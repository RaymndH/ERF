#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <cmath>
#include <vector>
#include <tuple>
#include <string>
#include <cstdio>

#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"

/**
 * @file ERF_GTestReinitDrift.cpp
 * @brief TASKS_levelset.md T1: how far reinitialize_phi() moves the zero
 *        level set per call on an EXACT signed-distance field, with no
 *        advection at all -- isolating reinit's own behaviour.
 *
 * Deviation from the TASKS_levelset.md T1 spec, deliberate: that spec
 * describes the normalized ([-1,1]) phi convention. The ACTUAL call site
 * (ERF_FireLayer.cpp ~line 1176-1180) always passes normalized=false
 * (signed-distance/metres mode) for the fire-front reinit call the campaign
 * runs use, so this test matches that: phi is the metres distance directly,
 * no banding, no per-iteration clamp (only normalized mode clamps).
 *
 * Domain: 5000 m square, fire dx = 50 m (100x100 cells) -- matches the
 * campaign's fire grid (grid_ratio=4 under a 200 m atm grid), not the
 * TASKS_levelset.md placeholder of 25 m. Single box (no MPI decomposition)
 * so a simple bilinear interpolant can read cell-centred phi anywhere in
 * the domain interior directly.
 *
 * Front-displacement measurement: for a signed-distance field, at a point
 * exactly ON the original (t=0) front, phi_exact(t=0) = 0. To leading order
 * near a front with finite curvature, if the true front moves outward
 * (grows the burned region) by delta along the local normal, a fixed point
 * on the OLD front then has distance approximately -delta from the NEW
 * front (it is now on the burned side if delta>0). So
 *   delta(s) = -phi_measured(point on original front, arc-length s)
 * is the outward normal displacement at that point, exact for a straight
 * front (zero curvature) and first-order accurate elsewhere (circle) --
 * adequate here since drift is a small perturbation being measured, not the
 * front's bulk shape.
 */

using namespace amrex;
using namespace fire_levelset;

namespace {

constexpr int  NCELL = 100;
constexpr Real LDOM  = 5000.0;
constexpr Real DX    = LDOM / NCELL;     // 50 m, matches the campaign fire grid
constexpr Real CX    = 0.5 * LDOM;
constexpr Real CY    = 0.5 * LDOM;
// Sub-cell offset so no shape is grid-aligned by accident (TASKS_levelset.md T1).
constexpr Real OFFX  = 0.37 * DX;
constexpr Real OFFY  = 0.61 * DX;

struct Grid
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;

    Grid ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);   // single box: no maxSize() split
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});
    }
};

Real xc (int i) { return (i + Real(0.5)) * DX; }
Real yc (int j) { return (j + Real(0.5)) * DX; }

// ---- Exact signed-distance shape generators (phi < 0 = burned) -----------

/// Half-plane through (CX+OFFX, CY+OFFY), normal (cos th, sin th); burned on
/// the side the normal points away from. Exact signed distance everywhere.
struct LineShape
{
    Real theta_deg;
    Real nx, ny, x0, y0;
    explicit LineShape (Real deg) : theta_deg(deg)
    {
        const Real t = deg * M_PI / 180.0;
        nx = std::cos(t); ny = std::sin(t);
        x0 = CX + OFFX; y0 = CY + OFFY;
    }
    Real phi (Real x, Real y) const { return (x - x0) * nx + (y - y0) * ny; }
    // ~50 probe points along the front, well clear of the domain edges
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        const Real tx = -ny, ty = nx; // tangent
        const Real half_len = 0.35 * LDOM; // stay >= ~15 cells from edges
        for (int k = 0; k < n; ++k) {
            const Real s = -half_len + 2.0 * half_len * k / (n - 1);
            pts.emplace_back(x0 + s * tx, y0 + s * ty);
        }
        return pts;
    }
};

/// Circle of radius R0 about (CX+OFFX, CY+OFFY). convex=true: burned inside
/// (phi = r - R0, convex burned region); convex=false: burned OUTSIDE
/// (phi = R0 - r, concave burned region, inverted circle).
struct CircleShape
{
    Real R0, cx, cy; bool convex;
    CircleShape (Real R, bool conv) : R0(R), cx(CX + OFFX), cy(CY + OFFY), convex(conv) {}
    Real phi (Real x, Real y) const
    {
        const Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        return convex ? (r - R0) : (R0 - r);
    }
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        for (int k = 0; k < n; ++k) {
            const Real a = 2.0 * M_PI * k / n;
            pts.emplace_back(cx + R0 * std::cos(a), cy + R0 * std::sin(a));
        }
        return pts;
    }
};

/// 90-degree convex corner: burned quarter-plane {x'>=0 and y'>=0} in a frame
/// rotated by theta_deg about (CX+OFFX, CY+OFFY). Exact signed distance:
/// nearest-edge distance when both x',y' >= 0 (inside, phi<0); nearest-edge
/// distance to the violated axis when exactly one is negative; distance to
/// the corner point itself when both are negative.
struct CornerShape
{
    Real theta_deg, cx, cy, c, s;
    explicit CornerShape (Real deg) : theta_deg(deg), cx(CX + OFFX), cy(CY + OFFY)
    {
        const Real t = deg * M_PI / 180.0; c = std::cos(t); s = std::sin(t);
    }
    void local (Real x, Real y, Real& xp, Real& yp) const
    {
        const Real dx = x - cx, dy = y - cy;
        xp =  dx * c + dy * s;
        yp = -dx * s + dy * c;
    }
    Real phi (Real x, Real y) const
    {
        Real xp, yp; local(x, y, xp, yp);
        if (xp >= 0.0 && yp >= 0.0) { return -amrex::min(xp, yp); }
        if (xp <  0.0 && yp >= 0.0) { return -xp; }
        if (xp >= 0.0 && yp <  0.0) { return -yp; }
        return std::sqrt(xp * xp + yp * yp);
    }
    // Probes along both edges, away from the corner singularity itself
    // (the exact-distance trick above is only first-order right at the kink).
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        const Real lo = 0.08 * LDOM, hi = 0.35 * LDOM; // skip near the corner
        for (int k = 0; k < n / 2; ++k) {
            const Real xp = lo + (hi - lo) * k / (n / 2 - 1);
            // edge y'=0, x'=xp>0: rotate back to world frame
            pts.emplace_back(cx + xp * c, cy + xp * s);
        }
        for (int k = 0; k < n / 2; ++k) {
            const Real yp = lo + (hi - lo) * k / (n / 2 - 1);
            pts.emplace_back(cx - yp * s, cy + yp * c);
        }
        return pts;
    }
};

template <class Shape>
void fill_phi (MultiFab& phi, const Shape& shape)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.array(mfi);
        const Box& gbx = mfi.growntilebox();
        ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            p(i, j, k) = shape.phi(xc(i), yc(j));
        });
    }
}

/// Bilinear interpolation of a cell-centred, single-box MultiFab at (x, y).
/// Valid away from the domain edges (>= 1 cell in from every side); callers
/// keep all front points >= ~8 cells from any edge.
Real interp_phi (const MultiFab& phi, Real x, Real y)
{
    Real fi = x / DX - 0.5, fj = y / DX - 0.5;
    int i0 = static_cast<int>(std::floor(fi));
    int j0 = static_cast<int>(std::floor(fj));
    Real tx = fi - i0, ty = fj - j0;
    Real v = 0.0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Real p00 = p(i0,     j0,     0);
        const Real p10 = p(i0 + 1, j0,     0);
        const Real p01 = p(i0,     j0 + 1, 0);
        const Real p11 = p(i0 + 1, j0 + 1, 0);
        v = (1 - tx) * (1 - ty) * p00 + tx * (1 - ty) * p10
          + (1 - tx) *      ty  * p01 + tx *      ty  * p11;
    }
    ParallelDescriptor::ReduceRealSum(v); // only one rank actually owns data
    return v;
}

Real signed_area_from_phi (const MultiFab& phi_d)
{
    // Sub-cell burned-area proxy: sum over cells of clamp(0.5 - phi_d/dx, 0, 1).
    Real total = 0.0;
    for (MFIter mfi(phi_d); mfi.isValid(); ++mfi) {
        auto p = phi_d.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                const Real frac = amrex::max(Real(0.0), amrex::min(Real(1.0), Real(0.5) - p(i, j, 0) / DX));
                total += frac;
            }
        }
    }
    ParallelDescriptor::ReduceRealSum(total);
    return total * DX * DX;
}

amrex::Long burned_cells (const MultiFab& phi)
{
    amrex::Long n = 0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p = phi.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (p(i, j, 0) < Real(0.0)) { ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

amrex::Long nonfinite_cells (const MultiFab& a)
{
    amrex::Long n = 0;
    for (MFIter mfi(a); mfi.isValid(); ++mfi) {
        auto p = a.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (!std::isfinite(p(i, j, 0))) { ++n; }
            }
        }
    }
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

struct DriftResult { Real mean_disp_dx, max_abs_disp_dx, area_change_m2, sub_area_change_m2, max_dist_err_m; };

template <class Shape>
DriftResult measure (const MultiFab& phi, const MultiFab& phi_exact0,
                      const Shape& shape, Real area0, Real sub_area0)
{
    const auto pts = shape.front_points();
    Real sum = 0.0, mx = 0.0;
    for (auto& xy : pts) {
        const Real d = -interp_phi(phi, xy.first, xy.second); // outward = +
        sum += d; mx = amrex::max(mx, std::abs(d));
    }
    const Real mean_disp = (sum / pts.size()) / DX;
    const Real max_disp  = mx / DX;

    Real max_err = 0.0;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        auto p  = phi.const_array(mfi);
        auto pe = phi_exact0.const_array(mfi);
        const Box& bx = mfi.validbox();
        for (int j = bx.smallEnd(1); j <= bx.bigEnd(1); ++j) {
            for (int i = bx.smallEnd(0); i <= bx.bigEnd(0); ++i) {
                if (std::abs(pe(i, j, 0)) < 3.0 * DX) {
                    max_err = amrex::max(max_err, std::abs(p(i, j, 0) - pe(i, j, 0)));
                }
            }
        }
    }
    ParallelDescriptor::ReduceRealMax(max_err);

    const Real area = burned_cells(phi) * DX * DX;
    const Real sub_area = signed_area_from_phi(phi);
    return {mean_disp, max_disp, area - area0, sub_area - sub_area0, max_err};
}

/// Runs N_max reinit calls (no advection between them), recording the drift
/// metrics at checkpoints {1, 10, 100, 1000} (intersected with N_max).
/// dtau matches ERF_FireLayer.cpp's auto default (0.25*min(dx,dy)) exactly,
/// since levelset_reinit_dtau is not overridden anywhere in the campaign.
template <class Shape>
void run_matrix (const Shape& shape, const char* shape_name, Grid& g,
                  bool skip_clamp, int iters,
                  std::vector<std::tuple<std::string,int,int,bool,DriftResult>>& all_results,
                  Real dtau = 0.25 * DX,   // default: ERF_FireLayer.cpp's auto default
                  bool disable_subcell_fix = false,
                  bool wrf_style_upwind = false,
                  bool exact_interface_freeze = false)
{
    MultiFab phi(g.ba, g.dm, 1, 3), phi0(g.ba, g.dm, 1, 3);
    fill_phi(phi,  shape);
    fill_phi(phi0, shape);
    fire_fill_boundary(phi, g.geom);

    const Real area0 = burned_cells(phi) * DX * DX;
    const Real sub_area0 = signed_area_from_phi(phi);

    const std::vector<int> checkpoints = {1, 10, 100, 1000};
    int done = 0;
    for (int cp : checkpoints) {
        const int n_calls = cp - done;
        for (int c = 0; c < n_calls; ++c) {
            reinitialize_phi(phi, g.geom, iters, dtau, -1.0, /*normalized=*/false,
                              nullptr, false, LevelSetGradient{}, skip_clamp,
                              /*tvd_rk3=*/false, disable_subcell_fix, wrf_style_upwind,
                              exact_interface_freeze);
            fire_fill_boundary(phi, g.geom);
        }
        done = cp;
        EXPECT_EQ(nonfinite_cells(phi), 0)
            << shape_name << " clamp=" << !skip_clamp << " iters=" << iters << " N=" << cp;
        DriftResult r = measure(phi, phi0, shape, area0, sub_area0);
        all_results.emplace_back(std::string(shape_name), iters, cp, skip_clamp, r);
        std::printf("[T1] shape=%-22s clamp=%s iters=%2d N=%4d  "
                    "mean_disp=% .4f dx  max_disp=%.4f dx  "
                    "area_change=% .1f m2  max_dist_err=%.3f m\n",
                    shape_name, skip_clamp ? "off" : "on ", iters, cp,
                    r.mean_disp_dx, r.max_abs_disp_dx, r.area_change_m2, r.max_dist_err_m);
        // Loose sanity bound only -- this test's job is to MEASURE drift
        // (including the known-bad unclamped iters=10 erosion), not to gate
        // on a pass/fail threshold that some configurations are expected to
        // fail. See TASKS_levelset.md T1 for the intended pass rule and
        // CLAUDE.md's "Reinit clamp / late-time divergence investigation"
        // for the results this test produced.
        EXPECT_LT(std::abs(r.mean_disp_dx), 20.0)
            << shape_name << " clamp=" << !skip_clamp << " iters=" << iters << " N=" << cp
            << " -- displacement blew up, not just drifted";
    }
}

} // namespace

/// T1 (TASKS_levelset.md): static, no-advection measurement of how far
/// reinitialize_phi() moves the zero level set per call, isolated from
/// advection/ROS/wind. Priority subset actually run here (see file-header
/// deviation note): straight line at 0/10/22.5/34/45 deg, a convex circle,
/// a concave (inverted) circle, and a convex corner at 0/34 deg; each times
/// {clamp on, clamp off} x {iters=1, iters=10} x N in {1,10,100,1000}.
TEST(ReinitDrift, StaticFrontDisplacement)
{
    Grid g;
    std::vector<std::tuple<std::string,int,int,bool,DriftResult>> results;

    for (Real deg : {0.0, 10.0, 22.5, 34.0, 45.0}) {
        LineShape shape(deg);
        char name[64]; std::snprintf(name, sizeof(name), "line_%gdeg", deg);
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, name, g, skip_clamp, iters, results);
            }
        }
    }

    {
        CircleShape shape(1000.0, /*convex=*/true);
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, "circle_convex_R1000", g, skip_clamp, iters, results);
            }
        }
    }
    {
        CircleShape shape(1000.0, /*convex=*/false);
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, "circle_concave_R1000", g, skip_clamp, iters, results);
            }
        }
    }
    for (Real deg : {0.0, 34.0}) {
        CornerShape shape(deg);
        char name[64]; std::snprintf(name, sizeof(name), "corner_%gdeg", deg);
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, name, g, skip_clamp, iters, results);
            }
        }
    }

    // Regression check on the mechanism actually found by this test (NOT the
    // originally-guessed symmetric ratchet -- see the file-header note and
    // CLAUDE.md's 2026-09-26 T1 addendum for the full story). Reinit's error
    // is CURVATURE-SIGN-DEPENDENT: it shrinks convex bumps of burned area
    // (front retreats, phi increases) and grows concave dents (front
    // advances, phi decreases), i.e. it behaves like an added curvature-flow
    // smoothing term. The min-clamp (phi_out = min(phi_out, phi_in)) forbids
    // phi from increasing anywhere, so it suppresses the RETREAT pathway
    // (convex shrink) hard -- confirmed below -- but does essentially
    // nothing about the ADVANCE pathway (concave growth), because that
    // pathway already moves phi in the direction the clamp allows. Do not
    // strengthen this into "clamp always reduces |drift|" -- it measurably
    // does not on the concave shape (clamped 0.082 dx vs unclamped 0.068 dx
    // at iters=10, N=1000, 2026-09-26 run): the clamp is agnostic to
    // magnitude on that pathway, not protective.
    for (auto& [name, iters, cp, skip_clamp, r] : results) {
        if (name.find("convex") == std::string::npos) { continue; } // "circle_convex_..." only, not "concave"
        if (iters != 10 || cp != 1000 || skip_clamp) { continue; }  // only clamped rows as anchor
        for (auto& [name2, iters2, cp2, skip_clamp2, r2] : results) {
            if (name2 == name && iters2 == 10 && cp2 == 1000 && skip_clamp2 && !skip_clamp) {
                EXPECT_LT(std::abs(r.mean_disp_dx), std::abs(r2.mean_disp_dx))
                    << name << ": clamp should strongly suppress the convex-retreat erosion pathway";
                break;
            }
        }
    }
}

/// Follow-up (2026-09-26) to the WRF-reference comparison: does ERF's OWN
/// scheme (Russo-Smereka subcell fix + smoothed-sign Sussman, unchanged)
/// stop compounding with N once dtau is set to WRF's much smaller value
/// (0.01*dx instead of ERF's default 0.25*dx, 25x smaller), isolating
/// whether dtau magnitude alone explains why WRF's algorithm (same
/// curvature-bias mechanism per ERF_WrfReinitReference.H, but ~15x smaller
/// and flat vs N) doesn't compound the way ERF's default does.
TEST(ReinitDrift, CurvedFrontCreepWithWrfDtau)
{
    Grid g;
    std::vector<std::tuple<std::string,int,int,bool,DriftResult>> results;
    const Real wrf_dtau = 0.01 * DX;

    for (bool convex : {true, false}) {
        CircleShape shape(1000.0, convex);
        const char* name = convex ? "circle_convex_R1000_dtau01" : "circle_concave_R1000_dtau01";
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, name, g, skip_clamp, iters, results, wrf_dtau);
            }
        }
    }
}

/// Follow-up to CurvedFrontCreepWithWrfDtau: matching dtau to WRF's value
/// did NOT stop the concave/advance pathway from compounding with N (still
/// grew ~14x from N=1 to N=1000, vs WRF's reference staying flat at the
/// identical dtau/geometry). This test isolates whether the Russo-Smereka
/// near-front subcell override -- the one piece of ERF's scheme with no WRF
/// analogue at all -- is what's actually responsible, by disabling JUST that
/// branch (disable_subcell_fix=true) so every cell, near-front or not, uses
/// the general Sussman/Godunov update, still at WRF's dtau=0.01*dx.
TEST(ReinitDrift, CurvedFrontCreepSubcellFixDisabled)
{
    Grid g;
    std::vector<std::tuple<std::string,int,int,bool,DriftResult>> results;
    const Real wrf_dtau = 0.01 * DX;

    for (bool convex : {true, false}) {
        CircleShape shape(1000.0, convex);
        const char* name = convex ? "circle_convex_R1000_nosubcell" : "circle_concave_R1000_nosubcell";
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, name, g, skip_clamp, iters, results, wrf_dtau,
                           /*disable_subcell_fix=*/true);
            }
        }
    }
}

/// Follow-up to CurvedFrontCreepSubcellFixDisabled: the subcell fix was
/// refuted as the sole cause (disabling it made iters=10 growth WORSE, not
/// better). This tests the other identified difference: ERF's Godunov
/// branch selects the SAME side for both axes from sign(S(phi0)) alone
/// (no gradient information at all), while WRF's selects PER AXIS from
/// sign(S(phi0) * 4th-order-central-diff) -- still with the subcell fix
/// disabled and dtau matched to WRF's value, isolating just this one piece.
TEST(ReinitDrift, CurvedFrontCreepWrfStyleUpwind)
{
    Grid g;
    std::vector<std::tuple<std::string,int,int,bool,DriftResult>> results;
    const Real wrf_dtau = 0.01 * DX;

    for (bool convex : {true, false}) {
        CircleShape shape(1000.0, convex);
        const char* name = convex ? "circle_convex_R1000_wrfupwind" : "circle_concave_R1000_wrfupwind";
        for (bool skip_clamp : {false, true}) {
            for (int iters : {1, 10}) {
                run_matrix(shape, name, g, skip_clamp, iters, results, wrf_dtau,
                           /*disable_subcell_fix=*/true, /*wrf_style_upwind=*/true);
            }
        }
    }
}

/// Candidate FIX (2026-09-26): Sussman & Fatemi (1999)-style exact
/// interface freeze. Changes the Russo-Smereka near-front update from an
/// iterative relaxation toward the target subcell distance to a direct
/// assignment to that (already-exact, phi0-derived) target -- immediately
/// invariant across iterations and calls, by construction, not just
/// suppressed. Tested at BOTH the campaign's default dtau (0.25*dx) and
/// WRF's much smaller dtau (0.01*dx), subcell fix ON (required -- nothing
/// to freeze without it), to see whether this flattens the growth the way
/// WRF's reference implementation already is, at either/both settings.
TEST(ReinitDrift, CurvedFrontCreepExactInterfaceFreeze)
{
    Grid g;
    std::vector<std::tuple<std::string,int,int,bool,DriftResult>> results;

    for (Real dtau_val : {0.25 * DX, 0.01 * DX}) {
        const char* dtau_tag = (dtau_val == Real(0.25 * DX)) ? "dtau025" : "dtau001";
        for (bool convex : {true, false}) {
            CircleShape shape(1000.0, convex);
            char name[80];
            std::snprintf(name, sizeof(name), "circle_%s_R1000_exactfreeze_%s",
                          convex ? "convex" : "concave", dtau_tag);
            for (bool skip_clamp : {false, true}) {
                for (int iters : {1, 10}) {
                    run_matrix(shape, name, g, skip_clamp, iters, results, dtau_val,
                               /*disable_subcell_fix=*/false, /*wrf_style_upwind=*/false,
                               /*exact_interface_freeze=*/true);
                }
            }
        }
    }
}
