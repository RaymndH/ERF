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
#include <string>
#include <cstdio>

#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"
#include "ERF_FireParams.H"
#include "ERF_DirectionalRos.H"

/**
 * @file ERF_GTestReinitAdvectDrift.cpp
 * @brief TASKS_levelset.md T1b: does the near-front distance-VALUE
 *        corruption T1 measured on a flat, grid-oblique front (crossing
 *        itself does not move, but phi's VALUES near it degrade, worse the
 *        more oblique) feed forward into a REAL, growing front-POSITION bias
 *        once a genuine advection step reads that corrupted gradient field,
 *        even though static reinit alone left the crossing exactly fixed?
 *
 * Design. T1 was static: reinit only, no advection. Here each "cycle" is
 * ONE reinit call (clamp on, iters=10, same dtau convention as T1/production)
 * followed by ONE advect_levelset_directional_rk3 step (the actual production
 * advection path for the campaign's directional_ros=true,
 * ros_model=rothermel, directional_wind_coupling=advective setup -- NOT
 * advect_levelset_weno5z_rk3, which is only the non-directional fallback;
 * confirmed by reading ERF_FireLayer.cpp's branch selection directly).
 *
 * Geometry, two cases:
 *  - "flank_34deg" / "flank_22p5deg": a STRAIGHT front at a fixed oblique
 *    angle to the grid (T1's worst value-corruption angles), with wind
 *    ALIGNED WITH THE FRONT'S OWN TANGENT. This makes the wind-normal
 *    component cos_w = 0 identically along the whole line -- an exact
 *    isotropic-rate (R=R0) translation, analytically exact, AND it means the
 *    front stays at this one persistently-oblique angle to the grid for
 *    every one of the N cycles: the flank-of-a-wind-driven-fire analogue,
 *    which in the full campaign runs is exactly where the drift was found to
 *    concentrate (not at the head, which sweeps through many angles as the
 *    perimeter curves).
 *  - "circle_headflank": T1's convex circle (radius 1000 m) in the SAME
 *    fixed wind. Because the front normal sweeps through every angle around
 *    the circle, this single geometry contains a wind-aligned "head" sector
 *    (cos_w near 1), a wind-tangent "flank" sector (cos_w near 0) and a
 *    "backing" sector (cos_w<0, clamped to R0) simultaneously -- the
 *    rotating/curvature-dominated comparison case, analogous to how the
 *    real fire head continuously changes local orientation.
 *
 * Because the true evolved shape is not elementary once ROS varies with
 * wind-projection angle around the circle, the metric here is NOT deviation
 * from a closed-form solution (that's only tractable for the wind-aligned
 * flank case, where R=R0 exactly and the exact answer is a rigid outward
 * translation by R0*T). The primary, always-available metric is instead the
 * DIFFERENCE between a "reinit+advect" run and an "advect-only" run started
 * from the identical initial field and run for the identical N cycles with
 * the identical wind/ROS/advection settings -- isolating exactly what
 * reinit-coupling adds on top of whatever bias advection has on its own.
 *
 * Synthetic Rothermel coefficients (rc): chosen for round numbers, not fuel
 * realism -- R0=0.16 m/s, wind_conv=1 (ft/min unit skipped, U taken directly
 * in the same units as R0), B=1 (linear in wind, avoids fractional powers),
 * beta_ratio_E=1, C=2.25 so that phi_w = C*U = 9 at U=4 m/s (matching the
 * campaign's actual free-stream wind and the "|V| ~ 10*R0" order-of-magnitude
 * noted in CLAUDE.md's "Leading hypothesis" section) -> R_head = R0*(1+9) =
 * 1.6 m/s, matching the campaign's actual target head rate.
 */

using namespace amrex;
using namespace fire_levelset;

namespace {

constexpr int  NCELL = 100;
constexpr Real LDOM  = 5000.0;
constexpr Real DX    = LDOM / NCELL;   // 50 m, matches T1 and the campaign fire grid
constexpr Real CX    = 0.5 * LDOM;
constexpr Real CY    = 0.5 * LDOM;
constexpr Real OFFX  = 0.37 * DX;
constexpr Real OFFY  = 0.61 * DX;

constexpr Real WIND_SPEED = 4.0;   // m/s, matches campaign LineFireFlat
constexpr Real DT_FIRE    = 0.25;  // s, matches campaign erf.fixed_dt

struct Grid
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Grid ()
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});
    }
};

Real xc (int i) { return (i + Real(0.5)) * DX; }
Real yc (int j) { return (j + Real(0.5)) * DX; }

struct LineShape
{
    Real theta_deg, nx, ny, x0, y0;
    explicit LineShape (Real deg) : theta_deg(deg)
    {
        const Real t = deg * M_PI / 180.0;
        nx = std::cos(t); ny = std::sin(t);
        x0 = CX + OFFX; y0 = CY + OFFY;
    }
    Real phi (Real x, Real y) const { return (x - x0) * nx + (y - y0) * ny; }
    // Tangent direction, for the wind-parallel ("flank") setup
    Real tx () const { return -ny; }
    Real ty () const { return  nx; }
    std::vector<std::pair<Real,Real>> front_points (int n = 50) const
    {
        std::vector<std::pair<Real,Real>> pts;
        const Real txv = -ny, tyv = nx;
        const Real half_len = 0.35 * LDOM;
        for (int k = 0; k < n; ++k) {
            const Real s = -half_len + 2.0 * half_len * k / (n - 1);
            pts.emplace_back(x0 + s * txv, y0 + s * tyv);
        }
        return pts;
    }
};

struct CircleShape
{
    Real R0, cx, cy;
    explicit CircleShape (Real R) : R0(R), cx(CX + OFFX), cy(CY + OFFY) {}
    Real phi (Real x, Real y) const
    {
        const Real r = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
        return r - R0;
    }
    std::vector<std::pair<Real,Real>> front_points (int n = 72) const
    {
        std::vector<std::pair<Real,Real>> pts;
        for (int k = 0; k < n; ++k) {
            const Real a = 2.0 * M_PI * k / n;
            pts.emplace_back(cx + R0 * std::cos(a), cy + R0 * std::sin(a));
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
    ParallelDescriptor::ReduceRealSum(v);
    return v;
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

/// Synthetic Rothermel coefficients: R0=0.16 m/s, phi_w = 2.25*U (linear,
/// B=1), so R_head = R0*(1+2.25*4) = 1.6 m/s at the campaign's 4 m/s wind --
/// matching the campaign's actual target head rate and wind/R0 ratio without
/// needing the full fuel-moisture lookup machinery T1b doesn't need.
DirectionalRosState make_synthetic_rothermel_state ()
{
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_ROTHERMEL;
    st.rc.R0 = 0.16;
    st.rc.C = 2.25;
    st.rc.B = 1.0;
    st.rc.beta_ratio_E = 1.0;
    st.rc.wind_conv = 1.0;
    st.rc.ros_conv = 1.0;
    st.rc.U_max_ftmin = 1.0e6;
    st.rc.phi_s_const = 0.0;
    st.rc.beta = 1.0;
    st.rc.I_R = 0.0;
    return st;
}

/// One reinit+advect (or advect-only) cycle, in place.
void one_cycle (MultiFab& phi, const MultiFab& wind, const MultiFab& slopes,
                 const Geometry& geom, const DirectionalRosState& st,
                 const LevelSetGradient& ls_grad, bool do_reinit)
{
    if (do_reinit) {
        const Real dtau = 0.25 * DX; // matches T1 / ERF_FireLayer.cpp's auto default
        reinitialize_phi(phi, geom, /*n_iters=*/10, dtau, -1.0, /*normalized=*/false,
                          nullptr, false, ls_grad, /*skip_final_clamp=*/false);
        fire_fill_boundary(phi, geom);
    }
    advect_levelset_directional_rk3(phi, wind, slopes, geom, DT_FIRE,
                                     /*eps_visc=*/0.4, st, nullptr, false, ls_grad,
                                     DIRECTIONAL_SHAPE_PROJECTION,
                                     DIRECTIONAL_ELLIPSE_LW_MODEL, 8.0,
                                     nullptr, DIRECTIONAL_WIND_COUPLING_ADVECTIVE);
    fire_fill_boundary(phi, geom);
}

/// Runs N cycles for both reinit+advect and advect-only from the SAME
/// initial field, and reports (a) the reinit-vs-advect-only difference at
/// each front-probe point (the primary, always-valid metric), and (b), only
/// when exact_R0 > 0 (the wind-aligned flank case, where the true solution
/// is a rigid translation by exact_R0*N*DT_FIRE), the deviation of EACH run
/// from that exact analytic reference.
template <class Shape>
void run_pair (const Shape& shape, const char* case_name, Grid& g,
                const MultiFab& wind, const MultiFab& slopes,
                const DirectionalRosState& st, const LevelSetGradient& ls_grad,
                Real exact_R0 /* <=0 disables the exact-reference check */)
{
    MultiFab phi_ra(g.ba, g.dm, 1, 3);   // reinit + advect
    MultiFab phi_ao(g.ba, g.dm, 1, 3);   // advect only
    fill_phi(phi_ra, shape);
    fill_phi(phi_ao, shape);
    fire_fill_boundary(phi_ra, g.geom);
    fire_fill_boundary(phi_ao, g.geom);

    const auto pts = shape.front_points();
    const std::vector<int> checkpoints = {10, 100, 1000};
    int done = 0;
    for (int cp : checkpoints) {
        for (int c = done; c < cp; ++c) {
            one_cycle(phi_ra, wind, slopes, g.geom, st, ls_grad, /*do_reinit=*/true);
            one_cycle(phi_ao, wind, slopes, g.geom, st, ls_grad, /*do_reinit=*/false);
        }
        done = cp;
        EXPECT_EQ(nonfinite_cells(phi_ra), 0) << case_name << " N=" << cp << " (reinit+advect)";
        EXPECT_EQ(nonfinite_cells(phi_ao), 0) << case_name << " N=" << cp << " (advect-only)";

        const Real T = cp * DT_FIRE;
        Real sum_diff = 0.0, max_abs_diff = 0.0;
        Real sum_err_ra = 0.0, max_abs_err_ra = 0.0;
        Real sum_err_ao = 0.0, max_abs_err_ao = 0.0;
        for (auto& xy : pts) {
            const Real p_ra = interp_phi(phi_ra, xy.first, xy.second);
            const Real p_ao = interp_phi(phi_ao, xy.first, xy.second);
            // Outward displacement relative to each other (always valid)
            const Real diff = (p_ao - p_ra); // >0: reinit+advect got further ahead (outward)
            sum_diff += diff; max_abs_diff = amrex::max(max_abs_diff, std::abs(diff));
            if (exact_R0 > 0.0) {
                // Exact position: the original front, translated outward by
                // R0*T along its own normal, has phi_exact = -R0*T at every
                // original front point (same sign convention as T1: negative
                // phi = burned = outward of the original front).
                const Real exact = -exact_R0 * T;
                const Real err_ra = p_ra - exact, err_ao = p_ao - exact;
                sum_err_ra += err_ra; max_abs_err_ra = amrex::max(max_abs_err_ra, std::abs(err_ra));
                sum_err_ao += err_ao; max_abs_err_ao = amrex::max(max_abs_err_ao, std::abs(err_ao));
            }
        }
        const Real mean_diff_dx = (sum_diff / pts.size()) / DX;
        const Real max_diff_dx  = max_abs_diff / DX;
        if (exact_R0 > 0.0) {
            const Real mean_err_ra_dx = (sum_err_ra / pts.size()) / DX;
            const Real mean_err_ao_dx = (sum_err_ao / pts.size()) / DX;
            std::printf("[T1b] case=%-18s N=%4d  (ra-ao)_mean=% .4f dx  (ra-ao)_max=%.4f dx  "
                        "|  ra_err_mean=% .4f dx  ao_err_mean=% .4f dx  ra_err_max=%.4f dx  ao_err_max=%.4f dx\n",
                        case_name, cp, mean_diff_dx, max_diff_dx,
                        mean_err_ra_dx, mean_err_ao_dx, max_abs_err_ra / DX, max_abs_err_ao / DX);
        } else {
            std::printf("[T1b] case=%-18s N=%4d  (ra-ao)_mean=% .4f dx  (ra-ao)_max=%.4f dx\n",
                        case_name, cp, mean_diff_dx, max_diff_dx);
        }
    }
}

} // namespace

/// T1b (TASKS_levelset.md): does the flat-oblique reinit distance-VALUE
/// corruption T1 found (crossing fixed, field corrupted) turn into a real,
/// growing front-POSITION bias once genuine advection (the actual
/// production advect_levelset_directional_rk3 path, directional_ros=true,
/// ros_model=rothermel, directional_wind_coupling=advective) reads that
/// corrupted gradient field between reinit calls?
TEST(ReinitAdvectDrift, FlankVsHeadBias)
{
    Grid g;
    const DirectionalRosState st = make_synthetic_rothermel_state();
    LevelSetGradient ls_grad;
    ls_grad.scheme = LEVELSET_GRAD_WENO5Z_FRONT;
    ls_grad.band = 4.0 * DX;               // levelset_weno_band_cells=4 (campaign)
    ls_grad.eps_visc_front = 0.1;          // campaign levelset_eps_visc_front
    ls_grad.visc_d0 = 2.0 * DX;            // levelset_visc_front_cells=2
    ls_grad.visc_d1 = ls_grad.visc_d0 + 2.0 * DX; // + levelset_visc_transition_cells=2

    MultiFab slopes(g.ba, g.dm, 2, 3);
    slopes.setVal(0.0);

    // --- Flank case: wind aligned with the front's own tangent (cos_w=0
    // identically along the whole line) -- exact solution is a rigid
    // outward translation at rate R0 (the directional/wind term contributes
    // nothing here since U_n=0 everywhere on this front).
    for (Real deg : {22.5, 34.0}) {
        LineShape shape(deg);
        MultiFab wind(g.ba, g.dm, 2, 3);
        wind.setVal(WIND_SPEED * shape.tx(), 0, 1, 3);
        wind.setVal(WIND_SPEED * shape.ty(), 1, 1, 3);
        char name[32]; std::snprintf(name, sizeof(name), "flank_%gdeg", deg);
        run_pair(shape, name, g, wind, slopes, st, ls_grad, /*exact_R0=*/st.rc.R0);
    }

    // --- Head/flank/backing case: same fixed wind direction (+x), but a
    // circle so the front normal sweeps through every angle relative to the
    // wind simultaneously -- the curvature-dominated, "local angle keeps
    // changing" analogue of the real fire head. No simple closed-form
    // reference (ROS varies with wind-projection angle around the circle),
    // so only the reinit-vs-advect-only difference is reported.
    {
        CircleShape shape(1000.0);
        MultiFab wind(g.ba, g.dm, 2, 3);
        wind.setVal(WIND_SPEED, 0, 1, 3);
        wind.setVal(0.0, 1, 1, 3);
        run_pair(shape, "circle_headflank", g, wind, slopes, st, ls_grad, /*exact_R0=*/-1.0);
    }
}
