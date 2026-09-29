#include <gtest/gtest.h>
#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Math.H>
#include <cmath>

static constexpr double REL = (sizeof(amrex::Real) == 8) ? 1e-10 : 1e-4;

#include "ERF_FireParams.H"   // ERF_CheneyGouldModel.H reads FireParams without including it
#include "ERF_NumericalSchemes.H"
#include "ERF_Reinitialize.H"
#include "ERF_LevelSetAdvection.H"
#include "ERF_DirectionalRos.H"

/**
 * @file ERF_GTestSplitHamiltonianAdvection.cpp
 * @brief Regression guard for the Hamiltonian-splitting fix
 *        (erf.fire.directional_split_hamiltonian): with
 *        directional_wind_coupling = "advective", R(n)|grad phi| is exactly
 *        the two-term Hamiltonian R0|grad phi| + max(V.grad phi, 0), and
 *        folding it into one Godunov flux built from an estimated front
 *        normal upwinds the wind-driven term along that normal instead of
 *        along V, where its information actually travels (a corner/curvature
 *        -seeded rotation-variance bulge, worst at oblique angles). The split
 *        path (advect_levelset_directional_rk3_split, ERF_DirectionalRos.H)
 *        upwinds the two terms separately, each by its own correct rule.
 *
 * A short, obliquely-oriented (34deg, this investigation's own convention)
 * capsule ignition line is advanced under a Rothermel wind tuned to a head
 * rate of 6 m/s -- deliberately high so a large, easily measured head
 * advance accumulates in a modest number of steps -- and the head's actual
 * position (found by a 1D bilinear-sampled bisection for the zero crossing
 * along the wind ray through the capsule's own centre, i.e. away from its
 * two rounded end corners) is compared to the analytic Rf * T. One test
 * checks the split path alone (advection only, no reinitialisation); a
 * second interleaves the production Jiang-Peng reinitialisation
 * (reinitialize_phi_jiang_peng, every step) to confirm the fix holds up with
 * reinit active too -- the combination that actually showed the multi-hundred
 * -metre wing with the baseline (un-split) scheme in the full campaign
 * (WRF_ERF/CLAUDE.md, "Hamiltonian splitting for the advective wind
 * coupling").
 */

using namespace amrex;
using namespace fire_levelset;

namespace {

constexpr int  NCELL = 120;
constexpr Real LDOM  = 600.0;
constexpr Real DX    = LDOM / NCELL;   // 5 m

constexpr Real WIND_DEG   = 34.0;   // math convention, 0 = +x; this investigation's own angle
constexpr Real TARGET_RF  = 6.0;    // deliberately high head rate [m/s]
constexpr Real HALF_WIDTH = 25.0;   // capsule half-width [m]
constexpr Real HALF_LEN   = 75.0;   // capsule half-length [m] (150 m line)

/// Rothermel coefficients close to fuel model 1 at 5.5% moisture, wind cap
/// lifted -- identical fixture to ERF_GTestDirectionalShape.cpp's
/// rothermel_state(), reused here rather than re-derived so both tests draw
/// on the same validated constants.
DirectionalRosState rothermel_state ()
{
    DirectionalRosState st;
    st.model = DIRECTIONAL_ROS_ROTHERMEL;
    st.rc = RothermelComputed{};
    st.rc.R0           = 0.0239;
    st.rc.C            = 5.4e-5;
    st.rc.B            = 2.07;
    st.rc.beta_ratio_E = 1.32;
    st.rc.beta         = 0.0010625;
    st.rc.phi_s_const  = 41.1;
    st.rc.U_max_ftmin  = 1.0e6;
    st.rc.wind_conv    = 196.85;
    st.rc.ros_conv     = 1.0;
    st.rc.I_R          = 0.0;
    return st;
}

/// Wind speed [m/s] whose head-on (cos_theta_wind=1) advective-coupling rate
/// is exactly target_rf, found by bisection against directional_ros_cell
/// itself -- not a hand-derived constant, so this stays correct if the
/// Rothermel coefficients above ever change.
Real wind_speed_for_head_rate (const DirectionalRosState& st, Real target_rf)
{
    Real lo = 0.0, hi = 30.0;
    for (int it = 0; it < 60; ++it) {
        const Real mid = 0.5 * (lo + hi);
        const Real r = directional_ros_cell(st, mid, Real(0.0),
                                             DIRECTIONAL_WIND_COUPLING_ADVECTIVE, mid, Real(0.0));
        if (r < target_rf) { lo = mid; } else { hi = mid; }
    }
    return 0.5 * (lo + hi);
}

struct CapsuleFixture
{
    BoxArray            ba;
    DistributionMapping dm;
    Geometry            geom;
    Real wind_deg, wind_speed, wx, wy;
    Real cx, cy;      // capsule centre (upwind of the domain centre)
    Real px, py;      // capsule line direction (perpendicular to the wind)

    CapsuleFixture (Real target_rf)
    {
        Box domain(IntVect(0, 0, 0), IntVect(NCELL - 1, NCELL - 1, 0));
        ba = BoxArray(domain);   // single box: no ghost-exchange concern here, already covered elsewhere
        dm = DistributionMapping(ba);
        RealBox rb({0.0, 0.0, 0.0}, {LDOM, LDOM, 1.0});
        geom = Geometry(domain, rb, CoordSys::cartesian, {0, 0, 0});

        wind_deg = WIND_DEG;
        const DirectionalRosState st = rothermel_state();
        wind_speed = wind_speed_for_head_rate(st, target_rf);
        const Real th = wind_deg * Math::pi<Real>() / 180.0;
        wx = wind_speed * std::cos(th);
        wy = wind_speed * std::sin(th);

        // Capsule centred at the domain centre, shifted upwind so the head
        // has room to advance without leaving the domain.
        const Real cxh = std::cos(th), cyh = std::sin(th);
        cx = 0.5 * LDOM - 200.0 * cxh;
        cy = 0.5 * LDOM - 200.0 * cyh;
        // Line direction perpendicular to the wind
        px = -cyh; py = cxh;
    }

    /// Signed distance to the capsule (stadium): negative (burned) within
    /// HALF_WIDTH of the line segment [centre - HALF_LEN*p, centre + HALF_LEN*p].
    void capsule (MultiFab& phi) const
    {
        const Real cxL = cx, cyL = cy, pxL = px, pyL = py;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto p = phi.array(mfi);
            const Box& gbx = mfi.growntilebox();
            ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real x = (i + Real(0.5)) * DX;
                const Real y = (j + Real(0.5)) * DX;
                const Real rx = x - cxL, ry = y - cyL;
                const Real t = amrex::Clamp(rx * pxL + ry * pyL, -HALF_LEN, HALF_LEN);
                const Real dx0 = rx - t * pxL, dy0 = ry - t * pyL;
                p(i, j, k) = std::sqrt(dx0 * dx0 + dy0 * dy0) - HALF_WIDTH;
            });
        }
    }

    /// Bilinear sample of a (single-box) cell-centred MultiFab at (x, y).
    Real sample (const MultiFab& phi, Real x, Real y) const
    {
        const Real fi = x / DX - 0.5, fj = y / DX - 0.5;
        int i0 = static_cast<int>(std::floor(fi));
        int j0 = static_cast<int>(std::floor(fj));
        const Real tx = fi - i0, ty = fj - j0;
        Real result = 0.0;
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.fabbox();   // includes ghosts
            if (!bx.contains(IntVect(i0, j0, 0)) || !bx.contains(IntVect(i0 + 1, j0 + 1, 0))) { continue; }
            auto p = phi.const_array(mfi);
            const Real v00 = p(i0, j0, 0),     v10 = p(i0 + 1, j0, 0);
            const Real v01 = p(i0, j0 + 1, 0), v11 = p(i0 + 1, j0 + 1, 0);
            result = v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty)
                   + v01 * (1 - tx) * ty       + v11 * tx * ty;
        }
        ParallelDescriptor::ReduceRealSum(result);   // exactly one rank contributes
        return result;
    }

    /// Distance from the capsule centre, along the wind ray, to phi's zero
    /// crossing -- bisection on the bilinear-sampled field. The ray starts
    /// at the capsule's own centre (far from both rounded end corners), so
    /// this is the head position, not a flank or corner effect.
    Real head_position (const MultiFab& phi) const
    {
        Real lo = 0.0, hi = 350.0;   // must stay inside the domain (see ctor: 200 m upwind margin)
        const Real th = wind_deg * Math::pi<Real>() / 180.0;
        const Real ux = std::cos(th), uy = std::sin(th);
        auto phi_at = [&](Real d) { return sample(phi, cx + d * ux, cy + d * uy); };
        EXPECT_LT(phi_at(lo), 0.0) << "capsule centre should start burned";
        EXPECT_GT(phi_at(hi), 0.0) << "search bound should stay ahead of the front";
        for (int it = 0; it < 40; ++it) {
            const Real mid = 0.5 * (lo + hi);
            if (phi_at(mid) < 0.0) { lo = mid; } else { hi = mid; }
        }
        return 0.5 * (lo + hi);
    }
};

} // namespace

/// Advection only (no reinitialisation), split-Hamiltonian path: the head
/// advances at the model's own analytic Rothermel rate (6 m/s, tuned above)
/// to within 2% over 30 s (180 m) at 34deg -- an oblique angle where the
/// baseline (un-split) scheme's front-normal estimate is least accurate.
TEST(SplitHamiltonianAdvection, HeadRateAt34DegreesMatchesAnalyticRf)
{
    const Real target_rf = TARGET_RF;
    CapsuleFixture f(target_rf);
    const DirectionalRosState st = rothermel_state();

    MultiFab phi(f.ba, f.dm, 1, 3), wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    f.capsule(phi);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);

    const Real dt = 0.1;
    const int  nsteps = 300;   // 30 s simulated -> ~180 m head advance
    const Real eps_visc = 0.4;

    const Real d0 = f.head_position(phi);
    for (int step = 0; step < nsteps; ++step) {
        advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st);
        fire_fill_boundary(phi, f.geom);
    }
    const Real d1 = f.head_position(phi);

    const Real traveled = d1 - d0;
    const Real expected = target_rf * (nsteps * dt);
    EXPECT_NEAR(traveled, expected, 0.02 * expected)
        << "traveled=" << traveled << " expected=" << expected;
}

/// Same setup, but with the production Jiang-Peng reinitialisation
/// (erf.fire.levelset.reinit_scheme = "jiang_peng") applied every step --
/// the combination that showed a multi-hundred-metre wing at 34deg with the
/// baseline (un-split) advection scheme in the full campaign (see
/// WRF_ERF/CLAUDE.md). With the split path, head rate still matches the
/// analytic Rf to within 3% (a slightly looser tolerance than the pure
/// advection test, since reinit's own known curvature-dependent bias, see
/// the same file's reinit sections, is a separate, smaller effect this test
/// does not aim to eliminate -- only to confirm the Hamiltonian-splitting
/// bug is not reintroducing a much larger error on top of it).
TEST(SplitHamiltonianAdvection, ReinitHeadRateAt34DegreesMatchesAnalyticRf)
{
    const Real target_rf = TARGET_RF;
    CapsuleFixture f(target_rf);
    const DirectionalRosState st = rothermel_state();

    MultiFab phi(f.ba, f.dm, 1, 3), wind(f.ba, f.dm, 2, 0), slopes(f.ba, f.dm, 2, 0);
    f.capsule(phi);
    wind.setVal(f.wx, 0, 1);
    wind.setVal(f.wy, 1, 1);
    slopes.setVal(0.0);

    const Real dt   = 0.015;
    const int  nsteps = 2000;  // same 30s/180m as the advection-only test above, but 2000 reinit calls -- see the reinit sections of WRF_ERF/CLAUDE.md on why bias scales with call count, not physical time
    const Real eps_visc = 0.4;
    const Real dtau  = 0.01 * DX;   // matches WRF-Fire's own default, ERF_FireLayer.cpp

    const Real d0 = f.head_position(phi);
    for (int step = 0; step < nsteps; ++step) {
        advect_levelset_directional_rk3_split(phi, wind, slopes, f.geom, dt, eps_visc, st);
        fire_fill_boundary(phi, f.geom);
        reinitialize_phi_jiang_peng(phi, f.geom, 1, dtau);
        fire_fill_boundary(phi, f.geom);
    }
    const Real d1 = f.head_position(phi);

    const Real traveled = d1 - d0;
    const Real expected = target_rf * (nsteps * dt);
    EXPECT_NEAR(traveled, expected, 0.03 * expected)
        << "traveled=" << traveled << " expected=" << expected;
}
