#ifndef ERF_REINITIALIZE_H
#define ERF_REINITIALIZE_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <ERF_NumericalSchemes.H>

/**
 * @file ERF_Reinitialize.H
 * @brief Level-set reinitialization, mirroring WRF-Fire's reinit_ls_rk3 /
 * advance_ls_reinit (module_fr_fire_core.F) exactly.
 *
 * RESET 2026-09-28. An extended investigation (WRF_ERF/CLAUDE.md and
 * TASKS_levelset.md in the campaign directory, and this file's own git
 * history) found that ERF's prior Sussman/Russo-Smereka scheme -- even
 * after fixing its subcell-distance formula for oblique fronts -- has a
 * non-saturating, curvature-driven drift on curved fronts that WRF's own
 * scheme does not show at a comparable magnitude, and that neither the
 * subcell fix, alternate upwind rules, an SSP/TVD RK3 pseudo-time
 * integrator, nor an exact-interface-freeze variant closed that gap
 * cleanly. Rather than continue patching ERF's own scheme, this
 * reimplements WRF's actual algorithm as ERF's production reinit path:
 *
 *   - smoothed sign S(phi0) = phi0/sqrt(phi0^2 + dx^2), computed ONCE from
 *     the field at call entry and held fixed across every outer iteration
 *     and RK stage.
 *   - Wicker & Skamarock (2002) 3-stage RK3 (NOT Shu-Osher SSP-RK3): every
 *     stage advances from the SAME fixed phi0 base, not additively from the
 *     previous stage --
 *       phi1 = phi0 + (dtau/3) L(phi0)
 *       phi2 = phi0 + (dtau/2) L(phi1)
 *       phi3 = phi0 +  dtau    L(phi2)
 *   - dtau = 0.01*dx by default (WRF's own default; ERF's old default was
 *     0.25*min(dx,dy), ~25x more aggressive at dx=dy).
 *   - n_outer pseudo-time steps per call (fire_lsm_reinit_iter; WRF's own
 *     comment: "1 iter each time step is enough" -- 1 is now ERF's default
 *     too, was 10).
 *   - Hybrid WENO5/ENO1 gradient: classic Jiang-Shu flux-form WENO5 on phi
 *     VALUES (not ERF's old HJ-WENO5-Z on divided differences) within
 *     `band` of the front, first-order ENO1 (Godunov) elsewhere, with WRF's
 *     own band test on phi DIRECTLY (not abs(phi) -- reproduced as WRF has
 *     it, asymmetric quirk and all, not "fixed").
 *   - WENO5's upwind side selected by sign(S(phi0) * D4th), D4th a
 *     4th-order CENTRAL difference of the current iterate (WRF's
 *     select_4th) -- not a Godunov derivative-sign selection.
 *   - NO subcell fix of any kind (no Russo-Smereka, no analogue). WRF has
 *     none; this is the headline structural simplification versus ERF's
 *     old scheme.
 *   - Final safeguard: phi_out = min(phi_out, phi_in) against the field at
 *     call entry ("fire area can only increase").
 *
 * Verified against WRF's exact source, not assumed -- this is the same
 * math as the test-only mirror in Tests/Unit/Fire/ERF_WrfReinitReference.H
 * (kept there as an independent from-scratch cross-check), now promoted to
 * production and extended with mask/wall-stencil support for non-burnable
 * terrain, which that test-only mirror deliberately omits.
 *
 * Deliberately different from WRF, both unavoidable and by design:
 *   - Units: WRF's LFN and ERF's fire_phi are both already metric signed
 *     distance at the call site (ERF_FireLayer.cpp), so no unit-convention
 *     complexity is needed here -- this file no longer has a normalized/
 *     banded [-1,1] mode at all (every call site always used the metric
 *     mode; the other mode existed but was never exercised in production).
 *   - Boundary handling: ghost cells are filled with fire_fill_boundary(),
 *     the same routine ERF's advection path uses, not WRF's own
 *     continue_at_boundary extrapolation or its bdy_eno1 forced-ENO1 band
 *     near the domain edge.
 *   - Non-burnable mask / wall_extrap: WRF has no equivalent to ERF's
 *     terrain mask. Masked cells pass through unchanged every stage; their
 *     neighbours are read through WallStencil, the same convention
 *     ERF_DirectionalRos.H's fill_directional_ros uses.
 *
 * Everything this file used to carry as opt-in diagnostics or candidate
 * fixes for the old scheme's compounding drift -- the per-axis subcell
 * distance formula, `reinit_skip_clamp`, `reinit_tvd_rk3`,
 * `disable_subcell_fix`, `wrf_style_upwind`, `exact_interface_freeze` (with
 * or without D-smoothing) -- is gone. That work is preserved in this
 * worktree's git history (branch `fire-reinit-drift-investigation`) and in
 * WRF_ERF/CLAUDE.md's dated addenda, not carried forward as production
 * surface area.
 */

namespace fire_levelset {

namespace wrf_reinit {

/** WRF's 1st-order ENO selection (module_fr_fire_core.F's select_eno). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real select_eno(amrex::Real diffL, amrex::Real diffR) noexcept
{
    using namespace amrex;
    if (!(diffL > 0) && !(diffR > 0)) { return diffR; }
    if (!(diffL < 0) && !(diffR < 0)) { return diffL; }
    if (!(diffL < 0) && !(diffR > 0)) {
        return (!(std::abs(diffR) < std::abs(diffL))) ? diffR : diffL;
    }
    return 0.0_rt;
}

/** WRF's 4th-order central difference, used only to pick WENO5's upwind
 * side (module_fr_fire_core.F's select_4th). */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real select_4th(amrex::Real dx, amrex::Real p_i, amrex::Real p_im1,
                        amrex::Real p_im2, amrex::Real p_ip1, amrex::Real p_ip2) noexcept
{
    using namespace amrex;
    const amrex::Real dp = (7.0_rt*p_ip1 + 7.0_rt*p_i - p_ip2 - p_im1) / (12.0_rt*dx);
    const amrex::Real dm = (7.0_rt*p_i + 7.0_rt*p_im1 - p_ip1 - p_im2) / (12.0_rt*dx);
    return dp - dm;
}

/** Classic Jiang-Shu flux-form WENO5 on phi VALUES (module_fr_fire_core.F's
 * select_weno5), NOT HJ-WENO5-Z on divided differences. `uf`'s sign selects
 * the upwind stencil side. Verified against the exact WRF source, including
 * that the i+1/2 and i-1/2 faces use different (not simply shifted) beta
 * formulas. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real select_weno5(amrex::Real dx,
                          amrex::Real p_i, amrex::Real p_im1, amrex::Real p_im2, amrex::Real p_im3,
                          amrex::Real p_ip1, amrex::Real p_ip2, amrex::Real p_ip3,
                          amrex::Real uf) noexcept
{
    using namespace amrex;
    constexpr Real g1 = 1.0_rt/10.0_rt, g2 = 3.0_rt/5.0_rt, g3 = 3.0_rt/10.0_rt, tol = 1.0e-6_rt;

    Real i0, im1, im2, im3, ip1, ip2;
    if (uf >= 0.0_rt) {
        i0 = p_i; im1 = p_im1; im2 = p_im2; im3 = p_im3; ip1 = p_ip1; ip2 = p_ip2;
    } else {
        i0 = p_i; im1 = p_ip1; im2 = p_ip2; im3 = p_ip3; ip1 = p_im1; ip2 = p_im2;
    }

    auto weno5_blend = [&](Real fh_1, Real fh_2, Real fh_3,
                            Real beta1, Real beta2, Real beta3) {
        const Real w1t = g1/((beta1+tol)*(beta1+tol));
        const Real w2t = g2/((beta2+tol)*(beta2+tol));
        const Real w3t = g3/((beta3+tol)*(beta3+tol));
        const Real wsum = w1t + w2t + w3t;
        return (w1t*fh_1 + w2t*fh_2 + w3t*fh_3) / wsum;
    };

    // Face at i+1/2, from (im2,im1,i,ip1,ip2).
    const Real flux_p = weno5_blend(
        (2.0_rt*im2 - 7.0_rt*im1 + 11.0_rt*i0) / 6.0_rt,
        (-1.0_rt*im1 + 5.0_rt*i0 + 2.0_rt*ip1) / 6.0_rt,
        (2.0_rt*i0 + 5.0_rt*ip1 - 1.0_rt*ip2) / 6.0_rt,
        (13.0_rt/12.0_rt)*(im2-2.0_rt*im1+i0)*(im2-2.0_rt*im1+i0) + 0.25_rt*(im2-4.0_rt*im1+3.0_rt*i0)*(im2-4.0_rt*im1+3.0_rt*i0),
        (13.0_rt/12.0_rt)*(im1-2.0_rt*i0+ip1)*(im1-2.0_rt*i0+ip1) + 0.25_rt*(im1-ip1)*(im1-ip1),
        (13.0_rt/12.0_rt)*(i0-2.0_rt*ip1+ip2)*(i0-2.0_rt*ip1+ip2) + 0.25_rt*(3.0_rt*i0-4.0_rt*ip1+ip2)*(3.0_rt*i0-4.0_rt*ip1+ip2));

    // Face at i-1/2, from (im3,im2,im1,i,ip1).
    const Real flux_m = weno5_blend(
        (2.0_rt*im3 - 7.0_rt*im2 + 11.0_rt*im1) / 6.0_rt,
        (-1.0_rt*im2 + 5.0_rt*im1 + 2.0_rt*i0) / 6.0_rt,
        (2.0_rt*im1 + 5.0_rt*i0 - 1.0_rt*ip1) / 6.0_rt,
        (13.0_rt/12.0_rt)*(im3-2.0_rt*im2+im1)*(im3-2.0_rt*im2+im1) + 0.25_rt*(im3-4.0_rt*im2+3.0_rt*im1)*(im3-4.0_rt*im2+3.0_rt*im1),
        (13.0_rt/12.0_rt)*(im2-2.0_rt*im1+i0)*(im2-2.0_rt*im1+i0) + 0.25_rt*(im2-i0)*(im2-i0),
        (13.0_rt/12.0_rt)*(im1-2.0_rt*i0+ip1)*(im1-2.0_rt*i0+ip1) + 0.25_rt*(3.0_rt*im1-4.0_rt*i0+ip1)*(3.0_rt*im1-4.0_rt*i0+ip1));

    return (uf >= 0.0_rt) ? (flux_p - flux_m) / dx : -(flux_p - flux_m) / dx;
}

/** Per-cell RHS L(phi) = S(phi0)*(1-|grad phi|), WRF's fire_upwinding_reinit=4
 * (hybrid WENO5/ENO1) branch, band test WITHOUT abs() -- mirrors WRF
 * exactly. Neighbour reads go through `wall` so a masked (non-burnable)
 * cell's stale value never enters the stencil -- WRF has no analogue to
 * this, it is ERF's own terrain-mask convention layered on top. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real reinit_rhs_cell(amrex::Array4<amrex::Real const> const& lfn_curr,
                             amrex::Array4<amrex::Real const> const& lfn_s0,
                             const WallStencil& wall,
                             int i, int j, int k,
                             amrex::Real dx, amrex::Real dy,
                             amrex::Real threshold_HLu) noexcept
{
    using namespace amrex;
    auto P = [&](int di, int dj) { return wall.at(lfn_curr, i, j, k, di, dj); };
    const Real p_ij = lfn_curr(i, j, k);

    Real diff2x, diff2y;
    if (p_ij < threshold_HLu) {
        const Real d4x = select_4th(dx, p_ij, P(-1,0), P(-2,0), P(1,0), P(2,0));
        const Real d4y = select_4th(dy, p_ij, P(0,-1), P(0,-2), P(0,1), P(0,2));
        diff2x = select_weno5(dx, p_ij, P(-1,0), P(-2,0), P(-3,0),
                               P(1,0), P(2,0), P(3,0), lfn_s0(i,j,k)*d4x);
        diff2y = select_weno5(dy, p_ij, P(0,-1), P(0,-2), P(0,-3),
                               P(0,1), P(0,2), P(0,3), lfn_s0(i,j,k)*d4y);
    } else {
        const Real diffLx = (p_ij - P(-1,0))/dx, diffRx = (P(1,0) - p_ij)/dx;
        const Real diffLy = (p_ij - P(0,-1))/dy, diffRy = (P(0,1) - p_ij)/dy;
        diff2x = select_eno(diffLx, diffRx);
        diff2y = select_eno(diffLy, diffRy);
    }

    const Real grad = std::sqrt(diff2x*diff2x + diff2y*diff2y);
    return lfn_s0(i,j,k) * (1.0_rt - grad);
}

} // namespace wrf_reinit

/**
 * @brief Reinitialize phi via WRF-Fire's reinit_ls_rk3 algorithm (metric
 * signed distance, no normalized/banded mode).
 *
 * @param[in,out] phi        Level-set field [m], modified in place.
 * @param[in]     geom       Domain geometry.
 * @param[in]     n_outer    Number of outer RK3 pseudo-time steps
 *                           (WRF's fire_lsm_reinit_iter; WRF's own comment:
 *                           "1 iter each time step is enough").
 * @param[in]     dtau       Pseudo-timestep [m]. WRF's own default is
 *                           0.01*dx; pass that explicitly, this function has
 *                           no implicit default (the campaign convention is
 *                           to resolve <=0 to 0.01*dx at the call site, the
 *                           same pattern ERF_FireLayer.cpp already uses for
 *                           other geometry-dependent defaults).
 * @param[in]     band_m     WENO5/ENO1 band threshold [m] (WRF's
 *                           threshold_HLu = fire_lsm_band_ngp*dx; pass that
 *                           product directly -- this function takes a
 *                           length, not a cell count, matching how
 *                           ERF_FireLayer.cpp already precomputes
 *                           LevelSetGradient::band for the advection path).
 * @param[in]     mask       Optional non-burnable mask, read only with
 *                           wall_extrap.
 * @param[in]     wall_extrap  Leave mask cells untouched and read
 *                           neighbours through a wall stencil.
 */
inline void reinitialize_phi(amrex::MultiFab& phi,
                              const amrex::Geometry& geom,
                              int n_outer,
                              amrex::Real dtau,
                              amrex::Real band_m,
                              const amrex::MultiFab* mask = nullptr,
                              bool wall_extrap = false)
{
    using namespace amrex;
    const auto* dx = geom.CellSize();
    const Real threshold_HLu = band_m;

    MultiFab phi_in(phi.boxArray(), phi.DistributionMap(), 1, 3);
    phi_in.ParallelCopy(phi); // lfn_in: fixed for the whole call (final clamp reference)

    MultiFab S(phi.boxArray(), phi.DistributionMap(), 1, 3);
    {
        // S(phi0) computed ONCE from the field at call entry, held fixed
        // across every outer iteration and every RK stage.
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p = phi.const_array(mfi);
            auto s = S.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real p0 = p(i, j, k);
                s(i, j, k) = p0 / std::sqrt(p0*p0 + dx[0]*dx[0]);
            });
        }
        fire_fill_boundary(S, geom);
    }

    MultiFab phi0(phi.boxArray(), phi.DistributionMap(), 1, 3);
    MultiFab stage(phi.boxArray(), phi.DistributionMap(), 1, 3);
    MultiFab stage2(phi.boxArray(), phi.DistributionMap(), 1, 3);
    phi0.ParallelCopy(phi); // lfn_s3, the base each outer iteration's 3 RK stages advance from

    for (int outer = 0; outer < n_outer; ++outer) {
        fire_fill_boundary(phi0, geom);

        // Stage 1: phi1 = phi0 + (dtau/3) L(phi0) -> into `stage`.
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage.array(mfi);
            const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (wall.masked(i, j, k)) { out(i, j, k) = p0(i, j, k); return; }
                const Real L = wrf_reinit::reinit_rhs_cell(p0, s, wall, i, j, k, dx[0], dx[1], threshold_HLu);
                out(i, j, k) = p0(i, j, k) + (dtau/3.0_rt) * L;
            });
        }
        fire_fill_boundary(stage, geom);

        // Stage 2: phi2 = phi0 + (dtau/2) L(phi1) -> into `stage2`.
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto p1 = stage.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage2.array(mfi);
            const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (wall.masked(i, j, k)) { out(i, j, k) = p0(i, j, k); return; }
                const Real L = wrf_reinit::reinit_rhs_cell(p1, s, wall, i, j, k, dx[0], dx[1], threshold_HLu);
                out(i, j, k) = p0(i, j, k) + (dtau/2.0_rt) * L;
            });
        }
        fire_fill_boundary(stage2, geom);

        // Stage 3: phi3 = phi0 + dtau*L(phi2) -> written into `stage` (its
        // stage-1 contents are no longer needed; does not alias phi0 or
        // stage2, both only read here).
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto p2 = stage2.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage.array(mfi);
            const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (wall.masked(i, j, k)) { out(i, j, k) = p0(i, j, k); return; }
                const Real L = wrf_reinit::reinit_rhs_cell(p2, s, wall, i, j, k, dx[0], dx[1], threshold_HLu);
                out(i, j, k) = p0(i, j, k) + dtau * L;
            });
        }

        MultiFab::Copy(phi0, stage, 0, 0, 1, 0); // phi3 becomes next outer iteration's base
    }

    // lfn_out = min(lfn_out, lfn_in): reinitialization can never revert an
    // already-burned cell (phi < 0) back to unburned.
    for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto out = phi0.array(mfi);
        auto pin = phi_in.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            out(i, j, k) = amrex::min(out(i, j, k), pin(i, j, k));
        });
    }

    phi.ParallelCopy(phi0);
}

} // namespace fire_levelset

#endif
