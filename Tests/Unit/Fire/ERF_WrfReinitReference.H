#ifndef ERF_WRF_REINIT_REFERENCE_H
#define ERF_WRF_REINIT_REFERENCE_H

#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <ERF_FireGrid.H>

/**
 * @file ERF_WrfReinitReference.H
 * @brief An exact C++/AMReX mirror of WRF-Fire's level-set reinitialization
 * (module_fr_fire_core.F: reinit_ls_rk3 + advance_ls_reinit), for direct
 * side-by-side comparison against ERF's reinitialize_phi() under the SAME
 * T1/T1b static-drift test harness (Tests/Unit/Fire/ERF_GTestReinitDrift.cpp,
 * ERF_GTestReinitAdvectDrift.cpp).
 *
 * This is a REFERENCE / TEST-ONLY implementation, not a production code
 * path. It deliberately reproduces WRF's algorithm exactly as written,
 * including its quirks -- it does NOT "fix" anything ERF's own
 * investigation found questionable in WRF's version. Specifically,
 * faithfully mirrored:
 *
 *   - S(phi0) = phi0/sqrt(phi0^2 + dx^2) computed ONCE from the field at
 *     call entry, held fixed across every outer iteration and RK stage
 *     (module_fr_fire_core.F:1576, ~1592).
 *   - Wicker & Skamarock (2002) 3-stage RK3, NOT Shu-Osher SSP-RK3: all
 *     three stages advance from the SAME fixed base phi0, not additively
 *     from the previous stage --
 *       phi1 = phi0 + (dtau/3) L(phi0)
 *       phi2 = phi0 + (dtau/2) L(phi1)
 *       phi3 = phi0 +  dtau    L(phi2)
 *     (lines 1602, 1623, 1644: all three calls pass lfn_ini=lfn_s3=phi0).
 *   - dtau = 0.01*dx (module_fr_fire_core.F:1593), NOT ERF's
 *     0.25*min(dx,dy) -- 25x smaller by default at dx=dy=50m.
 *   - Hybrid WENO5-ENO1 gradient (fire_upwinding_reinit=4, the campaign's
 *     actual namelist setting, module_fr_fire_util.F default): WENO5 where
 *     lfn_curr(i,j) < threshold_HLu (band_ngp*dx), ENO1 (Godunov 1st-order)
 *     elsewhere. Note the band test uses phi DIRECTLY, NOT abs(phi) --
 *     this is asymmetric (essentially all burned cells plus a band of
 *     unburned cells get WENO5; deep unburned cells get ENO1) and, per
 *     CLAUDE.md's code-comparison notes, looks like an inconsistency
 *     against WRF's own tend_ls (the advection RHS), which DOES use
 *     abs(). Reproduced here exactly as WRF has it, not "fixed".
 *   - WENO5 upwind side selected by sign(S(phi0) * D4th), where D4th is a
 *     4th-order CENTRAL difference of lfn_curr (module_fr_fire_core.F's
 *     select_4th) -- NOT the Godunov (derivative-sign) selection ERF uses.
 *   - Classic Jiang-Shu WENO5 weights (gamma = 1/10, 3/5, 3/10; tol=1e-6
 *     added inside (beta+tol)^2 in phi-units, scale-dependent) on a
 *     FLUX-FORM reconstruction of phi VALUES at cell faces, not ERF's
 *     HJ-WENO5-Z on divided differences.
 *   - NO subcell fix of any kind (no Russo-Smereka, no analogue). This is
 *     the headline structural difference from ERF's reinitialize_phi.
 *   - Final safeguard: lfn_out = min(lfn_out, lfn_in) against the field AT
 *     CALL ENTRY -- identical in spirit to ERF's clamp.
 *
 * Deliberately NOT mirrored (out of scope for a same-domain-interior
 * comparison test, and not expected to matter for T1/T1b's synthetic
 * shapes, which the T1 spec keeps >=20 cells from any domain edge):
 *   - WRF's bdy_eno1=10 forced-ENO1 band within 10 cells of the DOMAIN edge
 *     (module_fr_fire_core.F:1700). Ghost cells here are filled the same
 *     way ERF's own reinitialize_phi() fills them (fire_fill_boundary),
 *     not WRF's continue_at_boundary extrapolation-up-only rule. If a test
 *     ever places a front within 10 cells of the domain edge, this
 *     omission would matter; T1/T1b never do.
 *   - fire_upwinding_reinit cases 1/2/3 (WENO3-only, WENO5-only, hybrid
 *     WENO3-ENO1) -- only case 4 (the campaign's actual setting) is
 *     implemented.
 *
 * @param[in,out] phi        Level-set field (metres, signed distance),
 *                            modified in place -- same convention as
 *                            reinitialize_phi()'s normalized=false mode.
 * @param[in]     geom       Domain geometry (dx,dy read from here; WRF's
 *                            own S(phi0) epsilon and dtau use dx[0] only,
 *                            mirrored exactly -- assumes dx==dy, true for
 *                            the campaign's fire grid).
 * @param[in]     n_outer    fire_lsm_reinit_iter: number of outer RK3
 *                            pseudo-time steps (WRF's own comment: "1 iter
 *                            each time step is enough"; campaign namelist
 *                            uses 1). Exposed here so a test can vary it
 *                            the way T1 varies ERF's n_iters.
 * @param[in]     band_ngp   fire_lsm_band_ngp: WENO5 band half-width in
 *                            cells (campaign namelist: 4).
 */
namespace wrf_reinit_reference {

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real wrf_select_eno (amrex::Real diffL, amrex::Real diffR) noexcept
{
    using namespace amrex;
    if (!(diffL > 0) && !(diffR > 0)) { return diffR; }
    if (!(diffL < 0) && !(diffR < 0)) { return diffL; }
    if (!(diffL < 0) && !(diffR > 0)) {
        return (!(std::abs(diffR) < std::abs(diffL))) ? diffR : diffL;
    }
    return 0.0_rt;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real wrf_select_4th (amrex::Real dx, amrex::Real p_i, amrex::Real p_im1,
                            amrex::Real p_im2, amrex::Real p_ip1, amrex::Real p_ip2) noexcept
{
    using namespace amrex;
    const amrex::Real dp = (7.0_rt*p_ip1 + 7.0_rt*p_i - p_ip2 - p_im1) / (12.0_rt*dx);
    const amrex::Real dm = (7.0_rt*p_i + 7.0_rt*p_im1 - p_ip1 - p_im2) / (12.0_rt*dx);
    return dp - dm;
}

/** Classic Jiang-Shu flux-form WENO5 on phi VALUES (module_fr_fire_core.F's
 * select_weno5), NOT HJ-WENO5-Z on divided differences. uf's sign selects
 * the upwind stencil side, mirroring lfn_it/lfn_im1t/... exactly. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real wrf_select_weno5 (amrex::Real dx,
                               amrex::Real p_i, amrex::Real p_im1, amrex::Real p_im2, amrex::Real p_im3,
                               amrex::Real p_ip1, amrex::Real p_ip2, amrex::Real p_ip3,
                               amrex::Real uf) noexcept
{
    using namespace amrex;
    // Classical 3-stencil Jiang-Shu WENO5 on phi VALUES (module_fr_fire_core.F
    // select_weno5, lines 2301-2366) -- verified against the exact source,
    // not assumed. Note the i+1/2 and i-1/2 faces use DIFFERENT (not simply
    // shifted) fh/beta formulas in WRF's own code; mirrored as-is.
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
 * (hybrid WENO5/ENO1) branch, band test WITHOUT abs() -- mirrors WRF exactly. */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real wrf_reinit_rhs_cell (amrex::Array4<amrex::Real const> const& lfn_curr,
                                  amrex::Array4<amrex::Real const> const& lfn_s0,
                                  int i, int j, int k,
                                  amrex::Real dx, amrex::Real dy,
                                  amrex::Real threshold_HLu) noexcept
{
    using namespace amrex;

    auto P = [&](int ii, int jj) { return lfn_curr(ii, jj, k); };

    Real diff2x, diff2y;
    if (P(i,j) < threshold_HLu) {
        const Real d4x = wrf_select_4th(dx, P(i,j), P(i-1,j), P(i-2,j), P(i+1,j), P(i+2,j));
        const Real d4y = wrf_select_4th(dy, P(i,j), P(i,j-1), P(i,j-2), P(i,j+1), P(i,j+2));
        diff2x = wrf_select_weno5(dx, P(i,j), P(i-1,j), P(i-2,j), P(i-3,j),
                                   P(i+1,j), P(i+2,j), P(i+3,j), lfn_s0(i,j,k)*d4x);
        diff2y = wrf_select_weno5(dy, P(i,j), P(i,j-1), P(i,j-2), P(i,j-3),
                                   P(i,j+1), P(i,j+2), P(i,j+3), lfn_s0(i,j,k)*d4y);
    } else {
        const Real diffLx = (P(i,j)-P(i-1,j))/dx, diffRx = (P(i+1,j)-P(i,j))/dx;
        const Real diffLy = (P(i,j)-P(i,j-1))/dy, diffRy = (P(i,j+1)-P(i,j))/dy;
        diff2x = wrf_select_eno(diffLx, diffRx);
        diff2y = wrf_select_eno(diffLy, diffRy);
    }

    const Real grad = std::sqrt(diff2x*diff2x + diff2y*diff2y);
    return lfn_s0(i,j,k) * (1.0_rt - grad);
}

/** Driver: exact mirror of reinit_ls_rk3. Single-box-friendly (uses
 * fire_fill_boundary for ghosts, like reinitialize_phi() does -- see the
 * file doc comment for what this omits relative to WRF's own boundary
 * handling). */
inline void wrf_reinit_ls_rk3 (amrex::MultiFab& phi,
                                const amrex::Geometry& geom,
                                int n_outer = 1,
                                amrex::Real band_ngp = 4.0)
{
    using namespace amrex;
    const auto* dxa = geom.CellSize();
    const Real dx = dxa[0], dy = dxa[1];
    const Real dtau = 0.01_rt * dx;
    const Real threshold_HLu = band_ngp * dx;

    MultiFab phi_in(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow());
    phi_in.ParallelCopy(phi); // lfn_in: fixed for the whole call (final clamp reference)

    MultiFab S(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow());
    {
        // S(phi0) computed ONCE from the field at call entry, held fixed
        // across every outer iteration and every RK stage.
        for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p = phi.const_array(mfi);
            auto s = S.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real p0 = p(i,j,k);
                s(i,j,k) = p0 / std::sqrt(p0*p0 + dx*dx);
            });
        }
        fire_fill_boundary(S, geom);
    }

    MultiFab phi0(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow());
    MultiFab stage(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow());
    phi0.ParallelCopy(phi); // lfn_s3, the base each outer iteration's 3 RK stages advance from

    for (int outer = 0; outer < n_outer; ++outer) {
        fire_fill_boundary(phi0, geom);

        // Stage 1: phi1 = phi0 + (dtau/3) L(phi0) -> into `stage`
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real L = wrf_reinit_rhs_cell(p0, s, i, j, k, dx, dy, threshold_HLu);
                out(i,j,k) = p0(i,j,k) + (dtau/3.0_rt) * L;
            });
        }
        fire_fill_boundary(stage, geom);

        // Stage 2: phi2 = phi0 + (dtau/2) L(phi1) -> reuse `stage` in place is
        // unsafe (L reads neighbours of `stage` while we'd be overwriting it);
        // write into phi_in's ghost-sized twin scratch instead -- use a
        // second temporary.
        MultiFab stage2(phi.boxArray(), phi.DistributionMap(), 1, phi.nGrow());
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto p1 = stage.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage2.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real L = wrf_reinit_rhs_cell(p1, s, i, j, k, dx, dy, threshold_HLu);
                out(i,j,k) = p0(i,j,k) + (dtau/2.0_rt) * L;
            });
        }
        fire_fill_boundary(stage2, geom);

        // Stage 3: phi3 = phi0 + dtau*L(phi2) -> into `stage` (phi1's old
        // contents no longer needed; does not alias phi0 or stage2, both
        // only read here).
        for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto p0 = phi0.const_array(mfi);
            auto p2 = stage2.const_array(mfi);
            auto s  = S.const_array(mfi);
            auto out = stage.array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const Real L = wrf_reinit_rhs_cell(p2, s, i, j, k, dx, dy, threshold_HLu);
                out(i,j,k) = p0(i,j,k) + dtau * L;
            });
        }

        MultiFab::Copy(phi0, stage, 0, 0, 1, 0); // phi3 becomes next outer iteration's base
    }

    // lfn_out = min(lfn_out, lfn_in)
    for (MFIter mfi(phi0); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto out = phi0.array(mfi);
        auto pin = phi_in.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            out(i,j,k) = amrex::min(out(i,j,k), pin(i,j,k));
        });
    }

    phi.ParallelCopy(phi0);
}

} // namespace wrf_reinit_reference

#endif
