#ifndef ERF_REINITIALIZE_H
#define ERF_REINITIALIZE_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <ERF_NumericalSchemes.H>

/**
 * @file ERF_Reinitialize.H
 * @brief Sussman reinitialization of the level-set function
 *
 * Ported from wildfire_levelset/src/reinitialization.H
 * Resets φ to a normalized signed-distance function via Sussman's method.
 *
 * NOTE ON UNITS. The textbook Sussman update drives |∇φ| → 1, which presumes φ
 * is a signed distance measured in metres. ERF-Hazard's φ is not: it is
 * normalized and clamped to [-1, 1] (initialize_ignition sets φ = (d - r)/r,
 * and the FARSITE path and ember spotting use the same convention). Applying
 * the textbook form to a normalized φ leaves the bracket (1 - |∇φ|) near 1 for
 * every cell, so each iteration added ~dtau and the field diverged by
 * n_iters * dtau per call. The update below instead targets |∇φ| = 1/L for an
 * explicit band length L, so φ reaches ±1 at distance L from the front, which
 * is the convention the rest of the module already uses.
 */

namespace fire_levelset {

namespace detail {

/**
 * @brief Per-cell reinitialization RHS, L(phi), such that one forward-Euler
 * pseudo-time step is phi_new = phi_state + dtau * L(phi_state).
 *
 * Factored out of reinitialize_phi() so the same per-cell physics (Sussman
 * band-normalized update with WRF-Fire's smoothed sign, overridden by the
 * Russo-Smereka subcell update near the front, both referenced against the
 * ORIGINAL phi0 passed to the outer call) can be evaluated at each stage of
 * either forward Euler or TVD RK3 without duplicating the physics.
 *
 * @param[in] phi_state  The current pseudo-time iterate (or RK stage) to read
 *                       derivatives and the cell's own value from.
 * @param[in] phi_in     The ORIGINAL phi passed to reinitialize_phi(), fixed
 *                       for the whole call -- source of the near_front test,
 *                       the exact sign, and the Russo-Smereka distance D.
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real reinit_rhs_cell(amrex::Array4<amrex::Real const> const& phi_state,
                             amrex::Array4<amrex::Real const> const& phi_in,
                             const WallStencil& wall,
                             int i, int j, int k,
                             amrex::Real dx0, amrex::Real dx1,
                             amrex::Real h, amrex::Real hinv,
                             amrex::Real Lval, amrex::Real Linv,
                             const LevelSetGradient& grad,
                             bool disable_subcell_fix = false) noexcept
{
    using namespace amrex;

    // Mask cells are not part of the distance function -- zero rate leaves
    // them unchanged by any stage combination (0.75*m + 0.25*(m+dtau*0) = m,
    // etc.), matching the explicit copy-through the un-refactored code used.
    if (wall.masked(i, j, k)) { return 0.0_rt; }

    Real dxm, dxp, dym, dyp;
    one_sided_derivatives(phi_state, dx0, i, j, k, 1, 0, grad, wall, dxm, dxp);
    one_sided_derivatives(phi_state, dx1, i, j, k, 0, 1, grad, wall, dym, dyp);

    const Real phi0 = phi_in(i, j, k);
    Real sgn_phi0_hard = (phi0 > 0.0) ? 1.0 :
                         (phi0 < 0.0) ? -1.0 : 0.0;

    const Real eps_s = h * Linv;
    Real sgn_phi0_smooth = phi0 / std::sqrt(phi0 * phi0 + eps_s * eps_s);

    Real gx2, gy2;
    if (sgn_phi0_smooth > 0.0_rt) {
        Real a = amrex::max(dxm, 0.0_rt), b = amrex::min(dxp, 0.0_rt);
        gx2 = amrex::max(a * a, b * b);
        Real c = amrex::max(dym, 0.0_rt), d = amrex::min(dyp, 0.0_rt);
        gy2 = amrex::max(c * c, d * d);
    } else {
        Real a = amrex::min(dxm, 0.0_rt), b = amrex::max(dxp, 0.0_rt);
        gx2 = amrex::max(a * a, b * b);
        Real c = amrex::min(dym, 0.0_rt), d = amrex::max(dyp, 0.0_rt);
        gy2 = amrex::max(c * c, d * d);
    }
    Real norm_grad = std::sqrt(gx2 + gy2 + 1.0e-14);

    Real L_rate = sgn_phi0_smooth * (1.0_rt - Lval * norm_grad) * Linv;

    const Real p0c  = phi_in(i, j, k);
    const Real p0xp = wall.at(phi_in, i, j, k, 1, 0);
    const Real p0xm = wall.at(phi_in, i, j, k, -1, 0);
    const Real p0yp = wall.at(phi_in, i, j, k, 0, 1);
    const Real p0ym = wall.at(phi_in, i, j, k, 0, -1);
    const bool near_front = !disable_subcell_fix &&
        ((p0c * p0xp < 0.0_rt) || (p0c * p0xm < 0.0_rt) ||
         (p0c * p0yp < 0.0_rt) || (p0c * p0ym < 0.0_rt));

    if (near_front) {
        const Real d_xp = (p0xp - p0c) * hinv;
        const Real d_xm = (p0c - p0xm) * hinv;
        const Real d_yp = (p0yp - p0c) * hinv;
        const Real d_ym = (p0c - p0ym) * hinv;
        const bool cross_xp = (p0c * p0xp < 0.0_rt);
        const bool cross_xm = (p0c * p0xm < 0.0_rt);
        const bool cross_yp = (p0c * p0yp < 0.0_rt);
        const bool cross_ym = (p0c * p0ym < 0.0_rt);

        auto minmod = [](Real a, Real b) {
            if (a * b <= 0.0_rt) { return 0.0_rt; }
            return (std::abs(a) < std::abs(b)) ? a : b;
        };
        auto axis_grad = [&](bool cp, bool cm, Real dp, Real dm) {
            if (cp && cm) { return (std::abs(dp) > std::abs(dm)) ? dp : dm; }
            if (cp) { return dp; }
            if (cm) { return dm; }
            return minmod(dm, dp);
        };
        const Real gx = axis_grad(cross_xp, cross_xm, d_xp, d_xm);
        const Real gy = axis_grad(cross_yp, cross_ym, d_yp, d_ym);
        const Real gnorm = std::sqrt(gx * gx + gy * gy + 1.0e-14_rt);
        const Real D = p0c / gnorm;

        L_rate = -hinv * (sgn_phi0_hard * std::abs(phi_state(i, j, k)) - D * Linv);
    }

    return L_rate;
}

} // namespace detail

/**
 * @brief Reinitialize φ to a normalized signed-distance function
 *
 * Solves the band-normalized reinitialization PDE
 *
 *   ∂φ/∂τ = S(φ₀) * (1 - L |∇φ|) / L
 *
 * by explicit Euler in pseudo-time (default) or, if `tvd_rk3` is set, by the
 * 3-stage SSP/TVD Runge-Kutta scheme Jiang & Peng (2000) use for this same
 * equation (their eqs. (3.10)-(3.12) sketch it for the pure Godunov RHS; here
 * the same Shu-Osher time integrator is applied to ERF's actual RHS, i.e.
 * WITH the smoothed-sign Sussman forcing and the Russo-Smereka near-front
 * override, not the textbook RHS in isolation):
 *
 *   phi1 = phi0 + dtau L(phi0)
 *   phi2 = 3/4 phi0 + 1/4 (phi1 + dtau L(phi1))
 *   phi3 = 1/3 phi0 + 2/3 (phi2 + dtau L(phi2))
 *
 * where phi0 is the state at the start of the pseudo-time step (one call is
 * n_iters such steps) and L is factored out in fire_levelset::detail::
 * reinit_rhs_cell(). Each RK stage is clamped exactly like each Euler
 * iteration was (see below).
 *
 * S(φ₀) = φ₀/sqrt(φ₀²+eps²) is the
 * WRF-Fire-style smoothed sign (module_fr_fire_core.F's reinit_ls_rk3), not
 * the exact ±1 sign -- it tapers the forcing continuously to zero across the
 * front instead of switching at a single point. The fixed point is
 * |∇φ| = 1/L, i.e. φ varies linearly from 0 at the front to ±1 at distance L,
 * matching the normalization initialize_ignition applies. φ is clamped to
 * [-1, 1] after every iteration, which pins the far field (where |∇φ| = 0 and
 * the update would otherwise push φ past ±1 indefinitely).
 *
 * Cells whose original neighbourhood straddles the interface use the
 * Russo & Smereka (2000) subcell update instead, driven by the *exact* sign
 * of φ₀ (the smoothed sign would scale down this correction exactly where it
 * matters most). This fixes the interface location from φ₀ rather than
 * letting the pseudo-time iteration move it. Without it each pass erodes the
 * burned area, and because the level-set path never rebuilds φ from
 * fire_arrival_time the loss is permanent.
 *
 * After all iterations, a WRF-Fire-style monotonicity safeguard
 * (lfn_out = min(lfn_out, lfn_in), "fire area can only increase") clamps the
 * result against the pre-reinit field so reinitialization can never revert an
 * already-burned cell (φ < 0) back to unburned.
 *
 * A sign-preserving alternative was tried and rejected (2026-09-25): only
 * reverting cells where reinit flipped which side of the front they were on,
 * instead of clamping every cell to be no larger than before, lets error
 * accumulate without bound over repeated reinit calls (reinit_every=1, one
 * call per fire subcycle) -- on LineFireFlat/LineFireFlat34 it produced a
 * severe, unphysical wedge cutting across the burned interior within 600 s,
 * a strictly worse regression, not an improvement. The blunt min-clamp
 * being one-sided everywhere is a real defect (the failing upstream test
 * LevelSetAdvection.ReinitialisationRestoresUnitGradient documents it), but
 * whatever the correct fix is, it needs its own bound on error growth; a bare
 * sign check is not enough. Keep the min-clamp until a better-bounded
 * alternative is found. See CLAUDE.md's reinit review notes.
 *
 * Each iteration moves φ by at most dtau/L, so with the default
 * dtau = 0.5*min(dx,dy) the change per iteration is half a cell measured in
 * band lengths.
 *
 * @param[in,out] phi      Level-set field; modified in-place
 * @param[in]     geom     Domain geometry
 * @param[in]     n_iters  Number of Sussman iterations
 * @param[in]     dtau     Pseudo-timestep [m] for reinitialization
 * @param[in]     band_m   Band half-width L [m]: the distance from the front at
 *                         which φ reaches ±1. Values <= 0 select 3*min(dx,dy).
 * @param[in]     mask     Optional non-burnable mask, read only with wall_extrap
 * @param[in]     wall_extrap  Leave mask cells untouched and read neighbours
 *                         through a wall stencil, so the wall is a zero-gradient
 *                         boundary of the distance function rather than a front
 * @param[in]     skip_final_clamp  Diagnostic only (default false = current,
 *                         confirmed behavior). true disables the final
 *                         min(phi_out, phi_in) monotonicity safeguard below,
 *                         to test whether it is responsible for a slow,
 *                         one-sided drift accumulating over many reinit calls.
 *                         A prior sign-preserving-clamp variant of removing
 *                         this safeguard was tried and rejected (2026-09-25,
 *                         see CLAUDE.md): it let error run away unboundedly
 *                         within 600s at reinit_every=1. This flag removes the
 *                         safeguard entirely (no replacement clamp at all),
 *                         which is expected to be at least as unstable -- it
 *                         exists to confirm/quantify that failure mode, not as
 *                         a candidate fix.
 * @param[in]     tvd_rk3  Default false = forward Euler (current, confirmed
 *                         behavior, bit-for-bit unchanged). true replaces each
 *                         pseudo-time iteration with a 3-stage SSP/TVD RK3
 *                         step (Jiang & Peng's recommended time integrator for
 *                         this equation) using the SAME per-cell RHS -- see
 *                         the file doc comment above.
 * @param[in]     disable_subcell_fix  Diagnostic only (default false =
 *                         current, confirmed behavior). true disables the
 *                         Russo-Smereka near-front override entirely, so
 *                         EVERY cell (near-front or not) uses the general
 *                         Sussman/Godunov update -- isolates whether the
 *                         subcell branch itself is responsible for the
 *                         non-saturating curvature-driven creep on curved
 *                         fronts that WRF's reference implementation
 *                         (which has no subcell fix at all) does not show
 *                         at matched dtau. See CLAUDE.md's 2026-09-26
 *                         addendum for the test this was built for.
 */
inline void reinitialize_phi(amrex::MultiFab& phi,
                              const amrex::Geometry& geom,
                              int n_iters,
                              amrex::Real dtau,
                              amrex::Real band_m = -1.0,
                              bool normalized = true,
                              const amrex::MultiFab* mask = nullptr,
                              bool wall_extrap = false,
                              const LevelSetGradient& grad = LevelSetGradient{},
                              bool skip_final_clamp = false,
                              bool tvd_rk3 = false,
                              bool disable_subcell_fix = false)
{
    using namespace amrex;

    const auto* dx = geom.CellSize();

    // Band half-width. The default spreads the [-1, 1] range over three cells,
    // the usual interface thickness for a level-set band.
    // In signed-distance mode phi is already in metres, so the band length is
    // 1 m: the update targets |grad phi| = 1 and nothing is clamped. In
    // normalized mode phi is dimensionless and reaches +/-1 at distance L.
    const Real L    = normalized ? ((band_m > 0.0) ? band_m
                                                  : 3.0 * amrex::min(dx[0], dx[1]))
                                 : 1.0;
    const Real Linv = 1.0 / L;

    // Isotropic spacing for the Russo-Smereka subcell term. The fire grid is
    // square in practice; min() is the conservative choice if it ever is not.
    const Real h    = amrex::min(dx[0], dx[1]);
    const Real hinv = 1.0 / h;
    AMREX_ASSERT(L > 0.0 && h > 0.0);

    // Make a working copy for pseudo-time iteration
    MultiFab phi_work(phi.boxArray(), phi.DistributionMap(), 1, 3);
    phi_work.ParallelCopy(phi);

    // Jacobi scratch. The update below reads phi_work at i±1 / j±1 while writing
    // the centre cell, so writing back into phi_work in place would be a
    // read-write race between neighbouring GPU threads -- each cell would see
    // either the old or the new neighbour depending on scheduling, making the
    // result non-deterministic and different from the CPU sweep. Write into a
    // separate fab and swap once the pass is complete.
    MultiFab phi_next(phi.boxArray(), phi.DistributionMap(), 1, 3);
    // Freshly allocated fabs hold indeterminate values. Only the valid region is
    // written and only the valid region is copied back, but leaving the ghosts
    // indeterminate makes the run depend on whatever the allocator hands over.
    phi_next.setVal(0.0);

    // Second stage buffer, used only by the tvd_rk3 path (phi2 of the
    // 3-stage step; phi1 reuses phi_next, phi3 is written back into phi_next
    // once phi1's contents are no longer needed -- see below). Always
    // allocated for simplicity; the cost is negligible next to phi_work/
    // phi_next above and this keeps the euler path's allocations identical
    // to before this change.
    MultiFab phi_stage2(phi.boxArray(), phi.DistributionMap(), 1, 3);
    if (tvd_rk3) { phi_stage2.setVal(0.0); }

    for (int iter = 0; iter < n_iters; ++iter) {
        // Fill boundary ghosts
        fire_fill_boundary(phi_work, geom);

        if (tvd_rk3) {
            // 3-stage SSP/TVD RK3 pseudo-time step -- see file doc comment.
            // phi0 = phi_work (fixed reference for this step, read but not
            // written until the very end).

            // Stage 1: phi1 = phi0 + dtau*L(phi0), into phi_next.
            for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto phi0_arr = phi_work.const_array(mfi);
                auto phi_in   = phi.const_array(mfi);
                auto out      = phi_next.array(mfi);
                const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                    int i = iv[0], j = iv[1], k = iv[2];
                    if (wall.masked(i, j, k)) { out(i, j, k) = phi0_arr(i, j, k); return; }
                    Real Lr = detail::reinit_rhs_cell(phi0_arr, phi_in, wall, i, j, k,
                                                       dx[0], dx[1], h, hinv, L, Linv, grad, disable_subcell_fix);
                    Real v = phi0_arr(i, j, k) + dtau * Lr;
                    out(i, j, k) = normalized ? amrex::max(-1.0_rt, amrex::min(1.0_rt, v)) : v;
                });
            }
            fire_fill_boundary(phi_next, geom);

            // Stage 2: phi2 = 3/4 phi0 + 1/4 (phi1 + dtau*L(phi1)), into phi_stage2.
            for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto phi0_arr = phi_work.const_array(mfi);
                auto phi1_arr = phi_next.const_array(mfi);
                auto phi_in   = phi.const_array(mfi);
                auto out      = phi_stage2.array(mfi);
                const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                    int i = iv[0], j = iv[1], k = iv[2];
                    if (wall.masked(i, j, k)) { out(i, j, k) = phi0_arr(i, j, k); return; }
                    Real Lr = detail::reinit_rhs_cell(phi1_arr, phi_in, wall, i, j, k,
                                                       dx[0], dx[1], h, hinv, L, Linv, grad, disable_subcell_fix);
                    Real v = 0.75_rt * phi0_arr(i, j, k)
                           + 0.25_rt * (phi1_arr(i, j, k) + dtau * Lr);
                    out(i, j, k) = normalized ? amrex::max(-1.0_rt, amrex::min(1.0_rt, v)) : v;
                });
            }
            fire_fill_boundary(phi_stage2, geom);

            // Stage 3: phi3 = 1/3 phi0 + 2/3 (phi2 + dtau*L(phi2)), written
            // into phi_next -- phi1's contents there are no longer needed,
            // and this does not alias phi_work (phi0) or phi_stage2 (phi2),
            // which are only read in this pass.
            for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
                const Box& bx = mfi.validbox();
                auto phi0_arr = phi_work.const_array(mfi);
                auto phi2_arr = phi_stage2.const_array(mfi);
                auto phi_in   = phi.const_array(mfi);
                auto out      = phi_next.array(mfi);
                const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                    int i = iv[0], j = iv[1], k = iv[2];
                    if (wall.masked(i, j, k)) { out(i, j, k) = phi0_arr(i, j, k); return; }
                    Real Lr = detail::reinit_rhs_cell(phi2_arr, phi_in, wall, i, j, k,
                                                       dx[0], dx[1], h, hinv, L, Linv, grad, disable_subcell_fix);
                    Real v = (1.0_rt/3.0_rt) * phi0_arr(i, j, k)
                           + (2.0_rt/3.0_rt) * (phi2_arr(i, j, k) + dtau * Lr);
                    out(i, j, k) = normalized ? amrex::max(-1.0_rt, amrex::min(1.0_rt, v)) : v;
                });
            }

            MultiFab::Copy(phi_work, phi_next, 0, 0, 1, 0);
            continue;
        }

        // Update RHS via Sussman reinitialization
        for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto phi_arr = phi_work.const_array(mfi);
            auto phi_out = phi_next.array(mfi);
            auto phi_in  = phi.const_array(mfi);
            const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                int i = iv[0], j = iv[1], k = iv[2];

                // Mask cells are not part of the distance function: they keep
                // their value and their neighbours read through the wall.
                if (wall.masked(i, j, k)) { phi_out(i, j, k) = phi_arr(i, j, k); return; }

                // One-sided differences for the gradient, first order or
                // HJ-WENO5-Z as the advection uses (WRF-Fire solves the
                // reinitialisation with the same hybrid scheme).
                Real dxm, dxp, dym, dyp;
                one_sided_derivatives(phi_arr, dx[0], i, j, k, 1, 0, grad, wall, dxm, dxp);
                one_sided_derivatives(phi_arr, dx[1], i, j, k, 0, 1, grad, wall, dym, dyp);

                // Exact sign of the original level-set. Used only by the
                // Russo-Smereka subcell block below, which must combine
                // crossing distances with an exact +/-1 direction -- feeding
                // it the smoothed sign (next) would scale down the very
                // sub-cell correction that block exists to get right, since
                // near_front cells are exactly where the smoothed sign's
                // magnitude is smallest.
                const Real phi0 = phi_in(i, j, k);
                Real sgn_phi0_hard = (phi0 > 0.0) ? 1.0 :
                                     (phi0 < 0.0) ? -1.0 : 0.0;

                // WRF-Fire style smoothed sign, S(phi0) = phi0/sqrt(phi0^2+eps^2)
                // (module_fr_fire_core.F's reinit_ls_rk3), used for the general
                // (non-Russo-Smereka) update below. eps is one grid cell's
                // worth of phi-units: h*Linv is h itself (WRF's dx) in
                // signed-distance mode where L=1, and h/L in normalized mode.
                // This only damps the RHS magnitude near the front -- the sign
                // of sgn_phi0_smooth always matches sgn_phi0_hard, so it does
                // not change which upwind branch is selected below.
                const Real eps_s = h * Linv;
                Real sgn_phi0_smooth = phi0 / std::sqrt(phi0 * phi0 + eps_s * eps_s);

                // Godunov upwind norm for the reinitialization Hamiltonian.
                // The one-sided difference is selected by the sign of phi_0, not
                // by which has the larger magnitude: taking the larger magnitude
                // unconditionally is not the Godunov scheme and is unstable near
                // extrema, which showed up as the front flooding the domain once
                // enough pseudo-time iterations were taken.
                Real gx2, gy2;
                if (sgn_phi0_smooth > 0.0_rt) {
                    Real a = amrex::max(dxm, 0.0_rt), b = amrex::min(dxp, 0.0_rt);
                    gx2 = amrex::max(a * a, b * b);
                    Real c = amrex::max(dym, 0.0_rt), d = amrex::min(dyp, 0.0_rt);
                    gy2 = amrex::max(c * c, d * d);
                } else {
                    Real a = amrex::min(dxm, 0.0_rt), b = amrex::max(dxp, 0.0_rt);
                    gx2 = amrex::max(a * a, b * b);
                    Real c = amrex::min(dym, 0.0_rt), d = amrex::max(dyp, 0.0_rt);
                    gy2 = amrex::max(c * c, d * d);
                }
                Real norm_grad = std::sqrt(gx2 + gy2 + 1.0e-14);

                // Band-normalized Sussman RHS: S(φ₀) * (1 - L|∇φ|) / L, using
                // the smoothed sign so the forcing tapers continuously to zero
                // across the front rather than switching at a single point.
                Real rhs = sgn_phi0_smooth * (1.0_rt - L * norm_grad) * Linv;

                Real phi_new = phi_arr(i, j, k) + dtau * rhs;

                // Russo-Smereka (2000) subcell fix.
                //
                // The plain Sussman iteration does not preserve the zero level
                // set, so each pass walks the front and permanently erodes the
                // burned area -- the level-set path never rebuilds phi from
                // fire_arrival_time the way the FARSITE path does, so nothing
                // puts it back. For cells whose ORIGINAL neighbourhood straddles
                // the interface, replace the Godunov update with one driven by
                // the distance to the interface implied by phi_0 itself, so the
                // interface location is set by phi_0 and not by the iteration.
                const Real p0c = phi_in(i, j, k);
                const Real p0xp = wall.at(phi_in, i, j, k, 1, 0);
                const Real p0xm = wall.at(phi_in, i, j, k, -1, 0);
                const Real p0yp = wall.at(phi_in, i, j, k, 0, 1);
                const Real p0ym = wall.at(phi_in, i, j, k, 0, -1);
                const bool near_front = !disable_subcell_fix &&
                    ((p0c * p0xp < 0.0_rt) || (p0c * p0xm < 0.0_rt) ||
                     (p0c * p0yp < 0.0_rt) || (p0c * p0ym < 0.0_rt));

                if (near_front) {
                    // Per-axis gradient of phi_0, then D = phi_0 / |g| --
                    // exact for a planar front at ANY angle to the grid,
                    // unlike the reciprocal-sum rule this replaces.
                    //
                    // The reciprocal-sum rule (1/D^2 = sum 1/D_dir^2 over
                    // crossing directions) is exact only when BOTH an x- and
                    // a y-neighbour cross: on an oblique front many cells
                    // cross in only ONE axis, where the true distance is the
                    // axis intercept divided by cos(theta), not the intercept
                    // itself -- the old rule used the bare intercept, so it
                    // came out ~31% too large at 40deg from grid-aligned. It
                    // also DOUBLE-counts when both x+ and x- cross (thin
                    // strips, converging fronts near a corner), where the sum
                    // of two reciprocal squares overweights a single true
                    // crossing, making D too small by ~sqrt(2). Building an
                    // actual per-axis gradient component (one-sided
                    // difference toward whichever neighbour is nearer when
                    // both cross that axis, the lone crossing neighbour's
                    // difference when only one does, or a minmod of the two
                    // one-sided differences when neither does -- so the
                    // gradient stays defined and bounded even on axes with
                    // no crossing at all) and dividing phi_0 by its norm
                    // reduces to the ordinary planar-front distance formula
                    // at any orientation, and only reduces to the old
                    // reciprocal-sum answer in the special case both axes
                    // cross with equal steepness.
                    const Real d_xp = (p0xp - p0c) * hinv;  // forward one-sided derivative
                    const Real d_xm = (p0c - p0xm) * hinv;  // backward one-sided derivative
                    const Real d_yp = (p0yp - p0c) * hinv;
                    const Real d_ym = (p0c - p0ym) * hinv;
                    const bool cross_xp = (p0c * p0xp < 0.0_rt);
                    const bool cross_xm = (p0c * p0xm < 0.0_rt);
                    const bool cross_yp = (p0c * p0yp < 0.0_rt);
                    const bool cross_ym = (p0c * p0ym < 0.0_rt);

                    auto minmod = [](Real a, Real b) {
                        if (a * b <= 0.0_rt) { return 0.0_rt; }
                        return (std::abs(a) < std::abs(b)) ? a : b;
                    };
                    // "Nearer" crossing when both sides cross: the interface
                    // is closer along the steeper (larger-magnitude) one-sided
                    // difference (Ddir = h*p0c/diff shrinks as |diff| grows).
                    auto axis_grad = [&](bool cp, bool cm, Real dp, Real dm) {
                        if (cp && cm) { return (std::abs(dp) > std::abs(dm)) ? dp : dm; }
                        if (cp) { return dp; }
                        if (cm) { return dm; }
                        return minmod(dm, dp);
                    };
                    const Real gx = axis_grad(cross_xp, cross_xm, d_xp, d_xm);
                    const Real gy = axis_grad(cross_yp, cross_ym, d_yp, d_ym);
                    const Real gnorm = std::sqrt(gx * gx + gy * gy + 1.0e-14_rt);
                    const Real D = p0c / gnorm;

                    phi_new = phi_arr(i, j, k)
                            - dtau * hinv * (sgn_phi0_hard * std::abs(phi_arr(i, j, k)) - D * Linv);
                }

                // Clamp only in normalized mode, where it pins the far field:
                // there |grad phi| = 0 and the update would otherwise drive phi
                // past +/-1 without limit. In signed-distance mode phi is meant
                // to keep growing with distance, so clamping is exactly the
                // thing that breaks it.
                phi_out(i, j, k) = normalized
                    ? amrex::max(-1.0_rt, amrex::min(1.0_rt, phi_new))
                    : phi_new;
            });
        }

        // phi_next now holds the whole updated valid region; make it the input
        // for the next sweep. Ghost cells are refilled at the top of the loop.
        MultiFab::Copy(phi_work, phi_next, 0, 0, 1, 0);
    }

    // WRF-Fire style monotonicity safeguard (module_fr_fire_core.F's
    // reinit_ls_rk3: "lfn_out=min(lfn_out,lfn_in) ! fire area can only
    // increase"). phi < 0 is burned, so taking the elementwise min against
    // the pre-reinit field can only make a cell more (or equally) burned --
    // reinitialization can never revert an already-burned cell back to
    // unburned, regardless of how the iterations above moved it. This is a
    // blunt backstop on top of, not instead of, the Russo-Smereka fix above.
    //
    // A sign-preserving alternative (only revert cells where reinit flipped
    // which side of the front they were on, instead of clamping every cell)
    // was tried and rejected: it let error accumulate unbounded over 2400
    // steps of every-substep reinit, producing a severe wedge-shaped
    // interior distortion on LineFireFlat/LineFireFlat34 within 600 s --
    // strictly worse than this blunt clamp, not an improvement. See the
    // file-comment note above and CLAUDE.md's reinit review.
    if (!skip_final_clamp) {
        for (MFIter mfi(phi_work); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto phi_arr = phi_work.array(mfi);
            auto phi_orig = phi.const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
                int i = iv[0], j = iv[1], k = iv[2];
                phi_arr(i, j, k) = amrex::min(phi_arr(i, j, k), phi_orig(i, j, k));
            });
        }
    }

    // Copy result back to phi
    phi.ParallelCopy(phi_work);
}

} // namespace fire_levelset

#endif
