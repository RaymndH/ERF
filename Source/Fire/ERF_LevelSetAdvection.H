#ifndef ERF_LEVELSET_ADVECTION_H
#define ERF_LEVELSET_ADVECTION_H

#include <AMReX_MultiFab.H>
#include <ERF_FireGrid.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_NumericalSchemes.H>

/**
 * @file ERF_LevelSetAdvection.H
 * @brief Level-set advection using Godunov upwinding and SSP-RK3
 *
 * Ported from wildfire_levelset/src/advection.H
 * Solves ∂φ/∂t = −R(x,y) * (|∇φ| − ε Δφ) via:
 *   - First-order Godunov upwind |grad phi| for spatial discretization
 *   - 3-stage Strong Stability Preserving RK3 for time integration
 */

namespace fire_levelset {

/**
 * @brief Compute level-set RHS: ∂φ/∂t = −R * (|∇φ| − ε Δφ)
 *
 * Uses the Godunov upwind norm for |∇φ| and central differences for the
 * Laplacian. The one-sided derivatives are first order, or HJ-WENO5-Z with
 * a WENO `grad_scheme`, which reads all three ghost cells of
 * the stage fields.
 *
 * @param[out] rhs         RHS field (modified in-place)
 * @param[in]  phi         Level-set field with ghosts
 * @param[in]  R_mf        Rate-of-spread field
 * @param[in]  dx, dy      Cell sizes
 * @param[in]  eps_visc    Artificial viscosity coefficient away from the front
 *                         (grad_scheme carries the near-front value, if any)
 */
inline void compute_levelset_rhs(amrex::MultiFab& rhs,
                                  const amrex::MultiFab& phi,
                                  const amrex::MultiFab& R_mf,
                                  amrex::Real dx, amrex::Real dy,
                                  amrex::Real eps_visc,
                                  const amrex::MultiFab* slopes = nullptr,
                                  const amrex::MultiFab* mask = nullptr,
                                  bool wall_extrap = false,
                                  const LevelSetGradient& grad_scheme = LevelSetGradient{})
{
    using namespace amrex;

    const Real dx2i = 1.0 / (dx * dx);
    const Real dy2i = 1.0 / (dy * dy);

    const bool use_terrain = (slopes != nullptr);

    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto rhs_arr = rhs.array(mfi);
        auto phi_arr = phi.const_array(mfi);
        auto R_arr = R_mf.const_array(mfi);
        const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

        Array4<const Real> slope_arr;
        if (use_terrain) slope_arr = slopes->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            int i = iv[0], j = iv[1], k = iv[2];

            // Compute |∇φ|, projected onto the terrain surface when slopes are
            // supplied so that R is a spread rate along the ground rather than
            // in map view.
            const Real sx = use_terrain ? slope_arr(i, j, k, 0) : Real(0.0);
            const Real sy = use_terrain ? slope_arr(i, j, k, 1) : Real(0.0);
            Real norm_grad = fire_levelset::godunov_norm_grad_phi(phi_arr, dx, dy,
                                                                  i, j, k, sx, sy, wall,
                                                                  grad_scheme);

            // Compute Δφ = ∂²φ/∂x² + ∂²φ/∂y², through the same wall stencil
            Real laplacian = (wall.at(phi_arr, i, j, k, 1, 0) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, -1, 0)) * dx2i
                           + (wall.at(phi_arr, i, j, k, 0, 1) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, 0, -1)) * dy2i;

            // RHS: -R * (|∇φ| - ε Δφ), with ε the near-front value inside
            // the viscosity band when two values are set
            Real R_val = R_arr(i, j, k);
            const Real eps = grad_scheme.eps_at(phi_arr(i, j, k), eps_visc);
            rhs_arr(i, j, k) = -R_val * (norm_grad - eps * laplacian);
        });
    }
}

/**
 * @brief Compute level-set RHS with the wind/slope advective term upwinded
 * SEPARATELY from the isotropic term, instead of folding both into one
 * scalar R(n) multiplied by a single Godunov |grad phi| (compute_levelset_rhs
 * above).
 *
 * Only valid for erf.fire.directional_wind_coupling = "advective" with
 * directional_shape = "projection" and ros_model = "rothermel", where the
 * physical R(n)|grad phi| reduces exactly to a genuine two-term Hamiltonian:
 *
 *   phi_t = -R0*|grad phi| - max(Vw.grad phi, 0) - max(Vs.grad phi, 0)
 *
 * with Vw = R0*phi_w(|U|)/|U| * U (wind) and Vs = R0*phi_s(|slope|)/|slope| *
 * slope (terrain), each clamped to its own max(.,0) exactly as
 * rothermel_ros_cell's phi_w/phi_s cosine clamps do. R0 and Vw/Vs are known a
 * priori from the wind/slope fields (independent of phi), so each advective
 * term is upwinded by the SIGN OF ITS OWN V component-by-component -- ordinary
 * linear-advection upwind, not a front-normal-derived rule. See
 * ERF_DirectionalRos.H's advect_levelset_directional_rk3_split() and its
 * DirectionalRosParams::directional_split_hamiltonian doc comment for the
 * full rationale (this fixes a structurally different mismatch than Variant
 * G's front-normal construction, which tested negative for the flat-terrain
 * rotation-invariance bulge).
 *
 * @param[out] rhs      RHS field (modified in-place)
 * @param[in]  phi      Level-set field with ghosts
 * @param[in]  R0       Isotropic (no-wind, no-slope) rate of spread [m/s]
 * @param[in]  VwVs     4-component field: (Vwx, Vwy, Vsx, Vsy) [m/s], built
 *                      once from wind/slope (independent of phi)
 * @param[in]  dx, dy   Cell sizes
 * @param[in]  eps_visc Artificial viscosity coefficient
 */
inline void compute_levelset_rhs_split_advective(amrex::MultiFab& rhs,
                                                  const amrex::MultiFab& phi,
                                                  amrex::Real R0,
                                                  const amrex::MultiFab& VwVs,
                                                  amrex::Real dx, amrex::Real dy,
                                                  amrex::Real eps_visc,
                                                  const amrex::MultiFab* slopes = nullptr,
                                                  const amrex::MultiFab* mask = nullptr,
                                                  bool wall_extrap = false,
                                                  const LevelSetGradient& grad_scheme = LevelSetGradient{})
{
    using namespace amrex;

    const Real dx2i = 1.0 / (dx * dx);
    const Real dy2i = 1.0 / (dy * dy);
    const bool use_terrain = (slopes != nullptr);

    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto rhs_arr = rhs.array(mfi);
        auto phi_arr = phi.const_array(mfi);
        auto v_arr = VwVs.const_array(mfi);
        const WallStencil wall = make_wall_stencil(mask, wall_extrap, mfi);

        Array4<const Real> slope_arr;
        if (use_terrain) slope_arr = slopes->const_array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            int i = iv[0], j = iv[1], k = iv[2];

            const Real sx = use_terrain ? slope_arr(i, j, k, 0) : Real(0.0);
            const Real sy = use_terrain ? slope_arr(i, j, k, 1) : Real(0.0);
            Real norm_grad = fire_levelset::godunov_norm_grad_phi(phi_arr, dx, dy,
                                                                  i, j, k, sx, sy, wall,
                                                                  grad_scheme);

            Real laplacian = (wall.at(phi_arr, i, j, k, 1, 0) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, -1, 0)) * dx2i
                           + (wall.at(phi_arr, i, j, k, 0, 1) - 2.0*phi_arr(i, j, k)
                              + wall.at(phi_arr, i, j, k, 0, -1)) * dy2i;
            const Real eps = grad_scheme.eps_at(phi_arr(i, j, k), eps_visc);

            // Each advective term upwinded by the sign of its OWN velocity
            // component, independent of the front normal -- ordinary
            // linear-advection upwind (one_sided_derivatives returns D- in
            // dxm/dym, D+ in dxp/dyp; for phi_t + a*phi_x = 0 with a > 0 the
            // correct upwind derivative is the backward/left-biased one, D-).
            Real dxm, dxp, dym, dyp;
            fire_levelset::one_sided_derivatives(phi_arr, dx, i, j, k, 1, 0, grad_scheme, wall, dxm, dxp);
            fire_levelset::one_sided_derivatives(phi_arr, dy, i, j, k, 0, 1, grad_scheme, wall, dym, dyp);

            const Real Vwx = v_arr(i, j, k, 0), Vwy = v_arr(i, j, k, 1);
            const Real Vsx = v_arr(i, j, k, 2), Vsy = v_arr(i, j, k, 3);

            const Real phix_w = (Vwx > Real(0.0)) ? dxm : dxp;
            const Real phiy_w = (Vwy > Real(0.0)) ? dym : dyp;
            const Real adv_w = amrex::max(Vwx*phix_w + Vwy*phiy_w, Real(0.0));

            const Real phix_s = (Vsx > Real(0.0)) ? dxm : dxp;
            const Real phiy_s = (Vsy > Real(0.0)) ? dym : dyp;
            const Real adv_s = amrex::max(Vsx*phix_s + Vsy*phiy_s, Real(0.0));

            rhs_arr(i, j, k) = -R0*(norm_grad - eps*laplacian) - adv_w - adv_s;
        });
    }
}

/**
 * @brief Zero a rate-of-spread field where a non-burnable mask is set
 *
 * @param[in,out] R     Rate of spread [m/s]
 * @param[in]     mask  Non-burnable mask (>0.5 = non-burnable), or nullptr for no-op
 */
inline void zero_ros_in_mask(amrex::MultiFab& R, const amrex::MultiFab* mask)
{
    if (mask == nullptr) { return; }
    for (amrex::MFIter mfi(R, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& r = R.array(mfi);
        auto const& m = mask->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (m(i, j, k) > amrex::Real(0.5)) { r(i, j, k) = amrex::Real(0.0); }
        });
    }
}

/**
 * @brief Multiply a rate-of-spread field by a per-cell factor
 *
 * @param[in,out] R      Rate of spread [m/s]
 * @param[in]     scale  Factor on the same grid, or nullptr for no-op
 */
inline void scale_ros_field(amrex::MultiFab& R, const amrex::MultiFab* scale)
{
    if (scale == nullptr) { return; }
    for (amrex::MFIter mfi(R, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& r = R.array(mfi);
        auto const& s = scale->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            r(i, j, k) *= s(i, j, k);
        });
    }
}

/**
 * @brief Hold the level set at or above `value` where a non-burnable mask is set
 *
 * @param[in,out] phi    Level-set field
 * @param[in]     mask   Non-burnable mask (>0.5 = non-burnable), or nullptr for no-op
 * @param[in]     value  Level-set value kept in mask cells (positive = unburned)
 */
inline void hold_phi_in_mask(amrex::MultiFab& phi, const amrex::MultiFab* mask, amrex::Real value)
{
    if (mask == nullptr) { return; }
    for (amrex::MFIter mfi(phi, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.growntilebox(amrex::min(phi.nGrowVect(), mask->nGrowVect()));
        auto const& p = phi.array(mfi);
        auto const& m = mask->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            if (m(i, j, k) > amrex::Real(0.5)) { p(i, j, k) = amrex::max(p(i, j, k), value); }
        });
    }
}

/**
 * @brief SSP-RK3 stage updates on the valid cells
 *
 * Kept out of advect_levelset_rk3_with_fill: CUDA rejects a device lambda
 * inside a function template whose template argument is a caller's local
 * lambda type, which is how every ROS model passes its fill_ros.
 */
/// Stage 1: phi* = phi* + dt L(phi*)
inline void rk3_stage1_update(amrex::MultiFab& phi_star, const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi_star); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ps = phi_star.array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ps(iv) = ps(iv) + dt * r(iv);
        });
    }
}

/// Stage 2: phi** = (3/4) phi^n + (1/4)(phi* + dt L(phi*)), with phi_hat holding phi^n on entry
inline void rk3_stage2_update(amrex::MultiFab& phi_hat, const amrex::MultiFab& phi_star,
                              const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi_hat); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ph = phi_hat.array(mfi);
        auto ps = phi_star.const_array(mfi);
        auto r  = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ph(iv) = 0.75_rt * ph(iv) + 0.25_rt * (ps(iv) + dt * r(iv));
        });
    }
}

/// Stage 3: phi^{n+1} = (1/3) phi^n + (2/3)(phi** + dt L(phi**))
inline void rk3_stage3_update(amrex::MultiFab& phi, const amrex::MultiFab& phi_hat,
                              const amrex::MultiFab& phi_n, const amrex::MultiFab& rhs, amrex::Real dt)
{
    using namespace amrex;
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto pnew = phi.array(mfi);
        auto ph   = phi_hat.const_array(mfi);
        auto r    = rhs.const_array(mfi);
        auto pn   = phi_n.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            pnew(iv) = (1.0_rt/3.0_rt) * pn(iv) + (2.0_rt/3.0_rt) * (ph(iv) + dt * r(iv));
        });
    }
}

/**
 * @brief SSP-RK3 level-set step with a rate of spread rebuilt every stage
 *
 * Shared driver for every direction-dependent path: the caller supplies
 * fill_ros(R, phi_stage), which writes the ROS field to use for that stage
 * from the stage's level set (typically from its front normal). Same Godunov
 * spatial scheme and stage arithmetic as advect_levelset_weno5z_rk3().
 *
 * @param[in,out] phi       Level-set field
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]; |∇φ| is projected
 *                          onto the terrain surface
 * @param[in]     geom      Fire-grid geometry
 * @param[in]     dt        Timestep [s]
 * @param[in]     eps_visc  Artificial viscosity coefficient
 * @param[in]     fill_ros  Callable (MultiFab& R, const MultiFab& phi_stage)
 * @param[in]     mask      Optional non-burnable mask; the ROS is zeroed there
 *                          after every fill
 * @param[in]     wall_extrap  Extrapolate phi into mask cells inside every
 *                          stencil (see WallStencil); needs a mask
 * @param[in]     ros_scale Optional per-cell factor applied to the ROS after
 *                          every fill (the front acceleration clock)
 */
template <class FillRos>
inline void advect_levelset_rk3_with_fill(amrex::MultiFab& phi,
                                          const amrex::MultiFab& slopes,
                                          const amrex::Geometry& geom,
                                          amrex::Real dt,
                                          amrex::Real eps_visc,
                                          FillRos&& fill_ros,
                                          const amrex::MultiFab* mask = nullptr,
                                          bool wall_extrap = false,
                                          const LevelSetGradient& grad_scheme = LevelSetGradient{},
                                          const amrex::MultiFab* ros_scale = nullptr)
{
    using namespace amrex;

    const auto* dx = geom.CellSize();

    const int nghost = 3;
    MultiFab phi_n(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_star(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_hat(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab rhs(phi.boxArray(), phi.DistributionMap(), 1, 0);
    MultiFab R_dir(phi.boxArray(), phi.DistributionMap(), 1, 0);

    phi_n.ParallelCopy(phi);
    fire_fill_boundary(phi_n, geom);

    // Stage 1: phi* = phi^n + dt L(phi^n)
    phi_star.ParallelCopy(phi_n);
    fire_fill_boundary(phi_star, geom);
    fill_ros(R_dir, phi_star);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    rk3_stage1_update(phi_star, rhs, dt);
    fire_fill_boundary(phi_star, geom);

    // Stage 2: phi** = (3/4)phi^n + (1/4)(phi* + dt L(phi*))
    fill_ros(R_dir, phi_star);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_star, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    phi_hat.ParallelCopy(phi_n);
    rk3_stage2_update(phi_hat, phi_star, rhs, dt);
    fire_fill_boundary(phi_hat, geom);

    // Stage 3: phi^{n+1} = (1/3)phi^n + (2/3)(phi** + dt L(phi**))
    fill_ros(R_dir, phi_hat);
    scale_ros_field(R_dir, ros_scale);
    zero_ros_in_mask(R_dir, mask);
    compute_levelset_rhs(rhs, phi_hat, R_dir, dx[0], dx[1], eps_visc, &slopes, mask, wall_extrap, grad_scheme);
    rk3_stage3_update(phi, phi_hat, phi_n, rhs, dt);
}

/**
 * @brief Advect level-set using Godunov upwind spatial + SSP-RK3 temporal integration
 *
 * Solves ∂φ/∂t = −R(x,y) * (|∇φ| − ε Δφ) with a single RK3 step.
 * Multi-stage calls advance by dt; CFL-based subcycling is handled by caller.
 *
 * Note: vel_eff is accepted for interface consistency with other advection
 * routines but is not used here; ROS magnitude is taken directly from R_mf.
 *
 * @param[in,out] phi       Level-set field (modified in-place)
 * @param[in]     vel_eff   Wind velocity (unused; ROS taken from R_mf)
 * @param[in]     R_mf      Rate-of-spread field [m/s]
 * @param[in]     geom      Domain geometry
 * @param[in]     dt        Timestep
 * @param[in]     eps_visc  Artificial viscosity coefficient (erf.fire.levelset.eps_visc)
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]; when supplied, |∇φ| is
 *                          projected onto the terrain surface so that R is a
 *                          spread rate along the ground. nullptr keeps the
 *                          map-view interpretation.
 * @param[in]     mask      Optional non-burnable mask, only read with wall_extrap
 * @param[in]     wall_extrap  Extrapolate phi into mask cells inside every stencil
 */
inline void advect_levelset_weno5z_rk3(amrex::MultiFab& phi,
                                        const amrex::MultiFab& /*vel_eff*/,
                                        const amrex::MultiFab& R_mf,
                                        const amrex::Geometry& geom,
                                        amrex::Real dt,
                                        amrex::Real eps_visc,
                                        const amrex::MultiFab* slopes = nullptr,
                                        const amrex::MultiFab* mask = nullptr,
                                        bool wall_extrap = false,
                                        const LevelSetGradient& grad_scheme = LevelSetGradient{})
{
    using namespace amrex;

    const auto* dx = geom.CellSize();

    // Allocate stage MultiFabs with 3 ghost cells (the stencil reads one)
    const int nghost = 3;
    MultiFab phi_n(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_star(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab phi_hat(phi.boxArray(), phi.DistributionMap(), 1, nghost);
    MultiFab rhs(phi.boxArray(), phi.DistributionMap(), 1, 0);

    // Save initial phi (input)
    phi_n.ParallelCopy(phi);
    fire_fill_boundary(phi_n, geom);

    // Stage 1: φ* = φ^n + Δt L(φ^n)
    phi_star.ParallelCopy(phi_n);
    // ParallelCopy fills the valid region only, and phi_star was just allocated,
    // so without this its three ghost cells are uninitialized -- and the gradient-Z
    // stencil in compute_levelset_rhs reads all three. Stages 2 and 3 already
    // fill before their RHS; stage 1 was the one that did not.
    fire_fill_boundary(phi_star, geom);
    compute_levelset_rhs(rhs, phi_star, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    for (MFIter mfi(phi_star); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ps = phi_star.array(mfi);
        auto r = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ps(iv) = ps(iv) + dt * r(iv);
        });
    }
    fire_fill_boundary(phi_star, geom);

    // Stage 2: φ** = (3/4)φ^n + (1/4)(φ* + Δt L(φ*))
    compute_levelset_rhs(rhs, phi_star, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    phi_hat.ParallelCopy(phi_n);
    for (MFIter mfi(phi_hat); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto ph = phi_hat.array(mfi);
        auto ps = phi_star.const_array(mfi);
        auto r = rhs.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            ph(iv) = 0.75_rt * ph(iv) + 0.25_rt * (ps(iv) + dt * r(iv));
        });
    }
    fire_fill_boundary(phi_hat, geom);

    // Stage 3: φ^{n+1} = (1/3)φ^n + (2/3)(φ** + Δt L(φ**))
    compute_levelset_rhs(rhs, phi_hat, R_mf, dx[0], dx[1], eps_visc, slopes, mask, wall_extrap, grad_scheme);
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto pnew = phi.array(mfi);
        auto ph = phi_hat.const_array(mfi);
        auto r = rhs.const_array(mfi);
        auto pn = phi_n.const_array(mfi);
        ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) noexcept {
            pnew(iv) = (1.0_rt/3.0_rt) * pn(iv) + (2.0_rt/3.0_rt) * (ph(iv) + dt * r(iv));
        });
    }
}

} // namespace fire_levelset

#endif