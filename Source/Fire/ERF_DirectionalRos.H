#ifndef ERF_DIRECTIONAL_ROS_H
#define ERF_DIRECTIONAL_ROS_H

#include <AMReX_MultiFab.H>
#include <AMReX_Gpu.H>
#include <cmath>
#include <ERF_Rothermel.H>
#include <ERF_BehaveModel.H>
#include <ERF_MacArthurModel.H>
#include <ERF_CheneyGouldModel.H>
#include <ERF_FbpModel.H>
#include <ERF_LevelSetAdvection.H>
#include <ERF_LevelSetEllipse.H>

/**
 * @file ERF_DirectionalRos.H
 * @brief Direction-dependent rate of spread on the level-set path
 *
 * The level set propagates a front at the normal speed it is given, so the
 * shape of the fire follows from the rate of spread rather than from an imposed
 * template. Handed a single scalar rate, it grows a circle at the head-fire
 * rate: flanks and backing fire advance as fast as the head, which is the
 * behaviour of every rate-of-spread model on this path except Balbi with
 * erf.fire.balbi.directional.
 *
 * The fix is not to bolt on the FARSITE ellipse. That ellipse exists because a
 * Lagrangian marker method has to be told how fast each marker moves relative to
 * the wind, and its length-to-width ratio is an empirical function of midflame
 * wind speed — a stand-in for a flow field that ERF resolves. Imposing it on top
 * of a resolved wind is the same double counting as applying the FARSITE terrain
 * wind factors over resolved terrain flow.
 *
 * Instead the driving vectors are projected onto the front normal
 * n = grad(phi)/|grad(phi)| and the selected model is evaluated with the
 * projected scalars:
 *
 *   R(n) = model( max(U . n, 0), max(grad(z) . n, 0) )
 *
 * At the head n is aligned with the wind and the model sees the full wind and
 * upslope, so R is the head rate. On the flanks the projections vanish and R
 * falls back to the model's no-wind, no-slope rate. Backing is the same, since
 * a wind blowing out of the unburned fuel does not drive the front into it, and
 * Rothermel's slope factor is quadratic and so cannot represent downslope
 * retardation; both are clamped at zero rather than reversed.
 *
 * Three limits of this are worth knowing. The projection reproduces neither the
 * saturation of the observed length-to-width ratio, which the Anderson fit caps
 * at 8, nor a backing rate below the no-wind rate, both of which the empirical
 * ellipse carries as calibration. Making R depend on the normal turns the
 * level-set equation into a general Hamilton-Jacobi problem, while the Godunov
 * upwinding used for |grad(phi)| is derived for a direction-independent speed;
 * freezing R(n) from central differences at each Runge-Kutta stage, as here, is
 * the usual practical approximation rather than the correct anisotropic flux.
 *
 * And a normal speed is not a rate of spread in a direction. From a point, the
 * exact (viscosity) solution of phi_t + R(n) |grad(phi)| = 0 is the Wulff shape
 * {x : x . n <= t R(n) for every n}, whose extent along a direction d is the
 * minimum over n . d > 0 of R(n) / (n . d), not R(d); the two agree only when R
 * is the support function of a convex set. Rothermel's R0 (1 + phi_w (U n_x)^B)
 * is not once phi_w (B - 1) > 1, nor R0 (1 + phi_s (s n_x)^2) once phi_s > 1
 * (the terrain projection of |grad(phi)| adds a factor 1/sqrt(1 + (s . n)^2)
 * that changes little): the head of a point fire becomes a wedge of oblique
 * facets whose tip runs at R0 B/(B - 1) (phi_w (B - 1))^(1/B) in wind, about
 * 2 sqrt(phi_s) R0 on a slope, well below the head rate. A straight line fire
 * keeps n along the wind and is not affected. A better scheme would not help,
 * since it is the exact solution that falls short; this one lands between the
 * tip speed and the head rate (Exec/RegTests/FireDirectionalShape).
 *
 * erf.fire.directional_shape = "ellipse" (DIRECTIONAL_SHAPE_ELLIPSE, opt-in)
 * takes the normal speed instead from the support function of an ellipse, which
 * is convex and so its own Wulff shape, built from the same model:
 *
 *   head   R_h = R0 + g |dR_w w + dR_s s|, along that vector
 *   back   R0
 *   flank  R0
 *
 * with R0 = model(0, 0), dR_w = model(|U|, 0) - R0 and dR_s = model(0, |grad z|)
 * - R0 the wind and slope increments along the unit wind and upslope vectors w
 * and s, added as vectors the way BEHAVE adds Rothermel's wind and slope
 * factors, and g = (model(|U|, |grad z|) - R0) / (dR_w + dR_s), which is 1 for
 * the additive Rothermel form and makes aligned wind and slope give the model's
 * combined rate for the others. With aligned drivers the head, back and flanks
 * are the projection's own rates at those three normals, and the ellipse fills
 * in the directions between them convexly. Its semi-axes are b = (R_h + R0)/2
 * along the head and a = R0 across it, with the centre offset c = (R_h - R0)/2
 * (ERF_LevelSetEllipse.H has the support function), so the length-to-width ratio
 * is (R_h/R0 + 1)/2, uncapped like the projection's.
 *
 * That ratio is the projection's claim that the flanks spread at the no-wind
 * rate, and it runs far above the observed one: 5.7 for short grass in a
 * 1.5 m/s wind, where Anderson (1983) gives 1.5. erf.fire.directional_ellipse_lw
 * = "anderson" (DIRECTIONAL_ELLIPSE_LW_ANDERSON, opt-in) keeps the head and back
 * rates and sets the flank rate a = b / LW instead, with LW Anderson's ratio at
 * the effective wind speed, capped at erf.fire.directional_ellipse_lw_max. The
 * effective wind speed is the wind that alone would give the head rate R_h, the
 * BEHAVE definition, so a slope elongates the fire as the equivalent wind would;
 * for Rothermel and BEHAVE it is capped at the model's own wind limit. The head
 * still runs at R_h and the back at R0, and the ellipse stays convex.
 *
 * erf.fire.directional_wind_coupling = "advective" (DIRECTIONAL_WIND_COUPLING_ADVECTIVE,
 * opt-in, Rothermel with directional_shape = "projection" only) removes the
 * same non-convexity a different way: not by imposing a shape on top of the
 * projection's rates, as the ellipse does, but by changing how the model's
 * wind/slope factor couples to the front-normal direction in the first place.
 * Rothermel's phi_w is exponentiated from the raw, unprojected wind speed and
 * only the result is scaled by cos(theta) to the front normal afterward,
 * R(n) = R0 (1 + phi_w(|U|) max(U . n / |U|, 0)), rather than
 * R0 (1 + phi_w(max(U . n, 0))). Since n |grad(phi)| = grad(phi), this
 * particular R(n) turns the level-set equation into pure advection by a
 * wind-aligned velocity V = (phi_w(|U|) R0 / |U|) U plus isotropic growth at
 * R0, phi_t + V . grad(phi) + R0 |grad(phi)| = 0 -- the same reduction used in
 * the original level-set fire-spread formulation (Mandel, Beezley & Kochanski
 * 2011) and, not coincidentally, matching WRF-Fire's fire_ros
 * (module_fr_fire_phys.F). Being linear in cos(theta), it is also the support
 * function of a stadium (a disc of radius R0 swept along the wind vector,
 * since the clamp at zero flattens the back rather than reversing it) --
 * convex, and so its own Wulff shape, giving the same head rate as the
 * ellipse. Unlike the ellipse it is the model's own oblique rate rather than
 * an imposed shape: back and flank both come out at R0 here because that
 * happens to be Rothermel's own zero-wind rate, not because they were set
 * that way, so for Rothermel the two options happen to agree; they would not
 * necessarily agree for a model whose off-axis rate is not the no-wind rate
 * (BEHAVE has the same power-law wind term and would be a cheap extension,
 * but is not wired to this flag yet). It is also the form a user comparing
 * against WRF-Fire runs would expect.
 *
 * Balbi has its own directional path in ERF_BalbiModel.H, which carries the
 * per-cell moisture, surface temperature and heat-flux couplings that model
 * supports; this covers the remaining models.
 */

/**
 * @struct DirectionalRosState
 * @brief Precomputed coefficients for whichever ROS model is selected
 *
 * Trivially copyable so it can be captured by a device lambda. Only the member
 * matching `model` is read.
 */
struct DirectionalRosState
{
    /// 0 = Rothermel, 1 = BEHAVE, 2 = MacArthur, 3 = Cheney-Gould, 4 = FBP
    int model = 0;

    RothermelComputed   rc{};
    BehaveState         bs{};
    CheneyGouldComputed cgc{};
    FbpComputed         fbp{};

    /// Cheney-Gould inputs, which its per-cell function takes directly
    amrex::Real cg_moisture = 10.0;   ///< Dead fine fuel moisture [%]
    amrex::Real cg_curing   = 1.0;    ///< Degree of curing [0-1]
};

/// Model selectors, matching DirectionalRosState::model
inline constexpr int DIRECTIONAL_ROS_ROTHERMEL    = 0;
inline constexpr int DIRECTIONAL_ROS_BEHAVE       = 1;
inline constexpr int DIRECTIONAL_ROS_MACARTHUR    = 2;
inline constexpr int DIRECTIONAL_ROS_CHENEY_GOULD = 3;
inline constexpr int DIRECTIONAL_ROS_FBP          = 4;

/// |grad(phi)| below which the front normal is taken as undefined and the
/// isotropic wind and slope magnitudes are used instead.
inline constexpr amrex::Real DIRECTIONAL_ROS_GRAD_MIN = 1.0e-12;

/// Angular coupling of the wind/slope factor into the projected rate,
/// erf.fire.directional_wind_coupling. Rothermel only; every other model
/// ignores this and directional_ros_cell's U_mag/s_mag arguments.
inline constexpr int DIRECTIONAL_WIND_COUPLING_PROJECTION = 0;  ///< default: project wind, then exponentiate (Rothermel's U^B)
inline constexpr int DIRECTIONAL_WIND_COUPLING_ADVECTIVE  = 1;  ///< opt-in: exponentiate on the raw wind, then scale by cos(theta) -- turns the level set into pure advection plus isotropic growth

/**
 * @brief Rate of spread along one direction
 *
 * @param[in] st             Precomputed coefficients for the selected model
 * @param[in] U_n            Wind component along the spread direction [m/s]
 * @param[in] s_n            Terrain slope along the spread direction [-]
 * @param[in] wind_coupling  DIRECTIONAL_WIND_COUPLING_PROJECTION (default) or
 *                           _ADVECTIVE; Rothermel only, see below.
 * @param[in] U_mag          Raw (unprojected) wind speed [m/s]; read only
 *                           when wind_coupling == _ADVECTIVE.
 * @param[in] s_mag          Raw (unprojected) slope magnitude [-]; read only
 *                           when wind_coupling == _ADVECTIVE.
 * @return ROS [m/s]
 *
 * DIRECTIONAL_WIND_COUPLING_PROJECTION (default, backward compatible): the
 * Rothermel branch projects the wind onto the spread direction and then
 * raises it to Rothermel's B power, R(theta) ~ (U cos theta)^B -- the
 * original behaviour, byte-for-byte, since U_mag/s_mag are not read on this
 * path. This is non-convex once phi_w(B-1) > 1, which is what lets a finite
 * line's ends grow a Wulff wedge that falls behind the head rate over time
 * (see the file comment). DIRECTIONAL_WIND_COUPLING_ADVECTIVE (opt-in) instead
 * exponentiates the wind/slope factor from the raw magnitude U_mag/s_mag and
 * only afterward scales by the cosine to the spread direction (U_n/U_mag).
 * That keeps R(theta) linear in cos(theta) -- the convex support function of
 * a stadium (a disc of radius R0 swept along the wind vector, since the
 * clamp at zero for backing flattens rather than reverses it) -- so the
 * wedge does not form; equivalently, it turns the level-set equation into
 * pure advection by a wind-aligned velocity plus isotropic growth (see the
 * file comment). Selected via erf.fire.directional_wind_coupling.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_ros_cell(const DirectionalRosState& st,
                                 amrex::Real U_n,
                                 amrex::Real s_n,
                                 int wind_coupling = DIRECTIONAL_WIND_COUPLING_PROJECTION,
                                 amrex::Real U_mag = amrex::Real(0.0),
                                 amrex::Real s_mag = amrex::Real(0.0)) noexcept
{
    // Only a wind pushing into the unburned fuel drives the front, and only an
    // upslope component enhances it; the models take magnitudes, so the
    // projections are clamped rather than passed signed.
    const amrex::Real U = amrex::max(U_n, amrex::Real(0.0));
    const amrex::Real s = amrex::max(s_n, amrex::Real(0.0));

    switch (st.model) {
        case DIRECTIONAL_ROS_BEHAVE:
            return behave_ros_cell(U, amrex::Real(0.0), s, amrex::Real(0.0), st.bs);
        case DIRECTIONAL_ROS_MACARTHUR:
            return macarthur_ros(U);
        case DIRECTIONAL_ROS_CHENEY_GOULD:
            return cheney_gould_ros(U, st.cgc.ros_backing, st.cg_moisture, st.cg_curing);
        case DIRECTIONAL_ROS_FBP:
            return fbp_ros(st.fbp, U, s);
        default:
            if (wind_coupling == DIRECTIONAL_WIND_COUPLING_ADVECTIVE) {
                // The raw magnitude is at least the projected component (it is
                // the same vector, just not dotted with the normal); guard
                // against a caller passing 0 by mistake so cos_w falls back
                // to 1 rather than dividing by (near) zero.
                const amrex::Real Um = amrex::max(U_mag, U);
                const amrex::Real sm = amrex::max(s_mag, s);
                const amrex::Real cos_w = (Um > amrex::Real(1.0e-12)) ? U / Um : amrex::Real(1.0);
                const amrex::Real cos_s = (sm > amrex::Real(1.0e-12)) ? s / sm : amrex::Real(1.0);
                return rothermel_ros_cell(Um, amrex::Real(0.0), sm, amrex::Real(0.0), st.rc, cos_w, cos_s);
            }
            return rothermel_ros_cell(U, amrex::Real(0.0), s, amrex::Real(0.0), st.rc);
    }
}

/// Shape of the direction-dependent rate, erf.fire.directional_shape
inline constexpr int DIRECTIONAL_SHAPE_PROJECTION = 0;  ///< the model evaluated along the front normal
inline constexpr int DIRECTIONAL_SHAPE_ELLIPSE    = 1;  ///< support function of the model's spread ellipse

/// Length-to-width ratio of the ellipse shape, erf.fire.directional_ellipse_lw
inline constexpr int DIRECTIONAL_ELLIPSE_LW_MODEL    = 0;  ///< flanks at the model's no-wind, no-slope rate
inline constexpr int DIRECTIONAL_ELLIPSE_LW_ANDERSON = 1;  ///< Anderson (1983) at the effective wind speed, capped

/**
 * @brief Effective wind speed [m/s]: the wind that alone gives the head rate R_h
 *
 * The wind factor inverted at the combined head rate. Closed form for the
 * Rothermel power law (Rothermel and BEHAVE), capped at the model's own wind
 * limit; bisection on [0, 30] m/s for the others, whose rates increase with the
 * wind.
 *
 * @param[in] st   Precomputed coefficients for the selected model
 * @param[in] R_h  Head rate of spread [m/s]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_effective_wind(const DirectionalRosState& st, amrex::Real R_h) noexcept
{
    const amrex::Real zero_r = 0.0;
    const amrex::Real one_r  = 1.0;
    if (st.model == DIRECTIONAL_ROS_ROTHERMEL || st.model == DIRECTIONAL_ROS_BEHAVE) {
        const bool behave = (st.model == DIRECTIONAL_ROS_BEHAVE);
        const amrex::Real R0   = behave ? st.bs.r_0          : st.rc.R0;
        const amrex::Real cE   = behave ? st.bs.C * st.bs.beta_ratio_E : st.rc.C * st.rc.beta_ratio_E;
        const amrex::Real B    = behave ? st.bs.B            : st.rc.B;
        const amrex::Real Umax = behave ? st.bs.U_max_ftmin  : st.rc.U_max_ftmin;
        const amrex::Real conv = behave ? st.bs.wind_conv    : st.rc.wind_conv;
        if (R0 <= zero_r || cE <= zero_r || B <= zero_r || R_h <= R0) { return zero_r; }
        const amrex::Real U_ft = std::pow((R_h / R0 - one_r) / cE, one_r / B);
        return amrex::min(U_ft, Umax) / conv;
    }
    amrex::Real lo = zero_r, hi = amrex::Real(30.0);
    if (directional_ros_cell(st, hi, zero_r) <= R_h) { return hi; }
    for (int it = 0; it < 30; ++it) {
        const amrex::Real mid = amrex::Real(0.5) * (lo + hi);
        if (directional_ros_cell(st, mid, zero_r) < R_h) { lo = mid; } else { hi = mid; }
    }
    return amrex::Real(0.5) * (lo + hi);
}

/**
 * @struct DirectionalEllipse
 * @brief Spread ellipse built from the selected model, and the direction of its head
 */
struct DirectionalEllipse
{
    SpreadEllipse e{};       ///< head rate b + c, back b - c, flank a [m/s]
    amrex::Real hx = 1.0;    ///< unit vector along the head
    amrex::Real hy = 0.0;
};

/**
 * @brief Spread ellipse whose head, back and flank rates come from the model
 *
 * Head R0 + g |dR_w w + dR_s s| along that vector, back R0, flanks R0 or, with
 * DIRECTIONAL_ELLIPSE_LW_ANDERSON, b over Anderson's length-to-width ratio at
 * the effective wind speed; see the file comment.
 *
 * @param[in] st          Precomputed coefficients for the selected model
 * @param[in] u, v        Wind [m/s]
 * @param[in] dzdx, dzdy  Terrain slopes [-]
 * @param[in] lw_mode     DIRECTIONAL_ELLIPSE_LW_MODEL or DIRECTIONAL_ELLIPSE_LW_ANDERSON
 * @param[in] lw_max      Cap on Anderson's length-to-width ratio
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
DirectionalEllipse directional_ellipse_cell(const DirectionalRosState& st,
                                            amrex::Real u, amrex::Real v,
                                            amrex::Real dzdx, amrex::Real dzdy,
                                            int lw_mode = DIRECTIONAL_ELLIPSE_LW_MODEL,
                                            amrex::Real lw_max = amrex::Real(8.0)) noexcept
{
    constexpr amrex::Real tiny = 1.0e-12;
    const amrex::Real zero_r = 0.0;
    const amrex::Real U = std::sqrt(u*u + v*v);
    const amrex::Real S = std::sqrt(dzdx*dzdx + dzdy*dzdy);

    const amrex::Real R0   = directional_ros_cell(st, zero_r, zero_r);
    const amrex::Real dR_w = amrex::max(directional_ros_cell(st, U, amrex::Real(0.0)) - R0, amrex::Real(0.0));
    const amrex::Real dR_s = amrex::max(directional_ros_cell(st, amrex::Real(0.0), S) - R0, amrex::Real(0.0));
    const amrex::Real dR   = amrex::max(directional_ros_cell(st, U, S) - R0, amrex::Real(0.0));

    // The wind and slope increments as vectors along the wind and upslope
    amrex::Real px = 0.0, py = 0.0;
    if (U > tiny) { px += dR_w * u / U;    py += dR_w * v / U; }
    if (S > tiny) { px += dR_s * dzdx / S; py += dR_s * dzdy / S; }
    const amrex::Real P = std::sqrt(px*px + py*py);
    const amrex::Real g = (dR_w + dR_s > tiny) ? dR / (dR_w + dR_s) : amrex::Real(1.0);

    DirectionalEllipse d;
    if (P > tiny) { d.hx = px / P; d.hy = py / P; }
    const amrex::Real R_h = R0 + g * P;
    d.e.b  = amrex::Real(0.5) * (R_h + R0);
    d.e.c  = amrex::Real(0.5) * (R_h - R0);
    if (lw_mode == DIRECTIONAL_ELLIPSE_LW_ANDERSON) {
        // Flanks from Anderson's ratio at the effective wind speed [mph]
        const amrex::Real U_eff = directional_effective_wind(st, R_h);
        const amrex::Real LB = amrex::max(amrex::Real(1.0),
                                          amrex::min(anderson_LW_ratio(U_eff * amrex::Real(2.23694)), lw_max));
        d.e.a  = d.e.b / LB;
        d.e.LB = LB;
    } else {
        d.e.a  = R0;
        d.e.LB = (R0 > tiny) ? d.e.b / R0 : amrex::Real(1.0e12);
    }
    d.e.HB = (R0 > tiny) ? R_h / R0 : amrex::Real(1.0e12);
    return d;
}

/// Normal speed of a DirectionalEllipse for the unit front normal (nx, ny).
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real directional_ellipse_speed(const DirectionalEllipse& d,
                                      amrex::Real nx, amrex::Real ny) noexcept
{
    return ellipse_normal_speed(d.e, nx * d.hx + ny * d.hy, nx * d.hy - ny * d.hx);
}

/**
 * @brief Fill a direction-dependent ROS field from the front normal
 *
 * Cells away from the front, where |grad(phi)| underflows, fall back to the
 * isotropic magnitudes so the field stays defined everywhere.
 *
 * @param[out] R_mf    Directional ROS [m/s]
 * @param[in]  phi     Level-set field, needs one ghost cell
 * @param[in]  wind    Effective wind [m/s]
 * @param[in]  slopes  Terrain slopes [dz/dx, dz/dy]
 * @param[in]  geom    Fire-grid geometry
 * @param[in]  st      Precomputed coefficients for the selected model
 * @param[in]  mask    Optional non-burnable mask, read only with wall_extrap
 * @param[in]  wall_extrap  Take the front normal through a wall stencil, so a
 *                     cell next to a masked wall sees a normal along the wall
 *                     rather than into it
 * @param[in]  shape   DIRECTIONAL_SHAPE_PROJECTION (the model along the normal)
 *                     or DIRECTIONAL_SHAPE_ELLIPSE (the model's spread ellipse)
 * @param[in]  ellipse_lw      Flank rule of the ellipse, DIRECTIONAL_ELLIPSE_LW_*
 * @param[in]  ellipse_lw_max  Cap on Anderson's length-to-width ratio
 * @param[in]  wind_coupling   DIRECTIONAL_WIND_COUPLING_PROJECTION (default) or
 *                     _ADVECTIVE; Rothermel-with-DIRECTIONAL_SHAPE_PROJECTION only,
 *                     see directional_ros_cell()
 * @param[in]  grad_scheme  Scheme for the front normal's one-sided derivatives
 *                     (fire_levelset::LevelSetGradient; UPWIND1/WENO5Z/WENO5Z_FRONT).
 *                     Godunov-selected (the same max(D-,0)/min(D+,0) choice as
 *                     godunov_norm_grad_phi), NOT central differences -- a plain
 *                     central difference along the front is non-dissipative and
 *                     odd-even-blind, which lets the wind's tangential-to-the-front
 *                     component (dominant on the flanks, unlike the head-on normal
 *                     component) leak through undamped wherever the flank is not
 *                     grid-aligned. This was the rotation-invariance artifact:
 *                     absent when wind/flank are grid-aligned (90/0deg), present
 *                     and growing at oblique angles (34deg), with or without
 *                     reinit. See CLAUDE.md "Leading hypothesis for the
 *                     rotated-stack artifact" / Phase-1 Variant G.
 */
inline void fill_directional_ros(amrex::MultiFab& R_mf,
                                 const amrex::MultiFab& phi,
                                 const amrex::MultiFab& wind,
                                 const amrex::MultiFab& slopes,
                                 const amrex::Geometry& geom,
                                 const DirectionalRosState& st,
                                 const amrex::MultiFab* mask = nullptr,
                                 bool wall_extrap = false,
                                 int shape = DIRECTIONAL_SHAPE_PROJECTION,
                                 int ellipse_lw = DIRECTIONAL_ELLIPSE_LW_MODEL,
                                 amrex::Real ellipse_lw_max = amrex::Real(8.0),
                                 int wind_coupling = DIRECTIONAL_WIND_COUPLING_PROJECTION,
                                 const fire_levelset::LevelSetGradient& grad_scheme = fire_levelset::LevelSetGradient{})
{
    const amrex::Real dx = geom.CellSize(0);
    const amrex::Real dy = geom.CellSize(1);

    for (amrex::MFIter mfi(R_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto R_arr      = R_mf.array(mfi);
        auto phi_arr    = phi.const_array(mfi);
        auto wind_arr   = wind.const_array(mfi);
        auto slopes_arr = slopes.const_array(mfi);
        const fire_levelset::WallStencil wall =
            fire_levelset::make_wall_stencil(mask, wall_extrap, mfi);

        constexpr amrex::Real grad_min = DIRECTIONAL_ROS_GRAD_MIN;

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) {
            const int i = iv[0];
            const int j = iv[1];
            const int k = iv[2];

            const amrex::Real u = wind_arr(i, j, k, 0);
            const amrex::Real v = wind_arr(i, j, k, 1);
            const amrex::Real dzdx = slopes_arr(i, j, k, 0);
            const amrex::Real dzdy = slopes_arr(i, j, k, 1);

            // Front normal from Godunov-selected one-sided derivatives (same
            // max(D-,0)/min(D+,0) choice as godunov_norm_grad_phi), NOT a plain
            // central difference -- see the grad_scheme doc comment above for why.
            amrex::Real dxm, dxp, dym, dyp;
            fire_levelset::one_sided_derivatives(phi_arr, dx, i, j, k, 1, 0, grad_scheme, wall, dxm, dxp);
            fire_levelset::one_sided_derivatives(phi_arr, dy, i, j, k, 0, 1, grad_scheme, wall, dym, dyp);
            const amrex::Real ax = amrex::max(dxm, amrex::Real(0.0));
            const amrex::Real bx = amrex::min(dxp, amrex::Real(0.0));
            const amrex::Real ay = amrex::max(dym, amrex::Real(0.0));
            const amrex::Real by = amrex::min(dyp, amrex::Real(0.0));
            const amrex::Real dpdx = (ax * ax > bx * bx) ? ax : bx;
            const amrex::Real dpdy = (ay * ay > by * by) ? ay : by;
            const amrex::Real grad = std::sqrt(dpdx*dpdx + dpdy*dpdy);

            if (shape == DIRECTIONAL_SHAPE_ELLIPSE) {
                // Support function of the model's spread ellipse, and its head
                // rate where the normal is undefined
                const DirectionalEllipse de = directional_ellipse_cell(st, u, v, dzdx, dzdy,
                                                                      ellipse_lw, ellipse_lw_max);
                R_arr(i, j, k) = (grad > grad_min)
                    ? directional_ellipse_speed(de, dpdx / grad, dpdy / grad)
                    : de.e.b + de.e.c;
                return;
            }

            // Raw (unprojected) magnitudes: passed through so the Rothermel
            // branch of directional_ros_cell can exponentiate from these
            // (the advective coupling), then scale by the cosine to the
            // front normal below, rather than projecting first. Ignored by
            // every other model.
            const amrex::Real U_mag = std::sqrt(u*u + v*v);
            const amrex::Real s_mag = std::sqrt(dzdx*dzdx + dzdy*dzdy);

            amrex::Real U_n, s_n;
            if (grad > grad_min) {
                const amrex::Real nx = dpdx / grad;
                const amrex::Real ny = dpdy / grad;
                U_n = u*nx + v*ny;
                s_n = dzdx*nx + dzdy*ny;
            } else {
                U_n = U_mag;
                s_n = s_mag;
            }

            R_arr(i, j, k) = directional_ros_cell(st, U_n, s_n, wind_coupling, U_mag, s_mag);
        });
    }
}

/**
 * @brief Advect the level set with a direction-dependent ROS
 *
 * Same Godunov / SSP-RK3 scheme as advect_levelset_weno5z_rk3(), except that
 * the ROS field is rebuilt from the front normal at the start of every RK stage
 * so R follows the front as it turns.
 *
 * @param[in,out] phi       Level-set field
 * @param[in]     wind      Effective wind [m/s]
 * @param[in]     slopes    Terrain slopes [dz/dx, dz/dy]
 * @param[in]     geom      Fire-grid geometry
 * @param[in]     dt        Timestep [s]
 * @param[in]     eps_visc  Artificial viscosity coefficient
 * @param[in]     st        Precomputed coefficients for the selected model
 * @param[in]     shape     DIRECTIONAL_SHAPE_PROJECTION or DIRECTIONAL_SHAPE_ELLIPSE
 * @param[in]     ellipse_lw      Flank rule of the ellipse, DIRECTIONAL_ELLIPSE_LW_*
 * @param[in]     ellipse_lw_max  Cap on Anderson's length-to-width ratio
 * @param[in]     ros_scale       Optional per-cell factor on the rebuilt ROS (front acceleration clock)
 * @param[in]     wind_coupling   DIRECTIONAL_WIND_COUPLING_PROJECTION (default) or _ADVECTIVE;
 *                     Rothermel-with-DIRECTIONAL_SHAPE_PROJECTION only, see directional_ros_cell()
 */
inline void advect_levelset_directional_rk3(amrex::MultiFab& phi,
                                            const amrex::MultiFab& wind,
                                            const amrex::MultiFab& slopes,
                                            const amrex::Geometry& geom,
                                            amrex::Real dt,
                                            amrex::Real eps_visc,
                                            const DirectionalRosState& st,
                                            const amrex::MultiFab* nonburnable = nullptr,
                                            bool wall_extrap = false,
                                            const fire_levelset::LevelSetGradient& grad_scheme = fire_levelset::LevelSetGradient{},
                                            int shape = DIRECTIONAL_SHAPE_PROJECTION,
                                            int ellipse_lw = DIRECTIONAL_ELLIPSE_LW_MODEL,
                                            amrex::Real ellipse_lw_max = amrex::Real(8.0),
                                            const amrex::MultiFab* ros_scale = nullptr,
                                            int wind_coupling = DIRECTIONAL_WIND_COUPLING_PROJECTION)
{
    fire_levelset::advect_levelset_rk3_with_fill(phi, slopes, geom, dt, eps_visc,
        [&](amrex::MultiFab& R, const amrex::MultiFab& phi_stage) {
            fill_directional_ros(R, phi_stage, wind, slopes, geom, st, nonburnable, wall_extrap, shape,
                                 ellipse_lw, ellipse_lw_max, wind_coupling, grad_scheme);
        }, nonburnable, wall_extrap, grad_scheme, ros_scale);
}

#endif
