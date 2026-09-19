#ifndef ERF_ROTHERMEL_H
#define ERF_ROTHERMEL_H

#include <vector>
#include <limits>
#include <AMReX_MultiFab.H>
#include <cmath>

/**
 * @file ERF_Rothermel.H
 * @brief Rothermel fire spread model implementation
 *
 * Implements the Rothermel (1972) fire spread model for rate-of-spread (ROS)
 * computation as a function of fuel type, moisture, wind, and slope.
 *
 * Key references:
 * - Rothermel, R.C. (1972): "A Mathematical Model for Predicting Fire Spread
 *   in Wildland Fuels", USDA Forest Service Research Paper INT-115
 * - Anderson, H.E. (1982): "Aids to determining fuel models for estimating fire
 *   behavior", General Technical Report INT-122, USDA Forest Service
 * - Albini, F.A. (1976): "Estimating wildfire behavior and effects",
 *   General Technical Report INT-30, USDA Forest Service
 *
 * Ported from: wildfire_levelset/src/rothermel_model.H and
 *              wildfire_levelset/src/compute_rothermel_R.H
 */

/**
 * @struct FuelModelParams
 * @brief Anderson FBFM13 fuel model parameters
 *
 * Stores parameters for one of the 13 standard fuel models used by
 * the fire behavior community (Anderson 1982).
 */
struct FuelModelParams
{
    // Fuel load [lb/ft² or kg/m², converted as needed]
    amrex::Real w_d1;       ///< 1-hour dead fuel load
    amrex::Real w_d10;      ///< 10-hour dead fuel load
    amrex::Real w_d100;     ///< 100-hour dead fuel load
    amrex::Real w_lh;       ///< Live herbaceous fuel load
    amrex::Real w_lw;       ///< Live woody fuel load

    // Surface area to volume ratio [1/ft or 1/m]
    amrex::Real sigma_d1;   ///< 1-hour SAV
    amrex::Real sigma_lh;   ///< Herbaceous SAV
    amrex::Real sigma_lw;   ///< Woody SAV

    // Fuel bed depth [ft or m]
    amrex::Real delta;

    // Fuel moisture of extinction [fraction]
    amrex::Real Mx;

    // Heat content [BTU/lb or J/kg]
    amrex::Real heat_content;

    // Fuel particle density [lb/ft³ or kg/m³]
    amrex::Real rho_p;
};

/**
 * @brief Get Anderson FBFM13 fuel model parameters
 *
 * @param[in] model_id Anderson fuel model ID (1–13)
 * @return FuelModelParams for the specified model
 *
 * All 13 Anderson fuel models are hard-coded with standard values.
 */
// Anderson FBFM13 fuel model database
// Reference: Anderson, H.E. (1982) "Aids to determining fuel models..."
// All values are standard published values for these fuel models

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FuelModelParams get_anderson_fuel_params(int model_id)
{
    FuelModelParams fp;

    // All fuel loads in lb/ft² (standard Anderson 1982)
    // SAV in 1/ft (characteristic for 1-hour dead fuel; 10-hr = 109, 100-hr = 30 ft⁻¹ fixed)
    // Heat content in BTU/lb
    // Particle density in lb/ft³
    // Depth in ft
    // Reference: wildfire_levelset/src/fuel_database.H (Anderson 13 fuel models)

    switch (model_id) {
    case 1:  // Short grass (1 ft)
        fp.w_d1 = 0.034;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 3500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.12;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 2:  // Timber (Grass and Understory)
        fp.w_d1 = 0.046;   fp.w_d10 = 0.023;  fp.w_d100 = 0.023;
        fp.w_lh = 0.023;   fp.w_lw = 0.0;
        fp.sigma_d1 = 3000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.15;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 3:  // Tall grass (2.5 ft)
        fp.w_d1 = 0.138;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 4:  // Chaparral (6 ft)
        fp.w_d1 = 0.230;   fp.w_d10 = 0.184;  fp.w_d100 = 0.092;
        fp.w_lh = 0.230;   fp.w_lw = 0.0;
        fp.sigma_d1 = 1739.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 6.0;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 5:  // Brush (2 ft)
        fp.w_d1 = 0.046;   fp.w_d10 = 0.023;  fp.w_d100 = 0.0;
        fp.w_lh = 0.092;   fp.w_lw = 0.0;
        fp.sigma_d1 = 1739.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.0;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 6:  // Dormant Brush, Hardwood Slash
        fp.w_d1 = 0.069;   fp.w_d10 = 0.115;  fp.w_d100 = 0.115;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1750.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 7:  // Southern Rough
        fp.w_d1 = 0.052;   fp.w_d10 = 0.086;  fp.w_d100 = 0.069;
        fp.w_lh = 0.017;   fp.w_lw = 0.017;
        fp.sigma_d1 = 1550.0;  fp.sigma_lh = 1550.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.5;    fp.Mx = 0.40;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 8:  // Closed Timber Litter
        fp.w_d1 = 0.069;   fp.w_d10 = 0.046;  fp.w_d100 = 0.115;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 0.2;    fp.Mx = 0.30;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 9:  // Hardwood Litter
        fp.w_d1 = 0.134;   fp.w_d10 = 0.019;  fp.w_d100 = 0.007;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 2500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 0.2;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 10:  // Timber (Litter and Understory)
        fp.w_d1 = 0.138;   fp.w_d10 = 0.092;  fp.w_d100 = 0.230;
        fp.w_lh = 0.092;   fp.w_lw = 0.0;
        fp.sigma_d1 = 2000.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 11:  // Light Logging Slash
        fp.w_d1 = 0.069;   fp.w_d10 = 0.207;  fp.w_d100 = 0.253;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.15;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 12:  // Medium Logging Slash
        fp.w_d1 = 0.184;   fp.w_d10 = 0.644;  fp.w_d100 = 0.759;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 2.3;    fp.Mx = 0.20;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    case 13:  // Heavy Logging Slash
        fp.w_d1 = 0.322;   fp.w_d10 = 1.058;  fp.w_d100 = 1.288;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 1500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 3.0;    fp.Mx = 0.25;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;

    default:
        // GPU-safe fallback: an unknown id silently uses model 1. amrex::Print()
        // and the recursive call this replaced are both illegal on device, and
        // this function is called from inside ParallelFor kernels.
        fp.w_d1 = 0.034;   fp.w_d10 = 0.0;    fp.w_d100 = 0.0;
        fp.w_lh = 0.0;     fp.w_lw = 0.0;
        fp.sigma_d1 = 3500.0;  fp.sigma_lh = 1500.0;  fp.sigma_lw = 0.0;
        fp.delta = 1.0;    fp.Mx = 0.12;      fp.heat_content = 8000.0;
        fp.rho_p = 32.0;
        break;
    }

    return fp;
}

/**
 * @brief Midflame wind cap of the Rothermel wind factor [ft/min]
 *
 * 300 ft/min for fine fuels (sigma > 1000 1/ft) and 500 ft/min otherwise, or no
 * cap when erf.fire.use_wind_limit = false: the largest Real, which the kernels'
 * min() never selects.
 *
 * @param[in] sigma           Characteristic surface-area-to-volume ratio [1/ft]
 * @param[in] use_wind_limit  erf.fire.use_wind_limit
 * @return U_max [ft/min]
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_wind_cap_ftmin(amrex::Real sigma, bool use_wind_limit) noexcept
{
    if (!use_wind_limit) { return std::numeric_limits<amrex::Real>::max(); }
    return (sigma > amrex::Real(1000.0)) ? amrex::Real(300.0) : amrex::Real(500.0);
}

/**
 * @struct RothermelComputed
 * @brief Precomputed Rothermel coefficients (fuel + moisture dependent)
 *
 * Stores intermediate results computed once per timestep to avoid
 * redundant calculations in the per-cell GPU kernel.
 */
struct RothermelComputed
{
    amrex::Real R0;              ///< No-wind, no-slope ROS [m/s]
    amrex::Real C, B;            ///< Wind factor φ_w = C * U^B
    amrex::Real beta_ratio_E;    ///< (β/β_opt)^E packing ratio factor
    amrex::Real beta;            ///< Packing ratio
    amrex::Real phi_s_const;     ///< Base slope factor (before terrain effects)
    amrex::Real U_max_ftmin;     ///< MEWS wind speed cap [ft/min]
    amrex::Real wind_conv;       ///< m/s to ft/min conversion factor
    amrex::Real ros_conv;        ///< ft/min to m/s conversion factor
    amrex::Real I_R;             ///< Reaction intensity [BTU/ft²/min]
};

/**
 * @brief Compute Rothermel fuel/moisture coefficients
 *
 * Precomputes all Rothermel intermediate quantities that depend only on
 * fuel type and moisture content (not on wind/slope).
 *
 * @param[in] fp FuelModelParams for selected fuel model
 * @param[in] use_wind_limit  Apply the MEWS cap (erf.fire.use_wind_limit)
 * @param[in] use_rothermel_a_formula  Select the reaction-velocity exponent A:
 *                        false (default) Albini's (1976, GTR-INT-30) form,
 *                        A = 133*sigma^-0.7913 (ERF's original behavior);
 *                        true WRF-Fire's Rothermel (1972) original form,
 *                        A = 1/(4.774*sigma^0.1-7.27)
 *                        (erf.fire.reaction_velocity_formula = "rothermel").
 *                        The WRF form has a pole at sigma ~= 67 ft^-1; this
 *                        function's minimum-SAV guard (sigma >= 100) keeps
 *                        every call safely above it.
 * @param[in] wrf_bmst_compat  Deflate the total dry fuel load w_0 by
 *                        (1-bmst), bmst = M_f/(1+M_f), before it feeds
 *                        w_n/rho_b/beta/I_R/xi/R0 -- reproduces WRF-Fire's
 *                        fuelloadm = (1-bmst)*fgip (module_fr_fire_phys.F),
 *                        which treats its (Anderson-derived) fgip as a wet
 *                        mass and strips adsorbed water before Rothermel.
 *                        Default false: ERF's w_d1/w_d10/... are already
 *                        oven-dry, so this is a WRF-compatibility knob, not
 *                        a physical correction (erf.fire.wrf_bmst_compat).
 * @return RothermelComputed with precomputed coefficients
 *
 * Implements the full Rothermel model:
 * - Reaction intensity I_R from fuel load, SAV, moisture, heat content
 * - Propagating flux ratio Ξ and damping coefficients
 * - No-wind ROS R0 from I_R and fuel characteristics
 * - Wind and slope factor coefficients C, B, E
 * - MEWS cap U_max_ftmin (none when use_wind_limit is false)
 */
RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           amrex::Real moisture_1hr,
                                           amrex::Real moisture_10hr,
                                           amrex::Real moisture_100hr,
                                           bool use_wind_limit = true,
                                           bool use_rothermel_a_formula = false,
                                           bool wrf_bmst_compat = false);

/**
 * @brief Compute rate-of-spread for a single fire cell (GPU kernel)
 *
 * Computes ROS = R0 * (1 + φ_w + φ_s) where:
 * - φ_w is wind factor (uses effective midflame wind after WAF and terrain)
 * - φ_s is slope factor (vectorial combination with wind direction)
 * - Includes MEWS cap on effective wind speed
 *
 * @param[in] ux_eff, uy_eff   Effective midflame wind [m/s] after WAF and terrain
 * @param[in] sx, sy           Terrain slopes [dz/dx, dz/dy]
 * @param[in] rc               RothermelComputed coefficients
 * @param[in] cos_theta_wind   Cosine between the wind and the spread direction,
 *                             applied to phi_w after it is computed from
 *                             |(ux_eff, uy_eff)|; 1 (no-op) for the isotropic
 *                             call sites. Lets a caller (see
 *                             ERF_DirectionalRos.H's DIRECTIONAL_WIND_COUPLING_ADVECTIVE)
 *                             exponentiate phi_w from the raw wind speed and
 *                             scale by this cosine afterward, rather than
 *                             passing an already-projected wind in ux_eff/uy_eff.
 * @param[in] cos_theta_slope  Cosine between the slope gradient and the spread
 *                             direction, applied to phi_s the same way; 1
 *                             (no-op) for the isotropic call sites.
 * @return ROS [m/s]
 *
 * Wind factor:
 *   φ_w = C * U_eff^B * (β/β_opt)^E   where U_eff is effective wind speed [ft/min]
 *
 * Slope factor (vectorial combination):
 *   - Compute slope angle θ = atan(|∇z|)
 *   - Project slope elevation gradient in wind direction
 *   - φ_s = tan(θ) * cos(angle_between_wind_and_slope_gradient)
 *   - If slope opposes wind (e.g., upslope): φ_s can be negative (downwind spread)
 *   - φ_s is computed locally; actual terrain forcing comes from terrain wind corrections
 *
 * MEWS cap:
 *   - If U_eff exceeds U_max_ftmin, use U_max_ftmin for wind factor only
 *   - Preserves steady-state behavior: E ≤ 0.9 * I_R always
 *   - With erf.fire.use_wind_limit = false U_max_ftmin is the largest Real
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real rothermel_ros_cell(
    amrex::Real ux_eff, amrex::Real uy_eff,
    amrex::Real sx, amrex::Real sy,
    const RothermelComputed& rc,
    amrex::Real cos_theta_wind = amrex::Real(1.0),
    amrex::Real cos_theta_slope = amrex::Real(1.0)) noexcept
{
    // Effective wind speed [m/s]
    amrex::Real U_eff = std::sqrt(ux_eff*ux_eff + uy_eff*uy_eff);

    // Wind factor: φ_w = C * U_eff^B * (β/β_opt)^E
    // Convert to ft/min for Rothermel calculation
    amrex::Real U_eff_ftmin = U_eff * rc.wind_conv;

    // Apply MEWS cap
    U_eff_ftmin = amrex::min(U_eff_ftmin, rc.U_max_ftmin);

    // Slope magnitude and squared tangent (for Rothermel Eq. 51)
    amrex::Real tan_slope_sq = sx*sx + sy*sy;

    // Cap at tan²(80°) ≈ 32.16; steeper than any realistic or surveyed terrain.
    // Prevents corrupted boundary slopes from producing unphysical ROS.
    tan_slope_sq = amrex::min(tan_slope_sq, amrex::Real(32.0));

    // Slope factor (Rothermel Eq. 51): φ_s = 5.275 * β^(-0.3) * tan²(φ)
    amrex::Real phi_s = rc.phi_s_const * tan_slope_sq;

    // Wind factor
    amrex::Real phi_w = rc.C * std::pow(U_eff_ftmin, rc.B) * rc.beta_ratio_E;

    // The advective wind coupling (erf.fire.directional_wind_coupling =
    // "advective") couples direction in by scaling the already-exponentiated
    // wind/slope excess by the cosine to the spread direction, rather than
    // projecting (ux_eff,uy_eff) before raising it to the B power -- matching
    // WRF-Fire's fire_ros (module_fr_fire_phys.F). cos_theta_* = 1 (the
    // default) reproduces the omnidirectional rate exactly, unchanged from
    // before this parameter existed; a caller doing directional spread may
    // pass the raw wind speed as (ux_eff,uy_eff) and the true cosine here
    // instead, which keeps R linear in cos(theta) and so a convex (stadium)
    // support function instead of cos^B(theta) -- see ERF_DirectionalRos.H's
    // file comment.
    phi_w *= amrex::max(cos_theta_wind, amrex::Real(0.0));
    phi_s *= amrex::max(cos_theta_slope, amrex::Real(0.0));

    // Rate of spread: R = R0 * (1 + φ_w + φ_s)
    amrex::Real ROS = rc.R0 * (1.0 + phi_w + phi_s);

    return amrex::max(static_cast<amrex::Real>(ROS), static_cast<amrex::Real>(0.0));
}

/**
 * @brief Compute rate-of-spread field on fire grid
 *
 * Applies rothermel_ros_cell() to all cells on the fire grid.
 *
 * @param[out] fire_ros        Output ROS [m/s]
 * @param[in]  fire_wind       Effective midflame wind after WAF + terrain [m/s]
 * @param[in]  fire_slopes     Terrain slopes [dz/dx, dz/dy]
 * @param[in]  rc              RothermelComputed coefficients
 */
void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc);

/// Entries in the per-fuel Rothermel table: code 0 (non-burnable) plus FBFM 1-13.
inline constexpr int ROTHERMEL_TABLE_SIZE = 54;   ///< FUEL_SLOT_COUNT: 0 non-burnable, 1-13 Anderson, 14-53 Scott-Burgan

/**
 * @brief Build a per-fuel-code lookup table of Rothermel coefficients
 *
 * Mirrors build_fuel_balbi_table(): indexed by integer fuel code, with index 0
 * holding a zeroed entry (R0 = 0, no spread) and 1-13 the Anderson models at
 * the given dead fuel moistures.
 *
 * @param[in] moisture_1hr   1-hour dead fuel moisture [fraction]
 * @param[in] moisture_10hr  10-hour dead fuel moisture [fraction]
 * @param[in] moisture_100hr 100-hour dead fuel moisture [fraction]
 * @param[in] use_wind_limit Apply the MEWS cap (erf.fire.use_wind_limit)
 * @param[in] use_rothermel_a_formula Select the reaction-velocity A formula;
 *                        see compute_rothermel_params()
 *                        (erf.fire.reaction_velocity_formula = "rothermel")
 * @param[in] wrf_bmst_compat Deflate w_0 by (1-bmst) before Rothermel; see
 *                        compute_rothermel_params() (erf.fire.wrf_bmst_compat)
 * @return host-side vector of size ROTHERMEL_TABLE_SIZE
 */
std::vector<RothermelComputed> build_fuel_rothermel_table(
    amrex::Real moisture_1hr,
    amrex::Real moisture_10hr,
    amrex::Real moisture_100hr,
    int fuel_set = 0,
    amrex::Real moisture_live = -1.0,
    bool use_wind_limit = true,
    bool use_rothermel_a_formula = false,
    bool wrf_bmst_compat = false);

/**
 * @brief Compute the rate-of-spread field with per-cell Rothermel coefficients
 *
 * Cells whose fuel code lies inside the table use that code's coefficients;
 * any other cell, and every cell when no table is given, uses rc_default.
 * With fuel_model or table null this is identical to the uniform overload.
 *
 * @param[out] fire_ros     Output ROS [m/s]
 * @param[in]  fire_wind    Effective midflame wind after WAF + terrain [m/s]
 * @param[in]  fire_slopes  Terrain slopes [dz/dx, dz/dy]
 * @param[in]  rc_default   Coefficients for cells outside the table
 * @param[in]  fuel_model   Per-cell fuel code MultiFab, or nullptr
 * @param[in]  table        Device pointer to the per-fuel table, or nullptr
 * @param[in]  table_size   Entries in table
 * @param[in]  fuel_set     Fuel model set the codes belong to (ERF_FuelModels.H); with
 *                          the Anderson set a code outside 0-13 falls to rc_default as before
 */
void compute_ros_field(
    amrex::MultiFab& fire_ros,
    const amrex::MultiFab& fire_wind,
    const amrex::MultiFab& fire_slopes,
    const RothermelComputed& rc_default,
    const amrex::MultiFab* fuel_model,
    const RothermelComputed* table,
    int table_size,
    int fuel_set = 0);

#endif
