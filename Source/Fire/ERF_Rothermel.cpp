#include <ERF_Rothermel.H>
#include <ERF_FuelModels.H>


using namespace amrex;


RothermelComputed compute_rothermel_params(const FuelModelParams& fp,
                                           Real moisture_1hr,
                                           Real moisture_10hr,
                                           Real moisture_100hr,
                                           bool use_wind_limit,
                                           bool use_rothermel_a_formula,
                                           bool wrf_bmst_compat)
{
    RothermelComputed rc;

    // Unit conversions
    const Real FT_MIN_TO_M_S = 0.00508;

    // Standard mineral contents (Rothermel 1972)
    const Real S_T = 0.0555;   // Total mineral content
    const Real S_e = 0.010;    // Effective mineral content

    // ===================================================================
    // 1. Total fuel load and weighted moisture (single-class path)
    // ===================================================================
    Real w_0 = fp.w_d1 + fp.w_d10 + fp.w_d100 + fp.w_lh + fp.w_lw;  // total oven-dry load [lb/ft²]

    // Compute weighted dead fuel moisture
    Real w_d = fp.w_d1 + fp.w_d10 + fp.w_d100;
    Real M_f = 0.0;  // weighted fuel moisture fraction
    if (w_d > 1.0e-6) {
        Real r_d1 = fp.w_d1 / w_d;
        Real r_d10 = fp.w_d10 / w_d;
        Real r_d100 = fp.w_d100 / w_d;
        M_f = r_d1 * moisture_1hr + r_d10 * moisture_10hr + r_d100 * moisture_100hr;
    }

    // WRF-Fire compatibility: WRF's fgip is nominally the same Anderson
    // oven-dry load as w_0, but module_fr_fire_phys.F treats it as a wet
    // mass and strips out an adsorbed-water fraction before Rothermel:
    // fuelloadm = (1-bmst)*fgip, bmst = fmc_g/(1+fmc_g). Applying the same
    // deflation to w_0 here -- before w_n/rho_b/beta/I_R/xi/R0 are formed --
    // reproduces that behavior exactly through every downstream nonlinear
    // step, not just I_R's numerator.
    if (wrf_bmst_compat) {
        Real bmst = M_f / (1.0 + M_f);
        w_0 *= (1.0 - bmst);
    }

    // ===================================================================
    // 2. Net fuel load (Eq. 24)
    // ===================================================================
    Real w_n = w_0 * (1.0 - S_T);

    // ===================================================================
    // 3. Bulk density and packing ratio (Eq. 24)
    // ===================================================================
    Real rho_b = w_0 / fp.delta;  // lb/ft³
    Real beta = rho_b / fp.rho_p;

    // ===================================================================
    // 4-7. Reaction velocity components (Eqs. 36-38)
    // ===================================================================
    // Use sigma_d1 as characteristic SAV
    Real sigma = fp.sigma_d1;
    sigma = amrex::max(static_cast<amrex::Real>(sigma), static_cast<amrex::Real>(100.0));  // Minimum SAV guard

    Real beta_op = 3.348 * std::pow(sigma, -0.8189);           // Eq. 37: optimum packing ratio
    Real sigma_1p5 = std::pow(sigma, 1.5);
    Real Gamma_max = sigma_1p5 / (495.0 + 0.0594 * sigma_1p5); // Eq. 36: maximum reaction velocity
    // A coefficient (Eq. 38): Albini's (1976) reformulation (ERF's original
    // form) or WRF-Fire's original Rothermel (1972) form, selected by
    // erf.fire.reaction_velocity_formula. The WRF form has a pole at
    // sigma ~= 67 ft^-1 = (7.27/4.774)^10, below which A goes negative; the
    // minimum SAV guard above keeps sigma >= 100 here, safely above it.
    Real A = use_rothermel_a_formula
        ? 1.0 / (4.774 * std::pow(sigma, 0.1) - 7.27)
        : 133.0 * std::pow(sigma, -0.7913);
    Real beta_ratio = beta / beta_op;
    Real Gamma_prime = Gamma_max * std::pow(beta_ratio, A) * std::exp(A * (1.0 - beta_ratio)); // Eq. 38

    // ===================================================================
    // 8-10. Moisture damping (Eq. 29) and mineral damping (Eq. 30)
    // ===================================================================
    Real rm = amrex::min(static_cast<amrex::Real>(M_f / fp.Mx), static_cast<amrex::Real>(1.0));
    Real eta_M = amrex::max(0.0, 1.0 - 2.59*rm + 5.11*rm*rm - 3.52*rm*rm*rm);  // Eq. 29
    Real eta_s = 0.174 * std::pow(S_e, -0.19);                                  // Eq. 30

    // ===================================================================
    // 11. Reaction intensity (Eq. 27)
    // ===================================================================
    Real I_R = Gamma_prime * w_n * fp.heat_content * eta_M * eta_s;
    I_R = amrex::max(static_cast<amrex::Real>(I_R), static_cast<amrex::Real>(0.01));

    // ===================================================================
    // 12-15. Propagating flux ratio, heating number, heat of preignition, and R0 (Eqs. 1, 12, 14, 42)
    // ===================================================================
    Real xi = std::exp((0.792 + 0.681 * std::sqrt(sigma)) * (beta + 0.1))
              / (192.0 + 0.2595 * sigma);                                      // Eq. 42
    Real eps_h = std::exp(-138.0 / sigma);                                     // Eq. 14
    Real Q_ig = 250.0 + 1116.0 * M_f;                                          // Eq. 12: heat of preignition
    Real R0_ft_min = (I_R * xi) / (rho_b * eps_h * Q_ig);                      // Eq. 1: no-wind ROS [ft/min]

    // ===================================================================
    // 16-18. Wind factor coefficients (Eqs. 47-49)
    // ===================================================================
    Real C = 7.47 * std::exp(-0.133 * std::pow(sigma, 0.55));           // Eq. 47 (Rothermel 1972, raw sigma path)
    Real B = 0.02526 * std::pow(sigma, 0.54);                           // Eq. 48
    Real E = 0.715 * std::exp(-3.59e-4 * sigma);                        // Eq. 49

    // ===================================================================
    // 19. Packing ratio wind factor
    // ===================================================================
    Real beta_ratio_E = std::pow(beta_ratio, -E);

    // ===================================================================
    // 20. Slope factor coefficient (Eq. 51)
    // ===================================================================
    Real phi_s_const = 5.275 * std::pow(beta, -0.3);

    // ===================================================================
    // 21. MEWS wind speed cap (Andrews 2018 / Rothermel 1972)
    // ===================================================================
    // The formula phi_w_max = 0.9 * I_R mixes incompatible quantities:
    // phi_w (dimensionless) and I_R (BTU/ft²/min ~500 for fine fuels).
    // This produces a dimensionally incorrect cap of phi_w ~ 475 for FM1,
    // giving unrealistically high ROS (ROS ~ wind speed × 2-3).
    //
    // Instead, use fuel-type-based absolute cap on midflame wind speed,
    // consistent with published BEHAVE/BehavePlus validation tables:
    //   Fine fuels (sigma > 1000 ft⁻¹): cap at 300 ft/min (~1.5 m/s midflame)
    //   Coarse fuels (sigma <= 1000 ft⁻¹): cap at 500 ft/min (~2.5 m/s midflame)
    // erf.fire.use_wind_limit = false removes the cap (rothermel_wind_cap_ftmin).
    Real U_max_ftmin = rothermel_wind_cap_ftmin(sigma, use_wind_limit);

    // ===================================================================
    // Store results in RothermelComputed
    // ===================================================================
    rc.R0           = R0_ft_min * FT_MIN_TO_M_S;  // Convert ft/min → m/s
    rc.C            = C;
    rc.B            = B;
    rc.beta_ratio_E = beta_ratio_E;
    rc.beta         = beta;
    rc.phi_s_const  = phi_s_const;
    rc.U_max_ftmin  = U_max_ftmin;
    rc.wind_conv    = 196.85;    // m/s → ft/min
    rc.ros_conv     = 1.0;       // No double conversion: rc.R0 is already in m/s
    rc.I_R          = I_R;

    return rc;
}


void compute_ros_field(
    MultiFab& fire_ros,
    const MultiFab& fire_wind,
    const MultiFab& fire_slopes,
    const RothermelComputed& rc)
{
    for (MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> ros = fire_ros.array(mfi);
        Array4<const Real> wind = fire_wind.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            Real ux = wind(i, j, k, 0);
            Real uy = wind(i, j, k, 1);
            Real sx = slopes(i, j, k, 0);
            Real sy = slopes(i, j, k, 1);

            ros(i, j, k) = rothermel_ros_cell(ux, uy, sx, sy, rc);
        });
    }
}

std::vector<RothermelComputed> build_fuel_rothermel_table(
    Real moisture_1hr,
    Real moisture_10hr,
    Real moisture_100hr,
    int fuel_set,
    Real moisture_live,
    bool use_wind_limit,
    bool use_rothermel_a_formula,
    bool wrf_bmst_compat)
{
    std::vector<RothermelComputed> table(ROTHERMEL_TABLE_SIZE);

    // Code 0 is non-burnable: every coefficient zero, so R0 = 0 and the
    // kernel returns zero spread whatever the wind and slope.
    table[0] = RothermelComputed{};

    // Slots 1-13 hold the Anderson models at their own codes; 14-53 the Scott-Burgan models.
    for (int slot = 1; slot < ROTHERMEL_TABLE_SIZE; ++slot) {
        table[slot] = compute_rothermel_params(get_fuel_params(fuel_code_from_slot(slot), (slot >= 14) ? 1 : fuel_set, moisture_live),
                                              moisture_1hr, moisture_10hr, moisture_100hr, use_wind_limit,
                                              use_rothermel_a_formula, wrf_bmst_compat);
    }
    return table;
}

void compute_ros_field(
    MultiFab& fire_ros,
    const MultiFab& fire_wind,
    const MultiFab& fire_slopes,
    const RothermelComputed& rc_default,
    const MultiFab* fuel_model,
    const RothermelComputed* table,
    int table_size,
    int fuel_set)
{
    if (fuel_model == nullptr || table == nullptr || table_size <= 0) {
        compute_ros_field(fire_ros, fire_wind, fire_slopes, rc_default);
        return;
    }

    for (MFIter mfi(fire_ros, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        Array4<Real> ros = fire_ros.array(mfi);
        Array4<const Real> wind = fire_wind.array(mfi);
        Array4<const Real> slopes = fire_slopes.array(mfi);
        Array4<const Real> fuel = fuel_model->const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            int i = iv[0];
            int j = iv[1];
            int k = 0;

            Real ux = wind(i, j, k, 0);
            Real uy = wind(i, j, k, 1);
            Real sx = slopes(i, j, k, 0);
            Real sy = slopes(i, j, k, 1);

            const int code = static_cast<int>(fuel(i, j, k));
            const int idx  = fuel_table_index(code, fuel_set);
            const RothermelComputed& rc = (idx >= 0 && idx < table_size)
                                        ? table[idx] : rc_default;
            ros(i, j, k) = rothermel_ros_cell(ux, uy, sx, sy, rc);
        });
    }
}
