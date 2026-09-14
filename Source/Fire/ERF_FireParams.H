#ifndef ERF_FIRE_PARAMS_H
#define ERF_FIRE_PARAMS_H

#include <AMReX_ParmParse.H>
#include <ERF_FireBreak.H>
#include <ERF_FireAcceleration.H>
#include <string>
#include <vector>
#include <ERF_FuelModels.H>
#include <ERF_FbpModel.H>

/**
 * @file ERF_FireParams.H
 * @brief Fire simulation parameters container
 *
 * Loads fire configuration from ParmParse "erf.fire" prefix at runtime.
 * Stores all tunable parameters for the wildfire spread module including
 * fuel model selection, moisture conditions, wind reference height, and
 * wind adjustment factors.
 */

struct FireParams
{
    bool              enable              = false;       ///< Enable fire module
    bool              fire_debug          = false;       ///< Enable debug output for fire calculations
    int               grid_ratio          = 5;           ///< Fire grid refinement factor (C)
    int               fuel_model_id       = 1;           ///< Anderson FBFM13 model (1-13)
    amrex::Real       moisture_1hr        = 0.08;        ///< 1-hour fuel moisture [fraction]
    amrex::Real       moisture_10hr       = 0.08;        ///< 10-hour fuel moisture [fraction]
    amrex::Real       moisture_100hr      = 0.10;        ///< 100-hour fuel moisture [fraction]
    amrex::Real       ignition_x          = 0.0;         ///< Ignition center x [m]
    amrex::Real       ignition_y          = 0.0;         ///< Ignition center y [m]
    amrex::Real       ignition_r          = 20.0;        ///< Ignition radius [m]
    amrex::Real       wind_ref_ht         = 6.1;         ///< MOST wind reference height [m]
    bool              prescribed_wind     = false;       ///< Bypass atmospheric wind interpolation
    amrex::Real       prescribed_wind_x   = 0.0;         ///< Constant reference wind, x [m/s]; exact everywhere only with use_waf=false and use_terrain_wind=false (WAF and the FARSITE terrain corrections still apply downstream otherwise)
    amrex::Real       prescribed_wind_y   = 0.0;         ///< Constant reference wind, y [m/s]; see prescribed_wind_x
    // Phase 13A: Per-fuel wind height
    bool              use_per_fuel_wind_ht = false; ///< Use per-fuel fcwh table for wind height
    bool              use_waf             = true;        ///< Apply Wind Adjustment Factor
    std::string       waf_formula         = "andrews";   ///< "andrews" or "behaviorplus"
    bool              use_wind_limit      = true;        ///< Apply the MEWS midflame wind cap (Rothermel, BEHAVE)
    // Second-height wind sampling (the Community Fire Behavior Model's second
    // interpolation option): sample the atmospheric wind at wind_sample_ht
    // above the local ground, above the fire's own plume, and bring it down
    // to the target height (wind_ref_ht, or the per-fuel height) with a
    // neutral log profile of roughness wind_sample_z0,
    //   U(z_ref) = U(z_sample) ln(z_ref / z0) / ln(z_sample / z0).
    // 0 (default) samples at the target height as before.
    amrex::Real       wind_sample_ht      = 0.0;         ///< Height [m] above ground to sample the wind; 0 = at the target height
    amrex::Real       wind_sample_z0      = 0.1;         ///< Roughness length [m] of the log profile between the two heights
    bool              use_terrain_wind    = true;        ///< Apply terrain corrections
    amrex::Real       k_ridge             = 1.5;         ///< Ridge speed-up factor
    amrex::Real       k_shelter           = 0.6;         ///< Sheltered wind factor
    amrex::Real       k_valley            = 0.8;         ///< Valley channeling factor
    amrex::Real       k_deflect           = 0.3;         ///< Wind deflection factor

    // Fuel moisture control (Phase 4)
    bool              moisture_dynamic    = true;        ///< Advance moisture each step via ODEs
    // How the dead classes are advanced when moisture_dynamic is on:
    //   "timelag" (default) — one time constant per class (ERF_FuelMoisture.H)
    //   "stick"             — radial diffusion in a cylindrical stick per class
    //                         (ERF_FuelMoistureStick.H, Nelson 2000 framework)
    std::string       moisture_model      = "timelag";
    // Equilibrium moisture curves the dead classes relax toward (both models):
    //   "legacy" (default) — the adsorption/desorption quartics ported from
    //                        wildfire_levelset (provenance in ERF_FuelMoisture.H)
    //   "van_wagner"       — Van Wagner and Pickett (1985) drying and wetting EMC
    std::string       emc_model           = "legacy";
    struct StickParams {
        int                      n_shells = 6;                       ///< radial shells per stick (<= 16)
        std::vector<amrex::Real> radius_cm = {0.15, 0.635, 2.5};     ///< 1-h, 10-h, 100-h stick radii [cm]
        amrex::Real              rain_surface_moisture = 0.35;       ///< surface moisture while it rains
        amrex::Real              diffusivity_scale = 1.0;            ///< multiplier on the lag-calibrated diffusivity
    } stick;
    amrex::Real       precip_rate_mm_hr   = 0.0;         ///< Uniform precipitation rate [mm/hr]
    amrex::Real       moisture_live       = 0.60;        ///< Live fuel moisture [fraction]
    // How the live classes move when moisture_dynamic is on:
    //   "legacy" (default) — the dead-fuel time-lag update with the 100-h lag,
    //                        bounded to [0.30, 2.50]; its dead-range clamp
    //                        pulls a value above 0.40 down to 0.40 at once
    //   "fixed"            — held at moisture_live, also after a restart (the
    //                        checkpointed live classes are replaced), and the
    //                        domain average BEHAVE uses is not clamped
    std::string       moisture_live_model = "legacy";

    // Moisture of extinction (Phase 4)

    // Heat flux parameters (Phase 5)
    amrex::Real       tau_residence_s     = 0.0;         ///< Residence time [s]; 0 = derive from fuel SAV
    // Burnout of the fuel behind the front (the e-folding time of the fuel
    // load, which sets how long a cell keeps releasing heat):
    //   "residence" (default) — max(dx_fire / R, tau_SAV), or tau_residence_s
    //                           when positive: the cell crossing time floored
    //                           by the particle burnout time, as always.
    //   "sfire"               — a burn time per fuel model, as WRF-SFIRE and
    //                           the Community Fire Behavior Model do (Mandel
    //                           et al. 2011; Jimenez y Munoz et al. 2026,
    //                           Table 1: 7 s grass, 180 s chaparral, 100 s
    //                           brush, 900 s timber litter and slash), turned
    //                           into the e-folding time by dividing by 0.8514
    //                           (a 40 % loss in 10 min for 1000 s). Heavy fuels
    //                           then keep burning behind the front instead of
    //                           releasing their heat in the crossing time.
    std::string       burnout_model       = "residence";
    std::vector<amrex::Real> burnout_times_s;             ///< 13 burn times [s] for Anderson models 1-13 overriding the table (sfire)
    amrex::Real       burnout_time_to_efold = 0.8514;     ///< tau = burn time / this (WRF-SFIRE)

    // FARSITE parameters
    amrex::Real       farsite_phi_threshold   = 0.1;  ///< front detection level in phi
    int               farsite_use_anderson_lw = 1;       ///< derive coeff_a/b/c from Anderson L/W
    amrex::Real       farsite_coeff_a         = 0.5;  ///< Richards head coefficient
    amrex::Real       farsite_coeff_b         = 0.25; ///< Richards flank coefficient
    amrex::Real       farsite_coeff_c         = 0.1;  ///< Richards backing coefficient
    amrex::Real       farsite_gaussian_sigma  = -1.0; ///< phi stamping radius [m]
    amrex::Real       farsite_cfl_fire        = 0.5;  ///< fire CFL number
    // Front update of the FARSITE path, erf.fire.farsite.front_update (see
    // ERF_FarsiteEllipse.H). "front_cell" (1, default): unburned cells with a
    // burned neighbour burn when the spread shape grown from their neighbours'
    // arrival times reaches them, one row at a time. "legacy" (0): displacement
    // stamping from every cell at or below phi_threshold, which after the phi = 0
    // rebuild includes the first unburned row, so the front ran about twice the
    // rate of spread; kept to reproduce earlier results.
    int               farsite_front_update    = 1;

    // Level-set propagation parameters
    std::string       propagation_method      = "farsite";   ///< "farsite" (default, Lagrangian) or "levelset" (PDE-based)
    amrex::Real       levelset_cfl            = 0.4;   ///< CFL for level-set subcycle
    amrex::Real       levelset_eps_visc       = 0.4;   ///< Artificial viscosity coefficient ε
    int               levelset_reinit_every   = 5;     ///< Reinitialize φ every N fire subcycles
    int               levelset_reinit_iters   = 10;    ///< Sussman reinitialization iterations
    amrex::Real       levelset_reinit_band_m  = -1.0;  ///< Band half-width L [m] where |phi| reaches 1; <=0 selects 3*min(dx,dy)
    amrex::Real       levelset_reinit_dtau    = -1.0;  ///< Reinitialization pseudo-dt (<0 = auto = 0.5*min(dx,dy))
    std::string       levelset_gradient       = "weno5z_front"; ///< One-sided derivatives: "upwind" (first order), "weno5z" (HJ-WENO5-Z everywhere) or "weno5z_front" (WENO within weno_band_cells of the front, first order elsewhere; the WRF-Fire default)
    int               levelset_grad_scheme    = 2;     ///< levelset_gradient resolved: 0 upwind, 1 weno5z, 2 weno5z_front
    int               levelset_weno_band_cells = 4;    ///< Half-width in fire cells of the front band that uses WENO with "weno5z_front" (WRF-Fire fire_lsm_band_ngp)
    amrex::Real       levelset_eps_visc_front  = 0.1;  ///< Artificial viscosity within visc_front_cells of the front (WRF-Fire fire_viscosity_bg; Jimenez y Munoz et al. 2026 use 0.1); < 0 uses eps_visc everywhere
    int               levelset_visc_front_cells = 2;   ///< Half-width in fire cells of the near-front viscosity band (WRF-Fire fire_viscosity_ngp)
    amrex::Real       levelset_visc_transition_cells = 2.0; ///< Width in fire cells of the linear blend from eps_visc_front to eps_visc beyond that band (WRF-Fire: fire_viscosity_band x fire_lsm_band_ngp = 0.5 x 4)
    // Extrapolate the level set into non-burnable cells inside every stencil
    // (gradient, Laplacian, front normal, reinitialisation), so a masked wall
    // is a zero-gradient boundary of the distance function rather than a
    // front. Without it the cells along a wall burn down at the rate the
    // one-sided difference into the wall dictates and the flank along it runs
    // about 20% fast. Off by default; needs a mask source to do anything.
    bool              levelset_wall_extrapolate = false;
    // Wind-driven spread ellipse on the level-set path (Huygens principle as
    // in FARSITE, ELMFIRE, Prometheus, Cell2Fire): the normal speed is the
    // support function of an ellipse with Anderson's length-to-width ratio
    // from the midflame wind and Alexander's head-to-back ratio, head at the
    // model's rate. Off by default. It shapes the front itself, so it turns the
    // directional_ros default off; an explicit directional_ros = true, the
    // hybrid directional path or balbi.directional cannot be combined with
    // it. See ERF_LevelSetEllipse.H.
    bool              levelset_ellipse         = false;  ///< erf.fire.levelset.ellipse
    amrex::Real       levelset_ellipse_lw      = 0.0;    ///< Fixed length-to-width ratio; 0 = Anderson (1983) from the wind
    amrex::Real       levelset_ellipse_lw_max  = 8.0;    ///< Cap on the length-to-width ratio (Anderson's fit saturates at 8)

    // Terrain file for fine-grid terrain data
    std::string       terrain_file_name       = "";   ///< Path to terrain file for fine-grid interpolation

    // CSV output for fire statistics (burned area, perimeter, etc.)
    bool              write_fire_stats_csv    = true; ///< Write burned area and perimeter CSV
    std::string       fire_stats_csv_file     = "fire_stats.csv"; ///< Fire statistics output filename

    // Phase 6: fire-atmosphere coupling parameters
    amrex::Real       heat_flux_alfg          = 45.0;  ///< e-folding height for sensible heat distribution [m]
    amrex::Real       fire_atm_feedback       = 1.0;   ///< multiplier on fire flux (0 = one-way, 1 = full)
    // Coupling mode between fire and atmosphere.
    //   "passive"      — fire heat computed but not injected into atmosphere
    //   "lagged"       — lagged coupling: fire heat from step n enters atmosphere
    //                    at step n+1 via RK source term (default, Phase 6 behaviour)
    //   "synchronous"  — synchronous coupling: fire advances using post-dycore wind
    //                    (vars_new), fire heat still enters atmosphere at step n+1
    std::string       coupling_type           = "lagged";
    bool              inject_latent           = true;  ///< inject latent heat alongside sensible (WRF-Fire parity)
    // How the combustion heat release is split between the sensible and the
    // latent flux handed to the atmosphere.
    //   "legacy" (default) — the sensible flux is the full heat release of the
    //                        dry fuel, Q = w h / tau, and the latent flux is
    //                        derived from it as in WRF-SFIRE (the form in
    //                        ERF-Fire since the coupling was written).
    //   "cfbm"             — the sensible flux is scaled by the dry-fuel
    //                        fraction of the wet fuel mass, 1 / (1 + M_f),
    //                        Eq. 4 of Jimenez y Munoz et al. (2026, GMD 19,
    //                        3035), the Community Fire Behavior Model; the
    //                        latent flux is unchanged (their Eq. 5 is the
    //                        legacy form). The fire-grid heat flux kept for
    //                        diagnostics stays the unpartitioned release.
    std::string       heat_flux_partition     = "legacy";
    // How the fire tendency enters the cell source after make_sources has
    // rebuilt it for the Runge-Kutta stage. "overwrite" (default, legacy)
    // replaces the potential-temperature and vapour slots, which discards any
    // other source in them (immersed-forcing scalar relaxation, radiation,
    // Rayleigh damping). "add" accumulates into the freshly rebuilt source,
    // once per stage, so nothing is lost and nothing is double counted.
    std::string       source_mode             = "overwrite";
    // Place the injected heat only in the open part of each atmospheric column:
    // below the sampled roof height a cell's weight is scaled by its open
    // fraction (one minus the area covered by non-burnable structure cells),
    // and the column's total is renormalised so the injected energy is
    // unchanged. Needs erf.fire.structures.enable. Off by default.
    bool              heat_open_fraction      = false;
    // Tendency form. false (default): the energy-consistent
    // d(rho theta)/dt = -(1/Cp) dQ/dz, with which the coupling's energy
    // diagnostic reads 1 - exp(-z_top/alfg). true: the historical form with
    // an extra density factor, -rho (1/Cp) dQ/dz, which injects rho times the
    // fire flux (about 10-15% too much at sea level); kept to reproduce
    // coupled results from before 2026-09-03.
    bool              heat_tendency_density   = false;

    // Phase 8: Albini (1983) ember spotting parameters
    struct SpottingParams {
        bool              enable              = false;   ///< Master on/off switch
        amrex::Real       I_B_min             = 100.0;   ///< Minimum Byram intensity to launch [kW/m]
        amrex::Real       P_base              = 0.01;    ///< Base launch probability per front cell per step
        amrex::Real       terminal_velocity   = 0.5;     ///< Firebrand terminal fall velocity [m/s]
        int               n_traj_steps        = 20;      ///< Trajectory integration sub-steps
        amrex::Real       spot_radius         = 10.0;    ///< Ignition disk radius at landing [m]
        amrex::Real       P_catch             = 1.0;     ///< Catching probability at landing [0–1]
        amrex::Real       reentry_fuel_thresh = 0.05;    ///< Minimum residual fuel fraction for re-entry,
                                                       ///< as a fraction of the cell's initial load
        int               random_seed         = 0;       ///< 0 = time-based seed, >0 = fixed seed
        int               spotting_interval   = 1;       ///< Apply spotting every N fire subcycles
        std::string       fuel_system         = "13";    ///< "13" (FBFM13) or "40" (Scott-Burgan 40)
        // Which burned cells may launch a brand. "burned" (default, legacy):
        // every cell behind the front, so the brand count scales with the
        // burned area and P_base has to be retuned per domain. "front": only
        // burned cells with unburned fuel within front_band cells, the
        // fireline Albini's model assumes, so the count scales with its
        // length.
        std::string       launch_from         = "burned";
        int               front_band          = 2;       ///< "front": band width [fire cells]
    } spotting;

    // Phase 9: Crown fire initiation parameters.
    // Van Wagner (1977) surface-to-crown criterion + Cruz et al. (2005) active crown ROS.
    // Applicable to conifer forest stands. Not appropriate for grass or shrub fuels.
    struct CrownFireParams {
        bool          enable          = false;   ///< Enable crown fire initiation
        amrex::Real   canopy_base_ht  = 5.0;    ///< Canopy base height C_BH [m]
        amrex::Real   canopy_bulk_den = 0.2;    ///< Canopy bulk density C_BD [kg/m³]
        amrex::Real   canopy_depth    = 10.0;   ///< Canopy depth [m]
        amrex::Real   foliar_moisture = 1.0;    ///< Foliar moisture M_fold [fraction]
        amrex::Real   h_crown_BTU_lb  = 8000.0; ///< Canopy heat content [BTU/lb]
        amrex::Real   wind_10m_kmh    = 0.0;    ///< 10-m wind speed [km/h]; 0 = derive from fire_wind_eff
        std::string   ros_model       = "cruz"; ///< Crown ROS model: "cruz", "rothermel1991", "van_wagner_proxy"
        bool          use_passive_blend = false; ///< Use Van Wagner passive-to-active ROS blend
    } crown;

    // Phase 9: Flame temperature diagnostic
    std::string       flame_temp_method   = "byram_radiant"; ///< "byram_radiant", "mcalpine_heat", "nelson_emissivity"
    amrex::Real       flame_temp_T_amb    = 300.0;           ///< Ambient temperature for flame temp formula [K]

    // Phase 9: Flame tilt angle diagnostic
    bool              compute_flame_tilt  = false;           ///< Compute flame tilt angle diagnostic [degrees]
    amrex::Real       flame_tilt_T_amb    = 300.0;           ///< Ambient temperature for flame tilt [K]
    amrex::Real       flame_tilt_rho_air  = 1.2;             ///< Air density for flame tilt [kg/m³]

    // Phase 10: Spatial fuel map parameters.
    struct FuelMapParams {
        std::string   fuel_map_file       = "";        ///< Path to fuel map file; empty = uniform
        std::string   fuel_map_format     = "ascii";   ///< "ascii" or "lcp"
        amrex::Real   blending_fraction   = 0.0;       ///< ROS blending at fuel boundaries [0-1]
        /// Fuel codes that are non-burnable (e.g. 0 for nodata, 91-99 in the
        /// Scott-Burgan set). Empty by default, in which case an unknown code
        /// falls through to fuel model 1 in the Rothermel table as before.
        std::vector<int> nonburnable_codes;
        // Fuel model set (see ERF_FuelModels.H):
        //   "anderson13"     (default) the Anderson 13 only; codes outside 1-13 fall to model 1
        //   "scott_burgan40" the Scott and Burgan 40 (LANDFIRE FBFM40 codes 101-204) alongside
        //                    the 13, natively; 0 and 91-99 are non-burnable without listing them
        std::string fuel_set = "anderson13";
        /// With scott_burgan40: translate the 40 to the Anderson 13 at load (the CFBM way)
        bool sb40_crosswalk = false;
        /// Take each cell's initial fuel load from its own model instead of the uniform one
        bool load_from_map = false;
        /// Set id for the kernels: Scott-Burgan semantics only when native (no crosswalk)
        int  fuel_set_id() const { return (fuel_set == "scott_burgan40" && !sb40_crosswalk) ? 1 : 0; }
        /// The Scott-Burgan set is in use in either mode (non-burnable 0 and 91-99)
        bool sb40_active() const { return fuel_set == "scott_burgan40"; }
    } fuel_map;

    // Phase 11: Multi-ignition schedule parameters.
    // When ignition_schedule_file is non-empty, a CSV schedule of timed ignition
    // events is loaded and applied during advance(). The primary ignition from
    // ignition_x/y/r is always applied at t=0 regardless of this setting.
    //
    // CSV format: time_s  cx  cy  radius  [source_type]  [priority]
    //   source_type: "sphere" (default) or "polyline"
    //   priority: integer; higher priority events fire first within the same timestep
    //   Comment lines start with # or !
    //
    // Polygon ignition (initial perimeter from a vertex file):
    //   polygon_file: path to CSV of polygon vertices (x y per line)
    //   polygon_type: "polygon" (closed) or "polyline" (open line fire)
    //   polyline_width: half-width of line ignition zone [m]
    struct IgnitionParams {
        std::string   ignition_schedule_file  = "";         ///< Path to ignition schedule CSV; empty = disabled
        std::string   polygon_file            = "";         ///< Path to polygon vertex CSV; empty = disabled
        std::string   polygon_type            = "polygon";  ///< "polygon" or "polyline"
        amrex::Real   polyline_width          = 10.0;       ///< Half-width of polyline ignition zone [m]
        // Observed-perimeter ignition with spin-up (WRF-SFIRE's perimeter
        // time, as used by the Community Fire Behavior Model): the polygon is
        // applied at polygon_time instead of at initialisation, so the
        // atmosphere spins up first. With polygon_interior_ros > 0 the
        // interior is given the state of a fire that spread outward at that
        // rate: a cell at distance d inside the perimeter gets arrival time
        // polygon_time - d / R (clamped at the simulation start, since a
        // negative arrival time reads as unburned) and its fuel reduced by
        // exp(-d / (R tau)),
        // tau = polygon_interior_tau (dx_fire / R when 0), the exponential
        // burnout the heat-flux step applies. 0 keeps the whole interior
        // igniting at polygon_time with the fuel intact, as before.
        amrex::Real   polygon_time            = 0.0;        ///< Time [s] the polygon is applied; 0 = at initialisation
        amrex::Real   polygon_interior_ros    = 0.0;        ///< Spread rate [m/s] behind the interior arrival times; 0 = off
        amrex::Real   polygon_interior_tau    = 0.0;        ///< Burnout e-folding time [s] of the interior; 0 = dx_fire / polygon_interior_ros
        // Temperature-threshold ignition: every burnable, unburned fire cell
        // whose near-surface air temperature (the k = 0 potential temperature
        // the fuel-moisture update reads, mapped onto the fire grid) exceeds
        // threshold_temp is ignited as a disc of radius threshold_radius (one
        // fire cell when 0), from threshold_start_time on. In a coupled run
        // the fire's own heating can therefore start new fire ahead of the
        // front; in a one-way run only the initial state can. Off by default.
        bool          threshold_enable        = false;      ///< Ignite cells hotter than threshold_temp
        amrex::Real   threshold_temp          = 573.15;     ///< Ignition temperature [K] (300 C, piloted ignition of fine dead fuel)
        amrex::Real   threshold_radius        = 0.0;        ///< Radius [m] of the disc stamped around a hot cell; 0 = one fire cell
        amrex::Real   threshold_start_time    = 0.0;        ///< No threshold ignition before this time [s]
    } ignition;

    // Phase 12: Fire acceleration at startup.
    // Small fires do not immediately achieve their quasi-steady-state (QSS) ROS.
    // Two models are available, selected by use_temporal:
    //
    //   Size-based (use_temporal = false, default):
    //     alpha = 1 - exp(-r_fire / L_acc)
    //     R_effective = alpha * R_Rothermel
    //   where r_fire is the effective radius from burned area and L_acc is the
    //   acceleration length scale [m].
    //   Reference: Rothermel (1983) GTR INT-143.
    //
    //   Temporal (use_temporal = true):
    //     R(t) = R_E * (1 - exp(-A * t))
    //   where t is elapsed time [s] and A [1/min] is the acceleration
    //   constant, selected based on perimeter length relative to
    //   perim_limit (point vs line fire). accel.clock picks the meaning of t:
    //   "legacy" (default) a clock per burned cell, started when the cell
    //   burns, written to burned cells only; "front" the time since ignition,
    //   carried with the front and applied to the rate that moves it.
    //   Reference: McAlpine & Wakimoto (1991) Forest Science 37(5).
    //
    // Disabled by default (enable = false). When disabled, no MultiFab is
    // allocated and the call in advance() returns immediately.
    AccelerationParams accel;

    // Evaluate the rate of spread along the local front normal on the level-set
    // path, instead of handing it a single scalar rate that it then applies in
    // every direction. The wind and slope are projected onto the normal and the
    // selected ROS model is evaluated with the projected scalars, so head, flank
    // and backing spread come from the model itself.
    //
    // On by default (since 2026-09-08): this is the WRF-Fire form, and without
    // it the level set applied the head rate in every direction, so a grass
    // fire backed into a 35 m/s wind at a third of its head rate. Set it false
    // to recover that isotropic disc. No effect on the FARSITE path, which gets
    // its directionality from the Anderson ellipse; erf.fire.levelset.ellipse
    // turns this default off (see the end of the constructor). Balbi has its own
    // switch, erf.fire.balbi.directional, which additionally carries that
    // model's per-cell moisture, surface temperature and heat-flux couplings;
    // either flag enables direction-dependent spread when ros_model = "balbi".
    bool              directional_ros     = true;

    // Shape of that direction-dependent rate, erf.fire.directional_shape.
    // "projection" (0, default) evaluates the model with the wind and slope
    // projected onto the front normal. "ellipse" (1) takes the normal speed from
    // the support function of an ellipse with the model's head rate along the
    // vector sum of its wind and slope increments, and its no-wind, no-slope
    // rate at the back and flanks. From a point ignition the projection's head
    // falls short of the model's head rate once the wind or slope factor is
    // large, because the exact solution is the Wulff shape of R(n) (see
    // ERF_DirectionalRos.H); the ellipse is convex and spreads the head at the
    // model's rate. Needs directional_ros; implemented for rothermel, behave,
    // macarthur, cheney_gould and fbp.
    int               directional_shape   = 0;

    // Angular coupling of the wind/slope factor into that projection,
    // erf.fire.directional_wind_coupling. "projection" (0, default, backward
    // compatible) projects the wind onto the spread direction before
    // Rothermel's power-law wind factor, R(theta) ~ (U cos theta)^B -- this is
    // the original behaviour, unchanged. "advective" (1, opt-in) instead
    // computes the wind/slope factor from the raw, unprojected speed and
    // scales the result by the cosine afterward (matching WRF-Fire's
    // fire_ros, module_fr_fire_phys.F): R(theta) is then linear in cos(theta), the
    // convex support function of a stadium (a disc of radius R0 swept along
    // the wind vector), rather than cos^B(theta), which stops being convex
    // once phi_w(B-1) > 1 and lets a finite line's ends grow a Wulff wedge
    // that falls behind the head rate over time (see ERF_DirectionalRos.H).
    // Rothermel with directional_shape = "projection" only; rejected in every
    // other combination (see the abort below), since the ellipse path is
    // already convex and every other model or the hybrid path would silently
    // ignore this flag.
    int               directional_wind_coupling = 0;

    // Flank rule of that ellipse, erf.fire.directional_ellipse_lw. "model" (0,
    // default) keeps the model's no-wind, no-slope rate at the flanks, so the
    // length-to-width ratio (R_h/R0 + 1)/2 is not capped. "anderson" (1) sets the
    // flank rate from Anderson's (1983) ratio at the effective wind speed (the
    // wind that alone would give the combined head rate, so a slope elongates
    // the fire too), capped at directional_ellipse_lw_max; head and back rates
    // are unchanged. Needs directional_shape = "ellipse".
    int               directional_ellipse_lw     = 0;
    amrex::Real       directional_ellipse_lw_max = 8.0;  ///< Cap on that ratio (Anderson's fit saturates at 8)

    // Horizontal interpolation of the atmospheric wind onto the fire grid:
    // 0 = nearest atmospheric column (every fire cell in a column shares one
    // wind vector), 1 = bilinear between the four surrounding columns.
    // Set through erf.fire.wind_interp = "nearest" | "bilinear".
    int               wind_interp         = 1;

    // Phase 13B: Rate-of-spread model selection and parameters

    // Phase 13B: Rate-of-spread model selection.
    // "rothermel"    — Rothermel (1972) single-class model (default, Phases 1-12)
    // "macarthur"    — MacArthur (1966) Australian formula R = 0.18*exp(0.8424*U)
    // "balbi"        — Balbi (2009) physical model R = A*(1+sin α − cos α)
    // "cheney_gould" — Cheney & Gould (1998) empirical grassland model
    // "fbp"          — Canadian Forest Fire Behavior Prediction System (ERF_FbpModel.H)
    std::string       ros_model = "rothermel";  ///< ROS model selector

    // The Rothermel/BEHAVE reaction-velocity exponent A (Gamma' = Gamma'_max *
    // (beta/beta_op)^A * exp(A*(1-beta/beta_op))) has two non-equivalent
    // published forms, erf.fire.reaction_velocity_formula:
    //   "albini" (default, backward compatible) — Albini's (1976, GTR-INT-30)
    //                        reformulation, A = 133 * sigma^-0.7913, ERF's
    //                        form since this was first implemented.
    //   "rothermel"        — Rothermel's (1972) original form,
    //                        A = 1 / (4.774 * sigma^0.1 - 7.27), as used by
    //                        WRF-Fire (module_fr_fire_phys.F). This form has a
    //                        pole at sigma ~= 67 ft^-1, below which A goes
    //                        negative; ERF_Rothermel.cpp's minimum-SAV guard
    //                        (sigma >= 100) keeps every call safely above it.
    // Only affects ros_model = "rothermel"; BEHAVE (ERF_BehaveModel.H) keeps
    // its own independent copy of the Albini form regardless of this option.
    std::string       reaction_velocity_formula = "albini";

    // Phase 13B: Balbi (2009) physical fire spread model parameters.
    // Only used when ros_model = "balbi".
    // Defaults from Balbi (2009) Table 1.
    // The struct is copied to the device and evaluated inside GPU kernels, so
    // every member has to stay trivially copyable (no std::string here; the
    // "formulation" input string is mapped onto the integer flag below).
    struct BalbiParams {
        amrex::Real T_a      = 300.0;    ///< Ambient temperature [K]
        amrex::Real T_f      = 1000.0;   ///< Mean flame temperature [K]
        amrex::Real T_i      = 600.0;    ///< Ignition temperature [K]
        amrex::Real delta_H  = 2.26e6;   ///< Latent heat of vaporisation [J/kg]
        amrex::Real C_pf     = 1800.0;   ///< Specific heat of dry fuel [J/(kg·K)]
        amrex::Real r_00     = 2.5e-4;   ///< Radiation length scale [m]
        amrex::Real tau_0    = 75591.0;  ///< Residence time coefficient [s/m]

        // Formulation selector: 0 = Balbi (2009) steady explicit form (default),
        // 1 = Balbi (2020) convective-radiative implicit form.
        // Set through erf.fire.balbi.formulation = "2009" | "2020".
        int         formulation = 0;

        // Balbi (2020) constants. Defaults follow the nomenclature of
        // Balbi et al. (2020) as implemented in ForeFire's Balbi2020.
        amrex::Real chi_0    = 0.3;      ///< Radiant fraction [-]
        amrex::Real C_pa     = 1150.0;   ///< Specific heat of air [J/(kg·K)]
        amrex::Real C_pw     = 4180.0;   ///< Specific heat of water [J/(kg·K)]
        amrex::Real T_vap    = 373.0;    ///< Water vaporisation temperature [K]
        amrex::Real K1       = 130.0;    ///< Convective decay coefficient [-]
        amrex::Real st       = 17.0;     ///< Stoichiometric coefficient [-]
        amrex::Real rho_a    = 1.2;      ///< Air density [kg/m³]
        amrex::Real sigma_B  = 5.670374e-8; ///< Stefan-Boltzmann constant [W/(m²·K⁴)]
        int         max_iter = 40;       ///< Max root-solve iterations (2020 only)
        amrex::Real tol      = 1.0e-4;   ///< Root-solve bracket width [m/s] (2020 only)

        // Direction-dependent spread on the level-set path: evaluate the ROS
        // along the local front normal instead of using the wind and slope
        // magnitudes. No effect on the FARSITE path, which gets its
        // directionality from the ellipse.
        bool directional = false;

        // Take the ambient temperature per cell from fire_surface_temp instead
        // of the fixed balbi.T_a.
        bool use_surface_temp = false;

        // Derive the mean flame temperature from the combustion energy,
        // T_f = T_a + delta_H_combustion (1 - chi_0) / (C_pa (st + 1)),
        // instead of using the fixed balbi.T_f. Balbi (2009) form only; the
        // 2020 form always computes its own flame temperature.
        bool flame_temp_from_combustion = false;

        // Take the fuel moisture per cell from the Phase 4 moisture ODE state
        // (fire_fuel_mc component 0) instead of the domain average. Requires
        // erf.fire.moisture_dynamic = true.
        bool use_cell_moisture = false;

        // Zero the rate of spread once the fuel moisture reaches the fuel
        // model's moisture of extinction. Without this the model has no
        // extinction limit and a wet fuel bed still spreads.
        bool use_moisture_extinction = false;

        // Fraction of the live herbaceous load that has cured and is carried
        // as dead fine fuel. Only the Balbi (2020) form uses a fuel load, so
        // this has no effect under the 2009 formulation.
        amrex::Real herb_curing = 0.0;

        // Which wind the model consumes: 0 = the midflame wind after the
        // Wind Adjustment Factor and terrain corrections (fire_wind_eff,
        // default, what every other ROS model uses), 1 = the reference-height
        // wind before those corrections (fire_wind_ref). Balbi normalises the
        // wind by its own vertical velocity scale rather than by a midflame
        // reduction, so the WAF is arguably double counting.
        // Set through erf.fire.balbi.wind_source = "midflame" | "reference".
        int wind_source = 0;

        // Augment the buoyancy velocity with the fire-induced component built
        // from the surface heat flux:
        //   v_b_Q   = k_upward sqrt(g Q H_ref / (rho_a C_pa T_a))
        //   v_b_eff = sqrt(v_b^2 + v_b_Q^2)
        // The heat flux lags the ROS by one fire step.
        bool        heat_flux_coupling = false;
        amrex::Real k_upward           = 1.0;   ///< Scale on the induced velocity [-]
        amrex::Real hf_ref_height      = 10.0;  ///< Reference height for v_b_Q [m]
    } balbi;

    // Phase 13B: Cheney-Gould (1998) grassland model parameters.
    // Only used when ros_model = "cheney_gould".
    struct CheneyGouldParams {
        amrex::Real moisture = 10.0;  ///< Dead fine fuel moisture [%]
        amrex::Real curing   = 1.0;   ///< Degree of curing [0-1]; 1 = fully cured
    } cheney_gould;

    // Canadian Forest Fire Behavior Prediction System (ros_model = "fbp";
    // see ERF_FbpModel.H). The system's inputs are daily indices, uniform
    // over the domain, and its wind is the 10 m open wind.
    struct FbpParams {
        std::string fuel_type = "C2";   ///< C1-C7, D1, M1-M4, S1-S3, O1A, O1B
        amrex::Real ffmc      = 90.0;   ///< Fine Fuel Moisture Code
        amrex::Real bui       = 60.0;   ///< Buildup Index
        amrex::Real curing    = 60.0;   ///< Degree of curing of the grass types [%]
        amrex::Real pc        = 50.0;   ///< Percent conifer (M1, M2)
        amrex::Real pdf       = 50.0;   ///< Percent dead fir (M3, M4)
        bool        use_slope = true;   ///< Slope through the equivalent wind
        /// 0 = the reference-height wind (fire_wind_ref; set wind_ref_ht = 10),
        /// 1 = the midflame wind after the Wind Adjustment Factor (fire_wind_eff)
        int         wind_source = 0;
    } fbp;

    // Phase 15: BEHAVE multi-class Rothermel parameters.
    // Only used when ros_model = "behave".
    // Controls the moisture window of the live-to-dead herbaceous load transfer,
    // which acts only on fuels with a live herbaceous load (w_lh > 0).
    // All other fuel properties come from FuelModelParams via get_anderson_fuel_params().
    struct BehaveParams {
        amrex::Real dynamic_transfer_lo  = 0.30;  ///< Live herba moisture [fraction] at and below which the whole load transfers
        amrex::Real dynamic_transfer_hi  = 1.20;  ///< Live herba moisture [fraction] at and above which none transfers
    } behave;

    // Hybrid rate of spread: two of the models above evaluated on every cell
    // and blended with a per-cell weight in [0,1], where 0 takes the primary
    // model and 1 the secondary. Only used when ros_model = "hybrid".
    // Selectors: "region" and "fuel" set a static weight at initialisation,
    // "structure" a static weight from distance to a building footprint, and
    // "wind" a weight rebuilt every fire step from the effective wind speed.
    // Works on the isotropic path and, with directional_ros, on the
    // direction-dependent level-set path.
    struct HybridParams {
        std::string      primary   = "rothermel";  ///< Model where the weight is 0
        std::string      secondary = "balbi";      ///< Model where the weight is 1
        std::string      selector  = "region";     ///< "region" | "fuel" | "structure" | "wind"
        amrex::Real      region_x_lo = 0.0;        ///< selector = region: rectangle [m]
        amrex::Real      region_y_lo = 0.0;
        amrex::Real      region_x_hi = 0.0;
        amrex::Real      region_y_hi = 0.0;
        std::vector<int> secondary_fuel_codes;     ///< selector = fuel: codes that take the secondary
        std::string      structure_file;           ///< selector = structure: building heightmap; empty = erf.buildings_file_name
        amrex::Real      structure_min_height = 0.5; ///< selector = structure: height above which a cell is a structure [m]
        amrex::Real      structure_distance = 10.0; ///< selector = structure: cells within this distance take the secondary [m]
        amrex::Real      wind_lo = 8.0;            ///< selector = wind: weight 0 at and below this speed [m/s]
        amrex::Real      wind_hi = 12.0;           ///< selector = wind: weight 1 at and above this speed [m/s]
        amrex::Real      blend_width = 0.0;        ///< selector = region or structure: linear ramp across the edge [m]
    } hybrid;

    // Arrival-time probes: flat list of x y pairs [m]. Each probe reports the
    // fire arrival time at its fire cell once, on the step it first burns.
    std::vector<amrex::Real> probes;

    // Prescribed rate of spread, used only when ros_model = "prescribed":
    // R = max(min_ros, R_base + grad_x (x - x0) + grad_y (y - y0)) with R_base
    // the rate listed for the cell's fuel code in by_fuel, else ros. No wind,
    // slope or moisture dependence and no direction, so directional_ros is
    // turned off. For verification against closed-form arrival times; see
    // ERF_PrescribedFire.H.
    struct PrescribedRosParams {
        amrex::Real ros     = 0.0;   ///< R_base where by_fuel has no entry [m/s]
        amrex::Real grad_x  = 0.0;   ///< dR/dx [1/s]
        amrex::Real grad_y  = 0.0;   ///< dR/dy [1/s]
        amrex::Real x0      = 0.0;   ///< Point where R = R_base [m]
        amrex::Real y0      = 0.0;
        amrex::Real min_ros = 0.0;   ///< Floor on R [m/s]
        std::vector<amrex::Real> by_fuel; ///< Flat list of fuel code, R_base pairs; needs fuel_map.file
    } prescribed;

    // Prescribed surface heat flux over a disc, added to fire_heat_flux whatever
    // the fuel and the front, and injected by the coupling like fire heat. Off
    // unless flux > 0. For plume verification; see ERF_PrescribedFire.H.
    struct PrescribedHeatParams {
        amrex::Real flux       = 0.0;   ///< Flux inside the disc [W/m2]; 0 = off
        amrex::Real cx         = 0.0;   ///< Disc centre [m]
        amrex::Real cy         = 0.0;
        amrex::Real radius     = 0.0;   ///< Disc radius [m]
        amrex::Real start_time = 0.0;   ///< Switch-on time [s]
        amrex::Real end_time   = -1.0;  ///< Switch-off time [s]; negative = never
    } prescribed_heat;

    // Per-structure exposure diagnostics (needs structures.enable). Each
    // connected footprint of the heightmap gets an id, and every `interval`
    // fire steps one line per structure is appended to `file`: the first and
    // last arrival of the front in the ring of burnable cells within `ring`
    // fire cells of the footprint, the fraction of that ring burned, the peak
    // fireline intensity and the accumulated heat load there, and the number
    // of embers that landed on the footprint. Off by default.
    struct ExposureParams {
        bool        enable   = false;
        std::string file     = "fire_exposure.csv";
        int         interval = 100;   ///< fire steps between rows
        int         ring     = 1;     ///< wall band width [fire cells]
    } exposure;

    // Structures on the fire grid: a building heightmap in the ERF terrain
    // text format sampled onto fire cell centres by nearest point. With
    // enable = true every cell whose sampled height exceeds min_height is
    // non-burnable: zero rate of spread, the level set held outside the
    // burned region, FARSITE and spotting landings rejected, no fuel and no
    // heat. Off by default; the hybrid "structure" selector reads the same
    // file but by itself does not stop the fire.
    struct StructureParams {
        bool        enable     = false;   ///< Treat structure cells as non-burnable
        std::string file;                 ///< Heightmap; empty = hybrid.structure_file, then erf.buildings_file_name
        amrex::Real min_height = 0.5;     ///< Sampled height above which a cell is a structure [m]
        // Bilinear wind weights renormalised over the atmospheric columns
        // that are open at the fire cell's wind height: a column whose
        // sampled roof is above that height contributes with its open
        // fraction. With immersed-forcing buildings the columns inside a
        // footprint carry the relaxed in-building velocity, which otherwise
        // leaks into the wind of the fire cells around it. Off by default;
        // no effect with wind_interp = "nearest".
        bool        wind_open_columns = false;
    } structures;

    // Firebreak cells stamped at initialization are also held in the
    // non-burnable mask when true, so the level-set path cannot burn back
    // into them after reinitialisation. Off by default (legacy behaviour).
    bool firebreak_use_mask = false;

    // Evaluate the Rothermel coefficients per cell from the spatial fuel map
    // instead of the domain fuel_model_id. Off by default: the Rothermel
    // kernel then spreads with the domain fuel everywhere, as it always has,
    // and only the heat flux and the Balbi kernels see the map.
    bool rothermel_per_fuel = false;

    // Phase 10: Firebreak barriers (up to 10, indexed by N=0..9)
    int n_firebreaks = 0;
    std::vector<FireBreak> firebreaks;

    // Phase 4: smoke tracer
    bool  smoke_enable           = false;
    amrex::Real  smoke_emission_factor  = 0.02;       // kg smoke / kg fuel [-]
    amrex::Real  smoke_heat_of_comb     = 18700000.0; // J/kg fuel
    // Take the heat per kilogram that converts the flux back into fuel burnt
    // from the fuel model's own heat content (erf.fire.fuel_model_id) instead
    // of smoke_heat_of_comb, as the Community Fire Behavior Model does. With
    // a spatial fuel map the deck's fuel model is still the one used, since
    // the lagged flux lives on the atmospheric grid without the fuel codes.
    bool  smoke_heat_from_fuel   = false;

    /**
     * @brief Construct FireParams from ParmParse
     *
     * Reads from "erf.fire.*" parameters. Unspecified values use defaults above.
     */
    FireParams()
    {
       amrex::ParmParse pp("erf.fire");
       // Reject values the kernels cannot use instead of running on with a
       // silent fallback or a NaN. Every message names the key and the value.
       auto fail = [](const std::string& msg) { amrex::Abort("[FIRE] " + msg); };
       auto one_of = [&](const char* key, const std::string& v, std::initializer_list<const char*> allowed) {
           for (auto a : allowed) { if (v == a) return; }
           std::string list;
           for (auto a : allowed) { list += (list.empty() ? "" : ", ") + std::string("\"") + a + "\""; }
           fail(std::string("erf.fire.") + key + " = \"" + v + "\" is not one of " + list);
       };
       auto positive = [&](const char* key, amrex::Real v) {
           if (!(v > 0.0)) fail(std::string("erf.fire.") + key + " must be > 0, got " + std::to_string(v));
       };
       auto nonneg = [&](const char* key, amrex::Real v) {
           if (!(v >= 0.0)) fail(std::string("erf.fire.") + key + " must be >= 0, got " + std::to_string(v));
       };
       auto fraction = [&](const char* key, amrex::Real v) {
           if (!(v >= 0.0 && v <= 1.0)) fail(std::string("erf.fire.") + key + " must be in [0,1], got " + std::to_string(v));
       };
       pp.query("enable", enable);
       pp.query("fire_debug", fire_debug);
       pp.query("grid_ratio", grid_ratio);
       pp.query("fuel_model_id", fuel_model_id);
       pp.query("moisture_1hr", moisture_1hr);
       pp.query("moisture_10hr", moisture_10hr);
       pp.query("moisture_100hr", moisture_100hr);
       pp.query("ignition_x", ignition_x);
       pp.query("ignition_y", ignition_y);
       pp.query("ignition_r", ignition_r);
       pp.query("wind_ref_ht", wind_ref_ht);
       pp.query("prescribed_wind", prescribed_wind);
       pp.query("prescribed_wind_x", prescribed_wind_x);
       pp.query("prescribed_wind_y", prescribed_wind_y);
       fraction("moisture_1hr", moisture_1hr);
       fraction("moisture_10hr", moisture_10hr);
       fraction("moisture_100hr", moisture_100hr);
       nonneg("ignition_r", ignition_r);   // 0 = no ignition disc (schedule, polygon or line instead)
       positive("wind_ref_ht", wind_ref_ht);
       pp.query("wind_sample_ht", wind_sample_ht);
       pp.query("wind_sample_z0", wind_sample_z0);
       if (wind_sample_ht > 0.0) {
           if (wind_sample_ht <= wind_ref_ht) {
               amrex::Abort("erf.fire.wind_sample_ht must exceed erf.fire.wind_ref_ht (sample above the target height)");
           }
           if (wind_sample_z0 <= 0.0 || wind_sample_z0 >= wind_ref_ht) {
               amrex::Abort("erf.fire.wind_sample_z0 must be positive and below erf.fire.wind_ref_ht");
           }
       }
       // Phase 13A: Per-fuel wind height
       pp.query("use_per_fuel_wind_ht", use_per_fuel_wind_ht);
       pp.query("use_waf", use_waf);
       pp.query("waf_formula", waf_formula);
       one_of("waf_formula", waf_formula, {"andrews", "behaviorplus"});
       pp.query("use_wind_limit", use_wind_limit);
       pp.query("use_terrain_wind", use_terrain_wind);
       pp.query("k_ridge", k_ridge);
       pp.query("k_shelter", k_shelter);
       pp.query("k_valley", k_valley);
       pp.query("k_deflect", k_deflect);

       // Fuel moisture control (Phase 4)
       pp.query("moisture_dynamic", moisture_dynamic);
       pp.query("precip_rate_mm_hr", precip_rate_mm_hr);
       pp.query("moisture_live", moisture_live);
       pp.query("moisture_model", moisture_model);
       pp.query("emc_model", emc_model);
       if (emc_model != "legacy" && emc_model != "van_wagner") {
           amrex::Abort("erf.fire.emc_model must be \"legacy\" or \"van_wagner\"");
       }
       pp.query("moisture_live_model", moisture_live_model);
       pp.query("stick.n_shells", stick.n_shells);
       pp.queryarr("stick.radius_cm", stick.radius_cm);
       pp.query("stick.rain_surface_moisture", stick.rain_surface_moisture);
       pp.query("stick.diffusivity_scale", stick.diffusivity_scale);
       if (moisture_model != "timelag" && moisture_model != "stick") {
           amrex::Abort("erf.fire.moisture_model must be \"timelag\" or \"stick\"");
       }
       if (moisture_live_model != "legacy" && moisture_live_model != "fixed") {
           amrex::Abort("erf.fire.moisture_live_model must be \"legacy\" or \"fixed\"");
       }
       if (moisture_model == "stick") {
           if (stick.n_shells < 1 || stick.n_shells > 16) { amrex::Abort("erf.fire.stick.n_shells must be 1-16"); }
           if (stick.radius_cm.size() != 3) { amrex::Abort("erf.fire.stick.radius_cm needs 3 values (1-h, 10-h, 100-h)"); }
           for (auto r : stick.radius_cm) { if (r <= 0.0) amrex::Abort("erf.fire.stick.radius_cm must be positive"); }
           if (stick.diffusivity_scale <= 0.0) { amrex::Abort("erf.fire.stick.diffusivity_scale must be positive"); }
       }

       // Moisture of extinction (Phase 4)

       // Heat flux parameters (Phase 5)
       pp.query("tau_residence_s", tau_residence_s);
       pp.query("burnout_model", burnout_model);
       pp.queryarr("burnout_times_s", burnout_times_s);
       pp.query("burnout_time_to_efold", burnout_time_to_efold);
       if (burnout_model != "residence" && burnout_model != "sfire") {
           amrex::Abort("erf.fire.burnout_model must be \"residence\" or \"sfire\"");
       }
       if (!burnout_times_s.empty() && burnout_times_s.size() != 13) {
           amrex::Abort("erf.fire.burnout_times_s needs 13 values, one per Anderson fuel model");
       }
       for (auto t : burnout_times_s) { if (t <= 0.0) amrex::Abort("erf.fire.burnout_times_s must be positive"); }
       if (burnout_time_to_efold <= 0.0) { amrex::Abort("erf.fire.burnout_time_to_efold must be positive"); }

       // FARSITE parameters
       pp.query("farsite.phi_threshold", farsite_phi_threshold);
       pp.query("farsite.use_anderson_lw", farsite_use_anderson_lw);
       pp.query("farsite.coeff_a", farsite_coeff_a);
       pp.query("farsite.coeff_b", farsite_coeff_b);
       pp.query("farsite.coeff_c", farsite_coeff_c);
       pp.query("farsite.gaussian_sigma", farsite_gaussian_sigma);
       pp.query("farsite.cfl_fire", farsite_cfl_fire);
       {
           std::string front_update_str = (farsite_front_update == 0) ? "legacy" : "front_cell";
           pp.query("farsite.front_update", front_update_str);
           if (front_update_str == "legacy") {
               farsite_front_update = 0;
           } else if (front_update_str == "front_cell") {
               farsite_front_update = 1;
           } else {
               amrex::Abort("erf.fire.farsite.front_update must be \"front_cell\" or \"legacy\"");
           }
       }

       // Level-set propagation parameters
       pp.query("propagation_method", propagation_method);
       one_of("propagation_method", propagation_method, {"farsite", "levelset"});
       pp.query("levelset.cfl", levelset_cfl);
       pp.query("levelset.eps_visc", levelset_eps_visc);
       pp.query("levelset.reinit_every", levelset_reinit_every);
       pp.query("levelset.reinit_iters", levelset_reinit_iters);
       pp.query("levelset.reinit_band_m", levelset_reinit_band_m);
       pp.query("levelset.reinit_dtau", levelset_reinit_dtau);
       pp.query("levelset.wall_extrapolate", levelset_wall_extrapolate);
       pp.query("levelset.ellipse",        levelset_ellipse);
       pp.query("levelset.ellipse_lw",     levelset_ellipse_lw);
       pp.query("levelset.ellipse_lw_max", levelset_ellipse_lw_max);
       if (levelset_ellipse) {
           if (levelset_ellipse_lw < 0.0 || levelset_ellipse_lw_max < 1.0) {
               amrex::Abort("erf.fire.levelset.ellipse_lw must be >= 0 and ellipse_lw_max >= 1");
           }
       }
       pp.query("levelset.gradient", levelset_gradient);
       pp.query("levelset.weno_band_cells", levelset_weno_band_cells);
       pp.query("levelset.eps_visc_front", levelset_eps_visc_front);
       pp.query("levelset.visc_front_cells", levelset_visc_front_cells);
       pp.query("levelset.visc_transition_cells", levelset_visc_transition_cells);
       if (levelset_gradient == "upwind") {
           levelset_grad_scheme = 0;
       } else if (levelset_gradient == "weno5z") {
           levelset_grad_scheme = 1;
       } else if (levelset_gradient == "weno5z_front") {
           levelset_grad_scheme = 2;
       } else {
           amrex::Abort("erf.fire.levelset.gradient must be \"upwind\", \"weno5z\" or \"weno5z_front\", got \"" + levelset_gradient + "\"");
       }

       // Terrain file. The fire grid is grid_ratio times finer than the
       // atmosphere, so it can carry a finer digital elevation model than the
       // fitted mesh: erf.fire.terrain_file_name overrides the atmosphere's
       // file when it is given, and the slope that drives the rate of spread
       // is then read at the fire's own resolution. Unset, the fire shares
       // erf.terrain_file_name as before.
       amrex::ParmParse pp_erf("erf");
       pp_erf.query("terrain_file_name", terrain_file_name);
       pp.query("terrain_file_name", terrain_file_name);

       // CSV output parameters
       pp.query("write_fire_stats_csv", write_fire_stats_csv);
       pp.query("fire_stats_csv_file", fire_stats_csv_file);

       // Phase 6: fire-atmosphere coupling parameters
       pp.query("heat_flux_alfg", heat_flux_alfg);
       pp.query("fire_atm_feedback", fire_atm_feedback);
       pp.query("coupling_type", coupling_type);
       one_of("coupling_type", coupling_type, {"passive", "lagged", "synchronous"});
       positive("heat_flux_alfg", heat_flux_alfg);
       nonneg("fire_atm_feedback", fire_atm_feedback);   // 0 = one-way, the documented switch
       pp.query("inject_latent", inject_latent);
       pp.query("heat_flux_partition", heat_flux_partition);
       if (heat_flux_partition != "legacy" && heat_flux_partition != "cfbm") {
           amrex::Abort("erf.fire.heat_flux_partition must be \"legacy\" or \"cfbm\"");
       }
       pp.query("source_mode", source_mode);
       if (source_mode != "overwrite" && source_mode != "add") {
           amrex::Abort("erf.fire.source_mode must be \"overwrite\" or \"add\"");
       }
       pp.query("heat_open_fraction", heat_open_fraction);
       pp.query("heat_tendency_density", heat_tendency_density);

       // Phase 4: smoke tracer parameters
       pp.query("smoke_enable", smoke_enable);
       pp.query("smoke_emission_factor", smoke_emission_factor);
       pp.query("smoke_heat_of_comb", smoke_heat_of_comb);
       pp.query("smoke_heat_from_fuel", smoke_heat_from_fuel);
       if (smoke_enable) {
           positive("smoke_heat_of_comb", smoke_heat_of_comb);
           nonneg("smoke_emission_factor", smoke_emission_factor);
       }

       // Phase 8: Albini (1983) ember spotting parameters
       auto& sp = spotting;
       pp.query("spotting.enable",              sp.enable);
       pp.query("spotting.I_B_min",             sp.I_B_min);
       pp.query("spotting.P_base",              sp.P_base);
       pp.query("spotting.terminal_velocity",   sp.terminal_velocity);
       pp.query("spotting.n_traj_steps",        sp.n_traj_steps);
       pp.query("spotting.spot_radius",         sp.spot_radius);
       pp.query("spotting.launch_from",         sp.launch_from);
       pp.query("spotting.front_band",          sp.front_band);
       if (sp.launch_from != "burned" && sp.launch_from != "front") {
           amrex::Abort("erf.fire.spotting.launch_from must be \"burned\" or \"front\"");
       }
       if (sp.front_band < 1) { amrex::Abort("erf.fire.spotting.front_band must be >= 1"); }
       pp.query("spotting.P_catch",             sp.P_catch);
       pp.query("spotting.reentry_fuel_thresh", sp.reentry_fuel_thresh);
       pp.query("spotting.random_seed",         sp.random_seed);
       pp.query("spotting.spotting_interval",   sp.spotting_interval);
       pp.query("spotting.fuel_system",         sp.fuel_system);
       if (sp.enable) {
           positive("spotting.I_B_min", sp.I_B_min);
           positive("spotting.terminal_velocity", sp.terminal_velocity);
           if (sp.n_traj_steps < 1) fail("erf.fire.spotting.n_traj_steps must be >= 1, got " + std::to_string(sp.n_traj_steps));
           fraction("spotting.P_base", sp.P_base);
           fraction("spotting.P_catch", sp.P_catch);
           one_of("spotting.fuel_system", sp.fuel_system, {"13", "40"});
       }

       // Phase 9: Crown fire initiation parameters
       auto& cr = crown;
       pp.query("crown.enable",          cr.enable);
       pp.query("crown.canopy_base_ht",  cr.canopy_base_ht);
       pp.query("crown.canopy_bulk_den", cr.canopy_bulk_den);
       pp.query("crown.canopy_depth",    cr.canopy_depth);
       pp.query("crown.foliar_moisture", cr.foliar_moisture);
       pp.query("crown.h_crown_BTU_lb",  cr.h_crown_BTU_lb);
       pp.query("crown.wind_10m_kmh",    cr.wind_10m_kmh);
       pp.query("crown.ros_model",       cr.ros_model);
       one_of("crown.ros_model", cr.ros_model, {"cruz", "rothermel1991", "van_wagner_proxy"});
       pp.query("crown.use_passive_blend", cr.use_passive_blend);

       // Phase 9: Flame temperature diagnostic
       pp.query("flame_temp_method", flame_temp_method);
       one_of("flame_temp_method", flame_temp_method, {"byram_radiant", "mcalpine_heat", "nelson_emissivity"});
       pp.query("flame_temp_T_amb",  flame_temp_T_amb);

       // Phase 9: Flame tilt angle diagnostic
       pp.query("compute_flame_tilt", compute_flame_tilt);
       pp.query("flame_tilt_T_amb",   flame_tilt_T_amb);
       pp.query("flame_tilt_rho_air", flame_tilt_rho_air);

       // Phase 10: fuel map
       auto& fm = fuel_map;
       pp.query("fuel_map.file",               fm.fuel_map_file);
       pp.query("fuel_map.format",             fm.fuel_map_format);
       pp.query("fuel_map.blending_fraction",  fm.blending_fraction);
       one_of("fuel_map.format", fm.fuel_map_format, {"ascii", "lcp"});
       fraction("fuel_map.blending_fraction", fm.blending_fraction);

       // Phase 11: multi-ignition schedule and polygon ignition
       auto& ig = ignition;
       pp.query("ignition.schedule_file", ig.ignition_schedule_file);
       pp.query("ignition.polygon_file",  ig.polygon_file);
       pp.query("ignition.polygon_type",  ig.polygon_type);
       pp.query("ignition.polyline_width", ig.polyline_width);
       one_of("ignition.polygon_type", ig.polygon_type, {"polygon", "polyline"});
       if (!ig.polygon_file.empty() && ig.polygon_type == "polyline") positive("ignition.polyline_width", ig.polyline_width);
       pp.query("ignition.polygon_time",         ig.polygon_time);
       pp.query("ignition.polygon_interior_ros", ig.polygon_interior_ros);
       pp.query("ignition.polygon_interior_tau", ig.polygon_interior_tau);
       if (ig.polygon_time < 0.0 || ig.polygon_interior_ros < 0.0 || ig.polygon_interior_tau < 0.0) {
           amrex::Abort("erf.fire.ignition.polygon_time, polygon_interior_ros and polygon_interior_tau must be >= 0");
       }
       pp.query("ignition.threshold_enable",     ig.threshold_enable);
       pp.query("ignition.threshold_temp",       ig.threshold_temp);
       pp.query("ignition.threshold_radius",     ig.threshold_radius);
       pp.query("ignition.threshold_start_time", ig.threshold_start_time);
       if (ig.threshold_temp <= 0.0 || ig.threshold_radius < 0.0 || ig.threshold_start_time < 0.0) {
           amrex::Abort("erf.fire.ignition.threshold_temp must be > 0 K; threshold_radius and threshold_start_time must be >= 0");
       }

       // Phase 12: fire acceleration
       auto& ac = accel;
       pp.query("accel.enable",          ac.enable);
       pp.query("accel.use_temporal",    ac.use_temporal);
       pp.query("accel.L_acc",           ac.L_acc);
       pp.query("accel.A_point",         ac.A_point);
       pp.query("accel.A_line",          ac.A_line);
       pp.query("accel.perim_limit",     ac.perim_limit);
       pp.query("accel.enable_wind_lag", ac.enable_wind_lag);
       pp.query("accel.tau_wind",        ac.tau_wind);
       {
           std::string clock_str = (ac.clock == accel_clock::front) ? "front" : "legacy";
           pp.query("accel.clock", clock_str);
           if (clock_str == "legacy") {
               ac.clock = accel_clock::legacy;
           } else if (clock_str == "front") {
               ac.clock = accel_clock::front;
           } else {
               amrex::Abort("erf.fire.accel.clock must be \"legacy\" or \"front\"");
           }
       }
       // The wind lag restarts the legacy clock on every rise of the equilibrium
       // rate; the front clock follows the equilibrium rate at once instead.
       if (ac.enable && ac.use_temporal && ac.clock == accel_clock::front && ac.enable_wind_lag) {
           amrex::Abort("erf.fire.accel.enable_wind_lag works with erf.fire.accel.clock = \"legacy\" only");
       }

       // Remembered for the ellipse check at the end: the ellipse turns the
       // default off but conflicts with an explicit true.
       const bool directional_ros_given = pp.query("directional_ros", directional_ros);
       {
           std::string shape_str = (directional_shape == 1) ? "ellipse" : "projection";
           pp.query("directional_shape", shape_str);
           if (shape_str == "projection") {
               directional_shape = 0;
           } else if (shape_str == "ellipse") {
               directional_shape = 1;
           } else {
               amrex::Abort("erf.fire.directional_shape must be \"projection\" or \"ellipse\"");
           }

           std::string coupling_str = (directional_wind_coupling == 1) ? "advective" : "projection";
           pp.query("directional_wind_coupling", coupling_str);
           if (coupling_str == "projection") {
               directional_wind_coupling = 0;
           } else if (coupling_str == "advective") {
               directional_wind_coupling = 1;
           } else {
               amrex::Abort("erf.fire.directional_wind_coupling must be \"projection\" or \"advective\"");
           }

           std::string lw_str = (directional_ellipse_lw == 1) ? "anderson" : "model";
           pp.query("directional_ellipse_lw", lw_str);
           pp.query("directional_ellipse_lw_max", directional_ellipse_lw_max);
           if (lw_str == "model") {
               directional_ellipse_lw = 0;
           } else if (lw_str == "anderson") {
               directional_ellipse_lw = 1;
           } else {
               amrex::Abort("erf.fire.directional_ellipse_lw must be \"model\" or \"anderson\"");
           }
           if (directional_ellipse_lw == 1 && directional_shape != 1) {
               amrex::Abort("erf.fire.directional_ellipse_lw = \"anderson\" needs erf.fire.directional_shape = \"ellipse\"");
           }
           if (directional_ellipse_lw_max < 1.0) {
               amrex::Abort("erf.fire.directional_ellipse_lw_max must be >= 1");
           }
       }

       {
           std::string wind_interp_str = (wind_interp == 0) ? "nearest" : "bilinear";
           pp.query("wind_interp", wind_interp_str);
           if (wind_interp_str == "bilinear") {
               wind_interp = 1;
           } else if (wind_interp_str == "nearest") {
               wind_interp = 0;
           } else {
               amrex::Abort("erf.fire.wind_interp must be \"bilinear\" or \"nearest\"");
           }
       }

       // Phase 13B: ROS model selection
       pp.query("ros_model", ros_model);
       one_of("ros_model", ros_model, {"rothermel", "behave", "balbi", "fbp", "hybrid", "prescribed", "cheney_gould", "macarthur"});
       pp.query("reaction_velocity_formula", reaction_velocity_formula);
       one_of("reaction_velocity_formula", reaction_velocity_formula, {"albini", "rothermel"});
       pp.query("rothermel_per_fuel", rothermel_per_fuel);

       // Hybrid ROS
       {
           auto& hy = hybrid;
           pp.query("hybrid.primary",   hy.primary);
           pp.query("hybrid.secondary", hy.secondary);
           pp.query("hybrid.selector",  hy.selector);
           std::vector<amrex::Real> region;
           if (pp.queryarr("hybrid.region", region)) {
               if (region.size() != 4) {
                   amrex::Abort("erf.fire.hybrid.region needs four values: x_lo y_lo x_hi y_hi");
               }
               hy.region_x_lo = region[0];
               hy.region_y_lo = region[1];
               hy.region_x_hi = region[2];
               hy.region_y_hi = region[3];
           }
           pp.queryarr("hybrid.secondary_fuel_codes", hy.secondary_fuel_codes);
           pp.query("hybrid.structure_file",       hy.structure_file);
           pp.query("hybrid.structure_min_height", hy.structure_min_height);
           pp.query("hybrid.structure_distance",   hy.structure_distance);
           pp.query("hybrid.wind_lo",            hy.wind_lo);
           pp.query("hybrid.wind_hi",            hy.wind_hi);
           pp.query("hybrid.blend_width",        hy.blend_width);

           if (ros_model == "hybrid") {
               auto known = [](const std::string& m) {
                   return m == "rothermel" || m == "balbi" || m == "behave" ||
                          m == "macarthur" || m == "cheney_gould" || m == "fbp";
               };
               if (!known(hy.primary) || !known(hy.secondary)) {
                   amrex::Abort("erf.fire.hybrid.primary/secondary must each be one of "
                                "rothermel, balbi, behave, macarthur, cheney_gould, fbp");
               }
               if (hy.primary == hy.secondary) {
                   amrex::Abort("erf.fire.hybrid.primary and secondary must differ");
               }
               if (hy.selector == "region") {
                   if (!pp.contains("hybrid.region")) {
                       amrex::Abort("erf.fire.hybrid.selector = region needs erf.fire.hybrid.region");
                   }
                   if (hy.region_x_hi < hy.region_x_lo || hy.region_y_hi < hy.region_y_lo) {
                       amrex::Abort("erf.fire.hybrid.region must satisfy x_lo <= x_hi and y_lo <= y_hi");
                   }
               } else if (hy.selector == "fuel") {
                   if (hy.secondary_fuel_codes.empty()) {
                       amrex::Abort("erf.fire.hybrid.selector = fuel needs erf.fire.hybrid.secondary_fuel_codes");
                   }
               } else if (hy.selector == "structure") {
                   if (hy.structure_file.empty()) {
                       amrex::ParmParse pp_erf("erf");
                       pp_erf.query("buildings_file_name", hy.structure_file);
                   }
                   if (hy.structure_file.empty()) {
                       amrex::Abort("erf.fire.hybrid.selector = structure needs "
                                    "erf.fire.hybrid.structure_file or erf.buildings_file_name");
                   }
                   if (hy.structure_distance < 0.0) {
                       amrex::Abort("erf.fire.hybrid.structure_distance must be >= 0");
                   }
               } else if (hy.selector == "wind") {
                   if (!(hy.wind_hi > hy.wind_lo)) {
                       amrex::Abort("erf.fire.hybrid.wind_hi must exceed erf.fire.hybrid.wind_lo");
                   }
               } else {
                   amrex::Abort("erf.fire.hybrid.selector must be region, fuel, structure or wind");
               }
               if (hy.blend_width < 0.0) {
                   amrex::Abort("erf.fire.hybrid.blend_width must be >= 0");
               }
           }
       }

       pp.queryarr("probes", probes);
       if (probes.size() % 2 != 0) {
           amrex::Abort("erf.fire.probes must be a flat list of x y pairs");
       }

       // Prescribed rate of spread and heat flux (verification cases)
       {
           auto& pr = prescribed;
           pp.query("prescribed.ros",     pr.ros);
           pp.query("prescribed.min_ros", pr.min_ros);
           std::vector<amrex::Real> pair;
           if (pp.queryarr("prescribed.gradient", pair)) {
               if (pair.size() != 2) { amrex::Abort("erf.fire.prescribed.gradient needs two values: dR/dx dR/dy [1/s]"); }
               pr.grad_x = pair[0]; pr.grad_y = pair[1];
           }
           pair.clear();
           if (pp.queryarr("prescribed.origin", pair)) {
               if (pair.size() != 2) { amrex::Abort("erf.fire.prescribed.origin needs two values: x0 y0 [m]"); }
               pr.x0 = pair[0]; pr.y0 = pair[1];
           }
           pp.queryarr("prescribed.by_fuel", pr.by_fuel);
           if (pr.by_fuel.size() % 2 != 0) {
               amrex::Abort("erf.fire.prescribed.by_fuel must be a flat list of fuel code, rate pairs");
           }
           if (ros_model == "prescribed") {
               bool any_positive = pr.ros > 0.0;
               for (std::size_t n = 1; n < pr.by_fuel.size(); n += 2) {
                   if (pr.by_fuel[n] < 0.0) { amrex::Abort("erf.fire.prescribed.by_fuel rates must be >= 0"); }
                   any_positive = any_positive || pr.by_fuel[n] > 0.0;
               }
               if (pr.ros < 0.0 || pr.min_ros < 0.0) {
                   amrex::Abort("erf.fire.prescribed.ros and prescribed.min_ros must be >= 0");
               }
               if (!any_positive && pr.min_ros <= 0.0) {
                   amrex::Abort("erf.fire.ros_model = prescribed needs prescribed.ros > 0, a positive prescribed.by_fuel rate or prescribed.min_ros > 0");
               }
           }

           auto& ph = prescribed_heat;
           pp.query("prescribed_heat.flux", ph.flux);
           pair.clear();
           if (pp.queryarr("prescribed_heat.center", pair)) {
               if (pair.size() != 2) { amrex::Abort("erf.fire.prescribed_heat.center needs two values: x y [m]"); }
               ph.cx = pair[0]; ph.cy = pair[1];
           }
           pp.query("prescribed_heat.radius",     ph.radius);
           pp.query("prescribed_heat.start_time", ph.start_time);
           pp.query("prescribed_heat.end_time",   ph.end_time);
           if (ph.flux < 0.0) { amrex::Abort("erf.fire.prescribed_heat.flux must be >= 0"); }
           if (ph.flux > 0.0 && ph.radius <= 0.0) {
               amrex::Abort("erf.fire.prescribed_heat.flux > 0 needs prescribed_heat.radius > 0");
           }
       }
       pp.query("exposure.enable",   exposure.enable);
       pp.query("exposure.file",     exposure.file);
       pp.query("exposure.interval", exposure.interval);
       pp.query("exposure.ring",     exposure.ring);
       // Checked against structures.enable below, once that block is read.

       // Non-burnable mask: structures, fuel codes, firebreaks
       pp.query("structures.enable",     structures.enable);
       pp.query("structures.file",       structures.file);
       pp.query("structures.min_height", structures.min_height);
       pp.query("structures.wind_open_columns", structures.wind_open_columns);
       pp.queryarr("fuel_map.nonburnable_codes", fuel_map.nonburnable_codes);
       pp.query("fuel_map.fuel_set",       fuel_map.fuel_set);
       pp.query("fuel_map.sb40_crosswalk", fuel_map.sb40_crosswalk);
       pp.query("fuel_map.load_from_map",  fuel_map.load_from_map);
       if (fuel_map.fuel_set != "anderson13" && fuel_map.fuel_set != "scott_burgan40") {
           amrex::Abort("erf.fire.fuel_map.fuel_set must be \"anderson13\" or \"scott_burgan40\"");
       }
       if (fuel_map.sb40_crosswalk && fuel_map.fuel_set != "scott_burgan40") {
           amrex::Abort("erf.fire.fuel_map.sb40_crosswalk needs fuel_map.fuel_set = scott_burgan40");
       }
       // The spotting distance table has to match the fuel codes the map carries:
       // Scott-Burgan codes (101-204) without the crosswalk need the "40" table.
       {
           const std::string want = (fuel_map.fuel_set == "scott_burgan40" && !fuel_map.sb40_crosswalk) ? "40" : "13";
           if (!pp.contains("spotting.fuel_system")) {
               spotting.fuel_system = want;
           } else if (spotting.enable && spotting.fuel_system != want) {
               fail("erf.fire.spotting.fuel_system = \"" + spotting.fuel_system + "\" does not match fuel_map.fuel_set = \""
                    + fuel_map.fuel_set + "\"" + (fuel_map.sb40_crosswalk ? " with sb40_crosswalk" : "") + " (needs \"" + want + "\")");
           }
       }
       if (fuel_map.fuel_set == "scott_burgan40") {
           const bool ok13 = (fuel_model_id >= 1 && fuel_model_id <= 13);
           const bool ok40 = (fuel_slot(fuel_model_id) >= 14);
           if (!ok13 && !ok40) {
               amrex::Abort("erf.fire.fuel_model_id must be an Anderson (1-13) or Scott-Burgan (101-204) code with fuel_set = scott_burgan40");
           }
           if (fuel_map.sb40_crosswalk) { fuel_model_id = sb40_to_anderson(fuel_model_id); }
       }
       pp.query("firebreak.use_mask", firebreak_use_mask);
       if (structures.enable) {
           if (structures.file.empty()) { structures.file = hybrid.structure_file; }
           if (structures.file.empty()) {
               amrex::ParmParse pp_erf("erf");
               pp_erf.query("buildings_file_name", structures.file);
           }
           if (structures.file.empty()) {
               amrex::Abort("erf.fire.structures.enable needs erf.fire.structures.file "
                            "or erf.buildings_file_name");
           }
           if (structures.min_height < 0.0) {
               amrex::Abort("erf.fire.structures.min_height must be >= 0");
           }
       }
       if (structures.wind_open_columns && !structures.enable) {
           amrex::Abort("erf.fire.structures.wind_open_columns needs erf.fire.structures.enable");
       }
       if (exposure.enable) {
           if (!structures.enable) {
               amrex::Abort("erf.fire.exposure.enable needs erf.fire.structures.enable");
           }
           if (exposure.interval < 1) { amrex::Abort("erf.fire.exposure.interval must be >= 1"); }
           if (exposure.ring < 1)     { amrex::Abort("erf.fire.exposure.ring must be >= 1"); }
       }
       if (heat_open_fraction && !structures.enable) {
           amrex::Abort("erf.fire.heat_open_fraction needs erf.fire.structures.enable");
       }

       // Balbi parameters
       auto& bl = balbi;
       pp.query("balbi.T_a",     bl.T_a);
       pp.query("balbi.T_f",     bl.T_f);
       pp.query("balbi.T_i",     bl.T_i);
       pp.query("balbi.delta_H", bl.delta_H);
       pp.query("balbi.C_pf",    bl.C_pf);
       pp.query("balbi.r_00",    bl.r_00);
       pp.query("balbi.tau_0",   bl.tau_0);

       // Balbi formulation selector, mapped onto the device-side integer flag
       {
           std::string balbi_formulation = (bl.formulation == 1) ? "2020" : "2009";
           pp.query("balbi.formulation", balbi_formulation);
           if (balbi_formulation == "2020") {
               bl.formulation = 1;
           } else if (balbi_formulation == "2009") {
               bl.formulation = 0;
           } else {
               amrex::Abort("erf.fire.balbi.formulation must be \"2009\" or \"2020\"");
           }
       }

       // Balbi (2020) constants and solver controls
       // Balbi et al. (2020) fit r00 = 2.5e-5, while the 2009 radiation length
       // scale is 2.5e-4 m. Move the default with the formulation unless the
       // user has set it explicitly.
       if (bl.formulation == 1 && !pp.contains("balbi.r_00")) {
           bl.r_00 = 2.5e-5;
       }

       pp.query("balbi.chi_0",    bl.chi_0);
       pp.query("balbi.C_pa",     bl.C_pa);
       pp.query("balbi.C_pw",     bl.C_pw);
       pp.query("balbi.T_vap",    bl.T_vap);
       pp.query("balbi.K1",       bl.K1);
       pp.query("balbi.st",       bl.st);
       pp.query("balbi.rho_a",    bl.rho_a);
       pp.query("balbi.sigma_B",  bl.sigma_B);
       pp.query("balbi.max_iter", bl.max_iter);
       pp.query("balbi.tol",      bl.tol);

       // Balbi coupling options
       pp.query("balbi.directional",                bl.directional);
       pp.query("balbi.use_surface_temp",           bl.use_surface_temp);
       pp.query("balbi.flame_temp_from_combustion", bl.flame_temp_from_combustion);
       pp.query("balbi.use_cell_moisture",          bl.use_cell_moisture);
       pp.query("balbi.use_moisture_extinction",    bl.use_moisture_extinction);
       pp.query("balbi.herb_curing",                bl.herb_curing);
       pp.query("balbi.heat_flux_coupling",         bl.heat_flux_coupling);

       {
           std::string balbi_wind_source = (bl.wind_source == 1) ? "reference" : "midflame";
           pp.query("balbi.wind_source", balbi_wind_source);
           if (balbi_wind_source == "reference") {
               bl.wind_source = 1;
           } else if (balbi_wind_source == "midflame") {
               bl.wind_source = 0;
           } else {
               amrex::Abort("erf.fire.balbi.wind_source must be "
                            "\"midflame\" or \"reference\"");
           }
       }
       pp.query("balbi.k_upward",                   bl.k_upward);
       pp.query("balbi.hf_ref_height",              bl.hf_ref_height);

       // Cheney-Gould parameters
       auto& cg = cheney_gould;
       pp.query("cheney_gould.moisture", cg.moisture);
       pp.query("cheney_gould.curing",   cg.curing);

       auto& fb = fbp;
       pp.query("fbp.fuel_type", fb.fuel_type);
       pp.query("fbp.ffmc",      fb.ffmc);
       pp.query("fbp.bui",       fb.bui);
       pp.query("fbp.curing",    fb.curing);
       pp.query("fbp.pc",        fb.pc);
       pp.query("fbp.pdf",       fb.pdf);
       pp.query("fbp.use_slope", fb.use_slope);
       {
           std::string ws = (fb.wind_source == 1) ? "midflame" : "reference";
           pp.query("fbp.wind_source", ws);
           if (ws == "reference")     { fb.wind_source = 0; }
           else if (ws == "midflame") { fb.wind_source = 1; }
           else { amrex::Abort("erf.fire.fbp.wind_source must be \"reference\" or \"midflame\""); }
       }
       if (uses_model("fbp")) {
           if (fbp_type_from_name(fb.fuel_type) < 0) {
               amrex::Abort("erf.fire.fbp.fuel_type must be one of C1-C7, D1, M1-M4, S1-S3, O1A, O1B");
           }
           if (fb.ffmc < 0.0 || fb.ffmc > 101.0 || fb.bui < 0.0 || fb.curing < 0.0 || fb.curing > 100.0
               || fb.pc < 0.0 || fb.pc > 100.0 || fb.pdf < 0.0 || fb.pdf > 100.0) {
               amrex::Abort("erf.fire.fbp: ffmc in [0,101], bui >= 0, curing/pc/pdf in [0,100]");
           }
       }

       // Phase 15: BEHAVE parameters
       auto& bh = behave;
       pp.query("behave.dynamic_transfer_lo", bh.dynamic_transfer_lo);
       pp.query("behave.dynamic_transfer_hi", bh.dynamic_transfer_hi);
       if (uses_model("behave") && !(bh.dynamic_transfer_hi > bh.dynamic_transfer_lo)) {
           amrex::Abort("erf.fire.behave.dynamic_transfer_hi must be greater than dynamic_transfer_lo");
       }

       // Phase 10: firebreaks (indexed, stop at first missing)
       for (int nb = 0; nb < 10; ++nb) {
           std::string type_key = "firebreak." + std::to_string(nb) + ".type";
           std::string type_val;
           if (!pp.query(type_key.c_str(), type_val)) break;
           FireBreak fb;
           fb.type = type_val;
           one_of(type_key.c_str(), type_val, {"rect", "rectangle", "circle"});   // the kernel branches on the first letter
           std::string pfx = "firebreak." + std::to_string(nb) + ".";
           pp.query((pfx + "x_lo").c_str(),   fb.x_lo);
           pp.query((pfx + "y_lo").c_str(),   fb.y_lo);
           pp.query((pfx + "x_hi").c_str(),   fb.x_hi);
           pp.query((pfx + "y_hi").c_str(),   fb.y_hi);
           pp.query((pfx + "cx").c_str(),     fb.cx);
           pp.query((pfx + "cy").c_str(),     fb.cy);
           pp.query((pfx + "radius").c_str(), fb.radius);
           firebreaks.push_back(fb);
       }
       n_firebreaks = static_cast<int>(firebreaks.size());

       // The Huygens ellipse shapes the front itself, so it supersedes the
       // directional default; asking for both explicitly is a conflict.
       if (levelset_ellipse) {
           if ((directional_ros_given && directional_ros) || balbi.directional) {
               amrex::Abort("erf.fire.levelset.ellipse cannot be combined with directional_ros = true or balbi.directional");
           }
           directional_ros = false;
       }

       // A prescribed rate has no direction, so it supersedes the directional
       // default in the same way; asking for both explicitly is a conflict.
       if (ros_model == "prescribed") {
           if (directional_ros_given && directional_ros) {
               amrex::Abort("erf.fire.ros_model = prescribed is isotropic and cannot be combined with directional_ros = true");
           }
           if (levelset_ellipse) {
               amrex::Abort("erf.fire.ros_model = prescribed cannot be combined with erf.fire.levelset.ellipse");
           }
           directional_ros = false;
       }

       // The ellipse shape is a form of the directional rate of the generic
       // models; Balbi and the hybrid keep the projection. Checked last, once
       // the ellipse and prescribed options have had their say on
       // directional_ros.
       if (directional_shape == 1) {
           if (!directional_ros) {
               amrex::Abort("erf.fire.directional_shape = \"ellipse\" needs directional_ros = true and cannot be combined with erf.fire.levelset.ellipse or ros_model = prescribed");
           }
           if (ros_model == "balbi" || is_hybrid()) {
               amrex::Abort("erf.fire.directional_shape = \"ellipse\" is implemented for rothermel, behave, macarthur, cheney_gould and fbp, not balbi or hybrid");
           }
       }

       // directional_wind_coupling = "advective" is read only by directional_ros_cell's
       // Rothermel branch (and only reached with directional_shape = "projection");
       // every other shape, model or the hybrid path (whose fill_directional_ros
       // call in ERF_HybridRos.H does not thread this flag through) would silently
       // ignore it, so reject those combinations instead.
       if (directional_wind_coupling == 1) {
           if (directional_shape == 1) {
               amrex::Abort("erf.fire.directional_wind_coupling = \"advective\" needs erf.fire.directional_shape = \"projection\" (it has no effect with \"ellipse\")");
           }
           if (is_hybrid() || !uses_model("rothermel")) {
               amrex::Abort("erf.fire.directional_wind_coupling = \"advective\" is implemented for erf.fire.ros_model = \"rothermel\" only, not hybrid or another model");
           }
       }
    }

    /// True when ros_model = "hybrid".
    bool is_hybrid() const { return ros_model == "hybrid"; }

    /**
     * @brief Whether a rate-of-spread model is evaluated in this run.
     *
     * Covers the single-model case and both members of a hybrid pair.
     * "rothermel" is the fallback for any unrecognised ros_model string,
     * matching the dispatch in FireLayer.
     */
    bool uses_model(const std::string& m) const
    {
        if (is_hybrid()) { return hybrid.primary == m || hybrid.secondary == m; }
        if (m == "rothermel") {
            return ros_model != "balbi" && ros_model != "behave" &&
                   ros_model != "macarthur" && ros_model != "cheney_gould" && ros_model != "fbp" &&
                   ros_model != "prescribed";
        }
        return ros_model == m;
    }

    // Convenience accessors for coupling mode
    bool is_passive()     const { return coupling_type == "passive"; }
    bool is_lagged()      const { return coupling_type == "lagged"; }
    bool is_synchronous() const { return coupling_type == "synchronous"; }
    bool injects_flux()   const { return is_lagged() || is_synchronous(); }
    /// The CFBM partition: sensible flux scaled by the dry-fuel fraction 1 / (1 + M_f).
    bool cfbm_partition() const { return heat_flux_partition == "cfbm"; }
};

#endif
