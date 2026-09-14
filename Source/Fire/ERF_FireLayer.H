#ifndef ERF_FIRE_LAYER_H
#define ERF_FIRE_LAYER_H

#include <AMReX_MultiFab.H>
#include <string>
#include <AMReX_Geometry.H>
#include <memory>
#include <ERF_FireParams.H>
#include <ERF_FireGrid.H>
#include <ERF_Rothermel.H>
#include <ERF_FarsiteEllipse.H>
#include <ERF_FireIgnition.H>
#include <ERF_FuelMoisture.H>
#include <ERF_MoistureExtinction.H>
#include <ERF_FireStatsOutput.H>
#include <ERF_FireHeatFlux.H>
#include <ERF_FireDiagnostics.H>
#include <ERF_FireAtmCoupling.H>
#include <ERF_AlbiniSpotting.H>
#include <ERF_ScottSpottingTable.H>
#include <ERF_CrownFire.H>
#include <ERF_FuelMap.H>
#include <ERF_FireBreak.H>
#include <ERF_FuelBlending.H>
#include <ERF_IgnitionSchedule.H>
#include <ERF_PolygonIgnition.H>
#include <ERF_LevelSetEllipse.H>
#include <ERF_FuelModels.H>
#include <ERF_FbpModel.H>
#include <ERF_FuelMoistureStick.H>
#include <ERF_FireAcceleration.H>
// Phase 13A: Per-fuel wind height
#include <ERF_FuelWindHeight.H>
// Phase 13B: ROS models
#include <ERF_BalbiModel.H>
#include <ERF_DirectionalRos.H>
#include <ERF_CheneyGouldModel.H>
#include <ERF_MacArthurModel.H>
// Phase 15: BEHAVE multi-class Rothermel
#include <ERF_BehaveModel.H>
// Level-set propagation headers
#include <ERF_NumericalSchemes.H>
#include <ERF_Reinitialize.H>
#include <ERF_LevelSetAdvection.H>

// Forward declarations
class SurfaceLayer;
class ERF;

/**
 * @file ERF_FireLayer.H
 * @brief Main fire layer container and orchestrator
 *
 * The FireLayer class is the primary interface between the ERF atmospheric
 * solver and the wildfire spread module. It owns all fire-specific data
 * (level-set, wind, ROS, slopes, etc.) and orchestrates the computation
 * pipeline each timestep.
 *
 * Owned by ERF as std::unique_ptr<FireLayer> m_fire_layer, created in
 * ERF::InitData_post() if erf.fire.enable = true.
 */

/**
 * @class FireLayer
 * @brief Manages fire simulation on a 2D level-set grid
 *
 * Maintains all fire-related MultiFabs on the refined fire grid and calls
 * the wind extraction, WAF, terrain, and Rothermel computations each timestep.
 *
 * Multifabs (all on FireGrid, size C*nx × C*ny × 1):
 *
 * | Name           | ncomp | Description                                  |
 * |----------------|-------|----------------------------------------------|
 * | fire_phi       | 1     | Level-set: <0 burned, >0 unburned, ≈0 front  |
 * | fire_wind_ref  | 2     | u,v at z_ref (e.g., 6.1 m) from MOST        |
 * | fire_wind_eff  | 2     | After WAF + terrain corrections              |
 * | fire_slopes    | 2     | dz/dx, dz/dy (static, computed at init)      |
 * | fire_surface_z | 1     | Ground elevation [m] of the atmospheric column |
 * | fire_col_ground| 4     | Ground [m] of the 4 bilinear wind-stencil columns |
 * | fire_curvature | 1     | Terrain curvature (static, at init)          |
 * | fire_ros       | 1     | Rate of spread [m/s]                         |
 * | fire_fuel_load | 1     | Fuel load [kg/m²]                            |
 * | fire_fuel_mc   | 5     | M_1hr, M_10hr, M_100hr, M_lh, M_lw [fraction] |
 * | fire_mext      | 1     | Moisture of extinction [fraction] (Phase 4)  |
 * | fire_heat_flux | 1     | Heat flux [W/m²]                             |
 * | fire_spread_vec| 2     | Scratch for FARSITE spread direction (Phase 3)|
 * | fire_crown_active  | 1 | Crown fire status: 1=active crown, 0=surface fire (Phase 9) |
 * | fire_crown_load    | 1 | Remaining crown fuel load [kg/m²] (Phase 9)                 |
 * | fire_crown_fraction_burned | 1 | Crown fraction burned [0-1] (Phase 9)            |
 * | fire_flame_tilt    | 1 | Flame tilt angle [deg from vertical] (Phase 9)             |
 * | fire_flame_temp    | 1 | Flame temperature [K] (Phase 9)                             |
 *
 * **fire_fuel_mc component layout (Phase 15, 5 components):**
 *   - comp 0: M_1hr      [fraction] — 1-hour dead fuel moisture (single-class path)
 *   - comp 1: M_10hr     [fraction] — 10-hour dead fuel moisture (single-class path)
 *   - comp 2: M_100hr    [fraction] — 100-hour dead fuel moisture (single-class path)
 *   - comp 3: M_lh       [fraction] — Live herbaceous moisture (BEHAVE multi-class path only)
 *   - comp 4: M_lw       [fraction] — Live woody moisture (BEHAVE multi-class path only)
 */
class FireLayer
{
public:

    /** @brief Constructor */
    FireLayer() = default;

    /** @brief Destructor */
    ~FireLayer() = default;

    /**
     * @brief Initialize fire module after SurfaceLayer is ready.
     *
     * Called from ERF::InitData_post(). Performs:
     * 1. Prerequisite verification (verify_fire_prerequisites)
     * 2. Create fire grid (create_fire_grid)
     * 3. Allocate all MultiFabs
     * 4. Compute static fields: terrain slopes, curvature
     * 5. Initialize fire level-set (ignition circle)
     * 6. Set initial fuel properties (moisture, fuel load)
     *
     * The caller (ERF, a friend of itself) passes the private members as
     * explicit arguments so that FireLayer never needs to access ERF's
     * private/protected data directly.
     *
     * @param[in] erf                ERF instance — used only for public
     *                               AmrCore accessors (boxArray, DistributionMap,
     *                               Geom).
     * @param[in] surface_layer_ptr  Pointer to the SurfaceLayer owned by ERF
     *                               (pass m_SurfaceLayer.get() at the call site).
     *                               May be nullptr when called before the surface
     *                               layer is constructed (prereq check only).
     * @param[in] z_phys_nd_atm      Atmospheric nodal terrain-height MultiFab
     *                               (pass z_phys_nd[0].get() at the call site);
     *                               nullptr on flat terrain (TerrainType::None).
     * @param[in] fire_params        Fire configuration struct.
     *
     * @throw amrex::Error if prerequisites not met.
     */
    void initialize(const ERF&            erf,
                    const SurfaceLayer*   surface_layer_ptr,
                    const amrex::MultiFab* z_phys_nd_atm,
                    const FireParams&     fire_params);

    /**
     * @brief Advance fire simulation by one atmospheric timestep.
     *
     * Called from ERF::Advance() each timestep.
     * Executes the fire computation pipeline:
     * 1. Extract wind from atmosphere at reference height
     * 2. Apply Wind Adjustment Factor for canopy
     * 3. Apply terrain wind corrections (FARSITE)
     * 4. Update fuel moisture from atmospheric state (Phase 2)
     * 5. Compute effective midflame wind
     * 6. Compute rate-of-spread using Rothermel model
     * 7. (Phase 3) Advance level-set using ROS and wind
     * 8. (Phase 6) Compute heat flux to atmosphere
     *
     * @param[in]     dt             Timestep [s]
     * @param[in,out] surface_layer  SurfaceLayer with MOST diagnostics.
     *                               Non-const because get_u_star / get_z0 /
     *                               get_olen are not marked const in
     *                               ERF_SurfaceLayer.H.
     * @param[in]     xvel           x-velocity on atmospheric grid [m/s]
     * @param[in]     yvel           y-velocity on atmospheric grid [m/s]
     * @param[in]     z_phys_cc      Cell-center terrain height on atmospheric grid [m]
     * @param[in]     T_atm_k0       Atmospheric potential temperature at k=0 [K]
     *                               On atmospheric grid (coarser than fire grid).
     * @param[in]     RH_atm_k0      Atmospheric relative humidity at k=0 [fraction 0-1]
     *                               On atmospheric grid (coarser than fire grid).
     */
    void advance(amrex::Real time,
                 amrex::Real dt,
                 SurfaceLayer& surface_layer,
                 const amrex::MultiFab& xvel,
                 const amrex::MultiFab& yvel,
                 const amrex::MultiFab& z_phys_cc,
                 const amrex::MultiFab& T_atm_k0,
                 const amrex::MultiFab& RH_atm_k0);

    /**
     * @brief Get heat flux for atmospheric coupling.
     * @return Pointer to fire_heat_flux MultiFab (always allocated).
     */
    const amrex::MultiFab* get_heat_flux() const
    {
        return fire_heat_flux.get();
    }

    /**
     * @brief Get rate-of-spread field for diagnostics.
     * @return Pointer to fire_ros MultiFab [m/s].
     */
    const amrex::MultiFab* get_ros() const
    {
        return fire_ros.get();
    }

    /**
     * @brief Get the hybrid ROS per-cell weight of the secondary model.
     * @return Pointer to fire_ros_weight [0,1], or nullptr unless ros_model = "hybrid".
     */
    const amrex::MultiFab* get_ros_weight() const
    {
        return fire_ros_weight.get();
    }

    /**
     * @brief Get the structure height sampled onto the fire grid.
     * @return Pointer to fire_structure_height [m], or nullptr unless the
     *         hybrid structure selector is active.
     */
    const amrex::MultiFab* get_structure_height() const
    {
        return fire_structure_height.get();
    }

    /**
     * @brief Get the non-burnable mask (1 in structure, listed fuel-code and
     *        masked firebreak cells, 0 elsewhere).
     * @return Pointer to fire_nonburnable, or nullptr when no mask source is on.
     */
    const amrex::MultiFab* get_nonburnable() const
    {
        return fire_nonburnable.get();
    }

    /**
     * @brief Get effective wind field for diagnostics.
     * @return Pointer to fire_wind_eff MultiFab [m/s].
     */
    const amrex::MultiFab* get_wind_eff() const
    {
        return fire_wind_eff.get();
    }

    /**
     * @brief Get level-set field for diagnostics.
     * @return Pointer to fire_phi MultiFab.
     */
    const amrex::MultiFab* get_levelset() const
    {
        return fire_phi.get();
    }

    /**
     * @brief Get mutable level-set field for checkpoint restart.
     * @return Mutable pointer to fire_phi MultiFab.
     */
    amrex::MultiFab* get_levelset_mut()
    {
        return fire_phi.get();
    }

    /**
     * @brief Get fire arrival time field for diagnostics.
     * @return Pointer to fire_arrival_time MultiFab.
     */
    const amrex::MultiFab* get_arrival_time() const
    {
        return fire_arrival_time.get();
    }

    /**
     * @brief Get mutable fire arrival time field for checkpoint restart.
     * @return Mutable pointer to fire_arrival_time MultiFab.
     */
    amrex::MultiFab* get_arrival_time_mut()
    {
        return fire_arrival_time.get();
    }

    /**
     * @brief Get mutable fuel load field for checkpoint restart.
     * @return Mutable pointer to fire_fuel_load MultiFab.
     */
    amrex::MultiFab* get_fuel_load_mut()
    {
        return fire_fuel_load.get();
    }

    /**
     * @brief Get mutable fuel moisture field for checkpoint restart.
     * @return Mutable pointer to fire_fuel_mc MultiFab (5 components).
     */
    amrex::MultiFab* get_fuel_mc_mut()
    {
        return fire_fuel_mc.get();
    }

    /**
     * @brief Get the FARSITE sub-cell displacement accumulator.
     * @return Pointer to fire_disp_accum MultiFab (2 components), or nullptr.
     */
    const amrex::MultiFab* get_disp_accum() const
    {
        return fire_disp_accum.get();
    }

    /**
     * @brief Get mutable displacement accumulator for checkpoint restart.
     * @return Mutable pointer to fire_disp_accum MultiFab.
     */
    amrex::MultiFab* get_disp_accum_mut()
    {
        return fire_disp_accum.get();
    }

    /**
     * @brief Get mutable crown activation state for checkpoint restart.
     * @return Mutable pointer to fire_crown_active MultiFab, or nullptr when
     *         crown fire is disabled.
     */
    amrex::MultiFab* get_crown_active_mut()
    {
        return fire_crown_active.get();
    }

    /**
     * @brief Get mutable remaining crown fuel load for checkpoint restart.
     * @return Mutable pointer to fire_crown_load MultiFab, or nullptr when
     *         crown fire is disabled.
     */
    amrex::MultiFab* get_crown_load_mut()
    {
        return fire_crown_load.get();
    }

    /**
     * @brief Get fire geometry for plotfile output.
     * @return Const reference to the fire grid geometry.
     */
    const amrex::Geometry& get_fire_geom() const
    {
        return m_fg.geom;
    }

    /**
     * @brief Get fire grid refinement factor.
     * @return Fire-to-atmospheric grid ratio.
     */
    int get_grid_ratio() const
    {
       return m_fg.C;
    }

    /**
     * @brief Get reference wind field for plotfile output.
     * @return Pointer to fire_wind_ref MultiFab [m/s].
     */
    const amrex::MultiFab* get_wind_ref() const
    {
        return fire_wind_ref.get();
    }

    /**
     * @brief Get wind extraction height for plotfile output.
     * @return Pointer to fire_wind_extract_z MultiFab [m].
     */
    const amrex::MultiFab* get_wind_extract_z() const
    {
        return fire_wind_extract_z.get();
    }

    /**
     * @brief Get terrain slopes for plotfile output.
     * @return Pointer to fire_slopes MultiFab [dz/dx, dz/dy].
     */
    const amrex::MultiFab* get_slopes() const
    {
        return fire_slopes.get();
    }

    /**
     * @brief Get fuel moisture content for plotfile output.
     * @return Pointer to fire_fuel_mc MultiFab [M_1hr, M_10hr, M_100hr].
     */
    const amrex::MultiFab* get_fuel_mc() const
    {
        return fire_fuel_mc.get();
    }

    /**
     * @brief Get fuel load for plotfile output.
     * @return Pointer to fire_fuel_load MultiFab [kg/m²].
     */
    const amrex::MultiFab* get_fuel_load() const
    {
        return fire_fuel_load.get();
    }

    /**
     * @brief Get surface temperature for plotfile output.
     * @return Pointer to fire_surface_temp MultiFab [K].
     */
    const amrex::MultiFab* get_surface_temp() const
    {
        return fire_surface_temp.get();
    }

    /**
     * @brief Get surface relative humidity for plotfile output.
     * @return Pointer to fire_surface_rh MultiFab [0-1].
     */
    const amrex::MultiFab* get_surface_rh() const
    {
        return fire_surface_rh.get();
    }

    /**
     * @brief Get fireline intensity for diagnostics and plotfile output.
     * @return Pointer to fire_fireline_intensity MultiFab [kW/m].
     */
    const amrex::MultiFab* get_fireline_intensity() const
    {
        return fire_fireline_intensity.get();
    }

    /**
     * @brief Get flame length for diagnostics and plotfile output.
     * @return Pointer to fire_flame_length MultiFab [m].
     */
    const amrex::MultiFab* get_flame_length() const
    {
        return fire_flame_length.get();
    }


    /**
     * @brief Get flame temperature for diagnostics and plotfile output.
     * @return Pointer to fire_flame_temp MultiFab [K].
     */
    const amrex::MultiFab* get_flame_temp() const
    {
        return fire_flame_temp.get();
    }

    /**
     * @brief Get flame tilt angle for diagnostics and plotfile output.
     * @return Pointer to fire_flame_tilt MultiFab [deg from vertical], or nullptr when disabled.
     */
    const amrex::MultiFab* get_flame_tilt() const
    {
        return fire_flame_tilt.get();
    }

    /**
     * @brief Get crown fire activation field for diagnostics and plotfile output.
     * @return Pointer to fire_crown_active MultiFab, or nullptr when crown fire is disabled.
     */
    const amrex::MultiFab* get_crown_active() const
    {
        return fire_crown_active.get();
    }

    /**
     * @brief Get remaining crown fuel load for diagnostics and plotfile output.
     * @return Pointer to fire_crown_load MultiFab [kg/m²], or nullptr when crown fire is disabled.
     */
    const amrex::MultiFab* get_crown_load() const
    {
        return fire_crown_load.get();
    }

    /**
     * @brief Get crown fraction burned diagnostic for plotfile output.
     * @return Pointer to fire_crown_fraction_burned MultiFab [0-1].
     */
    const amrex::MultiFab* get_crown_fraction_burned() const
    {
        return fire_crown_fraction_burned.get();
    }

    /**
     * @brief Get Albini spotting diagnostics for plotfile output.
     * @return Pointer to fire_albini_data MultiFab (nullptr when spotting disabled).
     */
    const amrex::MultiFab* get_albini_data() const
    {
        return fire_albini_data.get();
    }

    /**
     * @brief Get fire grid reference for plotfile output.
     * @return Const reference to the FireGrid.
     */
    const FireGrid& get_fire_grid() const
    {
        return m_fg;
    }

    /**
     * @brief Get coarsened sensible heat flux on atmospheric grid from previous timestep.
     * @return Pointer to m_Q_atm_prev MultiFab (nullptr until first advance).
     */
    const amrex::MultiFab* get_Q_atm_prev() const
    {
        return m_Q_atm_prev.get();
    }

    /**
     * @brief Get coarsened latent heat flux on atmospheric grid from previous timestep.
     * @return Pointer to m_Q_lat_atm_prev MultiFab (nullptr until first advance).
     */
    const amrex::MultiFab* get_Q_lat_atm_prev() const
    {
        return m_Q_lat_atm_prev.get();
    }

    /// Mutable access to the lagged sensible flux buffer, for checkpoint restart.
    amrex::MultiFab* get_Q_atm_prev_mut() { return m_Q_atm_prev.get(); }
    amrex::MultiFab* get_heat_flux_mut()  { return fire_heat_flux.get(); }
    amrex::Real get_f_dry_prev() const { return m_f_dry_prev; }
    void set_f_dry_prev(amrex::Real f) { m_f_dry_prev = f; }

    /// Shell moistures of the stick model (3 classes x n_shells), null unless erf.fire.moisture_model = stick.
    const amrex::MultiFab* get_stick_mc() const { return fire_stick_mc.get(); }
    /// Mutable access to the stick state, for checkpoint restart.
    amrex::MultiFab* get_stick_mc_mut() { return fire_stick_mc.get(); }
    /// Allocate the stick shells, every shell at its class's current moisture (no-op if allocated).
    void allocate_stick_mc();

    /**
     * @brief Stamp the perimeter polygon (or polyline) of
     *        erf.fire.ignition.polygon_file into the level set.
     *
     * Called at initialisation when erf.fire.ignition.polygon_time is 0, or
     * from advance() on the step whose window contains that time. With
     * erf.fire.ignition.polygon_interior_ros > 0 the cells the polygon
     * newly ignites get the interior state of a fire that spread outward at
     * that rate: arrival time t_ign - d / R and fuel reduced by
     * exp(-d / (R tau)) with d the distance inside the perimeter and tau
     * erf.fire.ignition.polygon_interior_tau (dx_fire / R when 0).
     * @param t_ign Time [s] the perimeter is valid at.
     */
    void apply_polygon_ignition(amrex::Real t_ign);

    /**
     * @brief Burnout e-folding time of a fuel model for erf.fire.burnout_model = sfire.
     *
     * WRF-SFIRE's burn time per Anderson model (7, 7, 7, 180, 100, 100, 100,
     * 900, 900, 900, 900, 900, 900 s; Jimenez y Munoz et al. 2026, Table 1),
     * or erf.fire.burnout_times_s, divided by erf.fire.burnout_time_to_efold.
     * @param fuel_code Anderson fuel model 1-13 (others fall back to 1).
     */
    amrex::Real burnout_tau_s(int fuel_code) const;

    /// Parameters of the uniform fuel model, erf.fire.fuel_model_id, under the deck's fuel set.
    FuelModelParams uniform_fuel_params() const
    {
        FuelModelParams fp = get_fuel_params(m_params.fuel_model_id, m_params.fuel_map.fuel_set_id(), m_params.moisture_live);
        // erf.fire.heat_content_btu_lb: uniform override for matching a
        // specific WRF-Fire run's cmbcnst, see ERF_FireParams.H.
        if (m_params.heat_content_override_btu_lb > 0.0) {
            fp.heat_content = m_params.heat_content_override_btu_lb;
        }
        return fp;
    }

    /// The uniform fuel model's parameters for BEHAVE: the table without its curing transfer (behave_fuel_params).
    FuelModelParams uniform_behave_fuel_params() const
    {
        return behave_fuel_params(m_params.fuel_model_id, m_params.fuel_map.fuel_set_id());
    }

    /**
     * @brief Heat per kilogram of fuel that converts the lagged atmospheric
     *        sensible flux back into fuel burnt, for the smoke emission.
     *
     * The fuel model's heat content when erf.fire.smoke_heat_from_fuel is
     * true, otherwise erf.fire.smoke_heat_of_comb, multiplied by the
     * partition factor the buffer was last filled with (1 under the legacy
     * partition, 1 / (1 + M_f) under the CFBM one), so the fuel burnt is
     * recovered from the unpartitioned heat release and the smoke does not
     * change with erf.fire.heat_flux_partition.
     * @return Heat per kilogram of fuel [J/kg] to divide the lagged flux by.
     */
    amrex::Real smoke_heat_per_kg_atm() const;

    /// Mutable access to the lagged latent flux buffer, for checkpoint restart.
    amrex::MultiFab* get_Q_lat_atm_prev_mut() { return m_Q_lat_atm_prev.get(); }

    /// Set the fire step counter, for a restart: the spotting seed and the
    /// interval-based outputs derive from it.
    void set_step(int step) { m_step = step; }
    int  get_step() const { return m_step; }
    /// The level-set subcycle count that schedules the reinitialisation; carried
    /// by the checkpoint so a restart keeps the schedule of the uninterrupted run.
    int  get_levelset_subcycle_count() const { return m_levelset_subcycle_count; }
    void set_levelset_subcycle_count(int n) { m_levelset_subcycle_count = n; }
    /// Fields that carry across steps and are checkpointed alongside the ones
    /// above: the spotting diagnostics (recomputed only every spotting interval),
    /// the temporal acceleration state, the crown ROS carried between steps and
    /// the crown fraction burned.
    const amrex::MultiFab* get_accel_state() const      { return fire_accel_state.get(); }
    const amrex::MultiFab* get_crown_ros_active() const { return fire_crown_ros_active.get(); }
    amrex::MultiFab* get_albini_data_mut()           { return fire_albini_data.get(); }
    amrex::MultiFab* get_accel_state_mut()           { return fire_accel_state.get(); }
    amrex::MultiFab* get_crown_ros_active_mut()      { return fire_crown_ros_active.get(); }
    amrex::MultiFab* get_crown_fraction_burned_mut() { return fire_crown_fraction_burned.get(); }

    /// Structure id on the fire grid (0 = none); null unless exposure.enable.
    const amrex::MultiFab* get_structure_id() const { return fire_structure_id.get(); }
    /// Accumulated heat load [J/m2]; null unless exposure.enable.
    const amrex::MultiFab* get_heat_load() const { return fire_heat_load.get(); }
    amrex::MultiFab* get_heat_load_mut() { return fire_heat_load.get(); }
    /// Running maximum of the fireline intensity [kW/m]; null unless exposure.enable.
    const amrex::MultiFab* get_peak_intensity() const { return fire_peak_intensity.get(); }
    amrex::MultiFab* get_peak_intensity_mut() { return fire_peak_intensity.get(); }
    /// Embers that landed on each cell of a footprint; null unless exposure.enable.
    const amrex::MultiFab* get_ember_landings() const { return fire_ember_landings.get(); }
    amrex::MultiFab* get_ember_landings_mut() { return fire_ember_landings.get(); }

    /**
     * @brief Get fire latent heat flux on fire grid.
     * @return Pointer to fire_latent_flux MultiFab [W/m²].
     */
    const amrex::MultiFab* get_latent_flux() const
    {
        return fire_latent_flux.get();
    }

    /**
     * @brief Store the current fire heat flux into the previous-step buffer.
     *
     * Called from ERF::Advance() at the END of the fire advance, after
     * compute_heat_flux_and_diagnostics() has filled fire_heat_flux.
     * Coarsens fire_heat_flux from fire grid to atmospheric grid and
     * computes latent flux.
     *
     * @param[in] geom_atm  Atmospheric geometry (used for coarsening)
     */
    void update_atm_flux_buffer(const amrex::Geometry& geom_atm);

    /**
     * @brief Apply the previous-step fire fluxes to cc_source using WRF-Fire profile.
     *
     * Injects the one-step-lagged fire flux from previous timestep into cc_source
     * using the exponential distribution profile. Called from ERF::Advance()
     * BEFORE advance_dycore().
     *
     * @param[in,out] cc_source    3D source array for conserved variables
     * @param[in]     S_old        atmospheric conserved state at time n (for density)
     * @param[in]     z_phys_cc    cell-centre terrain height [m]
     * @param[in]     geom_atm     atmospheric geometry
     * @param[in]     has_moisture whether moisture is active
     */
    void apply_fire_coupling_to_cc_source(
        amrex::MultiFab& cc_source,
        const amrex::MultiFab& S_old,
        const amrex::MultiFab& z_phys_cc,
        const amrex::Geometry& geom_atm,
        bool has_moisture);

    /**
     * @brief Get the fire parameters for access to coupling type and other settings.
     * @return Const reference to the FireParams.
     */
    const FireParams& get_params() const { return m_params; }

    /**
     * @brief Get the per-cell fuel model MultiFab (Phase 10).
     * @return Const pointer to fire_fuel_model MultiFab (nullptr if uniform fuel).
     */
    const amrex::MultiFab* get_fuel_model() const { return fire_fuel_model.get(); }

    /**
     * @brief Check if spatial fuel map is active (Phase 10).
     * @return true if per-cell fuel model is loaded and active.
     */
    bool has_spatial_fuel() const { return m_has_spatial_fuel; }

private:

    // Fire grid and geometry
    FireGrid                                 m_fg;

    // Fire state MultiFabs (all on m_fg grid, C*nx × C*ny × 1)
    std::unique_ptr<amrex::MultiFab>         fire_phi;          // Level-set
    std::unique_ptr<amrex::MultiFab>         fire_wind_ref;     // Wind at z_ref
    std::unique_ptr<amrex::MultiFab>         fire_wind_eff;     // Effective wind
    std::unique_ptr<amrex::MultiFab>         fire_wind_extract_z; // Wind extraction height [m] (z_surf + z_ref)
    std::unique_ptr<amrex::MultiFab>         fire_slopes;       // Terrain slopes
    std::unique_ptr<amrex::MultiFab>         fire_surface_z;    // Ground elevation of the atm column
    std::unique_ptr<amrex::MultiFab>         fire_col_ground;   // Ground of the 4 wind-stencil columns
    std::unique_ptr<amrex::MultiFab>         fire_curvature;    // Terrain curvature
    std::unique_ptr<amrex::MultiFab>         fire_ros;          // Rate of spread
    std::unique_ptr<amrex::MultiFab>         fire_fuel_load;    // Fuel load
    std::unique_ptr<amrex::MultiFab>         fire_fuel_mc;      // Fuel moisture (5 comp: 1-h, 10-h, 100-h, live herb, live woody)
    std::unique_ptr<amrex::MultiFab>         fire_stick_mc;     // stick model shells: 3 classes x n_shells [fraction]
    std::unique_ptr<amrex::MultiFab>         fire_mext;         // Moisture of extinction (Phase 4)
    std::unique_ptr<amrex::MultiFab>         fire_heat_flux;    // Heat flux [W/m2] on the fire grid
    std::unique_ptr<amrex::MultiFab>         fire_spread_vec;   // FARSITE scratch
    std::unique_ptr<amrex::MultiFab>         fire_arrival_time; // Fire arrival time [s]
    std::unique_ptr<amrex::MultiFab>         fire_disp_accum;   // Persistent FARSITE accumulator (2 comp): displacement (legacy), or head-rate distance and front flag (front_cell)
    std::unique_ptr<amrex::MultiFab>         fire_surface_temp; // Surface temperature [K] (Phase 2)
    std::unique_ptr<amrex::MultiFab>         fire_surface_rh;   // Surface relative humidity [0-1] (Phase 2)
    std::unique_ptr<amrex::MultiFab>         fire_fireline_intensity; // Byram fireline intensity [kW/m]
    std::unique_ptr<amrex::MultiFab>         fire_flame_length;       // Thomas flame length [m]
    std::unique_ptr<amrex::MultiFab>         fire_flame_tilt;         // Flame tilt angle [deg from vertical] (Phase 9)
    std::unique_ptr<amrex::MultiFab>         fire_flame_temp;         // Flame temperature [K] (Phase 9)

    // Phase 9: Crown fire diagnostics (2 comp, on fire grid)
    // comp 0: crown fire status (1.0 = active crown, 0.0 = surface),
    // comp 1: remaining crown fuel load [kg/m²]
    std::unique_ptr<amrex::MultiFab>         fire_crown_active;       // Crown fire status: 1=active, 0=surface (Phase 9)
    std::unique_ptr<amrex::MultiFab>         fire_crown_load;         // Remaining crown fuel load [kg/m²] (Phase 9)
    std::unique_ptr<amrex::MultiFab>         fire_crown_ros_active;   // Active crown ROS [m/s], carried from
                                                                      // apply_crown_fire_ros() to the heat-flux pass
    std::unique_ptr<amrex::MultiFab>         fire_crown_fraction_burned; // Crown fraction burned [0-1] (Phase 9)

    // Phase 8: Albini spotting diagnostics (4 comp, on fire grid)
    // comp 0: lofting height H_z [m], comp 1: brand count,
    // comp 2: max landing dist [m], comp 3: active spot flag
    std::unique_ptr<amrex::MultiFab>         fire_albini_data;

    // Phase 12: Per-cell acceleration state for temporal model (3 components).
    // comp 0: ROS_current [m/s], comp 1: elapsed_time [s], comp 2: ROS_equilibrium [m/s].
    // Only allocated when accel.enable = true AND accel.use_temporal = true.
    // nullptr when acceleration is disabled or size-based model is used.
    std::unique_ptr<amrex::MultiFab>  fire_accel_state;

    // Phase 10: Per-cell fuel model (non-null when fuel_map.file is non-empty)
    std::unique_ptr<amrex::MultiFab>         fire_fuel_model;  // integer code stored as Real
    amrex::Gpu::DeviceVector<int>            m_d_fuel_codes;   // device copy of integer codes
    bool                                     m_has_spatial_fuel = false;

    // Phase 11: Multi-ignition schedule runtime state
    IgnitionSchedule  m_ignition_schedule;   // loaded from ignition_schedule_file
    bool              m_has_schedule = false;

    // Phase 6: fire-atmosphere coupling MultiFabs
    // Coarsened fire fluxes on the atmospheric level-0 grid (nx × ny × 1).
    // These hold fire fluxes from the previous timestep for injection into
    // cc_source at the start of the next timestep (one-step explicit lag).
    std::unique_ptr<amrex::MultiFab>         m_Q_atm_prev;            // sensible [W/m²]
    std::unique_ptr<amrex::MultiFab>         m_Q_lat_atm_prev;        // latent   [W/m²]
    amrex::Real                              m_f_dry_prev = 1.0;      // partition factor m_Q_atm_prev was filled with
    bool                                     m_polygon_applied = false; // the timed perimeter polygon has been stamped

    // Open-fraction heat placement (erf.fire.heat_open_fraction): per
    // atmospheric column, the fraction of the footprint not covered by
    // non-burnable structure cells and the mean roof height of those cells,
    // both area-averaged from the fire grid at initialisation.
    std::unique_ptr<amrex::MultiFab>         m_open_frac_atm;         // [0-1]
    std::unique_ptr<amrex::MultiFab>         m_roof_h_atm;            // [m]

    /// Fill m_open_frac_atm and m_roof_h_atm from the mask and structure height.
    void build_open_fraction(const amrex::Geometry& geom_atm);
    std::unique_ptr<amrex::MultiFab>         fire_latent_flux;        // latent heat flux on fire grid [W/m²]

    // Precomputed Rothermel coefficients
    RothermelComputed                        m_rc;

    // FARSITE parameters for level-set propagation
    FarsiteParams                            m_fp;

    // Time tracking for fire advancement
    amrex::Real                              m_current_time = 0.0;
    amrex::Real                              m_dt_atm       = 0.0;
    int                                      m_step         = 0;  ///< Simulation step counter

    // Level-set propagation tracking
    int                                      m_levelset_subcycle_count = 0;  ///< Counts fire subcycles for reinit scheduling

    // Vertical levels
    int                                      m_nz = 0;

    // Fuel bed depth [ft] (fuel-model dependent)
    amrex::Real                              m_fuel_bed_depth_ft = 6.5;

    // Initial fuel load [kg/m²] for diagnostics (Phase 5)
    amrex::Real                              m_fuel_load_initial_kg_m2 = 0.0;

    // Phase 13A: Per-fuel wind height device tables
    amrex::Gpu::DeviceVector<amrex::Real>    m_d_fcwh;          // fcwh[0..13], device copy
    amrex::Gpu::DeviceVector<amrex::Real> m_d_burnout_tau;   ///< burnout e-folding time by fuel code (index 1-13) [s], erf.fire.burnout_model = sfire
    amrex::Gpu::DeviceVector<amrex::Real>    m_d_fcz0;          // fcz0[0..13], device copy
    bool                                     m_use_per_fuel_wind_ht = false;

    // Phase 13B: Pre-computed ROS model coefficients (set in initialize, used in advance)
    BalbiComputed                            m_bc_default;      // Balbi global default (uniform fuel)
    CheneyGouldComputed                      m_cgc;             // Cheney-Gould computed params
    FbpComputed         m_fbp;     ///< Canadian FBP state (ros_model = fbp)
    amrex::Gpu::DeviceVector<BalbiComputed>  m_d_balbi_table;   // per-fuel Balbi table (spatial fuel)

    // Per-fuel Rothermel table; empty unless rothermel_per_fuel is set on a
    // spatial fuel map, in which case the Rothermel kernel looks it up per cell.
    amrex::Gpu::DeviceVector<RothermelComputed> m_d_rc_table;

    // Hybrid ROS (ros_model = "hybrid"): per-cell weight of the secondary
    // model in [0,1] and the scratch field the secondary model writes into.
    std::unique_ptr<amrex::MultiFab>         fire_ros_weight;
    std::unique_ptr<amrex::MultiFab>         fire_ros_scratch;

    /// Evaluate one named ROS model into `out` (uniform-fuel Balbi inputs in balbi_in).
    /// Rate of spread for ros_model = "prescribed" (ERF_PrescribedFire.H).
    void fill_prescribed_ros(amrex::MultiFab& out) const;

    /// Add erf.fire.prescribed_heat.flux to fire_heat_flux over its disc and time window.
    void add_prescribed_heat_flux();

    void fill_ros_for_model(const std::string& model,
                            amrex::MultiFab& out,
                            const BalbiFieldInputs& balbi_in);

    /// Fill fire_ros_weight from a static hybrid selector (region, fuel, structure).
    void init_ros_weight();

    /// Rebuild fire_ros_weight from the effective wind speed (selector = wind).
    void update_wind_weight();

    /// Debug line with the weight sum and the count of cells above one half.
    void print_hybrid_weight_summary() const;

    /// Precomputed coefficients of a named non-Balbi model for the directional path.
    DirectionalRosState make_directional_state(const std::string& model) const;

    /// Print the arrival time at each erf.fire.probes point the first time it burns.
    void report_probes();

    // Structure height [m] on the fire grid (structures.enable or the hybrid
    // structure selector).
    std::unique_ptr<amrex::MultiFab>         fire_structure_height;

    // Non-burnable mask on the fire grid, one ghost cell so the level-set
    // stencils and the FARSITE stamp can test neighbours. Null when no source
    // (structures, fuel codes, masked firebreaks) is configured.
    std::unique_ptr<amrex::MultiFab>         fire_nonburnable;

    // Exposure diagnostics (erf.fire.exposure.enable). The id carries
    // exposure.ring ghost cells so the wall band can be found from any cell;
    // the three accumulators are restart state and are checkpointed.
    std::unique_ptr<amrex::MultiFab>         fire_structure_id;     // 1..N, 0 outside
    std::unique_ptr<amrex::MultiFab>         fire_heat_load;        // sum Q dt [J/m2]
    std::unique_ptr<amrex::MultiFab>         fire_peak_intensity;   // max I_B [kW/m]
    std::unique_ptr<amrex::MultiFab>         fire_ember_landings;   // brands landed on a footprint
    int                                      m_n_structures = 0;
    std::vector<char>                        m_exposure_reported;   // first-arrival line printed

    /// Label the footprints of the structure heightmap and allocate the accumulators.
    void build_structure_ids();

    /// Append one line per structure to the exposure CSV and print first arrivals.
    void report_exposure();

    /// Read the structure heightmap onto the fire grid (nearest-cell sampling).
    void load_structure_height(const std::string& file);

    /// Build fire_nonburnable from structures, fuel codes and firebreaks.
    void build_nonburnable_mask();

    /// Hold phi outside the burned region in mask cells (level-set path).
    void enforce_nonburnable_phi();

    /// Zero the rate of spread in mask cells.
    void zero_ros_in_mask(amrex::MultiFab& ros) const;

    // Arrival-time probes already reported.
    std::vector<bool>                        m_probe_reported;

    /// Rebuild m_d_rc_table at the given dead fuel moistures.
    void rebuild_rothermel_table(amrex::Real m1, amrex::Real m10, amrex::Real m100);

    // Phase 15: BEHAVE pre-computed state (set in initialize from domain-avg moisture,
    // updated in advance when moisture_dynamic = false)
    BehaveState                              m_bs_default;

    // Ignition parameters
    amrex::Real                              m_ignition_x = 0.0;
    amrex::Real                              m_ignition_y = 0.0;
    amrex::Real                              m_ignition_r = 20.0;

    // Configuration
    FireParams                               m_params;


    /**
     * @brief Apply Wind Adjustment Factor to the effective wind field.
     *
     * Multiplies fire_wind_eff by a scalar WAF. Used inside advance().
     */
    void apply_waf_to_wind();

    // nvcc forbids an extended __device__ lambda inside a function with private or
    // protected class access, and both methods below launch ParallelFor kernels, so
    // they are public even though they are only called from advance().
public:

    /**
     * @brief Apply the crown-fire rate-of-spread enhancement to fire_ros.
     *
     * Must be called from advance() BEFORE the front is propagated: the whole
     * point of the enhancement is that the perimeter moves at the crowning rate.
     * It also latches fire_crown_active and records the active crown ROS in
     * fire_crown_ros_active, which compute_heat_flux_and_diagnostics() then uses
     * for the crown heat release and canopy fuel depletion.
     */
    void apply_crown_fire_ros();

    /**
     * @brief Update fire_fuel_mc and fire_mext for one atmospheric timestep.
     *
     * Called from advance() before ComputeRothermellSpreadRate. Updates
     * fuel moisture using time-lag ODEs and recomputes moisture of extinction.
     *
     * @param[in] dt_s       Atmospheric timestep [seconds]
     * @param[in] T_atm_k0   Atmospheric potential temperature at k=0 [K]
     * @param[in] RH_atm_k0  Atmospheric relative humidity at k=0 [fraction 0-1]
     */
    void advance_fuel_moisture(amrex::Real dt_s,
                               const amrex::MultiFab& T_atm_k0,
                               const amrex::MultiFab& RH_atm_k0);

    /**
     * @brief Compute heat flux and deplete fuel load for one atmospheric timestep (Phase 5).
     *
     * Called from advance() after fire_subcycle. Computes:
     * - Sensible heat flux [W/m²] in fire_heat_flux
     * - Fuel load depletion via exponential burnout model
     * - Byram fireline intensity [kW/m] in fire_fireline_intensity
     * - Thomas flame length [m] in fire_flame_length
     *
     * @param[in] dt_fire_s  Last fire subcycle timestep [s]
     */
    void compute_heat_flux_and_diagnostics(amrex::Real dt_fire_s);

};

#endif