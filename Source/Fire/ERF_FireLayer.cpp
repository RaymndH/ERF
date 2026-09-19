#include <ERF_FireLayer.H>
#include <ERF_PrescribedFire.H>
#include <ERF.H>
#include <ERF_SurfaceLayer.H>
#include <ERF_FirePrerequisites.H>
#include <ERF_FireGrid.H>
#include <ERF_FireWindExtract.H>
#include <ERF_TerrainSlope.H>
#include <ERF_HybridRos.H>
#include <ERF_FireTerrainReader.H>
#include <ERF_FireBoundaryGuard.H>
#include <ERF_HostFabView.H>
#include <fstream>
#include <sstream>
#include <limits>
#include <array>
#include <iomanip>

#include <AMReX_Reduce.H>

using namespace amrex;

// Debug diagnostics over the fire grid.
//
// These run as AMReX device reductions rather than host loops: with
// the_arena_is_managed defaulting to false, a MultiFab's data is device-only
// under CUDA and cannot be dereferenced from the host at all.
//
// Namespace scope, not FireLayer members, because nvcc forbids an extended
// __device__ lambda inside a function with private or protected class access.
namespace erf_fire_diag {

struct BurningRosStats {
    amrex::Real max_ros  = 0.0;   ///< Max ROS over burning cells [m/s]
    amrex::Real mean_ros = 0.0;   ///< Mean ROS over burning cells [m/s]
    amrex::Long n_cells  = 0;     ///< Number of burning cells
};

// Masked ROS statistics over burning cells (phi < 0), reduced across ranks.
// With no burning cells anywhere, max and mean are both 0, matching the
// host-loop version this replaces.
BurningRosStats burning_ros_stats (const amrex::MultiFab& ros,
                                   const amrex::MultiFab& phi)
{
    ReduceOps<ReduceOpMax, ReduceOpSum, ReduceOpSum> reduce_op;
    ReduceData<Real, Real, unsigned long long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(ros); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const ros_arr = ros.const_array(mfi);
        auto const phi_arr = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            if (phi_arr(i,j,k,0) < 0.0_rt) {
                const Real r = ros_arr(i,j,k,0);
                return {r, r, 1ull};
            }
            return {0.0_rt, 0.0_rt, 0ull};
        });
    }

    ReduceTuple hv = reduce_data.value(reduce_op);
    Real max_ros = amrex::get<0>(hv);
    Real sum_ros = amrex::get<1>(hv);
    amrex::Long n_cells = static_cast<amrex::Long>(amrex::get<2>(hv));

    ParallelDescriptor::ReduceRealMax(max_ros);
    ParallelDescriptor::ReduceRealSum(sum_ros);
    ParallelDescriptor::ReduceLongSum(n_cells);

    BurningRosStats stats;
    stats.max_ros  = max_ros;
    stats.n_cells  = n_cells;
    stats.mean_ros = (n_cells > 0) ? sum_ros / Real(n_cells) : 0.0_rt;
    return stats;
}

// Number of cells with phi < 0, reduced across ranks.
amrex::Long count_burning_cells (const amrex::MultiFab& phi)
{
    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<unsigned long long> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.tilebox();
        auto const phi_arr = phi.const_array(mfi);
        reduce_op.eval(bx, reduce_data,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
        {
            return { (phi_arr(i,j,k,0) < 0.0_rt) ? 1ull : 0ull };
        });
    }

    amrex::Long n = static_cast<amrex::Long>(amrex::get<0>(reduce_data.value(reduce_op)));
    ParallelDescriptor::ReduceLongSum(n);
    return n;
}

} // namespace erf_fire_diag

namespace {

// Where the fire grid sits, printed once at start-up, and the warning when a
// finer level covers part of it: the fire then reads that level's coarser wind,
// and its heat on this level is replaced by the finer level's average-down.
void report_fire_region(const ERF& erf, int lev, const FireGrid& fg)
{
    const Geometry& g = fg.geom;
    amrex::Print() << "[FIRE] Fire grid on level " << lev << " of 0-" << erf.finestLevel()
                   << ": x " << g.ProbLo(0) << " to " << g.ProbHi(0)
                   << " m, y " << g.ProbLo(1) << " to " << g.ProbHi(1) << " m, "
                   << g.Domain().length(0) << " x " << g.Domain().length(1) << " fire cells of "
                   << g.CellSize(0) << " x " << g.CellSize(1) << " m\n";

    if (lev >= erf.finestLevel()) { return; }

    // Share of the fire region that level lev + 1 covers, in columns of level lev.
    const Box region(IntVect(fg.atm_lo[0], fg.atm_lo[1], 0),
                     IntVect(fg.atm_lo[0] + g.Domain().length(0) / fg.C - 1,
                             fg.atm_lo[1] + g.Domain().length(1) / fg.C - 1, 0));
    BoxArray fine = erf.boxArray(lev + 1);
    fine.coarsen(erf.refRatio(lev));
    amrex::Long covered = 0;
    for (int n = 0; n < fine.size(); ++n) {
        Box b = fine[n];
        b.setSmall(2, 0);
        b.setBig(2, 0);
        b &= region;
        if (b.ok()) { covered += b.numPts(); }
    }
    const amrex::Real share = amrex::Real(covered) / amrex::Real(region.numPts());
    amrex::Print() << "[FIRE] WARNING: the fire grid is on level " << lev << " but the finest level is "
                   << erf.finestLevel() << " (erf.fire.anchor_level = " << lev << "), and level " << lev + 1
                   << " covers " << 100.0 * share << " % of the fire grid. The fire reads the wind of level "
                   << lev << " and its heat, moisture and smoke go into level " << lev << " only: with "
                   << "erf.coupling_type = TwoWay the average-down from the finer level replaces the heated cells "
                   << "under it, so that heat is lost, and with OneWay the finer level never sees the fire. "
                   << "Leave erf.fire.anchor_level unset to put the fire grid on the finest level.\n";
}

} // namespace

void FireLayer::initialize(const ERF& erf,
                            int lev,
                            const SurfaceLayer* surface_layer_ptr,
                            const MultiFab* z_phys_nd_atm,
                            const FireParams& fire_params)
{
    m_params = fire_params;
    m_lev    = lev;
    verify_fire_prerequisites(erf, lev, surface_layer_ptr, fire_params);
    m_fg = create_fire_grid(erf.boxArray(lev), erf.DistributionMap(lev),
                            erf.Geom(lev), fire_params.grid_ratio);
    m_fg.lev = lev;
    m_nz = erf.Geom(lev).Domain().length(2);
    report_fire_region(erf, lev, m_fg);

    fire_phi        = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 3);
    fire_wind_ref   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_wind_eff   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_wind_extract_z = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_slopes     = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 1);
    fire_surface_z  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_col_ground = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 4, 0);
    fire_curvature  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_ros        = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_fuel_load  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_fuel_mc    = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 5, 0);
    fire_mext       = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_heat_flux  = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_spread_vec = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_arrival_time = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_disp_accum   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 2, 0);
    fire_surface_temp = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_surface_rh   = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);

    // Phase 5: Heat flux and diagnostics fields
    fire_fireline_intensity = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_flame_length = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);

    // Phase 9 diagnostics
    fire_flame_temp = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_crown_fraction_burned = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    if (m_params.compute_flame_tilt) {
        fire_flame_tilt = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    }
    if (m_params.crown.enable) {
        fire_crown_active = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_crown_load = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_crown_ros_active = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    }

    fire_phi->setVal(1.0);
    fire_spread_vec->setVal(0.0_rt);
    fire_wind_ref->setVal(0.0);
    fire_wind_eff->setVal(0.0);
    fire_wind_extract_z->setVal(0.0);
    fire_slopes->setVal(0.0);
    fire_curvature->setVal(0.0);
    fire_ros->setVal(0.0);
    // Written by compute_heat_flux_and_diagnostics() from the first step on;
    // the initial plotfile carried whatever the allocator held before that.
    fire_heat_flux->setVal(0.0_rt);
    fire_arrival_time->setVal(-1.0_rt);
    fire_disp_accum->setVal(0.0_rt);
    fire_surface_temp->setVal(0.0);
    fire_surface_rh->setVal(0.0);
    // Rain per fire cell: the atmosphere's rain per column (erf.fire.precip_source =
    // atmosphere) or the uniform rate, kept only when it can wet the dead classes.
    if (m_params.moisture_dynamic &&
        (m_params.precip_source == "atmosphere" || m_params.precip_rate_mm_hr > 0.0_rt)) {
        fire_precip_rate = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_precip_rate->setVal(m_params.precip_source == "atmosphere" ? 0.0_rt : m_params.precip_rate_mm_hr);
    }
    fire_fireline_intensity->setVal(0.0_rt);
    fire_flame_length->setVal(0.0_rt);
    fire_flame_temp->setVal(0.0_rt);
    fire_crown_fraction_burned->setVal(0.0_rt);
    if (fire_flame_tilt) {
        fire_flame_tilt->setVal(0.0_rt);
    }
    if (fire_crown_active) {
        fire_crown_active->setVal(0.0_rt);
    }
    if (fire_crown_load) {
        fire_crown_load->setVal(m_params.crown.canopy_bulk_den * m_params.crown.canopy_depth);
    }
    if (fire_crown_ros_active) {
        fire_crown_ros_active->setVal(0.0_rt);
    }

    // Phase 6: fire-atmosphere coupling MultiFabs
    const BoxArray& ba_atm = erf.boxArray(lev);
    const DistributionMapping& dm_atm = erf.DistributionMap(lev);
    BoxArray ba_atm_2d = ba_atm;
    ba_atm_2d.coarsen(IntVect(1, 1, erf.Geom(lev).Domain().length(2)));
    // Ghost columns for the MRF fire thermal excess, which reads the fluxes in the
    // halo around every tile. ERF's state carries ComputeGhostCells() + 1 columns,
    // at most 5, and ComputeDiffusivityMRF keeps its halo within the state's and
    // checks these fluxes against it; update_atm_flux_buffer() fills them.
    const IntVect ng_flux(5, 5, 0);
    m_Q_atm_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, ng_flux);
    m_Q_lat_atm_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, ng_flux);
    m_Q_atm_prev->setVal(0.0_rt);
    m_Q_lat_atm_prev->setVal(0.0_rt);
    // The surface precipitation accumulation the rain per column is measured from
    // (erf.fire.precip_source = atmosphere): zero at a fresh start, as the scheme's
    // accumulators are; a restart restores it from the checkpoint.
    if (m_params.precip_source == "atmosphere") {
        m_precip_accum_prev = std::make_unique<MultiFab>(ba_atm_2d, dm_atm, 1, 0);
        m_precip_accum_prev->setVal(0.0_rt);
        m_precip_prev_valid = true;
    }

    fire_latent_flux = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_latent_flux->setVal(0.0_rt);

    // Phase 8: Albini spotting diagnostics
    if (m_params.spotting.enable) {
        fire_albini_data = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 4, 0);
        fire_albini_data->setVal(0.0_rt);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Albini spotting enabled: "
                           << "I_B_min=" << m_params.spotting.I_B_min << " kW/m, "
                           << "P_base=" << m_params.spotting.P_base << "\n";
        }
    }

    // Phase 12: Allocate per-cell acceleration state for temporal model.
    // Only allocated when both enable and use_temporal are true.
    // When disabled or size-based, fire_accel_state stays nullptr.
    if (m_params.accel.enable && m_params.accel.use_temporal) {
        fire_accel_state = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 3, 0);
        fire_accel_state->setVal(0.0_rt);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Fire acceleration enabled (temporal model): "
                           << "A_point=" << m_params.accel.A_point << " 1/min, "
                           << "A_line=" << m_params.accel.A_line << " 1/min\n";
        }
    } else if (m_params.accel.enable) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Fire acceleration enabled (size-based model): "
                           << "L_acc=" << m_params.accel.L_acc << " m\n";
        }
    }

    FuelModelParams fp = get_fuel_params(fire_params.fuel_model_id, fire_params.fuel_map.fuel_set_id(),
                                         fire_params.moisture_live);
    if (fire_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Uniform fuel model code=" << fire_params.fuel_model_id
                       << " set=" << fire_params.fuel_map.fuel_set << (fire_params.fuel_map.sb40_crosswalk ? " (crosswalk)" : "")
                       << " M_live=" << fire_params.moisture_live
                       << " w_d1=" << fp.w_d1 << " w_d10=" << fp.w_d10 << " w_d100=" << fp.w_d100
                       << " w_lh=" << fp.w_lh << " w_lw=" << fp.w_lw << " sigma_d1=" << fp.sigma_d1
                       << " delta=" << fp.delta << " Mx=" << fp.Mx << " heat=" << fp.heat_content
                       << " (lb/ft2, 1/ft, ft, -, BTU/lb)" << std::endl;
    }
    fire_fuel_load->setVal((fp.w_d1+fp.w_d10+fp.w_d100+fp.w_lh+fp.w_lw)*4.88243);
    m_fuel_load_initial_kg_m2 = (fp.w_d1+fp.w_d10+fp.w_d100+fp.w_lh+fp.w_lw)*4.88243_rt;
    m_fuel_bed_depth_ft = fp.delta;

    fire_fuel_mc->setVal(0.0);
    for (MFIter mfi(*fire_fuel_mc); mfi.isValid(); ++mfi) {
        Array4<Real> mc = fire_fuel_mc->array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            mc(iv,0) = fire_params.moisture_1hr;
            mc(iv,1) = fire_params.moisture_10hr;
            mc(iv,2) = fire_params.moisture_100hr;
            mc(iv,3) = fire_params.moisture_live;  // live herbaceous (Phase 15, new component)
            mc(iv,4) = fire_params.moisture_live;  // live woody      (Phase 15, new component)
        });
    }

    fire_mext->setVal(compute_moisture_of_extinction(
        dead_fuel_weighted_sav(fp.w_d1, fp.w_d10, fp.w_d100, fp.sigma_d1)));

    compute_terrain_slopes(*fire_slopes, z_phys_nd_atm, erf.Geom(lev), m_fg, m_params.terrain_file_name);

    // Ground elevation of each fire cell's atmospheric column, used as the datum
    // for wind extraction. Always from the atmospheric terrain, even when a finer
    // terrain file supplies the slopes, since the wind profile being interpolated
    // belongs to that column.
    compute_fire_surface_height(*fire_surface_z, z_phys_nd_atm, erf.Geom(lev), m_fg);

    // Grounds of the four columns the bilinear wind stencil blends, so each can
    // be sampled at the same height above its own terrain.
    compute_fire_column_grounds(*fire_col_ground, z_phys_nd_atm, erf.Geom(lev), m_fg);
    fire_fill_boundary(*fire_slopes, m_fg.geom);
    compute_terrain_curvature(*fire_curvature, *fire_slopes, m_fg.geom);

    m_ignition_x = fire_params.ignition_x;
    m_ignition_y = fire_params.ignition_y;
    m_ignition_r = fire_params.ignition_r;

    // The level-set solver needs a true signed distance in metres; the FARSITE
    // path keeps the normalized [-1, 1] indicator convention.
    const bool phi_normalized = (m_params.propagation_method != "levelset");
    initialize_ignition(*fire_phi, m_fg.geom, m_ignition_x, m_ignition_y, m_ignition_r,
                        phi_normalized);
    fire_fill_boundary(*fire_phi, m_fg.geom);

    for (MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
        auto phi_arr = fire_phi->const_array(mfi);
        auto at_arr  = fire_arrival_time->array(mfi);
        ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv) {
            if (phi_arr(iv) < 0.0_rt) at_arr(iv) = 0.0_rt;
        });
    }

    // Phase 10: Load spatial fuel map from file when specified.
    // File reading is CPU-only on rank 0; broadcast to all ranks; copy to device.
    if (!m_params.fuel_map.fuel_map_file.empty()) {
        const int fire_nx = m_fg.ba.minimalBox().length(0);
        const int fire_ny = m_fg.ba.minimalBox().length(1);
        std::vector<int> h_fuel_codes;
        int nodata_val = -9999;
        bool ok = false;
        if (m_params.fuel_map.fuel_map_format == "lcp") {
            ok = read_lcp_fuel_map(m_params.fuel_map.fuel_map_file,
                                   fire_nx, fire_ny, h_fuel_codes);
        } else {
            ok = read_ascii_fuel_map(m_params.fuel_map.fuel_map_file,
                                     fire_nx, fire_ny, h_fuel_codes, nodata_val,
                                     m_fg.geom.CellSize(0));
        }
        if (ok && m_params.fuel_map.sb40_crosswalk) {
            // The Community Fire Behavior Model's route for LANDFIRE data: the
            // Scott-Burgan codes become Anderson codes here, non-burnable 0.
            for (auto& c : h_fuel_codes) { c = sb40_to_anderson(c); }
        }
        if (ok) {
            m_d_fuel_codes.resize(h_fuel_codes.size());
            amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                             h_fuel_codes.begin(), h_fuel_codes.end(),
                             m_d_fuel_codes.begin());
            fire_fuel_model = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
            fill_fuel_model_mf(*fire_fuel_model, m_d_fuel_codes.data(),
                               m_fg.geom, fire_nx);
            m_has_spatial_fuel = true;
            if (m_params.fuel_map.load_from_map) {
                // Each cell starts with its own model's load rather than the
                // uniform one; non-burnable codes of the Scott-Burgan set carry
                // none, and unknown codes follow the set's fall-through.
                fill_fuel_load_from_map(*fire_fuel_load, *fire_fuel_model,
                                        m_params.fuel_map.fuel_set_id(),
                                        m_params.fuel_map.sb40_active(),
                                        m_params.moisture_live);
                if (m_params.fire_debug) {
                    const Real dA = m_fg.geom.CellSize(0) * m_fg.geom.CellSize(1);
                    amrex::Print() << "[FIRE DEBUG] Fuel load from the map: " << fire_fuel_load->sum(0) * dA
                                   << " kg on the grid\n";
                }
            }
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] Loaded spatial fuel map '"
                               << m_params.fuel_map.fuel_map_file << "': "
                               << fire_nx << "x" << fire_ny << " cells\n";
            }
        } else {
            amrex::Print() << "[FIRE] WARNING: Cannot read fuel map '"
                           << m_params.fuel_map.fuel_map_file
                           << "'; using uniform fuel_model_id="
                           << m_params.fuel_model_id << "\n";
        }
    }

    // Phase 10: Apply firebreak barriers after ignition stamp.
    if (!m_params.firebreaks.empty() && fire_phi) {
        apply_firebreaks(*fire_phi, m_params.firebreaks, m_fg.geom);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Applied "
                           << m_params.n_firebreaks << " firebreak(s)\n";
        }
    }

    // Phase 11: Polygon ignition (initial fire perimeter from vertex file).
    // Applied at t=0 as part of initialization, before the schedule, unless
    // erf.fire.ignition.polygon_time defers it to advance().
    if (!m_params.ignition.polygon_file.empty() && m_params.ignition.polygon_time <= 0.0) {
        apply_polygon_ignition(0.0);
    }

    // Phase 11: Load ignition schedule if specified.
    // File reading and broadcast are handled inside load_ignition_schedule().
    if (!m_params.ignition.ignition_schedule_file.empty()) {
        load_ignition_schedule(m_params.ignition.ignition_schedule_file,
                               m_ignition_schedule);
        m_has_schedule = true;
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Loaded ignition schedule: "
                           << m_ignition_schedule.events.size()
                           << " events\n";
        }
    }

    m_fp.phi_threshold      = fire_params.farsite_phi_threshold;
    m_fp.coeff_a            = fire_params.farsite_coeff_a;
    m_fp.coeff_b            = fire_params.farsite_coeff_b;
    m_fp.coeff_c            = fire_params.farsite_coeff_c;
    m_fp.use_anderson_lw    = fire_params.farsite_use_anderson_lw;
    m_fp.gaussian_sigma     = fire_params.farsite_gaussian_sigma;
    m_fp.cfl_fire           = fire_params.farsite_cfl_fire;
    m_fp.front_update       = fire_params.farsite_front_update;

    m_rc = compute_rothermel_params(fp, fire_params.moisture_1hr,
                                    fire_params.moisture_10hr,
                                    fire_params.moisture_100hr,
                                    fire_params.use_wind_limit,
                                    fire_params.reaction_velocity_formula == "rothermel",
                                    fire_params.wrf_bmst_compat);
    if (fire_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Init Rothermel coefficients: R0=" << m_rc.R0
                       << " m/s  I_R=" << m_rc.I_R << " beta=" << m_rc.beta << std::endl;
    }

    // Phase 13A: Build per-fuel wind height tables and copy to device.
    // When use_per_fuel_wind_ht = false, all entries equal wind_ref_ht (no-op).
    m_use_per_fuel_wind_ht = m_params.use_per_fuel_wind_ht;
    {
        auto h_fcwh = build_fcwh_table(m_params.wind_ref_ht, m_params.use_per_fuel_wind_ht);
        auto h_fcz0 = build_fcz0_table();
        m_d_fcwh.resize(h_fcwh.size());
        m_d_fcz0.resize(h_fcz0.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_fcwh.begin(), h_fcwh.end(), m_d_fcwh.begin());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_fcz0.begin(), h_fcz0.end(), m_d_fcz0.begin());
        if (m_params.fire_debug && m_params.use_per_fuel_wind_ht) {
            amrex::Print() << "[FIRE DEBUG] Per-fuel wind height enabled. "
                           << "FM1 fcwh=" << h_fcwh[1] << " m, FM4 fcwh=" << h_fcwh[4] << " m\n";
        }
    }

    // Phase 13B: Pre-compute alternative ROS model coefficients.
    {
        FuelModelParams fp_ros = uniform_fuel_params();
        if (m_params.uses_model("balbi")) {
            m_bc_default = compute_balbi_params(fp_ros, m_params.balbi,
                                                m_params.moisture_1hr);
            // Build per-fuel Balbi table when spatial fuel map is active
            if (m_has_spatial_fuel) {
                auto h_balbi = build_fuel_balbi_table(m_params.balbi, m_params.moisture_1hr, -1.0, m_params.fuel_map.fuel_set_id(), m_params.moisture_live);
                m_d_balbi_table.resize(h_balbi.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_balbi.begin(),
                                 h_balbi.end(), m_d_balbi_table.begin());
            }
            if (m_params.fire_debug) {
                if (m_params.balbi.formulation == 1) {
                    amrex::Print() << "[FIRE DEBUG] ROS model: Balbi (2020), A_rad="
                                   << m_bc_default.A_rad << ", Rb_coef="
                                   << m_bc_default.Rb_coef << " m/(s K^4), u0_coef="
                                   << m_bc_default.u0_coef << " m/s, s*r00="
                                   << m_bc_default.s_r00 << "\n";
                } else {
                    amrex::Print() << "[FIRE DEBUG] ROS model: Balbi (2009), A_coeff="
                                   << m_bc_default.A_coeff << " m/s, v_b="
                                   << m_bc_default.v_b << " m/s\n";
                }
                if (m_params.balbi.directional) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: direction-dependent ROS on "
                                   << "the level-set path\n";
                }
                if (m_params.balbi.use_surface_temp) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: per-cell ambient temperature "
                                   << "from fire_surface_temp\n";
                }
                if (m_params.balbi.use_cell_moisture) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: per-cell fuel moisture from "
                                   << "the moisture ODE state\n";
                }
                if (m_params.balbi.use_moisture_extinction) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: moisture-of-extinction cutoff "
                                   << "at M_x=" << fp_ros.Mx << "\n";
                }
                if (m_params.balbi.wind_source == 1) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: reference-height wind "
                                   << "(WAF and terrain correction bypassed)\n";
                }
                if (m_params.balbi.heat_flux_coupling) {
                    amrex::Print() << "[FIRE DEBUG] Balbi: heat-flux buoyancy coupling, "
                                   << "k_upward=" << m_params.balbi.k_upward
                                   << ", H_ref=" << m_params.balbi.hf_ref_height << " m\n";
                }
            }
        }
        if (m_params.uses_model("cheney_gould")) {
            m_cgc = compute_cheney_gould_params(m_params.cheney_gould);
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: Cheney-Gould (1998), "
                               << "moisture=" << m_params.cheney_gould.moisture
                               << "%, curing=" << m_params.cheney_gould.curing << "\n";
            }
        }
        if (m_params.uses_model("behave")) {
            // Phase 15: Pre-compute BEHAVE multi-class coefficients from the
            // untransferred table; BEHAVE cures the herbaceous load itself.
            FuelModelParams fp_bh = uniform_behave_fuel_params();
            m_bs_default = compute_behave_state(fp_bh,
                                                m_params.moisture_1hr,
                                                m_params.moisture_10hr,
                                                m_params.moisture_100hr,
                                                m_params.moisture_live,
                                                m_params.moisture_live,
                                                m_params.behave.dynamic_transfer_lo,
                                                m_params.behave.dynamic_transfer_hi,
                                                m_params.use_wind_limit);
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: BEHAVE multi-class Rothermel, "
                               << "R0=" << m_bs_default.r_0 << " m/s, live herbaceous transfer window "
                               << m_params.behave.dynamic_transfer_lo << "-"
                               << m_params.behave.dynamic_transfer_hi << "\n";
            }
        }
        if (m_params.uses_model("fbp")) {
            const auto& fb = m_params.fbp;
            m_fbp.type      = fbp_type_from_name(fb.fuel_type);
            m_fbp.fF        = fbp_ffmc_function(fb.ffmc);
            m_fbp.be        = fbp_buildup_effect(m_fbp.type, fb.bui);
            m_fbp.cf        = (m_fbp.type == FBP_O1A || m_fbp.type == FBP_O1B) ? fbp_curing_factor(fb.curing) : 1.0;
            m_fbp.pc        = fb.pc;
            m_fbp.pdf       = fb.pdf;
            m_fbp.use_slope = fb.use_slope;
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] FBP: fuel type " << fb.fuel_type << " FFMC=" << fb.ffmc
                               << " f(F)=" << m_fbp.fF << " BUI=" << fb.bui << " BE=" << m_fbp.be
                               << " CF=" << m_fbp.cf << " wind_source=" << (fb.wind_source == 1 ? "midflame" : "reference")
                               << " (wind_ref_ht=" << m_params.wind_ref_ht << " m; the system wants the 10 m wind)"
                               << " ROS at 0 and 20 km/h, flat: " << fbp_ros(m_fbp, 0.0, 0.0) << " " << fbp_ros(m_fbp, 20.0 / 3.6, 0.0) << " m/s\n";
            }
        }
        if (m_params.uses_model("macarthur")) {
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: MacArthur (1966) Australian formula\n";
            }
        }
        if (m_params.uses_model("rothermel")) {
            // Rothermel coefficients were initialised above via m_rc.
            if (m_params.fire_debug) {
                amrex::Print() << "[FIRE DEBUG] ROS model: Rothermel (1972)\n";
            }
        }
    }

    // Per-cell Rothermel coefficients on a spatial fuel map (opt-in). Without
    // this the Rothermel kernel spreads with the domain fuel_model_id everywhere.
    if (m_params.rothermel_per_fuel && m_has_spatial_fuel && m_params.uses_model("rothermel")) {
        rebuild_rothermel_table(m_params.moisture_1hr,
                                m_params.moisture_10hr,
                                m_params.moisture_100hr);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Rothermel: per-cell coefficients from the "
                           << "spatial fuel map (" << m_d_rc_table.size() << " codes)\n";
        }
    }

    // Hybrid ROS: allocate the per-cell weight and the secondary scratch field,
    // then fill the weight from the selector. The weight is static in step 1
    // (region and fuel selectors), so it is built once here.
    if (m_params.is_hybrid()) {
        fire_ros_weight  = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_ros_scratch = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        fire_ros_scratch->setVal(0.0_rt);
        if (m_params.hybrid.selector == "structure" && !fire_structure_height) {
            load_structure_height(m_params.hybrid.structure_file);
        }
        init_ros_weight();
        if (m_params.fire_debug) { print_hybrid_weight_summary(); }
    }

    // Non-burnable mask: structures, listed fuel codes, masked firebreaks.
    // Built after every ignition source has been stamped so that an ignition
    // overlapping a structure is pushed back out of it, and fuel in mask cells
    // is removed so no heat or intensity can ever come from them.
    if (m_params.structures.enable && !fire_structure_height) {
        load_structure_height(m_params.structures.file);
    }
    if (m_params.exposure.enable && m_params.structures.enable) {
        build_structure_ids();
    }
    build_nonburnable_mask();
    if (fire_nonburnable) {
        enforce_nonburnable_phi();
        for (MFIter mfi(*fire_fuel_load); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& fuel = fire_fuel_load->array(mfi);
            auto const& at   = fire_arrival_time->array(mfi);
            auto const& m    = fire_nonburnable->const_array(mfi);
            ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (m(i, j, k) > 0.5_rt) {
                    fuel(i, j, k) = 0.0_rt;
                    // An ignition stamped over a footprint is pushed back out:
                    // the FARSITE path rebuilds phi from the arrival time.
                    at(i, j, k) = -1.0_rt;
                }
            });
        }
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Non-burnable mask: "
                           << std::lround(fire_nonburnable->sum(0)) << " cells\n";
        }
        // Per-column open fraction and roof height: weights for the coupling
        // with heat_open_fraction and its diagnostics, and for the wind
        // extraction with structures.wind_open_columns.
        if (m_params.structures.enable) {
            build_open_fraction(erf.Geom(lev));
        }
    }

    // Structure ignition: needs the ids (exposure) and the mask (the wall
    // band), both built above.
    if (m_params.structures.ignition.enable) {
        if (!fire_structure_id || !fire_nonburnable) {
            amrex::Abort("[FIRE] erf.fire.structures.ignition.enable needs the structure ids and the "
                         "non-burnable mask (erf.fire.exposure.enable with erf.fire.structures.enable)");
        }
        m_struct_ign = std::make_unique<StructureIgnition>(m_fg, m_params.structures.ignition,
                                                           m_n_structures, m_params.exposure.ring,
                                                           m_params.fire_debug);
        m_struct_ign->build_footprints(*fire_structure_id);
    }

    // Suppression: its own fields, and a mask for both propagation paths to
    // read even when no static source exists (an all-zero mask changes nothing).
    if (m_params.suppression.enable) {
        m_suppression = std::make_unique<FireSuppression>();
        m_suppression->initialize(m_params.suppression.file, m_params.suppression.poll_interval,
                                  m_params.suppression.log, m_fg.ba, m_fg.dm, m_fg.geom,
                                  m_params.fire_debug);
        if (fire_nonburnable) {
            fire_nonburnable_static = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 3);
            amrex::MultiFab::Copy(*fire_nonburnable_static, *fire_nonburnable, 0, 0, 1, 3);
        } else {
            fire_nonburnable = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 3);
            fire_nonburnable->setVal(0.0_rt);
        }
        if (m_params.propagation_method != "levelset" && m_fp.front_update == farsite_front::legacy) {
            amrex::Print() << "[FIRE] WARNING: suppression with erf.fire.farsite.front_update = legacy "
                           << "is untested; the ros_factor of a drop is applied to fire_ros only\n";
        }
    }

    m_probe_reported.assign(m_params.probes.size() / 2, false);

    amrex::Print() << "[FIRE] FireLayer initialized: C=" << m_fg.C
                   << ", fuel_model=" << fire_params.fuel_model_id
                   << ", grid=" << m_fg.ba.size() << " boxes" << std::endl;

    if (m_params.fire_debug) {
        IntVect max_extent = m_fg.geom.Domain().size();
        RealBox prob_domain = m_fg.geom.ProbDomain();
        Real dx_fire = (prob_domain.hi(0)-prob_domain.lo(0))/max_extent[0];
        Real dy_fire = (prob_domain.hi(1)-prob_domain.lo(1))/max_extent[1];
        amrex::Print() << "[FIRE DEBUG] Fire grid: dx=" << dx_fire << " m, dy=" << dy_fire
                       << " m, extent=" << max_extent[0] << "x" << max_extent[1]
                       << ", grid_ratio=" << m_fg.C << std::endl;
        amrex::Print() << "[FIRE DEBUG] Coupling type: " << fire_params.coupling_type
                       << " (passive=" << fire_params.is_passive()
                       << ", lagged=" << fire_params.is_lagged()
                       << ", synchronous=" << fire_params.is_synchronous() << ")" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Fire-atmosphere feedback multiplier: "
                       << fire_params.fire_atm_feedback << std::endl;
        amrex::Print() << "[FIRE DEBUG] Heat flux alfg (e-folding height): "
                       << fire_params.heat_flux_alfg << " m" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Inject latent heat: " << fire_params.inject_latent << std::endl;
    }
}

// surface_layer is unused: the fire layer reads the atmospheric state it needs
// through the wind, temperature and humidity MultiFabs passed alongside it. The
// parameter stays for interface symmetry with initialize().
void FireLayer::advance(Real time, Real dt, SurfaceLayer& surface_layer,
                        const MultiFab& xvel, const MultiFab& yvel,
                        const MultiFab& z_phys_cc,
                        const MultiFab& T_atm_k0, const MultiFab& RH_atm_k0,
                        const MultiFab* precip_accum_k0)
{
    amrex::ignore_unused(surface_layer);

    m_current_time = time;
    m_dt_atm       = dt;
    ++m_step;

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Starting fire advance step with dt=" << dt << std::endl;

    if (m_params.prescribed_wind) {
        fire_wind_ref->setVal(m_params.prescribed_wind_x, 0, 1, 0);
        fire_wind_ref->setVal(m_params.prescribed_wind_y, 1, 1, 0);
    } else {
        const bool wind_open = m_params.structures.wind_open_columns && m_open_frac_atm && m_roof_h_atm;
        fill_fire_wind_from_interpolation(*fire_wind_ref, *fire_wind_extract_z, xvel, yvel, z_phys_cc,
                                          *fire_surface_z, *fire_col_ground,
                                          m_fg, m_params.wind_ref_ht, m_nz,
                                          m_use_per_fuel_wind_ht ? fire_fuel_model.get() : nullptr,
                                          m_use_per_fuel_wind_ht ? m_d_fcwh.data() : nullptr,
                                          m_use_per_fuel_wind_ht ? FUEL_SLOT_COUNT - 1 : 0,
                                          m_params.wind_interp,
                                          wind_open ? m_open_frac_atm.get() : nullptr,
                                          wind_open ? m_roof_h_atm.get() : nullptr,
                                          m_params.wind_sample_ht, m_params.wind_sample_z0);
    }
    if (m_params.fire_debug) {
        if (m_params.wind_sample_ht > 0.0) {
            amrex::Print() << "[FIRE DEBUG] Wind sampled at " << m_params.wind_sample_ht
                           << " m above ground, log-law factor to " << m_params.wind_ref_ht << " m: "
                           << std::log(m_params.wind_ref_ht / m_params.wind_sample_z0)
                              / std::log(m_params.wind_sample_ht / m_params.wind_sample_z0) << std::endl;
        }
        amrex::Print() << "[FIRE DEBUG] Wind extraction completed. Max reference wind: "
                       << fire_wind_ref->max(0) << " m/s" << std::endl;
        if (m_step > 0)
            amrex::Print() << "[FIRE DEBUG] Wind extraction height range: min="
                           << fire_wind_extract_z->min(0) << " m  max="
                           << fire_wind_extract_z->max(0) << " m" << std::endl;
    }

    MultiFab::Copy(*fire_wind_eff, *fire_wind_ref, 0, 0, 2, 0);

    if (m_params.use_waf) {
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Applying Wind Adjustment Factor (formula: "
                           << m_params.waf_formula << ")" << std::endl;
        apply_waf_to_wind();
    }

    if (m_params.use_terrain_wind) {
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Applying FARSITE terrain wind corrections" << std::endl;
        apply_farsite_terrain_wind(*fire_wind_eff, *fire_slopes, *fire_curvature,
                                   m_params.k_ridge, m_params.k_shelter,
                                   m_params.k_valley, m_params.k_deflect);
    }

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Effective wind computed. Max effective wind: "
                       << fire_wind_eff->max(0) << " m/s" << std::endl;

    if (m_params.fire_debug)
        amrex::Print() << "[FIRE DEBUG] Updating fuel moisture from atmospheric state" << std::endl;
    advance_fuel_moisture(dt, T_atm_k0, RH_atm_k0, precip_accum_k0);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fuel moisture update completed. Max 1-hour moisture: "
                       << fire_fuel_mc->max(0) << std::endl;
        amrex::Print() << "[FIRE DEBUG] Surface temp range: min="
                       << fire_surface_temp->min(0) << " K  max="
                       << fire_surface_temp->max(0) << " K" << std::endl;
        amrex::Print() << "[FIRE DEBUG] Surface RH range:   min="
                       << fire_surface_rh->min(0) << "    max="
                       << fire_surface_rh->max(0) << std::endl;
        if (fire_precip_rate) {
            amrex::Print() << "[FIRE DEBUG] Rain rate (" << m_params.precip_source << "): min="
                           << fire_precip_rate->min(0) << " mm/hr  max="
                           << fire_precip_rate->max(0) << " mm/hr" << std::endl;
        }
    }

    if (m_params.moisture_dynamic) {
        amrex::Long nc = fire_fuel_mc->boxArray().numPts();
        Real avg1   = (nc>0) ? fire_fuel_mc->sum(0)/Real(nc) : m_params.moisture_1hr;
        Real avg10  = (nc>0) ? fire_fuel_mc->sum(1)/Real(nc) : m_params.moisture_10hr;
        Real avg100 = (nc>0) ? fire_fuel_mc->sum(2)/Real(nc) : m_params.moisture_100hr;
        avg1   = amrex::max(0.01_rt, amrex::min(avg1,   0.40_rt));
        avg10  = amrex::max(0.01_rt, amrex::min(avg10,  0.40_rt));
        avg100 = amrex::max(0.01_rt, amrex::min(avg100, 0.40_rt));
        FuelModelParams fp_cur = uniform_fuel_params();
        m_rc = compute_rothermel_params(fp_cur, avg1, avg10, avg100, m_params.use_wind_limit,
                                        m_params.reaction_velocity_formula == "rothermel",
                                        m_params.wrf_bmst_compat);
        if (!m_d_rc_table.empty()) {
            rebuild_rothermel_table(avg1, avg10, avg100);
        }
        if (m_params.fire_debug)
            amrex::Print() << "[FIRE DEBUG] Updated Rothermel coefficients with avg moisture: "
                           << "M_1hr=" << avg1 << " M_10hr=" << avg10
                           << " M_100hr=" << avg100 << " R0=" << m_rc.R0 << " m/s" << std::endl;

        // Phase 13B: Moisture coupling for Balbi and Cheney-Gould models
        if (m_params.moisture_dynamic && m_params.uses_model("balbi")) {
            // Recompute Balbi coefficients with updated moisture
            // A_coeff carries the moisture dependence through B*, so both the
            // default coefficients and the per-fuel table have to be rebuilt.
            FuelModelParams fp_balbi = uniform_fuel_params();
            m_bc_default = compute_balbi_params(fp_balbi, m_params.balbi, avg1);
            if (m_has_spatial_fuel) {
                auto h_balbi = build_fuel_balbi_table(m_params.balbi, avg1, -1.0, m_params.fuel_map.fuel_set_id(), m_params.moisture_live);
                m_d_balbi_table.resize(h_balbi.size());
                amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_balbi.begin(),
                                 h_balbi.end(), m_d_balbi_table.begin());
            }
        }
        if (m_params.moisture_dynamic && m_params.uses_model("cheney_gould")) {
            // Update Cheney-Gould with current 1-hr moisture converted to percent
            FireParams::CheneyGouldParams cgp_cur = m_params.cheney_gould;
            cgp_cur.moisture = avg1 * 100.0_rt;  // fraction → percent
            m_cgc = compute_cheney_gould_params(cgp_cur);
        }
        // Phase 15: Update BEHAVE state when dynamic moisture is enabled.
        if (m_params.moisture_dynamic && m_params.uses_model("behave")) {
            FuelModelParams fp_bh = uniform_behave_fuel_params();
            // Domain-average live moisture from components 3 and 4; the
            // directional level-set path reads this state. "fixed" passes the
            // held value through, as the per-cell kernel does, so a
            // moisture_live outside [0.30, 2.50] reaches both paths. "legacy"
            // keeps the clamp; its update holds the classes in [0.30, 0.40], so
            // the clamp only catches round-off in the average there.
            amrex::Long nc_live = fire_fuel_mc->boxArray().numPts();
            Real avg_lh  = (nc_live > 0) ? fire_fuel_mc->sum(3) / Real(nc_live) : m_params.moisture_live;
            Real avg_lw  = (nc_live > 0) ? fire_fuel_mc->sum(4) / Real(nc_live) : m_params.moisture_live;
            if (m_params.moisture_live_model != "fixed") {
                avg_lh = amrex::max(0.30_rt, amrex::min(avg_lh, 2.50_rt));
                avg_lw = amrex::max(0.30_rt, amrex::min(avg_lw, 2.50_rt));
            }
            m_bs_default = compute_behave_state(fp_bh, avg1, avg10, avg100, avg_lh, avg_lw,
                                                m_params.behave.dynamic_transfer_lo,
                                                m_params.behave.dynamic_transfer_hi,
                                                m_params.use_wind_limit);
        }
    }

    // Observed-perimeter ignition with spin-up: the polygon of
    // erf.fire.ignition.polygon_file is stamped on the step whose window
    // (m_current_time - dt, m_current_time] contains polygon_time. After a
    // restart past that time the window never contains it, so the perimeter
    // restored from the checkpoint is not stamped again.
    if (!m_params.ignition.polygon_file.empty() && !m_polygon_applied && fire_phi
        && m_params.ignition.polygon_time > 0.0
        && m_params.ignition.polygon_time > m_current_time - dt
        && m_params.ignition.polygon_time <= m_current_time) {
        apply_polygon_ignition(m_params.ignition.polygon_time);
        enforce_nonburnable_phi();
        fire_fill_boundary(*fire_phi, m_fg.geom);
    }

    // Suppression actions for this step: lines, drops, hold test and the
    // burnout ignitions, which enter the schedule applied just below.
    if (m_suppression) {
        apply_suppression();
    }

    // Phase 11: Apply any scheduled ignition events due this timestep.
    // Time window: (m_current_time - dt, m_current_time].
    if (m_has_schedule && fire_phi) {
        apply_scheduled_ignitions(*fire_phi, m_fg.geom,
                                  m_ignition_schedule,
                                  m_current_time,
                                  m_current_time - dt);
        // fill_boundary after any phi modification to propagate ghost cells
        //amrex::FillBoundary(*fire_phi, m_fg.geom);
        enforce_nonburnable_phi();
        fire_fill_boundary(*fire_phi, m_fg.geom);
    }

    // Temperature-threshold ignition: cells whose near-surface air is hotter
    // than erf.fire.ignition.threshold_temp ignite. fire_surface_temp is the
    // k = 0 potential temperature mapped onto the fire grid by
    // advance_fuel_moisture() above, so a coupled run sees the fire's own
    // heating and a one-way run only the initial atmosphere. Stateless: the
    // ignited cells are ordinary burning cells from here on, and the arrival
    // time is set by the propagation step like any other new fire.
    if (m_params.ignition.threshold_enable && fire_phi && fire_surface_temp
        && m_current_time >= m_params.ignition.threshold_start_time) {
        const Real r_ign = (m_params.ignition.threshold_radius > 0.0)
                         ? m_params.ignition.threshold_radius
                         : static_cast<Real>(m_fg.geom.CellSize(0));
        const amrex::Long n_ign = apply_threshold_ignition(*fire_phi, *fire_surface_temp,
                                                    fire_nonburnable.get(), m_fg.geom,
                                                    m_params.ignition.threshold_temp, r_ign,
                                                    m_params.propagation_method != "levelset");
        if (n_ign > 0) {
            enforce_nonburnable_phi();
            fire_fill_boundary(*fire_phi, m_fg.geom);
        }
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Threshold ignition: " << n_ign
                           << " cells ignited above " << m_params.ignition.threshold_temp
                           << " K (max surface temp " << fire_surface_temp->max(0) << " K)\n";
        }
    }

    // Hybrid wind selector: the weight follows the effective wind, so it is
    // rebuilt every fire step; the other selectors are static.
    if (m_params.is_hybrid() && m_params.hybrid.selector == "wind") {
        update_wind_weight();
        if (m_params.fire_debug) { print_hybrid_weight_summary(); }
    }

    // Phase 13B: ROS model dispatch.
    // All models write into fire_ros [m/s].
    // Rothermel is the default (ros_model = "rothermel" or unrecognised string).
    // With ros_model = "hybrid" the primary model writes fire_ros, the
    // secondary writes fire_ros_scratch, and the two are blended per cell with
    // fire_ros_weight (0 = primary, 1 = secondary).
    // Optional per-cell fields for the Balbi kernels. Declared here because the
    // level-set path below reuses them when balbi.directional is set.
    BalbiFieldInputs balbi_in;
    if (m_params.uses_model("balbi")) {
        balbi_in.fuel_model   = m_has_spatial_fuel ? fire_fuel_model.get() : nullptr;
        balbi_in.table        = m_d_balbi_table.empty() ? nullptr : m_d_balbi_table.data();
        balbi_in.table_size   = static_cast<int>(m_d_balbi_table.size());
        balbi_in.fp           = uniform_fuel_params();
        balbi_in.fuel_set     = m_params.fuel_map.fuel_set_id();
        balbi_in.M_live       = m_params.moisture_live;
        balbi_in.M_f          = m_params.moisture_1hr;
        balbi_in.surface_temp = fire_surface_temp.get();
        // Per-cell moisture needs the Phase 4 ODE state; without dynamic
        // moisture that field never evolves, so the domain value is used.
        balbi_in.fuel_mc      = m_params.moisture_dynamic ? fire_fuel_mc.get() : nullptr;
        // fire_heat_flux is filled at the end of the step, so this is the
        // previous step's flux: the buoyancy feedback lags the ROS by one
        // fire step.
        balbi_in.heat_flux    = fire_heat_flux.get();

        if (m_params.moisture_dynamic && fire_fuel_mc) {
            amrex::Long nc_mc = fire_fuel_mc->boxArray().numPts();
            if (nc_mc > 0) {
                Real avg_mc = fire_fuel_mc->sum(0) / Real(nc_mc);
                balbi_in.M_f = amrex::max(0.01_rt, amrex::min(avg_mc, 0.40_rt));
            }
        }

    }

    if (m_params.is_hybrid()) {
        fill_ros_for_model(m_params.hybrid.primary,   *fire_ros,         balbi_in);
        fill_ros_for_model(m_params.hybrid.secondary, *fire_ros_scratch, balbi_in);
        // Blend: R = (1 - w) R_primary + w R_secondary. With w = 0 or 1
        // everywhere this reproduces the single-model result exactly.
        blend_ros_fields(*fire_ros, *fire_ros_scratch, *fire_ros_weight);
    } else {
        fill_ros_for_model(m_params.ros_model, *fire_ros, balbi_in);
    }
    zero_ros_in_mask(*fire_ros);
    if (m_params.fire_debug) {
        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Print() << "[FIRE DEBUG] Rate-of-spread computed. Max: " << ros_stats.max_ros
                       << " m/s, Mean: " << ros_stats.mean_ros
                       << " m/s" << std::endl;
    }

    // Phase 10: Fuel boundary blending when spatial fuel map is active.
    if (m_has_spatial_fuel &&
        m_params.fuel_map.blending_fraction > 0.0_rt &&
        fire_fuel_model && fire_ros) {
        apply_fuel_boundary_blending(*fire_ros, *fire_fuel_model,
                                      m_params.fuel_map.blending_fraction);
    }

    fire_fill_boundary(*fire_phi, m_fg.geom);

    // The reach estimate wants the equilibrium rate, before the acceleration
    // ramp scales it down; taken only on the step that prints it.
    const Real ros_max_equilibrium =
        (m_params.edge_reach_check && !m_edge_reach_checked && fire_ros) ? fire_ros->max(0) : Real(0.0);

    // Phase 12: Apply fire acceleration scaling to ROS.
    // Reduces ROS for small fires not yet at quasi-steady-state.
    // Returns immediately when accel.enable = false (zero cost when disabled).
    if (m_params.accel.enable && fire_ros && fire_phi) {
        apply_fire_acceleration(*fire_ros, *fire_phi, m_fg.geom,
                                m_params.accel, dt,
                                fire_accel_state.get(),
                                m_params.fire_debug);
    }

    // Phase 9: crown-fire ROS enhancement.
    // Must run before propagation so the front actually advances at the
    // crowning rate; it also latches crown activation for the heat-flux pass.
    apply_crown_fire_ros();

    int n_substeps = 0;

    // With the front acceleration clock, the level-set paths that rebuild the
    // rate inside every RK stage apply the factor fire_ros already carries.
    std::unique_ptr<amrex::MultiFab> accel_factor;
    if (m_params.accel.enable && m_params.accel.use_temporal
        && m_params.accel.clock == accel_clock::front && fire_accel_state) {
        accel_factor = std::make_unique<amrex::MultiFab>(*fire_accel_state, amrex::make_alias, 2, 1);
    }

    // A retardant drop with 0 < ros_factor < 1 multiplies the rate the front
    // update receives, after the acceleration ramp and the crown enhancement.
    // The directional level-set paths rebuild the rate inside every RK stage,
    // so the factor is folded into the per-stage scale they take.
    if (m_suppression && m_suppression->factor_active()) {
        amrex::MultiFab::Multiply(*fire_ros, *m_suppression->ros_factor(), 0, 0, 1, 0);
        auto combined = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        if (accel_factor) {
            amrex::MultiFab::Copy(*combined, *accel_factor, 0, 0, 1, 0);
        } else {
            combined->setVal(1.0_rt);
        }
        amrex::MultiFab::Multiply(*combined, *m_suppression->ros_factor(), 0, 0, 1, 0);
        accel_factor = std::move(combined);
    }

    if (m_params.fire_debug && m_params.levelset_ellipse && m_params.propagation_method == "levelset") {
        // Shape of the spread ellipse at the strongest midflame wind on the grid.
        amrex::MultiFab umag(m_fg.ba, m_fg.dm, 1, 0);
        for (amrex::MFIter mfi(umag); mfi.isValid(); ++mfi) {
            auto const& w = fire_wind_eff->const_array(mfi);
            auto const& m = umag.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                m(i, j, k) = std::sqrt(w(i, j, k, 0) * w(i, j, k, 0) + w(i, j, k, 1) * w(i, j, k, 1));
            });
        }
        const amrex::Real U = umag.max(0);
        const SpreadEllipse e = spread_ellipse(1.0, U, m_params.levelset_ellipse_lw, m_params.levelset_ellipse_lw_max);
        amrex::Print() << "[FIRE DEBUG] Spread ellipse at max midflame wind " << U << " m/s: LB=" << e.LB
                       << " HB=" << e.HB << " flank/head=" << e.a << " back/head=" << (e.b - e.c) << std::endl;
    }

    if (m_params.propagation_method == "levelset") {
        // --- Level-set path ---
        // CFL-based subcycling (same structure as FARSITE)
        amrex::Real time_remaining = dt;
        int n_ls_substeps = 0;
        while (time_remaining > 1.0e-14) {
            amrex::Real max_ros = fire_ros->max(0);
            amrex::Real dt_ls   = (max_ros > 1.0e-10)
                ? m_params.levelset_cfl * std::min(m_fg.geom.CellSize()[0],
                                                   m_fg.geom.CellSize()[1]) / max_ros
                : time_remaining;
            dt_ls = std::min(dt_ls, time_remaining);
            AMREX_ASSERT(dt_ls > 0.0 && std::isfinite(dt_ls));

            // With directional spread enabled the ROS is rebuilt from the front
            // normal inside every RK stage, so the front gets head, flank and
            // backing spread from the model rather than a single head-fire
            // magnitude. fire_ros still holds the isotropic head ROS and sets
            // the CFL, which stays conservative since the directional rate never
            // exceeds it.
            const bool hybrid_directional =
                m_params.is_hybrid() &&
                (m_params.directional_ros ||
                 (m_params.balbi.directional && m_params.uses_model("balbi")));
            const bool balbi_directional =
                (m_params.ros_model == "balbi") &&
                (m_params.balbi.directional || m_params.directional_ros);
            const bool generic_directional =
                m_params.directional_ros && !m_params.is_hybrid() &&
                (m_params.ros_model != "balbi") && (m_params.ros_model != "prescribed");
            // Wall extrapolation only means something with a mask.
            const bool wall_extrap = m_params.levelset_wall_extrapolate && (fire_nonburnable != nullptr);
            // One-sided derivatives of the level set: the front band of the
            // hybrid scheme is a number of fire cells, phi is in metres here.
            fire_levelset::LevelSetGradient ls_grad;
            ls_grad.scheme = m_params.levelset_grad_scheme;
            const amrex::Real h_fire = std::min(m_fg.geom.CellSize()[0], m_fg.geom.CellSize()[1]);
            ls_grad.band   = m_params.levelset_weno_band_cells * h_fire;
            // Two-value artificial viscosity: the near-front value inside
            // visc_front_cells, blended to eps_visc over visc_transition_cells.
            ls_grad.eps_visc_front = m_params.levelset_eps_visc_front;
            ls_grad.visc_d0 = m_params.levelset_visc_front_cells * h_fire;
            ls_grad.visc_d1 = ls_grad.visc_d0 + m_params.levelset_visc_transition_cells * h_fire;

            if (hybrid_directional) {
                // Both members are rebuilt along the front normal at every RK
                // stage and blended with the same weight the isotropic path uses.
                HybridDirectionalSpec spec;
                auto set_member = [&](const std::string& name, int& model,
                                      DirectionalRosState& state) {
                    if (name == "balbi") {
                        model = HYBRID_MODEL_BALBI;
                    } else {
                        state = make_directional_state(name);
                        model = state.model;
                    }
                };
                set_member(m_params.hybrid.primary,   spec.primary_model,   spec.primary_state);
                set_member(m_params.hybrid.secondary, spec.secondary_model, spec.secondary_state);
                spec.wind_eff   = fire_wind_eff.get();
                spec.balbi_wind = (m_params.balbi.wind_source == 1)
                                ? fire_wind_ref.get() : fire_wind_eff.get();
                spec.bc         = &m_bc_default;
                spec.bp         = &m_params.balbi;
                spec.balbi_in   = &balbi_in;
                spec.weight     = fire_ros_weight.get();
                advect_levelset_hybrid_rk3(*fire_phi, *fire_slopes, m_fg.geom, dt_ls,
                                           m_params.levelset_eps_visc, spec,
                                           fire_nonburnable.get(), wall_extrap,
                                           ls_grad, accel_factor.get());
            } else if (balbi_directional) {
                advect_levelset_balbi_rk3(*fire_phi,
                                          (m_params.balbi.wind_source == 1)
                                              ? *fire_wind_ref : *fire_wind_eff,
                                          *fire_slopes,
                                          m_fg.geom, dt_ls,
                                          m_params.levelset_eps_visc,
                                          m_bc_default, m_params.balbi, balbi_in,
                                          fire_nonburnable.get(), wall_extrap,
                                          ls_grad, accel_factor.get());
            } else if (generic_directional) {
                const DirectionalRosState dir_state = make_directional_state(m_params.ros_model);
                // FBP reads the reference-height wind unless told otherwise;
                // every other model takes the midflame wind.
                const bool fbp_ref = (m_params.ros_model == "fbp") && (m_params.fbp.wind_source == 0);
                advect_levelset_directional_rk3(*fire_phi, fbp_ref ? *fire_wind_ref : *fire_wind_eff,
                                                *fire_slopes, m_fg.geom, dt_ls,
                                                m_params.levelset_eps_visc,
                                                dir_state, fire_nonburnable.get(), wall_extrap,
                                                ls_grad, m_params.directional_shape,
                                                m_params.directional_ellipse_lw,
                                                m_params.directional_ellipse_lw_max,
                                                accel_factor.get(),
                                                m_params.directional_wind_coupling);
            } else if (m_params.levelset_ellipse) {
                // Huygens ellipse: the model's rate is the head rate and the
                // normal speed follows the ellipse set by the midflame wind.
                advect_levelset_ellipse_rk3(*fire_phi, *fire_wind_eff, *fire_ros,
                                            *fire_slopes, m_fg.geom, dt_ls,
                                            m_params.levelset_eps_visc,
                                            m_params.levelset_ellipse_lw, m_params.levelset_ellipse_lw_max,
                                            fire_nonburnable.get(), wall_extrap, ls_grad);
            } else {
                fire_levelset::advect_levelset_weno5z_rk3(*fire_phi, *fire_wind_eff,
                                                *fire_ros, m_fg.geom, dt_ls,
                                                m_params.levelset_eps_visc,
                                                fire_slopes.get(),
                                                fire_nonburnable.get(), wall_extrap,
                                                ls_grad);
            }
            enforce_nonburnable_phi();
            fire_fill_boundary(*fire_phi, m_fg.geom);

            ++m_levelset_subcycle_count;
            if (m_levelset_subcycle_count % m_params.levelset_reinit_every == 0) {
                // Sussman reinitialization is stable for dtau <= dx/2; 0.5*dx sits
                // exactly on that limit and went unstable once enough iterations
                // were taken, so default to half of it.
                amrex::Real dtau = (m_params.levelset_reinit_dtau > 0.0)
                    ? m_params.levelset_reinit_dtau
                    : 0.25 * std::min(m_fg.geom.CellSize()[0], m_fg.geom.CellSize()[1]);
                fire_levelset::reinitialize_phi(*fire_phi, m_fg.geom,
                                      m_params.levelset_reinit_iters, dtau,
                                      m_params.levelset_reinit_band_m,
                                      /*normalized=*/false,
                                      fire_nonburnable.get(), wall_extrap, ls_grad);
                enforce_nonburnable_phi();
                fire_fill_boundary(*fire_phi, m_fg.geom);
            }

            // Update arrival time for newly burned cells (phi < 0)
            {
                const amrex::Real t_now = m_current_time + (dt - time_remaining);
                for (amrex::MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
                    auto p  = fire_phi->const_array(mfi);
                    auto at = fire_arrival_time->array(mfi);
                    amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) noexcept {
                        if (p(iv) < 0.0_rt && at(iv) < 0.0_rt) at(iv) = t_now;
                    });
                }
            }
            time_remaining -= dt_ls;

            // Mirrors the guard the FARSITE path has had all along. Without it a
            // pathological dt_ls turns this into a silent hang, and each pass
            // costs a global fire_ros->max(0) reduction.
            if (++n_ls_substeps > 1000) {
                amrex::Abort("[FIRE] level-set subcycle: too many substeps in one "
                             "atmospheric step; check erf.fire.levelset.cfl");
            }
        }
        // Substeps taken this step, not the run-to-date total that
        // m_levelset_subcycle_count accumulates.
        n_substeps = n_ls_substeps;
    } else {
        // --- Default: FARSITE Lagrangian path (unchanged) ---
        // fire_slopes turns the ROS into a spread rate along the ground: a step
        // of ds up a slope covers ds cos(theta) in map view.
        n_substeps = advance_fire_subcycle(*fire_phi, *fire_spread_vec,
                                          *fire_disp_accum,
                                          *fire_arrival_time,
                                          *fire_wind_eff, *fire_ros,
                                          m_fg.geom, dt,
                                          m_current_time,
                                          m_fp,
                                          fire_slopes.get(), fire_nonburnable.get());
    }

    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Fire front propagation completed with "
                       << n_substeps << " fire subcycles" << std::endl;
        const amrex::Long num_fire_cells = erf_fire_diag::count_burning_cells(*fire_phi);
        amrex::Print() << "[FIRE DEBUG] Number of active fire cells: " << num_fire_cells << std::endl;
    }

    fire_fill_boundary(*fire_phi, m_fg.geom);

    // The fire at the edge of the fire grid: the reach estimate once, the guard
    // band every step (both read the propagated front).
    if (m_params.edge_reach_check && !m_edge_reach_checked) {
        report_edge_reach(ros_max_equilibrium, dt);
    }
    check_edge_guard();

    compute_heat_flux_and_diagnostics(dt);

    // Burning structures: their release joins fire_heat_flux on their
    // footprint cells (and so the atmosphere, through update_atm_flux_buffer),
    // their equivalent intensity is handed to the spotting launch below, and
    // their radiant fraction becomes the incident flux on the cells around them.
    if (m_struct_ign) {
        m_struct_ign->apply_heat_sources(m_current_time, *fire_heat_flux, *fire_structure_id);
    }

    // Exposure accumulators: heat load integrates the flux over the
    // atmospheric step (plus the incident radiant flux of burning structures,
    // which is how a burning house loads its neighbours), the peak keeps the
    // largest intensity seen.
    if (fire_heat_load && fire_heat_flux && fire_fireline_intensity) {
        const amrex::MultiFab* rad = m_struct_ign ? m_struct_ign->rad_flux() : nullptr;
        for (MFIter mfi(*fire_heat_load, TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.tilebox();
            auto const& hl = fire_heat_load->array(mfi);
            auto const& pk = fire_peak_intensity->array(mfi);
            auto const& q  = fire_heat_flux->const_array(mfi);
            auto const& ib = fire_fireline_intensity->const_array(mfi);
            if (rad) {
                auto const& qr = rad->const_array(mfi);
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    hl(i, j, k) += (q(i, j, k) + qr(i, j, k)) * dt;
                    pk(i, j, k)  = amrex::max(pk(i, j, k), ib(i, j, k));
                });
            } else {
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    hl(i, j, k) += q(i, j, k) * dt;
                    pk(i, j, k)  = amrex::max(pk(i, j, k), ib(i, j, k));
                });
            }
        }
    }

    // Phase 8: Albini ember spotting
    // Apply stochastic spotting at the specified interval.
    // fire_wind_eff provides the 2-D wind field for trajectory integration.
    // fire_fuel_load provides residual fuel for re-entry filtering.
    if (m_params.spotting.enable && fire_albini_data && fire_wind_eff) {
        if (m_step % m_params.spotting.spotting_interval == 0) {
            fire_albini_data->setVal(0.0_rt);
            FuelModelParams fp_sp = uniform_fuel_params();
            std::string fuel_sys  = m_params.spotting.fuel_system;
            compute_albini_spotting(
                *fire_phi,
                *fire_albini_data,
                *fire_wind_eff,
                *fire_ros,
                m_fg.geom,
                fp_sp,
                m_params.spotting,
                m_step,
                fire_fuel_load.get(),
                &fuel_sys,
                m_params.fuel_model_id,
                m_params.fire_debug,
                fire_fuel_load.get(),
                m_fuel_load_initial_kg_m2,
                fire_surface_z.get(),
                fire_nonburnable.get(),
                fire_ember_landings.get(),
                /*phi_normalized=*/ m_params.propagation_method != "levelset",
                m_struct_ign ? m_struct_ign->launch_intensity() : nullptr);

            // A spot landing lowers phi below zero directly, outside the
            // substep loop that stamps arrival time, so spot-ignited cells kept
            // the -1 sentinel while burning: an isolated spot ahead of the front
            // had no arrival time at all until the main front overran it. Stamp
            // them here with the end of this step, the same rule the loop uses.
            const amrex::Real t_spot = m_current_time + dt;
            for (amrex::MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
                auto p  = fire_phi->const_array(mfi);
                auto at = fire_arrival_time->array(mfi);
                amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const amrex::IntVect& iv) noexcept {
                    if (p(iv) < 0.0_rt && at(iv) < 0.0_rt) at(iv) = t_spot;
                });
            }
        }
    }

    if (m_params.fire_debug && fire_nonburnable) {
        // Burned cells inside the mask must stay at zero; the regression
        // scripts read this line.
        amrex::MultiFab flag(m_fg.ba, m_fg.dm, 1, 0);
        for (amrex::MFIter mfi(flag); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& f  = flag.array(mfi);
            auto const& m  = fire_nonburnable->const_array(mfi);
            auto const& at = fire_arrival_time->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                f(i, j, k) = (m(i, j, k) > 0.5_rt && at(i, j, k) >= 0.0_rt) ? 1.0_rt : 0.0_rt;
            });
        }
        amrex::Print() << "[FIRE DEBUG] Non-burnable: mask_cells="
                       << std::lround(fire_nonburnable->sum(0))
                       << " burned_inside=" << std::lround(flag.sum(0)) << "\n";
    }

    if (m_params.fire_debug) {
        amrex::Real phi_min  = fire_phi->min(0, 0);   // nghost=0
        amrex::Real phi_max  = fire_phi->max(0, 0);

        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Real ros_max  = ros_stats.max_ros;
        amrex::Real ros_mean = ros_stats.mean_ros;

        amrex::Real Q_max    = fire_heat_flux ? fire_heat_flux->max(0) : 0.0;
        amrex::Real I_B_max  = fire_fireline_intensity ? fire_fireline_intensity->max(0) : 0.0;
        amrex::Real L_max    = fire_flame_length ? fire_flame_length->max(0) : 0.0;
        amrex::Print() << "[FIRE] t=" << m_current_time
                       << "  substeps=" << n_substeps
                       << "  phi_min=" << phi_min
                       << "  phi_max=" << phi_max
                       << "  max_ROS=" << ros_max << " m/s"
                       << "  mean_ROS=" << ros_mean << " m/s"
                       << "  Q_max=" << Q_max << " W/m2"
                       << "  I_B_max=" << I_B_max << " kW/m"
                       << "  L_max=" << L_max << " m\n";
    }

    report_probes();

    // The ignition rule reads the accumulators after this step's update and
    // stamps new ignitions with the end of the step, as the arrival time is.
    if (m_struct_ign) {
        m_struct_ign->update_state(m_current_time + dt, dt, *fire_structure_id, *fire_nonburnable,
                                   *fire_heat_load, *fire_fireline_intensity, *fire_ember_landings);
    }

    if (m_params.exposure.enable && fire_structure_id &&
        (m_step % m_params.exposure.interval == 0)) {
        report_exposure();
    }

    if (m_params.write_fire_stats_csv) {
        static bool csv_header_written = false;
        if (!csv_header_written) {
            write_fire_stats_header(m_params.fire_stats_csv_file);
            csv_header_written = true;
        }
        append_fire_stats(*fire_phi, *fire_arrival_time, m_fg.geom,
                         m_step, m_current_time, m_params.fire_stats_csv_file,
                         fire_ros.get(), fire_heat_flux.get(), fire_albini_data.get(),
                         m_n_edge_cells, m_edge_contact_time, fire_precip_rate.get());
    }
}


void FireLayer::report_edge_reach(Real ros_max, Real dt)
{
    // Wait for a burning cell: a scheduled or threshold ignition may come later.
    std::array<Real, 4> dist;
    const amrex::Long n_burning = erf_fire_edge::burning_distance_to_edges(*fire_phi, m_fg.geom, dist);
    if (n_burning == 0) return;
    m_edge_reach_checked = true;

    // Time left in the run: the earlier of stop_time and max_step at this step's dt.
    Real t_left = Real(-1.0);
    if (m_run_stop_time < std::numeric_limits<Real>::max() && m_run_stop_time > m_current_time) {
        t_left = m_run_stop_time - m_current_time;
    }
    if (m_run_max_step > m_step && dt > Real(0.0)) {
        const Real t_steps = Real(m_run_max_step - m_step) * dt;
        t_left = (t_left < Real(0.0)) ? t_steps : std::min(t_left, t_steps);
    }
    const Real reach = (t_left > Real(0.0)) ? ros_max * t_left : Real(-1.0);

    amrex::Print() << "[FIRE] Reach at ignition (t=" << m_current_time << " s): largest rate of spread "
                   << ros_max << " m/s";
    if (reach >= Real(0.0)) {
        amrex::Print() << ", " << t_left << " s left in the run, reach " << reach << " m";
    } else {
        amrex::Print() << ", run length unknown (no stop_time or max_step)";
    }
    amrex::Print() << "\n";

    bool any_periodic_only = true;
    for (int f = 0; f < 4; ++f) {
        if (dist[f] < Real(0.0)) continue;   // periodic edge
        any_periodic_only = false;
        const Real t_edge = (ros_max > Real(0.0)) ? dist[f] / ros_max : Real(-1.0);
        amrex::Print() << "[FIRE]   " << erf_fire_edge::face_name(f) << " edge: " << dist[f] << " m from the burning region";
        if (t_edge >= Real(0.0)) amrex::Print() << ", reached in about " << t_edge << " s at that rate";
        amrex::Print() << "\n";
        if (reach >= Real(0.0) && reach > dist[f]) {
            amrex::Print() << "[FIRE] WARNING: the fire can reach the " << erf_fire_edge::face_name(f)
                           << " edge of the fire grid before the run ends (" << dist[f] << " m at "
                           << ros_max << " m/s, about " << t_edge << " s). Nothing outside the grid burns:"
                           << (m_lev > 0 ? " enlarge the refinement region of level " + std::to_string(m_lev)
                                         : std::string(" enlarge the domain"))
                           << " or move the ignition; erf.fire.boundary_guard_action"
                           << " decides what happens when the fire gets there.\n";
        }
    }
    if (any_periodic_only) {
        amrex::Print() << "[FIRE]   every edge of the fire grid is periodic; the fire wraps around\n";
    }
}

void FireLayer::check_edge_guard()
{
    m_n_edge_cells = 0;
    if (!m_params.guards_edge()) return;
    if (m_fg.geom.isPeriodic(0) && m_fg.geom.isPeriodic(1)) return;   // nothing to guard

    std::array<amrex::Long, 4> touched;
    m_n_edge_cells = erf_fire_edge::count_burning_in_edge_band(*fire_phi, m_fg.geom,
                                                               m_params.boundary_guard_cells, touched);
    if (m_n_edge_cells == 0 || m_edge_contact_time >= Real(0.0)) return;

    m_edge_contact_time = m_current_time;
    std::string faces;
    for (int f = 0; f < 4; ++f) {
        if (touched[f] > 0) { faces += (faces.empty() ? "" : ", ") + std::string(erf_fire_edge::face_name(f)); }
    }
    if (m_params.guard_aborts()) {
        std::ostringstream msg;
        msg << "[FIRE] The fire reached the boundary guard band (" << m_params.boundary_guard_cells
            << " fire cells) at the " << faces << " edge of the fire grid at t = " << m_current_time
            << " s. Nothing outside the grid burns. Enlarge the "
            << (m_lev > 0 ? "refinement region of level " + std::to_string(m_lev) : std::string("domain"))
            << ", move the ignition, or set"
               " erf.fire.boundary_guard_action = warn to continue with the front clipped at the edge.";
        amrex::Abort(msg.str());
    }
    amrex::Print() << "[FIRE] WARNING: the fire entered the boundary guard band (" << m_params.boundary_guard_cells
                   << " fire cells) at the " << faces << " edge of the fire grid at t = " << m_current_time
                   << " s with " << m_n_edge_cells << " burning cells in it. The front stops at the edge and"
                   << " nothing outside the grid burns; the contact time is in the statistics CSV"
                   << " (erf.fire.boundary_guard_action = warn).\n";
}

void FireLayer::apply_waf_to_wind()
{
    Real waf = 0.4_rt;
    if      (m_params.waf_formula == "andrews")      waf = compute_waf_unsheltered(m_fuel_bed_depth_ft);
    else if (m_params.waf_formula == "behaviorplus")  waf = compute_waf_behaviorplus(m_fuel_bed_depth_ft);
    else amrex::Abort("[FIRE] Unknown waf_formula '" + m_params.waf_formula + "'");
    fire_wind_eff->mult(waf, 0, 2, 0);
}

void FireLayer::advance_fuel_moisture(Real dt_s,
                                      const MultiFab& T_atm_k0,
                                      const MultiFab& RH_atm_k0,
                                      const MultiFab* precip_accum_k0)
{
    // Map atmospheric T and RH to fire grid and store persistently.
    // This operation runs every timestep to ensure
    // fire_surface_temp and fire_surface_rh are available for plotfile output.
    fill_fire_from_atm_k0(*fire_surface_temp, T_atm_k0,  m_fg);
    fill_fire_from_atm_k0(*fire_surface_rh,   RH_atm_k0, m_fg);

    if (!m_params.moisture_dynamic) { return; }
    Real dt_hours = dt_s / 3600.0_rt;
    FuelModelParams fp = uniform_fuel_params();
    Real precip_mm_hr = m_params.precip_rate_mm_hr;

    // Rain per fire cell for this step (erf.fire.precip_source). With
    // "atmosphere" the rate of every atmospheric column is the change of its
    // surface precipitation accumulation over the step, mapped onto the fire
    // grid like T and RH: every fire cell takes the value of its column.
    if (fire_precip_rate) {
        if (m_params.precip_source == "atmosphere") {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(precip_accum_k0 != nullptr && m_precip_accum_prev != nullptr,
                "[FIRE] Internal error: erf.fire.precip_source = atmosphere but no surface precipitation "
                "accumulation reached the fire layer");
            MultiFab rate_2d(m_precip_accum_prev->boxArray(), m_precip_accum_prev->DistributionMap(), 1, 0);
            fire_precip_rate_from_accum(rate_2d, *m_precip_accum_prev, *precip_accum_k0, dt_s, m_precip_prev_valid);
            m_precip_prev_valid = true;
            fill_fire_from_atm_k0(*fire_precip_rate, rate_2d, m_fg);
        } else {
            fire_precip_rate->setVal(precip_mm_hr);
        }
    }

    // Stick model: allocate the shells on first use, every shell at the
    // class's current moisture (a restart restores them afterwards, before
    // the first advance, through get_stick_mc_mut()).
    const bool use_stick = (m_params.moisture_model == "stick");
    const int  n_sh      = m_params.stick.n_shells;
    const bool live_fixed = (m_params.moisture_live_model == "fixed");
    if (use_stick && !fire_stick_mc) {
        allocate_stick_mc();
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Fuel moisture stick model: " << n_sh << " shells, radii "
                           << m_params.stick.radius_cm[0] << " " << m_params.stick.radius_cm[1] << " "
                           << m_params.stick.radius_cm[2] << " cm, diffusivities "
                           << stick_diffusivity(m_params.stick.radius_cm[0], FuelMoistureConst::TAU_1HR) << " "
                           << stick_diffusivity(m_params.stick.radius_cm[1], FuelMoistureConst::TAU_10HR) << " "
                           << stick_diffusivity(m_params.stick.radius_cm[2], FuelMoistureConst::TAU_100HR)
                           << " cm2/h (x " << m_params.stick.diffusivity_scale << ")\n";
        }
    }
    const Real r1 = m_params.stick.radius_cm[0], r10 = m_params.stick.radius_cm[1], r100 = m_params.stick.radius_cm[2];
    const Real rain_ms = m_params.stick.rain_surface_moisture, dscale = m_params.stick.diffusivity_scale;
    // Equilibrium moisture curves (erf.fire.emc_model), for both update models
    const int emc_model = (m_params.emc_model == "van_wagner") ? FuelMoistureEMC::VAN_WAGNER : FuelMoistureEMC::LEGACY;

    for (MFIter mfi(*fire_fuel_mc); mfi.isValid(); ++mfi) {
        Array4<Real> mc   = fire_fuel_mc->array(mfi);
        Array4<Real> mext = fire_mext->array(mfi);
        Array4<const Real> T_f  = fire_surface_temp->const_array(mfi);
        Array4<const Real> RH_f = fire_surface_rh->const_array(mfi);
        Array4<Real> st;
        if (use_stick) { st = fire_stick_mc->array(mfi); }
        // The rain of this cell: the per-cell field when there is one, else the
        // uniform rate (zero without a rain source).
        const bool has_pr = (fire_precip_rate != nullptr);
        Array4<const Real> pr;
        if (has_pr) { pr = fire_precip_rate->const_array(mfi); }
        const Real sw_dead = dead_fuel_weighted_sav(fp.w_d1, fp.w_d10, fp.w_d100, fp.sigma_d1);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv_f) {
            int i = iv_f[0], j = iv_f[1];
            Real T_C = T_f(i,j,0) - 273.15_rt;
            Real RH  = RH_f(i,j,0) * 100.0_rt;
            const Real precip_cell = has_pr ? pr(i,j,0) : precip_mm_hr;
            if (use_stick) {
                // Radial diffusion per class; the class value is the volume average.
                Real M[FuelStickConst::MAX_SHELLS];
                const Real radii[3] = {r1, r10, r100};
                const Real taus[3]  = {FuelMoistureConst::TAU_1HR, FuelMoistureConst::TAU_10HR, FuelMoistureConst::TAU_100HR};
                for (int c = 0; c < 3; ++c) {
                    for (int n = 0; n < n_sh; ++n) { M[n] = st(i, j, 0, c * n_sh + n); }
                    mc(i, j, 0, c) = stick_advance_class(M, n_sh, radii[c], taus[c], RH, T_C, precip_cell,
                                                         rain_ms, dscale, dt_hours, emc_model);
                    for (int n = 0; n < n_sh; ++n) { st(i, j, 0, c * n_sh + n) = M[n]; }
                }
            } else {
            // Existing 3 dead fuel classes (unchanged)
            mc(i,j,0,0) = advance_fuel_moisture_one_class(mc(i,j,0,0),RH,T_C,precip_cell,dt_hours,FuelMoistureConst::TAU_1HR,emc_model);
            mc(i,j,0,1) = advance_fuel_moisture_one_class(mc(i,j,0,1),RH,T_C,precip_cell,dt_hours,FuelMoistureConst::TAU_10HR,emc_model);
            mc(i,j,0,2) = advance_fuel_moisture_one_class(mc(i,j,0,2),RH,T_C,precip_cell,dt_hours,FuelMoistureConst::TAU_100HR,emc_model);
            }
            // Live herbaceous and live woody moisture (components 3 and 4):
            // held where they are with erf.fire.moisture_live_model = "fixed",
            // otherwise the legacy dead-fuel update (see advance_live_fuel_moisture).
            if (mc.nComp() >= 5) {
                mc(i,j,0,3) = advance_live_fuel_moisture(mc(i,j,0,3),RH,T_C,dt_hours,live_fixed,emc_model);
                mc(i,j,0,4) = advance_live_fuel_moisture(mc(i,j,0,4),RH,T_C,dt_hours,live_fixed,emc_model);
            }
            mext(i,j,0) = compute_moisture_of_extinction(sw_dead);
        });
    }
}

void FireLayer::compute_heat_flux_and_diagnostics(Real dt_fire_s)
{
    FuelModelParams fp = uniform_fuel_params();

    Real sigma_agg = dead_fuel_weighted_sav(fp.w_d1, fp.w_d10, fp.w_d100, fp.sigma_d1);

    Real tau_sav = compute_residence_time_s(sigma_agg, fp.rho_p);
    Real tau_sav_floor = (m_params.tau_residence_s > 0.0_rt) ? m_params.tau_residence_s : tau_sav;

    if (m_params.fire_debug) {
        // Masked ROS diagnostics (only for burning cells where phi < 0)
        const auto ros_stats = erf_fire_diag::burning_ros_stats(*fire_ros, *fire_phi);

        amrex::Print() << "[FIRE DEBUG] tau_sav=" << tau_sav
                       << " s  (dx_fire=" << m_fg.geom.CellSize(0)
                       << " m, max_ROS=" << ros_stats.max_ros << " m/s)" << std::endl;
    }

    // Fuel-model burnout time (erf.fire.burnout_model = sfire): the table by
    // fuel code is built once, on first use.
    const bool sfire_burnout = (m_params.burnout_model == "sfire");
    if (sfire_burnout && m_d_burnout_tau.empty()) {
        std::vector<Real> h_tau(FUEL_SLOT_COUNT, 0.0_rt);
        for (int slot = 1; slot < FUEL_SLOT_COUNT; ++slot) { h_tau[slot] = burnout_tau_s(fuel_code_from_slot(slot)); }
        m_d_burnout_tau.resize(h_tau.size());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_tau.begin(), h_tau.end(), m_d_burnout_tau.begin());
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Burnout model sfire: tau by fuel model 1-13 [s]:";
            for (int c = 1; c <= 13; ++c) { amrex::Print() << " " << h_tau[c]; }
            amrex::Print() << " (burn time / " << m_params.burnout_time_to_efold << ")\n";
            amrex::Print() << "[FIRE DEBUG] Burnout model sfire: uniform fuel model " << m_params.fuel_model_id
                           << " tau=" << burnout_tau_s(m_params.fuel_model_id) << " s, w0="
                           << m_fuel_load_initial_kg_m2 << " kg/m2, fresh-cell heat flux w0 h / tau="
                           << m_fuel_load_initial_kg_m2 * fp.heat_content * 2326.0_rt / burnout_tau_s(m_params.fuel_model_id)
                           << " W/m2\n";
        }
    }
    fill_fire_heat_flux(*fire_heat_flux, *fire_fuel_load,
                        *fire_phi, *fire_ros, fp,
                        m_fg.geom.CellSize(0), tau_sav_floor, dt_fire_s,
                        m_has_spatial_fuel ? fire_fuel_model.get() : nullptr,
                        (sfire_burnout && !m_has_spatial_fuel) ? burnout_tau_s(m_params.fuel_model_id) : 0.0_rt,
                        (sfire_burnout && m_has_spatial_fuel) ? m_d_burnout_tau.data() : nullptr,
                        m_params.fuel_map.fuel_set_id(), m_params.moisture_live);
    add_prescribed_heat_flux();

    const Real h_kJ_per_kg = fp.heat_content * 2.326_rt;
    const Real h_fuel_Jkg = fp.heat_content * 2326.0_rt;

    // Phase 9 (part 2): crown heat release and canopy fuel depletion.
    // The ROS enhancement and the crown-activation latch happen in
    // apply_crown_fire_ros(), before the front is propagated; this pass only
    // consumes what it recorded.
    if (m_params.crown.enable && fire_crown_active && fire_crown_load && fire_crown_ros_active) {
        const Real canopy_depth  = m_params.crown.canopy_depth;
        const Real h_crown_Jkg   = m_params.crown.h_crown_BTU_lb * 2326.0_rt;

        for (MFIter mfi(*fire_ros); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const crown_active_arr = fire_crown_active->const_array(mfi);
            auto const crown_ros_arr    = fire_crown_ros_active->const_array(mfi);
            auto crown_load_arr = fire_crown_load->array(mfi);
            auto heat_flux_arr  = fire_heat_flux->array(mfi);

            ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                if (crown_active_arr(i, j, k) < 0.5_rt) { return; }
                if (crown_load_arr(i, j, k) <= 0.0_rt)  { return; }

                const Real R_active  = crown_ros_arr(i, j, k);
                const Real tau_crown = amrex::max(canopy_depth / amrex::max(R_active, 1.0e-6_rt), 1.0_rt);

                heat_flux_arr(i, j, k) += crown_load_arr(i, j, k) * h_crown_Jkg / tau_crown;
                crown_load_arr(i, j, k) *= std::exp(-dt_fire_s / tau_crown);
                crown_load_arr(i, j, k) = amrex::max(crown_load_arr(i, j, k), 0.0_rt);
            });
        }
    }

    fill_fire_diagnostics(*fire_fireline_intensity, *fire_flame_length,
                          *fire_phi, *fire_ros, *fire_fuel_load,
                          m_fuel_load_initial_kg_m2, h_kJ_per_kg);

    const Real dead_load = fp.w_d1 + fp.w_d10 + fp.w_d100;
    Real M_f = m_params.moisture_1hr;
    if (m_params.moisture_dynamic && fire_fuel_mc) {
        const amrex::Long nc = fire_fuel_mc->boxArray().numPts();
        const Real avg1 = (nc > 0) ? fire_fuel_mc->sum(0) / Real(nc) : m_params.moisture_1hr;
        const Real avg10 = (nc > 0) ? fire_fuel_mc->sum(1) / Real(nc) : m_params.moisture_10hr;
        const Real avg100 = (nc > 0) ? fire_fuel_mc->sum(2) / Real(nc) : m_params.moisture_100hr;
        M_f = (dead_load > 1.0e-10_rt)
            ? (fp.w_d1 * avg1 + fp.w_d10 * avg10 + fp.w_d100 * avg100) / dead_load
            : avg1;
    }

    if (fire_flame_temp) {
        fill_flame_temperature(*fire_flame_temp, *fire_fireline_intensity, *fire_phi,
                               m_params.flame_temp_method, h_fuel_Jkg, M_f,
                               m_params.flame_temp_T_amb, m_params.fire_debug);
    }

    if (fire_flame_tilt) {
        fill_flame_tilt_angle(*fire_flame_tilt, *fire_fireline_intensity, *fire_wind_eff,
                              m_params.flame_tilt_rho_air, m_params.flame_tilt_T_amb,
                              m_params.fire_debug);
    }
}

void FireLayer::apply_crown_fire_ros()
{
    if (!m_params.crown.enable || !fire_crown_active || !fire_crown_load
        || !fire_crown_ros_active) {
        return;
    }

    if (fire_crown_fraction_burned) {
        fire_crown_fraction_burned->setVal(0.0_rt);
    }

    const FuelModelParams fp = uniform_fuel_params();
    const Real h_kJ_per_kg   = fp.heat_content * 2.326_rt;

    // Surface-only quantities: the crown criterion is driven by the surface
    // fireline intensity, so both are evaluated from fire_ros before this
    // routine overwrites it with the crown-enhanced value.
    MultiFab surface_ros(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab surface_intensity(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab surface_flame_length(m_fg.ba, m_fg.dm, 1, 0);
    MultiFab::Copy(surface_ros, *fire_ros, 0, 0, 1, 0);

    fill_fire_diagnostics(surface_intensity, surface_flame_length,
                          *fire_phi, surface_ros, *fire_fuel_load,
                          m_fuel_load_initial_kg_m2, h_kJ_per_kg);

    const auto& crown = m_params.crown;
    const Real canopy_base_ht = crown.canopy_base_ht;
    const Real canopy_bulk_den = crown.canopy_bulk_den;
    const Real foliar_moisture = crown.foliar_moisture;
    const Real default_moisture_10hr = m_params.moisture_10hr;
    const Real I_B_crit = van_wagner_critical_intensity(canopy_base_ht, foliar_moisture);
    const Real fixed_u10_ms = (crown.wind_10m_kmh > 0.0_rt)
        ? crown.wind_10m_kmh / 3.6_rt
        : -1.0_rt;
    const int crown_model_id = (crown.ros_model == "rothermel1991") ? 1
        : (crown.ros_model == "van_wagner_proxy") ? 2 : 0;
    const bool use_dynamic_mc = (m_params.moisture_dynamic && fire_fuel_mc != nullptr);
    const bool use_passive_blend = crown.use_passive_blend;
    // With the front acceleration clock the crown rate accelerates with the
    // fire, as the surface rate it is compared with does: Cruz and the Van
    // Wagner proxy take the factor here, Rothermel (1991) scales the surface
    // rate, which already carries it.
    const bool accel_front = m_params.accel.enable && m_params.accel.use_temporal
                          && (m_params.accel.clock == accel_clock::front) && fire_accel_state;

    for (MFIter mfi(*fire_ros); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        auto const phi_arr = fire_phi->const_array(mfi);
        // Cruz's U_10 is an open wind: the reference wind at wind_ref_ht, not the
        // WAF-reduced midflame wind (about 0.4x, which made R_crown 2.3x low).
        auto const wind_arr = fire_wind_ref->const_array(mfi);
        auto const surface_ros_arr = surface_ros.const_array(mfi);
        auto const surface_I_B_arr = surface_intensity.const_array(mfi);
        Array4<const Real> mc_arr;
        if (use_dynamic_mc) {
            mc_arr = fire_fuel_mc->const_array(mfi);
        }
        Array4<const Real> acc_arr;
        if (accel_front) {
            acc_arr = fire_accel_state->const_array(mfi);
        }
        auto ros_arr = fire_ros->array(mfi);
        auto crown_active_arr = fire_crown_active->array(mfi);
        auto crown_ros_arr = fire_crown_ros_active->array(mfi);
        auto crown_frac_arr = fire_crown_fraction_burned->array(mfi);

        ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            if (phi_arr(i, j, k) >= 0.0_rt) {
                ros_arr(i, j, k) = surface_ros_arr(i, j, k);
                crown_active_arr(i, j, k) = 0.0_rt;
                crown_ros_arr(i, j, k) = 0.0_rt;
                crown_frac_arr(i, j, k) = 0.0_rt;
                return;
            }

            const Real R_surface = amrex::max(surface_ros_arr(i, j, k), 0.0_rt);
            const Real I_surface = amrex::max(surface_I_B_arr(i, j, k), 0.0_rt);
            const Real U10_ms = (fixed_u10_ms > 0.0_rt)
                ? fixed_u10_ms
                : std::sqrt(wind_arr(i, j, k, 0) * wind_arr(i, j, k, 0)
                          + wind_arr(i, j, k, 1) * wind_arr(i, j, k, 1));
            const Real moisture_10hr = use_dynamic_mc ? mc_arr(i, j, k, 1) : default_moisture_10hr;

            Real R_active = R_surface;
            if (crown_model_id == 1) {
                R_active = compute_rothermel_1991_crown_ros(R_surface);
            } else if (crown_model_id == 2) {
                R_active = compute_van_wagner_proxy_ros(canopy_bulk_den, foliar_moisture);
            } else {
                R_active = cruz_crown_ros(U10_ms, canopy_bulk_den, moisture_10hr);
            }
            if (accel_front && crown_model_id != 1) {
                R_active *= acc_arr(i, j, k, 2);
            }
            R_active = amrex::max(R_active, R_surface);

            const bool crown_now_active = (crown_active_arr(i, j, k) >= 0.5_rt) || (I_surface >= I_B_crit);
            crown_active_arr(i, j, k) = crown_now_active ? 1.0_rt : 0.0_rt;
            crown_ros_arr(i, j, k) = R_active;

            Real R_total = R_surface;
            if (use_passive_blend) {
                R_total = compute_van_wagner_passive_blend(R_surface, R_active, I_surface, I_B_crit);
            } else if (crown_now_active) {
                R_total = R_active;
            }
            ros_arr(i, j, k) = amrex::max(R_total, 0.0_rt);
            crown_frac_arr(i, j, k) = compute_crown_fraction_burned(ros_arr(i, j, k), R_surface, R_active);
        });
    }
}

void FireLayer::update_atm_flux_buffer(const amrex::Geometry& geom_atm)
{
    if (!m_params.injects_flux() && !m_params.smoke_enable) {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping flux buffer update: coupling_type is passive (injects_flux=false) and smoke_enable=false" << std::endl;
        }
        return;
    }
    if (!fire_heat_flux || !m_Q_atm_prev) { return; }

    if (m_params.fire_debug) {
        const amrex::Real dA = m_fg.geom.CellSize(0) * m_fg.geom.CellSize(1);
        amrex::Print() << "[FIRE DEBUG] Updating atmosphere flux buffer for coupling_type="
                       << m_params.coupling_type << ". Current max heat flux: "
                       << fire_heat_flux->max(0) << " W/m2"
                       << " total_power_W=" << fire_heat_flux->sum(0) * dA
                       << " fuel_kg=" << fire_fuel_load->sum(0) * dA << std::endl;
    }

    // Fuel moisture seen by the flux partition and the latent flux: the
    // deck's 1-h value, or the load-weighted mean of the dynamic moisture
    // map. One value for the whole fire grid, as the latent flux has always
    // used, so the two fluxes see the same moisture.
    FuelModelParams fp = uniform_fuel_params();
    const amrex::Real h_fuel_Jkg = fp.heat_content * 2326.0_rt;
    amrex::Real M_f = m_params.moisture_1hr;
    if (m_params.moisture_dynamic && fire_fuel_mc) {
        amrex::Long nc = fire_fuel_mc->boxArray().numPts();
        amrex::Real avg1   = (nc > 0) ? fire_fuel_mc->sum(0) / amrex::Real(nc) : m_params.moisture_1hr;
        amrex::Real avg10  = (nc > 0) ? fire_fuel_mc->sum(1) / amrex::Real(nc) : m_params.moisture_10hr;
        amrex::Real avg100 = (nc > 0) ? fire_fuel_mc->sum(2) / amrex::Real(nc) : m_params.moisture_100hr;
        amrex::Real dead_load = fp.w_d1 + fp.w_d10 + fp.w_d100;
        M_f = (dead_load > 1e-10_rt)
            ? (fp.w_d1*avg1 + fp.w_d10*avg10 + fp.w_d100*avg100) / dead_load
            : avg1;
    }

    // Sensible flux handed to the atmosphere. With the legacy partition it is
    // the full heat release of the dry fuel on the fire grid; with the CFBM
    // partition (erf.fire.heat_flux_partition = cfbm) it is scaled by the
    // dry-fuel fraction of the wet fuel mass, f_dry = 1 / (1 + M_f), Eq. 4 of
    // Jimenez y Munoz et al. (2026). The fire-grid field itself is left as the
    // unpartitioned release for the diagnostics and the latent flux below.
    const amrex::Real f_dry = m_params.cfbm_partition() ? 1.0_rt / (1.0_rt + M_f) : 1.0_rt;
    m_f_dry_prev = f_dry;   // the smoke emission divides this back out
    if (m_params.cfbm_partition()) {
        amrex::MultiFab Q_sens(fire_heat_flux->boxArray(), fire_heat_flux->DistributionMap(), 1, 0);
        amrex::MultiFab::Copy(Q_sens, *fire_heat_flux, 0, 0, 1, 0);
        Q_sens.mult(f_dry, 0, 1, 0);
        coarsen_fire_flux_to_atm(*m_Q_atm_prev, Q_sens, geom_atm, m_fg);
    } else {
        coarsen_fire_flux_to_atm(*m_Q_atm_prev, *fire_heat_flux,
                                 geom_atm, m_fg);
    }
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Sensible flux to the atmosphere: max " << m_Q_atm_prev->max(0)
                       << " W/m2 (partition=" << m_params.heat_flux_partition
                       << ", f_dry_fuel=" << f_dry << ", fuel moisture=" << M_f << ")" << std::endl;
    }

    if (m_params.inject_latent && m_Q_lat_atm_prev) {
        compute_fire_latent_flux(*fire_latent_flux, *fire_heat_flux, M_f, h_fuel_Jkg);
        coarsen_fire_flux_to_atm(*m_Q_lat_atm_prev, *fire_latent_flux,
                                 geom_atm, m_fg);
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Latent heat flux computed. Max latent flux: "
                           << fire_latent_flux->max(0) << " W/m2, fuel moisture: " << M_f << std::endl;
        }
    } else {
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Skipping latent heat injection (inject_latent=false or no moisture)" << std::endl;
        }
        m_Q_lat_atm_prev->setVal(0.0_rt);
    }

    // The MRF fire thermal excess reads these fluxes in the halo columns around
    // every tile; give them the neighbouring box's value (and the edge column's
    // outside a non-periodic face), so a column sees the same flux whichever box
    // holds it. On a finer level the columns outside the refined region have no
    // fire grid and keep the zero they were allocated with.
    fire_fill_boundary(*m_Q_atm_prev, geom_atm);
    if (m_Q_lat_atm_prev) { fire_fill_boundary(*m_Q_lat_atm_prev, geom_atm); }
}

void FireLayer::apply_polygon_ignition(amrex::Real t_ign)
{
    std::vector<amrex::Real> xs, ys;
    // Vertex file is read on rank 0 only; broadcast to all ranks inside
    // read_polygon_vertices() before returning.
    read_polygon_vertices(m_params.ignition.polygon_file, xs, ys);

    const amrex::Real R   = m_params.ignition.polygon_interior_ros;
    const bool interior   = (R > 0.0);
    // Cells the polygon newly ignites are those that were unburned before it.
    std::unique_ptr<amrex::MultiFab> phi_before;
    if (interior) {
        phi_before = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
        amrex::MultiFab::Copy(*phi_before, *fire_phi, 0, 0, 1, 0);
    }

    if (m_params.ignition.polygon_type == "polyline") {
        init_phi_from_polyline(*fire_phi, m_fg.geom, xs, ys,
                               m_params.ignition.polyline_width);
    } else {
        init_phi_from_polygon(*fire_phi, m_fg.geom, xs, ys);
    }
    fire_fill_boundary(*fire_phi, m_fg.geom);
    m_polygon_applied = true;

    if (interior) {
        // Interior state of a fire that reached the perimeter at t_ign after
        // spreading outward at R: |phi| is the distance inside the perimeter.
        const amrex::Real tau = (m_params.ignition.polygon_interior_tau > 0.0)
                              ? m_params.ignition.polygon_interior_tau
                              : m_fg.geom.CellSize(0) / R;
        amrex::Real fuel_before = fire_fuel_load->sum(0);
        for (amrex::MFIter mfi(*fire_phi); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& p0   = phi_before->const_array(mfi);
            auto const& p    = fire_phi->const_array(mfi);
            auto const& at   = fire_arrival_time->array(mfi);
            auto const& fuel = fire_fuel_load->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (p0(i, j, k) >= 0.0_rt && p(i, j, k) < 0.0_rt) {
                    const amrex::Real d = -p(i, j, k);
                    // Arrival times before the simulation start are clamped to
                    // 0: the field's negative values mean "unburned" everywhere
                    // in the fire layer (the -1 sentinel), so a negative arrival
                    // would be overwritten at the first substep. The fuel keeps
                    // the full burnout, which is what the heat release needs.
                    at(i, j, k)   = amrex::max(t_ign - d / R, 0.0_rt);
                    fuel(i, j, k) = fuel(i, j, k) * std::exp(-d / (R * tau));
                }
            });
        }
        if (m_params.fire_debug) {
            amrex::Print() << "[FIRE DEBUG] Polygon interior state: ros=" << R << " m/s tau=" << tau
                           << " s, fuel load sum " << fuel_before << " -> " << fire_fuel_load->sum(0)
                           << " kg/m2 (cell sum)\n";
        }
    }
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Polygon ignition applied from '"
                       << m_params.ignition.polygon_file << "' ("
                       << m_params.ignition.polygon_type << ") at t=" << t_ign << " s\n";
    }
}

void FireLayer::allocate_stick_mc()
{
    if (fire_stick_mc || !fire_fuel_mc) { return; }
    const int n_sh = m_params.stick.n_shells;
    fire_stick_mc = std::make_unique<MultiFab>(m_fg.ba, m_fg.dm, 3 * n_sh, 0);
    for (MFIter mfi(*fire_stick_mc); mfi.isValid(); ++mfi) {
        Array4<Real> st = fire_stick_mc->array(mfi);
        Array4<const Real> mc0 = fire_fuel_mc->const_array(mfi);
        amrex::ParallelFor(mfi.tilebox(), [=] AMREX_GPU_DEVICE (const IntVect& iv_f) {
            for (int c = 0; c < 3; ++c) {
                for (int n = 0; n < n_sh; ++n) { st(iv_f[0], iv_f[1], 0, c * n_sh + n) = mc0(iv_f[0], iv_f[1], 0, c); }
            }
        });
    }
}

amrex::Real FireLayer::burnout_tau_s(int fuel_code) const
{
    // WRF-SFIRE / CFBM burn times for the Anderson 13 (Jimenez y Munoz et al. 2026, Table 1)
    static const amrex::Real sfire_burn_time_s[14] = {0.0, 7.0, 7.0, 7.0, 180.0, 100.0, 100.0, 100.0,
                                                      900.0, 900.0, 900.0, 900.0, 900.0, 900.0};
    // Scott-Burgan codes take the burn time of their crosswalked Anderson model.
    const int a = sb40_to_anderson(fuel_code);
    const int c = (a >= 1 && a <= 13) ? a : 1;
    const amrex::Real burn_time = m_params.burnout_times_s.empty() ? sfire_burn_time_s[c]
                                                                   : m_params.burnout_times_s[c - 1];
    return burn_time / m_params.burnout_time_to_efold;
}

amrex::Real FireLayer::smoke_heat_per_kg_atm() const
{
    amrex::Real h = m_params.smoke_heat_of_comb;
    if (m_params.smoke_heat_from_fuel) {
        const FuelModelParams fp = uniform_fuel_params();
        h = fp.heat_content * 2326.0_rt;    // BTU/lb to J/kg
    }
    return h * m_f_dry_prev;
}

void FireLayer::apply_fire_coupling_to_cc_source(
    amrex::MultiFab& cc_source,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& z_phys_cc,
    const amrex::Geometry& geom_atm,
    bool has_moisture)
{
    if (!m_params.injects_flux()) { return; }
    if (!m_Q_atm_prev) { return; }
    if (m_params.fire_atm_feedback <= 0.0_rt) { return; }

    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Applying fire coupling to atmosphere (coupling_type="
                       << m_params.coupling_type << ", feedback=" << m_params.fire_atm_feedback
                       << ", max_Q_prev=" << m_Q_atm_prev->max(0) << " W/m2)" << std::endl;
    }

    const amrex::MultiFab* Q_lat_ptr = (m_params.inject_latent && has_moisture)
        ? m_Q_lat_atm_prev.get() : nullptr;

    // Pass fire_debug so apply_fire_tendency_to_cc_source prints tendency diagnostics.
    // With fire_debug=true you will see per-RK-stage output:
    //   [FIRE COUPLING] Q_atm_max=...  alfg=...
    //   [FIRE COUPLING] RhoTheta tendency sum=...  max=...  expected_surface_max=...
    apply_fire_tendency_to_cc_source(
        cc_source,
        *m_Q_atm_prev,
        Q_lat_ptr,
        z_phys_cc,
        S_old,
        geom_atm,
        m_params.heat_flux_alfg,
        m_params.fire_atm_feedback,
        has_moisture,
        m_params.fire_debug,
        m_params.source_mode == "add",
        m_open_frac_atm.get(),
        m_roof_h_atm.get(),
        m_params.heat_open_fraction,
        m_params.heat_tendency_density);
}

void FireLayer::build_open_fraction(const amrex::Geometry& geom_atm)
{
    if (!fire_nonburnable || !fire_structure_height || !m_Q_atm_prev) { return; }
    const amrex::BoxArray& ba2d = m_Q_atm_prev->boxArray();
    const amrex::DistributionMapping& dm = m_Q_atm_prev->DistributionMap();
    // One ghost cell: the bilinear wind stencil reaches the neighbouring column.
    m_open_frac_atm = std::make_unique<amrex::MultiFab>(ba2d, dm, 1, 1);
    m_roof_h_atm    = std::make_unique<amrex::MultiFab>(ba2d, dm, 1, 1);
    m_open_frac_atm->setVal(1.0_rt);
    m_roof_h_atm->setVal(0.0_rt);

    // Only structure cells count toward the roof height, not fuel-code or
    // firebreak cells of the mask, which have no height.
    amrex::MultiFab smask(m_fg.ba, m_fg.dm, 1, 0);
    amrex::MultiFab sh(m_fg.ba, m_fg.dm, 1, 0);
    const amrex::Real hmin = m_params.structures.min_height;
    for (amrex::MFIter mfi(smask); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& m = smask.array(mfi);
        auto const& w = sh.array(mfi);
        auto const& h = fire_structure_height->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const bool s = h(i, j, k) > hmin;
            m(i, j, k) = s ? 1.0_rt : 0.0_rt;
            w(i, j, k) = s ? h(i, j, k) : 0.0_rt;
        });
    }
    // Area averages onto the atmospheric columns: blocked fraction and
    // (height x indicator), whose ratio is the mean roof height.
    amrex::MultiFab blocked(ba2d, dm, 1, 0), hsum(ba2d, dm, 1, 0);
    coarsen_fire_flux_to_atm(blocked, smask, geom_atm, m_fg);
    coarsen_fire_flux_to_atm(hsum,    sh,    geom_atm, m_fg);
    for (amrex::MFIter mfi(*m_open_frac_atm); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& fo = m_open_frac_atm->array(mfi);
        auto const& hr = m_roof_h_atm->array(mfi);
        auto const& b  = blocked.const_array(mfi);
        auto const& hs = hsum.const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real f = amrex::max(0.0_rt, amrex::min(1.0_rt, b(i, j, k)));
            fo(i, j, k) = 1.0_rt - f;
            hr(i, j, k) = (f > 1.0e-6_rt) ? hs(i, j, k) / f : 0.0_rt;
        });
    }
    // The bilinear wind stencil reads the column past a non-periodic face for
    // the outermost fire cells; FillBoundary leaves that ghost at the setVal
    // defaults (open, no roof), so copy the edge column into it instead.
    fire_fill_boundary(*m_open_frac_atm, geom_atm);
    fire_fill_boundary(*m_roof_h_atm, geom_atm);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Open-fraction heat placement: min open fraction "
                       << m_open_frac_atm->min(0) << ", max roof height "
                       << m_roof_h_atm->max(0) << " m\n";
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Rate-of-spread model helpers
// ───────────────────────────────────────────────────────────────────────────

void FireLayer::fill_ros_for_model(const std::string& model,
                                   amrex::MultiFab& out,
                                   const BalbiFieldInputs& balbi_in)
{
    if (model == "balbi") {
        // Balbi normalises the wind by its own vertical velocity scale rather
        // than by a midflame reduction, so balbi.wind_source can bypass the
        // Wind Adjustment Factor and hand it the reference-height wind.
        const amrex::MultiFab& balbi_wind = (m_params.balbi.wind_source == 1)
                                          ? *fire_wind_ref : *fire_wind_eff;
        fill_balbi_ros(out, balbi_wind, *fire_slopes,
                       m_bc_default, m_params.balbi, balbi_in);
    } else if (model == "cheney_gould") {
        fill_cheney_gould_ros(out, *fire_wind_eff, m_cgc);
    } else if (model == "behave") {
        // Phase 15: BEHAVE multi-class Rothermel model
        FuelModelParams fp_behave = uniform_behave_fuel_params();
        fill_behave_ros(out, *fire_wind_eff, *fire_slopes,
                        fp_behave,
                        m_bs_default,
                        m_params.behave.dynamic_transfer_lo,
                        m_params.behave.dynamic_transfer_hi,
                        m_params.moisture_dynamic ? fire_fuel_mc.get() : nullptr,
                        m_params.moisture_dynamic,
                        m_params.use_wind_limit);
    } else if (model == "macarthur") {
        fill_macarthur_ros(out, *fire_wind_eff);
    } else if (model == "fbp") {
        fill_fbp_ros(out, (m_params.fbp.wind_source == 1) ? *fire_wind_eff : *fire_wind_ref, *fire_slopes, m_fbp);
    } else if (model == "prescribed") {
        fill_prescribed_ros(out);
    } else {
        // Default: Rothermel (1972). The per-fuel table is empty unless
        // rothermel_per_fuel is set on a spatial fuel map, in which case the
        // overload falls through to the uniform kernel.
        const bool per_fuel = !m_d_rc_table.empty() && fire_fuel_model;
        compute_ros_field(out, *fire_wind_eff, *fire_slopes, m_rc,
                          per_fuel ? fire_fuel_model.get() : nullptr,
                          per_fuel ? m_d_rc_table.data() : nullptr,
                          per_fuel ? static_cast<int>(m_d_rc_table.size()) : 0,
                          m_params.fuel_map.fuel_set_id());
    }
}

void FireLayer::fill_prescribed_ros(amrex::MultiFab& out) const
{
    const auto& pr = m_params.prescribed;
    const int n_pairs = static_cast<int>(pr.by_fuel.size() / 2);
    if (n_pairs > 0 && !fire_fuel_model) {
        amrex::Abort("erf.fire.prescribed.by_fuel needs a spatial fuel map (erf.fire.fuel_map.file)");
    }
    const bool by_code = (n_pairs > 0);

    amrex::Gpu::DeviceVector<amrex::Real> d_pairs(pr.by_fuel.size());
    if (by_code) {
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, pr.by_fuel.begin(), pr.by_fuel.end(), d_pairs.begin());
    }
    const amrex::Real* pairs = d_pairs.data();

    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();
    const amrex::Real R_default = pr.ros;
    const amrex::Real gx = pr.grad_x, gy = pr.grad_y, x0 = pr.x0, y0 = pr.y0, rmin = pr.min_ros;

    for (amrex::MFIter mfi(out); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& R = out.array(mfi);
        amrex::Array4<const amrex::Real> fuel;
        if (by_code) { fuel = fire_fuel_model->const_array(mfi); }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
            const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
            amrex::Real base = R_default;
            if (by_code) {
                const int code = static_cast<int>(fuel(i, j, k));
                for (int n = 0; n < n_pairs; ++n) {
                    if (static_cast<int>(pairs[2 * n]) == code) { base = pairs[2 * n + 1]; break; }
                }
            }
            R(i, j, k) = prescribed_ros_at(base, gx, gy, x0, y0, rmin, x, y);
        });
    }
    amrex::Gpu::streamSynchronize();
}

void FireLayer::add_prescribed_heat_flux()
{
    const auto& ph = m_params.prescribed_heat;
    if (ph.flux <= 0.0_rt || !fire_heat_flux) { return; }

    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();
    const amrex::Real flux = ph.flux, cx = ph.cx, cy = ph.cy, radius = ph.radius;
    const amrex::Real t0 = ph.start_time, t1 = ph.end_time, t = m_current_time;

    for (amrex::MFIter mfi(*fire_heat_flux); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& q = fire_heat_flux->array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
            const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
            q(i, j, k) += prescribed_heat_flux_at(flux, cx, cy, radius, t0, t1, x, y, t);
        });
    }
    if (m_params.fire_debug) {
        const bool on = prescribed_heat_flux_at(flux, cx, cy, radius, t0, t1, cx, cy, t) > 0.0_rt;
        amrex::Print() << "[FIRE DEBUG] Prescribed heat flux " << (on ? "on" : "off") << " at t=" << t
                       << " s: " << flux << " W/m2 over radius " << radius << " m at ("
                       << cx << ", " << cy << ")\n";
    }
}

void FireLayer::rebuild_rothermel_table(amrex::Real m1, amrex::Real m10, amrex::Real m100)
{
    auto h_table = build_fuel_rothermel_table(m1, m10, m100, m_params.fuel_map.fuel_set_id(), m_params.moisture_live,
                                              m_params.use_wind_limit,
                                              m_params.reaction_velocity_formula == "rothermel",
                                              m_params.wrf_bmst_compat);
    m_d_rc_table.resize(h_table.size());
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_table.begin(), h_table.end(),
                     m_d_rc_table.begin());
}

void FireLayer::init_ros_weight()
{
    const auto& hy = m_params.hybrid;
    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();

    fire_ros_weight->setVal(0.0_rt);

    if (hy.selector == "region") {
        const amrex::Real x_lo = hy.region_x_lo;
        const amrex::Real y_lo = hy.region_y_lo;
        const amrex::Real x_hi = hy.region_x_hi;
        const amrex::Real y_hi = hy.region_y_hi;
        const amrex::Real bw   = hy.blend_width;
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w = fire_ros_weight->array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const amrex::Real x = prob_lo[0] + (i + 0.5_rt) * dx[0];
                const amrex::Real y = prob_lo[1] + (j + 0.5_rt) * dx[1];
                if (bw > 0.0_rt) {
                    // Signed distance into the rectangle (positive inside),
                    // ramped linearly over blend_width centred on the edge.
                    const amrex::Real d = amrex::min(amrex::min(x - x_lo, x_hi - x),
                                                     amrex::min(y - y_lo, y_hi - y));
                    w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, 0.5_rt + d / bw));
                } else {
                    w(i, j, k) = (x >= x_lo && x < x_hi && y >= y_lo && y < y_hi) ? 1.0_rt : 0.0_rt;
                }
            });
        }
    } else if (hy.selector == "fuel") {
        if (!fire_fuel_model) {
            amrex::Abort("erf.fire.hybrid.selector = fuel needs a spatial fuel map "
                         "(erf.fire.fuel_map.file)");
        }
        const int n_codes = static_cast<int>(hy.secondary_fuel_codes.size());
        amrex::Gpu::DeviceVector<int> d_codes(n_codes);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         hy.secondary_fuel_codes.begin(), hy.secondary_fuel_codes.end(),
                         d_codes.begin());
        const int* codes = d_codes.data();
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w    = fire_ros_weight->array(mfi);
            auto const& fuel = fire_fuel_model->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int code = static_cast<int>(fuel(i, j, k));
                amrex::Real wt = 0.0_rt;
                for (int n = 0; n < n_codes; ++n) {
                    if (codes[n] == code) { wt = 1.0_rt; break; }
                }
                w(i, j, k) = wt;
            });
        }
        // The kernels above read d_codes asynchronously; let them finish
        // before the device vector is freed at scope exit.
        amrex::Gpu::streamSynchronize();
    } else if (hy.selector == "structure") {
        // Cells within structure_distance of a structure cell (height above
        // structure_min_height) take the secondary model; blend_width ramps
        // the weight linearly across that distance instead of stepping.
        const amrex::Real D    = hy.structure_distance;
        const amrex::Real bw   = hy.blend_width;
        const amrex::Real hmin = hy.structure_min_height;
        const amrex::Real search = D + 0.5_rt * bw;
        const int ri = static_cast<int>(std::ceil(search / dx[0]));
        const int rj = static_cast<int>(std::ceil(search / dx[1]));
        const int ng = amrex::max(ri, rj) + 1;

        amrex::MultiFab mask(m_fg.ba, m_fg.dm, 1, ng);
        mask.setVal(0.0_rt);
        for (amrex::MFIter mfi(mask); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& m = mask.array(mfi);
            auto const& h = fire_structure_height->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                m(i, j, k) = (h(i, j, k) > hmin) ? 1.0_rt : 0.0_rt;
            });
        }
        fire_fill_boundary(mask, m_fg.geom);

        const amrex::Real dxf = dx[0];
        const amrex::Real dyf = dx[1];
        for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& w = fire_ros_weight->array(mfi);
            auto const& m = mask.const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                amrex::Real dmin = 1.0e30_rt;
                for (int dj = -rj; dj <= rj; ++dj) {
                    for (int di = -ri; di <= ri; ++di) {
                        if (m(i + di, j + dj, k) > 0.5_rt) {
                            const amrex::Real d = std::sqrt(di * dxf * di * dxf + dj * dyf * dj * dyf);
                            dmin = amrex::min(dmin, d);
                        }
                    }
                }
                if (bw > 0.0_rt) {
                    w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, 0.5_rt + (D - dmin) / bw));
                } else {
                    w(i, j, k) = (dmin <= D) ? 1.0_rt : 0.0_rt;
                }
            });
        }
    } else if (hy.selector == "wind") {
        // Rebuilt from the effective wind at every fire step; the wind is not
        // known yet here, so the weight starts at zero.
    } else {
        amrex::Abort("FireLayer::init_ros_weight: unsupported hybrid selector " + hy.selector);
    }
}

void FireLayer::update_wind_weight()
{
    const amrex::Real lo = m_params.hybrid.wind_lo;
    const amrex::Real hi = m_params.hybrid.wind_hi;
    for (amrex::MFIter mfi(*fire_ros_weight); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& w    = fire_ros_weight->array(mfi);
        auto const& wind = fire_wind_eff->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            const amrex::Real u = wind(i, j, k, 0);
            const amrex::Real v = wind(i, j, k, 1);
            const amrex::Real U = std::sqrt(u * u + v * v);
            w(i, j, k) = amrex::max(0.0_rt, amrex::min(1.0_rt, (U - lo) / (hi - lo)));
        });
    }
}

void FireLayer::print_hybrid_weight_summary() const
{
    const amrex::Real w_sum = fire_ros_weight->sum(0);
    // Count cells that take mostly the secondary model (weight > 0.5).
    amrex::MultiFab flag(m_fg.ba, m_fg.dm, 1, 0);
    for (amrex::MFIter mfi(flag); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        auto const& f = flag.array(mfi);
        auto const& w = fire_ros_weight->const_array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            f(i, j, k) = (w(i, j, k) > 0.5_rt) ? 1.0_rt : 0.0_rt;
        });
    }
    const long n_half = std::lround(flag.sum(0));
    amrex::Print() << "[FIRE DEBUG] Hybrid ROS: primary=" << m_params.hybrid.primary
                   << " secondary=" << m_params.hybrid.secondary
                   << " selector=" << m_params.hybrid.selector
                   << " weight_sum=" << w_sum
                   << " secondary_cells=" << n_half << "\n";
}

DirectionalRosState FireLayer::make_directional_state(const std::string& model) const
{
    DirectionalRosState st;
    if (model == "behave") {
        st.model = DIRECTIONAL_ROS_BEHAVE;
        st.bs    = m_bs_default;
    } else if (model == "macarthur") {
        st.model = DIRECTIONAL_ROS_MACARTHUR;
    } else if (model == "fbp") {
        st.model = DIRECTIONAL_ROS_FBP;
        st.fbp   = m_fbp;
    } else if (model == "cheney_gould") {
        st.model       = DIRECTIONAL_ROS_CHENEY_GOULD;
        st.cgc         = m_cgc;
        st.cg_moisture = m_params.cheney_gould.moisture;
        st.cg_curing   = m_params.cheney_gould.curing;
    } else {
        st.model = DIRECTIONAL_ROS_ROTHERMEL;
        st.rc    = m_rc;
    }
    return st;
}

void FireLayer::report_probes()
{
    const int np = static_cast<int>(m_params.probes.size() / 2);
    if (np == 0 || !fire_arrival_time) { return; }

    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();
    const amrex::Box& domain = m_fg.geom.Domain();

    for (int n = 0; n < np; ++n) {
        if (m_probe_reported[n]) { continue; }
        const amrex::Real x = m_params.probes[2 * n];
        const amrex::Real y = m_params.probes[2 * n + 1];
        const int ip = static_cast<int>(std::floor((x - prob_lo[0]) / dx[0]));
        const int jp = static_cast<int>(std::floor((y - prob_lo[1]) / dx[1]));
        const amrex::IntVect cell(ip, jp, 0);
        if (!domain.contains(cell)) {
            amrex::Print() << "[FIRE PROBE] " << n << " x=" << x << " y=" << y
                           << " is outside the fire domain; ignored\n";
            m_probe_reported[n] = true;
            continue;
        }

        // Arrival time at the probe cell from whichever rank owns it; the
        // field is -1 until the cell burns, and the min over other ranks'
        // sentinel keeps that value.
        amrex::ReduceOps<amrex::ReduceOpMin> reduce_op;
        amrex::ReduceData<amrex::Real> reduce_data(reduce_op);
        using ReduceTuple = typename decltype(reduce_data)::Type;
        for (amrex::MFIter mfi(*fire_arrival_time); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            if (!bx.contains(cell)) { continue; }
            auto const& at = fire_arrival_time->const_array(mfi);
            reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {
                    return { (i == ip && j == jp) ? at(i, j, k) : amrex::Real(1.0e30) };
                });
        }
        amrex::Real val = amrex::get<0>(reduce_data.value(reduce_op));
        amrex::ParallelDescriptor::ReduceRealMin(val);

        if (val >= 0.0_rt && val < 1.0e29_rt) {
            amrex::Print() << "[FIRE PROBE] " << n << " x=" << x << " y=" << y
                           << " cell=(" << ip << "," << jp << ")"
                           << " arrival_time_s=" << val << "\n";
            m_probe_reported[n] = true;
        }
    }
}


// ───────────────────────────────────────────────────────────────────────────
// Structure exposure diagnostics
// ───────────────────────────────────────────────────────────────────────────

void FireLayer::build_structure_ids()
{
    const int ring = m_params.exposure.ring;
    fire_structure_id = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, ring);
    fire_structure_id->setVal(0.0_rt);
    m_n_structures = read_structure_ids_nearest_onto_fire_cells(
        *fire_structure_id, m_fg, m_params.structures.file, m_params.structures.min_height);
    if (m_n_structures < 0) {
        amrex::Abort("[FIRE] cannot read structure heightmap '" + m_params.structures.file
                     + "' for the exposure diagnostics");
    }
    fire_fill_boundary(*fire_structure_id, m_fg.geom);
    m_exposure_reported.assign(m_n_structures + 1, 0);

    fire_heat_load      = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_peak_intensity = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_ember_landings = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    fire_heat_load->setVal(0.0_rt);
    fire_peak_intensity->setVal(0.0_rt);
    fire_ember_landings->setVal(0.0_rt);

    amrex::Print() << "[FIRE] Exposure diagnostics: " << m_n_structures
                   << " structures in '" << m_params.structures.file
                   << "', wall band " << ring << " fire cell(s), rows every "
                   << m_params.exposure.interval << " steps to "
                   << m_params.exposure.file << "\n";
}

void FireLayer::report_exposure()
{
    if (!fire_structure_id || m_n_structures <= 0) { return; }
    const int  N    = m_n_structures;
    const int  ring = m_params.exposure.ring;
    const auto prob_lo = m_fg.geom.ProbLoArray();
    const auto dx      = m_fg.geom.CellSizeArray();

    // Per-structure partial sums on this rank, index 0 unused.
    std::vector<amrex::Real> foot(N + 1, 0.0), sx(N + 1, 0.0), sy(N + 1, 0.0),
        hmax(N + 1, 0.0), wall(N + 1, 0.0), wall_burned(N + 1, 0.0),
        tfirst(N + 1, 1.0e30), tlast(N + 1, -1.0), imax(N + 1, 0.0),
        hl_sum(N + 1, 0.0), hl_max(N + 1, 0.0), emb(N + 1, 0.0), rad_max(N + 1, 0.0);
    const amrex::MultiFab* rad = m_struct_ign ? m_struct_ign->rad_flux() : nullptr;

    for (amrex::MFIter mfi(*fire_structure_id); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const ERFHostFabView id_v((*fire_structure_id)[mfi]);
        const ERFHostFabView mk_v((*fire_nonburnable)[mfi]);
        const ERFHostFabView at_v((*fire_arrival_time)[mfi]);
        const ERFHostFabView hl_v((*fire_heat_load)[mfi]);
        const ERFHostFabView pk_v((*fire_peak_intensity)[mfi]);
        const ERFHostFabView em_v((*fire_ember_landings)[mfi]);
        const ERFHostFabView hh_v((*fire_structure_height)[mfi]);
        auto id = id_v.array();  auto mk = mk_v.array();  auto at = at_v.array();
        auto hl = hl_v.array();  auto pk = pk_v.array();  auto em = em_v.array();
        auto hh = hh_v.array();
        std::unique_ptr<ERFHostFabView> rad_v;
        amrex::Array4<const amrex::Real> rf;
        if (rad) { rad_v = std::make_unique<ERFHostFabView>((*rad)[mfi]); rf = rad_v->array(); }
        amrex::LoopOnCpu(bx, [&](int i, int j, int /*k*/) {
            const int sid = static_cast<int>(id(i, j, 0) + 0.5_rt);
            if (sid > 0) {
                foot[sid] += 1.0;
                sx[sid]   += prob_lo[0] + (i + 0.5_rt) * dx[0];
                sy[sid]   += prob_lo[1] + (j + 0.5_rt) * dx[1];
                hmax[sid]  = std::max(hmax[sid], hh(i, j, 0));
                emb[sid]  += em(i, j, 0);
                return;
            }
            if (mk(i, j, 0) > 0.5_rt) { return; }   // street or firebreak: not a wall cell
            // Distinct structures within the ring of this burnable cell; a
            // cell between two footprints counts for both.
            int seen[8]; int ns = 0;
            for (int dj = -ring; dj <= ring; ++dj) {
                for (int di = -ring; di <= ring; ++di) {
                    const int nid = static_cast<int>(id(i + di, j + dj, 0) + 0.5_rt);
                    if (nid <= 0) { continue; }
                    bool dup = false;
                    for (int n = 0; n < ns; ++n) { if (seen[n] == nid) { dup = true; break; } }
                    if (!dup && ns < 8) { seen[ns++] = nid; }
                }
            }
            for (int n = 0; n < ns; ++n) {
                const int s = seen[n];
                wall[s] += 1.0;
                const amrex::Real t = at(i, j, 0);
                if (t >= 0.0_rt) {
                    wall_burned[s] += 1.0;
                    tfirst[s] = std::min(tfirst[s], t);
                    tlast[s]  = std::max(tlast[s], t);
                }
                imax[s]    = std::max(imax[s], pk(i, j, 0));
                hl_sum[s] += hl(i, j, 0);
                hl_max[s]  = std::max(hl_max[s], hl(i, j, 0));
                if (rad) { rad_max[s] = std::max(rad_max[s], rf(i, j, 0)); }
            }
        });
    }
    amrex::ParallelDescriptor::ReduceRealSum(foot.data(),        N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(sx.data(),          N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(sy.data(),          N + 1);
    amrex::ParallelDescriptor::ReduceRealMax(hmax.data(),        N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(wall.data(),        N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(wall_burned.data(), N + 1);
    amrex::ParallelDescriptor::ReduceRealMin(tfirst.data(),      N + 1);
    amrex::ParallelDescriptor::ReduceRealMax(tlast.data(),       N + 1);
    amrex::ParallelDescriptor::ReduceRealMax(imax.data(),        N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(hl_sum.data(),      N + 1);
    amrex::ParallelDescriptor::ReduceRealMax(hl_max.data(),      N + 1);
    amrex::ParallelDescriptor::ReduceRealSum(emb.data(),         N + 1);
    amrex::ParallelDescriptor::ReduceRealMax(rad_max.data(),     N + 1);

    if (!amrex::ParallelDescriptor::IOProcessor()) { return; }

    // Header only when the file does not exist yet (or is empty), so a
    // restarted run appends to the file the first leg wrote.
    bool need_header = true;
    {
        std::ifstream probe(m_params.exposure.file, std::ios::ate);
        if (probe.good() && probe.tellg() > 0) { need_header = false; }
    }
    std::ofstream csv(m_params.exposure.file, std::ios::app);
    if (need_header) {
        csv << "time_s,structure_id,x_m,y_m,height_m,footprint_cells,wall_cells,"
               "wall_burned_frac,t_first_s,t_last_s,residence_s,"
               "peak_intensity_kWm,heat_load_mean_MJm2,heat_load_max_MJm2,embers";
        // The structure ignition columns only when it is on, so the positional
        // readers of the plain exposure CSV keep working.
        if (m_struct_ign) {
            csv << ",state,t_ignition_s,cause,structure_flux_Wm2,incident_flux_max_Wm2";
        }
        csv << "\n";
    }
    int reached = 0;
    amrex::Real i_top = 0.0, hl_top = 0.0;
    for (int s = 1; s <= N; ++s) {
        const bool burned = wall_burned[s] > 0.0;
        const amrex::Real t0  = burned ? tfirst[s] : -1.0;
        const amrex::Real t1  = burned ? tlast[s]  : -1.0;
        const amrex::Real res = burned ? (t1 - t0) : 0.0;
        const amrex::Real xc  = (foot[s] > 0.0) ? sx[s] / foot[s] : 0.0;
        const amrex::Real yc  = (foot[s] > 0.0) ? sy[s] / foot[s] : 0.0;
        const amrex::Real wf  = (wall[s] > 0.0) ? wall_burned[s] / wall[s] : 0.0;
        const amrex::Real hlm = (wall[s] > 0.0) ? hl_sum[s] / wall[s] * 1.0e-6 : 0.0;
        csv << std::setprecision(10)
            << m_current_time << "," << s << "," << xc << "," << yc << "," << hmax[s] << ","
            << static_cast<long>(foot[s]) << "," << static_cast<long>(wall[s]) << ","
            << wf << "," << t0 << "," << t1 << "," << res << ","
            << imax[s] << "," << hlm << "," << hl_max[s] * 1.0e-6 << ","
            << static_cast<long>(emb[s]);
        if (m_struct_ign) {
            csv << "," << erf_structure_ignition::state_name(m_struct_ign->state_of(s))
                << "," << m_struct_ign->ignition_time(s)
                << "," << erf_structure_ignition::cause_name(m_struct_ign->cause_of(s))
                << "," << m_struct_ign->flux_of(s, m_current_time)
                << "," << rad_max[s];
        }
        csv << "\n";
        if (burned) {
            ++reached;
            i_top  = std::max(i_top,  imax[s]);
            hl_top = std::max(static_cast<amrex::Real>(hl_top), static_cast<amrex::Real>(hl_max[s] * 1.0e-6));
            if (!m_exposure_reported[s]) {
                amrex::Print() << "[FIRE EXPOSURE] structure " << s
                               << " x=" << xc << " y=" << yc
                               << " reached_at_s=" << t0 << "\n";
                m_exposure_reported[s] = 1;
            }
        }
    }
    amrex::Print() << "[FIRE EXPOSURE] t=" << m_current_time
                   << " reached=" << reached << "/" << N
                   << " peak_intensity_kWm=" << i_top
                   << " heat_load_max_MJm2=" << hl_top;
    if (m_struct_ign) {
        amrex::Print() << " burning=" << m_struct_ign->n_burning()
                       << " burned_out=" << m_struct_ign->n_burned_out();
    }
    amrex::Print() << "\n";
}

// ───────────────────────────────────────────────────────────────────────────
// Structures and the non-burnable mask
// ───────────────────────────────────────────────────────────────────────────

void FireLayer::load_structure_height(const std::string& file)
{
    fire_structure_height = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 0);
    if (!read_heightmap_nearest_onto_fire_cells(*fire_structure_height, m_fg, file)) {
        amrex::Abort("[FIRE] cannot read structure heightmap '" + file + "'");
    }
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Structure heightmap '" << file
                       << "' max height " << fire_structure_height->max(0) << " m\n";
    }
}

void FireLayer::build_nonburnable_mask()
{
    const bool from_structures = m_params.structures.enable && fire_structure_height;
    // The Scott-Burgan set makes 0 and 91-99 non-burnable without listing them.
    std::vector<int> nb_codes = m_params.fuel_map.nonburnable_codes;
    if (m_params.fuel_map.sb40_active()) {
        for (int c : {0, 91, 92, 93, 94, 95, 96, 97, 98, 99}) { nb_codes.push_back(c); }
    }
    const bool from_codes      = !nb_codes.empty() && fire_fuel_model;
    const bool from_breaks     = m_params.firebreak_use_mask && !m_params.firebreaks.empty();
    if (!from_structures && !from_codes && !from_breaks) { return; }

    // Three ghost cells, the reach of the HJ-WENO5-Z stencil: with
    // levelset.wall_extrapolate the wall stencil tests the mask at every cell
    // of the stencil, and with one ghost that read ran past the fab.
    fire_nonburnable = std::make_unique<amrex::MultiFab>(m_fg.ba, m_fg.dm, 1, 3);
    fire_nonburnable->setVal(0.0_rt);

    if (from_structures) {
        const amrex::Real hmin = m_params.structures.min_height;
        for (amrex::MFIter mfi(*fire_nonburnable); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& m = fire_nonburnable->array(mfi);
            auto const& h = fire_structure_height->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (h(i, j, k) > hmin) { m(i, j, k) = 1.0_rt; }
            });
        }
    }
    if (from_codes) {
        const int n_codes = static_cast<int>(nb_codes.size());
        amrex::Gpu::DeviceVector<int> d_codes(n_codes);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         nb_codes.begin(),
                         nb_codes.end(), d_codes.begin());
        const int* codes = d_codes.data();
        for (amrex::MFIter mfi(*fire_nonburnable); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& m    = fire_nonburnable->array(mfi);
            auto const& fuel = fire_fuel_model->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                const int code = static_cast<int>(fuel(i, j, k));
                for (int n = 0; n < n_codes; ++n) {
                    if (codes[n] == code) { m(i, j, k) = 1.0_rt; break; }
                }
            });
        }
        amrex::Gpu::streamSynchronize();
    }
    if (from_breaks) {
        // Firebreak cells carry the phi sentinel from apply_firebreaks(); anything
        // at or above it is a firebreak.
        const amrex::Real sentinel = 0.5_rt * FIREBREAK_PHI_SENTINEL;
        for (amrex::MFIter mfi(*fire_nonburnable); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.validbox();
            auto const& m = fire_nonburnable->array(mfi);
            auto const& p = fire_phi->const_array(mfi);
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                if (p(i, j, k) >= sentinel) { m(i, j, k) = 1.0_rt; }
            });
        }
    }
    fire_fill_boundary(*fire_nonburnable, m_fg.geom);
}

void FireLayer::apply_suppression()
{
    if (!m_suppression || !fire_phi || !fire_arrival_time || !fire_nonburnable) { return; }
    m_suppression->update(m_current_time, m_dt_atm, m_step, *fire_phi, *fire_arrival_time,
                          fire_flame_length.get(), m_ignition_schedule, m_has_schedule);
    // The mask both propagation paths read: the static sources OR the suppression cells.
    const bool has_static = (fire_nonburnable_static != nullptr);
    for (amrex::MFIter mfi(*fire_nonburnable, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& m  = fire_nonburnable->array(mfi);
        auto const& sm = m_suppression->mask()->const_array(mfi);
        amrex::Array4<const amrex::Real> st;
        if (has_static) { st = fire_nonburnable_static->const_array(mfi); }
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            const bool on = (sm(i, j, k) > 0.5_rt) || (has_static && st(i, j, k) > 0.5_rt);
            m(i, j, k) = on ? 1.0_rt : 0.0_rt;
        });
    }
    fire_fill_boundary(*fire_nonburnable, m_fg.geom);
    enforce_nonburnable_phi();
    fire_fill_boundary(*fire_phi, m_fg.geom);
    if (m_params.fire_debug) {
        amrex::Print() << "[FIRE DEBUG] Suppression: mask_cells=" << std::lround(m_suppression->mask()->sum(0))
                       << " factor_active=" << (m_suppression->factor_active() ? 1 : 0) << "\n";
    }
}

void FireLayer::enforce_nonburnable_phi()
{
    if (!fire_nonburnable || !fire_phi) { return; }
    // Mask cells are only ever clamped at zero, never lifted to a fixed
    // positive level. With a zero rate of spread their level-set value does
    // not evolve during advection, and reinitialisation keeps its sign, so
    // the clamp is a guard against round-off. Holding them at a positive
    // distance instead would break the signed-distance property around the
    // footprint and let its edge act like a front: the masked runs burned
    // more than the unmasked ones when that was tried. On the FARSITE path
    // zero is simply the unburned indicator.
    fire_levelset::hold_phi_in_mask(*fire_phi, fire_nonburnable.get(), 0.0_rt);
}

void FireLayer::zero_ros_in_mask(amrex::MultiFab& ros) const
{
    fire_levelset::zero_ros_in_mask(ros, fire_nonburnable.get());
}
