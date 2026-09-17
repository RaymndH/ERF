# OpenBCRidgeCorner

Just the wind and the ridge: the smallest atmosphere-only setup built to
investigate the corner instability behind the Open BC corner double-write
fix (`Source/Advection/ERF_AdvectionSrcForMom.cpp` /
`ERF_AdvectionSrcForState.cpp`), with everything fire-specific stripped out
except a with/without fire-mesh comparison. Partial fix stops ERF's advection 
code from writing a corner cell twice at Open-BC corners (once correctly, 
once with corrupted ghost-cell data from the perpendicular direction's stencil) 
by shrinking each direction's box so every corner is written exactly once.

**Read this whole file before trusting a single number out of context.**
The headline result is more nuanced than "the fix solves it": the fix is a
real, verified improvement, but this deck also demonstrates that a deeper,
separate, still-open robustness question remains. See "What we actually
found" below.

## The case

Same idealized N-S ridge, domain and wind as `FireRidgeLineAdvective34deg`
(terrain files and sounding copied unchanged from there): a 5200x5200x2000 m
domain, 100 m atmosphere cells, an idealized ridge (WRF `fire_mountain_type=3`,
300 m tall, crest at x=2500) and a steady 3 m/s wind at 34 deg compass bearing
(`u=1.678, v=2.487`, both positive). `xlo/ylo/xhi/yhi.type = "Open"` on all
four lateral faces (Klemp-Wilhelmson radiative BC, matching WRF's
`open_xs/xe/ys/ye`). This wind direction makes `xlo`/`ylo` genuine inflow and
`xhi`/`yhi` genuine outflow, so the two "mixed" corners -- `(xhi,ylo)` and
`(xlo,yhi)`, each pairing an inflow-Open face with an outflow-Open face -- are
where the fixed double-write bug used to corrupt the advection RHS.

Everything fire-ignition-specific (fuel model, WRF fuel-moisture parity
settings) is removed. `inputs_fire` turns the fire mesh on (one-way coupled,
`coupling_type="lagged"`, `fire_atm_feedback=0.0`, same convention as
`FireAdvectiveWindCoupling`); ERF applies its own mandatory default point
ignition near the origin corner since none is set explicitly here -- harmless
for this test's purpose, which is checking whether an *active* fire mesh
changes the atmosphere-side corner behavior at all, not fire spread.

## What we actually found

**1. The fix is real and measurable.** Built and ran this exact deck two
ways: the fix commit (`f12c3c178ce5`, this branch) and the unpatched parent
(`330f11d46`, `origin/ERF-Fire` tip, branch `baseline-unpatched`). Tracking
domain-wide `max|z_velocity|` at matching output times:

| t(s) | fixed | unpatched |
|---|---|---|
| 0   | 3.0   | 3.0   |
| 40  | 29.0  | 42.1  |
| 80  | 81.1  | 148.4 |
| 120 | 133.9 | 221.5 |
| 160 | 208.1 | 287.1 |
| 200 | 263.7 | 336.5 |
| 240 | 308.8 | 393.7 |
| 280 | 390.2 | 373.1 (baseline dips here, likely noise near saturation) |

Through t=240s the fixed build is consistently 20-30% lower than unpatched
(e.g. 308.8 vs 393.7 m/s at t=240s) -- a real, reproducible improvement,
matching the "roughly halves the growth rate" finding from the original
production `FireRidgeLineAdvective34deg` investigation.

**2. Neither version stays bounded.** Both grow from the 3 m/s background
wind to hundreds of m/s over the 300 s run -- this is NOT the corner
double-write bug alone. An earlier version of this README and
`check_openbccorner.py` incorrectly claimed the fixed build "stays bounded";
that was wrong and has been corrected. The growth is not confined to the two
marked mixed corners either -- `wind_terrain_*_t*.png` (see below) shows it
spreading along the entire `ylo` boundary row and, later, into the domain
interior. This matches the deeper, separate issue already flagged in this
worktree's `CLAUDE.md`: ERF's `"Open"` BC (Klemp & Wilhelmson 1978,
radiative/outflow) is inherently less robust when applied to a genuine
*inflow* face, and the corner fix narrows that gap without closing it.

**3. Fire mesh has zero effect on the atmosphere, exactly as expected.**
`check_openbccorner.py`'s `nofire` vs `fire` comparison at the two mixed
corners matches to `0.0000` m/s at every common output time (one-way
coupling, `fire_atm_feedback=0.0` -- this is the one hard pass/fail check
in the script, and it's a real one).

## Reproducing this

### Build the fixed version
```
git clone https://github.com/RaymndH/ERF.git erf-fixed
cd erf-fixed
git checkout fix/openbc-corner-double-write
git submodule update --init Submodules/AMReX
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release -DERF_ENABLE_MPI=ON -DERF_ENABLE_FIRE=ON
cmake --build build_cpu -j <N>
```

### Build the unpatched ("nofix") baseline
```
git clone https://github.com/RaymndH/ERF.git erf-baseline
cd erf-baseline
git checkout baseline-unpatched     # = origin/ERF-Fire tip, no fix applied
git submodule update --init Submodules/AMReX
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release -DERF_ENABLE_MPI=ON -DERF_ENABLE_FIRE=ON
cmake --build build_cpu -j <N>
```
CMake needs >= 3.20 (a system `cmake` that reports older may need a newer
one from elsewhere on `PATH`, e.g. a conda environment's).

### Run
```
cd Exec/RegTests/OpenBCRidgeCorner
./run_openbccorner.sh /path/to/erf_exec       # runs both inputs_nofire/inputs_fire, then checks
SKIP_RUN=1 ./run_openbccorner.sh x            # checks only, on existing output
```
In a sandboxed environment where MPI's default fabric can't probe cgroup/
network devices, set `FI_PROVIDER=tcp I_MPI_FABRICS=tcp` first (or
`FI_PROVIDER=shm` for single-rank; `tcp` is required for multi-rank in that
kind of sandbox -- see this worktree's `CLAUDE.md` for the full story).

### Plot wind + terrain at any timestep
```
python3 plot_wind_terrain.py nofire         # latest available timestep
python3 plot_wind_terrain.py nofire 30      # t=30s specifically
python3 plot_wind_terrain.py fire 0
```
Wind is plotted at the lowest atmosphere model level, k=0, which is a
uniform 20 m above *local terrain* (AGL) everywhere -- not a fixed absolute
elevation (confirmed directly from the `z_phys` field: 20.0 m at a flat
cell, 284.9 m at the ~265 m ridge crest column). Terrain comes from
`terrain_atm.txt`, the same raster ERF itself loads. The two mixed corners
are marked with dashed red boxes. Arrow length uses a fixed physical scale
(3 m/s <-> 100 m on the plot), the same convention used elsewhere in this
project's wind-diff plots, so arrow length is comparable across timesteps
and between the fixed/unpatched runs.

## Figures

All at the lowest model level (20 m AGL); terrain colored 0-300 m.

- `wind_terrain_nofire_t00000.png`, `_t00030.png`, `_t00300.png` -- fixed
  build, `nofire`, t=0/30/300s. At t=0 the wind is exactly uniform; by t=30s
  distortion is already visible near the `(xlo,yhi)` corner; by t=300s it's
  severe and domain-wide.
- `wind_terrain_fire_t00000.png`, `_t00030.png`, `_t00250.png` --
  fixed build, `fire` (one-way coupled, unignited), same pattern, confirming
  finding 3 above.
- `wind_terrain_baseline_nofire_t00000.png`, `_t00030.png`, `_t00270.png` --
  unpatched baseline, `nofire`. Compare directly against the fixed-build
  `nofire` figures at the same times for the visual counterpart of the
  table above.

## WRF-Fire comparison: is this ERF-specific?

`wrf_comparison/` runs the identical physical setup (same terrain, same
Open BCs on all four sides, same wind, same top damping, same horizontal
advection order) in WRF-Fire instead. **Result: WRF stays stable where ERF
does not** -- WRF's wind speed grows only 3 -> ~6 m/s over the same 300 s
window where ERF's grows past 400 m/s. See `wrf_comparison/README.md` for
the full settings-matching table, build/run instructions, and the wind
extraction script.

**Update, see `../OpenBCEllipsoidBuffer/`: this is NOT an inherent ERF
numerical robustness gap.** The follow-up test there keeps everything
about this setup identical except swapping the ridge (which crosses/runs
near all four boundaries) for a finite ellipsoid kept >600 m clear of
every boundary. With that one change, **ERF is completely stable too**
(3 -> ~4.3 m/s, no growth) -- matching WRF closely. Terrain-to-boundary
distance was the controlled variable and it flipped the outcome
completely. The severe growth documented above was very likely driven by
having terrain intersect/run close to an Open boundary specifically, not
by a general fragility in ERF's Open BC on inflow faces regardless of
geometry. Read `../OpenBCEllipsoidBuffer/README.md` before drawing any
conclusion from this file alone.

## Still open

The corner double-write bug is fixed and verified (this deck, and directly
on the original `FireRidgeLineAdvective34deg` production case) -- that
part stands regardless of the finding above. What's now understood
differently, and what remains genuinely open:

- **Understood, not open anymore**: the severe, domain-wide (not just
  corner-localized) growth in this deck is explained by terrain crossing
  close to the Open boundaries (see `../OpenBCEllipsoidBuffer/`), not by
  an inherent ERF Open-BC weakness. Standard modeling practice already
  avoids this domain layout for exactly this kind of reason.
- **Still open**: whether ERF's `"Open"` BC could still be made more
  robust for the (poor-practice, but sometimes unavoidable, e.g. a real
  fire domain where terrain genuinely reaches the boundary) case where
  terrain does approach an Open boundary -- e.g. per `CLAUDE.md`,
  switching genuine upwind faces to Dirichlet `"Inflow"` and keeping
  `"Open"` only on genuine downwind faces was identified as "not yet
  tried." This is now a robustness/hardening question, not a correctness
  bug blocking normal use.
