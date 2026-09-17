# WRF-Fire comparison for OpenBCRidgeCorner

Same physical setup as `../inputs_nofire`/`../inputs_fire`, run in WRF-Fire
instead of ERF, to check whether the severe wind runaway found in
`OpenBCRidgeCorner` (see `../README.md`) is an ERF-specific numerical
robustness gap or an unavoidable consequence of the domain design itself
(a ridge crossing close to/through open lateral boundaries).

**Result: WRF stays stable where ERF does not.** Given an *identical*
physical setup -- same terrain, same Open BCs on all four sides, same wind
forcing, same top-damping and horizontal-advection-order settings -- WRF's
wind speed grows only modestly (3 -> ~6 m/s) over the same 300 s window
where ERF's grows from 3 to over 400 m/s. This points at an ERF-specific
issue, not an inherent flaw in having terrain intersect an open boundary.

## Confirmed-matching settings (namelist.input, this directory)

| Setting | WRF | ERF (`../inputs_nofire`) |
|---|---|---|
| Lateral BCs | `open_xs/xe/ys/ye = .true.` (all 4 sides) | `xlo/xhi/ylo/yhi.type = "Open"` (all 4 sides) |
| Top damping | `damp_opt=2`, `zdamp=800`, `dampcoef=0.2` | `erf.rayleigh_damp_W=true`, `erf.rayleigh_zdamp=800.0`, `erf.rayleigh_dampcoef=0.2` |
| Horizontal advection order | `h_mom_adv_order=5`, `h_sca_adv_order=5` | `erf.dycore_horiz_adv_type="Upwind_5th"`, `erf.dryscal_horiz_adv_type="Upwind_5th"` |
| Wind | 3 m/s @ 34 deg bearing (`u=1.67758, v=2.48711` in `input_sounding`) | identical `input_sounding` |
| Terrain | rotated N-S ridge, `s(x,y)` formula, 300 m tall, crest through domain center | same formula, `terrain_atm.txt`/`terrain_fire.txt` generated from it |
| Domain | 5200x5200x2000 m, 52x52 (100 m) atmosphere cells | identical |

This is the exact companion case already used for the production
`FireRidgeLineAdvective34deg` investigation (`WRF4_fire/WRF/test/em_fire/
case_ridge_line_advective_34deg/`) -- not a new setup built for this
comparison. It carries a fire module too (`namelist.fire`), but since
`erf.fire.fire_atm_feedback = 0.0` on the ERF side and WRF's own fire
coupling in that companion case is likewise the standard one-way
convention, the atmosphere fields are unaffected by the fire; only the
atmosphere output is used here.

## Results: max horizontal wind speed at the lowest model level

| t(s) | ERF (`OpenBCRidgeCorner/inputs_nofire`) | WRF |
|---|---|---|
| 0   | 3.0   | 3.000 |
| 60  | ~50 (interpolated from 40/80s samples) | 4.719 |
| 120 | 133.9 | 5.399 |
| 180 | ~230 (interpolated) | 5.966 |
| 240 | 308.8 | 6.092 |
| 300 | ~410 (extrapolated, run stops at 300s) | 6.148 |

Longer WRF run, for context (not compared against ERF, which was not run
this long in this deck):

| t(s) | WRF max\|U\| |
|---|---|
| 600  | 6.778 |
| 900  | 7.747 |
| 1200 | 11.818 |
| 1500 | 22.221 |
| 1800 | 21.226 |

WRF does pick up more over 30 min (plausible terrain-wake/gravity-wave
accumulation, not investigated further here), but stays roughly two orders
of magnitude below ERF's growth over the same timescale.

## Figures

`plot_wind_terrain_wrf.py` is the WRF-side counterpart of
`../plot_wind_terrain.py`, using the same conventions (fixed 3 m/s <-> 100 m
quiver scale, same mixed-corner boxes, same 0-300 m terrain color range):

```
python3 plot_wind_terrain_wrf.py 0      # t=0s
python3 plot_wind_terrain_wrf.py 300    # t=300s
```
Needs `matplotlib` and `netCDF4`. Run from the WRF case directory (where
`wrfout_d01_*` live), not from here.

- `wind_terrain_wrf_t00000.png` -- t=0s, wind exactly uniform (identical
  setup to ERF's `../wind_terrain_nofire_t00000.png`).
- `wind_terrain_wrf_t00060.png` -- t=60s, still essentially uniform.
- `wind_terrain_wrf_t00300.png` -- t=300s, only mild distortion near the
  ridge crest -- compare directly against ERF's `../wind_terrain_nofire_t00300.png`
  at the same simulation time, which by then shows severe, domain-wide
  chaotic flow. This is the visual counterpart of the growth-rate table
  above.

## Build

This uses the WRF-Fire checkout at `WRF4_fire/WRF` on this machine,
`v4.6.1` (commit `d66e442f`, `origin` = `https://github.com/wrf-model/WRF.git`).
Standard WRF build (already built as `ideal.exe`/`wrf.exe` in that checkout);
no source changes were needed for this case.

```
cd WRF4_fire/WRF/test/em_fire
./configure   # if not already configured; ARW core, dmpar or serial as needed
./compile em_fire
```

## Run

```
mkdir case_ridge_line_advective_34deg && cd case_ridge_line_advective_34deg
# copy namelist.input, namelist.fire, input_sounding, input_ht, input_zsf
# from this directory, plus the standard WRF static data files
# (LANDUSE.TBL, GENPARM.TBL, SOILPARM.TBL, VEGPARM.TBL, URBPARM.TBL,
# ETAMPNEW_DATA, RRTMG_LW_DATA, RRTMG_SW_DATA) from WRF's run/ directory --
# those are standard WRF data, not case-specific, and not duplicated here.
ln -s ../ideal.exe ../wrf.exe .
./ideal.exe        # generates wrfinput_d01 from input_sounding/input_ht/input_zsf
./wrf.exe          # or mpirun -np N ./wrf.exe
```

`input_ht`/`input_zsf` were generated by `gen_wrf_rotated_ridge.py` (also in
this directory) -- WRF has no namelist angle parameter for a rotated ridge,
so the terrain is read from a file instead of using `fire_mountain_type`'s
built-in formula (`fire_read_atm_ht=.true.`, `fire_read_fire_ht=.true.` in
`namelist.fire`). See this worktree's `CLAUDE.md` for how the exact
`input_ht`/`input_zsf` dimensions/index convention were determined
empirically.

## Extract the wind-growth numbers

```
cd case_ridge_line_advective_34deg   # wherever wrfout_d01_* landed
python3 /path/to/this/extract_wind_growth.py
```
Needs `numpy` and `netCDF4`. Reads `U`/`V` at the lowest mass level from
every `wrfout_d01_2006-01-01_09:*:00` file, destaggers to mass points, and
prints max/mean horizontal wind speed (`sqrt(u^2+v^2)`) vs time.

Note this is a different quantity from what `../check_openbccorner.py`
tracks (`max|z_velocity|` in the two corner boxes specifically) -- the ERF
numbers in the table above come from a separate, ad hoc read of
`x_velocity`/`y_velocity` over the whole domain (domain-wide max horizontal
speed, matching what this script computes for WRF), not from the corner
checker. Both are legitimate ways to see the same runaway; they just
weren't unified into one script for this comparison.
