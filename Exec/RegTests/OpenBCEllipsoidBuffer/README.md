# OpenBCEllipsoidBuffer

The control case for `../OpenBCRidgeCorner`: same domain, wind, Open BCs on
all four sides, top damping, and horizontal advection order, but the
infinite ridge (which ran through/near the domain boundaries) is replaced
by a **finite elongated ellipsoid kept well clear of every boundary**, to
test whether terrain-boundary proximity -- not a general ERF Open-BC
weakness -- was driving the severe runaway found there.

**Result: yes. With terrain kept away from the boundaries, ERF is
completely stable** -- no runaway at all, closely matching WRF. This
overturns the more pessimistic conclusion drawn in
`../OpenBCRidgeCorner/README.md` (a "deeper, separate, still-open
Open-BC-on-inflow-face robustness question" independent of geometry); see
"Conclusion" below.

## The case

Same 5200x5200x2000 m domain, 100 m atmosphere cells, `xlo/xhi/ylo/yhi.type
= "Open"`, `erf.rayleigh_damp_W`/`dampcoef=0.2`/`zdamp=800`, `Upwind_5th`
horizontal advection, 3 m/s @ 34 deg wind (identical `input_sounding` to
`OpenBCRidgeCorner`) as the ridge case. No fire mesh (`erf.fire.enable =
false`) -- `OpenBCRidgeCorner` already established the fire mesh has zero
effect on the atmosphere for one-way coupling, so this test doesn't repeat
that check.

Terrain: an elongated ellipsoid (`gen_ellipsoid_terrain.py`), semi-major
axis `a=1700 m` along the direction the ridge crest used to run (`phi =
-34 deg`, perpendicular to the wind, same head-on-incidence convention as
the ridge), semi-minor axis `b=1000 m` -- **chosen to match the ridge's
own minor-direction slope steepness exactly** (300 m rise over 1000 m),
so this is a fair like-for-like comparison and not just "a smaller/gentler
feature." Centered at the domain center (2600, 2600); rotated bounding
half-extent is 1969 m in x and 1780 m in y, giving 631 m / 820 m of flat
clearance to the nearest boundary on each axis -- comfortably past the
600 m minimum, confirmed both analytically and by checking the actual
raster (`terrain_atm.txt`'s edge rows/columns are exactly 0).

**A first attempt used `a=1500, b=500`** (steeper: 300 m over just 500 m
in the minor direction, almost double the ridge's steepness) and crashed
WRF itself almost immediately (t=9.5 s, NaN, `w-cfl` up to 6.85) -- a
pure numerics/steepness artifact unrelated to the boundary-distance
question, not a finding. Corrected to the ridge-matched `a=1700, b=1000`
before drawing any conclusion; see git history of `gen_ellipsoid_terrain.py`
if you need the original attempt for reference.

## Results

| t(s) | ERF max\|U\| | WRF max\|U\| |
|---|---|---|
| 0   | 3.00 | 3.00 |
| 40/60  | 4.32 | 4.12 |
| 80/120 | 4.34 | 4.59 |
| 120/180| 4.31 | 4.99 |
| 160/240| 4.30 | 5.09 |
| 200/300| 4.30 | 4.84 |
| 240/360| 4.28 | 4.42 |
| 280 | 4.28 | -- |
| 300 | 4.28 | -- |

ERF plateaus almost immediately (~4.3 m/s from t=40s on) and stays flat
through the full 300 s -- no growth trend at all. WRF likewise stays
bounded (peaks ~5.1 m/s around t=240s, settles back to 4.4 m/s by t=360s).
Both codes handle this terrain essentially identically.

## Conclusion (supersedes part of `OpenBCRidgeCorner/README.md`)

| Case | Terrain vs. boundary | ERF | WRF |
|---|---|---|---|
| Ridge (`OpenBCRidgeCorner`) | crosses/runs near all 4 Open boundaries | **runaway, 3->390 m/s** | stable, 3->~6 m/s |
| Ellipsoid (this test) | >600 m clear of every boundary | stable, 3->4.3 m/s | stable, 3->~5 m/s |

Terrain-to-boundary distance was the controlled variable and it flipped
the outcome completely, with everything else (wind, Open BC on all sides,
damping, advection order, slope steepness) held fixed. This means the
ridge case's catastrophic growth was very likely driven by having terrain
intersect/run close to an Open boundary specifically, not by a general
fragility in ERF's Open BC when applied to a genuine inflow face
regardless of geometry, as `OpenBCRidgeCorner/README.md`'s "Still open"
section had concluded. The Open BC corner double-write bug is still a
real, independently-verified bug worth fixing on its own merits (see the
fix commit and `OpenBCRidgeCorner`) -- it just isn't the dominant
explanation for that case's severe growth.

Standard modeling practice already avoids letting a terrain feature reach
an open/radiative boundary for exactly this kind of reason (see the
Klemp & Wilhelmson / Orlanski radiation-condition literature referenced
elsewhere in this project) -- this test is a controlled, quantitative
confirmation of that practice for ERF's specific Open BC implementation,
rather than a new instability mechanism.

## Reproducing this

Same build as `../OpenBCRidgeCorner` (`fix/openbc-corner-double-write`
branch). Run:
```
cd Exec/RegTests/OpenBCEllipsoidBuffer
FI_PROVIDER=tcp I_MPI_FABRICS=tcp /path/to/erf_exec inputs
python3 plot_wind_terrain.py ellipsoid_buffer 0     # or 30, or omit for latest
```
WRF companion case: `wrf_comparison/` (same structure as
`../OpenBCRidgeCorner/wrf_comparison/`).

## Figures

- `wind_terrain_ellipsoid_buffer_t00000.png`, `_t00030.png`, `_t00300.png`
  -- ERF, t=0/30/300s. Visually near-identical at all three times: uniform
  background flow with mild acceleration/deflection directly over the
  terrain, nothing resembling the ridge case's chaos.
- `wrf_comparison/wind_terrain_wrf_t00000.png`, `_t00060.png`,
  `_t00300.png` -- WRF, same pattern.
