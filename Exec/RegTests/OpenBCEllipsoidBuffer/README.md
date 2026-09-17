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

Terrain: an elongated ellipsoid (`gen_ellipsoid_terrain.py`), major axis
along the direction the ridge crest used to run (`phi = -34 deg`,
perpendicular to the wind, same head-on-incidence convention as the
ridge), semi-minor axis `b=1000 m` -- **chosen to match the ridge's own
minor-direction slope steepness exactly** (300 m rise over 1000 m), so
this is a fair like-for-like comparison and not just "a smaller/gentler
feature." Centered at the domain center (2600, 2600).

Two semi-major-axis values have been run, both reported below:

| `a` | rotated half-extent (x, y) | clearance (x, y) |
|---|---|---|
| 1700 m | 1516 m, 1261 m | **1084 m, 1339 m** |
| 2500 m | 2147 m, 1625 m | **453 m, 975 m** |

using the correct rotated-**ellipse** extremum formula, `x_half =
sqrt((a*cos(34deg))^2 + (b*sin(34deg))^2)` (and the analogous swapped
form for `y_half`) -- confirmed both analytically and against the actual
raster (`terrain_atm.txt`'s edge rows/columns are exactly 0 in both
cases). An earlier version of this file used the wrong formula (the
Minkowski-sum bounding box of a *rotated rectangle*, `a*cos+b*sin`,
which overestimates for an ellipse) and reported 631 m/820 m for `a=1700`
-- the real clearance there is significantly larger, ~1084 m/1339 m.
`a=2500`'s x-clearance (453 m) was knowingly accepted just under the
original 600 m minimum specifically to test a longer ellipsoid; see
Results below for whether that mattered.

**A first attempt used `a=1500, b=500`** (steeper: 300 m over just 500 m
in the minor direction, almost double the ridge's steepness) and crashed
WRF itself almost immediately (t=9.5 s, NaN, `w-cfl` up to 6.85) -- a
pure numerics/steepness artifact unrelated to the boundary-distance
question, not a finding. Corrected to the ridge-matched `a=1700, b=1000`
before drawing any conclusion; see git history of `gen_ellipsoid_terrain.py`
if you need the original attempt for reference.

## Results

### a=1700 (clearance ~1084 m / 1339 m)

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

### a=2500 (clearance ~453 m / 975 m -- deliberately under the 600 m target)

| t(s) | ERF max\|U\| | WRF max\|U\| |
|---|---|---|
| 0   | 3.00 | 3.00 |
| 40/60  | 4.49 | 4.27 |
| 80/120 | 4.50 | 4.76 |
| 120/180| 4.44 | 5.16 |
| 160/240| 4.44 | 5.26 |
| 200/300| 4.30 | 4.97 |
| 240/360| 4.28 | 4.55 |
| 280 | 4.28 | -- |
| 300 | 4.41 | -- |

**Still completely stable, both codes, even with the narrower ~453 m
x-clearance.** The `a=2500` curve is essentially the same shape and
magnitude as `a=1700`'s -- no sign that the reduced clearance (still
under the originally-requested 600 m minimum) started reproducing any of
the ridge case's instability. This suggests the relevant threshold, if
one exists, is well below ~450 m for this wind/terrain-height
combination -- or that clearance distance isn't actually the binding
parameter once terrain simply doesn't *touch* the boundary, and the
ridge case's failure mode specifically needed terrain reaching all the
way to (or past) the boundary itself, not just being "close." Not
further narrowed down in this session.

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
  -- ERF, `a=1700`, t=0/30/300s. Visually near-identical at all three
  times: uniform background flow with mild acceleration/deflection
  directly over the terrain, nothing resembling the ridge case's chaos.
- `wrf_comparison/wind_terrain_wrf_t00000.png`, `_t00060.png`,
  `_t00300.png` -- WRF, `a=1700`, same pattern.
- `wind_terrain_ellipsoid_a2500_t00000.png`, `_t00030.png`, `_t00300.png`
  -- ERF, `a=2500` (narrower ~453 m x-clearance). Same visual story as
  `a=1700` -- no sign of instability despite the reduced buffer.
- `wrf_comparison/wind_terrain_wrf_a2500_t00000.png`, `_t00060.png`,
  `_t00300.png` -- WRF, `a=2500`, same pattern.
