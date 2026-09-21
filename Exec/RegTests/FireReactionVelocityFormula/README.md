# FireReactionVelocityFormula

Does ERF's actual simulated head rate of spread (ROS) match the closed-form
theoretical Rothermel `Rf` for every combination of the two ParmParse
options this PR adds -- `erf.fire.reaction_velocity_formula` ("albini" |
"rothermel") and `erf.fire.wrf_bmst_compat` (false | true)?

## The case

A short (60s), one-way-coupled, single-fuel-model (FM1, short grass) line
fire under a constant prescribed wind (`erf.fire.prescribed_wind`, 4.005
m/s) aligned with the fire's head-spread direction. `fire_atm_feedback=0`
so the fire never reads or feeds back onto the atmosphere -- the wind
driving Rothermel's `phi_w` term is exact and known. `use_wind_limit=false`
so the theoretical calc stays a simple closed form. The run is short
because this checks an algebraic quantity (ROS from a fixed fuel/wind
state), not front-tracking over time -- ERF's own `fire_stats_csv`
`head_ros_ms` column is already constant from the first step onward.

Four decks, one per combination:

| deck | reaction_velocity_formula | wrf_bmst_compat |
|---|---|---|
| `inputs_albini` | "albini" (default) | false (default) |
| `inputs_albini_bmst` | "albini" (default) | true |
| `inputs_rothermel` | "rothermel" | false (default) |
| `inputs_rothermel_bmst` | "rothermel" | true |

## Running

```
./run_regtest.sh /path/to/erf_exec
```

(needs `FI_PROVIDER=tcp` exported first if MPICH's default OFI provider
fails to init in your environment -- `run_regtest.sh` does this for you).
`SKIP_RUN=1 ./run_regtest.sh x` re-checks already-run decks.

## Check

`check_regtest.py` reads each deck's `head_ros_ms` from its
`fire_stats_*.csv`, and compares it against an independent Python
re-implementation of `compute_rothermel_params()`
(`Source/Fire/ERF_Rothermel.cpp`), parameterized over the same two flags.
Pass/fail: every deck's relative error against its own theoretical `Rf`
must be under 0.05%.
