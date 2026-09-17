#!/usr/bin/env python3
"""OpenBCRidgeCorner: just the wind and the ridge, fire mesh off vs on.

    python3 check_openbccorner.py [nofire fire]

Reads the two mixed-corner regions (xhi,ylo) and (xlo,yhi) -- where an
inflow-Open face meets an outflow-Open face for this case's 34 deg wind --
from every plt3d_{nofire,fire} plotfile, and checks:

1. The fire mesh being active doesn't change the atmosphere-side corner
   behavior (coupling is one-way, fire_atm_feedback=0.0, so it shouldn't).

IMPORTANT, read before trusting a "PASS" here: this checker does NOT assert
that the corner stays bounded. An earlier version of this script did, and
that assertion was WRONG -- measured directly (see README.md), both the
fixed and unpatched builds develop a severe, still-growing runaway in this
deck (3 -> ~390 m/s over 280 s), not just at the two corners but along the
whole ylo boundary row. The Open-BC corner double-write fix reduces the
growth rate by roughly 20-30% relative to unpatched (confirmed by running
both), it does not eliminate it. This script only prints the growth curve
for the record and checks the fire-mesh-independence property above, which
IS expected to hold exactly regardless of that open issue.

See README.md for the full comparison and the still-open question.
"""

import glob, re, sys
import numpy as np
try:
    import yt
    yt.set_log_level(50)
except ImportError:
    sys.exit("needs numpy and yt")

CORNER_SIZE = 500.0      # m, square region at each mixed corner
DOMAIN_HI = 5200.0
MATCH_TOL = 0.05          # m/s, nofire vs fire corner |w| must agree this closely

CORNERS = {
    "(xhi,ylo)": (DOMAIN_HI - CORNER_SIZE, DOMAIN_HI, 0.0, CORNER_SIZE),
    "(xlo,yhi)": (0.0, CORNER_SIZE, DOMAIN_HI - CORNER_SIZE, DOMAIN_HI),
}


def corner_max_w(prefix):
    files = sorted(glob.glob(f"plt3d_{prefix}[0-9]*"),
                    key=lambda f: int(re.search(r"(\d+)$", f).group(1)))
    times, vals = [], {name: [] for name in CORNERS}
    for f in files:
        ds = yt.load(f)
        ad = ds.all_data()
        x = ad["boxlib", "x"].to_value()
        y = ad["boxlib", "y"].to_value()
        w = ad["boxlib", "z_velocity"].to_value()
        times.append(float(ds.current_time.to_value()))
        for name, (xlo, xhi, ylo, yhi) in CORNERS.items():
            m = (x >= xlo) & (x <= xhi) & (y >= ylo) & (y <= yhi)
            vals[name].append(np.max(np.abs(w[m])) if m.any() else float("nan"))
    return np.array(times), {name: np.array(v) for name, v in vals.items()}


def main(variants):
    results = {}
    for v in variants:
        t, vals = corner_max_w(v)
        results[v] = (t, vals)
        print(f"\n=== {v} ===")
        print(f"{'t(s)':>8}" + "".join(f"{name:>16}" for name in CORNERS))
        for i in range(len(t)):
            print(f"{t[i]:>8.1f}" + "".join(f"{vals[name][i]:>16.4f}" for name in CORNERS))

    ok = True

    # Informational only -- NOT a pass/fail bound. See module docstring:
    # both fixed and unpatched builds grow well past this in 300 s.
    for v, (t, vals) in results.items():
        for name, series in vals.items():
            peak = np.nanmax(series)
            print(f"[info] {v} corner {name}: peak |w| = {peak:.4f} m/s (grows throughout the run -- not bounded)")

    # Check: nofire vs fire agreement, if both were run. This IS a real
    # pass/fail check -- one-way coupling means the fire mesh should have
    # zero effect on the atmosphere regardless of the open BC question.
    if "nofire" in results and "fire" in results:
        t_nf, v_nf = results["nofire"]
        t_f, v_f = results["fire"]
        n = min(len(t_nf), len(t_f))
        if not np.allclose(t_nf[:n], t_f[:n], atol=1e-6):
            print("[FAIL] nofire/fire output times do not line up")
            ok = False
        else:
            for name in CORNERS:
                diff = np.max(np.abs(v_nf[name][:n] - v_f[name][:n]))
                status = "PASS" if diff <= MATCH_TOL else "FAIL"
                if diff > MATCH_TOL:
                    ok = False
                print(f"[{status}] corner {name}: max|nofire - fire| = {diff:.4f} m/s (tol {MATCH_TOL})")

    if not ok:
        sys.exit("OpenBCRidgeCorner: FAILED")
    print("\nOpenBCRidgeCorner: all checks passed")


if __name__ == "__main__":
    main(sys.argv[1:] or ["nofire", "fire"])
