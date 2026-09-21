#!/usr/bin/env python3
"""FireReactionVelocityFormula check: for each combination of
erf.fire.reaction_velocity_formula ("albini" | "rothermel") and
erf.fire.wrf_bmst_compat (false | true), does ERF's actual simulated head
ROS (aligned with the prescribed wind) match the closed-form theoretical
Rothermel Rf for that combination?

Pass/fail: every deck's relative error must be below TOL.
"""

import csv, math, sys

U = 4.005
TOL_PCT = 0.05  # tight: this is an algebraic, not front-tracking, comparison
FM1 = dict(w0=0.034, sigma=3500.0, delta=1.0, Mx=0.12, h=8000.0, S_T=0.0555, S_e=0.010, rho_p=32.0)
DECKS = {
    ("albini", False):    "fire_stats_albini.csv",
    ("albini", True):     "fire_stats_albini_bmst.csv",
    ("rothermel", False): "fire_stats_rothermel.csv",
    ("rothermel", True):  "fire_stats_rothermel_bmst.csv",
}


def rothermel_fm1(formula, wrf_bmst_compat, M1, U):
    """Independent Python re-implementation of compute_rothermel_params()
    (Source/Fire/ERF_Rothermel.cpp), parameterized over the same two flags
    this regtest exercises."""
    fp = FM1
    w0 = fp['w0']
    M_f = M1  # FM1 has only w_d1 nonzero, so the weighted dead moisture is M1 exactly
    if wrf_bmst_compat:
        bmst = M_f / (1.0 + M_f)
        w0 = w0 * (1.0 - bmst)
    w_n = w0 * (1 - fp['S_T'])
    rho_b = w0 / fp['delta']
    beta = rho_b / fp['rho_p']
    s = max(fp['sigma'], 100.0)
    beta_op = 3.348 * s ** -0.8189
    s15 = s ** 1.5
    Gmax = s15 / (495.0 + 0.0594 * s15)
    A = (1.0 / (4.774 * s ** 0.1 - 7.27)) if formula == "rothermel" else (133.0 * s ** -0.7913)
    br = beta / beta_op
    Gp = Gmax * br ** A * math.exp(A * (1 - br))
    rm = min(M_f / fp['Mx'], 1.0)
    eta_M = max(0.0, 1 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3)
    eta_s = 0.174 * fp['S_e'] ** -0.19
    I_R = max(Gp * w_n * fp['h'] * eta_M * eta_s, 0.01)
    xi = math.exp((0.792 + 0.681 * math.sqrt(s)) * (beta + 0.1)) / (192.0 + 0.2595 * s)
    eps_h = math.exp(-138.0 / s)
    Q_ig = 250.0 + 1116.0 * M_f
    R0_ftmin = (I_R * xi) / (rho_b * eps_h * Q_ig)
    C = 7.47 * math.exp(-0.133 * s ** 0.55)
    B = 0.02526 * s ** 0.54
    E = 0.715 * math.exp(-3.59e-4 * s)
    U_ftmin = U * 196.85
    phi_w = C * (U_ftmin ** B) * (br ** -E)
    R_ftmin = R0_ftmin * (1.0 + phi_w)
    return R_ftmin * 0.00508  # ft/min -> m/s


def simulated_head_ros(csv_file):
    with open(csv_file) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"'{csv_file}': empty")
    return float(rows[-1]["head_ros_ms"])


def main():
    print(f"U = {U} m/s, FM1 at 6% 1hr moisture\n")
    worst = 0.0
    for (formula, bmst), csv_file in DECKS.items():
        try:
            ros_sim = simulated_head_ros(csv_file)
        except FileNotFoundError:
            sys.exit(f"'{csv_file}' not found -- run the decks first (run_regtest.sh)")
        Rf = rothermel_fm1(formula, bmst, 0.06, U)
        err = abs(ros_sim - Rf) / Rf * 100.0
        worst = max(worst, err)
        print(f"formula={formula:10s} wrf_bmst_compat={bmst!s:5s}  "
              f"ERF head ROS={ros_sim:.6f} m/s  theory Rf={Rf:.6f} m/s  error={err:.4f}%")

    print()
    if worst < TOL_PCT:
        print(f"PASS: worst-case error {worst:.4f}% < {TOL_PCT}%")
    else:
        sys.exit(f"FAIL: worst-case error {worst:.4f}% >= {TOL_PCT}%")


if __name__ == "__main__":
    main()
