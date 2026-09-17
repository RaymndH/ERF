#!/usr/bin/env python3
"""Terrain heatmap + near-surface wind quivers for OpenBCRidgeCorner, at the
latest available timestep of the given variant ("ellipsoid_buffer" or "fire").

    python3 plot_wind_terrain.py [ellipsoid_buffer|fire] [t_seconds]   # default: ellipsoid_buffer, latest

Terrain comes from terrain_atm.txt (the same raster ERF loads, not
re-derived from a plotfile field). Wind is the lowest-model-level
x_velocity/y_velocity from the latest plt3d_<variant>##### plotfile,
strided for legibility. The two mixed corners -- (xhi,ylo) and (xlo,yhi),
where the Open-BC corner double-write bug used to corrupt the flow -- are
marked with dashed boxes.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yt
yt.set_log_level(50)

QUIVER_STRIDE = 3
DOMAIN_HI = 5200.0
CORNER_SIZE = 500.0


def load_terrain(fname):
    with open(fname) as f:
        nx = int(f.readline()); ny = int(f.readline())
        xs = np.array([float(f.readline()) for _ in range(nx)])
        ys = np.array([float(f.readline()) for _ in range(ny)])
        z = np.array([float(f.readline()) for _ in range(nx * ny)]).reshape(nx, ny)
    return xs, ys, z


PLOT_CADENCE_S = 10.0  # erf.plot_int_1=40 steps at fixed_dt=0.25


def latest_plotfile(variant):
    files = sorted(glob.glob("plt3d?????"),
                    key=lambda f: int(re.search(r"(\d{5})$", f).group(1)))
    if not files:
        sys.exit("no plt3d##### plotfiles found -- run inputs first")
    return files[-1]


def plotfile_at_time(variant, t_seconds):
    step = round(t_seconds / PLOT_CADENCE_S) * int(PLOT_CADENCE_S / 0.25)
    fname = f"plt3d{step:05d}"
    if not os.path.isdir(fname):
        sys.exit(f"{fname} not found (requested t={t_seconds}s) -- check available plt3d##### files")
    return fname


def load_wind(fname):
    ds = yt.load(fname)
    g = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    u = np.asarray(g[("boxlib", "x_velocity")])[:, :, 0]
    v = np.asarray(g[("boxlib", "y_velocity")])[:, :, 0]
    nx, ny = u.shape
    dx = float((ds.domain_right_edge[0] - ds.domain_left_edge[0]).d) / nx
    dy = float((ds.domain_right_edge[1] - ds.domain_left_edge[1]).d) / ny
    xc = float(ds.domain_left_edge[0].d) + (np.arange(nx) + 0.5) * dx
    yc = float(ds.domain_left_edge[1].d) + (np.arange(ny) + 0.5) * dy
    t = float(ds.current_time.to_value())
    return xc, yc, u, v, t


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "ellipsoid_buffer"
    pf = plotfile_at_time(variant, float(sys.argv[2])) if len(sys.argv) > 2 else latest_plotfile(variant)
    xc, yc, u, v, t = load_wind(pf)
    txs, tys, tz = load_terrain("terrain_atm.txt")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(txs, tys, tz.T, cmap="terrain", shading="auto", vmin=0, vmax=300)
    fig.colorbar(im, ax=ax, label="Terrain elevation [m]", shrink=0.85)

    s = QUIVER_STRIDE
    # Fixed physical scale (3 m/s <-> 100 m on the plot), matching the
    # convention used elsewhere in this project's wind-diff plots, rather
    # than autoscaled arrows that are hard to compare across timesteps/runs.
    ax.quiver(xc[::s], yc[::s], u[::s, ::s].T, v[::s, ::s].T,
              color="black", angles="xy", scale_units="xy", scale=3.0 / 100.0,
              width=0.003)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"OpenBCEllipsoidBuffer ({variant}): terrain + wind (20 m AGL, lowest model level) at t={t:.0f}s\n{pf}")

    out = f"wind_terrain_{variant}_t{int(round(t)):05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} (t={t:.0f}s, source {pf})")


if __name__ == "__main__":
    main()
