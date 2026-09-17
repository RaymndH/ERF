#!/usr/bin/env python3
"""Terrain heatmap + near-surface wind quivers for the WRF-Fire companion
run, at a given time -- the WRF-side counterpart of ../plot_wind_terrain.py,
using the same conventions (fixed 3 m/s <-> 100 m quiver scale, same mixed-
corner boxes, same 0-300 m terrain color range) so the two are visually
comparable side by side.

    python3 plot_wind_terrain_wrf.py [t_seconds]   # default: t=0

Wind is U/V at the lowest mass level (destaggered), the WRF analog of
ERF's lowest model level -- WRF's namelist stretches the vertical grid
(stretch_grd/z_grd_scale) so this is NOT the same physical height as ERF's
uniform-grid 20 m AGL; see ../README.md's height-matching discussion for
why that distinction mattered elsewhere in this project. This plot is
concerned with terrain + wind pattern, not a height-matched quantitative
diff, so the lowest available level is used as-is.
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc

DOMAIN_HI = 5200.0
CORNER_SIZE = 500.0
QUIVER_STRIDE = 3


def wrf_file_at_time(t_seconds):
    files = sorted(glob.glob("wrfout_d01_2006-01-01_09:*:00"))
    def tsec(f):
        m = re.search(r"09:(\d{2}):(\d{2})$", f)
        return int(m.group(1)) * 60 + int(m.group(2))
    for f in files:
        if tsec(f) == t_seconds:
            return f, t_seconds
    sys.exit(f"no wrfout for t={t_seconds}s -- available times: "
              f"{[tsec(f) for f in files]}")


def main():
    t = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    fname, t = wrf_file_at_time(t)
    ds = nc.Dataset(fname)

    hgt = np.array(ds.variables["HGT"][0])          # (y,x), mass points
    u = np.array(ds.variables["U"][0, 0, :, :])      # (y, x+1), x-staggered
    v = np.array(ds.variables["V"][0, 0, :, :])      # (y+1, x), y-staggered
    ud = 0.5 * (u[:, :-1] + u[:, 1:])
    vd = 0.5 * (v[:-1, :] + v[1:, :])

    ny, nx = hgt.shape
    xc = (np.arange(nx) + 0.5) * 100.0
    yc = (np.arange(ny) + 0.5) * 100.0

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.pcolormesh(xc, yc, hgt, cmap="terrain", shading="auto", vmin=0, vmax=300)
    fig.colorbar(im, ax=ax, label="Terrain elevation [m]", shrink=0.85)

    s = QUIVER_STRIDE
    ax.quiver(xc[::s], yc[::s], ud[::s, ::s], vd[::s, ::s],
              color="black", angles="xy", scale_units="xy", scale=3.0 / 100.0,
              width=0.003)

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"WRF-Fire OpenBCEllipsoidBuffer comparison: terrain + wind (lowest model level) at t={t:.0f}s\n{fname}")

    out = f"wind_terrain_wrf_t{t:05d}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out} (t={t:.0f}s, source {fname})")
    ds.close()


if __name__ == "__main__":
    main()
