#!/usr/bin/env python3
"""Generate WRF's input_ht (atm mesh, 53x53 @ dx=100) and input_zsf (fire
mesh, 212x212 @ fdx=25) for a ridge whose crest is rotated to be
PERPENDICULAR to the 34deg-bearing wind (instead of the native NS-ridge
formula's fixed x-only dependence), mirroring the 90deg case's head-on
incidence. Matches WRF's own internal index convention exactly, confirmed
via a placeholder-file probe of module_initialize_fire.F's
read_array_2d_real calls (expects ids:ide=1:53 and ifds:ifde=1:212, cell
center at (i-0.5)*dx_local).

s(x,y) = (x-cx)*sin(theta) + (y-cy)*cos(theta) + cx   -- projection onto the
wind-bearing axis, offset so s=cx at the domain center (reduces to s=x for
theta=90, exactly reproducing the native NS-ridge formula for the 90deg case).
"""
import math

CX, CY = 2500.0, 2500.0
HEIGHT = 300.0
X_START, X_END = 1500.0, 3500.0
THETA_DEG = 34.0

th = math.radians(THETA_DEG)
SIN_T, COS_T = math.sin(th), math.cos(th)


def ridge_z(x, y):
    s = (x - CX) * SIN_T + (y - CY) * COS_T + CX
    f = max(0.0, min((s - X_START) / (X_END - X_START), 1.0))
    ms = math.pi + 2 * math.pi * f
    return HEIGHT * 0.5 * (1.0 + math.cos(ms))


def write_wrf_terrain(fname, n, dx):
    with open(fname, "w") as f:
        f.write(f"{n} {n}\n")
        for i in range(1, n + 1):
            x = (i - 0.5) * dx
            row = []
            for j in range(1, n + 1):
                y = (j - 0.5) * dx
                row.append(f"{ridge_z(x, y):.4f}")
            f.write(" ".join(row) + "\n")
    print(f"wrote {fname}: {n}x{n} @ dx={dx}")


write_wrf_terrain("input_ht", 53, 100.0)
write_wrf_terrain("input_zsf", 212, 25.0)
