#!/usr/bin/env python3
"""WRF-side counterpart of ERF's gen_ellipsoid_terrain.py -- same ellipsoid
formula (major axis at phi=-theta, theta=34deg wind bearing; semi-axes
a=1700/b=1000 m; height 300 m; centered at domain center), written in WRF's
input_ht/input_zsf cell-center convention (matching gen_wrf_rotated_ridge.py's
established index convention: 1-indexed, cell center at (i-0.5)*dx).
"""
import math

CX, CY = 2600.0, 2600.0
HEIGHT = 300.0
A, B = 2500.0, 1000.0
THETA_DEG = 34.0

_TH = math.radians(THETA_DEG)
_PHI = -_TH
_COS_P, _SIN_P = math.cos(_PHI), math.sin(_PHI)


def ellipsoid_z(x, y):
    dx, dy = x - CX, y - CY
    xp = dx * _COS_P + dy * _SIN_P
    yp = -dx * _SIN_P + dy * _COS_P
    r = math.sqrt((xp / A) ** 2 + (yp / B) ** 2)
    if r > 1.0:
        return 0.0
    return HEIGHT * 0.5 * (1.0 + math.cos(math.pi * r))


def write_wrf_terrain(fname, n, dx):
    with open(fname, "w") as f:
        f.write(f"{n} {n}\n")
        for i in range(1, n + 1):
            x = (i - 0.5) * dx
            row = []
            for j in range(1, n + 1):
                y = (j - 0.5) * dx
                row.append(f"{ellipsoid_z(x, y):.4f}")
            f.write(" ".join(row) + "\n")
    print(f"wrote {fname}: {n}x{n} @ dx={dx}")


write_wrf_terrain("input_ht", 53, 100.0)
write_wrf_terrain("input_zsf", 212, 25.0)
