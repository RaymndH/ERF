#!/usr/bin/env python3
"""Elongated ellipsoid terrain for OpenBCEllipsoidBuffer: same domain, wind,
and Open-BC-on-all-sides setup as OpenBCRidgeCorner's ridge, but with the
infinite ridge replaced by a FINITE elliptical bump, major axis aligned
with where the ridge crest was (perpendicular to the 34deg wind bearing,
same head-on-incidence convention as gen_ridge_terrain.py), and kept well
clear of every boundary -- unlike the ridge, which ran through the domain
edges/corners.

Semi-axes a=1700 m (major, along the crest direction), b=1000 m (minor,
matching the ridge's minor-direction slope so this is a fair steepness
comparison, not just a smaller feature);
height 300 m, same as the ridge. Centered at domain center (2600,2600).
Rotated bounding half-extent from the center: ~1523 m in x, ~1253 m in y
-- more than 1000 m of flat clearance to every boundary, comfortably past
the requested 600 m minimum.

Raised-cosine radial profile in the elliptical metric (C1 at the r=1
edge, same smoothness convention as the ridge's clamped-cosine profile):

    xp =  (x-cx)*cos(phi) + (y-cy)*sin(phi)     # along major axis
    yp = -(x-cx)*sin(phi) + (y-cy)*cos(phi)     # along minor axis
    r  = sqrt((xp/a)^2 + (yp/b)^2)
    z  = height * 0.5 * (1 + cos(pi*r))   for r <= 1, else 0

phi = -theta (theta = 34 deg wind bearing) is the ridge-crest direction:
the ridge's crest ran along the direction perpendicular to the wind
(constant s = (x-cx)sin(theta) + (y-cy)cos(theta)), which is exactly the
vector (cos(theta), -sin(theta)) -- so phi = -theta reproduces that same
orientation for the ellipse's major axis.

Writes terrain_atm.txt only (100 m, 53x53) -- this test has no fire mesh
(erf.fire.enable = false), so no terrain_fire.txt is needed.
"""
import math

L = 5200.0
HEIGHT = 300.0
CX, CY = 2600.0, 2600.0
A, B = 1700.0, 1000.0        # semi-major, semi-minor [m]
THETA_DEG = 34.0            # same wind bearing as OpenBCRidgeCorner

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


def write_raster(fname, dx):
    n = int(round(L / dx)) + 1
    xs = [i * dx for i in range(n)]
    ys = [j * dx for j in range(n)]
    with open(fname, "w") as f:
        f.write(f"{n}\n{n}\n")
        for x in xs:
            f.write(f"{x:.2f}\n")
        for y in ys:
            f.write(f"{y:.2f}\n")
        for x in xs:
            for y in ys:
                f.write(f"{ellipsoid_z(x, y):.4f}\n")
    zmax = max(ellipsoid_z(x, y) for x in xs for y in ys)
    print(f"{fname}: {n}x{n} nodes at dx={dx} m, peak z={zmax:.3f} m")


write_raster("terrain_atm.txt", 100.0)
