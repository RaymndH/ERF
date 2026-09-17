#!/usr/bin/env python3
"""Lowest-model-level max|U| vs time from a WRF-Fire wrfout series, for
direct comparison against ERF's OpenBCRidgeCorner max|z_velocity|... note:
ERF's number tracked is 3D max|U_h| too (see ../check_openbccorner.py and
the README table) -- both are horizontal wind speed magnitude at the
lowest model level, destaggered to mass points.

    python3 extract_wind_growth.py [stride_seconds]   # default: every output (60s)
"""
import glob, re, sys
import numpy as np
import netCDF4 as nc

STRIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 60


def tsec(f):
    m = re.search(r"09:(\d{2}):(\d{2})$", f)
    return int(m.group(1)) * 60 + int(m.group(2))


def main():
    files = sorted(glob.glob("wrfout_d01_2006-01-01_09:*:00"), key=tsec)
    if not files:
        sys.exit("no wrfout_d01_2006-01-01_09:*:00 files found -- run wrf.exe first")
    print(f"{'t(s)':>8}{'max|U|':>12}{'mean|U|':>12}")
    for f in files:
        t = tsec(f)
        if t % STRIDE != 0:
            continue
        ds = nc.Dataset(f)
        u = ds.variables["U"][0, 0, :, :]   # lowest mass level, x-staggered
        v = ds.variables["V"][0, 0, :, :]   # lowest mass level, y-staggered
        ud = 0.5 * (u[:, :-1] + u[:, 1:])   # destagger to mass points
        vd = 0.5 * (v[:-1, :] + v[1:, :])
        spd = np.sqrt(ud**2 + vd**2)
        print(f"{t:>8d}{spd.max():>12.3f}{spd.mean():>12.3f}")
        ds.close()


if __name__ == "__main__":
    main()
