#!/usr/bin/env python3
"""
STEP 8 — SBAS time-series analysis.

Invert the network of unwrapped interferometric pairs (built in Step 3-7) into
a per-date cumulative LOS displacement time series and a mean LOS velocity,
using PyGMTSAR's least-squares (SBAS) solver.

Needs >=3 acquisitions to be meaningful; with only a co-event pair this reduces
to a single displacement map (in which case prefer the Step 3-7 output).
Pure-Python — no GMTSAR/snaphu binaries required here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402
from Data.wayanad_insar.common import log, open_project  # noqa: E402


def main():
    import xarray as xr

    wd = C.workdir()
    upath = os.path.join(wd, "unwrap.nc")
    if not os.path.exists(upath):
        log(f"Missing {upath}. Run 03_process_insar.py first.")
        return

    stack = open_project(drop=False)

    # Coherence weights come from the saved interferogram stack; unwrapped
    # phase from the NetCDF that Step 3-7 persisted.
    corr = stack.open_stack("intf").correlation
    unwrap = xr.open_dataset(upath)

    log("SBAS least-squares inversion (lstsq) ...")
    disp = stack.lstsq(unwrap.phase, weight=corr)   # cumulative per-date phase
    disp_mm = stack.los_displacement_mm(disp)       # -> mm
    velocity = stack.velocity(disp_mm)              # mm/year

    disp_path = os.path.join(wd, "timeseries_los_mm.nc")
    vel_path = os.path.join(wd, "velocity_los_mm_yr.nc")
    disp_mm.to_netcdf(disp_path)
    velocity.to_netcdf(vel_path)
    log(f"Saved time series -> {disp_path}")
    log(f"Saved velocity    -> {vel_path}")

    # Optional inversion-quality metric.
    try:
        rmse = stack.rmse(unwrap.phase, disp, weight=corr)
        rmse.to_netcdf(os.path.join(wd, "rmse.nc"))
        log("Saved rmse.nc")
    except Exception as e:  # non-fatal
        log(f"rmse skipped: {e}")
    log("Step 8 complete.")


if __name__ == "__main__":
    main()
