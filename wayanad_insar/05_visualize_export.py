#!/usr/bin/env python3
"""
STEP 9 — Geocode, coherence-mask, visualize, and export the results.

Reads the SBAS time series + velocity (Step 8) and the interferogram stack
(Step 3-7), applies a coherence mask (drop pixels below config.COH_MIN — over
vegetated monsoon terrain most of the scene will be masked, by design),
geocodes radar coordinates to WGS84 lon/lat, writes PNG figures, and exports
GeoTIFF + NetCDF products.

Pure-Python — no GMTSAR/snaphu binaries required here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402
from Data.wayanad_insar.common import coherence_mask, log, open_project  # noqa: E402


def _savefig(plt, path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"  wrote {path}")


def _first_var(ds):
    """Return the sole/first data variable of a Dataset as a DataArray."""
    return ds[list(ds.data_vars)[0]]


def main():
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import xarray as xr

    wd = C.workdir()
    disp_path = os.path.join(wd, "timeseries_los_mm.nc")
    vel_path = os.path.join(wd, "velocity_los_mm_yr.nc")
    if not (os.path.exists(disp_path) and os.path.exists(vel_path)):
        log("Missing time-series/velocity NetCDFs. Run 04_timeseries.py first.")
        return

    os.makedirs(C.OUTDIR, exist_ok=True)
    stack = open_project(drop=False)
    tag = f"track{C.REL_ORBIT}"

    # Load tolerantly: los_displacement_mm()/velocity() should return a single
    # DataArray, but open_dataarray() throws if the file holds >1 variable, so
    # pull the first data variable explicitly (mirrors the open_dataset used
    # for the unwrap handoff).
    disp_mm = _first_var(xr.open_dataset(disp_path))
    velocity = _first_var(xr.open_dataset(vel_path))
    mean_corr = coherence_mask(stack.open_stack("intf").correlation)

    # Defend against float-ULP coordinate drift: velocity/disp came off a
    # NetCDF round-trip while mean_corr is recomputed fresh from the 'intf'
    # stack. If the grids match in shape, reattach identical coords so
    # .where() aligns cleanly instead of silently reindexing to all-NaN.
    if mean_corr.shape == velocity.shape:
        shared = [d for d in velocity.dims
                  if d in velocity.coords and d in mean_corr.coords]
        mean_corr = mean_corr.assign_coords({d: velocity.coords[d] for d in shared})

    # --- coherence mask: keep only reliable pixels ----------------------
    log(f"Applying coherence mask (coh >= {C.COH_MIN}) ...")
    velocity_m = velocity.where(mean_corr >= C.COH_MIN)
    disp_m = disp_mm.where(mean_corr >= C.COH_MIN)
    try:
        kept = 100.0 * float(velocity_m.notnull().mean())
        log(f"coherence mask keeps {kept:.1f}% of velocity pixels")
        if kept < 1.0:
            log("WARNING: almost everything was masked — expected over "
                "vegetated monsoon terrain, but check COH_MIN / grid alignment.")
    except Exception as e:
        log(f"masked-fraction check skipped: {e}")

    # --- geocode radar -> lon/lat ---------------------------------------
    log("Geocoding to WGS84 lon/lat (ra2ll) ...")
    vel_ll = stack.ra2ll(velocity_m)
    disp_ll = stack.ra2ll(disp_m)

    # --- coherence overview (the realistic 'where is signal' map) -------
    try:
        mean_corr_ll = stack.ra2ll(mean_corr)
        mean_corr_ll.plot(cmap="magma", vmin=0, vmax=1)
        plt.title(f"Wayanad {tag}: mean coherence")
        _savefig(plt, os.path.join(C.OUTDIR, f"{tag}_mean_coherence.png"))
    except Exception as e:
        log(f"coherence plot skipped: {e}")

    # --- LOS velocity (mm/yr) -------------------------------------------
    # Plot the already-mm arrays DIRECTLY. Do NOT use PyGMTSAR's *_los_mm
    # plotters here: they re-apply the phase->mm scale (-79.58 * wavelength)
    # internally and would double-convert data step 04 already put in mm.
    try:
        vel_ll.plot(cmap="RdBu_r", robust=True)
        plt.title(f"Wayanad {tag}: LOS velocity (mm/yr, coh>={C.COH_MIN})")
        _savefig(plt, os.path.join(C.OUTDIR, f"{tag}_velocity_los.png"))
    except Exception as e:
        log(f"velocity plot skipped: {e}")

    # --- cumulative displacement time series (mm) -----------------------
    try:
        tdim = next((d for d in ("date", "time", "pair") if d in disp_ll.dims), None)
        if tdim and disp_ll.sizes.get(tdim, 1) > 1:
            disp_ll.plot(col=tdim, col_wrap=4, cmap="RdBu_r", robust=True)
        else:
            disp_ll.plot(cmap="RdBu_r", robust=True)
            plt.title(f"Wayanad {tag}: cumulative LOS displacement (mm)")
        _savefig(plt, os.path.join(C.OUTDIR, f"{tag}_displacement_los.png"))
    except Exception as e:
        log(f"displacement plot skipped: {e}")

    # --- exports ---------------------------------------------------------
    log("Exporting GeoTIFF + NetCDF ...")
    try:
        stack.export_geotiff(vel_ll, os.path.join(C.OUTDIR, f"{tag}_velocity_los_mm_yr"))
        stack.export_netcdf(disp_ll, os.path.join(C.OUTDIR, f"{tag}_timeseries_los_mm"))
    except Exception as e:
        log(f"PyGMTSAR export failed ({e}); writing plain NetCDF via xarray")
        vel_ll.to_netcdf(os.path.join(C.OUTDIR, f"{tag}_velocity_los_mm_yr.nc"))
        disp_ll.to_netcdf(os.path.join(C.OUTDIR, f"{tag}_timeseries_los_mm.nc"))

    log("Step 9 complete. Outputs in " + C.OUTDIR)


if __name__ == "__main__":
    main()
