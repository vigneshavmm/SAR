#!/usr/bin/env python3
"""
STEPS 3-7 (insar.dev version) — Preprocessing with the insar.dev toolkit.

This script replaces the pygmtsar/GMTSAR/snaphu-based `03_process_insar.py`
with a unified, Python-native workflow using the `insardev` library.

The goal is to produce the same key output — a NetCDF or Zarr store of
co-registered, unwrapped phase — that the existing `insar_advanced.py`
engine can directly consume for time-series analysis.

This approach aims to:
1. Eliminate the dependency on external C/Fortran binaries (GMTSAR, snaphu).
2. Unify the entire pipeline into a single Python environment.
3. Leverage the performance and scalability of the insar.dev ecosystem.

NOTE: The `insardev` API is illustrative and based on the project's design
philosophy. Actual function names and parameters may vary.
"""
import os
import insardev
import config as C
from common import log

def main():
    # The insardev library often uses its own methods for finding scenes
    # and managing credentials, simplifying the setup.
    log("Initializing insar.dev project...")
    stack = insardev.Stack(
        workdir=C.workdir(),
        name=f"Wayanad_Track{C.REL_ORBIT}",
        # The insardev stack often takes scene discovery parameters directly
        satellite="Sentinel-1",
        bbox=C.AOI_BBOX,
        date_range=C.DATE_RANGE,
        polarization="VV",
        relative_orbit=C.REL_ORBIT,
    )

    # Step 1: Prepare the data stack (download, DEM, orbits)
    # This single command can replace `01_download_slc.py` and `02_download_dem.py`
    log("Preparing stack: downloading scenes, orbits, and DEM...")
    stack.prepare(dem_provider="Copernicus GLO-30")
    log(f"Found {len(stack.scenes)} scenes.")

    # Step 2: Process the stack to generate unwrapped phase
    # This single, high-level command encapsulates the complex chain of
    # co-registration, interferogram generation, and unwrapping.
    log("Processing stack to generate unwrapped phase...")
    # The `process` method is configured with parameters for the entire chain.
    # It's designed for out-of-core, parallel execution on Dask/GPU.
    unwrap = stack.process(
        temporal_baseline_days=C.SBAS_DAYS,
        multilook=C.COARSEN,
        filter_strength=0.5, # Corresponds to Goldstein `psize`
        unwrap_method="snaphu_mcf", # Or potentially a pure-python unwrapper
    )
    log("Processing complete.")

    # Step 3: Save the output for the advanced analysis engine
    upath = os.path.join(C.workdir(), "unwrap.zarr")
    unwrap.to_zarr(upath, mode="w", consolidated=True)
    log(f"Saved unwrapped phase stack -> {upath}")
    log("insar.dev preprocessing complete. Ready for insar_advanced.py.")

if __name__ == "__main__":
    main()