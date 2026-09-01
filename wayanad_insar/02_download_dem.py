#!/usr/bin/env python3
"""
STEP 2 — Download a DEM for the AOI (Copernicus GLO-30, SRTM fallback).

The DEM is used to remove the topographic phase and to geocode results.
PyGMTSAR's ``Tiles().download_dem`` fetches and mosaics the 1x1-degree tile(s)
covering the AOI (for Wayanad: N11 E076) into a single NetCDF.

Notes
-----
* ``provider`` in {'GLO','SRTM','ALOS'}; ``product`` in {'1s' (~30 m),'3s' (~90 m)}.
* GLO/SRTM heights are orthometric (geoid); Step 3's ``stack.load_dem`` adds
  the EGM96 geoid to produce WGS84 *ellipsoidal* heights (``DEM_WGS84.nc``),
  which is what GMTSAR consumes. Do not pre-convert here.
* The old ``AWS().download_dem`` / ``GMT().download_dem`` entry points were
  removed in current PyGMTSAR — ``Tiles().download_dem`` is the supported path.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402
from Data.wayanad_insar.common import log  # noqa: E402


def main():
    from pygmtsar import Tiles

    C.ensure_dirs()
    if os.path.exists(C.DEM_FILE):
        log(f"DEM already present: {C.DEM_FILE}")
        return

    log(f"Downloading Copernicus GLO-30 DEM for bbox {C.AOI_BBOX} -> {C.DEM_FILE}")
    try:
        Tiles().download_dem(
            C.aoi_geometry(), filename=C.DEM_FILE, product="1s", provider="GLO"
        )
    except Exception as e:  # network / void handling that raises
        log(f"GLO DEM download failed: {e}")

    # Fall back to SRTM if GLO produced no file OR a mostly-void tile. GLO
    # writes the mosaic even when tiles have voids (NaN cells, not a missing
    # file), so check the void fraction, not just file existence.
    if (not os.path.exists(C.DEM_FILE)) or _void_fraction(C.DEM_FILE) > 0.2:
        log("GLO DEM missing or void-heavy — falling back to SRTM 1-arcsec")
        if os.path.exists(C.DEM_FILE):
            os.remove(C.DEM_FILE)  # else skip_exist=True would keep the void tile
        Tiles().download_dem(
            C.aoi_geometry(), filename=C.DEM_FILE, product="1s", provider="SRTM"
        )
    log("Step 2 complete: " + C.DEM_FILE)


def _void_fraction(path):
    """Fraction of NaN cells in a DEM file (0.0 if it can't be read)."""
    try:
        import xarray as xr

        da = xr.open_dataarray(path)
        return float(da.isnull().mean())
    except Exception:
        return 0.0


if __name__ == "__main__":
    main()
