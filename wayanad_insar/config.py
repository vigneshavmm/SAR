"""
Central configuration for the Wayanad landslide InSAR pipeline (PyGMTSAR).

Everything tunable lives here so the step scripts (01..05) stay generic.
Edit the constants below, not the code in the step scripts.

Study area
----------
Chooralmala / Mundakkai, Wayanad district, Kerala, India — site of the
catastrophic debris flow of 30 July 2024. The AOI is a small box around the
failure and its runout so the (expensive) InSAR processing stays fast.

Data reality (read WAYANAD_INSAR_README.md)
-------------------------------------------
InSAR needs Sentinel-1 **SLC** (Single Look Complex — it preserves phase).
The ``*_GRDH_*_rtc_*.tif`` files already in ../Wayanad are GRD *amplitude*
only and CANNOT be used here. Step 1 downloads SLC from ASF instead.
"""
from __future__ import annotations

import os
from datetime import date

# --- Area of interest ----------------------------------------------------
# bbox as (W, S, E, N) in EPSG:4326 covering the landslide scar + runout.
AOI_BBOX = (76.00, 11.40, 76.25, 11.60)

# The landslide date (used only to label / bracket the co-event pair).
EVENT_DATE = date(2024, 7, 30)

# --- Sentinel-1 track selection -----------------------------------------
# Wayanad is covered by TWO descending tracks (verified from the on-disk GRD
# acquisition times ~00:40 and ~00:49 UTC). Process ONE track per project dir;
# never mix relative orbits in a single interferometric stack.
#
#   165 -> ~00:40 UTC, co-event 12-day pair 2024-07-20 / 2024-08-01 (PRIMARY)
#    63 -> ~00:49 UTC, co-event 12-day pair 2024-07-25 / 2024-08-06 (cross-check)
FLIGHT_DIRECTION = "DESCENDING"
REL_ORBIT = int(os.environ.get("WAYANAD_REL_ORBIT", 165))

# Acquisition window to search / stack (pre-monsoon baseline -> post-event).
SEARCH_START = "2024-05-01"
SEARCH_END = "2024-09-30"

# Reference (super-master) scene date. None -> PyGMTSAR auto-picks the first
# scene. Pinning the pre-event date is often more stable for a co-event study.
REFERENCE_DATE = None  # e.g. "2024-07-20"

# Co-event pair (pre, post) for damage / coherence-change mapping, per track.
COEVENT_PAIRS = {
    165: ("2024-07-20", "2024-08-01"),
    63: ("2024-07-25", "2024-08-06"),
}

# --- Sentinel-1 product parameters --------------------------------------
SUBSWATHS = 2         # IW2 covers the AOI; widen (e.g. '123') if you extend it
POLARIZATION = "VV"   # co-pol gives the best land coherence

# --- Paths ---------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PROJROOT = os.path.dirname(HERE)                              # .../Downloads/SAR
INSAR_ROOT = os.path.join(PROJROOT, "Wayanad", "insar")
DATADIR = os.path.join(INSAR_ROOT, "slc")                    # raw SLC + .EOF orbits
DEMDIR = os.path.join(INSAR_ROOT, "dem")
OUTDIR = os.path.join(INSAR_ROOT, "outputs")
DEM_FILE = os.path.join(DEMDIR, "dem.nc")


def workdir(rel_orbit: int | None = None) -> str:
    """Per-track processing dir (kept separate so tracks never mix)."""
    return os.path.join(INSAR_ROOT, f"track{rel_orbit or REL_ORBIT}")


# --- Processing / tuning knobs ------------------------------------------
WAVELENGTH = 200        # m, Gaussian anti-alias cutoff for multilooking
PSIZE = 32              # Goldstein filter patch size (px); triggers filtering
COARSEN = (1, 4)        # range/azimuth multilook (square-ish output cells)
GEOCODE_COARSEN = 60.0  # radar<->geo transform coarsening
SBAS_DAYS = 24          # max temporal baseline (days) for SBAS pairs
SBAS_METERS = 150       # max perpendicular baseline (m) for SBAS pairs

# Coherence mask: displacement below this correlation is dropped as noise.
# Over vegetated monsoon terrain keep this HIGH; expect most of the scene
# (forest) to be masked out. 0.3-0.5 is typical.
COH_MIN = 0.35

# --- Earthdata credentials ----------------------------------------------
# Preferred: a ~/.netrc entry for machine urs.earthdata.nasa.gov (chmod 600).
# Fallback: export EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
EARTHDATA_USERNAME = os.environ.get("EARTHDATA_USERNAME")
EARTHDATA_PASSWORD = os.environ.get("EARTHDATA_PASSWORD")


def ensure_dirs() -> None:
    for d in (DATADIR, DEMDIR, OUTDIR, workdir()):
        os.makedirs(d, exist_ok=True)


def aoi_geometry():
    """Shapely box for the AOI (accepted directly by PyGMTSAR)."""
    from shapely.geometry import box

    return box(*AOI_BBOX)


def aoi_wkt() -> str:
    """AOI as a WKT polygon for asf_search.intersectsWith."""
    w, s, e, n = AOI_BBOX
    return f"POLYGON(({w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}))"
