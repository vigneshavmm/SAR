"""
Shared helpers for the Wayanad InSAR step scripts (01..05).

Kept deliberately small: credential handling, a binary sanity-check, and one
``open_project()`` factory so every step reconstructs the same per-track Stack.
"""
from __future__ import annotations

import os
import sys

# Make ``import config`` / ``import common`` work no matter the CWD.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402


def log(msg: str) -> None:
    print(f"[wayanad-insar] {msg}", flush=True)


def asf_session():
    """asf_search session: explicit env creds if set, else ~/.netrc."""
    import asf_search as asf

    if C.EARTHDATA_USERNAME and C.EARTHDATA_PASSWORD:
        return asf.ASFSession().auth_with_creds(
            C.EARTHDATA_USERNAME, C.EARTHDATA_PASSWORD
        )
    # Reads ~/.netrc entry for machine urs.earthdata.nasa.gov.
    return asf.ASFSession()


def check_binaries() -> bool:
    """Warn early if the native binaries PyGMTSAR shells out to are missing.

    Reframe + co-registration call GMTSAR (make_s1a_tops / ext_orb_s1a /
    assemble_tops / esarp); phase unwrapping calls snaphu; GMT underlies both.
    Pure-Python steps (DEM, multilook, Goldstein, LOS, SBAS, export) do not.
    """
    import shutil

    needed = ("make_s1a_tops", "esarp", "snaphu", "gmt")
    missing = [b for b in needed if shutil.which(b) is None]
    if missing:
        log(
            "WARNING: missing binaries on PATH: "
            + ", ".join(missing)
            + ". GMTSAR (>=6.7) + snaphu + GMT 6.x are required for "
            "reframe/coregistration/unwrapping — see README §Setup. "
            "Pure-Python steps will still run."
        )
    return not missing


def open_project(rel_orbit: int | None = None, drop: bool = False):
    """Create (``drop=True``) or reopen a per-track PyGMTSAR ``Stack`` with the
    scenes and DEM attached.

    Reopening lets steps 04/05 pick up where 03 left off: PyGMTSAR persists the
    heavy intermediates (aligned SLCs, ``DEM_WGS84.nc``, geocode tables, saved
    interferogram stacks) inside WORKDIR, so re-attaching scenes + DEM restores
    the context cheaply without recomputing alignment.

    [VERIFY] If your installed PyGMTSAR build needs a different reopen sequence
    (e.g. an explicit re-align), adjust it here — this is the single place all
    steps go through.
    """
    from pygmtsar import S1, Stack

    wd = C.workdir(rel_orbit)
    scenes = S1.scan_slc(C.DATADIR)
    if scenes is None or len(scenes) == 0:
        raise RuntimeError(
            f"open_project: no Sentinel-1 SLCs found in {C.DATADIR}. "
            "Steps 03-05 re-scan the raw SLC zips on open, so they must remain "
            "on disk until the pipeline finishes. Re-run 01_download_slc.py "
            "(or keep the SLCs) and try again."
        )
    stack = Stack(wd, drop_if_exists=drop)
    stack.set_scenes(scenes)
    if C.REFERENCE_DATE:
        stack.set_reference(C.REFERENCE_DATE)
    if os.path.exists(C.DEM_FILE):
        stack.load_dem(C.DEM_FILE, geometry="auto")
    return stack


def coherence_mask(corr):
    """Mean coherence across the pair dimension (whatever it is named)."""
    for dim in ("pair", "pairs", "date"):
        if dim in getattr(corr, "dims", ()):
            return corr.mean(dim)
    return corr
