#!/usr/bin/env python3
"""
STEPS 3-7 — Initialize project, co-register, build interferograms, filter and
unwrap phase, and convert phase to line-of-sight (LOS) displacement.

This is the heavy stage. It requires GMTSAR (>=6.7) + snaphu + GMT 6.x on PATH
(reframe/align use GMTSAR; unwrapping uses snaphu). All intermediates are
persisted inside the per-track WORKDIR so steps 04/05 can reopen the project.

Pipeline mapping
----------------
  Step 3  init project   -> Stack(...) + set_scenes + compute_reframe + load_dem
  Step 4  co-register    -> compute_align + compute_geocode (+ baseline_table)
  Step 5  interferograms -> sbas_pairs + compute_interferogram_multilook
  Step 6  filter+unwrap  -> Goldstein (inline via psize) + unwrap_snaphu
  Step 7  displacement   -> los_displacement_mm
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402
from Data.wayanad_insar.common import check_binaries, log  # noqa: E402


def main():
    check_binaries()
    from pygmtsar import S1, Stack

    wd = C.workdir()
    scenes = S1.scan_slc(C.DATADIR)
    if len(scenes) < 2:
        log(
            f"Need >=2 SLC scenes in {C.DATADIR}; found {len(scenes)}. "
            "Run 01_download_slc.py first."
        )
        return
    log(f"Init project {wd} with {len(scenes)} scenes (track {C.REL_ORBIT})")

    # ---- STEP 3: init + reframe to AOI + DEM ----------------------------
    stack = Stack(wd, drop_if_exists=True)
    stack.set_scenes(scenes)
    if C.REFERENCE_DATE:
        stack.set_reference(C.REFERENCE_DATE)
    log("Reframing bursts to AOI (GMTSAR) ...")
    stack.compute_reframe(geometry=C.aoi_geometry())
    log("Loading DEM (EGM96 -> WGS84 ellipsoidal) ...")
    stack.load_dem(C.DEM_FILE, geometry="auto")

    # ---- STEP 4: co-registration + geocode tables -----------------------
    log("Co-registering the stack (align) ...")
    stack.compute_align()
    log("Computing radar<->geo geocode tables ...")
    stack.compute_geocode(coarsen=C.GEOCODE_COARSEN)
    try:
        log("Baseline table:\n" + str(stack.baseline_table()))
    except Exception as e:  # non-fatal: informational only
        log(f"baseline_table unavailable: {e}")

    # ---- STEP 5: interferograms (Goldstein applied inline via psize) ----
    pairs = stack.sbas_pairs(days=C.SBAS_DAYS, meters=C.SBAS_METERS)
    log(f"{len(pairs)} SBAS pair(s) within {C.SBAS_DAYS} d / {C.SBAS_METERS} m")
    if len(pairs) == 0:
        log(
            f"No SBAS pairs within {C.SBAS_DAYS} d / {C.SBAS_METERS} m — "
            "widen SBAS_DAYS / SBAS_METERS in config.py or add more scenes."
        )
        return
    weight = stack.psfunction()  # persistent-scatterer-like pixel weight
    log("Computing multilook interferograms + correlation (Goldstein psize="
        f"{C.PSIZE}) ...")
    stack.compute_interferogram_multilook(
        pairs,
        "intf",
        weight=weight,
        wavelength=C.WAVELENGTH,
        psize=C.PSIZE,
        coarsen=C.COARSEN,
    )
    ds = stack.open_stack("intf")
    intf, corr = ds.phase, ds.correlation

    # ---- STEP 6: unwrap phase (snaphu), weighted by coherence -----------
    log("Unwrapping phase with snaphu (weighted by coherence) ...")
    unwrap = stack.unwrap_snaphu(intf, weight=corr)
    upath = os.path.join(wd, "unwrap.nc")
    unwrap.to_netcdf(upath)  # persist for the SBAS step (04)
    log(f"Saved unwrapped phase -> {upath}")

    # ---- STEP 7: phase -> LOS displacement (mm), per pair ---------------
    # This is the standalone Step-7 product: LOS displacement for each
    # interferometric pair (incl. the co-event 07-20/08-01 pair). The SBAS
    # route in 04_timeseries.py re-inverts the network from unwrap.nc instead,
    # so this file is a deliverable in its own right, not a 04/05 input.
    los_mm = stack.los_displacement_mm(unwrap.phase)
    lpath = os.path.join(wd, "los_pairs_mm.nc")
    los_mm.to_netcdf(lpath)
    log(f"Saved per-pair LOS displacement (Step-7 product) -> {lpath}")
    log("Steps 3-7 complete.")


if __name__ == "__main__":
    main()
