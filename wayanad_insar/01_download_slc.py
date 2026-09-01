#!/usr/bin/env python3
"""
STEP 1 — Download Sentinel-1 IW SLC scenes + precise orbits for ONE track.

InSAR needs SLC (phase-preserving). We search ASF (NASA Earthdata) for a
single relative orbit so the whole stack shares one viewing geometry,
download the .zip SLCs, then fetch the matching precise-orbit (.EOF) files.

Prerequisites
-------------
* A free NASA Earthdata account with the ASF data-access EULA accepted
  (log in once at https://search.asf.alaska.edu/ — otherwise downloads 401/403).
* Credentials via ~/.netrc (machine urs.earthdata.nasa.gov) OR the env vars
  EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Tips
----
* Each IW SLC is ~4-8 GB. For a quick co-event-only run:
      WAYANAD_COEVENT_ONLY=1 python 01_download_slc.py
  which keeps only the two dates in config.COEVENT_PAIRS[REL_ORBIT].
* Process the second Wayanad track separately:
      WAYANAD_REL_ORBIT=63 python 01_download_slc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Data.wayanad_insar.config as C  # noqa: E402
from Data.wayanad_insar.common import asf_session, log  # noqa: E402


def main():
    import asf_search as asf
    from pygmtsar import S1

    C.ensure_dirs()
    log(
        f"Searching ASF: S1A IW SLC, {C.FLIGHT_DIRECTION} rel.orbit "
        f"{C.REL_ORBIT}, {C.SEARCH_START}..{C.SEARCH_END}"
    )
    results = asf.geo_search(
        intersectsWith=C.aoi_wkt(),
        platform=asf.PLATFORM.SENTINEL1A,       # S1A only for a mid-2024 event
        processingLevel=asf.PRODUCT_TYPE.SLC,   # full IW SLC (carries phase)
        beamMode=asf.BEAMMODE.IW,
        flightDirection=getattr(asf.FLIGHT_DIRECTION, C.FLIGHT_DIRECTION),
        relativeOrbit=C.REL_ORBIT,              # one track -> coherent stack
        start=f"{C.SEARCH_START}T00:00:00Z",
        end=f"{C.SEARCH_END}T23:59:59Z",
    )
    scenes = sorted(results, key=lambda x: x.properties["startTime"])
    log(f"{len(scenes)} SLC scenes found:")
    for p in scenes:
        pr = p.properties
        log(
            f"  {pr['sceneName']}  {pr['startTime'][:10]}  "
            f"orbit={pr['orbit']}  {pr['polarization']}"
        )
    if not scenes:
        log(
            "No scenes — check track/date/AOI, or that the ASF EULA is "
            "accepted for your Earthdata account."
        )
        return

    # Optional: keep only the co-event pair for a fast first run.
    if os.environ.get("WAYANAD_COEVENT_ONLY"):
        pre, post = C.COEVENT_PAIRS[C.REL_ORBIT]
        keep = {pre, post}
        scenes = [p for p in scenes if p.properties["startTime"][:10] in keep]
        log(f"CO-EVENT ONLY: keeping {len(scenes)} scenes {sorted(keep)}")

    session = asf_session()
    log(f"Downloading {len(scenes)} SLC .zip(s) into {C.DATADIR} (~4-8 GB each) ...")
    for p in scenes:
        name = p.properties["sceneName"] + ".zip"
        if os.path.exists(os.path.join(C.DATADIR, name)):
            log(f"  skip (exists): {name}")
            continue
        p.download(path=C.DATADIR, session=session)
        log(f"  downloaded: {name}")

    # SLC downloads do NOT include orbit files; fetch them explicitly.
    # download_orbits prefers precise POEORB, falls back to restituted RESORB.
    log("Fetching precise/restituted orbit (.EOF) files ...")
    S1.download_orbits(C.DATADIR, S1.scan_slc(C.DATADIR))
    log("Step 1 complete: SLC + orbits in " + C.DATADIR)


if __name__ == "__main__":
    main()
