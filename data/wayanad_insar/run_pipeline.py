#!/usr/bin/env python3
"""
Orchestrate the full Wayanad landslide InSAR pipeline (steps 1-9) for the
configured track, running each numbered step in order, in one process.

Usage
-----
    python run_pipeline.py                 # all steps, track from config
    python run_pipeline.py 2 3 4 5         # only these steps (e.g. skip download)
    WAYANAD_REL_ORBIT=63 python run_pipeline.py     # process the other track
    WAYANAD_COEVENT_ONLY=1 python run_pipeline.py   # fast: co-event pair only

Process the two Wayanad tracks (165 and 63) in SEPARATE runs — never mixed in
one interferometric stack (different geometry). Combine only at the
displacement/decomposition stage.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = {
    1: "01_download_slc.py",
    2: "02_download_dem.py",
    3: "03_process_insar.py",
    4: "04_timeseries.py",
    5: "05_visualize_export.py",
}


def _load(filename):
    """Import a numbered step module by path (leading-digit names aren't
    importable the normal way) and return it."""
    path = os.path.join(HERE, filename)
    mod_name = "step_" + filename.split("_")[0]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    sys.path.insert(0, HERE)
    import Data.wayanad_insar.config as C

    want = [int(a) for a in argv] if argv else sorted(STEPS)
    print(f"=== Wayanad InSAR pipeline | track {C.REL_ORBIT} | steps {want} ===")
    for n in want:
        print(f"\n----- STEP {n}: {STEPS[n]} -----")
        _load(STEPS[n]).main()
    print("\n=== pipeline finished ===")


if __name__ == "__main__":
    main(sys.argv[1:])
