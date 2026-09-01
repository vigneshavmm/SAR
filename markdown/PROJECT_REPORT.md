# SAR Disaster Monitoring — Project Report

**Project:** Satellite-radar monitoring of water/flood, drought, and landslide
hazards over India using Sentinel-1 SAR
**Date:** 2026-07-16
**Status:** Track 1 operational · Track 2 preprocessing coded + advanced analytics verified · Track 3 planned

---

## 1. Executive summary

This project uses **Sentinel-1 Synthetic Aperture Radar (SAR)** — which images
through cloud and darkness — to monitor three hazard classes over India from a
single sensor. It comprises **two independent analysis tracks plus a
model-serving plan**:

1. **Water / flood (amplitude, GRD)** — an operational, deterministic pipeline
   that maps open water, tracks reservoir drawdown, and flags new inundation over
   the **Tirupati** district using classical thresholds *and* a self-supervised
   deep change detector. Fully implemented and run end-to-end.
2. **Landslide deformation (InSAR, SLC)** — an interferometric pipeline for the
   **Wayanad / Mundakkai** landslide (30 July 2024), producing millimetric
   line-of-sight (LOS) ground-velocity maps. Preprocessing is coded against
   PyGMTSAR; the **advanced time-series engine (SBAS + PSI + STL, GPU/Dask/Zarr)
   is new and verified** against a synthetic benchmark.
3. **Model serving** — a costed plan to expose a Qwen3-1.7B model as an API.

Headline results: the water track resolves the Tirupati dry-season hydrological
cycle at ~40 m; the InSAR track's Mundakkai study reports LOS velocities of
**−30.01 to +23.67 mm/year**; and the new SBAS engine recovers a known velocity
field to **0.8 % peak error / 1.75 mm/yr RMSE** in self-test.

---

## 2. Motivation and objectives

Optical satellites go blind under the South-West monsoon — exactly when flood,
drought, and slope-failure information is most needed. SAR is all-weather and
day/night, and its two observables map cleanly onto the two hazards:

- **Amplitude (backscatter):** calm water is specular and reads near-black →
  thresholding isolates open water (Track 1).
- **Phase (interferometry):** the phase difference between two passes measures
  sub-centimetre ground motion (Track 2).

**Objectives:**
- O1 — Operational surface-water / flood / drought change-detection at district scale.
- O2 — Millimetric landslide-precursor deformation mapping via multi-temporal InSAR.
- O3 — Advanced, scalable time-series analytics (GPU/cloud-native) for O2.
- O4 — A deployment path to serve project models as an API.

---

## 3. Data

| Attribute | Track 1 — Water/flood | Track 2 — Landslide InSAR |
| --- | --- | --- |
| Sensor / product | Sentinel-1 **RTC GRD** (amplitude) | Sentinel-1 **SLC** (phase) + Copernicus GLO-30 DEM |
| Source | Microsoft Planetary Computer (`sentinel-1-rtc`) | ASF / NASA Earthdata (SLC); orbits via s1orbits |
| AOI | Tirupati, AP — `[79.298, 13.498, 79.552, 13.752]` (~27×28 km) | Mundakkai/Chooralmala, Wayanad — `[76.00, 11.40, 76.25, 11.60]` |
| Scenes | 16 acquisitions, **2026-01-06 → 2026-06-30** | 16 acquisitions, **2024-05-01 → 2024-09-30** (brackets the 30 Jul 2024 event) |
| Polarization | Dual-pol **VV + VH** | **VV** (best land coherence) |
| Grid / CRS | 10 m, **EPSG:32644** (UTM 44N), nodata −32768 | IW subswath 2, descending, relative orbit 165 (63 cross-check) |
| Volume | ~823 MB (`data/Tirupati/`) | downloaded on demand into `data/Wayanad/insar/` |

**Critical data constraint:** GRD and SLC are **not interchangeable**. GRD is
amplitude-only (no phase → no interferometry); InSAR requires SLC. The Wayanad
GRD scenes on disk therefore do **not** feed the InSAR pipeline — it downloads
its own SLC.

---

## 4. Methodology

### 4.1 Track 1 — Water & flood (`analyze_water.py`)

A single-pass ETL → analyse → visualise pipeline. All scenes are co-registered to
one downsampled (~40 m) reference grid via `WarpedVRT`, converted to decibels,
and processed by three detectors:

1. **Fixed −17 dB** VV threshold — the classical, well-understood baseline.
2. **Per-scene adaptive Otsu** — Otsu computed on the *dark tail* of each
   histogram (open water is ~0.1 % of pixels, so whole-histogram Otsu would
   mis-split), clamped to ±3 dB around −17 dB.
3. **SAFNet (Siamese Adaptive Fusion Network)** — a self-supervised,
   multi-scale CNN. A shared-weight encoder embeds both dates; per-scale
   difference features are fused by a learned per-pixel attention into a change
   probability. Trained with **no labels** via reconstruction + synthetic
   copy-paste change (so it learns SAR-speckle invariance, not the −17 dB rule).

**Design enhancements added this cycle:**
- **Dual-pol water test** — a pixel is water only if dark in **both** VV
  (< −17 dB) *and* VH (< −24 dB), rejecting smooth *dry* land (roads, sand) that
  fools VV alone.
- **Speckle cleanup** — morphological opening + removal of connected components
  < 5 px on the change masks.
- **Georeferenced export** — all masks + the SAFNet change-probability written as
  EPSG:32644 GeoTIFFs (QGIS-ready), alongside the PNG figures.
- Grid alignment guards `np.stack` against footprint drift; guarded torch/scipy
  imports (graceful degradation); fully deterministic (`SEED = 0`).

### 4.2 Track 2 — Landslide InSAR (`data/wayanad_insar/`)

**Preprocessing (steps 01–03, PyGMTSAR + GMTSAR/snaphu native binaries):**
SLC + DEM + orbit download → burst reframing to AOI → co-registration → SBAS pair
selection (≤ 24 days temporal, ≤ 150 m perpendicular baseline) → multilook
interferograms with Goldstein filtering → snaphu phase unwrapping →
phase-to-LOS-displacement.

**Advanced time-series engine (`insar_advanced.py`, new — the O3 deliverable):**
a pure-Python analytical stage in the *insar.dev* style that consumes the
unwrapped-phase stack and produces velocity + displacement + quality layers:

- **Weighted SBAS network inversion** — per-pixel normal equations built with
  `einsum`, batched linear solve (no pixel loop), Tikhonov-damped for stability.
- **PSI point selection** — temporal coherence + mean coherence + amplitude-
  dispersion index isolate persistent scatterers.
- **STL decomposition** — trend/seasonal/residual separation → de-seasonalised
  velocity (critical where a seasonal signal aliases into a linear fit).
- **Robust Theil–Sen velocity** option (outlier-resistant).
- **Advanced compute:** GPU-optional (torch MPS/CUDA) linear-algebra backend,
  Dask-style chunked out-of-core pixel processing, and cloud-native **Zarr**
  output (local / `s3://` / `gcs://` via fsspec) or NetCDF.

Phase→displacement uses the Sentinel-1 C-band wavelength λ = 55.465 mm
(d_LOS = λ·Δφ / 4π); velocities are scaled to mm/year (×365.25).

### 4.3 Track 3 — Model serving (`QWEN.md`)

Plan to move a **Qwen3-1.7B** model (Apache-2.0) from a local machine to a
callable API. Recommendation: **RunPod Serverless + vLLM** (OpenAI-compatible
endpoint, scale-to-zero, ~5–10 s cold start), with local-tunnel / on-device as
zero-budget fallbacks.

---

## 5. Results

### 5.1 Track 1 — Water/flood (verified run, ~6 min, deterministic)

- **Surface-water cycle (dual-pol "confident open water"):** ~**1.4 km²** in
  early January, drawn down to a ~**0.5 km²** minimum by late June, then a sharp
  rebound to ~**2.4 km²** on the **30 June** pass as the monsoon onset refills
  irrigation tanks — a clean drought/drawdown signal.
- **New-water vs dry baseline:** ~0 km² (16 px threshold / 25 px SAFNet). This is
  correct for the window: June water overlaps the *permanent* reservoirs, so
  little is *newly* inundated — this is a drawdown period, not a flood event.
- **SAFNet vs threshold (labelled PROXY, not ground truth):** precision 0.64,
  recall 1.00, F1 0.78, IoU 0.64.
- **Outputs:** 3 figures (`water_area_timeseries.png`, `flood_change_map.png`,
  `safnet_change_map.png`) + 3 GeoTIFFs (`new_water_threshold.tif`,
  `new_water_safnet.tif`, `safnet_change_prob.tif`).

### 5.2 Track 2 — Landslide InSAR

**Existing deliverable — Mundakkai Hill InSAR Displacement Report** (18-page
figure report, `Mundakkai_Hill_Landslide_Report.pdf`): Sentinel-1 C-band,
16 acquisitions (10 Jan – 01 Aug 2024), standard chain (DEM topographic-phase
removal → wrapped phase → coherence → unwrapping → 1.5 km Gaussian detrend →
least-squares velocity). Result: a concentric, localized deformation core with
**LOS velocity −30.01 to +23.67 mm/year**. Stated caveats: LOS-relative only,
subsidence-vs-uplift sign not ground-truth validated, ~8 % border pixels masked.

**New advanced engine — verified by synthetic benchmark (`--selftest`, exit 0):**

| Check | Result |
| --- | --- |
| SBAS velocity recovery (known −30 mm/yr field) | peak error **0.8 %**, bias +0.48, **RMSE 1.75 mm/yr**, spatial corr 0.957, temporal coherence 0.998 |
| STL seasonal/trend separation | recovers seasonal (1.37 mm std vs 2.0 injected) + negative subsidence trend |
| Zarr / NetCDF dataset export | all layers present |

The synthetic stack mirrors the study geometry (16 dates, 12-day cadence,
≤ 24-day baselines, coherence-scaled noise), so the benchmark exercises the real
inversion, not a toy.

---

## 6. Implementation and architecture

```
SAR/
├── analyze_water.py         # Track 1 pipeline (threshold + Otsu + SAFNet)
├── stac.py                  # Track 1 downloader (Planetary Computer STAC)
├── data/
│   ├── Tirupati/            #   Track-1 RTC GRD, VV+VH  (~823 MB)
│   ├── Wayanad/             #   Track-2 GRD + downloaded SLC/DEM
│   ├── wayanad_insar/       #   Track-2 pipeline (01…05) + insar_advanced.py + .venv-insar
│   ├── IND_shp/, Srikalahasti/, Tirupati District/
├── outputs/                 # PNGs + GeoTIFFs
├── markdown/                # documentation (this report, README, CLAUDE, QWEN, …)
└── venv/                    # Track-1 Python env
```

- **Two isolated environments:** `venv` (Py 3.9.6) for the water track;
  `data/wayanad_insar/.venv-insar` (Py 3.11.15) for InSAR — not interchangeable.
- **Graceful degradation:** both tracks guard optional dependencies (torch,
  scipy, statsmodels, zarr) so core products always emit.
- **Determinism / reproducibility:** `SEED = 0` throughout the water track;
  headless Matplotlib (`Agg`).
- **Not under version control** as a git repo; large/binary artifacts kept out of
  context via `.gitignore` / `.claudeignore`.

---

## 7. Validation and verification

- **Track 1** runs end-to-end deterministically; SAFNet metrics are explicitly
  labelled a **proxy** (scored against the threshold method) until a real
  `reference_water.tif` ground-truth mask is supplied via `REF_WATER`.
- **Track 2 analytics** are validated against a synthetic stack with a *known*
  velocity/seasonal field (Section 5.2), decoupling velocity accuracy from STL
  seasonal separation so each is independently checked.
- **Track 2 preprocessing** is coded but **not executed in this environment**
  (requires GMTSAR + snaphu native binaries, not installed here); the one
  integration point to confirm against real data is the `unwrap.nc` schema.

---

## 8. Limitations

- **Dual-pol is deliberately conservative** — requiring VV *and* VH dark reduces
  false positives but undercounts thin/rough water; reported km² are *confident*
  open water, not a maximal estimate.
- **SAFNet is unsupervised and unvalidated on ground truth** — synthetic
  copy-paste change only approximates real hydrology.
- **InSAR reports LOS motion only** — no vertical/horizontal separation, no
  sign validation; C-band coherence over Wayanad's vegetated, monsoon-wet slopes
  is the dominant limiting factor.
- **`insardev` (insar.dev) core is subscription-licensed** and its preprocessing
  still needs GMTSAR; only the analysis core is pure-Python/GPU. The advanced
  engine here reimplements that analytical recipe openly on the project stack.
- **NRT ≠ real-time** — detection lags each satellite pass plus processing.

---

## 9. Future work

1. **Ground-truth validation** — supply a labelled water/change mask
   (`REF_WATER`) and fine-tune SAFNet on labelled pairs (e.g. Sen1Floods11).
2. **Run the InSAR track end-to-end** on a GMTSAR-provisioned host; wire
   `insar_advanced.py` in as step 6 with GeoTIFF velocity/PS-map export.
3. **Multi-track InSAR fusion** (orbits 165 + 63) toward vertical/horizontal
   decomposition of the LOS field.
4. **GPU acceleration in production** — exercise the torch MPS/CUDA backend on a
   GPU host; evaluate a subscription `insardev` core for snaphu-free unwrapping.
5. **Operationalise** — ingest a Copernicus/ASF NRT stream; deploy the Qwen API
   (Track 3) for report generation / analytics.

---

## 10. Conclusion

The project demonstrates a coherent, dual-modality SAR disaster-monitoring
capability: an operational, verified water/flood pipeline at district scale, and
a landslide-InSAR track whose advanced SBAS/PSI/STL analytics are benchmarked to
sub-2-mm/yr velocity accuracy. Both tracks are engineered for reproducibility and
graceful degradation, and a clear path exists to ground-truth validation, GPU/
cloud scaling, and API deployment.

---

## Appendix A — Reproduction commands

```bash
# Track 1 — water/flood (~6 min; SAFNet trains on CPU)
./venv/bin/python stac.py                 # download/clip Sentinel-1 RTC scenes
./venv/bin/python analyze_water.py         # 3 figures + 3 GeoTIFFs in outputs/

# Track 2 — InSAR (requires GMTSAR + snaphu on PATH)
./data/wayanad_insar/.venv-insar/bin/python data/wayanad_insar/run_pipeline.py

# Track 2 — advanced analytics (no SLC/GMTSAR needed to verify)
./data/wayanad_insar/.venv-insar/bin/python data/wayanad_insar/insar_advanced.py --selftest
./data/wayanad_insar/.venv-insar/bin/python data/wayanad_insar/insar_advanced.py \
    --input <workdir>/unwrap.nc --out outputs/timeseries.zarr --stl --robust
```

## Appendix B — Key parameters

| Track | Parameter | Value | Meaning |
| --- | --- | --- | --- |
| 1 | `WATER_DB` / `WATER_DB_VH` | −17 / −24 dB | VV / VH open-water thresholds |
| 1 | `DS` | 4 | downsample → ~40 m pixels |
| 1 | `MIN_WATER_PX` | 5 | min connected-component size (despeckle) |
| 1 | `SEED` | 0 | deterministic sampling + init |
| 2 | `SBAS_DAYS` / `SBAS_METERS` | 24 / 150 | max temporal / perpendicular baseline |
| 2 | `COH_MIN` | 0.35 | coherence mask floor |
| 2 | λ (C-band) | 55.465 mm | phase→displacement scaling |

## Appendix C — Documentation index (`markdown/`)

`README.md` (project front door) · `CLAUDE.md` (operational guide) ·
`analyze_water.md` (water pipeline walkthrough) ·
`Mundakkai_Hill_Landslide_Report.pdf` + `landslide_text.txt` (InSAR deliverable) ·
`QWEN.md` (model-serving plan) · `PROJECT_REPORT.md` (this report).

## Appendix D — References / sources

- Microsoft Planetary Computer — Sentinel-1 RTC collection.
- Alaska Satellite Facility (ASF) / NASA Earthdata — Sentinel-1 SLC.
- Copernicus GLO-30 Global DEM.
- PyGMTSAR / insar.dev — InSAR processing ecosystem (`insar.dev`,
  `github.com/AlexeyPechnikov/pygmtsar`).
- 2024 Wayanad (Mundakkai–Chooralmala) landslides, 30 July 2024.
