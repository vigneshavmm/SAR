# Mundakkai / Wayanad Landslide — InSAR Displacement Project Report

**Site:** Mundakkai–Chooralmala, Wayanad district, Kerala, India
**Event:** Catastrophic debris flow of **30 July 2024**
**Sensor:** Sentinel-1 C-band SAR (SLC)
**Date:** 2026-07-16 · **Status:** Preprocessing coded (PyGMTSAR/GMTSAR); advanced time-series analytics implemented and verified

---

## Abstract

This report presents a multi-temporal Interferometric SAR (InSAR) analysis of
ground deformation around Mundakkai Hill, Wayanad — the slope that failed in the
catastrophic Mundakkai–Chooralmala debris flow of 30 July 2024. Using Sentinel-1
C-band **Single-Look-Complex (SLC)** acquisitions over a pre-monsoon-to-post-event
window, the interferometric phase is processed into coherence, unwrapped phase,
and line-of-sight (LOS) displacement, and inverted into a per-pixel LOS velocity
field via a Small-BAseline-Subset (SBAS) network. The reference study
(*Mundakkai_Hill_Landslide_Report.pdf*, 16 acquisitions, 10 Jan – 01 Aug 2024)
reports a highly localized, concentric deformation core with **LOS velocity from
−30.01 to +23.67 mm/year**. This report additionally documents an **advanced
time-series engine** (weighted SBAS + PSI + STL, GPU/Dask/Zarr) that reproduces a
known synthetic velocity field to **0.8 % peak error / 1.75 mm/yr RMSE**, and
lays out the implementation, results, and limitations of the end-to-end pipeline.

---

## Area of Interest

The AOI is a small box around the failure scar and its runout, kept tight so the
(expensive) interferometric processing stays fast and coherent.

| Parameter | Value |
| --- | --- |
| Processing bounding box (W,S,E,N) | `76.00, 11.40, 76.25, 11.60` (EPSG:4326) |
| Reference-report bbox (tight core) | `76.12683, 11.45700, 76.14517, 11.47500` |
| Location | Mundakkai / Chooralmala, Meppadi panchayat, Wayanad, Kerala |
| Terrain | Steep, densely vegetated Western Ghats slope; monsoon-wet |
| Event date bracketed | 30 July 2024 |

Wayanad is covered by **two descending tracks** (relative orbits **165** and
**63**, acquisition times ~00:40 and ~00:49 UTC). Each track is processed in a
separate stack — geometry differs, so relative orbits are never mixed in one
interferometric network; they are combined only at the displacement stage.

---

## Pre-processing and Auxiliary Data

| Component | Detail |
| --- | --- |
| **SLC scenes** | Sentinel-1 IW, **VV** polarization (best land coherence), **subswath IW2** (covers the AOI). Search window **2024-05-01 → 2024-09-30**. Source: ASF / NASA Earthdata. |
| **Co-event pairs** | Track 165: `2024-07-20 / 2024-08-01`; Track 63: `2024-07-25 / 2024-08-06` (12-day pairs bracketing the event). |
| **Precise orbits** | Sentinel-1 restituted/precise `.EOF` orbit files (s1orbits mirror, `sentineleof` fallback). |
| **DEM** | **Copernicus GLO-30** (1 arc-second, ~30 m), EGM96 → WGS-84 ellipsoidal, used to simulate and remove the topographic phase. |
| **Burst reframing** | SLC bursts are reframed to the AOI polygon so only the relevant sub-scene is processed. |

Why SLC and not GRD: interferometry needs the **phase** that only SLC preserves.
The Sentinel-1 GRD (amplitude) products already on disk **cannot** be used — the
pipeline downloads SLC independently.

---

## Methodology and Techniques

The workflow is a Small-BAseline-Subset (SBAS) multi-temporal InSAR chain, with an
advanced analytical stage layered on top:

1. **Baseline network (SBAS):** interferometric pairs are formed only within a
   **≤ 24-day temporal** and **≤ 150-m perpendicular** baseline, limiting temporal
   and spatial decorrelation.
2. **Topographic-phase removal** using the Copernicus DEM, isolating the
   deformation + atmospheric + noise phase.
3. **Multilooking** (range/azimuth coarsening, Gaussian anti-alias cutoff) to
   suppress speckle and square-up output cells.
4. **Goldstein adaptive filtering** (patch size 32 px) to raise fringe
   visibility in low-coherence terrain.
5. **Coherence-weighted phase unwrapping** to resolve 2π ambiguities.
6. **Phase → LOS displacement** conversion, then **network inversion** to a
   per-date cumulative displacement time series and a per-pixel **LOS velocity**.
7. **Advanced analytics** (this project's extension): weighted SBAS inversion,
   **PSI** persistent-scatterer selection, **STL** seasonal/trend decomposition,
   and robust velocity estimation — GPU-accelerated and cloud-native (below).

Key phase relation (C-band, λ ≈ 55.465 mm):

> Δφ = (4π / λ) · d_LOS  →  d_LOS = (λ / 4π) · Δφ

Cumulative displacement d(tᵢ) is referenced to the first epoch (d(t₀)=0), and the
long-term velocity is the least-squares slope of d(t), scaled to mm/year (×365.25).

---

## Core InSAR Processing Chain

The heavy preprocessing stage (`03_process_insar.py`) is built on **PyGMTSAR**
and shells out to native binaries (GMTSAR ≥ 6.7, snaphu, GMT 6.x):

```
Step 3  Init project     Stack(workdir) → set_scenes → compute_reframe(AOI) → load_dem
Step 4  Co-registration  compute_align → compute_geocode → baseline_table
Step 5  Interferograms   sbas_pairs(≤24 d, ≤150 m) → compute_interferogram_multilook
                          (Goldstein psize=32, weighted by PS-like pixel weight)
Step 6  Filter + unwrap  Goldstein (inline) → unwrap_snaphu (weighted by coherence)
Step 7  Displacement     los_displacement_mm(unwrapped_phase)  → per-pair LOS mm
```

Products persisted per track (`track165/`, `track63/`): aligned SLCs,
`DEM_WGS84.nc`, geocode tables, interferogram + correlation stacks,
`unwrap.nc`, and `los_pairs_mm.nc`.

**Advanced time-series engine (`insar_advanced.py`) — the analytical core:**
consumes the unwrapped-phase stack and produces velocity, cumulative
displacement, and quality layers using the *insar.dev*-style recipe:

- **Weighted SBAS network inversion** — per-pixel normal equations assembled with
  `einsum`, batched linear solve (no pixel loop), Tikhonov-damped for stability.
- **Temporal coherence** — per-pixel consistency of the inverted network, used to
  mask unreliable pixels.
- **PSI point selection** — temporal coherence + mean coherence + amplitude-
  dispersion index isolate persistent scatterers.
- **STL decomposition** — trend / seasonal / residual separation → de-seasonalised
  velocity (removes seasonal aliasing from the linear fit).
- **Robust Theil–Sen** velocity option (outlier-resistant).
- **Advanced compute** — GPU-optional (torch Apple MPS / NVIDIA CUDA) linear
  algebra, Dask-style chunked out-of-core processing, and cloud-native **Zarr**
  output (local / `s3://` / `gcs://`) or NetCDF.

The unwrapping and interferogram stages require the native GMTSAR/snaphu stack;
the analytical engine is pure Python and runs unchanged on any unwrapped stack.

---

## Implementation

```
data/wayanad_insar/
├── config.py               # single source of truth (AOI, track, baselines, tuning)
├── 01_download_slc.py       # SLC search + download (ASF / Earthdata)
├── 02_download_dem.py       # Copernicus GLO-30 DEM
├── 03_process_insar.py      # init → align → interferograms → unwrap → LOS (GMTSAR/snaphu)
├── 04_timeseries.py         # SBAS network inversion → displacement time series
├── 05_visualize_export.py   # maps / exports
├── insar_advanced.py        # NEW: advanced SBAS + PSI + STL engine (GPU/Dask/Zarr)
├── common.py, run_pipeline.py
├── requirements-insar.txt   # pygmtsar==2025.4.8.post1, asf_search, sentineleof
└── .venv-insar/             # Python 3.11.15
```

- **Environment:** dedicated `.venv-insar` (Py 3.11.15); PyGMTSAR 2025.4.8.post1 +
  `asf_search` + `sentineleof`; native GMTSAR/snaphu/GMT on PATH.
- **Per-track isolation:** each relative orbit gets its own working directory;
  tracks are combined only at the displacement/decomposition stage.
- **Configuration-driven:** all tunables (AOI, track, baselines, coherence floor,
  filter sizes) live in `config.py`; step scripts stay generic.
- **Graceful degradation:** the advanced engine guards optional torch / zarr /
  statsmodels, mirroring the project's water-track convention.

**Selected configuration** (`config.py`): descending, rel-orbit 165 (63 as
cross-check); IW2 / VV; SBAS ≤ 24 days & ≤ 150 m; Goldstein psize 32; multilook
coarsen (1,4); geocode coarsen 60; coherence floor **COH_MIN = 0.35**.

---

## Result and Outcome

**Reference deliverable — Mundakkai Hill InSAR Displacement Report** (16
Sentinel-1 C-band acquisitions, 10 Jan – 01 Aug 2024): the LOS velocity field
shows a **highly localized, concentric deformation core** in the central AOI,
with values ranging **−30.01 to +23.67 mm/year** over a stable background, and a
near-linear cumulative-displacement time series at the point of maximum motion.
The monitoring window brackets the 30 July 2024 failure, capturing pre-event
slope behaviour.

**Advanced engine — verified by synthetic benchmark (`--selftest`, exit 0):**
the benchmark mirrors the study geometry (16 dates, 12-day cadence, ≤ 24-day
baselines, coherence-scaled noise) with a *known* −30 mm/yr subsidence bump.

| Check | Result |
| --- | --- |
| SBAS velocity recovery | peak error **0.8 %**, bias +0.48, **RMSE 1.75 mm/yr**, spatial correlation 0.957, temporal coherence 0.998 |
| STL seasonal/trend separation | seasonal recovered (1.37 mm std vs 2.0 injected) + negative subsidence trend |
| Zarr / NetCDF export | velocity, cumulative displacement, temporal coherence, PS mask all written |

**Outcome:** an end-to-end, per-track InSAR pipeline for the Wayanad landslide,
with an analytical stage benchmarked to sub-2-mm/yr velocity accuracy and
engineered for GPU/cloud scaling — ready to run on real unwrapped stacks once the
native preprocessing stack is provisioned.

---

## Limitation

- **Coherence over vegetated monsoon terrain is the dominant limiter.** Wayanad's
  dense forest/tea cover and wet monsoon conditions collapse C-band coherence,
  especially in the back half of the window; much of the scene is expected to be
  masked (COH_MIN = 0.35).
- **LOS-only.** A single descending geometry measures motion along the
  radar line-of-sight; vertical vs horizontal cannot be separated without a
  second (ascending) track, and the subsidence-vs-uplift sign is **not validated
  against ground truth**.
- **Boundary masking.** ~8 % of border pixels are dropped for coverage/processing
  reasons (per the reference report); single-date maps are atmospherically noisy —
  the velocity map (fit across the series) is the robust product.
- **InSAR cannot image the failure itself.** The 30 July 2024 debris flow is a
  rapid, meters-scale, fully-decorrelating event; InSAR captures only slow
  *precursory* deformation, not the collapse.
- **Native-binary dependency.** Preprocessing requires GMTSAR + snaphu + GMT
  (not installed in this environment), so steps 03–05 were **not executed here**;
  the analytical engine was verified independently on synthetic data.
- **`insardev` (insar.dev) constraints.** Its core is subscription-licensed and
  its preprocessing still needs GMTSAR; only the analytical recipe is reproduced
  openly here.

---

## References

1. Microsoft/ASF/NASA Earthdata — Sentinel-1 SLC (IW) products.
2. Copernicus GLO-30 Global Digital Elevation Model (1 arc-second).
3. Pechnikov, A. — *PyGMTSAR / insar.dev*: Python InSAR ecosystem
   (`insar.dev`, `github.com/AlexeyPechnikov/pygmtsar`).
4. Berardino, P. et al. (2002) — *A new algorithm for surface deformation
   monitoring based on Small BAseline differential SAR interferograms (SBAS)*,
   IEEE TGRS.
5. Ferretti, A., Prati, C., Rocca, F. (2001) — *Permanent Scatterers in SAR
   interferometry (PSI)*, IEEE TGRS.
6. Goldstein, R. M., Werner, C. L. (1998) — *Radar interferogram filtering for
   geophysical applications*, GRL.
7. Chen, C. W., Zebker, H. A. (2002) — *snaphu* statistical-cost phase unwrapping.
8. Cleveland, R. B. et al. (1990) — *STL: A Seasonal-Trend decomposition
   procedure based on Loess*, J. Official Statistics.
9. 2024 Wayanad (Mundakkai–Chooralmala) landslides, 30 July 2024.
10. Mundakkai Hill InSAR Displacement Report (project deliverable,
    `Mundakkai_Hill_Landslide_Report.pdf`).
