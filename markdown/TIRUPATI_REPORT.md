# Tirupati Surface-Water & Flood Monitoring — Project Report

**Site:** Tirupati district & surrounds, Andhra Pradesh, India
**Sensor:** Sentinel-1 C-band SAR (RTC GRD, amplitude)
**Date:** 2026-07-16 · **Status:** Operational — implemented, verified, run end-to-end

---

## Abstract

This report presents a near-real-time surface-water, drought, and flood
change-detection analysis for the Tirupati district using **Sentinel-1 Synthetic
Aperture Radar (SAR)** amplitude imagery. Because radar images through cloud and
darkness, the workflow keeps operating through the South-West monsoon, when
optical sensors fail. A 16-date Sentinel-1 time series (2026-01-06 → 2026-06-30)
is processed by a single deterministic pipeline that classifies **open water**
from backscatter and detects **new inundation** against a dry-season baseline,
using three complementary detectors: a fixed decibel threshold, a per-scene
adaptive Otsu threshold, and a self-supervised deep change detector (**SAFNet**).
Detection is made **dual-polarization** (VV ∧ VH) to reject dry smooth surfaces,
and outputs are exported as georeferenced GeoTIFFs. Over the analysis window the
pipeline resolves the district's hydrological cycle — confident open water
falling from ~**1.4 km²** in January to a ~**0.5 km²** minimum by late June, then
rebounding to ~**2.4 km²** at monsoon onset — a directly usable drought/drawdown
signal.

---

## Area of Interest

| Parameter | Value |
| --- | --- |
| Location | Tirupati district & coastal Andhra Pradesh |
| Bounding box (W,S,E,N) | `79.298, 13.498, 79.552, 13.752` (EPSG:4326) |
| Extent | ~27 × 28 km tile centred on Tirupati city |
| Projection | **EPSG:32644** (UTM zone 44N) |
| Terrain / hydrology | Semi-arid plateau; irrigation tanks and reservoirs subject to strong seasonal drawdown |

The AOI is deliberately compact so the full multi-detector pipeline (including the
CNN) runs at district scale in minutes.

---

## Pre-processing and Auxiliary Data

| Component | Detail |
| --- | --- |
| **Product** | Sentinel-1 **RTC GRD** (Radiometrically Terrain-Corrected γ⁰, linear backscatter), IW mode. |
| **Source** | Microsoft Planetary Computer (`sentinel-1-rtc` collection), signed access. |
| **Scenes** | **16 acquisitions, 2026-01-06 → 2026-06-30**, ~12-day cadence (dry winter → pre-monsoon → monsoon onset). |
| **Polarizations** | Dual-pol **VV + VH** (~823 MB total in `data/Tirupati/`). |
| **Native grid** | 10 m pixels; nodata = −32768. |
| **Grid alignment** | Every scene (VV and paired VH) is warped onto **one common ~40 m reference grid** (`WarpedVRT`, downsample ×4), co-registering all dates and guarding array stacking against footprint drift. |
| **Radiometric prep** | Average resampling on read (cheap speckle reduction); linear γ⁰ → decibels (`10·log₁₀`); nodata / non-positive samples masked to `NaN` so invalid pixels are never counted as water. |

No DEM or external mask is required for the amplitude workflow; the reference-grid
transform is retained so all raster outputs are georeferenced (EPSG:32644).

---

## Methodology and Techniques

**Dual-polarization water primitive.** A pixel is classed **open water** only when
dark in **both** polarizations — `VV < −17 dB` **and** `VH < −24 dB`. Requiring
co- *and* cross-pol darkness rejects smooth dry surfaces (roads, runways, dry
sand) that fool a VV-only test, cutting false positives.

**Three complementary detectors:**
1. **Fixed −17 dB** VV threshold — the classical, well-understood baseline.
2. **Per-scene adaptive Otsu** — Otsu computed on the *dark tail* of each scene's
   histogram (open water is ~0.1 % of pixels, so whole-histogram Otsu would
   over-detect), clamped to ±3 dB around −17 dB so noise cannot drift the cut.
3. **SAFNet — Siamese Adaptive Fusion Network** (self-supervised, no labels): a
   shared-weight multi-scale encoder embeds both dates; per-scale absolute
   difference features are combined by a learned **per-pixel softmax attention**
   into a change probability. Trained on **reconstruction** (learn SAR texture) +
   **synthetic copy-paste change** (pasted content = change, added speckle = no
   change), so it learns SAR-noise invariance rather than the −17 dB rule.

**Change detection.** New water = *open water now (dual-pol)* **and** *dry in the
baseline*. Both the threshold and SAFNet change masks are **speckle-cleaned**
(morphological opening + removal of connected components < 5 px).

---

## Core SAR Processing Chain

*(This track uses SAR **amplitude** (GRD), so there is no interferometric phase —
the "core chain" is amplitude classification + change detection, the amplitude
analogue of an InSAR chain.)*

```
Discover        glob VV scenes, parse YYYYMMDD, sort chronologically
   │
Align + dB      WarpedVRT → common ~40 m grid → to_db → nodata/NaN   (VV and VH)
   │
Water mask      dual-pol:  VV < −17 dB  ∧  VH < −24 dB               (per scene)
   │
Time series     count water px → km²  (fixed cut + adaptive Otsu)    → figure 1
   │
Baseline        mean dB of first ~8 (dry) scenes  vs  latest scene
   │
New water (A)   water now (dual-pol)  ∧  dry in baseline  → despeckle → figure 2 + GeoTIFF
   │
SAFNet (B)      self-supervised train → change probability → Otsu cut
                → changed ∧ water now → despeckle → figure 3 + GeoTIFFs
   │
Validate        SAFNet scored vs threshold (PROXY) or vs REF_WATER (ground truth)
```

**SAFNet architecture:** multi-scale Siamese encoder (channels 1→16→32→64,
max-pool between scales); a decoder used only for reconstruction pre-training; and
an adaptive-fusion head that projects each scale's `|fA − fB|`, upsamples, and
combines them via per-pixel softmax attention into a change logit. Losses:
reconstruction (MSE) + synthetic-change (BCE), optimised jointly.

Water-area conversion uses `px_area = (10·DS)² / 1e6` km² per pixel; the whole
chain is a pure function of the input scenes and the tunable constants.

---

## Implementation

```
SAR/
├── stac.py              # download: query Planetary Computer STAC, clip to bbox, save GeoTIFFs
├── analyze_water.py     # analyse: 3 detectors → figures + georeferenced GeoTIFFs
├── data/Tirupati/       # 16 RTC GRD scenes (VV + VH)  (~823 MB)
├── outputs/             # generated PNGs + GeoTIFFs
└── venv/                # Python 3.9.6
```

- **Environment:** `venv` (Python 3.9.6) — pystac-client, planetary-computer,
  rasterio, numpy, matplotlib, **scipy** (speckle cleanup), **torch** (SAFNet,
  optional).
- **Deterministic:** `SEED = 0` → identical reruns; a changed result means the
  code or data changed.
- **Headless:** Matplotlib `Agg` backend (safe over SSH / CI).
- **Graceful degradation:** guarded imports — without `torch` the two threshold
  figures still emit; without `scipy` speckle cleanup is skipped.
- **Configuration-driven:** all tunables are module-level constants at the top of
  `analyze_water.py`; behaviour is changed there, not in the body.
- **Runtime:** ≈ 6 minutes, dominated by SAFNet training on CPU.

---

## Result and Outcome

**Surface-water area — the hydrological cycle** (dual-pol "confident open water"):

| Epoch | Open water |
| --- | --- |
| 2026-01-06 | ~1.4 km² |
| spring (drawdown) | declining |
| 2026-06-23 (minimum) | ~0.5 km² |
| 2026-06-30 (monsoon onset) | ~2.4 km² (sharp rebound) |

The January→June drawdown as tanks and reservoirs dry, followed by a monsoon-onset
refill on the final pass, is itself a **drought-monitoring signal**: a flat or
continued decline into June would flag water stress for the district.

**New-water (flood) vs dry baseline:** ~0 km² (16 px by threshold, 25 px by
SAFNet). This is **correct** for the window — June water overlaps the *permanent*
reservoirs, so almost nothing is *newly* inundated; this is a drawdown period, not
a flood event. The dual-pol gate keeps the estimate conservative.

**SAFNet vs threshold — scored as a labelled PROXY (not ground truth):**
precision 0.64 · recall 1.00 · F1 0.78 · IoU 0.64 (SAFNet's 25 px is a superset
of the threshold's 16 px).

**Outputs (`outputs/`):** `water_area_timeseries.png`, `flood_change_map.png`,
`safnet_change_map.png`, plus georeferenced GeoTIFFs `new_water_threshold.tif`,
`new_water_safnet.tif`, `safnet_change_prob.tif` (EPSG:32644, QGIS-ready).

**Outcome:** an operational, reproducible surface-water and flood change-detection
capability at ~40 m district scale, with three complementary detectors and
GIS-ready products; the 2026 run cleanly captures the district's reservoir
drawdown and monsoon-onset refill.

---

## Limitation

- **Dual-pol is deliberately conservative** — reported km² are *confident* open
  water, not a maximal estimate; thin or wind-roughened water may be undercounted.
- **SAFNet is unsupervised and unvalidated on ground truth** — its precision/
  recall/F1/IoU are a **proxy** against the threshold method until a real mask is
  supplied via `REF_WATER`; synthetic copy-paste change only approximates real
  hydrology.
- **The fixed −17 dB cut can be biased** by wind-roughened water and very smooth
  dry surfaces; the adaptive Otsu curve mitigates this partially, but a DEM/HAND or
  permanent-water mask would further exclude terrain that cannot hold water.
- **Amplitude-only.** GRD carries no phase, so no interferometric deformation or
  fine-scale coherence information is available on this track.
- **NRT ≠ real-time** — alerts lag by the next satellite pass plus processing;
  an operational feed should ingest a Copernicus/ASF NRT stream rather than the
  (multi-day-latency) archive.

---

## References

1. Microsoft Planetary Computer — Sentinel-1 RTC (`sentinel-1-rtc`) collection.
2. Sentinel-1 mission — ESA / Copernicus Programme.
3. Otsu, N. (1979) — *A threshold selection method from gray-level histograms*,
   IEEE TSMC.
4. Siamese / attention-fusion change-detection networks for remote sensing
   (SAFNet design lineage).
5. Bonafilia, D. et al. (2020) — *Sen1Floods11*: a georeferenced dataset for
   Sentinel-1 flood mapping (ground-truth / future validation).
6. Radiometric Terrain Correction (RTC) γ⁰ processing background.
