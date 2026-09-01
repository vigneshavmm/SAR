# SAR Disaster Monitoring — Water/Flood + Landslide InSAR

Satellite-radar disaster monitoring over India, built on **Sentinel-1 SAR**.
Radar sees through cloud, smoke, and darkness, so it keeps working through the
monsoon — exactly when flood, drought, and landslide information matters most.

The repo holds **two independent analysis tracks** that share nothing but the
sensor, plus a **model-serving plan**:

| Track | Hazard | Radar product | What it measures |
| --- | --- | --- | --- |
| **1 · Water / flood** | Inundation, drought, reservoir drawdown | GRD (amplitude) | Where backscatter says *open water* |
| **2 · Landslide InSAR** | Slope deformation / subsidence | SLC (phase) | Millimetric ground motion along line-of-sight |

> GRD and SLC are **not interchangeable**: amplitude thresholding (Track 1)
> cannot do interferometry, and InSAR (Track 2) needs the phase that only SLC
> preserves. They are separate pipelines with separate Python environments.

---

## Documentation map — what's in this folder

| File | It is… |
| --- | --- |
| **README.md** | This front door — project overview + how the pieces fit. |
| **[CLAUDE.md](CLAUDE.md)** | Operational guide: exact commands, current `data/` layout, the two-venv rule, and the gotchas that bite. |
| **[analyze_water.md](analyze_water.md)** | Line-referenced walkthrough of the water/flood pipeline (`analyze_water.py`). |
| **Mundakkai_Hill_Landslide_Report.pdf** | The Track-2 deliverable — an 18-page, figure-heavy InSAR displacement report for Mundakkai Hill. |
| **landslide_text.txt** | Plain-text extract of that report (the PDF is a binary; this is the readable copy). |
| **[QWEN.md](QWEN.md)** | Plan + cost comparison for serving a Qwen3-1.7B model as an API. |
| **.claudeignore** | Keeps large/binary/generated files out of Claude's context. |

---

## Why SAR

- **All-weather, day/night** — C-band radar penetrates cloud and haze; optical
  sensors go blind under the SW monsoon.
- **Water is unmistakable in amplitude** — calm water is specular, reflecting
  radar *away* from the sensor, so it reads near-black. A backscatter threshold
  isolates open water (Track 1).
- **Phase encodes motion** — the interferometric phase between two SLC passes
  measures sub-centimetre ground displacement (Track 2).
- **Dense revisit** — Sentinel-1 repeats every ~12 days (tighter with two
  satellites), giving a steady cadence to track change.

---

## Track 1 — Water & flood (amplitude / GRD)

**AOI:** Tirupati & surrounds, Andhra Pradesh — bbox
`[79.298, 13.498, 79.552, 13.752]` (W,S,E,N), a ~27 × 28 km tile centred on the
city.
**Data:** 16 Sentinel-1 **RTC GRD** acquisitions, **2026-01-06 → 2026-06-30**
(dry winter → pre-monsoon → monsoon onset), ~12-day cadence with a tighter final
pass. Dual-pol **VV + VH**, 10 m, **EPSG:32644** (UTM 44N), linear γ⁰,
nodata `-32768`. ~823 MB in `data/Tirupati/` (from Microsoft Planetary
Computer, `sentinel-1-rtc`).

**Method — three detectors, one run:**
1. **Fixed −17 dB** VV cut (classic baseline).
2. **Per-scene adaptive Otsu** on the histogram's dark tail, clamped near −17 dB.
3. **SAFNet** — a Siamese Adaptive Fusion Network, self-supervised (no labels),
   giving a learned per-pixel change probability.

All water calls are **dual-pol** (`VV < −17 dB` **and** `VH < −24 dB`), which
rejects smooth *dry* land that fools VV alone; change masks are **speckle-cleaned**
(morphological opening + small-blob removal). Full logic in
**[analyze_water.md](analyze_water.md)**.

```bash
./venv/bin/python stac.py            # download/clip Sentinel-1 RTC scenes
./venv/bin/python analyze_water.py   # analyse (~6 min — SAFNet trains on CPU)
```

**Outputs → `outputs/`:**
- `water_area_timeseries.png` — open-water area (km²) per date, fixed vs adaptive.
- `flood_change_map.png` — dry baseline vs latest, new-water in red.
- `safnet_change_map.png` — threshold vs SAFNet, side by side, with F1/IoU.
- **GeoTIFFs** (georeferenced, EPSG:32644): `new_water_threshold.tif`,
  `new_water_safnet.tif`, `safnet_change_prob.tif` — drop straight into QGIS.

**What the 2026 stack shows (dual-pol, conservative "confident open water"):**
~**1.4 km²** in early January, drawn down to a ~**0.5 km²** minimum by late June,
then a sharp rebound to ~**2.4 km²** on the **June 30** pass as the monsoon onset
refills tanks. New-water-vs-dry-baseline is ~0 km²: the June water overlaps the
*permanent* reservoirs, so little qualifies as *newly* inundated — this window is
a **drought/drawdown** signal, not a flood scene.

---

## Track 2 — Landslide deformation (InSAR / SLC)

**Site:** **Mundakkai Hill, Wayanad, Kerala** — the slope above the
Mundakkai–Chooralmala **debris flow of 30 July 2024**.
**Pipeline:** `data/wayanad_insar/` — a numbered chain
`01_download_slc → 02_download_dem → 03_process_insar → 04_timeseries →
05_visualize_export`, driven by `run_pipeline.py`, tunables in `config.py`
(`AOI_BBOX = (76.00, 11.40, 76.25, 11.60)`, DESCENDING, rel-orbit 165, IW
subswath 2, VV, search 2024-05-01…2024-09-30). It downloads its own **SLC + DEM**
(Copernicus GLO-30) — the `data/Wayanad/` GRD scenes do **not** feed it.

**Chain:** DEM removes topographic phase → wrapped interferograms → coherence →
phase unwrapping → 1.5 km Gaussian high-pass detrend → cumulative displacement →
per-pixel least-squares **LOS velocity**.

**Result (see the PDF / `landslide_text.txt`):** 16 acquisitions, **Jan 10 → Aug 1
2024**, over a tight AOI `[76.12683, 11.45700, 76.14517, 11.47500]`; a concentric,
localized deformation core with LOS velocity **−30.01 to +23.67 mm/year**. Honest
caveats are stated: LOS-relative only, subsidence-vs-uplift sign not ground-truth
validated, ~8% border pixels masked.

```bash
./data/wayanad_insar/.venv-insar/bin/python data/wayanad_insar/run_pipeline.py
```

---

## Serving — Qwen3-1.7B as an API

**[QWEN.md](QWEN.md)** captures the plan to move the project's Qwen3-1.7B model
off a local machine into a callable API. Recommendation: **RunPod Serverless +
vLLM** (OpenAI-compatible `/v1/chat/completions`, scale-to-zero, ~5–10 s cold
start), with local+tunnel or on-device as zero-budget fallbacks.

---

## Repository layout

```
SAR/
├── analyze_water.py       # Track 1: threshold + adaptive-Otsu + SAFNet water/flood maps
├── stac.py                # Track 1: query Planetary Computer STAC, clip to bbox, save GeoTIFFs
├── data/                  # all heavy inputs (git/Claude-ignored)
│   ├── Tirupati/          #   16 RTC GRD scenes, VV+VH  (~823 MB)
│   ├── Wayanad/           #   Wayanad RTC GRD
│   ├── wayanad_insar/     #   Track 2 pipeline (01…05) + config.py + .venv-insar
│   ├── IND_shp/           #   GADM India admin boundaries
│   ├── Srikalahasti/      #   clipped rasters + AOI geojson
│   └── Tirupati District/ #   older Tirupati rasters
├── outputs/               # generated PNGs + GeoTIFFs
├── markdown/              # this documentation folder
├── venv/                  # Track-1 Python env (3.9.6)
└── .gitignore
```

---

## Environments — two venvs, don't cross them

| Track | Interpreter | Key packages |
| --- | --- | --- |
| **Water / flood** | `./venv/bin/python` (Py **3.9.6**) | pystac-client, planetary-computer, rasterio, numpy, matplotlib, **scipy** (speckle cleanup), **torch** (SAFNet, optional) |
| **Wayanad InSAR** | `./data/wayanad_insar/.venv-insar/bin/python` (Py **3.11.15**) | InSAR stack (see `data/wayanad_insar/requirements-insar.txt`) |

Track 1 degrades gracefully: without `torch` it still emits the two threshold
figures; without `scipy` it skips speckle cleanup. Both are guarded imports.

---

## Notes & limitations

- **Dual-pol is deliberately conservative.** Requiring VV **and** VH dark cuts
  false positives but undercounts thin/rough water — the km² are *confident* open
  water, not a maximal estimate. Tune `WATER_DB` / `WATER_DB_VH` per site.
- **SAFNet metrics are a PROXY.** With no labels, its precision/recall/F1/IoU are
  scored against the threshold method unless you supply a real mask via
  `REF_WATER`. For an operational model, fine-tune on labelled change pairs (e.g.
  **Sen1Floods11**) and score against held-out ground truth.
- **InSAR reports LOS motion only** — no vertical/horizontal separation, no
  ground-truth sign validation, and C-band coherence over Wayanad's vegetated,
  monsoon-wet slopes is the real limiting factor.
- **NRT ≠ real-time.** Detection lags each satellite pass plus processing; for a
  live feed, ingest the Copernicus/ASF NRT stream rather than the archive.
- **`data/` is large and regenerable** — kept out of version control; see
  `.gitignore` and `markdown/.claudeignore`.
