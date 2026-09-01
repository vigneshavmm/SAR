# Agent.md — SAR Disaster Monitoring

Operational guidance for working in this repo (commands, layout, gotchas). The
end-to-end vision lives in `markdown/ROOT.md`; each track has its own README.

## The repo has TWO independent tracks + shared docs


| Track                             | Purpose                                                                     | Data                                             | Code                                               | venv                                              |
| ----------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **Water / flood** (GRD amplitude) | Surface-water, drought, flood change-detection for**Tirupati** & coastal AP | `data/Tirupati/` (RTC GRD)                       | `analyze_water.py`, `stac.py` (repo root)          | **`venv`** (Py 3.9.6)                             |
| **Wayanad InSAR** (landslide)     | Interferometric processing for the**Wayanad / Mundakkai** landslide         | `data/Wayanad/` (RTC GRD) + SLC/DEM it downloads | `data/wayanad_insar/` (numbered `01…05` pipeline) | **`data/wayanad_insar/.venv-insar`** (Py 3.11.15) |

⚠️ **Two venvs — use the right one.** `./venv/bin/python` for the water track;
`./data/wayanad_insar/.venv-insar/bin/python` for anything under
`data/wayanad_insar/`. They are not interchangeable (different Python versions
and dependency stacks). This repo is **not a git repo** — don't run git
operations unless the user initialises one.

> **Layout note.** All heavy data + the InSAR track now live under **`data/`**
> (`data/Tirupati`, `data/Tirupati District`, `data/Wayanad`,
> `data/wayanad_insar`, `data/IND_shp`, `data/Srikalahasti`). Source scripts
> (`analyze_water.py`, `stac.py`) sit at the repo root; docs live in `markdown/`.

---

## Water / flood track (the focus of recent work)

- `stac.py` — download: query Planetary Computer STAC, clip to bbox, save GeoTIFFs.
- `analyze_water.py` — analyse: water-area time series + flood/new-water maps →
  PNGs in `outputs/`.

```bash
./venv/bin/python analyze_water.py     # main analysis (~6 min — see below)
./venv/bin/python stac.py              # (re)download scenes
```

- `analyze_water.py` reads **`SRC = "data/Tirupati"`** and writes to `outputs/`.
- **`analyze_water.py` takes ~6 minutes** — SAFNet (a self-supervised CNN) trains
  on CPU and dominates the runtime. It is **not hung**; use a long timeout.
- Fully **deterministic** (`SEED = 0`): reruns reproduce identical numbers, so a
  changed result means the code or data changed.
- Matplotlib uses the headless **`Agg`** backend — safe over SSH / no display.
- The deep block is **guarded**: if `torch` import fails the script still emits
  the two threshold figures. Keep it that way.

### Three detection methods in `analyze_water.py`

1. **Fixed −17 dB** VV threshold (classic, well-understood baseline).
2. **Per-scene adaptive Otsu** — Otsu on the **dark tail** of each scene, clamped
   to ±3 dB around −17. (Open water is only ~1 % of pixels, so whole-histogram
   Otsu would split land-vs-land and massively over-detect — hence dark-tail +
   clamp.) Both water curves are drawn on the time-series figure.
3. **SAFNet** (Siamese Adaptive Fusion Network) — self-supervised, no labels;
   multi-scale shared encoder + attention-gated fusion → change probability.

### Water-track data facts (they shape the code)

- RTC γ⁰ **linear** backscatter; **nodata = -32768**; **EPSG:32644** (UTM 44N);
  10 m pixels; `DS=4` → ~40 m (~695×680 per scene). Products are **GRD, not SLC**
  — no phase, so no interferometry on this track.
- `to_db()` converts to decibels and masks non-finite / non-positive to `NaN`;
  every comparison treats `NaN` as `False`.

---

## Wayanad InSAR track → `data/wayanad_insar/`

- Numbered pipeline `01_download_slc → 02_download_dem → 03_process_insar → 04_timeseries → 05_visualize_export`, driven by `run_pipeline.py`; tunables in
  `config.py`. Its own README: `data/wayanad_insar/WAYANAD_INSAR_README.md`.
- Targets the **Mundakkai / Chooralmala debris flow of 30 July 2024**.
  `config.py`: `AOI_BBOX = (76.00, 11.40, 76.25, 11.60)`, DESCENDING,
  rel-orbit 165, search 2024-05-01…2024-09-30, IW subswath 2, VV.
- Output report: `markdown/Mundakkai_Hill_Landslide_Report.pdf` (13 MB binary;
  plain-text extract in `markdown/landslide_text.txt`). Reports LOS velocity
  −30.01…+23.67 mm/yr over 16 Sentinel-1 acquisitions, Jan–Aug 2024.

```bash
./data/wayanad_insar/.venv-insar/bin/python data/wayanad_insar/run_pipeline.py
```

---

## Ancillary / shared

- `data/Srikalahasti/` — clipped/reprojected rasters + `Srikalahasti.geojson` (an AOI).
- `data/IND_shp/` — GADM India admin boundaries (shapefile).
- `markdown/ROOT.md` — end-to-end disaster-monitoring problem statement (three
  systems: data capturing & pipelining, change detection & labelling, inference
  & analysis). Contains embedded base64 images → large file.
- `markdown/README.md` — water/flood track README (Tirupati & coastal AP).
- `markdown/QWEN.md` — notes on hosting a Qwen3-1.7B model as an API.

---

## Gotchas

- **`stac.py` and `analyze_water.py` point at different areas.** `stac.py`'s
  `BBOX` is now the **Wayanad/Chooralmala** box
  (`[76.0850, 11.4920, 76.1650, 11.5120]`, 2024-05…08) writing to
  **`sar_downloads/`**, while `analyze_water.py` reads **`data/Tirupati/`**.
  Re-running `stac.py` as-is will **not** repopulate the water-analysis inputs —
  repoint its `BBOX` / `OUT_DIR` if that is the intent.
- **SAFNet metrics are a PROXY.** It has no labels, so its printed
  precision/recall/F1/IoU are scored against the threshold method unless you set
  `REF_WATER` to a real ground-truth mask GeoTIFF. Don't present them as validated.
- **`data/Wayanad/` holds GRD, but the InSAR pipeline needs SLC** —
  `data/wayanad_insar/` downloads SLC + DEM itself (`01_download_slc.py`,
  `02_download_dem.py`) into `data/Wayanad/insar/`. Don't assume the GRD scenes
  feed the interferometry.
- **`config.py` paths still resolve after the move.** It derives `INSAR_ROOT`
  relatively (`dirname(config.py)/../Wayanad/insar`); because `wayanad_insar/`
  and `Wayanad/` moved into `data/` **together**, paths correctly resolve to
  `data/Wayanad/insar`. The `# .../Downloads/SAR` comment on `PROJROOT` is now
  cosmetically stale (it actually resolves to `.../SAR/data`) — behaviour is fine.

## Conventions

- Tunables live as **module-level constants at the top** of each script
  (`SRC`, `OUT`, `DS`, `WATER_DB`, `REF_WATER`, SAFNet block; InSAR uses
  `data/wayanad_insar/config.py`). Change behaviour there, not in the body.
- Match the existing terse, comment-light style; keep helpers pure and near the
  top so the time-series loop and SAFNet can share them (e.g. `otsu`).
- Large/binary artifacts (`data/`, `outputs/`, both venvs, `*.tif`, `*.pdf`) stay
  out of git and out of Claude's context — see `.gitignore` (repo root) and
  `markdown/.claudeignore`.
- **File-location caveat:** `CLAUDE.md` and `.claudeignore` only take full effect
  from the **repo root**. Keeping them in `markdown/` means Claude won't
  auto-load them while working on root-level `analyze_water.py` / `stac.py` —
  place copies at the repo root if you want them always active.
