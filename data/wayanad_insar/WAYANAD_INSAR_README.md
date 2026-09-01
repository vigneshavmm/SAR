# Wayanad Landslide InSAR — Sentinel-1 displacement pipeline (PyGMTSAR)

Line-of-sight **ground-displacement** analysis for the **Chooralmala /
Mundakkai, Wayanad** region (Kerala, India), around the catastrophic debris
flow of **30 July 2024**, using **Sentinel-1 SAR interferometry (InSAR)**
driven entirely from Python via **PyGMTSAR**.

This is a separate track from the repo's flood/water backscatter work: it needs
**phase**, not amplitude.

---

## ⚠️ Read this first — two hard realities

### 1. GRD ≠ SLC. The scenes already on disk cannot be used for InSAR.
The `../Wayanad/S1A_IW_GRDH_1SDV_..._rtc_vv.tif` files (from Microsoft
Planetary Computer) are **GRD — ground-range, *detected amplitude* only**. The
interferometric **phase has been discarded**, so InSAR is impossible with them.
InSAR needs **IW SLC** (`S1A_IW_SLC__1SDV_...`), which PC does **not** host —
Step 1 downloads SLC from **ASF / NASA Earthdata** instead.

### 2. PyGMTSAR is Python, but it wraps native binaries.
"Python only" holds for the *workflow* — every step here is Python. But
PyGMTSAR shells out to the **GMTSAR** C toolchain (co-registration) and
**snaphu** (phase unwrapping). You install those once (§Setup). The download,
DEM, multilook, Goldstein, LOS, SBAS and export steps are pure Python and need
no binaries.

### 3. Scientific honesty about what InSAR can deliver here
C-band (5.6 cm) Sentinel-1 over the **steep, densely-vegetated Western Ghats**
during **peak SW monsoon** is close to the worst case for InSAR:

- **Severe decorrelation** — 12-day coherence over forest routinely < 0.2;
  most of the scene is noise.
- **The slide itself is not recoverable as a displacement field** — a rapid,
  metres-scale, discontinuous failure fully decorrelates; it shows up as a
  **coherence-loss hole / amplitude-change anomaly**, not a smooth phase
  gradient. DInSAR only measures slow mm–cm creep.
- **Monsoon atmosphere** adds a strong, deformation-like phase screen.

**Realistic deliverable:** coherence-masked LOS displacement where coherence
survives (rock outcrops, roads, riverbeds, built-up patches) plus
coherence/amplitude-change **damage mapping** of the scar. For genuine slow
deformation, run the same pipeline on a **dry-season (Nov–Mar)** stack, and use
**both ascending and descending** tracks (steep slopes blind to one geometry
are seen by the other). The pipeline applies a coherence mask (`COH_MIN`) so
you never over-interpret decorrelation noise as motion.

---

## The 9-step workflow → code map

| # | Step | PyGMTSAR calls | File |
|---|------|----------------|------|
| 1 | Download Sentinel-1 SLC | `asf.geo_search` + `.download`; `S1.download_orbits`; `S1.scan_slc` | `01_download_slc.py` |
| 2 | Download DEM | `Tiles().download_dem(..., provider='GLO')` | `02_download_dem.py` |
| 3 | Init project | `Stack(...)`, `set_scenes`, `compute_reframe`, `load_dem` | `03_process_insar.py` |
| 4 | Co-register | `compute_align`, `compute_geocode`, `baseline_table` | `03_process_insar.py` |
| 5 | Interferograms | `sbas_pairs`, `psfunction`, `compute_interferogram_multilook` | `03_process_insar.py` |
| 6 | Filter + unwrap | Goldstein (inline via `psize`) + `unwrap_snaphu` | `03_process_insar.py` |
| 7 | Phase → displacement | `los_displacement_mm` | `03_process_insar.py` |
| 8 | Time-series (SBAS) | `lstsq`, `los_displacement_mm`, `velocity`, `rmse` | `04_timeseries.py` |
| 9 | Visualize + export | `ra2ll`, `plot_*`, `export_geotiff`, `export_netcdf` | `05_visualize_export.py` |

`config.py` holds every tunable (AOI, track, dates, filter/SBAS/mask knobs).
`common.py` has the credential/binary/`open_project` helpers.
`run_pipeline.py` runs steps 1→9 in order.

---

## Setup

### A. Python environment (3.10+)  ✅ already set up
The repo's original `venv` is Python 3.9; **PyGMTSAR requires ≥3.10**. A ready
env has been created at **`.venv-insar`** (Homebrew Python **3.11**, with
`pygmtsar==2025.4.8.post1` + `asf_search` + the full scientific stack). Use it
directly: `./.venv-insar/bin/python ...`.

To recreate it from scratch:

```bash
brew install python@3.11
/opt/homebrew/bin/python3.11 -m venv .venv-insar
./.venv-insar/bin/pip install -U pip
./.venv-insar/bin/pip install -r wayanad_insar/requirements-insar.txt
```

Note: avoid Python 3.14 — `numba`/`pygmtsar` wheels lag the newest releases; 3.11
is the safe target.

### B. Native binaries — GMTSAR + snaphu + GMT (macOS)
PyGMTSAR shells out to these. Primary path — the InSAR Homebrew tap (prebuilt,
works on Apple Silicon, GMTSAR ≥ 6.7 which bundles snaphu):

```bash
xcode-select --install
brew tap dsandwell/homebrew-insar
brew install gmtsar          # pulls GMT; provides make_s1a_tops, esarp, snaphu, ...
```

Add GMTSAR to the shell you actually run (zsh login default; prefix is
`/opt/homebrew` on Apple Silicon, `/usr/local` on Intel):

```bash
# ~/.zshrc
export GMTSAR=/opt/homebrew/share/gmtsar
export PATH="$GMTSAR/bin:$PATH"
```

**Verify the whole toolchain** (all three must resolve):

```bash
which make_s1a_tops esarp snaphu gmt
```

Fallbacks if the native build fights you:
- **MacPorts:** `sudo port install gmtsar` (pulls gmt6).
- **Docker:** run the maintainer's Linux image (GMTSAR+snaphu+GMT+PyGMTSAR
  preinstalled) and mount this repo as a volume. Check Docker Hub for the
  current image name (`mobigroup/pygmtsar` / `pechnikov/pygmtsar`).

Pitfalls: GMT must be 6.x (a stale GMT5 on PATH breaks GMTSAR); GMT needs
ghostscript for PostScript plotting; keep Homebrew arch consistent (no
arm64/x86_64 mix under Rosetta).

### C. NASA Earthdata credentials (for SLC download)
1. Register (free): <https://urs.earthdata.nasa.gov/>
2. Log into ASF Vertex **once** (<https://search.asf.alaska.edu/>) and **accept
   the ASF data-access EULA** — until you do, downloads return HTTP 401/403.
3. Store credentials in `~/.netrc` (`chmod 600 ~/.netrc`):
   ```
   machine urs.earthdata.nasa.gov
       login YOUR_EARTHDATA_USERNAME
       password YOUR_EARTHDATA_PASSWORD
   ```
   (Or export `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.)

---

## Usage

```bash
# Full pipeline for the primary track (165), all 9 steps:
./.venv-insar/bin/python wayanad_insar/run_pipeline.py

# Fast trial — download only the co-event pair (2 scenes) and process it:
WAYANAD_COEVENT_ONLY=1 ./.venv-insar/bin/python wayanad_insar/run_pipeline.py

# Re-run only later steps (e.g. after tuning filter/mask params):
./.venv-insar/bin/python wayanad_insar/run_pipeline.py 3 4 5

# Process the independent cross-check track (63) into its own project dir:
WAYANAD_REL_ORBIT=63 ./.venv-insar/bin/python wayanad_insar/run_pipeline.py
```

Each step is also runnable standalone: `python wayanad_insar/03_process_insar.py`.

### Data volume & time
Each IW SLC is ~4–8 GB. The full 2024-05→09 window is ~8 scenes per track
(~40+ GB). Co-event-only is 2 scenes. Co-registration + unwrapping the stack
takes tens of minutes to hours depending on CPU/RAM.

---

## Wayanad tracks & dates (verified from the on-disk acquisition times)

Two **descending** Sentinel-1 tracks cover Wayanad; **process each separately**.

| Rel. orbit | Pass (UTC) | Co-event 12-day pair | SBAS window dates (2024) |
|-----------:|-----------|----------------------|--------------------------|
| **165** (primary) | ~00:40 | **2024-07-20 / 2024-08-01** (10 d before / 2 d after) | 06-02, 06-14, 06-26, 07-08, 07-20, 08-01, 08-13, 08-25 |
| **63** (cross-check) | ~00:49 | **2024-07-25 / 2024-08-06** | 06-07, 06-19, 07-01, 07-13, 07-25, 08-06, 08-18, 08-30 |

Only Sentinel-1A was operational in mid-2024 (S1B failed Dec 2021; S1C came
online in 2025) → strict **12-day** repeat per track.

---

## Outputs (`../Wayanad/insar/outputs/`)

- `track165_mean_coherence.png` — where interferometric signal actually exists.
- `track165_velocity_los.png` — coherence-masked mean LOS velocity (mm/yr).
- `track165_displacement_los.png` — cumulative SBAS displacement time series.
- `track165_velocity_los_mm_yr.tif` — geocoded GeoTIFF (WGS84).
- `track165_timeseries_los_mm.nc` — geocoded NetCDF time series (WGS84).

Sign convention: PyGMTSAR LOS displacement — positive = motion **toward** the
satellite. Convert to slope-parallel motion only after decomposing asc+desc.

---

## Notes / caveats for the code

Method names were verified against **PyGMTSAR 2025.4.8.post1** source. A few
calls are marked `[VERIFY]` in comments — tune them against your installed
build and the observed coherence:

- Interferogram/SBAS knobs (`WAVELENGTH`, `PSIZE`, `SBAS_DAYS`, `SBAS_METERS`)
  are sensible defaults, not tuned runs.
- `open_project()` reopens a saved project by re-attaching scenes + DEM; if
  your build needs a different reopen sequence, fix it in that one helper.
- Export/plot helper signatures have generic fallbacks so Step 9 still produces
  output if a helper differs across versions.
- Do **not** install `insardev-pygmtsar` — it's a separate Zarr-based rewrite
  with different method names.
