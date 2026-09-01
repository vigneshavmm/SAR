# SAR Analysis Toolkit — Water & Landslide Detection

Multi-track Sentinel-1 SAR analysis system for **water/flood mapping** and **InSAR-based ground displacement monitoring** across India. Built for rapid response to hydrological and geotechnical hazards.

---

## 📍 Regions & Datasets

| Region | Track | Purpose | Data Source |
|--------|-------|---------|------------|
| **Tirupati, Andhra Pradesh** | Water/Flood | Backscatter-based water extent detection | Sentinel-1 GRD (Planetary Computer) |
| **Nepal (Kathmandu Valley)** | Water/Flood | Pre/post flood comparison (monsoon) | Sentinel-1 GRD (Planetary Computer) |
| **Wayanad, Kerala** | InSAR / Landslide | Line-of-sight displacement & damage mapping | Sentinel-1 SLC (ASF Earthdata) |
| **Srikalahasti, Andhra Pradesh** | Reference | Boundary / validation geometries | GeoJSON |

---

## 🌊 Track 1: Water & Flood Detection

**Amplitude-based water classification from Sentinel-1 Ground Range Detected (GRD) scenes.**

### Key Scripts

- **`stac.py`** — Download Sentinel-1 GRD for Tirupati region from Planetary Computer
- **`stac2.py`** — Download Sentinel-1 GRD for Nepal pre/post flood comparison (monsoon events)
- **`analyze_water.py`** — VV/VH thresholding, morphological filtering, spatial-temporal change detection, optional SAFNet neural refinement
- **`safnet.py`** — Self-supervised Siamese Adaptive Fusion network for pixel-level water confidence (optional; improves speckle robustness)
- **`place.py`** — Validate results against reference water masks (e.g., ground-truth inundation polygons)

### Workflow

```bash
# 1. Download Sentinel-1 GRD data to SAR catalog
python stac.py

# 2. Run water detection (threshold-based + optional SAFNet refinement)
python analyze_water.py

# 3. (Optional) Validate against reference mask
python place.py
```

### Outputs

- **GeoTIFFs** in `outputs/`
  - `water_extent.tif` — Binary water mask (threshold or SAFNet)
  - `water_confidence.tif` — Continuous 0–1 scores (SAFNet)
  - `change_map.tif` — New inundation / recession per interval
- **Plots** — Time-series extent trends, before/after comparisons

### Configuration

Edit settings in `analyze_water.py`:

```python
SRC = "data/Tirupati"        # Input scene directory
WATER_DB = -17.0             # VV co-pol threshold (dB); < this = water
USE_VH = True                # Require cross-pol confirmation (rejects smooth bare soil)
WATER_DB_VH = -24.0          # VH cross-pol threshold
DS = 4                        # Downsample factor for speed (40 m pixels)
```

### Algorithm Notes

Open water has distinct electromagnetic signature in SAR:

- **Co-pol VV (vertical-vertical)** — Open water → specular reflection (smooth surface) → very **dark** (< –17 dB); rough land/vegetation → diffuse scattering → brighter
- **Cross-pol VH (vertical-horizontal)** — Water is extremely poor at cross-pol (EM waves don't depolarize) → **very dark** (< –24 dB); wet bare soil & vegetation → brighter VH due to depolarization
- **Dual-pol logic** — Require **both** VV and VH dark to classify as water (rejects false positives: smooth, wet bare soil tricks VV alone)
- **Morphology** — Remove speckle blobs < 5 pixels; close small gaps; retain structure
- **SAFNet** (optional) — Self-supervised CNN pre-trained on Sentinel-1 dual-pol pairs; learns adaptive fusion weights to further suppress speckle and smooth-soil noise

---

## 🌍 Track 2: InSAR & Ground Displacement

**Line-of-sight displacement monitoring using Sentinel-1 Single-Look Complex (SLC) data and PyGMTSAR.**

⚠️ **Note:** GRD data (from Planetary Computer) cannot be used for InSAR — SLC is required and must be downloaded separately.

### Key Scripts

- **`data/wayanad_insar/01_download_slc.py`** — Fetch Sentinel-1 IW SLC from ASF Earthdata
- **`data/wayanad_insar/02_download_dem.py`** — Acquire GLO-30 DEM for co-registration & geocoding
- **`data/wayanad_insar/03_process_insar.py`** — Stack build, co-registration, interferogram formation, filtering, phase unwrapping, LOS conversion
- **`data/wayanad_insar/04_timeseries.py`** — SBAS inversion, time-series stack, velocity, coherence, PSI filtering
- **`data/wayanad_insar/05_visualize_export.py`** — Plotting & export (GeoTIFF, NetCDF, Zarr)
- **`insar_advanced.py`** — Standalone time-series engine with GPU acceleration, STL decomposition, cloud-native output (Zarr/S3)

### Workflow

```bash
cd data/wayanad_insar

# One-shot pipeline (steps 1–9)
python run_pipeline.py

# Or step-by-step
python 01_download_slc.py
python 02_download_dem.py
python 03_process_insar.py
python 04_timeseries.py
python 05_visualize_export.py

# Advanced post-processing (SBAS + STL + GPU)
cd ../..
python insar_advanced.py --input data/wayanad_insar/unwrap.nc \
                         --out outputs/ts.zarr \
                         --device mps --robust --stl
```

### Configuration

Edit `data/wayanad_insar/config.py`:

```python
# Area of interest (W, S, E, N)
AOI_BBOX = [76.08, 11.49, 76.17, 11.52]

# Sentinel-1 track (ascending=1, descending=2)
TRACK = 1

# Acquisition range
START_DATE = "2024-05-01"
END_DATE = "2024-08-31"

# Co-registration baseline thresholds
BASELINE_MAX_SPATIAL = 300    # m
BASELINE_MAX_TEMP = 180       # days (12-day repeat + buffer)

# Coherence masking
COH_MIN = 0.3                 # Mask pixels < this coherence

# Phase unwrapping
UNWRAP_METHOD = "snaphu"      # Default; set to "skip" for testing
```

### Dependencies

- **PyGMTSAR** — Python wrapper around GMTSAR (C-based InSAR processor)
- **GMTSAR + snaphu + GMT** — Native binaries (install via Homebrew on macOS)
- **asf_search** — ASF Earthdata authentication & SLC discovery
- **xarray + Zarr** — Cloud-native n-dimensional arrays

See `data/wayanad_insar/WAYANAD_INSAR_README.md` for detailed setup & caveats.

### Output Products

| File | Description | Units | Typical Range |
|------|-------------|-------|---|
| `los_displacement_mm.tif` | LOS displacement (geocoded, coherence-masked) | mm | ±100 |
| `los_velocity_mm_yr.tif` | LOS velocity from SBAS inversion | mm/year | ±50 |
| `temporal_coherence.tif` | SBAS inversion quality per pixel | 0–1 | > 0.3 reliable |
| `incidence_angle.tif` | SAR look angle (needed for vertical decomposition) | degrees | ~30–45 |
| `unwrap.nc` | Unwrapped phase stack (intermediate; large) | rad/2π | –1 to +1 |

### Understanding Output

- **Negative LOS displacement** = motion **toward** satellite; **positive** = **away from** satellite
- **Coherence < 0.2** over vegetation = noise; mask these pixels out (already done in export)
- **Displacement anomalies** at landslide scar = coherence loss (phase decorrelation), not smooth motion
- **Velocity ambiguity over single geometry** — ascending tracks blind to pure E–W motion; ascending + descending needed for 3-D decomposition

### Limitations & Best Practices

**C-band (5.6 cm) Sentinel-1 over monsoon-season Western Ghats is a challenging case:**

| Limitation | Workaround | Impact |
|------------|-----------|--------|
| **Dense vegetation decorrelation** | Use dry season (Nov–Mar); apply temporal baseline shortening | Coherence drops below 0.2 in summer; SBAS baseline tuning helps marginally |
| **Steep terrain ambiguity** | Combine ascending + descending tracks; use unwrapped phase for local analysis | Single-geometry LOS velocity is blind to E–W motion (30° incidence angle) |
| **Monsoon atmospheric screen** | Use multi-pass phase-linking; compute atmospheric estimates from ECMWF | Strong phase perturbations that mimic deformation; seasonal models available |
| **Rapid failure not recoverable** | Use coherence-loss / amplitude-change mapping as damage indicator (not displacement) | Meters-scale discontinuous failures fully decorrelate; appear as coherence holes, not phase gradients |
| **Phase unwrapping ambiguity** | Inspect intermediate wrapped phases; use multi-scale unwrapping (snaphu residue classification) | Snaphu may fail or produce artifacts in low-coherence zones; manual inspection recommended |

**Realistic Deliverables:**
- ✅ **Coherence-masked LOS displacement** — only where temporal coherence > 0.3 (rock, roads, riverbeds)
- ✅ **Coherence-loss anomaly map** — damage extent of the scar
- ✅ **Velocity field** — creep on stable slopes outside the failure zone
- ❌ **NOT** — smooth displacement gradient through the landslide (will be noise)

---

## 🛠 Setup

### Step 1: Clone Repository

```bash
git clone git@github.com:vigneshavmm/SAR.git
cd SAR
```

### Step 2: Water Track Environment (Python 3.8+)

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### Step 3: InSAR Track Environment (Python 3.10+)

```bash
# Install Python 3.11 (if not available)
brew install python@3.11

# Create isolated venv
/opt/homebrew/bin/python3.11 -m venv .venv-insar
source .venv-insar/bin/activate
pip install -U pip
pip install -r data/wayanad_insar/requirements-insar.txt
```

### Step 4: Native Binaries (InSAR only)

**macOS (Apple Silicon & Intel):**
```bash
xcode-select --install
brew tap dsandwell/homebrew-insar
brew install gmtsar  # Includes snaphu, GMT
```

**Linux (Ubuntu 20.04+):**
```bash
sudo apt-get update
sudo apt-get install gmtsar snaphu gmt
```

**Verify installation:**
```bash
which make_s1a_tops esarp snaphu gmt
```

---

## 📊 Project Structure

```
SAR/
├── stac.py                      # Water track: Download STAC data (Tirupati)
├── stac2.py                     # Water track: Pre/post flood comparison (Nepal)
├── analyze_water.py             # Water track: Detection & classification
├── safnet.py                    # Water track: SAFNet neural model (optional)
├── insar_advanced.py            # InSAR track: Advanced time-series (GPU, STL)
├── place.py                     # Validation utilities
│
├── requirements.txt             # Water track dependencies (Python 3.8+)
├── .gitignore                   # Git exclusions (data, outputs, cache)
├── README.md                    # This file
│
├── data/
│   ├── Srikalahasti/
│   │   └── Srikalahasti.geojson # Reference boundary polygon (tracked)
│   ├── Tirupati/                # Water track GRD scenes (ignored: large)
│   ├── Tirupati District/       # Water track GRD scenes (ignored: large)
│   ├── Wayanad/                 # RTC reference data (ignored: large)
│   └── wayanad_insar/           # InSAR pipeline (all tracked)
│       ├── 01_download_slc.py      # Step 1: Fetch SLC from ASF
│       ├── 02_download_dem.py      # Step 2: GLO-30 DEM download
│       ├── 03_process_insar.py     # Steps 3–7: Stack, coregister, unwrap
│       ├── 04_timeseries.py        # Step 8: SBAS inversion & velocity
│       ├── 05_visualize_export.py  # Step 9: Plotting & export
│       ├── run_pipeline.py         # Runs steps 1–9 in sequence
│       ├── config.py               # Configurable parameters
│       ├── common.py               # Shared helpers & credentials
│       ├── requirements-insar.txt  # InSAR dependencies (Python 3.10+)
│       └── WAYANAD_INSAR_README.md # Detailed InSAR workflow & caveats
│
├── outputs/                     # Analysis results (ignored: regenerable)
├── markdown/                    # Markdown reports & documentation (tracked)
├── sar_downloads_nepal/         # Downloaded SAR scenes (ignored: large)
└── docs/                        # Additional reference docs (if any)
```

---

## 🚀 Quick Start

### Water Detection (Tirupati, ~5 min)

```bash
# Activate environment
source venv/bin/activate

# 1. Download latest Sentinel-1 GRD scenes (30–60 min, ~100–200 MB)
python stac.py

# 2. Detect water extent (5–10 min)
python analyze_water.py

# 3. Inspect results
ls -lh outputs/
open outputs/water_extent.tif  # or use QGIS
```

**Expected outputs:**
- `water_extent.tif` — Binary water mask
- `water_confidence.tif` — Continuous 0–1 scores
- `comparison.png` — Before/after plots

### InSAR Displacement (Wayanad, ~2–4 hours)

```bash
# Activate InSAR environment
source .venv-insar/bin/activate

# Full automated pipeline (steps 1–9)
cd data/wayanad_insar
python run_pipeline.py

# Or run individual steps
python 01_download_slc.py     # ~30 min
python 02_download_dem.py     # ~5 min
python 03_process_insar.py    # ~1.5 hours (includes phase unwrapping)
python 04_timeseries.py       # ~15 min
python 05_visualize_export.py # ~10 min

# Return to repo root for advanced processing
cd ../..

# (Optional) Advanced time-series with GPU & STL
python insar_advanced.py --input data/wayanad_insar/unwrap.nc \
                         --out outputs/ts.zarr \
                         --device mps --robust --stl
```

**Expected outputs:**
- `los_displacement_mm.tif` — Line-of-sight displacement
- `los_velocity_mm_yr.tif` — Annual LOS velocity
- `temporal_coherence.tif` — Inversion quality mask
- `unwrap.nc` — Intermediate unwrapped phase stack

---

## 📋 Requirements

**Water track:**
- Python 3.8+
- `pystac-client`, `planetary-computer`, `rasterio`, `numpy`, `scipy`, `matplotlib`

**InSAR track:**
- Python 3.10+
- `pygmtsar >= 2025.4.8`, `asf_search`, `xarray`, `zarr`, `dask`
- macOS/Linux with GMTSAR + snaphu binaries

See `requirements.txt` (water) and `data/wayanad_insar/requirements-insar.txt` (InSAR) for pinned versions.

---

## 🔐 Credentials & Authentication

### Planetary Computer (Water Track)

- **Free** — No registration required
- Used by `stac.py` for Sentinel-1 GRD downloads
- Public catalog with no auth overhead

### ASF Earthdata (InSAR Track)

- **Free NASA account** required for SLC download
- Register: https://urs.earthdata.nasa.gov/
- Store credentials in `~/.netrc`:
  ```bash
  machine urs.earthdata.nasa.gov
  login YOUR_USERNAME
  password YOUR_PASSWORD
  ```
- Or set environment variables:
  ```bash
  export ASF_USERNAME="your_username"
  export ASF_PASSWORD="your_password"
  ```

---

## 🐛 Troubleshooting

### Water Track Issues

| Problem | Solution |
|---------|----------|
| `pystac_client` import fails | Activate venv: `source venv/bin/activate` |
| `rasterio` installation fails (macOS) | Install GDAL: `brew install gdal` |
| No scenes found in STAC query | Check BBOX order (W, S, E, N) and date range validity |
| Out of memory on `analyze_water.py` | Increase `DS` (downsample factor) to reduce resolution |

### InSAR Track Issues

| Problem | Solution |
|---------|----------|
| `pygmtsar` import fails | Use `.venv-insar/bin/python` (requires Python 3.10+) |
| GMTSAR binaries not found | Verify: `which make_s1a_tops snaphu` — reinstall if missing |
| ASF download authentication fails | Check `~/.netrc` or environment variables for typos |
| Phase unwrapping hangs | Set `UNWRAP_METHOD = "skip"` in config.py for testing; snaphu can be slow |
| GPU acceleration not available | Ensure PyTorch with CUDA/MPS support; falls back to CPU if not found |

### General

| Problem | Solution |
|---------|----------|
| Git won't push (repo too large) | Verify `.gitignore` is working: `git status` should not show `*.tif`, `outputs/`, etc. |
| Conda/pip conflicts | Create fresh venv from scratch; avoid mixing conda + pip |
| Outdated dependencies | Run: `pip install --upgrade -r requirements.txt` |

---

## 📚 References & Further Reading

**SAR & Water Mapping:**
- Twele et al. (2018) — [Sentinel-1 SAR backscatter for water mapping](https://www.mdpi.com/2072-4292/11/15/1779)
- Schlaffer et al. (2016) — Global water body detection & monitoring via SAR

**InSAR Fundamentals:**
- [NASA Earth Observatory: Measuring deformation via Radar Interferometry](https://earthobservatory.nasa.gov/images/4603/measuring-earth-deformation-with-radar-interferometry)
- Bürgmann, Rosen, Fielding (2000) — Synthetic Aperture Radar interferometry to measure Earth's surface topography and its deformation (*Reviews of Geophysics*)

**Software & Processing:**
- [PyGMTSAR documentation](https://pygmtsar.github.io/)
- [GMTSAR source](https://topex.ucsd.edu/gmtsar/), co-registration & unwrapping
- [snaphu](https://web.stanford.edu/~rmcleod/snaphu/) — Phase unwrapping algorithm

**Time-Series Analysis:**
- [STL decomposition](https://otexts.com/fpp2/stl.html) — Seasonal & Trend decomposition
- SBAS inversion — Berardino et al. (2002), Lanari et al. (2004)

**Case Study:**
- [Wayanad Landslides (30 July 2024)](https://en.wikipedia.org/wiki/2024_Wayanad_landslides) — Kerala, India

---

## 📄 License & Citation

**License:** CC0 (Public Domain) — Use freely for research, disaster response, and operational monitoring.

**Data:** Sentinel-1 SAR data is open under [Copernicus](https://www.copernicus.eu/en/about-copernicus/legal-notice) terms.

**Citation:** If you use this toolkit in research or operational work, please cite:
```bibtex
@software{sar_toolkit_2024,
  title={SAR Analysis Toolkit -- Water \& Landslide Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/vigneshavmm/SAR},
  note={Sentinel-1 GRD water mapping \& InSAR displacement monitoring}
}
```

---

## 👥 Contact & Support

For questions on water detection, contact the water-track lead.  
For InSAR setup & processing, see `data/wayanad_insar/WAYANAD_INSAR_README.md` or the InSAR track maintainer.

---

**Last updated:** September 2026
