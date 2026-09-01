import os, re, glob
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# scipy is optional: if absent, despeckle() becomes a no-op (script still runs).
try:
    from scipy import ndimage as ndi
    HAVE_NDI = True
except Exception as e:                                    # pragma: no cover
    HAVE_NDI = False
    print(f"[despeckle] scipy unavailable ({e}); morphological cleanup skipped.")

SRC = "data/Tirupati"
OUT = "outputs"; os.makedirs(OUT, exist_ok=True)
DS = 4                      # downsample factor -> ~40 m pixels (district scale, fast)
WATER_DB = -17.0           # VV threshold: below this = open water (standard-ish)
USE_VH = True              # dual-pol: require BOTH co- and cross-pol dark (rejects smooth dry land)
WATER_DB_VH = -24.0        # VH threshold; cross-pol over water sits well below VV
MIN_WATER_PX = 5           # drop connected new-water blobs smaller than this (speckle)
REF_WATER = "reference_water.tif"   # optional ground-truth mask; if present, scores SAFNet against it

# ---- SAFNet (Siamese Adaptive Fusion Network) hyper-parameters ----
SAFNET_CONFIG = dict(
    SEED = 0,                    # reproducible patch sampling + weight init
    PATCH = 64,                  # training patch size (px)
    BASE = 16,                   # encoder base width (scales use BASE, 2*BASE, 4*BASE)
    FUSE = 16,                   # channel width of the adaptive-fusion space
    EPOCHS = 15,                 # self-supervised (reconstruction + synthetic-change) epochs
    PATCHES_PER_SCENE = 64,      # random crops sampled from each scene
    BATCH = 32,
    LR = 1e-3,
)

def datekey(f):            # pull YYYYMMDD from filename
    m = re.search(r"_(\d{8})T", os.path.basename(f))
    if not m:
        raise SystemExit(f"Cannot parse an acquisition date (_YYYYMMDDT) from: {f}")
    return m.group(1)

def to_db(lin):
    lin = np.where(np.isfinite(lin) & (lin > 0), lin, np.nan)
    return 10.0 * np.log10(lin)

def build_ref_grid(path):
    """Common downsampled (crs, transform, width, height) taken from one scene.
    Every scene is then resampled onto THIS grid so all arrays are co-registered
    (guards np.stack against footprint drift) and the transform georeferences the
    exported GeoTIFF masks."""
    with rasterio.open(path) as s:
        w = s.width // DS; h = s.height // DS
        transform = s.transform * s.transform.scale(s.width / w, s.height / h)
        return dict(crs=s.crs, transform=transform, width=w, height=h)

def read_aligned(path, ref):
    """Read band 1 (average-resampled) onto the reference grid; nodata -> NaN."""
    with rasterio.open(path) as s:
        with WarpedVRT(s, crs=ref["crs"], transform=ref["transform"],
                       width=ref["width"], height=ref["height"],
                       resampling=Resampling.average,
                       src_nodata=s.nodata, nodata=s.nodata) as vrt:
            a = vrt.read(1).astype("float64")
            nod = vrt.nodata
    if nod is not None:
        a[a == nod] = np.nan
    return a

def water_mask(db_vv, db_vh, t_vv):
    """Boolean open-water mask. Dual-pol when VH is available: dark in BOTH co-
    and cross-pol, which rejects smooth dry surfaces (roads, dry sand) that fool
    a VV-only test."""
    m = np.isfinite(db_vv) & (db_vv < t_vv)
    if db_vh is not None:
        m &= np.isfinite(db_vh) & (db_vh < WATER_DB_VH)
    return m

def despeckle(mask, min_px=MIN_WATER_PX):
    """Binary opening + small-component removal to kill single-pixel speckle
    flips. No-op if scipy is unavailable or the mask is empty."""
    if not HAVE_NDI or not mask.any():
        return mask
    m = ndi.binary_opening(mask, structure=np.ones((3, 3), bool))
    lab, n = ndi.label(m)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0     # drop background
        m = (sizes >= min_px)[lab]
    return m

def write_gtiff(path, arr, ref, dtype, nodata=None):
    """Write a single-band GeoTIFF on the reference grid (so masks land in GIS)."""
    prof = dict(driver="GTiff", height=ref["height"], width=ref["width"], count=1,
                dtype=dtype, crs=ref["crs"], transform=ref["transform"], compress="deflate")
    if nodata is not None:
        prof["nodata"] = nodata
    with rasterio.open(path, "w", **prof) as d:
        d.write(arr.astype(dtype), 1)

def otsu(values):
    """Otsu bimodal threshold on a 1-D array (numpy-only, no scikit-image)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist).astype("float64")            # weight of class below thr
    total = w0[-1]
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = csum / w0
        m1 = (csum[-1] - csum) / w1
    between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = 0.0
    return float(centers[int(np.argmax(between))])

def adaptive_water_db(db, window=(-24.0, -12.0), clamp=(-20.0, -14.0)):
    """Per-scene water/land cut via Otsu — but computed only on the DARK tail.
    Open water is a tiny minority here (~1% of pixels), so Otsu on the full
    histogram would split land-vs-land; restricting to the [window] dark band
    isolates the water/dark-land boundary. The result is clamped to a sane band
    around the -17 dB default so a noisy scene can't drift the cut absurdly."""
    v = db[np.isfinite(db)]
    v = v[(v >= window[0]) & (v <= window[1])]
    if v.size < 1000:                      # too few dark pixels -> trust the default
        return WATER_DB
    return float(np.clip(otsu(v), clamp[0], clamp[1]))

def binary_scores(pred, ref, valid=None):
    """Pixel-wise precision / recall / F1 / IoU of a boolean prediction vs a
    boolean reference (both 2-D). Restricts to `valid` pixels when given."""
    pred = np.asarray(pred, bool); ref = np.asarray(ref, bool)
    if valid is not None:
        pred = pred & valid; ref = ref & valid
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    iou  = tp / max(tp + fp + fn, 1)
    return dict(precision=prec, recall=rec, f1=f1, iou=iou, tp=tp, fp=fp, fn=fn)

vv = sorted(glob.glob(f"{SRC}/*_rtc_vv.tif"), key=datekey)
dates = [datekey(f) for f in vv]
if not vv:
    raise SystemExit(f"No VV scenes found under {SRC}/")
if len(vv) < 2:
    raise SystemExit("Need at least two VV scenes to build the baseline change map.")
ref = build_ref_grid(vv[0])        # common grid every scene is aligned to
print(f"{len(vv)} VV scenes: {dates[0]} -> {dates[-1]}  "
      f"(dual-pol={'on' if USE_VH else 'off'}, grid {ref['width']}x{ref['height']})")

# ---- water area time series (fixed -17 dB and per-scene adaptive Otsu) ----
labels, water_km2, frac = [], [], []
water_km2_ad, thr_ad = [], []
stack_db, stack_vh = [], []
px_area = (10*DS)**2 / 1e6         # km^2 per pixel
for f in vv:
    db = to_db(read_aligned(f, ref))
    vh_path = f.replace("_rtc_vv.tif", "_rtc_vh.tif")
    vh = to_db(read_aligned(vh_path, ref)) if (USE_VH and os.path.exists(vh_path)) else None
    stack_db.append(db); stack_vh.append(vh)
    valid = np.isfinite(db)
    wet = water_mask(db, vh, WATER_DB)
    water_km2.append(wet.sum() * px_area)
    frac.append(100.0 * wet.sum() / max(valid.sum(), 1))
    t = adaptive_water_db(db)                       # per-scene dark-tail Otsu (clamped)
    thr_ad.append(t)
    water_km2_ad.append(water_mask(db, vh, t).sum() * px_area)
    labels.append(f"{datekey(f)[4:6]}/{datekey(f)[6:8]}")
    print(f"  {datekey(f)}: water = {water_km2[-1]:6.1f} km^2  ({frac[-1]:4.1f}% of scene)"
          f"   | adaptive {t:5.1f} dB -> {water_km2_ad[-1]:6.1f} km^2")

# ---- baseline (dry: first ~8 scenes Jan-Apr) vs latest, flood/new-water map ----
n_base = min(8, len(stack_db)-1)
# A high percentile is more robust to wet-day outliers than the mean.
baseline = np.nanpercentile(np.stack(stack_db[:n_base]), 80, axis=0)
latest = stack_db[-1]; latest_vh = stack_vh[-1]
# dry before (VV above cut), open water now (dual-pol), then speckle-cleaned.
new_water = despeckle(water_mask(latest, latest_vh, WATER_DB) & (baseline >= WATER_DB))
write_gtiff(f"{OUT}/new_water_threshold.tif", new_water.astype("uint8"), ref, "uint8")

# ---- figure 1: water-area time series (fixed vs per-scene adaptive threshold) ----
fig, ax = plt.subplots(figsize=(11,4.5))
ax.plot(range(len(labels)), water_km2, "-o", color="#1f77b4", lw=2, label="fixed −17 dB")
ax.fill_between(range(len(labels)), water_km2, alpha=0.15, color="#1f77b4")
ax.plot(range(len(labels)), water_km2_ad, "--s", color="#d62728", lw=1.6, ms=4,
        label="adaptive Otsu (per scene)")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("Open-water area (km²)"); ax.set_xlabel("2026 (month/day)")
ax.set_title(f"Surface-water area over time — Tirupati / coastal AP "
             f"(Sentinel-1 {'VV∧VH' if USE_VH else 'VV'})")
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/water_area_timeseries.png", dpi=130); plt.close(fig)

# ---- figure 2: baseline vs latest vs new-water map (threshold method) ----
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].imshow(baseline, cmap="gray", vmin=-25, vmax=0); axs[0].set_title(f"Dry baseline (Jan–Apr mean)")
axs[1].imshow(latest, cmap="gray", vmin=-25, vmax=0);   axs[1].set_title(f"Latest: {dates[-1]}")
axs[2].imshow(latest, cmap="gray", vmin=-25, vmax=0)
mask = np.ma.masked_where(~new_water, new_water)
axs[2].imshow(mask, cmap="autumn", alpha=0.9); axs[2].set_title("NEW water vs baseline (red)")
for a in axs: a.axis("off")
fig.tight_layout(); fig.savefig(f"{OUT}/flood_change_map.png", dpi=130); plt.close(fig)

new_km2 = new_water.sum() * px_area
print(f"\nBaseline = mean of first {n_base} scenes ({dates[0]}–{dates[n_base-1]})")
print(f"[threshold]  NEW open water on {dates[-1]} vs dry baseline: {new_km2:.1f} km^2")


# ============================================================
#  SAFNet — Siamese Adaptive Fusion Network (PyTorch)
# ============================================================
# A shared-weight, MULTI-SCALE encoder (the Siamese branch) embeds each date.
try:
    # All SAFNet logic is now in its own module for clarity.
    import safnet
    if not safnet.HAVE_TORCH:
        raise ImportError("PyTorch not found inside safnet module")
except Exception as e:                                    # pragma: no cover
    safnet = None
    print(f"\n[safnet] PyTorch unavailable ({e}); skipping deep change detection.")


if safnet:
    safnet.torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(SAFNET_CONFIG['SEED'])
    model = safnet.train_safnet(stack_db, rng, SAFNET_CONFIG)

    prob, valid = safnet.safnet_change_prob(model, baseline, latest)
    thr = otsu(prob)                                       # data-driven change cut
    changed = np.isfinite(prob) & (prob > thr)            # learned "something changed here"
    # New water = the network sees a change AND the latest scene is open water
    # (dual-pol). The learned change mask replaces the threshold method's hard
    # "baseline was dry" test; the two methods differ only in how "became water"
    # is decided. Speckle-cleaned like the threshold map.
    safnet_new_water = despeckle(changed & valid & water_mask(latest, latest_vh, WATER_DB))

    saf_km2 = safnet_new_water.sum() * px_area

    # ---- validation: score SAFNet against a ground-truth mask if one exists,
    #      else against the threshold method as a clearly-labelled PROXY ----
    if os.path.exists(REF_WATER):
        gt = read_aligned(REF_WATER, ref)
        ref_mask = np.isfinite(gt) & (gt > 0.5)            # binarize (avg-resampled)
        ref_name = "ground truth"
    else:
        ref_mask = new_water
        ref_name = "threshold method (PROXY — not ground truth)"
    scores = binary_scores(safnet_new_water, ref_mask, valid)
    iou = scores["iou"]

    # ---- export georeferenced SAFNet products (open in QGIS on the source grid) ----
    write_gtiff(f"{OUT}/new_water_safnet.tif", safnet_new_water.astype("uint8"), ref, "uint8")
    write_gtiff(f"{OUT}/safnet_change_prob.tif",
                np.where(np.isfinite(prob), prob, -1.0), ref, "float32", nodata=-1.0)

    # ---- figure 3: threshold vs SAFNet, side by side ----
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(latest, cmap="gray", vmin=-25, vmax=0)
    m0 = np.ma.masked_where(~new_water, new_water)
    axs[0].imshow(m0, cmap="autumn", alpha=0.9)
    axs[0].set_title(f"Threshold new-water (red)\n{new_km2:.1f} km²")

    im = axs[1].imshow(prob, cmap="magma", vmin=0, vmax=1)
    axs[1].set_title("SAFNet change probability\n(adaptive-fusion head)")
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    axs[2].imshow(latest, cmap="gray", vmin=-25, vmax=0)
    m2 = np.ma.masked_where(~safnet_new_water, safnet_new_water)
    axs[2].imshow(m2, cmap="autumn", alpha=0.9)
    axs[2].set_title(f"SAFNet new-water (red)\n{saf_km2:.1f} km²  ·  F1={scores['f1']:.2f}  IoU={scores['iou']:.2f}")

    for a in axs:
        a.axis("off")
    fig.suptitle("New-water detection — threshold vs SAFNet (Siamese Adaptive Fusion Network)", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/safnet_change_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"[safnet]    NEW open water on {dates[-1]} vs dry baseline: {saf_km2:.1f} km^2")
    print(f"[safnet]    scored vs {ref_name}:")
    print(f"[safnet]      precision {scores['precision']:.2f}  recall {scores['recall']:.2f}  "
          f"F1 {scores['f1']:.2f}  IoU {scores['iou']:.2f}")
    print(f"Saved -> {OUT}/*.png (3 figures)  +  GeoTIFFs: new_water_threshold.tif, "
          f"new_water_safnet.tif, safnet_change_prob.tif")
else:
    print(f"Saved -> {OUT}/water_area_timeseries.png  {OUT}/flood_change_map.png  "
          f"+  GeoTIFF: new_water_threshold.tif")
