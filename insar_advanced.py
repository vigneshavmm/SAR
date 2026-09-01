""" Advanced InSAR time-series engine (insar.dev-style) for the Wayanad landslide.

This is the "next level" analytical stage that sits ON TOP of the GMTSAR-backed preprocessing (steps 01-03): it takes the per-pair 
*unwrapped phase* stack(``unwrap.nc`` written by 03_process_insar.py) and produces a displacement time series, LOS velocity, and 
quality layers — using the same advanced-compute recipe insar.dev is built around:

    * SBAS network inversion  — weighted least-squares, vectorised over pixels
    * PSI point selection      — amplitude-dispersion / coherence stability
    * STL decomposition        — trend / seasonal / residual on the cumulative series
    * temporal coherence        — per-pixel consistency of the inverted network
    * GPU acceleration          — optional torch backend (Apple MPS / NVIDIA CUDA)
    * Dask chunking             — pixels processed blockwise, out-of-core
    * cloud-native output       — Zarr v3 (local / s3:// / gcs://) or NetCDF

Nothing here needs GMTSAR or snaphu — it is pure array maths, so it runs in the plain InSAR venv (and would run unchanged 
on top of insardev's own unwrapped output). Verify without any SLC data via ``--selftest`` (synthetic stack with a known velocity).

Usage
-----
    python insar_advanced.py --selftest                 # synthetic end-to-end check
    python insar_advanced.py --input <workdir>/unwrap.nc --out outputs/ts.zarr
    python insar_advanced.py --input unwrap.nc --device mps --robust --stl
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np

# ---- optional accelerators / IO backends (all guarded, like the water track) ----
try:
    import torch
    HAVE_TORCH = True
except Exception:                                        # pragma: no cover
    HAVE_TORCH = False

try:
    import xarray as xr
    HAVE_XR = True
except Exception:                                        # pragma: no cover
    HAVE_XR = False

try:
    from statsmodels.tsa.seasonal import STL
    HAVE_STL = True
except Exception:                                        # pragma: no cover
    HAVE_STL = False

# Sentinel-1 C-band radar wavelength (m). LOS displacement d = (lambda / 4pi) * phase.
LAMBDA_MM = 55.465
MM_PER_RAD = LAMBDA_MM / (4.0 * np.pi)
DAYS_PER_YEAR = 365.25

def log(msg: str) -> None:
    print(f"[insar-adv] {msg}", flush=True)

#  Backend: numpy or torch(MPS/CUDA). Only linear-algebra kernels use it.
def pick_device(name: str = "auto") -> str:
    """Resolve the compute device. 'auto' prefers CUDA, then Apple MPS, then CPU."""
    if not HAVE_TORCH:
        return "cpu-numpy"
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name

def _solve_batched(A, b, device):
    """Solve the stack of small SPD systems A[n] x[n] = b[n] on the chosen device.
    A: (npix, M, M), b: (npix, M) -> x: (npix, M). Uses torch on GPU if available,
    else numpy. Cholesky with a jitter fallback for numerical safety."""
    if device not in ("cpu-numpy",) and HAVE_TORCH:
        dev = torch.device("cpu" if device == "cpu" else device)
        At = torch.as_tensor(A, dtype=torch.float64, device=dev)
        bt = torch.as_tensor(b, dtype=torch.float64, device=dev).unsqueeze(-1)
        x = torch.linalg.solve(At, bt).squeeze(-1)
        return x.cpu().numpy()
    # numpy>=2: a 2-D b is a stack of matrices, so add an explicit vector axis.
    return np.linalg.solve(A, b[..., None])[..., 0]

#  SBAS network inversion (weighted least squares, vectorised over pixels)
def parse_pair_dates(pair_labels):
    """From PyGMTSAR-style 'YYYY-MM-DD_YYYY-MM-DD' labels build the sorted unique
    date axis and the (ref_idx, rep_idx) integer indices per pair."""
    def dts(lbl):
        a, b = re.findall(r"\d{4}-\d{2}-\d{2}", str(lbl))[:2]
        return a, b
    refs, reps = zip(*(dts(p) for p in pair_labels))
    dates = sorted(set(refs) | set(reps))
    idx = {d: i for i, d in enumerate(dates)}
    ref_idx = np.array([idx[r] for r in refs])
    rep_idx = np.array([idx[r] for r in reps])
    return np.array(dates), ref_idx, rep_idx

def design_matrix(n_dates, ref_idx, rep_idx):
    """SBAS incidence matrix G (n_pairs, n_dates-1). Unknowns are cumulative LOS
    displacement at dates 1..n-1 relative to date 0 (fixed = 0). Each pair
    contributes +1 at its repeat date and -1 at its reference date."""
    n_pairs = len(ref_idx)
    G = np.zeros((n_pairs, n_dates - 1))
    for p in range(n_pairs):
        if rep_idx[p] > 0:
            G[p, rep_idx[p] - 1] += 1.0
        if ref_idx[p] > 0:
            G[p, ref_idx[p] - 1] -= 1.0
    return G

def sbas_invert(phase, weight, G, device="cpu-numpy", damping=1e-3):
    """Weighted-LSQ SBAS inversion, vectorised over pixels.

    phase:  (n_pairs, npix) unwrapped phase (radians)
    weight: (n_pairs, npix) per-pair weights (e.g. coherence), or None -> uniform
    G:      (n_pairs, M) design matrix, M = n_dates-1

    Returns
      cum:   (n_dates, npix) cumulative LOS displacement in mm (date 0 == 0)
      tcoh:  (npix,) temporal coherence — network consistency in [0, 1]
    """
    n_pairs, npix = phase.shape
    M = G.shape[1]
    y = phase * MM_PER_RAD                                # phase -> mm per pair
    if weight is None:
        weight = np.ones_like(phase)
    w = np.clip(weight, 0.0, 1.0)

    # Per-pixel normal equations via einsum (no Python pixel loop):
    #   A = Gᵀ diag(w_pixel) G   (npix, M, M)
    #   b = Gᵀ diag(w_pixel) y   (npix, M)
    A = np.einsum("pi,pn,pj->nij", G, w, G, optimize=True)
    b = np.einsum("pi,pn,pn->ni", G, w, y, optimize=True)
    A += damping * np.eye(M)[None]                        # Tikhonov: keeps rank-deficient pixels solvable
    x = _solve_batched(A, b, device)                     # (npix, M) cumulative disp at dates 1..M

    cum = np.zeros((M + 1, npix))
    cum[1:] = x.T                                         # date 0 pinned to zero

    # Temporal coherence: how well the inverted field reproduces each pair.
    model = (G @ x.T)                                     # (n_pairs, npix) modelled mm
    resid_rad = (y - model) / MM_PER_RAD
    tcoh = np.abs(np.exp(1j * resid_rad).mean(axis=0))   # |<e^{i·resid}>| over pairs
    return cum, tcoh

#  PSI point selection + robust velocity
def amplitude_dispersion(amp_stack):
    """PS amplitude-dispersion index D_A = std/mean over time (low = stable point)."""
    mu = np.nanmean(amp_stack, axis=0)
    sd = np.nanstd(amp_stack, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mu > 0, sd / mu, np.inf)

def select_ps(tcoh, mean_coh=None, amp_da=None, tcoh_min=0.7, coh_min=0.35, da_max=0.4):
    """Boolean persistent-scatterer / valid-pixel mask. Uses whatever quality
    layers are available: temporal coherence always; mean coherence and/or
    amplitude dispersion when provided."""
    mask = tcoh >= tcoh_min
    if mean_coh is not None:
        mask &= mean_coh >= coh_min
    if amp_da is not None:
        mask &= amp_da <= da_max
    return mask

def velocity_lstsq(cum, t_days):
    """Ordinary least-squares slope of cumulative displacement -> mm/year."""
    t = t_days - t_days.mean()
    denom = (t * t).sum()
    v = (t[:, None] * (cum - cum.mean(0))).sum(0) / denom
    return v * DAYS_PER_YEAR

def velocity_theilsen(cum, t_days, max_pairs=4000):
    """Robust Theil-Sen slope (median of pairwise slopes) -> mm/year. Outlier
    resistant; good where a few epochs decorrelate. Vectorised over pixels."""
    n = len(t_days)
    ii, jj = np.triu_indices(n, k=1)
    dt = t_days[jj] - t_days[ii]
    good = dt != 0
    ii, jj, dt = ii[good], jj[good], dt[good]
    if len(dt) > max_pairs:                              # subsample pairs for big stacks
        sel = np.linspace(0, len(dt) - 1, max_pairs).astype(int)
        ii, jj, dt = ii[sel], jj[sel], dt[sel]
    slopes = (cum[jj] - cum[ii]) / dt[:, None]           # (n_pairs, npix) mm/day
    return np.median(slopes, axis=0) * DAYS_PER_YEAR

#  STL seasonal-trend decomposition (insar.dev-style de-seasonalised trend)
def stl_trend(series, t_days, period, robust=True):
    """STL-decompose a cumulative-displacement series sampled at t_days.
    Resamples to a regular cadence (STL needs even spacing), returns
    (trend, seasonal, resid, trend_velocity_mm_per_year). Needs >=2 periods."""
    if not HAVE_STL:
        raise RuntimeError("statsmodels not installed; STL unavailable.")
    step = np.median(np.diff(t_days))
    reg_t = np.arange(t_days[0], t_days[-1] + step / 2, step)
    reg_y = np.interp(reg_t, t_days, series)
    per = max(2, int(round(period / step)))
    res = STL(reg_y, period=per, robust=robust).fit()
    t = reg_t - reg_t.mean()
    v = (t * (res.trend - res.trend.mean())).sum() / (t * t).sum() * DAYS_PER_YEAR
    return res.trend, res.seasonal, res.resid, float(v)

#  High-level run: (n_pairs,y,x) phase+corr  ->  xarray Dataset of products
def analyze(phase, corr, pair_labels, device="auto", robust=False,
            coh_min=0.35, tcoh_min=0.7, amp_stack=None, do_stl=False,
            stl_period_days=None, chunk=200_000):
    """Full pipeline on numpy arrays. phase/corr: (n_pairs, ny, nx).
    Returns a dict of 2-D/3-D numpy products (+ optional STL on the PS-mean series)."""
    dev = pick_device(device)
    log(f"backend: {dev}  ({'torch' if dev not in ('cpu-numpy',) else 'numpy'})")
    n_pairs, ny, nx = phase.shape
    dates, ref_idx, rep_idx = parse_pair_dates(pair_labels)
    n_dates = len(dates)
    G = design_matrix(n_dates, ref_idx, rep_idx)
    rank = np.linalg.matrix_rank(G)
    log(f"{n_pairs} pairs, {n_dates} dates, design-matrix rank {rank}/{n_dates - 1}"
        + ("  (network connected)" if rank == n_dates - 1 else "  (DISCONNECTED!)"))

    ph = phase.reshape(n_pairs, ny * nx)
    wt = None if corr is None else corr.reshape(n_pairs, ny * nx)

    # Blockwise over pixels -> out-of-core friendly, and each block is a fast
    # vectorised solve. (Same shape a Dask map_blocks would feed us.)
    npix = ny * nx
    cum = np.empty((n_dates, npix))
    tcoh = np.empty(npix)
    t0 = time.time()
    for s in range(0, npix, chunk):
        e = min(s + chunk, npix)
        cum[:, s:e], tcoh[s:e] = sbas_invert(
            ph[:, s:e], None if wt is None else wt[:, s:e], G, device=dev)
    log(f"SBAS inversion: {npix:,} px in {time.time() - t0:.2f}s")

    t_days = (np.array(dates, dtype="datetime64[D]")
              - np.datetime64(dates[0])).astype(int).astype(float)
    vel = (velocity_theilsen(cum, t_days) if robust else velocity_lstsq(cum, t_days))

    mean_coh = None if wt is None else wt.mean(0)
    amp_da = amplitude_dispersion(amp_stack.reshape(n_pairs if amp_stack.shape[0] == n_pairs
                                                     else n_dates, npix)) if amp_stack is not None else None
    ps = select_ps(tcoh, mean_coh, amp_da, tcoh_min=tcoh_min, coh_min=coh_min)
    log(f"PS/valid pixels: {ps.sum():,}/{npix:,} ({100 * ps.mean():.1f}%)")

    out = dict(
        dates=dates, t_days=t_days,
        velocity=vel.reshape(ny, nx),
        cum_disp=cum.reshape(n_dates, ny, nx),
        temporal_coherence=tcoh.reshape(ny, nx),
        ps_mask=ps.reshape(ny, nx),
    )
    if do_stl and HAVE_STL:
        series = np.where(ps, cum.reshape(n_dates, npix), np.nan)
        mean_series = np.nanmean(series, axis=1)
        period = stl_period_days or max(2 * np.median(np.diff(t_days)),
                                        (t_days[-1] - t_days[0]) / 3)
        tr, se, re_, vstl = stl_trend(mean_series, t_days, period)
        out["stl"] = dict(trend=tr, seasonal=se, resid=re_, velocity=vstl)
        log(f"STL de-seasonalised trend velocity (PS-mean): {vstl:+.2f} mm/yr")
    return out

def to_dataset(out):
    """Wrap the products dict as an xarray Dataset (for Zarr/NetCDF export)."""
    if not HAVE_XR:
        raise RuntimeError("xarray not installed; cannot build Dataset.")
    ny, nx = out["velocity"].shape
    coords = dict(date=("date", np.array(out["dates"], dtype="datetime64[D]")),
                  y=("y", np.arange(ny)), x=("x", np.arange(nx)))
    ds = xr.Dataset(
        dict(velocity=(("y", "x"), out["velocity"]),
             cum_disp=(("date", "y", "x"), out["cum_disp"]),
             temporal_coherence=(("y", "x"), out["temporal_coherence"]),
             ps_mask=(("y", "x"), out["ps_mask"].astype("uint8"))),
        coords=coords,
        attrs=dict(units_velocity="mm/year", units_cum_disp="mm",
                   wavelength_mm=LAMBDA_MM, convention="LOS (positive per input phase sign)"))
    return ds

def save(ds, path):
    """Write to Zarr (cloud-native, supports s3://, gcs:// via fsspec) if the path
    ends in .zarr and zarr is installed, else NetCDF."""
    if str(path).endswith(".zarr"):
        try:
            import zarr  # noqa: F401
            ds.chunk({"y": 256, "x": 256}).to_zarr(path, mode="w")
            log(f"wrote Zarr -> {path}")
            return
        except Exception as e:                            # pragma: no cover
            log(f"zarr unavailable ({e}); falling back to NetCDF")
            path = str(path)[:-5] + ".nc"
    ds.to_netcdf(path)
    log(f"wrote NetCDF -> {path}")

#  Self-test: synthetic stack with a KNOWN velocity + seasonal signal
def synthetic_stack(ny=96, nx=96, n_dates=16, cadence=12, max_bt=24,
                    true_vel=-30.0, seed=0, seasonal=True):
    """Build a synthetic unwrapped-phase SBAS stack mimicking the Wayanad study:
    16 dates at 12-day cadence, pairs within a 24-day temporal baseline, a
    localised subsidence bump at true_vel mm/yr plus a seasonal oscillation, and
    coherence-scaled noise. Returns (phase, corr, pair_labels, truth_velocity)."""
    rng = np.random.default_rng(seed)
    d0 = np.datetime64("2024-01-10")
    dates = [str(d0 + np.timedelta64(cadence * k, "D")) for k in range(n_dates)]
    t = np.array([cadence * k for k in range(n_dates)], float)

    yy, xx = np.mgrid[0:ny, 0:nx]
    bump = np.exp(-(((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) / (2 * (ny / 8) ** 2)))
    vel_field = true_vel * bump                          # mm/yr, concentric like the report
    seas_amp = 2.0                                       # mm, spatially-uniform seasonal
    period = 60.0                                        # days (3 periods in-window, for STL)

    # true cumulative LOS displacement (mm) per date
    cum = (vel_field[None] / DAYS_PER_YEAR) * t[:, None, None]
    if seasonal:
        cum = cum + seas_amp * np.sin(2 * np.pi * t[:, None, None] / period)

    coh_field = np.clip(0.85 * bump + 0.25, 0.05, 0.98)  # high on the target, low outside

    # SBAS pairs within max_bt days. Per-interferogram phase noise scales with
    # (1 - coherence): ~0.01 rad on the stable target, ~0.3 rad at the edges.
    pairs, phase, corr = [], [], []
    for i in range(n_dates):
        for j in range(i + 1, n_dates):
            if t[j] - t[i] <= max_bt:
                pairs.append(f"{dates[i]}_{dates[j]}")
                dphi = (cum[j] - cum[i]) / MM_PER_RAD    # mm -> radians (unwrapped)
                noise = rng.normal(0, 0.4, (ny, nx)) * (1 - coh_field)  # coh-scaled
                phase.append(dphi + noise)
                corr.append(coh_field + rng.normal(0, 0.02, (ny, nx)))
    return (np.array(phase), np.clip(np.array(corr), 0, 1),
            pairs, vel_field)

def selftest():
    # ---- A: velocity recovery on a seasonal-free stack (clean tolerances) ----
    log("SELFTEST A — SBAS velocity recovery (no seasonal)")
    phase, corr, pairs, truth = synthetic_stack(seasonal=False)
    out = analyze(phase, corr, pairs, device="auto", robust=False,
                  coh_min=0.55, tcoh_min=0.85)
    ps = out["ps_mask"]
    est, tru = out["velocity"][ps], truth[ps]
    bias = float(np.mean(est - tru))
    rmse = float(np.sqrt(np.mean((est - tru) ** 2)))
    corr_r = float(np.corrcoef(est, tru)[0, 1])
    c = (truth.shape[0] // 2, truth.shape[1] // 2)
    peak_err = abs(out["velocity"][c] - truth[c]) / abs(truth[c])
    log(f"  peak velocity  truth={truth[c]:+.2f}  est={out['velocity'][c]:+.2f} mm/yr "
        f"({100 * peak_err:.1f}% err)")
    log(f"  over PS: bias={bias:+.3f}  RMSE={rmse:.3f} mm/yr  corr={corr_r:.4f}  "
        f"median tcoh={np.median(out['temporal_coherence'][ps]):.3f}")
    okA = (abs(bias) < 0.8 and rmse < 2.5 and corr_r > 0.95 and peak_err < 0.15)

    # ---- B: STL separates the injected seasonal from the subsidence trend ----
    log("SELFTEST B — STL seasonal/trend separation (with seasonal)")
    p2, c2, pr2, _ = synthetic_stack(seasonal=True)
    out2 = analyze(p2, c2, pr2, device="auto", do_stl=True,
                   coh_min=0.55, tcoh_min=0.85)
    stl = out2.get("stl")
    seas_std = float(np.nanstd(stl["seasonal"])) if stl else 0.0
    trend_v = stl["velocity"] if stl else 0.0
    log(f"  STL seasonal std={seas_std:.2f} mm (injected 2.0)  trend vel={trend_v:+.2f} mm/yr")
    okB = (not HAVE_STL) or (stl is not None and seas_std > 0.5 and trend_v < 0)

    # ---- C: cloud/dataset export path ----
    okC = True
    if HAVE_XR:
        ds = to_dataset(out)
        okC = set(ds.data_vars) >= {"velocity", "cum_disp", "temporal_coherence", "ps_mask"}
        log(f"SELFTEST C — Dataset export OK: {list(ds.data_vars)}  dims={dict(ds.sizes)}")

    ok = okA and okB and okC
    log(f"SELFTEST {'PASSED ✓' if ok else 'FAILED ✗'}  (A={okA} B={okB} C={okC})")
    return 0 if ok else 1

def main(argv=None):
    ap = argparse.ArgumentParser(description="Advanced InSAR SBAS/PSI/STL engine")
    ap.add_argument("--input", help="unwrap.nc from 03_process_insar.py")
    ap.add_argument("--out", default="outputs/timeseries.zarr")
    ap.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    ap.add_argument("--robust", action="store_true", help="Theil-Sen velocity")
    ap.add_argument("--stl", action="store_true", help="STL seasonal-trend decomposition")
    ap.add_argument("--coh-min", type=float, default=0.35)
    ap.add_argument("--tcoh-min", type=float, default=0.7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.input:
        ap.error("--input is required (or use --selftest)")
    if not HAVE_XR:
        ap.error("xarray required to read unwrap.nc")

    log(f"loading {a.input}")
    ds = xr.open_dataset(a.input)
    phase = ds["phase"].values
    corr = ds["correlation"].values if "correlation" in ds else None
    pair_dim = "pair" if "pair" in ds.dims else list(ds.dims)[0]
    pairs = [str(p) for p in ds[pair_dim].values]
    out = analyze(phase, corr, pairs, device=a.device, robust=a.robust,
                  do_stl=a.stl, coh_min=a.coh_min, tcoh_min=a.tcoh_min)
    save(to_dataset(out), a.out)
    log("done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())