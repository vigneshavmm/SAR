# How analyze_water.py works
It's a two-method surface-water / flood change-detector on a Sentinel-1 time series. The flow, top to bottom:

1. Load & co-register (:143-172) — glob the VV scenes sorted by date, build one reference grid from scene 0 downsampled ×4 (~40 m, for speed), then warp every scene — VV and its paired VH — onto that grid (read_aligned). Everything ends up pixel-aligned, nodata→NaN, converted linear→dB (to_db).
2. The water primitive (water_mask) — a pixel is open water if it's dark in both VV (< −17 dB) and VH (< −24 dB). Water is specular → low backscatter in both pols; requiring both rejects smooth dry land (roads, sand) that fools VV alone.
3. Time series (:158-172) — per date, count water pixels → km². Two thresholds are tracked: the fixed −17 dB, and a per-scene adaptive Otsu (adaptive_water_db) computed only on the dark tail of the histogram (water is ~0.1% of pixels, so full-histogram Otsu would mis-split), clamped near −17. → water_area_timeseries.png.
4. New-water, method A — threshold (:174-180) — baseline = mean dB of the first ~8 (assumed-dry) scenes. A pixel is new water if it's water now (dual-pol) and the baseline was dry (VV ≥ −17). Then despeckle (morphological opening + drop blobs < 5 px) kills speckle flips. → flood_change_map.png + new_water_threshold.tif.
5. New-water, method B — SAFNet (:210-408) — a Siamese multi-scale CNN trained self-supervised, no labels via two tasks: reconstruction (learn SAR texture) + synthetic change (copy-paste a rectangle = change; add speckle = no-change → the net learns noise invariance). At inference it emits a per-pixel change probability (baseline vs latest); Otsu cuts it; safnet_new_water = "changed" and "water now (dual-pol)", despeckled. The whole block is try-guarded — no torch, script still finishes step 4. → change-prob + mask GeoTIFFs + safnet_change_map.png.
6. Validation (:393-403) — SAFNet's mask is scored (precision/recall/F1/IoU) against a real reference_water.tif if present, else against the threshold mask as an explicitly-labeled PROXY (honest: those numbers aren't ground-truth-validated).

Cross-cutting design: deterministic (SEED=0), torch and scipy both guarded (graceful degradation), everything at 40 m for a ~6-min CPU run, georeferenced GeoTIFF outputs.

# TL;DR: 

Aligns a Sentinel-1 VV+VH time series to one 40 m grid → flags open water where both pols are dark → tracks water-area over time → finds new water two ways: a simple threshold (water-now-but-dry-before) and a self-supervised CNN (SAFNet) that learns change without labels. Both masks are speckle-cleaned and exported as georeferenced GeoTIFFs; SAFNet is scored against the threshold as a labeled proxy since there's no ground truth.