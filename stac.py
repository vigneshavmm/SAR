#import libraries
import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import os

BBOX = [76.0850,11.4920,76.1650,11.5120] # Chooralmala, W,S,E,N
DATE_RANGE = "2024-05-01/2024-08-31"
OUT_DIR = "sar_downloads"
POLARIZATIONS = ["vv", "vh"]

os.makedirs(OUT_DIR, exist_ok=True)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-1-rtc"],
    bbox=BBOX,
    datetime=DATE_RANGE,
    query={"sar:instrument_mode": {"eq": "IW"}},
)

items = list(search.items())
print(f"{len(items)} scenes found")

for item in items:
    for pol in POLARIZATIONS:
        if pol not in item.assets:
            continue
        href = item.assets[pol].href
        out_path = os.path.join(OUT_DIR, f"{item.id}_{pol}.tif")
        if os.path.exists(out_path):
            continue
        with rasterio.open(href) as src:
            bounds = transform_bounds("EPSG:4326", src.crs, *BBOX)
            window = from_bounds(*bounds, transform=src.transform)
            data = src.read(1, window=window)
            transform = src.window_transform(window)
            profile = src.profile.copy()
            profile.update(
                height=data.shape[0],
                width=data.shape[1],
                transform=transform,
            )
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"saved {out_path}")

print("done")