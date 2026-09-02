import pystac_client
import planetary_computer as pc
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import os

BBOX = [84.9500, 27.7500, 85.4500, 28.2500]  # W,S,E,N #nepal_floods

PRE_FLOOD_RANGE  = "2026-07-15/2026-08-25"
POST_FLOOD_RANGE = "2026-08-26/2026-09-10"

OUT_DIR = "sar_downloads_nepal"
POLARIZATIONS = ["vv", "vh"]
COLLECTION = "sentinel-1-grd"

os.makedirs(OUT_DIR, exist_ok=True)

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

def fetch(date_range, tag):
    search = catalog.search(
        collections=[COLLECTION],
        bbox=BBOX,
        datetime=date_range,
        query={"sar:instrument_mode": {"eq": "IW"}},
    )
    items = list(search.items())
    print(f"{tag}: {len(items)} scenes found")

    for item in items:
        for pol in POLARIZATIONS:
            if pol not in item.assets:
                continue
            href = item.assets[pol].href
            out_path = os.path.join(OUT_DIR, f"{tag}_{item.id}_{pol}.tif")
            if os.path.exists(out_path):
                continue

            with rasterio.open(href) as src:
                if src.crs is None:
                    with WarpedVRT(src, crs="EPSG:4326") as vrt:
                        window = from_bounds(*BBOX, transform=vrt.transform)
                        data = vrt.read(1, window=window)
                        transform = vrt.window_transform(window)
                        dtype = vrt.dtypes[0]
                        nodata = vrt.nodata
                else:
                    bounds = transform_bounds("EPSG:4326", src.crs, *BBOX)
                    window = from_bounds(*bounds, transform=src.transform)
                    data = src.read(1, window=window)
                    transform = src.window_transform(window)
                    dtype = src.dtypes[0]
                    nodata = src.nodata

            if data.size == 0:
                print(f"skip {item.id}_{pol}: empty window (outside scene extent)")
                continue

            profile = {
                "driver": "GTiff",
                "height": data.shape[0],
                "width": data.shape[1],
                "count": 1,
                "dtype": dtype,
                "crs": "EPSG:4326",
                "transform": transform,
                "nodata": nodata,
                "compress": "lzw",
            }
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            print(f"saved {out_path}")

fetch(PRE_FLOOD_RANGE, "pre")
fetch(POST_FLOOD_RANGE, "post")
print("done")