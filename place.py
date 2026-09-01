import geopandas as gpd
import pyogrio
from shapely.geometry import box
from pyproj import Transformer

path = "e075_n15_e080_n10.geojson"
tirupati_lonlat = (79.36, 13.58, 79.47, 13.69)  # minx, miny, maxx, maxy

# 1. inspect without loading
info = pyogrio.read_info(path)
src_crs = info["crs"]
print("source CRS:", src_crs)
print("total_bounds:", info["total_bounds"])
print("feature_count:", info["features"])

# 2. transform bbox into source CRS if needed
if src_crs and src_crs.upper() != "EPSG:4326":
    transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    minx, miny = transformer.transform(tirupati_lonlat[0], tirupati_lonlat[1])
    maxx, maxy = transformer.transform(tirupati_lonlat[2], tirupati_lonlat[3])
    read_bbox = (minx, miny, maxx, maxy)
else:
    read_bbox = tirupati_lonlat

print("bbox used for read:", read_bbox)

# 3. attempt indexed bbox read
gdf = gpd.read_file(path, bbox=read_bbox, engine="pyogrio")
print("rows from bbox read:", len(gdf))

# 4. fallback: bbox read returned 0 but file claims coverage -> full scan + clip
if len(gdf) == 0 and info["features"] > 0:
    tb = info["total_bounds"]
    covers = (tb[0] <= tirupati_lonlat[2] and tb[2] >= tirupati_lonlat[0]
              and tb[1] <= tirupati_lonlat[3] and tb[3] >= tirupati_lonlat[1])
    if covers:
        print("bbox kwarg failed silently, falling back to full read + clip")
        gdf = gpd.read_file(path, engine="pyogrio")
        if gdf.crs and gdf.crs.to_string().upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
    else:
        print("tile does not cover Tirupati bbox — wrong tile")

# 5. final clip to exact extent
if len(gdf) > 0:
    if gdf.crs and gdf.crs.to_string().upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    tirupati_buildings = gdf.clip(box(*tirupati_lonlat))
    tirupati_buildings.to_file("tirupati_buildings.geojson", driver="GeoJSON")
    print(f"Buildings extracted: {len(tirupati_buildings)}")
else:
    print("Buildings extracted: 0 — check tile coverage/CRS above")