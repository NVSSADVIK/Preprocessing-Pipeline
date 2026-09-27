"""AOI (GeoJSON, lon/lat) -> tiles that cover EXACTLY the AOI's bounding box, in the AOI's UTM zone.

Old behaviour used a fixed global UTM grid, so tile_0000 could be a corner tile that hangs outside the AOI.
Now: tiling starts at the AOI's own corner, tiles are equal-sized (<= tile_size_m), a small AOI becomes ONE tile
equal to its bounding box, and tiles are sorted nearest-to-AOI-centre first (so --max-tiles 1 = the centre).
"""
import json
import math

from sentinelhub import CRS, BBox, Geometry, get_utm_crs
from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union


def load_geometries(path, swap_latlon=False):
    """Read GeoJSON polygons. GeoJSON order is [lon, lat]; swap_latlon=True fixes files written as [lat, lon]."""
    with open(path) as f:
        gj = json.load(f)
    if gj["type"] == "FeatureCollection":
        geoms = [shape(ft["geometry"]) for ft in gj["features"]]
    elif gj["type"] == "Feature":
        geoms = [shape(gj["geometry"])]
    else:
        geoms = [shape(gj)]
    if swap_latlon:
        geoms = [transform(lambda x, y, z=None: (y, x), g) for g in geoms]
    for g in geoms:
        minx, miny, maxx, maxy = g.bounds
        if not (-180 <= minx and maxx <= 180 and -90 <= miny and maxy <= 90):
            raise ValueError(
                f"AOI bounds lon[{minx}, {maxx}] lat[{miny}, {maxy}] are impossible - "
                "your coordinates look like [lat, lon]. Re-run with --swap-latlon."
            )
    return geoms


def aoi_from_center(lat, lon, size_km, resolution=10):
    """Square of exactly size_km x size_km (grid-aligned to `resolution`) centred on (lat, lon), as a lon/lat polygon.
    Built in UTM so 2.56 km at 10 m is exactly 256 x 256 px."""
    r = resolution
    size_m = max(r, round(size_km * 1000 / r) * r)
    utm = get_utm_crs(lon, lat)
    x, y = Transformer.from_crs("EPSG:4326", f"EPSG:{utm.epsg}", always_xy=True).transform(lon, lat)
    cx, cy, half = round(x / r) * r, round(y / r) * r, size_m / 2
    return [Geometry(box(cx - half, cy - half, cx + half, cy + half), utm).transform(CRS.WGS84).geometry]


def _split(lo, hi, step, res):
    """Equal-sized segments (multiples of `res`) covering [lo, hi], each <= step."""
    n = max(1, math.ceil((hi - lo) / step))
    w = math.ceil((hi - lo) / n / res) * res
    return [(lo + i * w, min(lo + (i + 1) * w, hi)) for i in range(n) if lo + i * w < hi]


def make_tiles(geoms, tile_size_m, resolution=10):
    union = unary_union(geoms)
    c = union.centroid
    utm = get_utm_crs(c.x, c.y)
    g = Geometry(union, CRS.WGS84).transform(utm).geometry  # AOI in metres

    minx, miny, maxx, maxy = g.bounds
    r = resolution
    eps = 1e-6  # float round-trip noise must not add a pixel
    minx, miny = math.floor(minx / r + eps) * r, math.floor(miny / r + eps) * r
    maxx, maxy = math.ceil(maxx / r - eps) * r, math.ceil(maxy / r - eps) * r
    step = max(r, (tile_size_m // r) * r)

    tiles = []
    for x0, x1 in _split(minx, maxx, step, r):
        for y0, y1 in _split(miny, maxy, step, r):
            if g.intersects(box(x0, y0, x1, y1)):  # skip tiles that miss a non-rectangular AOI
                tiles.append(BBox((x0, y0, x1, y1), crs=utm))

    cx, cy = g.centroid.x, g.centroid.y
    tiles.sort(key=lambda b: (b.middle[0] - cx) ** 2 + (b.middle[1] - cy) ** 2)
    return tiles


def lonlat_center(bbox):
    """(lon, lat) of a tile centre."""
    pt = Geometry(BBox(bbox.geometry.centroid.bounds, crs=bbox.crs).geometry, bbox.crs).transform(CRS.WGS84).geometry
    return pt.centroid.x, pt.centroid.y


def write_footprints(tiles, geoms, out_path):
    """Save AOI + tile outlines as GeoJSON (WGS84) -> drag into geojson.io / QGIS to SEE where tiles are."""
    feats = []
    for g in geoms:
        feats.append({"type": "Feature", "properties": {"name": "AOI"}, "geometry": g.__geo_interface__})
    for i, t in enumerate(tiles):
        geom = Geometry(t.geometry, t.crs).transform(CRS.WGS84).geometry
        feats.append({"type": "Feature", "properties": {"name": f"tile_{i:04d}"}, "geometry": geom.__geo_interface__})
    with open(out_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
