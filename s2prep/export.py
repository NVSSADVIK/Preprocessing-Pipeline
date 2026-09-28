"""EOPatch -> georeferenced GeoTIFF (tifffile, no rasterio/GDAL)."""
import json

import numpy as np
import tifffile
from eolearn.core import EOPatch
from sentinelhub import CRS, BBox


def write_geotiff(arr, bbox, path, band_names=None):
    """Write [h,w] or [h,w,c] float32 as a north-up GeoTIFF in the bbox's projected CRS (UTM)."""
    if arr.ndim == 2:
        arr = arr[..., None]
    h, w, _ = arr.shape
    sx, sy = (bbox.max_x - bbox.min_x) / w, (bbox.max_y - bbox.min_y) / h
    epsg = bbox.crs.epsg
    geokeys = (1, 1, 0, 3,  # GeoKey directory header: version, revision, minor, n_keys
               1024, 0, 1, 1,  # GTModelType = projected
               1025, 0, 1, 1,  # GTRasterType = PixelIsArea
               3072, 0, 1, epsg)  # ProjectedCSType = EPSG code
    extratags = [
        (33550, 12, 3, (sx, sy, 0.0), True),  # ModelPixelScale
        (33922, 12, 6, (0.0, 0.0, 0.0, bbox.min_x, bbox.max_y, 0.0), True),  # ModelTiepoint (top-left)
        (34735, 3, len(geokeys), geokeys, True),  # GeoKeyDirectory
    ]
    tifffile.imwrite(
        path, np.moveaxis(arr, -1, 0).astype(np.float32),
        photometric="minisblack", planarconfig="separate", compression="zlib",
        extratags=extratags, description=json.dumps({"bands": list(band_names or [])}), metadata=None,
    )


def read_geotiff(path):
    """Read a GeoTIFF written by write_geotiff -> (arr[h,w,c], BBox, band_names)."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        arr = np.moveaxis(tif.asarray(), 0, -1)
        sx, sy = page.tags[33550].value[:2]
        tx, ty = page.tags[33922].value[3:5]
        gk = page.tags[34735].value
        epsg = next(gk[i + 3] for i in range(4, len(gk), 4) if gk[i] == 3072)
        bands = json.loads(page.description).get("bands", [])
    h, w = arr.shape[:2]
    return arr, BBox((tx, ty - h * sy, tx + w * sx, ty), crs=CRS(epsg)), bands


def export_tile(eopatch_path, settings, name):
    patch = EOPatch.load(str(eopatch_path))
    settings.tiff_dir.mkdir(parents=True, exist_ok=True)

    write_geotiff(patch.data_timeless["BANDS_MEDIAN"], patch.bbox,
                  settings.tiff_dir / f"{name}_reflectance.tif", settings.bands)
    write_geotiff(patch.data_timeless["BANDS_NORM"], patch.bbox,
                  settings.tiff_dir / f"{name}_norm.tif", settings.bands)

    return name
