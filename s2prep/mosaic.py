"""Stitch all processed tiles into ONE georeferenced image covering the whole AOI (no API calls)."""
import numpy as np
from eolearn.core import EOPatch
from sentinelhub import BBox

from .export import write_geotiff


def build_mosaic(settings, names):
    patches = [EOPatch.load(str(settings.eopatch_dir / n)) for n in names]
    r = settings.resolution
    x0 = min(p.bbox.min_x for p in patches)
    y0 = min(p.bbox.min_y for p in patches)
    x1 = max(p.bbox.max_x for p in patches)
    y1 = max(p.bbox.max_y for p in patches)
    W, H = round((x1 - x0) / r), round((y1 - y0) / r)
    C = patches[0].data_timeless["BANDS_MEDIAN"].shape[-1]

    raw = np.zeros((H, W, C), np.float32)
    norm = np.zeros((H, W, C), np.float32)
    for p in patches:
        h, w = p.data_timeless["BANDS_MEDIAN"].shape[:2]
        ox, oy = round((p.bbox.min_x - x0) / r), round((y1 - p.bbox.max_y) / r)
        raw[oy : oy + h, ox : ox + w] = p.data_timeless["BANDS_MEDIAN"]
        norm[oy : oy + h, ox : ox + w] = p.data_timeless["BANDS_NORM"]

    bbox = BBox((x0, y0, x1, y1), crs=patches[0].bbox.crs)
    settings.tiff_dir.mkdir(parents=True, exist_ok=True)
    write_geotiff(raw, bbox, settings.tiff_dir / "mosaic_reflectance.tif", settings.bands)
    write_geotiff(norm, bbox, settings.tiff_dir / "mosaic_norm.tif", settings.bands)
    return (H, W, C)
