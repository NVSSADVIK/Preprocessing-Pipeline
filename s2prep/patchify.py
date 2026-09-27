"""Whole-AOI mosaic -> normalised N x (size x size x C) patches (+ index/meta, + optional per-patch GeoTIFFs).

Cutting happens on the MOSAIC, so patches never break at tile borders. The image is padded on the
bottom/right so that every pixel is covered by at least one patch; `pad_frac` in the index says how much of a
patch is padding. unpatchify() puts model output back together (averaging overlaps).
"""
import csv
import json
import math

import numpy as np
from PIL import Image
from pyproj import Transformer
from sentinelhub import BBox

from .export import read_geotiff, write_geotiff


def normalize(arr, mode, ref_max):
    if mode == "01":
        return (np.clip(arr, 0, ref_max) / ref_max).astype(np.float32), {"mode": "01", "reflectance_max": ref_max}
    if mode == "zscore":
        mean, std = arr.reshape(-1, arr.shape[-1]).mean(0), arr.reshape(-1, arr.shape[-1]).std(0) + 1e-8
        return ((arr - mean) / std).astype(np.float32), {"mode": "zscore", "mean": mean.tolist(), "std": std.tolist()}
    if mode == "none":
        return arr.astype(np.float32), {"mode": "none"}
    raise ValueError(f"unknown norm {mode!r}")


def cut_patches(arr, size, overlap, pad_mode="reflect"):
    """arr[h,w,c] -> patches[N,size,size,c], rows(list of dict). Row-major order (top-left to bottom-right)."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch size")
    h, w, c = arr.shape
    stride = size - overlap
    ny = math.ceil(max(h - size, 0) / stride) + 1
    nx = math.ceil(max(w - size, 0) / stride) + 1
    hp, wp = (ny - 1) * stride + size, (nx - 1) * stride + size
    padded = np.pad(arr, ((0, hp - h), (0, wp - w), (0, 0)), mode="reflect" if pad_mode == "reflect" else "constant")

    patches = np.empty((ny * nx, size, size, c), np.float32)
    rows = []
    for i in range(ny):
        for j in range(nx):
            k, y0, x0 = i * nx + j, i * stride, j * stride
            patches[k] = padded[y0 : y0 + size, x0 : x0 + size]
            vh, vw = min(size, max(0, h - y0)), min(size, max(0, w - x0))
            rows.append({"id": k, "row": i, "col": j, "y0": y0, "x0": x0, "pad_frac": round(1 - vh * vw / size**2, 4)})
    return patches, rows, {"H": h, "W": w, "Hp": hp, "Wp": wp, "n_rows": ny, "n_cols": nx, "stride": stride}


def unpatchify(patches, meta, scale=1):
    """Inverse of cut_patches. `scale` = upscaling factor of the model (SR x2.5 -> use a model whose output is an integer x)."""
    n, s, _, c = patches.shape
    stride = meta["stride"] * scale
    hp, wp = meta["Hp"] * scale, meta["Wp"] * scale
    acc, cnt = np.zeros((hp, wp, c), np.float64), np.zeros((hp, wp, 1), np.float64)
    for k in range(n):
        i, j = divmod(k, meta["n_cols"])
        y0, x0 = i * stride, j * stride
        acc[y0 : y0 + s, x0 : x0 + s] += patches[k]
        cnt[y0 : y0 + s, x0 : x0 + s] += 1
    return (acc / cnt)[: meta["H"] * scale, : meta["W"] * scale].astype(np.float32)


def save_patch_pngs(patches, bands, out_dir, max_n=500):
    """True-colour PNG per patch, ONE shared 2-98% stretch so patches are comparable."""
    ix = [bands.index(b) for b in ("B04", "B03", "B02")] if all(b in bands for b in ("B04", "B03", "B02")) else [0, 1, 2]
    rgb = patches[..., ix]
    lo, hi = np.percentile(rgb[:, ::4, ::4], [2, 98])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(rgb), max_n)
    for k in range(n):
        img = (np.clip((rgb[k] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(out_dir / f"patch_{k:04d}.png")
    return n


def patchify_mosaic(mosaic_path, out_dir, size=256, overlap=0, norm="01", ref_max=0.3, pad_mode="reflect",
                    save_tiffs=False, save_png=True):
    arr, bbox, bands = read_geotiff(mosaic_path)  # raw reflectance mosaic
    normed, norm_meta = normalize(arr, norm, ref_max)
    patches, rows, grid = cut_patches(normed, size, overlap, pad_mode)

    res = (bbox.max_x - bbox.min_x) / arr.shape[1]
    tr = Transformer.from_crs(f"EPSG:{bbox.crs.epsg}", "EPSG:4326", always_xy=True)
    for r in rows:
        r["min_x"], r["max_x"] = bbox.min_x + r["x0"] * res, bbox.min_x + (r["x0"] + size) * res
        r["max_y"], r["min_y"] = bbox.max_y - r["y0"] * res, bbox.max_y - (r["y0"] + size) * res
        r["center_lon"], r["center_lat"] = (round(v, 6) for v in tr.transform((r["min_x"] + r["max_x"]) / 2,
                                                                              (r["min_y"] + r["max_y"]) / 2))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "patches.npy", patches)
    with open(out_dir / "patches_index.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    meta = {"patch_size": size, "overlap": overlap, "n_patches": len(rows), "bands": bands, "epsg": bbox.crs.epsg,
            "resolution_m": res, "mosaic_bbox": list(bbox), "pad_mode": pad_mode, "normalization": norm_meta, **grid}
    with open(out_dir / "patches_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if save_png:
        meta["n_png"] = save_patch_pngs(patches, bands, out_dir / "png")
    if save_tiffs:
        (out_dir / "tiffs").mkdir(exist_ok=True)
        for r, p in zip(rows, patches):
            write_geotiff(p, BBox((r["min_x"], r["min_y"], r["max_x"], r["max_y"]), crs=bbox.crs),
                          out_dir / "tiffs" / f"patch_{r['id']:04d}.tif", bands)
    return patches.shape, meta
