"""Sentinel-2 L2A -> cloud-free, normalised 10 m composites + LR patches for super-resolution training.

    python check_auth.py
    python run_pipeline.py --aoi aoi/example_aoi.geojson --start 2025-11-01 --end 2026-02-28 --max-tiles 2
"""
import argparse
from pathlib import Path

import json

from eolearn.core import EOExecutor
from shapely.ops import unary_union

from s2prep.config import get_collection, get_config
from s2prep.export import export_tile
from s2prep.grid import aoi_from_center, lonlat_center, load_geometries, make_tiles, write_footprints
from s2prep.mosaic import build_mosaic
from s2prep.patchify import patchify_mosaic
from s2prep.settings import PipelineSettings
from s2prep.workflow import build_workflow


def parse_args():
    d = PipelineSettings()
    p = argparse.ArgumentParser()
    p.add_argument("--aoi", type=Path, default=d.aoi_path)
    p.add_argument("--swap-latlon", action="store_true", help="AOI file is [lat, lon] instead of GeoJSON [lon, lat]")
    p.add_argument("--center", type=float, nargs=2, metavar=("LAT", "LON"), help="skip the GeoJSON: use a point (Google Maps order)")
    p.add_argument("--size-km", type=float, default=None,
                   help="side of the square around the centre (default: exactly one patch = patch-size x 10 m)")
    p.add_argument("--around-center", action="store_true",
                   help="find the centre of the given AOI/coordinates and image only a square around it")
    p.add_argument("--max-dates", type=int, default=d.max_dates, help="use only the N clearest dates (0 = all); fewer = faster")
    p.add_argument("--threads", type=int, default=d.threads, help="parallel downloads per tile")
    p.add_argument("--no-png", action="store_true", help="skip PNG previews")
    p.add_argument("--start", default=d.time_interval[0])
    p.add_argument("--end", default=d.time_interval[1])
    p.add_argument("--maxcc", type=float, default=d.maxcc)
    p.add_argument("--min-valid", type=float, default=d.min_valid_fraction)
    p.add_argument("--tile-size", type=int, default=d.tile_size_m, help="metres; multiple of 10")
    p.add_argument("--patch-size", type=int, default=d.patch_size, help="model-input patch size in px")
    p.add_argument("--overlap", type=int, default=d.patch_overlap, help="px overlap between patches")
    p.add_argument("--norm", choices=["01", "zscore", "none"], default=d.norm)
    p.add_argument("--save-patch-tiffs", action="store_true", help="also save each patch as a GeoTIFF")
    p.add_argument("--workers", type=int, default=d.workers)
    p.add_argument("--max-tiles", type=int, default=d.max_tiles, help="0 = all")
    p.add_argument("--out", type=Path, default=d.out_dir)
    return p.parse_args()


def main():
    a = parse_args()
    s = PipelineSettings(
        aoi_path=a.aoi, time_interval=(a.start, a.end), maxcc=a.maxcc, min_valid_fraction=a.min_valid,
        tile_size_m=a.tile_size, patch_size=a.patch_size, patch_overlap=a.overlap, norm=a.norm,
        save_patch_tiffs=a.save_patch_tiffs, max_dates=a.max_dates, threads=a.threads, save_png=not a.no_png,
        workers=a.workers, max_tiles=a.max_tiles, out_dir=a.out,
    )
    for d in (s.eopatch_dir, s.cache_dir, s.log_dir):
        d.mkdir(parents=True, exist_ok=True)

    sh_config, provider = get_config()
    workflow, in_node, save_node = build_workflow(s, sh_config, get_collection(provider))

    size_km = a.size_km or s.patch_size * s.resolution / 1000  # default: exactly one patch
    if a.center:
        geoms = aoi_from_center(a.center[0], a.center[1], size_km, s.resolution)
    else:
        geoms = load_geometries(s.aoi_path, swap_latlon=a.swap_latlon)
        if a.around_center:
            c = unary_union(geoms).centroid
            geoms = aoi_from_center(c.y, c.x, size_km, s.resolution)
    ac = unary_union(geoms).centroid
    print(f"AOI centre: lat={ac.y:.5f}, lon={ac.x:.5f}  ->  https://www.google.com/maps?q={ac.y:.5f},{ac.x:.5f}")

    bboxes = make_tiles(geoms, s.tile_size_m, s.resolution)
    write_footprints(bboxes, geoms, s.out_dir / "tiles.geojson")  # open in geojson.io / QGIS to see the tiles
    if s.max_tiles:
        bboxes = bboxes[: s.max_tiles]
    print(f"{len(bboxes)} tile(s) (nearest-to-AOI-centre first), provider={provider}, "
          f"{s.time_interval[0]} -> {s.time_interval[1]}")
    lon, lat = lonlat_center(bboxes[0])
    print(f"tile_0000 centre: lon={lon:.5f}, lat={lat:.5f}  ->  https://www.google.com/maps?q={lat:.5f},{lon:.5f}")
    print(f"tile footprints saved to {s.out_dir / 'tiles.geojson'}")
    info = {"center_lat": ac.y, "center_lon": ac.x, "n_tiles": len(bboxes), "tile_size_m": s.tile_size_m,
            "max_dates": s.max_dates, "time_interval": list(s.time_interval)}

    names = [f"tile_{i:04d}" for i in range(len(bboxes))]
    exec_args = [
        {in_node: {"bbox": bb, "time_interval": s.time_interval}, save_node: {"eopatch_folder": n}}
        for bb, n in zip(bboxes, names)
    ]
    executor = EOExecutor(workflow, exec_args, save_logs=True, logs_folder=str(s.log_dir))
    executor.run(workers=s.workers)

    failed = set(executor.get_failed_executions())
    done = []
    for i, n in enumerate(names):
        if i in failed:
            print(f"[FAILED] {n} - see {s.log_dir}")
            continue
        export_tile(s.eopatch_dir / n, s, n)
        done.append(n)
        print(f"[ok] {n}")

    if not done:
        return
    h, w, _ = build_mosaic(s, done)
    print(f"[ok] whole-AOI mosaic {w}x{h} px -> {s.tiff_dir / 'mosaic_reflectance.tif'}")
    shape, meta = patchify_mosaic(s.tiff_dir / "mosaic_reflectance.tif", s.patch_dir, s.patch_size, s.patch_overlap,
                                  s.norm, s.reflectance_max, s.pad_mode, s.save_patch_tiffs, s.save_png)
    print(f"[ok] {meta['n_patches']} patches of {s.patch_size}x{s.patch_size} ({meta['n_rows']}x{meta['n_cols']} grid), "
          f"norm={s.norm} -> {s.patch_dir / 'patches.npy'}")
    if s.save_png:
        import preview

        preview.main(str(s.tiff_dir), str(s.out_dir / "previews"), "mosaic_reflectance.tif")
        print(f"[ok] PNGs: {s.out_dir / 'previews' / 'mosaic_truecolor.png'} (+ mosaic_falsecolor.png), "
              f"{meta.get('n_png', 0)} patch PNGs in {s.patch_dir / 'png'}")
    info.update(mosaic_px=[w, h], n_patches=meta["n_patches"], patch_size=s.patch_size, tiles_done=len(done))
    (s.out_dir / "aoi_info.json").write_text(json.dumps(info, indent=2))
    if len(done) < len(bboxes):
        print(f"NOTE: only {len(done)}/{len(bboxes)} tiles processed - drop --max-tiles for the whole AOI.")


if __name__ == "__main__":
    main()
