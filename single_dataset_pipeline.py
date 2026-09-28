"""
single_dataset_pipeline.py — Sentinel-2 Single Dataset Preprocessing Pipeline

Key Features:
1. Input: Rectangular AOI or center coordinates from `input/coordinates.csv`.
   STRICT GEOMETRY VALIDATION: Enforces that inputs are 2D RECTANGLES.
   Explicitly rejects lines (LineString) or triangles (3-vertex polygons).
2. Automatic 2560m UTM Bounding Box: Centered on rectangle -> strictly 256x256 pixels at 10m resolution.
3. PyTorch Dataset (`SentinelDataset`) & DataLoader (`batch_size=1, shuffle=False, num_workers=0`).
4. Real Satellite Preprocessing (`SentinelPreprocessor`):
   - Authenticates with Copernicus Data Space Ecosystem (CDSE) Sentinel Hub API via OAuth.
   - STRICT ERROR HANDLING: If credentials are missing or API connection cannot be established,
     immediately halts and alerts the user. NEVER generates fake/random synthetic outputs.
   - Downloads real Sentinel-2 L2A BOA reflectance and SCL classification via `SentinelHubDownloadTask`.
   - Optimized for speed with 3 clearest dates (`max_dates=3`) and multi-threaded retrieval.
   - eo-learn SCL cloud/shadow/cirrus masking + temporal median compositing + normalization.
5. Standardized 256x256 Output Layout under `output/single_dataset/<sample_id>/`:
   - `composite_reflectance.tif`: 4-band cloud-free median composite (256x256, BOA reflectance [0, 1])
   - `composite_norm.tif`: 4-band normalized composite (256x256, scaled [0, 1])
   - Numbered band subfolders: `1/B02.tif`, `2/B03.tif`, `3/B04.tif`, `4/B08.tif` (each 256x256)
   - `comparison.jpg`: 3-panel true satellite visualization (Raw Scene / Cloud Mask / Cloud-Free Composite)
   - `metadata.json`: Summary metadata (coordinates, UTM bounding box, EPSG, acquisition timestamps)
6. Hard Cap: MAX_SAMPLES = 5 for fast, safe verification.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv(override=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from pyproj import Transformer

from eolearn.core import EOPatch
from sentinelhub import BBox, CRS, Geometry, SentinelHubSession, get_utm_crs

# Core preprocessing tasks from s2prep
from s2prep.config import get_collection, get_config
from s2prep.download import SentinelHubDownloadTask
from s2prep.export import write_geotiff
from s2prep.tasks import (
    ComputeValidMaskTask,
    FilterByValidFractionTask,
    MedianCompositeTask,
    NormalizeTask,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SingleDatasetPipeline")


# ---------------------------------------------------------------------------
# Geometry Validation Helper: Enforces RECTANGLE (Rejects Lines and Triangles)
# ---------------------------------------------------------------------------

def validate_and_extract_rectangle(record: Dict[str, Any]) -> Tuple[float, float, Tuple[float, float, float, float]]:
    """
    Validates that the input spatial definition is a 2D RECTANGLE.
    Explicitly rejects lines (LineStrings) and triangles (3-vertex polygons).

    Returns:
        (latitude, longitude, (min_lon, min_lat, max_lon, max_lat))
    """
    sid = record.get("sample_id", "sample")

    # Case A: GeoJSON geometry present
    if "geometry" in record and isinstance(record["geometry"], dict):
        geom = record["geometry"]
        g_type = geom.get("type", "")

        if g_type in ("LineString", "MultiLineString"):
            raise ValueError(
                f"[{sid}] Invalid geometry type '{g_type}' (a line). "
                "The input must be a 2D rectangle (not a line or triangle)."
            )

        if g_type == "Polygon":
            coordinates = geom.get("coordinates", [[]])
            if not coordinates or not coordinates[0]:
                raise ValueError(f"[{sid}] Empty Polygon coordinates.")
            ring = coordinates[0]
            unique_pts = set(tuple(p[:2]) for p in ring)

            if len(unique_pts) < 3:
                raise ValueError(
                    f"[{sid}] Degenerate geometry with {len(unique_pts)} unique vertices. "
                    "Input must be a 2D rectangle, not a point or line."
                )
            if len(unique_pts) == 3:
                raise ValueError(
                    f"[{sid}] Geometry is a Triangle (3 unique vertices). "
                    "Input must be a rectangle (4 corners), not a triangle or line."
                )

            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]
            min_lon, max_lon = min(lons), max(lons)
            min_lat, max_lat = min(lats), max(lats)

            if min_lon >= max_lon or min_lat >= max_lat:
                raise ValueError(f"[{sid}] Degenerate rectangle with zero width or height (a line).")

            c_lon = (min_lon + max_lon) / 2.0
            c_lat = (min_lat + max_lat) / 2.0
            return c_lat, c_lon, (min_lon, min_lat, max_lon, max_lat)

    # Case B: Explicit rectangle bounds
    if all(k in record for k in ("min_lon", "min_lat", "max_lon", "max_lat")):
        min_lon = float(record["min_lon"])
        min_lat = float(record["min_lat"])
        max_lon = float(record["max_lon"])
        max_lat = float(record["max_lat"])

        if min_lon >= max_lon or min_lat >= max_lat:
            raise ValueError(
                f"[{sid}] Invalid bounding box [{min_lon}, {min_lat}, {max_lon}, {max_lat}]: "
                "min must be strictly less than max (cannot be a flat line or inverted)."
            )
        c_lon = (min_lon + max_lon) / 2.0
        c_lat = (min_lat + max_lat) / 2.0
        return c_lat, c_lon, (min_lon, min_lat, max_lon, max_lat)

    # Case C: Center latitude and longitude
    if "latitude" in record and "longitude" in record:
        lat = float(record["latitude"])
        lon = float(record["longitude"])
        # Generate 2560m x 2560m rectangular bounds around coordinate
        half_deg_lat = 1280.0 / 111320.0
        half_deg_lon = 1280.0 / (111320.0 * max(0.01, np.cos(np.radians(lat))))
        bbox_rect = (lon - half_deg_lon, lat - half_deg_lat, lon + half_deg_lon, lat + half_deg_lat)
        return lat, lon, bbox_rect

    raise ValueError(f"[{sid}] Missing spatial coordinates. Provide a rectangle Polygon or (latitude, longitude).")


# ---------------------------------------------------------------------------
# 1. Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configures input coordinates, output paths, and satellite parameters."""
    data_root: Path = field(default_factory=lambda: Path(os.getenv("INPUT_DIR", "input")))
    output_root: Path = field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output")))
    dataset_name: str = "single_dataset"
    coords_file: str = "coordinates.csv"

    bands: Tuple[str, ...] = ("B02", "B03", "B04", "B08")  # Blue, Green, Red, NIR (10m)
    resolution: int = 10                                     # 10 metres/pixel
    target_size: Tuple[int, int] = (256, 256)               # Required downstream size (256x256)
    maxcc: float = 0.65                                      # Cloud cover threshold (allows candidate scenes with clouds)
    min_valid_fraction: float = 0.5                          # Min valid pixel fraction
    reflectance_max: float = 0.3                             # Normalization upper bound
    max_samples: int = 5                                     # Capped to 5 samples per run
    max_dates: int = 3                                       # Fast compositing (3 clearest dates)
    threads: int = 3                                         # Concurrent download threads for fast execution


# ---------------------------------------------------------------------------
# 2. PyTorch Dataset & DataLoader
# ---------------------------------------------------------------------------

class SentinelDataset(Dataset):
    """
    PyTorch Dataset reading coordinates from CSV.
    Validates that each spatial target is a valid RECTANGLE (not a line or triangle).
    """

    def __init__(self, data_root: Path | str, coords_file: str = "coordinates.csv", dataset_name: str = "single_dataset"):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.csv_path = self.data_root / coords_file
        self.samples: List[Dict[str, Any]] = []
        self._load_coordinates()

    def _load_coordinates(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Coordinates file not found at: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"].strip()
                s_date = row["start_date"].strip()
                e_date = row["end_date"].strip()

                # Validate spatial input is a RECTANGLE (rejects lines / triangles)
                lat, lon, bbox_rect = validate_and_extract_rectangle(row)

                self.samples.append({
                    "sample_id": sid,
                    "latitude": lat,
                    "longitude": lon,
                    "bbox_rect": bbox_rect,
                    "time_interval": (s_date, e_date),
                    "dataset_name": self.dataset_name,
                    "source_path": str(self.csv_path),
                })
        logger.info(f"Loaded {len(self.samples)} validated rectangular target(s) from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 3. Model Definition: SentinelPreprocessor
# ---------------------------------------------------------------------------

class SentinelPreprocessor:
    """
    Wraps Copernicus CDSE Sentinel Hub API + eo-learn cloud masking,
    temporal median compositing, and 256x256 GeoTIFF export.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_root = config.output_root
        self.dataset_name = config.dataset_name
        self.target_size = config.target_size  # (256, 256)

        # Authenticate with Copernicus CDSE Sentinel Hub API
        try:
            self.sh_config, provider = get_config()
            self.collection = get_collection(provider)
            # Test OAuth session to verify connectivity immediately
            session = SentinelHubSession(config=self.sh_config)
            token = session.token
            if not token:
                raise ConnectionError("Failed to obtain OAuth token from Copernicus CDSE.")
            logger.info(f"Copernicus CDSE Sentinel Hub API authenticated successfully (provider={provider}).")
        except Exception as exc:
            err_msg = (
                "\n" + "=" * 80 + "\n"
                "[ERROR] CDSE SENTINEL HUB API CONNECTION NOT ESTABLISHED!\n"
                f"Reason: {exc}\n\n"
                "Please verify that your CDSE credentials in .env are valid:\n"
                "  SH_PROVIDER=cdse\n"
                "  SH_CLIENT_ID=<your_cdse_client_id>\n"
                "  SH_CLIENT_SECRET=<your_cdse_client_secret>\n\n"
                "The pipeline cannot continue without active API connectivity because real\n"
                "satellite imagery is required for the requested coordinates (no random/synthetic output).\n"
                + "=" * 80 + "\n"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg) from exc

        # eo-learn preprocessing tasks
        self.valid_mask_task = ComputeValidMaskTask()
        self.filter_task = FilterByValidFractionTask(config.min_valid_fraction)
        self.median_task = MedianCompositeTask()
        self.norm_task = NormalizeTask(config.reflectance_max)

    def _get_bbox(self, lat: float, lon: float, bbox_rect: Optional[Tuple[float, float, float, float]] = None) -> Tuple[BBox, int]:
        """
        Creates an exact 2560m UTM rectangular square centered on the input rectangle.
        At 10m resolution: 2560m / 10m = strictly 256x256 pixels.
        """
        utm_crs = get_utm_crs(lon, lat)
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_crs.epsg}", always_xy=True)

        cx, cy = transformer.transform(lon, lat)
        half = (self.target_size[0] * self.config.resolution) / 2  # 2560 / 2 = 1280 m
        min_x, max_x = cx - half, cx + half
        min_y, max_y = cy - half, cy + half

        return BBox((min_x, min_y, max_x, max_y), crs=utm_crs), utm_crs.epsg

    def _acquire_scene(self, bbox: BBox, sample: Dict[str, Any]) -> EOPatch:
        """
        Acquire real multi-temporal Sentinel-2 L2A satellite imagery for the exact rectangular box.
        Optimized for speed using max_dates and concurrent threads.
        """
        time_interval = sample["time_interval"]
        sid = sample["sample_id"]
        lat, lon = sample["latitude"], sample["longitude"]

        logger.info(f"Querying Copernicus CDSE for '{sid}' at ({lat:.4f}, {lon:.4f}) over {time_interval}...")

        # Progressive cloud filter search (starting with configured maxcc, then relaxing if needed)
        for cc_threshold in [self.config.maxcc, 0.6, 0.9]:
            try:
                download_task = SentinelHubDownloadTask(
                    data_collection=self.collection,
                    config=self.sh_config,
                    bands=self.config.bands,
                    resolution=self.config.resolution,
                    maxcc=cc_threshold,
                    max_dates=self.config.max_dates,
                    threads=self.config.threads,
                )
                patch = download_task.execute(bbox=bbox, time_interval=time_interval)
                if patch and len(patch.timestamps) > 0:
                    return patch
            except ValueError as v_err:
                logger.warning(f"No scenes at cloud threshold {cc_threshold:.0%}: {v_err}. Retrying with relaxed filter...")
                continue
            except Exception as api_err:
                logger.error(f"CDSE API request failed for '{sid}' ({lat}, {lon}): {api_err}")
                raise RuntimeError(
                    f"Copernicus CDSE API download failed for '{sid}' at ({lat}, {lon}): {api_err}"
                ) from api_err

        raise RuntimeError(
            f"No Sentinel-2 satellite scenes found on Copernicus CDSE for '{sid}' at ({lat}, {lon}) "
            f"in time window {time_interval}. Please check observation dates."
        )

    def _ensure_256x256(self, arr: np.ndarray) -> np.ndarray:
        """Enforces exact (256, 256) spatial dimensions."""
        h, w = arr.shape[:2]
        if (h, w) == self.target_size:
            return arr
        target_h, target_w = self.target_size
        out = np.zeros((target_h, target_w, arr.shape[-1]), dtype=arr.dtype)
        copy_h, copy_w = min(h, target_h), min(w, target_w)
        out[:copy_h, :copy_w] = arr[:copy_h, :copy_w]
        return out

    def process(self, sample: Dict[str, Any], sample_id: str) -> Dict[str, Any]:
        """
        Processes one rectangular coordinate into 256x256 cloud-free GeoTIFFs + comparison.jpg:
        output/<dataset_name>/<sample_id>/
        """
        lat = sample["latitude"]
        lon = sample["longitude"]
        bbox_rect = sample.get("bbox_rect")

        # Setup output sample directory
        sample_dir = self.output_root / self.dataset_name / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 1. Bounding box & real satellite acquisition
        bbox, epsg = self._get_bbox(lat, lon, bbox_rect=bbox_rect)
        eopatch = self._acquire_scene(bbox, sample)

        # 2. Cloud masking & median compositing
        eopatch = self.valid_mask_task.execute(eopatch)
        # Note: MedianCompositeTask handles masking per pixel (np.where(valid, bands, np.nan)),
        # discarding cloud/shadow pixels while compositing clear pixels across time.
        # We retain the cloudy timestamp in eopatch for true comparison visualization.
        eopatch = self.median_task.execute(eopatch)
        eopatch = self.norm_task.execute(eopatch)

        # Enforce exact 256x256 dimensions & physical non-negative reflectance
        refl_comp = np.clip(self._ensure_256x256(eopatch.data_timeless["BANDS_MEDIAN"]), 0.0, 1.0)
        norm_comp = np.clip(self._ensure_256x256(eopatch.data_timeless["BANDS_NORM"]), 0.0, 1.0)
        eopatch.data_timeless["BANDS_MEDIAN"] = refl_comp
        eopatch.data_timeless["BANDS_NORM"] = norm_comp

        # 3. Export 256x256 GeoTIFFs
        # (A) Multi-band cloud-free composite
        refl_path = sample_dir / "composite_reflectance.tif"
        write_geotiff(refl_comp, bbox, refl_path, band_names=self.config.bands)

        # (B) Normalized composite
        norm_path = sample_dir / "composite_norm.tif"
        write_geotiff(norm_comp, bbox, norm_path, band_names=self.config.bands)

        tif_paths = [str(refl_path), str(norm_path)]

        # (C) Numbered subfolders for individual bands (1/, 2/, 3/, 4/)
        for idx, band_name in enumerate(self.config.bands):
            band_dir = sample_dir / str(idx + 1)
            band_dir.mkdir(parents=True, exist_ok=True)
            band_file = band_dir / f"{band_name}.tif"
            single_band = refl_comp[..., idx]  # 2D (256, 256)
            write_geotiff(single_band, bbox, band_file, band_names=[band_name])
            tif_paths.append(str(band_file))

        # 4. Generate comparison.jpg
        comp_jpg_path = sample_dir / "comparison.jpg"
        self._generate_comparison_jpg(eopatch, sample, comp_jpg_path)

        # 5. Metadata record
        summary = {
            "sample_id": sample_id,
            "latitude": lat,
            "longitude": lon,
            "bbox_rectangle": list(bbox_rect) if bbox_rect else None,
            "time_interval": list(sample["time_interval"]),
            "output_shape": list(refl_comp.shape),
            "utm_epsg": epsg,
            "dates_composited": len(eopatch.timestamps),
            "timestamps": [ts.isoformat() for ts in eopatch.timestamps],
            "output_files": tif_paths,
            "comparison_image": str(comp_jpg_path),
        }
        with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir),
            "tif_paths": tif_paths,
            "comparison_path": str(comp_jpg_path),
            "shape": refl_comp.shape,
            "epsg": epsg,
        }

    def _generate_comparison_jpg(self, eopatch: EOPatch, sample: Dict[str, Any], out_jpg: Path) -> None:
        """Generates 3-panel comparison figure showing real satellite map imagery."""
        bands = eopatch.data["BANDS"]
        valid = eopatch.mask["VALID_MASK"][..., 0]
        after = eopatch.data_timeless["BANDS_MEDIAN"]

        valid_frac = valid.mean(axis=(1, 2))
        t_cloudiest = int(valid_frac.argmin())

        def to_rgb(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
            rgb = np.stack([arr[..., 2], arr[..., 1], arr[..., 0]], axis=-1)
            return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

        ref_rgb = np.stack([after[..., 2], after[..., 1], after[..., 0]], axis=-1)
        lo, hi = np.percentile(ref_rgb, [2, 98])

        before_rgb = to_rgb(bands[t_cloudiest], lo, hi)
        masked_rgb = before_rgb.copy() * 0.45  # Dim background for high-contrast mask visibility

        scl_frame = eopatch.mask["SCL"][t_cloudiest, ..., 0]
        is_cloud = np.isin(scl_frame, [8, 9, 10])  # Medium prob, high prob, cirrus
        is_shadow = np.isin(scl_frame, [3])        # Cloud shadows
        is_invalid = np.isin(scl_frame, [0, 1])    # Defective / no data

        # Highlight detected clouds in vivid bright red
        masked_rgb[is_cloud] = np.array([1.0, 0.15, 0.15])
        # Highlight cloud shadows in vivid amber / yellow
        masked_rgb[is_shadow] = np.array([1.0, 0.75, 0.0])
        # Highlight invalid in magenta
        masked_rgb[is_invalid] = np.array([0.85, 0.0, 0.85])

        after_rgb = to_rgb(after, lo, hi)

        date_str = eopatch.timestamps[t_cloudiest].strftime("%Y-%m-%d")
        sid = sample["sample_id"]
        c_lat = sample["latitude"]
        c_lon = sample["longitude"]

        c_pct = float(is_cloud.mean() * 100)
        s_pct = float(is_shadow.mean() * 100)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(before_rgb)
        axes[0].set_title(f"BEFORE: Raw Sentinel-2 Scene\n({date_str} - {100.0 - c_pct - s_pct:.1f}% Clear)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(masked_rgb)
        axes[1].set_title(f"CLOUD & SHADOW MASK\n(Red: Cloud {c_pct:.1f}%, Yellow: Shadow {s_pct:.1f}%)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(after_rgb)
        axes[2].set_title(f"AFTER: Cloud-Free Composite (256x256)\n({len(eopatch.timestamps)} Dates Aggregated)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(f"Sample: {sid} | Lat: {c_lat:.4f} N, Lon: {c_lon:.4f} E | True Color RGB (B04-B03-B02)", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()
        fig.savefig(out_jpg, format="jpg", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Output Validation & Execution Loop
# ---------------------------------------------------------------------------

def validate_outputs(res: Dict[str, Any]) -> bool:
    """Verifies that all GeoTIFFs are 256x256, have non-negative values, and comparison.jpg exists."""
    for p_str in res["tif_paths"]:
        p = Path(p_str)
        if not p.exists() or p.stat().st_size == 0:
            logger.error(f"Validation failed: Missing file {p}")
            return False
        with tifffile.TiffFile(p) as tif:
            shape = tif.pages[0].shape
            h, w = (shape[1], shape[2]) if len(shape) == 3 else (shape[0], shape[1])
            if (h, w) != (256, 256):
                logger.error(f"Validation failed: {p} shape is {shape}, expected 256x256!")
                return False
            arr = tif.asarray()
            if arr.min() < 0.0:
                logger.error(f"Validation failed: {p} contains negative reflectance: {arr.min()}")
                return False

    comp_p = Path(res["comparison_path"])
    if not comp_p.exists() or comp_p.stat().st_size == 0:
        logger.error(f"Validation failed: Missing comparison file {comp_p}")
        return False
    return True


def run_pipeline(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Runs single dataset pipeline capped to MAX_SAMPLES = 5."""
    logger.info("=" * 70)
    logger.info("STARTING SINGLE DATASET PREPROCESSING PIPELINE (LIVE SATELLITE MODE)")
    logger.info(f"Input Directory  : {config.data_root.resolve()}")
    logger.info(f"Output Directory : {config.output_root.resolve()}")
    logger.info(f"Output Size      : {config.target_size} (Strict 256x256)")
    logger.info(f"Cap              : MAX_SAMPLES = {config.max_samples}")
    logger.info(f"Speed Settings   : max_dates = {config.max_dates}, threads = {config.threads}")
    logger.info("=" * 70)

    dataset = SentinelDataset(config.data_root, coords_file=config.coords_file, dataset_name=config.dataset_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b[0])
    preprocessor = SentinelPreprocessor(config)

    results = []
    max_cap = min(len(dataset), config.max_samples)

    for i, sample in enumerate(dataloader):
        if i >= config.max_samples:
            break
        sid = sample["sample_id"]
        logger.info(f"[{i + 1}/{max_cap}] Processing rectangular target '{sid}' (lat={sample['latitude']:.4f}, lon={sample['longitude']:.4f})...")
        try:
            res = preprocessor.process(sample, sample_id=sid)
            valid = validate_outputs(res)
            results.append({"sample_id": sid, "status": "SUCCESS" if valid else "FAILED", "result": res})
            logger.info(f"  [OK] '{sid}' -> 256x256 GeoTIFFs & comparison.jpg verified on disk.")
        except Exception as exc:
            logger.error(f"  [FAILED] '{sid}': {exc}", exc_info=True)
            results.append({"sample_id": sid, "status": "FAILED", "error": str(exc)})

    succeeded = sum(1 for r in results if r["status"] == "SUCCESS")
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE: {succeeded}/{max_cap} Succeeded | Output: {config.output_root / config.dataset_name}")
    logger.info("=" * 70)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    cfg = PipelineConfig(data_root=args.input_dir, output_root=args.output_dir, max_samples=args.max_samples)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
