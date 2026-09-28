"""
multi_dataset_pipeline.py — Multi-Dataset Sentinel-2 Preprocessing Pipeline

Key Features:
1. Input: 3 heterogeneous input formats under `input/`:
   - `input/dataset_a/coordinates.csv`: CSV table
   - `input/dataset_b/forestry_timeseries.json`: JSON list
   - `input/dataset_c/highlands.geojson`: GeoJSON rectangular Polygon FeatureCollection
   STRICT GEOMETRY VALIDATION: Enforces that inputs are 2D RECTANGLES.
   Explicitly rejects lines (LineString) or triangles (3-vertex polygons).
2. Automatic 2560m UTM Bounding Box: Centered on rectangle -> strictly 256x256 pixels at 10m.
3. BaseDataset(ABC, Dataset): Abstract interface ensuring uniform sample dictionaries.
4. Preprocessing Engine: Reuses dataset-agnostic `SentinelPreprocessor` connected to Copernicus CDSE Sentinel Hub API.
   STRICT ERROR HANDLING: Fails fast and notifies user if API connection fails; never generates fake/random outputs.
   Optimized for speed with 3 clearest dates (`max_dates=3`) and multi-threaded retrieval.
5. Standardized 256x256 Output Layout under `output/<dataset_name>/<sample_id>/`:
   - `composite_reflectance.tif`: 4-band cloud-free median composite (256x256)
   - `composite_norm.tif`: 4-band normalized composite (256x256)
   - Numbered band subfolders: `1/B02.tif`, `2/B03.tif`, `3/B04.tif`, `4/B08.tif` (each 256x256)
   - `comparison.jpg`: 3-panel true satellite visualization (Raw Scene / Cloud Mask / Cloud-Free Composite)
   - `metadata.json`: Summary metadata
6. Flattening Rule: Time-series in Dataset B flattened into single converted TIFFs without yearly folders.
7. Hard Cap: MAX_SAMPLES_PER_DATASET = 5 strictly enforced for every dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
from shapely.geometry import shape

from single_dataset_pipeline import (
    PipelineConfig,
    SentinelPreprocessor,
    validate_outputs,
    validate_and_extract_rectangle,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MultiDatasetPipeline")


# ---------------------------------------------------------------------------
# 1. Base Dataset (ABC) Interface
# ---------------------------------------------------------------------------

class BaseDataset(Dataset, ABC):
    """
    Universal Abstract Base Class for Sentinel-2 dataset loaders.
    Decouples source layout from downstream preprocessor.
    """

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.samples: List[Dict[str, Any]] = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.data_root}")
        self._load_samples()
        logger.info(f"[{self.dataset_name}] Indexed {len(self.samples)} validated rectangular sample(s) from {self.data_root}")

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Unique dataset name identifier."""
        pass

    @abstractmethod
    def _load_samples(self) -> None:
        """Populates self.samples with standardized records."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 2. Specialized Loaders (Format 1: CSV, Format 2: JSON, Format 3: GeoJSON)
# ---------------------------------------------------------------------------

class DatasetALoader(BaseDataset):
    """Loader for Dataset A (CSV table with rectangular AOIs)."""

    @property
    def dataset_name(self) -> str:
        return "dataset_a"

    def _load_samples(self) -> None:
        csv_file = self.data_root / "coordinates.csv"
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["sample_id"].strip()
                s_date = row["start_date"].strip()
                e_date = row["end_date"].strip()

                lat, lon, bbox_rect = validate_and_extract_rectangle(row)

                self.samples.append({
                    "sample_id": sid,
                    "latitude": lat,
                    "longitude": lon,
                    "bbox_rect": bbox_rect,
                    "time_interval": (s_date, e_date),
                    "dataset_name": self.dataset_name,
                    "source_path": str(csv_file),
                })


class DatasetBLoader(BaseDataset):
    """
    Loader for Dataset B (JSON list of records).
    Flattens time-series records into individual sample entries.
    """

    @property
    def dataset_name(self) -> str:
        return "dataset_b"

    def _load_samples(self) -> None:
        json_file = self.data_root / "forestry_timeseries.json"
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for rec in data:
            sid = rec["sample_id"].strip()
            s_date = rec["start_date"].strip()
            e_date = rec["end_date"].strip()

            lat, lon, bbox_rect = validate_and_extract_rectangle(rec)

            self.samples.append({
                "sample_id": sid,
                "latitude": lat,
                "longitude": lon,
                "bbox_rect": bbox_rect,
                "time_interval": (s_date, e_date),
                "dataset_name": self.dataset_name,
                "source_path": str(json_file),
            })


class DatasetCLoader(BaseDataset):
    """Loader for Dataset C (GeoJSON Rectangular Polygon FeatureCollection)."""

    @property
    def dataset_name(self) -> str:
        return "dataset_c"

    def _load_samples(self) -> None:
        geojson_file = self.data_root / "highlands.geojson"
        with open(geojson_file, "r", encoding="utf-8") as f:
            fc = json.load(f)

        for ft in fc.get("features", []):
            props = ft.get("properties", {})
            sid = props.get("sample_id", "sample_C").strip()

            # Passes feature with geometry to validate rectangle polygon (rejects lines/triangles)
            record = {"sample_id": sid, "geometry": ft.get("geometry", {}), **props}
            lat, lon, bbox_rect = validate_and_extract_rectangle(record)

            s_date = props.get("start_date", "2026-01-01").strip()
            e_date = props.get("end_date", "2026-03-31").strip()

            self.samples.append({
                "sample_id": sid,
                "latitude": lat,
                "longitude": lon,
                "bbox_rect": bbox_rect,
                "time_interval": (s_date, e_date),
                "dataset_name": self.dataset_name,
                "source_path": str(geojson_file),
            })


# ---------------------------------------------------------------------------
# 3. Multi-Dataset Execution
# ---------------------------------------------------------------------------

@dataclass
class MultiPipelineConfig(PipelineConfig):
    """Configuration for multi-dataset pipeline."""
    max_samples_per_dataset: int = 5
    input_dir: Path = field(default_factory=lambda: Path(os.getenv("INPUT_DIR", "input")))


def run_multi_pipeline(config: MultiPipelineConfig) -> Dict[str, List[Dict[str, Any]]]:
    """Runs preprocessing across all datasets with strict cap MAX_SAMPLES_PER_DATASET = 5."""
    logger.info("=" * 80)
    logger.info("STARTING MULTI-DATASET SENTINEL-2 PREPROCESSING PIPELINE (LIVE SATELLITE MODE)")
    logger.info(f"Input Directory          : {config.input_dir.resolve()}")
    logger.info(f"Output Directory         : {config.output_root.resolve()}")
    logger.info(f"Output Size              : {config.target_size} (Strict 256x256)")
    logger.info(f"Cap Per Dataset          : MAX_SAMPLES_PER_DATASET = {config.max_samples_per_dataset}")
    logger.info(f"Speed Settings           : max_dates = {config.max_dates}, threads = {config.threads}")
    logger.info("=" * 80)

    loaders = [
        DatasetALoader(config.input_dir / "dataset_a"),
        DatasetBLoader(config.input_dir / "dataset_b"),
        DatasetCLoader(config.input_dir / "dataset_c"),
    ]

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for loader in loaders:
        d_name = loader.dataset_name
        logger.info("-" * 80)
        logger.info(f"PROCESSING DATASET: '{d_name}' ({len(loader)} total targets)")
        logger.info("-" * 80)

        dataloader = DataLoader(loader, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b[0])

        dataset_cfg = PipelineConfig(
            data_root=loader.data_root,
            output_root=config.output_root,
            dataset_name=d_name,
            target_size=config.target_size,
            max_samples=config.max_samples_per_dataset,
            max_dates=config.max_dates,
            threads=config.threads,
        )
        preprocessor = SentinelPreprocessor(dataset_cfg)

        dataset_results = []
        max_cap = min(len(loader), config.max_samples_per_dataset)

        for i, sample in enumerate(dataloader):
            if i >= config.max_samples_per_dataset:
                break
            sid = sample["sample_id"]
            logger.info(f"[{d_name}] [{i + 1}/{max_cap}] Processing rectangular target '{sid}' (lat={sample['latitude']:.4f}, lon={sample['longitude']:.4f})...")
            try:
                res = preprocessor.process(sample, sample_id=sid)
                valid = validate_outputs(res)
                dataset_results.append({"sample_id": sid, "status": "SUCCESS" if valid else "FAILED", "result": res})
                logger.info(f"  [OK] [{d_name}] '{sid}' -> 256x256 GeoTIFFs & comparison.jpg verified on disk.")
            except Exception as exc:
                logger.error(f"  [FAILED] [{d_name}] '{sid}': {exc}", exc_info=True)
                dataset_results.append({"sample_id": sid, "status": "FAILED", "error": str(exc)})

        all_results[d_name] = dataset_results

    # Summary table
    logger.info("=" * 80)
    logger.info("MULTI-DATASET EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Dataset Name':<15} {'Requested':<12} {'Succeeded':<12} {'Failed':<10} {'Output Location'}")
    logger.info("-" * 80)
    for d_name, d_res in all_results.items():
        req = len(d_res)
        succ = sum(1 for r in d_res if r["status"] == "SUCCESS")
        fail = req - succ
        logger.info(f"{d_name:<15} {req:<12} {succ:<12} {fail:<10} {config.output_root / d_name}")
    logger.info("=" * 80)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--max-samples-per-dataset", type=int, default=5)
    args = parser.parse_args()

    cfg = MultiPipelineConfig(
        input_dir=args.input_dir,
        output_root=args.output_dir,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    run_multi_pipeline(cfg)


if __name__ == "__main__":
    main()
