"""
multi_dataset_pipeline.py — Multi-Dataset Sentinel-2 Preprocessing Pipeline (Deliverable 2)

Features:
- Abstract Base Class: BaseDataset(ABC, Dataset) defining the universal contract.
- Specialized Loaders:
    - DatasetALoader: Ingests CSV-indexed + GeoJSON tile structure (Urban/Agricultural).
    - DatasetBLoader: Ingests nested yearly hierarchy (Time-series / Forestry) and flattens it.
    - DatasetCLoader: Ingests single FeatureCollection GeoJSON (Highlands / Topographic).
- Reusable, dataset-agnostic SentinelPreprocessor wrapping Sentinel Hub + eo-learn.
- Common standardized output tree: processed_dataset/<Dataset_Name>/<sample_index>/*.tif.
- Flattened time-series structure & numbered band subfolders (1/, 2/, 3/, 4/).
- Strict hard cap: MAX_SAMPLES_PER_DATASET = 5 enforced for all datasets.
- Matplotlib 3-panel visualization (visualization.png) per processed sample.
- Disk verification & comprehensive multi-dataset quality report.
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
from typing import Any, Dict, List, Optional, Tuple, Type

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
    validate_sample_outputs,
)

# Configure logging
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
    Abstract Base Class for Sentinel-2 dataset loaders.
    
    Each subclass handles its own unique folder structure, metadata formats,
    and hierarchy internally, but MUST return the standardized sample dictionary:
        - sample_id: str
        - coordinates: [min_lon, min_lat, max_lon, max_lat]
        - metadata: dict (time_interval, name, custom attributes)
        - source_path: str
    """

    def __init__(self, data_root: Path | str):
        self.data_root = Path(data_root)
        self.samples: List[Dict[str, Any]] = []
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset root directory does not exist: {self.data_root}")
        self._load_samples()
        logger.info(f"[{self.dataset_name}] Successfully indexed {len(self.samples)} sample(s) from {self.data_root}")

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Unique identifier name for this dataset."""
        pass

    @abstractmethod
    def _load_samples(self) -> None:
        """Internal method to discover, parse, and populate self.samples."""
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset '{self.dataset_name}' (size {len(self.samples)})")
        return self.samples[idx]


# ---------------------------------------------------------------------------
# 2. Dataset Loaders for Heterogeneous Sources
# ---------------------------------------------------------------------------

class DatasetALoader(BaseDataset):
    """
    Loader for Dataset A (Flat CSV index with tiles/ folder containing individual GeoJSONs).
    Example: Urban and agricultural satellite target regions.
    """

    @property
    def dataset_name(self) -> str:
        return "dataset_a"

    def _load_samples(self) -> None:
        csv_file = self.data_root / "samples.csv"
        tiles_dir = self.data_root / "tiles"

        if csv_file.exists():
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = row["sample_id"]
                    coords = [
                        float(row["min_lon"]),
                        float(row["min_lat"]),
                        float(row["max_lon"]),
                        float(row["max_lat"]),
                    ]
                    self.samples.append({
                        "sample_id": sid,
                        "coordinates": coords,
                        "metadata": {
                            "name": row.get("name", sid),
                            "time_interval": ("2025-11-01", "2026-02-28"),
                            "dataset_name": self.dataset_name,
                            "type": "urban_agri",
                        },
                        "source_path": row.get("file_path", str(csv_file)),
                    })
        elif tiles_dir.exists():
            for gf in sorted(tiles_dir.glob("*.geojson")):
                with open(gf, "r", encoding="utf-8") as f:
                    gj = json.load(f)
                geom = shape(gj["geometry"])
                minx, miny, maxx, maxy = geom.bounds
                props = gj.get("properties", {})
                sid = props.get("sample_id", gf.stem)
                self.samples.append({
                    "sample_id": sid,
                    "coordinates": [minx, miny, maxx, maxy],
                    "metadata": {
                        "name": props.get("name", sid),
                        "time_interval": tuple(props.get("time_interval", ["2025-11-01", "2026-02-28"])),
                        "dataset_name": self.dataset_name,
                    },
                    "source_path": str(gf),
                })
        else:
            raise FileNotFoundError(f"Neither samples.csv nor tiles/ found in {self.data_root}")


class DatasetBLoader(BaseDataset):
    """
    Loader for Dataset B (Nested Time-Series / Yearly folders: e.g. <year>/<sample_id>/metadata.json).
    Example: Tropical forestry & watershed monitoring.
    Flattens the nested time-series hierarchy into standard sample entries.
    """

    @property
    def dataset_name(self) -> str:
        return "dataset_b"

    def _load_samples(self) -> None:
        # Search for all metadata.json files in yearly subfolders
        meta_files = sorted(self.data_root.glob("*/*/metadata.json"))
        for mf in meta_files:
            with open(mf, "r", encoding="utf-8") as f:
                data = json.load(f)

            sid = data.get("sample_id", mf.parent.name)
            coords = data.get("target_bounds")
            time_window = tuple(data.get("time_window", ["2025-01-01", "2025-04-30"]))

            self.samples.append({
                "sample_id": sid,
                "coordinates": coords,
                "metadata": {
                    "name": data.get("region_name", sid),
                    "year": data.get("year"),
                    "ecosystem": data.get("ecosystem", "Forest"),
                    "time_interval": time_window,
                    "dataset_name": self.dataset_name,
                },
                "source_path": str(mf),
            })


class DatasetCLoader(BaseDataset):
    """
    Loader for Dataset C (Single FeatureCollection GeoJSON containing multiple target features).
    Example: Nilgiris Highland ecological reserve polygons.
    """

    @property
    def dataset_name(self) -> str:
        return "dataset_c"

    def _load_samples(self) -> None:
        geojson_file = self.data_root / "highlands_regions.geojson"
        if not geojson_file.exists():
            # Search for any .geojson in root
            candidates = list(self.data_root.glob("*.geojson"))
            if candidates:
                geojson_file = candidates[0]
            else:
                raise FileNotFoundError(f"No FeatureCollection GeoJSON found in {self.data_root}")

        with open(geojson_file, "r", encoding="utf-8") as f:
            fc = json.load(f)

        features = fc.get("features", [])
        for i, ft in enumerate(features):
            props = ft.get("properties", {})
            geom = shape(ft["geometry"])
            minx, miny, maxx, maxy = geom.bounds
            sid = props.get("id", f"sample_C_{i:02d}")

            self.samples.append({
                "sample_id": sid,
                "coordinates": [minx, miny, maxx, maxy],
                "metadata": {
                    "name": props.get("label", sid),
                    "elevation_m": props.get("elevation_m"),
                    "time_interval": tuple(props.get("time_interval", ["2026-01-01", "2026-03-31"])),
                    "dataset_name": self.dataset_name,
                },
                "source_path": str(geojson_file),
            })


# ---------------------------------------------------------------------------
# 3. Multi-Dataset Pipeline Runner
# ---------------------------------------------------------------------------

@dataclass
class MultiDatasetPipelineConfig(PipelineConfig):
    """Configuration for multi-dataset pipeline execution."""
    max_samples_per_dataset: int = 5  # Strict hard cap per dataset
    datasets_root: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_ROOT", "sample_datasets"))
    )


def run_multi_dataset_pipeline(
    config: MultiDatasetPipelineConfig,
    dataset_loaders: Optional[List[BaseDataset]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the end-to-end multi-dataset pipeline across all registered datasets.
    
    Guarantees:
    - Enforces hard cap MAX_SAMPLES_PER_DATASET = 5 on every dataset.
    - Re-uses dataset-agnostic SentinelPreprocessor without modification.
    - Preserves standardized output tree: processed_dataset/<Dataset_Name>/<sample_index>/*.tif.
    - Flattens time-series hierarchies (never replicates yearly folder trees in output).
    - Produces 3-panel visualization and validates all output files on disk.
    """
    logger.info("=" * 80)
    logger.info("STARTING MULTI-DATASET SENTINEL-2 PREPROCESSING PIPELINE (Deliverable 2)")
    logger.info(f"Output Root              : {config.output_root.resolve()}")
    logger.info(f"Max Samples Per Dataset  : {config.max_samples_per_dataset} (STRICT HARD CAP)")
    logger.info(f"Authentication Mode      : {'LIVE CDSE OAUTH' if config.has_live_credentials() else 'OFFLINE/CACHED FALLBACK'}")
    logger.info("=" * 80)

    # Instantiate default loaders if not provided
    if dataset_loaders is None:
        d_root = config.datasets_root
        dataset_loaders = [
            DatasetALoader(d_root / "dataset_a"),
            DatasetBLoader(d_root / "dataset_b"),
            DatasetCLoader(d_root / "dataset_c"),
        ]

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    for loader in dataset_loaders:
        d_name = loader.dataset_name
        logger.info("-" * 80)
        logger.info(f"STARTING DATASET INGESTION: '{d_name}' ({len(loader)} total samples available)")
        logger.info("-" * 80)

        # PyTorch DataLoader: batch_size=1, shuffle=False, num_workers=0
        dataloader = DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda batch: batch[0],
        )

        # Configure dataset-agnostic preprocessor for this dataset
        dataset_config = PipelineConfig(
            data_root=loader.data_root,
            output_root=config.output_root,
            dataset_name=d_name,
            sh_client_id=config.sh_client_id,
            sh_client_secret=config.sh_client_secret,
            sh_provider=config.sh_provider,
            bands=config.bands,
            resolution=config.resolution,
            maxcc=config.maxcc,
            min_valid_fraction=config.min_valid_fraction,
            reflectance_max=config.reflectance_max,
            max_samples=config.max_samples_per_dataset,
            cache_dir=config.cache_dir,
            fallback_eopatch_path=config.fallback_eopatch_path,
        )
        preprocessor = SentinelPreprocessor(dataset_config)

        dataset_results: List[Dict[str, Any]] = []
        max_cap = min(len(loader), config.max_samples_per_dataset)

        for i, sample in enumerate(dataloader):
            if i >= config.max_samples_per_dataset:
                logger.info(f"[{d_name}] Reached MAX_SAMPLES_PER_DATASET ({config.max_samples_per_dataset}). Halting loop.")
                break

            sample_id = sample["sample_id"]
            sample_index = f"{i:04d}"
            logger.info(f"[{d_name}] [{i + 1}/{max_cap}] Processing sample '{sample_id}' (index {sample_index})...")

            try:
                sample_res = preprocessor.process(sample, sample_index=sample_index)
                val = validate_sample_outputs(sample_res)

                if val["all_valid"]:
                    logger.info(f"  [OK] Successfully processed '{sample_id}'. Created {len(sample_res['tif_paths'])} TIFFs + preview.")
                else:
                    logger.warning(f"  [WARN] Processed with validation issues: {val['errors']}")

                dataset_results.append({
                    "sample_id": sample_id,
                    "status": "SUCCESS" if val["all_valid"] else "VALIDATION_FAILED",
                    "result": sample_res,
                    "validation": val,
                })
            except Exception as exc:
                logger.error(f"  [FAILED] [{d_name}] Sample '{sample_id}' failed: {exc}", exc_info=True)
                dataset_results.append({
                    "sample_id": sample_id,
                    "status": "FAILED",
                    "error": str(exc),
                })

        all_results[d_name] = dataset_results

    # -----------------------------------------------------------------------
    # Comprehensive Multi-Dataset Validation & Summary Report
    # -----------------------------------------------------------------------
    logger.info("=" * 80)
    logger.info("MULTI-DATASET PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Dataset Name':<15} {'Requested':<12} {'Succeeded':<12} {'Failed':<10} {'Output Location'}")
    logger.info("-" * 80)

    grand_total_requested = 0
    grand_total_succeeded = 0

    for d_name, d_res in all_results.items():
        req = len(d_res)
        succ = sum(1 for r in d_res if r["status"] == "SUCCESS")
        fail = req - succ
        grand_total_requested += req
        grand_total_succeeded += succ
        out_path = config.output_root / d_name
        logger.info(f"{d_name:<15} {req:<12} {succ:<12} {fail:<10} {out_path}")

    logger.info("-" * 80)
    logger.info(f"{'TOTAL':<15} {grand_total_requested:<12} {grand_total_succeeded:<12} {grand_total_requested - grand_total_succeeded:<10}")
    logger.info("=" * 80)

    return all_results


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 Multi-Dataset Preprocessing Pipeline")
    parser.add_argument("--datasets-root", type=Path, default=Path("sample_datasets"), help="Root path containing datasets")
    parser.add_argument("--output-root", type=Path, default=Path("processed_dataset"), help="Root path for standardized outputs")
    parser.add_argument("--max-samples-per-dataset", type=int, default=5, help="Cap per dataset (default: 5)")
    args = parser.parse_args()

    cfg = MultiDatasetPipelineConfig(
        datasets_root=args.datasets_root,
        output_root=args.output_root,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )

    run_multi_dataset_pipeline(cfg)


if __name__ == "__main__":
    main()
