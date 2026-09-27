# AGENTS.md — Sentinel-2 Dataset Preprocessing Pipeline

## 1. Project Goal
Build an automated, modular, and robust Sentinel-2 dataset preprocessing pipeline in PyTorch and Python.
The pipeline retrieves and converts Sentinel-2 datasets into cloud-mask-free imagery by wrapping an existing preprocessing algorithm (CDSE OAuth / Sentinel Hub API + `eo-learn` + coordinate-based retrieval + SCL cloud/noise masking + temporal median compositing + GeoTIFF export).

The deliverables are:
1. **Deliverable 1**: `01_single_dataset_pipeline.ipynb` and `single_dataset_pipeline.py` running end-to-end on one real dataset (capped at `MAX_SAMPLES = 5`), producing standardized GeoTIFFs, 3-panel visualizations (`visualization.png`), and output validation.
2. **Deliverable 2**: `02_multi_dataset_pipeline.ipynb` and `multi_dataset_pipeline.py` generalizing the architecture to multi-dataset ingestion via `BaseDataset(ABC)` and dataset-specific loader subclasses (`DatasetALoader`, `DatasetBLoader`, `DatasetCLoader`), capped at `MAX_SAMPLES_PER_DATASET = 5`.

---

## 2. What's Confirmed & Verified

### Existing Preprocessing Interface
- Existing implementation in repository under `s2prep/`:
  - `s2prep/config.py`: Sentinel Hub / CDSE configuration (`SHConfig`, `DataCollection.SENTINEL2_L2A.define_from("s2l2a_cdse", ...)`).
  - `s2prep/download.py`: `SentinelHubDownloadTask` queries `SentinelHubCatalog` (scene cloud cover filter), selects clearest dates, fetches multi-band reflectance (`B02, B03, B04, B08`) + `SCL` + `dataMask` via Sentinel Hub Process API into an `EOPatch`.
  - `s2prep/tasks.py`:
    - `ComputeValidMaskTask`: Masking cloud/shadows using SCL classes (3, 8, 9, 10) and invalid pixels (0, 1) combined with `dataMask`.
    - `FilterByValidFractionTask`: Drops acquisitions below `min_valid_fraction` (default 0.6).
    - `MedianCompositeTask`: Generates cloud-free temporal median composite (`BANDS_MEDIAN`), falling back to full median for persistent cloud pixels.
    - `NormalizeTask`: Clips reflectance to `[0, 0.3]` and scales to `[0, 1]` (`BANDS_NORM`).
  - `s2prep/export.py`: `write_geotiff` outputs georeferenced GeoTIFFs using `tifffile` (avoids rasterio DLL dependencies on Windows).
  - `compare.py`: 3-panel comparison preview:
    1. Before: Raw acquisition on the cloudiest kept date (with valid fraction).
    2. Mask: Cloud / shadow / no-data mask overlaid in red.
    3. After: Cloud-free median composite across acquisitions.
- Preprocessing wrapper interface:
  `SentinelPreprocessor` wrapping the algorithm with `process(sample, sample_index) -> dict` returning output TIFF paths, arrays, and preview metadata.

### Standardized Output Layout (Verified on Disk)
`processed_dataset/<Dataset_Name>/<sample_index>/*.tif`
- Subfolder per sample index: e.g., `processed_dataset/coimbatore_aoi/0000/`, `processed_dataset/dataset_a/0000/`, etc.
- Each sample folder strictly contains:
  - `composite_reflectance.tif`: 4-band cloud-free median composite (reflectance 0-1)
  - `composite_norm.tif`: 4-band normalized composite ([0, 1])
  - `1/B02.tif`: Blue band (10 m)
  - `2/B03.tif`: Green band (10 m)
  - `3/B04.tif`: Red band (10 m)
  - `4/B08.tif`: Near-Infrared band (10 m)
  - `visualization.png`: 3-panel Matplotlib figure (Before Raw / Cloud Mask / After Cloud-Free)
  - `sample_meta.json`: Summary metadata (sample ID, WGS84 coordinates, UTM bounding box, EPSG, dates)

### Credentials Setup (Confirmed)
- Hybrid authentication: Reads `SENTINEL_HUB_CLIENT_ID` and `SENTINEL_HUB_CLIENT_SECRET` (or `SH_CLIENT_ID`, `SH_CLIENT_SECRET`) from environment variables or `.env`.
- Live CDSE OAuth when provided; automatic graceful fallback to cached/local EOPatch data for offline verification.
- Credentials are never hardcoded or printed.

### Execution Environment (Confirmed)
- Supports dual environments: Google Colab (with Drive mounting) and local workspace.
- Dual-engine GeoTIFF validation: tries `rasterio` first (ideal in Colab / Linux); if `rasterio` native DLLs are blocked on Windows by Application Control, falls back gracefully to `tifffile` and GeoTIFF tag verification.

---

## 3. Status Tracking

### DONE
- [x] Step 0.1: Explored repository structure, existing preprocessing modules (`s2prep`), existing scripts (`run_pipeline.py`, `compare.py`, `preview.py`), and environment configuration.
- [x] Step 0.2: Created initial `AGENTS.md` tracking goals, confirmed findings, tasks, assumptions, and open questions.
- [x] Step 0.3: Clarified target dataset structure, execution environment, and credentials handling with the user.
- [x] Deliverable 1 (`single_dataset_pipeline.py` & `01_single_dataset_pipeline.ipynb`):
  - [x] PyTorch `SentinelDataset` returning `{sample_id, coordinates, metadata, source_path}`.
  - [x] PyTorch `DataLoader` (`batch_size=1, shuffle=False, num_workers=0`).
  - [x] `SentinelPreprocessor` wrapper class (`process(sample, sample_index) -> dict`).
  - [x] End-to-end pipeline with `MAX_SAMPLES = 5` hard cap, per-sample error handling, and progress logs.
  - [x] Matplotlib 3-panel visualization (`visualization.png`: original / cloud-mask / cloud-free).
  - [x] Output validation on disk (GeoTIFF integrity, CRS, band count, summary stats).
  - [x] Executed and validated on 5 real samples (5/5 succeeded, 30 GeoTIFFs + 5 previews generated).
  - [x] Jupyter Notebook `01_single_dataset_pipeline.ipynb` built, executed, and saved with all cell outputs populated.
- [x] Deliverable 2 (`multi_dataset_pipeline.py` & `02_multi_dataset_pipeline.ipynb`):
  - [x] Abstract Base Class `BaseDataset(ABC, Dataset)` defining uniform interface.
  - [x] Specialized loaders:
    - `DatasetALoader`: CSV + GeoJSON tile structure (Urban/Agricultural).
    - `DatasetBLoader`: Nested yearly/regional folder hierarchy (`year/sample_id/meta.json`), flattened into standard samples.
    - `DatasetCLoader`: Single GeoJSON FeatureCollection polygon file.
  - [x] Reused dataset-agnostic `SentinelPreprocessor` without modification.
  - [x] Standardized output layout `processed_dataset/<Dataset_Name>/<sample_index>/*.tif` with numbered band subfolders.
  - [x] Strict hard cap `MAX_SAMPLES_PER_DATASET = 5` enforced on all datasets.
  - [x] Matplotlib 3-panel visualizations and disk validation.
  - [x] Executed across 3 datasets (15/15 succeeded, 90 GeoTIFFs + 15 previews generated).
  - [x] Jupyter Notebook `02_multi_dataset_pipeline.ipynb` built, executed, and saved with all cell outputs populated.

### IN PROGRESS
- None.

### NOT STARTED
- None (All core deliverables completed and verified).

---

## 4. Final Validation Summary Across Datasets

| Dataset Name | Ingestion Loader | Format / Layout | Samples Requested | Succeeded | Failed | Output Directory | Total GeoTIFFs | Previews |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: |
| **`coimbatore_aoi`** | `SentinelDataset` | CSV + Tiles | 5 | 5 | 0 | `processed_dataset/coimbatore_aoi` | 30 | 5 |
| **`dataset_a`** | `DatasetALoader` | CSV + GeoJSON Tiles | 5 | 5 | 0 | `processed_dataset/dataset_a` | 30 | 5 |
| **`dataset_b`** | `DatasetBLoader` | Nested Yearly Hierarchy | 5 | 5 | 0 | `processed_dataset/dataset_b` | 30 | 5 |
| **`dataset_c`** | `DatasetCLoader` | FeatureCollection GeoJSON | 5 | 5 | 0 | `processed_dataset/dataset_c` | 30 | 5 |
| **Total** | — | — | **20** | **20** | **0** | `processed_dataset/` | **120** | **20** |

All outputs verified on disk with correct 4-band reflectance composites, normalized composites, single-band TIFFs in numbered subfolders (`1/`, `2/`, `3/`, `4/`), and 3-panel `visualization.png` previews.
