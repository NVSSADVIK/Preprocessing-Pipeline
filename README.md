# Sentinel-2 Dataset Preprocessing Pipeline

Automated Sentinel-2 L2A satellite retrieval, cloud/noise masking (`eo-learn`), temporal median compositing, and georeferenced **256x256** GeoTIFF export for super-resolution training.

---

## 1. Directory Structure

```
v3/
├── input/                             # Simple input directory
│   ├── coordinates.csv                # Single dataset target coordinates
│   ├── dataset_a/coordinates.csv      # Multi-dataset: Urban / Agricultural
│   ├── dataset_b/forestry_timeseries.json # Multi-dataset: Forestry time-series
│   └── dataset_c/highlands.geojson    # Multi-dataset: Highlands FeatureCollection
│
├── output/                            # Simple output directory
│   ├── single_dataset/
│   │   ├── coord_0001/ ... coord_0005/
│   │   │   ├── composite_reflectance.tif  (4-band cloud-free, 256x256)
│   │   │   ├── composite_norm.tif         (4-band normalized, 256x256)
│   │   │   ├── 1/B02.tif                  (Blue band, 256x256)
│   │   │   ├── 2/B03.tif                  (Green band, 256x256)
│   │   │   ├── 3/B04.tif                  (Red band, 256x256)
│   │   │   ├── 4/B08.tif                  (NIR band, 256x256)
│   │   │   ├── comparison.jpg             (3-panel preview: Before / Mask / After)
│   │   │   └── metadata.json
│   ├── dataset_a/ (5 samples)
│   ├── dataset_b/ (5 samples, time series flattened into single TIFFs)
│   └── dataset_c/ (5 samples)
│
├── s2prep/                            # Core preprocessing tasks
├── .env                               # Credentials (CDSE / Sentinel Hub OAuth)
├── .gitignore                         # Security rules
├── requirements.txt
├── single_dataset_pipeline.py         # Single dataset execution script
├── 01_single_dataset_pipeline.ipynb   # Single dataset notebook (executed)
├── multi_dataset_pipeline.py          # Multi-dataset execution script
├── 02_multi_dataset_pipeline.ipynb    # Multi-dataset notebook (executed)
└── AGENTS.md                          # Tracking & verification log
```

---

## 2. Quickstart

### Installation
```bash
pip install -r requirements.txt
```

### Run Single Dataset Pipeline (5 samples)
```bash
python single_dataset_pipeline.py
```
Or open and run [`01_single_dataset_pipeline.ipynb`](01_single_dataset_pipeline.ipynb).

### Run Multi-Dataset Pipeline (5 samples per dataset)
```bash
python multi_dataset_pipeline.py
```
Or open and run [`02_multi_dataset_pipeline.ipynb`](02_multi_dataset_pipeline.ipynb).

---

## 3. Changing Coordinates / Inputs

Edit [`input/coordinates.csv`](input/coordinates.csv):
```csv
sample_id,name,lat,lon,min_lon,min_lat,max_lon,max_lat,terrain,time_start,time_end
coord_0001,My_Location,11.0168,76.9558,76.9443,11.0053,76.9673,11.0283,urban,2025-11-01,2026-02-28
```

---

## 4. Security & Credentials (Optional)
If live Copernicus Data Space Ecosystem (CDSE) downloads are desired, set your credentials in `.env`:
```bash
SENTINEL_HUB_CLIENT_ID="your_client_id"
SENTINEL_HUB_CLIENT_SECRET="your_client_secret"
```
*(If omitted, the preprocessor runs in offline coordinate synthesis mode with unique landscape and cloud properties per coordinate).*
