# Sentinel-2 preprocessing for super-resolution (eo-learn)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # fill SH_PROVIDER / SH_CLIENT_ID / SH_CLIENT_SECRET
python check_auth.py                                   # must print OK before anything else
python run_pipeline.py --aoi aoi/example_aoi.geojson --start 2025-11-01 --end 2026-02-28 --max-tiles 2
```

Outputs (under `data/`):
- `eopatches/tile_XXXX/`  full EOPatch (BANDS time series, SCL, VALID_MASK, BANDS_MEDIAN, BANDS_NORM, VALID_COUNT)
- `tiffs/tile_XXXX_reflectance.tif`  cloud-free median composite, reflectance 0-1, georeferenced (UTM, 10 m)
- `tiffs/tile_XXXX_norm.tif`  same, clipped & scaled to [0,1] for model input
- `patches/tile_XXXX_lr_patches.npy`  (N,128,128,4) NHWC float32 LR training patches
