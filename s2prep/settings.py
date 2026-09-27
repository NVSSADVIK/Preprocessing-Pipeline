from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineSettings:
    aoi_path: Path = Path("aoi/example_aoi.geojson")
    time_interval: tuple = ("2025-11-01", "2026-02-28")
    bands: tuple = ("B02", "B03", "B04", "B08")  # native 10 m bands (B,G,R,NIR)
    resolution: int = 10  # metres/pixel
    tile_size_m: int = 5120  # 5120 m / 10 m = 512 px tiles (must be multiple of resolution)
    maxcc: float = 0.3  # catalog-level scene cloud cover filter
    min_valid_fraction: float = 0.6  # per-timestamp valid-pixel share to keep a date
    reflectance_max: float = 0.3  # clip point for [0,1] normalisation
    patch_size: int = 256  # model-input patch size (px)
    patch_overlap: int = 0  # px shared between neighbouring patches (stride = size - overlap)
    norm: str = "01"  # "01" (clip/scale by reflectance_max), "zscore" (per-band), or "none"
    pad_mode: str = "reflect"  # how edge patches are filled: "reflect" or "constant" (zeros)
    save_patch_tiffs: bool = False  # also write every patch as its own georeferenced GeoTIFF
    max_dates: int = 8  # download only the N clearest dates (0 = all) - biggest speed lever
    threads: int = 3  # parallel downloads per tile
    save_png: bool = True  # PNG previews of mosaic + patches
    workers: int = 1
    max_tiles: int = 0  # 0 = all tiles
    out_dir: Path = Path("data")

    @property
    def eopatch_dir(self):
        return self.out_dir / "eopatches"

    @property
    def cache_dir(self):
        return self.out_dir / "cache"

    @property
    def tiff_dir(self):
        return self.out_dir / "tiffs"

    @property
    def patch_dir(self):
        return self.out_dir / "patches"

    @property
    def log_dir(self):
        return self.out_dir / "logs"
