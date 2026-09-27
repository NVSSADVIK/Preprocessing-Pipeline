"""Re-cut the whole-AOI mosaic into model-input patches. NO API calls (0 credits).

    python make_patches.py                                  # 256x256, no overlap, [0,1] normalisation
    python make_patches.py --size 256 --overlap 32 --norm zscore --save-tiffs
"""
import argparse
from pathlib import Path

from s2prep.patchify import patchify_mosaic
from s2prep.settings import PipelineSettings


def main():
    d = PipelineSettings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=d.out_dir)
    ap.add_argument("--size", type=int, default=d.patch_size)
    ap.add_argument("--overlap", type=int, default=d.patch_overlap)
    ap.add_argument("--norm", choices=["01", "zscore", "none"], default=d.norm)
    ap.add_argument("--ref-max", type=float, default=d.reflectance_max, help="clip point for --norm 01")
    ap.add_argument("--pad", choices=["reflect", "constant"], default=d.pad_mode)
    ap.add_argument("--save-tiffs", action="store_true")
    ap.add_argument("--no-png", action="store_true", help="skip PNG export of the patches")
    a = ap.parse_args()
    shape, meta = patchify_mosaic(a.data / "tiffs" / "mosaic_reflectance.tif", a.data / "patches", a.size, a.overlap,
                                  a.norm, a.ref_max, a.pad, a.save_tiffs, not a.no_png)
    print(f"{meta['n_patches']} patches {shape} ({meta['n_rows']} rows x {meta['n_cols']} cols) -> {a.data / 'patches'}")


if __name__ == "__main__":
    main()
