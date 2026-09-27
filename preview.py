"""Quick-look PNGs of preprocessed tiles: true colour (B04,B03,B02) + false colour (B08,B04,B03).

    python preview.py            # reads data/tiffs/*_reflectance.tif -> data/previews/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def stretch(x, lo=2, hi=98):
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / max(b - a, 1e-6), 0, 1)


def to_png(img, path):
    rgb = np.dstack([stretch(img[..., i]) for i in range(3)])
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)


def main(tiff_dir="data/tiffs", out_dir="data/previews", pattern="*_reflectance.tif"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(tiff_dir).glob(pattern))
    if not files:
        sys.exit(f"No {pattern} in {Path(tiff_dir).resolve()} - run run_pipeline.py first.")
    for f in files:
        with tifffile.TiffFile(f) as tif:
            bands = json.loads(tif.pages[0].description)["bands"]
            arr = tif.asarray()  # [C,H,W]
        ix = {b: i for i, b in enumerate(bands)}
        stem = f.stem.replace("_reflectance", "")
        to_png(np.stack([arr[ix["B04"]], arr[ix["B03"]], arr[ix["B02"]]], -1), out / f"{stem}_truecolor.png")
        if "B08" in ix:
            to_png(np.stack([arr[ix["B08"]], arr[ix["B04"]], arr[ix["B03"]]], -1), out / f"{stem}_falsecolor.png")
        print(f"wrote previews for {stem} -> {out.resolve()}")


if __name__ == "__main__":
    main()
