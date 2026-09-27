"""Before/after: raw cloudiest date vs cloud-masked median composite. Reads saved EOPatch, NO API calls (0 credits).

    python compare.py                    # tile_0000, cloudiest kept date
    python compare.py --tile tile_0003 --date 5   # pick a specific date index
"""
import argparse
from pathlib import Path

import numpy as np
from eolearn.core import EOPatch
from PIL import Image, ImageDraw, ImageFont


def to_rgb(x, lo, hi):
    """x[..., bands B02,B03,B04,...] reflectance -> uint8 RGB with a shared stretch."""
    img = np.stack([x[..., 2], x[..., 1], x[..., 0]], -1)
    return (np.clip((img - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def label(img, text):
    pad = 34
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + pad), "white")
    canvas.paste(Image.fromarray(img), (0, pad))
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # old Pillow
        font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((8, 7), text, fill="black", font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", default="tile_0000")
    ap.add_argument("--date", type=int, default=None, help="time index for 'before' (default: cloudiest kept date)")
    ap.add_argument("--data", type=Path, default=Path("data"))
    a = ap.parse_args()

    p = EOPatch.load(str(a.data / "eopatches" / a.tile))
    bands, valid = p.data["BANDS"], p.mask["VALID_MASK"][..., 0]
    after = p.data_timeless["BANDS_MEDIAN"]
    frac = valid.mean(axis=(1, 2))
    t = int(frac.argmin()) if a.date is None else a.date

    ref = np.stack([after[..., 2], after[..., 1], after[..., 0]], -1)
    lo, hi = np.percentile(ref, [2, 98])  # SAME stretch for both panels -> fair comparison

    before_rgb = to_rgb(bands[t], lo, hi)
    masked = before_rgb.copy()
    bad = ~valid[t]
    masked[bad] = (0.5 * masked[bad] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    after_rgb = to_rgb(after, lo, hi)

    date = p.timestamps[t].strftime("%Y-%m-%d")
    panels = [
        label(before_rgb, f"BEFORE: raw {date} ({frac[t]:.0%} valid)"),
        label(masked, "Cloud / shadow / no-data mask (red)"),
        label(after_rgb, f"AFTER: median of {len(p.timestamps)} dates"),
    ]
    gap = 8
    W = sum(x.width for x in panels) + gap * 2
    out = Image.new("RGB", (W, panels[0].height), "white")
    x = 0
    for pn in panels:
        out.paste(pn, (x, 0))
        x += pn.width + gap
    dest = a.data / "previews"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{a.tile}_before_after.png"
    out.save(path)
    print(f"saved {path.resolve()}")
    print(f"before: {date}, {frac[t]:.0%} valid pixels | after: {len(p.timestamps)} dates, "
          f"{(p.mask_timeless['VALID_COUNT'][..., 0] == 0).mean():.2%} pixels never valid")


if __name__ == "__main__":
    main()
