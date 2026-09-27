"""Custom EOTasks: cloud masking -> date filtering -> median composite -> normalisation."""
import warnings

import numpy as np
from eolearn.core import EOTask

# Sentinel-2 L2A Scene Classification (SCL) classes
SCL_CLOUDY = (3, 8, 9, 10)  # cloud shadow, cloud medium/high prob, thin cirrus
SCL_INVALID = (0, 1)  # no data, saturated/defective


class ComputeValidMaskTask(EOTask):
    """MASK/VALID_MASK[t,h,w,1] = True where the pixel is usable (has data, not cloud/shadow/cirrus)."""

    def execute(self, eopatch):
        scl = eopatch.mask["SCL"]
        has_data = eopatch.mask["dataMask"].astype(bool)
        bad = np.isin(scl, SCL_CLOUDY + SCL_INVALID)
        eopatch.mask["VALID_MASK"] = has_data & ~bad
        return eopatch


class FilterByValidFractionTask(EOTask):
    """Drop timestamps whose valid-pixel fraction is below a threshold."""

    def __init__(self, min_valid_fraction=0.6):
        self.min_valid_fraction = min_valid_fraction

    def execute(self, eopatch):
        frac = eopatch.mask["VALID_MASK"].mean(axis=(1, 2, 3))
        keep = np.flatnonzero(frac >= self.min_valid_fraction)
        if keep.size == 0:
            raise ValueError(
                f"No acquisition with >= {self.min_valid_fraction:.0%} valid pixels "
                f"(best was {frac.max():.0%}). Widen time_interval or relax min_valid_fraction/maxcc."
            )
        eopatch = eopatch.temporal_subset(keep.tolist())  # subsets every temporal feature + timestamps together
        return eopatch


class MedianCompositeTask(EOTask):
    """Cloud-free median over time -> DATA_TIMELESS/BANDS_MEDIAN (+ MASK_TIMELESS/VALID_COUNT)."""

    def execute(self, eopatch):
        bands = eopatch.data["BANDS"]  # [t,h,w,c]
        valid = eopatch.mask["VALID_MASK"]  # [t,h,w,1]
        masked = np.where(valid, bands, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices are handled below
            comp = np.nanmedian(masked, axis=0)
        holes = np.isnan(comp)  # pixels that were cloudy at every date -> fall back to plain median
        if holes.any():
            comp = np.where(holes, np.median(bands, axis=0), comp)
        eopatch.data_timeless["BANDS_MEDIAN"] = comp.astype(np.float32)
        eopatch.mask_timeless["VALID_COUNT"] = valid.sum(axis=0).astype(np.uint8)
        return eopatch


class NormalizeTask(EOTask):
    """Reflectance [0, reflectance_max] -> [0, 1] float32 (DATA_TIMELESS/BANDS_NORM)."""

    def __init__(self, reflectance_max=0.3):
        self.reflectance_max = reflectance_max

    def execute(self, eopatch):
        x = eopatch.data_timeless["BANDS_MEDIAN"]
        eopatch.data_timeless["BANDS_NORM"] = (np.clip(x, 0, self.reflectance_max) / self.reflectance_max).astype(
            np.float32
        )
        return eopatch
