"""rasterio-free replacement for eolearn.io.SentinelHubInputTask.

Why: `import eolearn.io` imports rasterio at package level, so if rasterio's DLLs are blocked
(Windows Application Control / Smart App Control) nothing from eolearn.io can be used.
This task uses only sentinelhub-py (Catalog + Process API) and eolearn.core, and produces the
same EOPatch layout: DATA/BANDS, MASK/SCL, MASK/dataMask, timestamps.
"""
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from eolearn.core import EOPatch, EOTask
from sentinelhub import MimeType, SentinelHubCatalog, SentinelHubRequest, bbox_to_dimensions


def _evalscript(bands):
    names = [*bands, "SCL", "dataMask"]
    inputs = ", ".join(f'"{b}"' for b in names)
    outputs = ", ".join(f"s.{b}" for b in names)
    return (
        "//VERSION=3\n"
        f'function setup() {{ return {{ input: [{inputs}], output: {{ bands: {len(names)}, sampleType: "FLOAT32" }} }}; }}\n'
        f"function evaluatePixel(s) {{ return [{outputs}]; }}\n"
    )


def _parse_utc_naive(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None)


class SentinelHubDownloadTask(EOTask):
    def __init__(self, data_collection, config, bands, resolution=10, maxcc=0.3,
                 time_difference=dt.timedelta(minutes=120), cache_folder=None, max_dates=0, threads=3):
        self.data_collection = data_collection
        self.config = config
        self.bands = list(bands)
        self.resolution = resolution
        self.maxcc = maxcc
        self.time_difference = time_difference
        self.cache_folder = str(cache_folder) if cache_folder else None
        self.max_dates = max_dates  # keep only the N least-cloudy acquisitions (0 = all)
        self.threads = max(1, threads)

    def _find_groups(self, bbox, time_interval):
        """Catalog search -> acquisition times, merging same-pass granules (< time_difference apart)."""
        catalog = SentinelHubCatalog(config=self.config)
        results = catalog.search(
            self.data_collection, bbox=bbox, time=time_interval,
            filter=f"eo:cloud_cover < {self.maxcc * 100:g}",
        )
        clouds = {}
        for item in results:
            props = item["properties"]
            clouds[_parse_utc_naive(props["datetime"])] = props.get("eo:cloud_cover", 100)
        groups = []
        for t in sorted(clouds):
            if groups and t - groups[-1][-1] < self.time_difference:
                groups[-1].append(t)
            else:
                groups.append([t])
        if self.max_dates and len(groups) > self.max_dates:
            # Sort candidate groups by cloud cover
            sorted_groups = sorted(groups, key=lambda g: min(clouds[t] for t in g))
            # Pick (max_dates - 1) clearest dates for composite + 1 representative cloudy date for mask evaluation
            selected = sorted(sorted_groups[: self.max_dates - 1] + [sorted_groups[-1]], key=lambda g: g[0])
            groups = selected
        return groups

    def execute(self, eopatch=None, *, bbox=None, time_interval=None):
        groups = self._find_groups(bbox, time_interval)
        if not groups:
            raise ValueError(
                f"Catalog returned no scenes with cloud cover < {self.maxcc:.0%} for {time_interval}. "
                "Widen the time interval or raise --maxcc."
            )
        size = bbox_to_dimensions(bbox, resolution=self.resolution)
        script = _evalscript(self.bands)

        def fetch(g):
            window = (g[0] - dt.timedelta(seconds=1), g[-1] + dt.timedelta(seconds=1))
            request = SentinelHubRequest(
                evalscript=script,
                input_data=[SentinelHubRequest.input_data(data_collection=self.data_collection, time_interval=window)],
                responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                bbox=bbox,
                size=size,
                config=self.config,
                data_folder=self.cache_folder,
            )
            return request.get_data(save_data=bool(self.cache_folder))[0]

        print(f"  {len(groups)} dates, {size[0]}x{size[1]} px -> downloading with {self.threads} threads", flush=True)
        frames = [fetch(groups[0])]  # first call alone: creates the OAuth session before threads share it
        if len(groups) > 1:
            with ThreadPoolExecutor(max_workers=self.threads) as pool:
                frames += list(pool.map(fetch, groups[1:]))

        arr = np.stack(frames)  # [t, h, w, n_bands + 2]
        nb = len(self.bands)
        out = EOPatch(bbox=bbox)
        out.timestamps = [g[0] for g in groups]
        out.data["BANDS"] = arr[..., :nb].astype(np.float32)
        out.mask["SCL"] = np.rint(arr[..., nb : nb + 1]).astype(np.uint8)
        out.mask["dataMask"] = arr[..., nb + 1 : nb + 2] > 0.5
        return out
