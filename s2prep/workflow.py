from datetime import timedelta

from eolearn.core import EOWorkflow, OverwritePermission, SaveTask, linearly_connect_tasks

from .download import SentinelHubDownloadTask
from .tasks import ComputeValidMaskTask, FilterByValidFractionTask, MedianCompositeTask, NormalizeTask


def build_workflow(settings, sh_config, collection):
    """Returns (workflow, input_node, save_node)."""
    download = SentinelHubDownloadTask(
        data_collection=collection,
        config=sh_config,
        bands=settings.bands,
        resolution=settings.resolution,
        maxcc=settings.maxcc,
        time_difference=timedelta(minutes=120),  # merge same-pass acquisitions
        cache_folder=settings.cache_dir,
        max_dates=settings.max_dates,
        threads=settings.threads,
    )
    save = SaveTask(str(settings.eopatch_dir), overwrite_permission=OverwritePermission.OVERWRITE_FEATURES)

    nodes = linearly_connect_tasks(
        download,
        ComputeValidMaskTask(),
        FilterByValidFractionTask(settings.min_valid_fraction),
        MedianCompositeTask(),
        NormalizeTask(settings.reflectance_max),
        save,
    )
    return EOWorkflow(nodes), nodes[0], nodes[-1]
