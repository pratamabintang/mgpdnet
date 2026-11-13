"""Dataloaders for few-shot segmentation."""

from .landslide import (
    DEFAULT_DATA_DIR,
    LandslideFewShotDataset,
    episodic_collate,
    get_dataloader,
    get_label_names,
    get_norm_params,
)

__all__ = [
    "DEFAULT_DATA_DIR",
    "LandslideFewShotDataset",
    "episodic_collate",
    "get_dataloader",
    "get_label_names",
    "get_norm_params",
]
