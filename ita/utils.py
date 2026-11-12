"""Small shared helpers: config loading, seeding, device, LeRobot import."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = PROJECT_ROOT / "configs" / "default.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def import_lerobot():
    """Return (LeRobotDataset, LeRobotDatasetMetadata); the module path moved between LeRobot versions."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    return LeRobotDataset, LeRobotDatasetMetadata


def task_list(meta) -> list[str]:
    """List the language instructions from LeRobot metadata (format differs between dataset versions)."""
    tasks = meta.tasks
    if hasattr(tasks, "index") and hasattr(tasks, "columns"):  # pandas DataFrame indexed by task string
        return [str(t) for t in tasks.index]
    if isinstance(tasks, dict):
        return [str(t) for t in tasks.values()]
    return [str(t) for t in tasks]


def chw_to_hwc_uint8(img: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a LeRobot image (C,H,W float in [0,1]) to (H,W,C) uint8 for saving/plotting."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    return img
