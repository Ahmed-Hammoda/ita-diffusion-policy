"""Min-max normalisation of states and actions to [-1, 1].

Diffusion works best when the target (actions) lives in a fixed range like [-1, 1].
The SAME statistics must be used at training time and in the simulator at evaluation time,
so they are saved to disk next to the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class MinMaxNormalizer:
    def __init__(self, min_val, max_val, eps: float = 1e-6):
        self.min = torch.as_tensor(np.asarray(min_val), dtype=torch.float32)
        self.max = torch.as_tensor(np.asarray(max_val), dtype=torch.float32)
        self.eps = eps

    @classmethod
    def from_lerobot_stats(cls, stats: dict, key: str) -> "MinMaxNormalizer":
        """Build from `dataset.meta.stats`, e.g. key='action' or 'observation.state'."""
        return cls(stats[key]["min"], stats[key]["max"])

    def _range(self, x: torch.Tensor):
        lo, hi = self.min.to(x.device), self.max.to(x.device)
        return lo, (hi - lo).clamp_min(self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        lo, rng = self._range(x)
        return 2.0 * (x - lo) / rng - 1.0

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        lo, rng = self._range(x)
        return (x + 1.0) / 2.0 * rng + lo

    def state_dict(self) -> dict:
        return {"min": self.min.tolist(), "max": self.max.tolist()}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MinMaxNormalizer":
        d = json.loads(Path(path).read_text())
        return cls(d["min"], d["max"])
