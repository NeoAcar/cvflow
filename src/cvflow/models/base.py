from __future__ import annotations

from typing import Protocol

import numpy as np
import torch


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FlowModel(Protocol):
    name: str

    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """img1, img2: uint8 HxWx3. Returns float32 HxWx2 flow (frame1 -> frame2)."""
        ...
