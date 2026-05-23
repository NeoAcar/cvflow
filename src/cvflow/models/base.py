from __future__ import annotations

from typing import Protocol

import numpy as np


class FlowModel(Protocol):
    name: str

    def predict(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """img1, img2: uint8 HxWx3. Returns float32 HxWx2 flow (frame1 -> frame2)."""
        ...
