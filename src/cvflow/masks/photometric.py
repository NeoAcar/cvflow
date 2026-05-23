"""Photometric residual mask (methodology §2.5): warp I1 by GT flow, |I1 - I2(x+flow)|.

Pixels with large residual on the non-occluded GT are pixels where brightness-
constancy is violated by the data itself (illumination, shadow, specularity) —
"the dataset's fault, not the model's".
"""

from __future__ import annotations

import cv2
import numpy as np


def photometric_residual(img1: np.ndarray, img2: np.ndarray, gt_flow: np.ndarray) -> np.ndarray:
    """Return float32 HxW per-pixel residual in [0..255] intensity units (mean over RGB)."""
    h, w = img1.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xx + gt_flow[..., 0]
    map_y = yy + gt_flow[..., 1]
    warped = cv2.remap(img2, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.abs(img1.astype(np.float32) - warped.astype(np.float32)).mean(axis=-1)
