"""Blur / defocus mask via local Laplacian variance (methodology §2.6, §2.8).

A pixel x is "blurred" iff the variance of the Laplacian in a window around x
is below a threshold. Sharp pixels have a wide-spread Laplacian distribution;
defocus / motion blur flattens it.
"""

from __future__ import annotations

import cv2
import numpy as np


def blur_mask(img_uint8: np.ndarray, window: int = 7, var_thresh: float = 20.0) -> np.ndarray:
    """Return bool HxW mask of blurred pixels (1 = blurred).

    img_uint8 : HxWx3 RGB uint8 (or HxW grayscale uint8).
    window    : odd integer; box-window size for the local variance estimate.
    var_thresh: threshold on local var(Laplacian); pixels below are blurred.
    """
    if img_uint8.ndim == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    # local mean and mean-of-squares via box filter → local variance
    mean = cv2.boxFilter(lap, ddepth=cv2.CV_32F, ksize=(window, window), normalize=True)
    mean_sq = cv2.boxFilter(lap * lap, ddepth=cv2.CV_32F, ksize=(window, window), normalize=True)
    local_var = np.maximum(mean_sq - mean * mean, 0.0)
    return local_var < var_thresh
