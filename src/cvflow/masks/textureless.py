"""Untext mask (Baker et al. 2011, §4.3): gradient magnitude threshold then 3x3 dilate."""

from __future__ import annotations

import cv2
import numpy as np


def untext_mask(img_uint8: np.ndarray, grad_thresh: float = 5.0) -> np.ndarray:
    """Return bool HxW mask of textureless pixels (1 = textureless).

    img_uint8: HxWx3 RGB uint8 (or HxW grayscale uint8).
    grad_thresh: Sobel gradient magnitude threshold in intensity units (0..255).
    """
    if img_uint8.ndim == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    low = (grad_mag < grad_thresh).astype(np.uint8)
    # Baker §4.3 dilates the *excluded* (high-gradient) pixels with a 3x3 box —
    # but the standard practice is to dilate the low-grad region to be inclusive.
    # We dilate the textureless mask so the boundary buffer is symmetric with Disc.
    dilated = cv2.dilate(low, np.ones((3, 3), np.uint8))
    return dilated.astype(bool)
