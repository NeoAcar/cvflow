"""Disc mask (Baker et al. 2011, §4.3): gradient of GT flow, threshold, 9x9 dilate."""

from __future__ import annotations

import cv2
import numpy as np


def disc_mask(gt_flow: np.ndarray, grad_thresh: float = 1.0) -> np.ndarray:
    """Return bool HxW mask of motion-discontinuity pixels (1 = on Disc).

    gt_flow: HxWx2 float32.
    grad_thresh: threshold on ‖∇u‖₁ + ‖∇v‖₁ in px/px units.
    """
    u, v = gt_flow[..., 0], gt_flow[..., 1]
    ux = cv2.Sobel(u, cv2.CV_32F, 1, 0, ksize=3)
    uy = cv2.Sobel(u, cv2.CV_32F, 0, 1, ksize=3)
    vx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3)
    vy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(ux) + np.abs(uy) + np.abs(vx) + np.abs(vy)
    high = (grad > grad_thresh).astype(np.uint8)
    dilated = cv2.dilate(high, np.ones((9, 9), np.uint8))
    return dilated.astype(bool)
