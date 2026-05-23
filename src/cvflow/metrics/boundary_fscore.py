"""F-score between predicted-flow Disc mask and GT-flow Disc mask, with tolerance."""

from __future__ import annotations

import cv2
import numpy as np

from cvflow.masks.motion_boundary import disc_mask


def boundary_fscore(pred_flow: np.ndarray, gt_flow: np.ndarray,
                    grad_thresh: float = 1.0, tol_px: int = 2) -> tuple[float, float, float]:
    """Compute (precision, recall, F1) between Disc(pred) and Disc(gt).

    A pixel in pred's boundary mask counts as a true positive if it lies within
    `tol_px` of any GT boundary pixel — standard practice in boundary detection
    (BSDS / DAVIS) to avoid penalizing 1-pixel offsets.
    """
    p_raw = (cv2.Sobel(pred_flow[..., 0], cv2.CV_32F, 1, 0, ksize=3).__abs__()
             + cv2.Sobel(pred_flow[..., 0], cv2.CV_32F, 0, 1, ksize=3).__abs__()
             + cv2.Sobel(pred_flow[..., 1], cv2.CV_32F, 1, 0, ksize=3).__abs__()
             + cv2.Sobel(pred_flow[..., 1], cv2.CV_32F, 0, 1, ksize=3).__abs__())
    pred_b = (p_raw > grad_thresh).astype(np.uint8)
    g_raw = (cv2.Sobel(gt_flow[..., 0], cv2.CV_32F, 1, 0, ksize=3).__abs__()
             + cv2.Sobel(gt_flow[..., 0], cv2.CV_32F, 0, 1, ksize=3).__abs__()
             + cv2.Sobel(gt_flow[..., 1], cv2.CV_32F, 1, 0, ksize=3).__abs__()
             + cv2.Sobel(gt_flow[..., 1], cv2.CV_32F, 0, 1, ksize=3).__abs__())
    gt_b = (g_raw > grad_thresh).astype(np.uint8)

    k = 2 * tol_px + 1
    pred_dil = cv2.dilate(pred_b, np.ones((k, k), np.uint8))
    gt_dil = cv2.dilate(gt_b, np.ones((k, k), np.uint8))

    n_pred = int(pred_b.sum())
    n_gt = int(gt_b.sum())
    tp_p = int(((pred_b == 1) & (gt_dil == 1)).sum())   # pred matched to GT
    tp_g = int(((gt_b == 1) & (pred_dil == 1)).sum())   # GT matched to pred

    precision = tp_p / n_pred if n_pred else 0.0
    recall = tp_g / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
