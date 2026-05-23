"""Middlebury-style metrics (Baker et al. 2011): EE, AE, R0.5/R1.0/R2.0, A50/A75/A95."""

from __future__ import annotations

import numpy as np


def endpoint_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=-1)


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Barron et al. 1994 / Baker et al. 2011 Sect. 5.2.1, returned in degrees."""
    u, v = pred[..., 0], pred[..., 1]
    uG, vG = gt[..., 0], gt[..., 1]
    num = 1.0 + u * uG + v * vG
    den = np.sqrt(1.0 + u * u + v * v) * np.sqrt(1.0 + uG * uG + vG * vG)
    cos = np.clip(num / np.maximum(den, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


_UNKNOWN_FLOW_THRESH = 1e9  # Middlebury convention: |u|>=1e9 or |v|>=1e9 means invalid GT


def gt_valid_mask(gt: np.ndarray) -> np.ndarray:
    return (np.abs(gt[..., 0]) < _UNKNOWN_FLOW_THRESH) & (np.abs(gt[..., 1]) < _UNKNOWN_FLOW_THRESH)


def summary(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute all Middlebury-style numbers for one pair, masking invalid GT pixels."""
    valid = gt_valid_mask(gt)
    ee = endpoint_error(pred, gt)[valid]
    ae = angular_error_deg(pred, gt)[valid]
    return {
        "ee_mean":  float(ee.mean()),
        "ee_sd":    float(ee.std()),
        "ae_mean":  float(ae.mean()),
        "ae_sd":    float(ae.std()),
        "R0.5":     float((ee > 0.5).mean()),
        "R1.0":     float((ee > 1.0).mean()),
        "R2.0":     float((ee > 2.0).mean()),
        "A50":      float(np.percentile(ee, 50)),
        "A75":      float(np.percentile(ee, 75)),
        "A95":      float(np.percentile(ee, 95)),
        "valid_frac": float(valid.mean()),
    }
