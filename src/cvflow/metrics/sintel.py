"""Per-pair Sintel metric accumulators.

Conventions (matches GMFlow's official evaluate.py for the no-refinement model):
    valid     = invalid == 0
    matched   = valid AND (occlusion == 0)        # non-occluded
    unmatched = valid AND (occlusion == 255)      # occluded
    s0_10     = valid AND speed < 10
    s10_40    = valid AND 10 <= speed < 40
    s40+      = valid AND speed >= 40
    Bad-Xpx   = fraction of `valid` pixels with EPE > X
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Accum:
    sum_epe: float = 0.0
    sum_n: int = 0
    sum_bad1: int = 0
    sum_bad3: int = 0
    sum_bad5: int = 0

    def add(self, epe: np.ndarray, mask: np.ndarray) -> None:
        if not mask.any():
            return
        e = epe[mask]
        self.sum_epe += float(e.sum())
        self.sum_n += int(e.size)
        self.sum_bad1 += int((e > 1).sum())
        self.sum_bad3 += int((e > 3).sum())
        self.sum_bad5 += int((e > 5).sum())

    def epe(self) -> float:
        return self.sum_epe / self.sum_n if self.sum_n else float("nan")

    def bad(self, t: int) -> float:
        if self.sum_n == 0:
            return float("nan")
        return {1: self.sum_bad1, 3: self.sum_bad3, 5: self.sum_bad5}[t] / self.sum_n


@dataclass
class SintelMetrics:
    """Accumulates EPE + Bad-X over multiple masks, globally and per-sequence."""
    _g: dict[str, _Accum] = field(default_factory=lambda: defaultdict(_Accum))
    _by_seq: dict[str, dict[str, _Accum]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(_Accum)))

    def update(self, pred: np.ndarray, gt: np.ndarray, occlusion: np.ndarray, invalid: np.ndarray, seq: str,
               disc: np.ndarray | None = None, untex: np.ndarray | None = None) -> None:
        epe = np.linalg.norm(pred - gt, axis=-1)
        valid = invalid == 0
        speed = np.linalg.norm(gt, axis=-1)
        masks = {
            "all":       valid,
            "matched":   valid & (occlusion == 0),
            "unmatched": valid & (occlusion == 255),
            "s0_10":     valid & (speed < 10),
            "s10_40":    valid & (speed >= 10) & (speed < 40),
            "s40+":      valid & (speed >= 40),
        }
        if disc is not None:
            masks["disc"] = valid & disc
        if untex is not None:
            masks["untex"] = valid & untex
        for name, m in masks.items():
            self._g[name].add(epe, m)
            self._by_seq[seq][name].add(epe, m)

    def global_summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, acc in self._g.items():
            out[f"epe/{name}"] = acc.epe()
        a = self._g["all"]
        out["bad1/all"] = a.bad(1)
        out["bad3/all"] = a.bad(3)
        out["bad5/all"] = a.bad(5)
        return out

    def per_seq_epe_all(self) -> dict[str, float]:
        return {s: accs["all"].epe() for s, accs in self._by_seq.items()}
