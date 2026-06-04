"""Per-pair Sintel metric accumulators.

Conventions (matches GMFlow's official evaluate.py for the no-refinement model):
    valid     = invalid == 0
    matched   = valid AND (occlusion == 0)        # non-occluded
    unmatched = valid AND (occlusion == 255)      # occluded
    s0_10     = valid AND speed < 10
    s10_40    = valid AND 10 <= speed < 40
    s40+      = valid AND speed >= 40
    s60+      = valid AND speed >= 60             # methodology §3 H1 extra bin
    Bad-Xpx   = fraction of `valid` pixels with EPE > X (X ∈ {1, 3, 5, 10})
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Accum:
    sum_epe: float = 0.0
    sum_epe_sq: float = 0.0
    sum_ae: float = 0.0
    sum_n: int = 0
    sum_bad1: int = 0
    sum_bad3: int = 0
    sum_bad5: int = 0
    sum_bad10: int = 0  # catastrophic-failure threshold (methodology §1.2)
    # Normalized EPE = EPE / |gt_flow| computed only where |gt_flow| > _NEPE_FLOOR
    sum_nepe: float = 0.0
    sum_nepe_n: int = 0
    _NEPE_FLOOR: float = 1.0
    # Paired (epe, ae) reservoir for AE/EPE correlations
    epe_sample: list = field(default_factory=list)
    ae_sample: list = field(default_factory=list)
    _SAMPLE_CAP: int = 5_000_000

    def add(self, epe: np.ndarray, mask: np.ndarray, ae: np.ndarray | None = None,
            speed: np.ndarray | None = None) -> None:
        if not mask.any():
            return
        e = epe[mask]
        self.sum_epe += float(e.sum())
        self.sum_epe_sq += float((e.astype(np.float64) ** 2).sum())
        self.sum_n += int(e.size)
        self.sum_bad1 += int((e > 1).sum())
        self.sum_bad3 += int((e > 3).sum())
        self.sum_bad5 += int((e > 5).sum())
        self.sum_bad10 += int((e > 10).sum())
        a = ae[mask] if ae is not None else None
        if a is not None:
            self.sum_ae += float(a.sum())
        if speed is not None:
            s = speed[mask]
            big = s > self._NEPE_FLOOR
            if big.any():
                self.sum_nepe += float((e[big] / s[big]).sum())
                self.sum_nepe_n += int(big.sum())
        # Paired reservoir for correlation; cap on TOTAL pixels across both arrays.
        remaining = self._SAMPLE_CAP - sum(arr.size for arr in self.epe_sample)
        if remaining > 0:
            take = min(remaining, e.size)
            self.epe_sample.append(e[:take].astype(np.float32))
            if a is not None:
                self.ae_sample.append(a[:take].astype(np.float32))

    def epe(self) -> float:
        return self.sum_epe / self.sum_n if self.sum_n else float("nan")

    def sd(self) -> float:
        if self.sum_n == 0:
            return float("nan")
        mean = self.sum_epe / self.sum_n
        var = max(self.sum_epe_sq / self.sum_n - mean * mean, 0.0)
        return float(np.sqrt(var))

    def ae(self) -> float:
        return self.sum_ae / self.sum_n if self.sum_n else float("nan")

    def bad(self, t: int) -> float:
        if self.sum_n == 0:
            return float("nan")
        return {1: self.sum_bad1, 3: self.sum_bad3, 5: self.sum_bad5, 10: self.sum_bad10}[t] / self.sum_n

    def percentile(self, q: float) -> float:
        if not self.epe_sample:
            return float("nan")
        return float(np.percentile(np.concatenate(self.epe_sample), q))

    def nepe(self) -> float:
        return self.sum_nepe / self.sum_nepe_n if self.sum_nepe_n else float("nan")

    def correlation(self) -> tuple[float, float]:
        """Return (pearson, spearman) on the paired (epe, ae) reservoir."""
        if not self.epe_sample or not self.ae_sample:
            return float("nan"), float("nan")
        e = np.concatenate(self.epe_sample).astype(np.float64)
        a = np.concatenate(self.ae_sample).astype(np.float64)
        if e.size < 2 or e.std() == 0 or a.std() == 0:
            return float("nan"), float("nan")
        pearson = float(np.corrcoef(e, a)[0, 1])
        from scipy.stats import spearmanr
        spearman = float(spearmanr(e, a, alternative="two-sided").statistic)
        return pearson, spearman


@dataclass
class SintelMetrics:
    """Accumulates EPE/AE/Bad-X/percentiles/SD over multiple masks, global + per-sequence."""
    _g: dict[str, _Accum] = field(default_factory=lambda: defaultdict(_Accum))
    _by_seq: dict[str, dict[str, _Accum]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(_Accum)))

    def update(self, pred: np.ndarray, gt: np.ndarray, occlusion: np.ndarray, invalid: np.ndarray, seq: str,
               disc: np.ndarray | None = None, untex: np.ndarray | None = None,
               blur: np.ndarray | None = None) -> None:
        from cvflow.metrics.middlebury import angular_error_deg
        epe = np.linalg.norm(pred - gt, axis=-1)
        ae = angular_error_deg(pred, gt)
        valid = invalid == 0
        speed = np.linalg.norm(gt, axis=-1)
        masks = {
            "all":       valid,
            "matched":   valid & (occlusion == 0),
            "unmatched": valid & (occlusion == 255),
            "s0_1":      valid & (speed < 1),
            "s0_10":     valid & (speed < 10),
            "s10_40":    valid & (speed >= 10) & (speed < 40),
            "s40+":      valid & (speed >= 40),
            "s60+":      valid & (speed >= 60),
        }
        if disc is not None:
            masks["disc"] = valid & disc
        if untex is not None:
            masks["untex"] = valid & untex
        if blur is not None:
            masks["blur"] = valid & blur
        for name, m in masks.items():
            self._g[name].add(epe, m, ae, speed)
            self._by_seq[seq][name].add(epe, m, ae, speed)

    def global_summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, acc in self._g.items():
            out[f"epe/{name}"] = acc.epe()
            out[f"sd/{name}"] = acc.sd()
            out[f"ae/{name}"] = acc.ae()
            out[f"nepe/{name}"] = acc.nepe()
            out[f"bad1/{name}"] = acc.bad(1)
            out[f"bad3/{name}"] = acc.bad(3)
            out[f"bad5/{name}"] = acc.bad(5)
            out[f"bad10/{name}"] = acc.bad(10)
            out[f"A50/{name}"] = acc.percentile(50)
            out[f"A75/{name}"] = acc.percentile(75)
            out[f"A95/{name}"] = acc.percentile(95)
            p, s = acc.correlation()
            out[f"pearson/{name}"] = p
            out[f"spearman/{name}"] = s
        return out

    def per_seq_epe_all(self) -> dict[str, float]:
        return {s: accs["all"].epe() for s, accs in self._by_seq.items()}

    def dump_per_seq(self, path) -> None:
        """Serialize per-sequence per-mask raw sums + counts as JSON.

        Output shape:
            { mask_name: { seq_name: {
                "sum_epe", "sum_epe_sq", "sum_ae", "sum_n",
                "sum_bad1", "sum_bad3", "sum_bad5", "sum_bad10",
                "sum_nepe", "sum_nepe_n"
            } } }

        These are enough to recompute every reported metric AND to bootstrap
        across sequences without re-reading any prediction file.
        """
        import json
        from pathlib import Path as _P
        masks: set[str] = set()
        for accs in self._by_seq.values():
            masks.update(accs.keys())
        out: dict[str, dict[str, dict[str, float | int]]] = {m: {} for m in masks}
        for seq, accs in self._by_seq.items():
            for m, acc in accs.items():
                out[m][seq] = {
                    "sum_epe":    acc.sum_epe,
                    "sum_epe_sq": acc.sum_epe_sq,
                    "sum_ae":     acc.sum_ae,
                    "sum_n":      acc.sum_n,
                    "sum_bad1":   acc.sum_bad1,
                    "sum_bad3":   acc.sum_bad3,
                    "sum_bad5":   acc.sum_bad5,
                    "sum_bad10":  acc.sum_bad10,
                    "sum_nepe":   acc.sum_nepe,
                    "sum_nepe_n": acc.sum_nepe_n,
                }
        p = _P(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(out, f, indent=2)
