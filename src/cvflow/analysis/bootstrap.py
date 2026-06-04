"""Paired sequence-level bootstrap for hair-margin metric comparisons.

Inputs: per-sequence per-mask raw (sum, n) statistics emitted by
`SintelMetrics.dump_per_seq`. We resample the 23 Sintel sequences with
replacement (the same indices for both models in a paired-bootstrap fashion)
and recompute the pixel-weighted mean per resample.

Returns the bootstrap distribution of the difference (b − a) plus a 95% CI
and a two-sided p-value for "Δ = 0".

Why sequence-level: pixels within a sequence are not independent (shared
scene, shared camera path); the dataset's effective sample size is the number
of independently-rendered sequences, not the pixel count. Pixel-level
bootstrap dramatically underestimates the CI width.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_METRIC_NUMER_DENOM = {
    "epe":   ("sum_epe",   "sum_n"),
    "ae":    ("sum_ae",    "sum_n"),
    "bad1":  ("sum_bad1",  "sum_n"),
    "bad3":  ("sum_bad3",  "sum_n"),
    "bad5":  ("sum_bad5",  "sum_n"),
    "bad10": ("sum_bad10", "sum_n"),
    "nepe":  ("sum_nepe",  "sum_nepe_n"),
}


def load_per_seq(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    with Path(path).open() as f:
        return json.load(f)


def _weighted_mean(seqs: list[str], stats: dict[str, dict[str, float]],
                   num_key: str, den_key: str) -> float:
    num = sum(stats[s][num_key] for s in seqs)
    den = sum(stats[s][den_key] for s in seqs)
    return num / den if den > 0 else float("nan")


@dataclass
class BootResult:
    mean_a: float
    mean_b: float
    diff: float           # b − a (point estimate)
    ci_low: float         # 2.5% percentile of (b − a)*
    ci_high: float        # 97.5% percentile of (b − a)*
    p_value: float        # 2 × min(P(Δ* ≤ 0), P(Δ* ≥ 0))
    crosses_zero: bool    # True iff the 95% CI includes zero


def paired_bootstrap(stats_a: dict, stats_b: dict, mask: str, metric: str,
                     n_boot: int = 10_000, seed: int = 0) -> BootResult:
    """Compute paired bootstrap CI for `metric` on `mask` between two models.

    stats_{a,b}: outputs of `SintelMetrics.dump_per_seq`.
    metric:    one of _METRIC_NUMER_DENOM keys.
    Same sequence resample applied to both models (paired).
    """
    num_key, den_key = _METRIC_NUMER_DENOM[metric]
    seqs_a = set(stats_a[mask].keys())
    seqs_b = set(stats_b[mask].keys())
    seqs = sorted(seqs_a & seqs_b)
    if not seqs:
        return BootResult(float("nan"), float("nan"), float("nan"),
                          float("nan"), float("nan"), float("nan"), True)

    mean_a = _weighted_mean(seqs, stats_a[mask], num_key, den_key)
    mean_b = _weighted_mean(seqs, stats_b[mask], num_key, den_key)
    diff = mean_b - mean_a

    rng = np.random.default_rng(seed)
    n = len(seqs)
    # Precompute per-sequence numerator and denominator arrays
    num_a = np.array([stats_a[mask][s][num_key] for s in seqs])
    den_a = np.array([stats_a[mask][s][den_key] for s in seqs])
    num_b = np.array([stats_b[mask][s][num_key] for s in seqs])
    den_b = np.array([stats_b[mask][s][den_key] for s in seqs])

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        da = num_a[idx].sum() / max(den_a[idx].sum(), 1e-12)
        db = num_b[idx].sum() / max(den_b[idx].sum(), 1e-12)
        diffs[i] = db - da

    ci_low, ci_high = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # Two-sided p-value via the percentile principle (Efron-style)
    p_left  = float((diffs <= 0).mean())
    p_right = float((diffs >= 0).mean())
    p_value = 2.0 * min(p_left, p_right)
    return BootResult(mean_a=mean_a, mean_b=mean_b, diff=diff,
                      ci_low=ci_low, ci_high=ci_high,
                      p_value=p_value, crosses_zero=(ci_low <= 0 <= ci_high))
