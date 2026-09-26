"""
src/eval/bootstrap.py — Paired bootstrap confidence intervals for F0.5.

The keep rule (§13) requires:
    OOF macro F0.5 improves by ≥ max(0.002, CI lower bound > 0)
    AND LOCO does not drop by > 0.003.

This module computes the paired bootstrap 95% CI for the delta between
two experiment configurations over S1 entities.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .metric import f05


def paired_bootstrap_ci(
    gts: List[List[str]],
    preds_a: List[List[str]],
    preds_b: List[List[str]],
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Compute paired bootstrap CI for the delta F0.5(B) - F0.5(A).

    Args:
        gts: Ground-truth lists (one per S1 entity).
        preds_a: Predictions from baseline experiment A.
        preds_b: Predictions from new experiment B.
        n_boot: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        alpha: Significance level (default 0.05 → 95% CI).

    Returns:
        Tuple (delta, ci_low, ci_high) where:
        - delta = mean F0.5(B) - mean F0.5(A) on the original sample
        - ci_low, ci_high = bootstrap percentile CI at (alpha/2, 1-alpha/2)

    Keep rule: delta ≥ 0.002 AND ci_low > 0.
    """
    rng = np.random.RandomState(seed)
    n = len(gts)

    scores_a = np.array([f05(g, p) for g, p in zip(gts, preds_a)])
    scores_b = np.array([f05(g, p) for g, p in zip(gts, preds_b)])

    delta = float(np.mean(scores_b) - np.mean(scores_a))

    boot_deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_deltas[i] = np.mean(scores_b[idx]) - np.mean(scores_a[idx])

    ci_low = float(np.percentile(boot_deltas, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_deltas, 100 * (1 - alpha / 2)))

    return delta, ci_low, ci_high


def passes_keep_rule(
    gts: List[List[str]],
    preds_baseline: List[List[str]],
    preds_new: List[List[str]],
    min_delta: float = 0.002,
    loco_delta_baseline: float = 0.0,
    loco_delta_new: float = 0.0,
    max_loco_drop: float = 0.003,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, object]:
    """Apply the full keep rule and return a structured result dict.

    Args:
        gts: Ground-truth lists.
        preds_baseline: Baseline predictions.
        preds_new: New-experiment predictions.
        min_delta: Minimum absolute OOF delta (default 0.002).
        loco_delta_baseline: LOCO F0.5 for baseline.
        loco_delta_new: LOCO F0.5 for new experiment.
        max_loco_drop: Maximum allowable LOCO drop.
        n_boot: Bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with keys:
        - ``delta``, ``ci_low``, ``ci_high``: F0.5 delta and 95% CI
        - ``loco_drop``: loco_delta_new - loco_delta_baseline (negative = worse)
        - ``oof_passes``: bool — delta ≥ min_delta AND ci_low > 0
        - ``loco_passes``: bool — loco_drop ≥ -max_loco_drop
        - ``keep``: bool — oof_passes AND loco_passes
    """
    delta, ci_low, ci_high = paired_bootstrap_ci(
        gts, preds_baseline, preds_new, n_boot=n_boot, seed=seed
    )
    loco_drop = loco_delta_new - loco_delta_baseline
    oof_passes = delta >= min_delta and ci_low > 0
    loco_passes = loco_drop >= -max_loco_drop

    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "loco_drop": loco_drop,
        "oof_passes": oof_passes,
        "loco_passes": loco_passes,
        "keep": oof_passes and loco_passes,
    }
