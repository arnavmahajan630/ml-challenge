"""
src/eval/metric.py — Exact macro F0.5 metric for Business Entity Resolution.

Closed-form: F0.5 = 1.25 · TP / (k + 0.25 · G)
where k = |predicted|, G = |ground truth|, TP = |predicted ∩ ground truth|.

Special cases (§0):
    GT empty, pred empty  → 1.0
    GT empty, pred non-empty → 0.0
    GT non-empty, pred empty → 0.0
    GT non-empty, pred non-empty → 1.25·P·R / (0.25·P + R)

The spec guarantees that duplicate IDs in a prediction are deduplicated
before scoring.  We always dedup here for safety.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np


def f05(gt: List[str], pred: List[str]) -> float:
    """Compute F0.5 for a single S1 entity.

    Args:
        gt: Ground-truth list of matched IDs (empty for singletons).
        pred: Predicted list of matched IDs.

    Returns:
        F0.5 score in [0, 1].
    """
    # Deduplicate prediction while preserving order
    seen: dict[str, None] = {}
    for x in pred:
        seen[x] = None
    pred_dedup = list(seen.keys())

    if not gt and not pred_dedup:
        return 1.0
    if not gt or not pred_dedup:
        return 0.0

    tp = len(set(gt) & set(pred_dedup))
    if tp == 0:
        return 0.0

    k = len(pred_dedup)
    g = len(gt)
    # Closed form: 1.25·TP / (k + 0.25·G)
    return 1.25 * tp / (k + 0.25 * g)


def macro_f05(
    gts: Iterable[List[str]],
    preds: Iterable[List[str]],
) -> float:
    """Compute macro-averaged F0.5 over all S1 entities.

    Args:
        gts: Iterable of ground-truth ID lists (one per S1).
        preds: Iterable of predicted ID lists (one per S1, same order).

    Returns:
        Macro F0.5 in [0, 1].  Returns 0.0 if there are no entities.
    """
    scores = [f05(g, p) for g, p in zip(gts, preds)]
    if not scores:
        return 0.0
    return float(np.mean(scores))


def evaluate(
    gt_map: Mapping[str, List[str]],
    pred_map: Mapping[str, List[str]],
    return_per_entity: bool = False,
) -> Dict[str, float]:
    """Full evaluation: macro F0.5 + breakdown by singleton / non-singleton.

    Also reports per-country-value breakdown (computed generically, not from
    country literals).

    Args:
        gt_map: ``{source1_entity_id: [matched_ids]}``
        pred_map: ``{source1_entity_id: [predicted_ids]}``
        return_per_entity: If True, include per-entity scores in the result.

    Returns:
        Dict with keys: ``macro_f05``, ``singleton_f05``, ``nonsingleton_f05``,
        ``n_entities``, ``n_singletons``, ``n_nonsingleton``,
        and optionally ``per_entity`` (dict of entity_id → score).
    """
    all_ids = sorted(gt_map.keys())
    gts = [gt_map[eid] for eid in all_ids]
    preds = [pred_map.get(eid, []) for eid in all_ids]

    scores = [f05(g, p) for g, p in zip(gts, preds)]

    singleton_scores = [s for s, g in zip(scores, gts) if not g]
    nonsingleton_scores = [s for s, g in zip(scores, gts) if g]

    result: Dict[str, float] = {
        "macro_f05": float(np.mean(scores)) if scores else 0.0,
        "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
        "nonsingleton_f05": float(np.mean(nonsingleton_scores)) if nonsingleton_scores else 0.0,
        "n_entities": float(len(scores)),
        "n_singletons": float(len(singleton_scores)),
        "n_nonsingleton": float(len(nonsingleton_scores)),
    }

    if return_per_entity:
        result["per_entity"] = dict(zip(all_ids, scores))  # type: ignore[assignment]

    return result


def evaluate_by_group(
    gt_map: Mapping[str, List[str]],
    pred_map: Mapping[str, List[str]],
    group_map: Mapping[str, str],
) -> Dict[str, float]:
    """Compute macro F0.5 per group value (e.g. country).

    Args:
        gt_map: Ground truth mapping.
        pred_map: Prediction mapping.
        group_map: ``{source1_entity_id: group_value}`` — e.g. country string.

    Returns:
        Dict ``{group_value: f05_score}`` plus ``"overall": macro_f05``.
    """
    groups: Dict[str, List[str]] = {}
    for eid in gt_map:
        grp = group_map.get(eid, "unknown")
        groups.setdefault(grp, []).append(eid)

    result: Dict[str, float] = {}
    all_scores = []
    for grp, eids in sorted(groups.items()):
        gts = [gt_map[eid] for eid in eids]
        preds = [pred_map.get(eid, []) for eid in eids]
        scores = [f05(g, p) for g, p in zip(gts, preds)]
        result[grp] = float(np.mean(scores)) if scores else 0.0
        all_scores.extend(scores)

    result["overall"] = float(np.mean(all_scores)) if all_scores else 0.0
    return result
