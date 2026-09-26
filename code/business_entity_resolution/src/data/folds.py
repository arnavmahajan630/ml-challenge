"""
src/data/folds.py — Cross-validation and LOCO splits for the BER pipeline.

Fold strategy:
- StratifiedGroupKFold(5) over train S1 IDs.
- Stratification key: country × singleton_flag (concatenated string).
- Groups: S1 entity_id (so the same S1 never straddles two folds).
- Unmatched S2/S3 records: in the candidate pool for ALL folds (they never
  appear in a held-out GT, so no leakage).
- Retrieval for a held-out fold: searches the full train S2/S3 pool
  (identical to test-time setup).

LOCO (Leave One Country Out) splits:
- train-on-{country_A} → eval-on-{country_B} for every ordered pair.
- ``loco_f05`` = mean of all pairwise LOCO F0.5 values.
- LOCO uses only the training country in all learned components.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_folds(
    gt_df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a fold index (0..n_splits-1) to every train S1 entity.

    Args:
        gt_df: Ground-truth DataFrame with columns:
               ``source1_entity_id``, ``matched_entity_ids``, ``country``
               (joined from S1 source file).
        n_splits: Number of folds.
        seed: Random state for the splitter.

    Returns:
        DataFrame with columns: ``source1_entity_id``, ``fold``, ``is_singleton``,
        ``country``, ``strat_key``.
    """
    df = gt_df.copy()
    df["is_singleton"] = df["matched_entity_ids"].apply(
        lambda s: 1 if not str(s).strip() else 0
    )
    # Stratification key: country value × singleton flag
    df["strat_key"] = df["country"].astype(str) + "__" + df["is_singleton"].astype(str)

    X = np.zeros(len(df))  # dummy features — we only need the splits
    y = df["strat_key"].values
    groups = df["source1_entity_id"].values

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_col = np.full(len(df), -1, dtype=int)

    for fold_idx, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_col[val_idx] = fold_idx

    df["fold"] = fold_col
    assert (df["fold"] >= 0).all(), "Some S1 rows were not assigned a fold."
    return df[["source1_entity_id", "fold", "is_singleton", "country", "strat_key"]]


def make_loco_splits(
    gt_df: pd.DataFrame,
    country_col: str = "country",
) -> List[Dict[str, object]]:
    """Generate all Leave-One-Country-Out splits from the training data.

    For N distinct countries, produces N*(N-1) ordered (train, eval) pairs.
    Each split is a dict with keys: ``train_s1_ids``, ``eval_s1_ids``,
    ``train_country``, ``eval_country``.

    Args:
        gt_df: Ground truth with ``source1_entity_id`` and *country_col*.
        country_col: Column name that holds the country string value.

    Returns:
        List of split dicts.
    """
    countries = sorted(gt_df[country_col].unique().tolist())
    splits = []
    for eval_c in countries:
        train_mask = gt_df[country_col] != eval_c
        eval_mask = gt_df[country_col] == eval_c
        splits.append(
            {
                "train_country": [c for c in countries if c != eval_c],
                "eval_country": eval_c,
                "train_s1_ids": set(
                    gt_df.loc[train_mask, "source1_entity_id"].tolist()
                ),
                "eval_s1_ids": set(
                    gt_df.loc[eval_mask, "source1_entity_id"].tolist()
                ),
            }
        )
    return splits


def get_fold_mask(
    fold_df: pd.DataFrame,
    fold_idx: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return boolean masks for (train_s1, val_s1) for a given fold.

    Args:
        fold_df: Output of :func:`make_folds`.
        fold_idx: Zero-based fold index.

    Returns:
        Tuple of (train_mask, val_mask) as boolean Series indexed by
        ``source1_entity_id``.
    """
    val_mask = fold_df["fold"] == fold_idx
    train_mask = ~val_mask
    return train_mask, val_mask


def pool_subsample_oof(
    candidate_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    target_ratio: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Sub-sample or up-sample unmatched candidates to hit *target_ratio*.

    Used for pool-matched OOF robustness check (§5.5).

    Args:
        candidate_df: Full candidate pairs DataFrame with column ``label``
                      (1=positive, 0=negative/unmatched).
        gt_df: Ground-truth DataFrame (used to identify positives).
        target_ratio: Desired ratio of |unmatched| / |S1|.
        seed: Random seed for sampling.

    Returns:
        Subsampled candidate DataFrame.
    """
    rng = np.random.RandomState(seed)
    pos_mask = candidate_df["label"] == 1
    neg_mask = ~pos_mask

    pos_df = candidate_df[pos_mask]
    neg_df = candidate_df[neg_mask]

    n_s1 = gt_df["source1_entity_id"].nunique()
    target_neg = int(target_ratio * n_s1)

    if len(neg_df) >= target_neg:
        neg_sample = neg_df.sample(n=target_neg, random_state=rng.randint(0, 2**31))
    else:
        # Up-sample with replacement
        neg_sample = neg_df.sample(
            n=target_neg, replace=True, random_state=rng.randint(0, 2**31)
        )

    return pd.concat([pos_df, neg_sample], ignore_index=True)
