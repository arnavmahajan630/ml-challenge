"""
src/decode/assign.py — 1-to-1 conflict resolution across S1 entities.

When S0 EDA shows that each S2/S3 ID appears in at most one GT list,
we can apply conflict resolution: a candidate c should not be assigned to
multiple S1 entities simultaneously.

Three variants (all tried on OOF; best kept):
1. Hard: sort all (S1, c) pairs by prob descending; assign c to first S1.
2. Gap-gated hard: resolve only when prob gap between top-2 claimants ≥ δ;
   else drop c from both (precision play).
3. Soft normalization: p′ᵢ = pᵢ / (Σⱼ pⱼ + λ₀) with tunable null mass λ₀.
   Keeps probability coherence; λ₀ acts as a prior on "no one owns c".
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def resolve_hard(
    pairs_df: pd.DataFrame,
    prob_col: str = "prob",
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """Hard 1-to-1 conflict resolution.

    Sort all (S1, c) pairs by prob descending; assign each c to the first
    S1 that claims it; drop c's other (S1, c) pairs.

    Args:
        pairs_df: DataFrame with (source1_entity_id, candidate_entity_id, prob).
        prob_col: Column name for calibrated probability.
        s1_col: Column name for S1 entity ID.
        cand_col: Column name for candidate entity ID.

    Returns:
        Filtered pairs_df with 1-to-1 assignment.
    """
    df = pairs_df.sort_values(prob_col, ascending=False).copy()
    claimed: Dict[str, str] = {}  # candidate_id → s1_id
    keep_mask = np.zeros(len(df), dtype=bool)

    for idx, row in df.iterrows():
        c = row[cand_col]
        s1 = row[s1_col]
        if c not in claimed:
            claimed[c] = s1
            keep_mask[df.index.get_loc(idx)] = True

    return df[keep_mask].reset_index(drop=True)


def resolve_gap_gated(
    pairs_df: pd.DataFrame,
    delta: float = 0.2,
    prob_col: str = "prob",
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
) -> pd.DataFrame:
    """Gap-gated hard conflict resolution.

    For each candidate c with multiple claimants:
    - If the gap between the top-2 claimants' probs ≥ delta → assign to top.
    - Else → drop c from ALL claimants (precision play).

    Args:
        pairs_df: Pairs DataFrame.
        delta: Minimum prob gap to assign.
        prob_col, s1_col, cand_col: Column names.

    Returns:
        Filtered pairs_df.
    """
    df = pairs_df.copy()

    # Find candidates with multiple claimants
    claimants = (
        df.groupby(cand_col)[prob_col]
        .apply(lambda x: sorted(x.tolist(), reverse=True))
        .reset_index()
    )
    claimants.columns = [cand_col, "sorted_probs"]
    claimants["n_claimants"] = claimants["sorted_probs"].apply(len)
    claimants["gap"] = claimants["sorted_probs"].apply(
        lambda ps: ps[0] - ps[1] if len(ps) > 1 else 999.0
    )

    # Candidates to drop entirely (gap < delta and multiple claimants)
    drop_cands = set(
        claimants.loc[
            (claimants["n_claimants"] > 1) & (claimants["gap"] < delta),
            cand_col,
        ].tolist()
    )

    # For remaining contested candidates (gap ≥ delta): keep only top claimant
    df_sorted = df.sort_values(prob_col, ascending=False)
    claimed: Dict[str, str] = {}
    keep_indices = []

    for idx, row in df_sorted.iterrows():
        c = row[cand_col]
        s1 = row[s1_col]
        if c in drop_cands:
            continue
        if c not in claimed:
            claimed[c] = s1
            keep_indices.append(idx)

    return df.loc[keep_indices].reset_index(drop=True)


def resolve_soft(
    pairs_df: pd.DataFrame,
    lambda0: float = 0.1,
    prob_col: str = "prob",
    s1_col: str = "source1_entity_id",
    cand_col: str = "candidate_entity_id",
    out_col: str = "prob_soft",
) -> pd.DataFrame:
    """Soft 1-to-1 normalization with null-mass prior.

    For each candidate c with claimants S1₁…S1ₘ:
        p′ᵢ = pᵢ / (Σⱼ pⱼ + λ₀)

    This keeps probabilities coherent (they sum to ≤ 1) and distributes
    the residual null mass λ₀ to the "no one owns c" hypothesis.

    Args:
        pairs_df: Pairs DataFrame.
        lambda0: Null-mass prior (tuned on OOF).
        prob_col: Input probability column.
        s1_col, cand_col: Column names.
        out_col: Name for the output normalized probability column.

    Returns:
        pairs_df with new column ``out_col``.
    """
    df = pairs_df.copy()
    # Sum of probs per candidate across all its S1 claimants
    denom_map = (
        df.groupby(cand_col)[prob_col].sum() + lambda0
    ).to_dict()

    df[out_col] = df.apply(
        lambda row: row[prob_col] / denom_map.get(row[cand_col], 1.0),
        axis=1,
    )
    return df


def apply_conflict_resolution(
    pairs_df: pd.DataFrame,
    mode: str = "hard",
    delta: float = 0.2,
    lambda0: float = 0.1,
    prob_col: str = "prob",
) -> pd.DataFrame:
    """Dispatch to the appropriate conflict resolution strategy.

    Args:
        pairs_df: Candidate pairs with probabilities.
        mode: One of "hard", "gap", "soft", "none".
        delta: Gap-gated threshold (used when mode="gap").
        lambda0: Null-mass (used when mode="soft").
        prob_col: Probability column name.

    Returns:
        pairs_df after conflict resolution; soft mode adds a ``prob_soft`` col.
    """
    if mode == "hard":
        return resolve_hard(pairs_df, prob_col=prob_col)
    elif mode == "gap":
        return resolve_gap_gated(pairs_df, delta=delta, prob_col=prob_col)
    elif mode == "soft":
        df = resolve_soft(pairs_df, lambda0=lambda0, prob_col=prob_col)
        # Use prob_soft as the effective probability for downstream decoding
        df[prob_col] = df["prob_soft"]
        return df
    elif mode == "none":
        return pairs_df.copy()
    else:
        raise ValueError(f"Unknown conflict resolution mode: {mode!r}")
