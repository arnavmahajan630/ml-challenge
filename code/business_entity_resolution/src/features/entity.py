"""
src/features/entity.py — Aggregate features for the entity-level models.

Generates features PER S1 ENTITY, such as max probability, mean probability,
probability gaps, and candidate pool size, used by the has-match classifier.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_entity_features(
    cands_df: pd.DataFrame,
    prob_col: str = "prob"
) -> pd.DataFrame:
    """
    Args:
        cands_df: Candidate pairs dataframe with calibrated probabilities.
        
    Returns:
        DataFrame with one row per source1_entity_id, containing aggregate features.
    """
    # Sort candidates by probability descending
    df = cands_df.sort_values(["source1_entity_id", prob_col], ascending=[True, False]).copy()
    
    aggs = {}
    
    # Simple aggregates
    aggs["cand_count"] = df.groupby("source1_entity_id")[prob_col].count()
    aggs["prob_max"] = df.groupby("source1_entity_id")[prob_col].max()
    aggs["prob_mean"] = df.groupby("source1_entity_id")[prob_col].mean()
    aggs["prob_sum"] = df.groupby("source1_entity_id")[prob_col].sum()
    
    # Top-K specific aggregates
    def _top_k_agg(series, k, agg_func):
        return series.head(k).agg(agg_func)
        
    aggs["prob_mean_top3"] = df.groupby("source1_entity_id")[prob_col].apply(lambda x: x.head(3).mean())
    aggs["prob_sum_top3"] = df.groupby("source1_entity_id")[prob_col].apply(lambda x: x.head(3).sum())
    
    # Gaps
    def _gap_1_2(x):
        if len(x) >= 2:
            return x.iloc[0] - x.iloc[1]
        return 0.0
        
    aggs["prob_gap_1_2"] = df.groupby("source1_entity_id")[prob_col].apply(_gap_1_2)
    
    res = pd.DataFrame(aggs).reset_index()
    
    # Fill NAs for S1s with 0 candidates (handled externally but good practice)
    res = res.fillna(0.0)
    
    return res
