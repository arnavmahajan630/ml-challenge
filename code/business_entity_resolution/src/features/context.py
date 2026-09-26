"""
src/features/context.py — Scale-invariant context features.

Features are computed as percentiles/fractions of the candidate pool size
to ensure robustness when the test pool ratio changes.

Includes:
- max_blocker_score_pct: percentile rank of the blocker score among S1's candidates.
- prob_gap_next: gap to the next best probability in S1's pool.
- is_mutual_best: 1 if this pair is the top probability for S1 AND top for Candidate.
- sibling_competition: max probability that this candidate is assigned to a *different* S1.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_context_features(
    df: pd.DataFrame,
    prob_col: str,
) -> pd.DataFrame:
    """Add context features based on a base probability/score.

    Modifies the dataframe and returns it.
    """
    res = df.copy()
    
    # Sort by S1 and score descending
    res = res.sort_values(["source1_entity_id", prob_col], ascending=[True, False])
    
    # 1. Intra-S1 features (ranking within the S1's candidate pool)
    res["rank_in_s1"] = res.groupby("source1_entity_id").cumcount() + 1
    
    # Size of S1's candidate pool
    s1_counts = res.groupby("source1_entity_id")[prob_col].transform("count")
    # Percentile rank (scale invariant): 1.0 is top, ~0.0 is bottom
    res["score_pct_in_s1"] = (s1_counts - res["rank_in_s1"] + 1) / np.maximum(s1_counts, 1)
    
    # Gap to next best in S1
    res["prob_gap_next"] = res[prob_col] - res.groupby("source1_entity_id")[prob_col].shift(-1).fillna(0.0)
    
    # Sibling competition: for this candidate c, what is its max prob among OTHER S1s?
    # This requires looking globally across all S1s that claim c.
    # First, find max prob for each candidate overall
    cand_max = res.groupby("candidate_entity_id")[prob_col].transform("max")
    
    # If the candidate's max prob equals *this* row's prob, then the *second* highest
    # is the actual competition. We need the top-2 to do this accurately.
    def _sibling_max(group):
        if len(group) == 1:
            return pd.Series([0.0] * len(group), index=group.index)
        
        # Sort group by prob descending
        sorted_idx = group.sort_values(ascending=False).index
        max1 = group.loc[sorted_idx[0]]
        max2 = group.loc[sorted_idx[1]]
        
        res_series = pd.Series([max1] * len(group), index=group.index)
        # For the top item, the competition is the 2nd best
        res_series.loc[sorted_idx[0]] = max2
        return res_series
        
    res["sibling_max_prob"] = res.groupby("candidate_entity_id")[prob_col].transform(_sibling_max)
    
    # Is mutual best? (Top rank for S1 AND top rank for Candidate)
    # Top rank for candidate = its prob equals the cand_max
    res["is_top_for_cand"] = (res[prob_col] == cand_max).astype(int)
    res["is_mutual_best"] = ((res["rank_in_s1"] == 1) & (res["is_top_for_cand"] == 1)).astype(int)

    return res
