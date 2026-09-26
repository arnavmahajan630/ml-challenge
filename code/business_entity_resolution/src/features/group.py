"""
src/features/group.py — Group consistency features (E10 Rec2Rec model output).

Uses p_same(c, c') from the Rec2Rec model to generate cluster-level consistency
scores for each candidate.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_group_features(
    s1_cands_df: pd.DataFrame,
    rec2rec_df: pd.DataFrame,
    prob_col: str = "prob"
) -> pd.DataFrame:
    """
    Args:
        s1_cands_df: Base DataFrame with S1-C pairs and their base probabilities (prob_col).
        rec2rec_df: Pairwise Rec2Rec probabilities: 
                    [source1_entity_id, cand1, cand2, p_same]
                    
    Returns:
        DataFrame with new cluster consistency features merged.
    """
    if rec2rec_df.empty:
        s1_cands_df["cluster_consistency"] = 0.0
        return s1_cands_df
        
    # We want to measure: for candidate c in S1's pool, how well does it agree
    # with the *other* high-probability candidates in S1's pool?
    
    # Calculate weighted consistency: sum_c'( p_base(c') * p_same(c, c') )
    # This rewards a candidate if it is similar to other candidates that the base model likes.
    
    # Create base prob lookup
    base_probs = s1_cands_df.set_index(["source1_entity_id", "candidate_entity_id"])[prob_col].to_dict()
    
    consistencies = []
    
    for _, row in rec2rec_df.iterrows():
        s1 = row["source1_entity_id"]
        c1 = row["cand1"]
        c2 = row["cand2"]
        p_same = row["p_same"]
        
        p1 = base_probs.get((s1, c1), 0.0)
        p2 = base_probs.get((s1, c2), 0.0)
        
        # c1's consistency gets a boost from c2
        consistencies.append({"source1_entity_id": s1, "candidate_entity_id": c1, "cluster_consistency": p_same * p2})
        # c2's consistency gets a boost from c1
        consistencies.append({"source1_entity_id": s1, "candidate_entity_id": c2, "cluster_consistency": p_same * p1})
        
    cons_df = pd.DataFrame(consistencies)
    if not cons_df.empty:
        cons_df = cons_df.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False)["cluster_consistency"].sum()
        
    res = pd.merge(s1_cands_df, cons_df, on=["source1_entity_id", "candidate_entity_id"], how="left")
    res["cluster_consistency"] = res["cluster_consistency"].fillna(0.0)
    
    return res
