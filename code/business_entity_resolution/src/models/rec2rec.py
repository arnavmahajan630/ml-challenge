"""
src/models/rec2rec.py — Same-entity model for candidates (S2/S3 ↔ S2/S3).

Trained as a full standalone LightGBM with the exact same pairwise features
used by the main model, but applied to pairs of S2/S3 records that share
a ground-truth parent.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from .gbm import GBMPairModel

class Rec2RecModel:
    def __init__(self, params: dict):
        self.gbm = GBMPairModel(params=params)
        
    def fit(self, s23_pairs: pd.DataFrame, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "Rec2RecModel":
        self.gbm.fit(X, y, groups)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.gbm.predict(X)
        
def generate_s23_training_pairs(gt_df: pd.DataFrame) -> pd.DataFrame:
    """Generate positive and negative S2/S3 pairs from ground truth."""
    # Positive pairs: S2/S3 IDs that appear in the SAME matched_ids_list
    # Negative pairs: S2/S3 IDs from DIFFERENT matched_ids_lists (sample)
    
    pos_pairs = []
    cands_by_s1 = {}
    
    for _, row in gt_df.iterrows():
        s1 = row["source1_entity_id"]
        matched = row.get("matched_ids_list", [])
        cands_by_s1[s1] = matched
        
        if len(matched) >= 2:
            for i in range(len(matched)):
                for j in range(i + 1, len(matched)):
                    pos_pairs.append({
                        "source1_entity_id": s1, # keep for grouping
                        "cand1": matched[i],
                        "cand2": matched[j],
                        "label": 1
                    })
                    
    pos_df = pd.DataFrame(pos_pairs)
    
    # Simple negatives: for each S1, pick one cand, pair with random cand from other S1
    # Implement as needed for E10
    
    return pos_df
