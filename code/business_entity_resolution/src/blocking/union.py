"""
src/blocking/union.py — Combine all blockers, cap candidates, and prefilter.

Combines B1-B8 outputs.
- Outer join on (source1_entity_id, candidate_entity_id).
- blocker_mask bitmask creation.
- Hard cap at K_max (e.g. 50).
- (Optional) LightGBM prefilter to trim down to K_final before cross-encoder.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def union_blockers(
    blocker_dfs: Dict[str, pd.DataFrame],
    k_max: int = 50,
) -> pd.DataFrame:
    """Union all blocker dataframes into one candidate set.

    Args:
        blocker_dfs: Dict mapping blocker_id (e.g. "B1", "B4") to df.
                     Each df must have source1_entity_id, candidate_entity_id.
        k_max: Hard cap per S1 on the final number of candidates.

    Returns:
        Unified DataFrame with all scores/ranks and a `blocker_mask`.
    """
    if not blocker_dfs:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id", "blocker_mask"])

    # Start with all unique pairs
    pairs = set()
    for bid, df in blocker_dfs.items():
        if not df.empty:
            pairs.update(
                zip(df["source1_entity_id"], df["candidate_entity_id"])
            )
            
    unified = pd.DataFrame(list(pairs), columns=["source1_entity_id", "candidate_entity_id"])
    
    # Merge each blocker's features and build mask
    blocker_names = sorted(list(blocker_dfs.keys()))
    mask_series = np.zeros(len(unified), dtype=int)
    
    for i, bid in enumerate(blocker_names):
        df = blocker_dfs[bid]
        if df.empty:
            continue
            
        unified = pd.merge(unified, df, on=["source1_entity_id", "candidate_entity_id"], how="left")
        
        # Check if pair came from this blocker (e.g. score > 0 or rank not null)
        # B6,B7,B8 use bX_hit = 1
        if f"{bid.lower()}_hit" in df.columns:
            hit = unified[f"{bid.lower()}_hit"].fillna(0) > 0
        else:
            # TFIDF/Dense use ranks
            hit = unified[f"{bid.lower()}_rank"].notna()
            
        mask_series |= (hit.astype(int) << i)
        
    unified["blocker_mask"] = mask_series
    
    # Fill NAs
    score_cols = [c for c in unified.columns if "score" in c]
    rank_cols = [c for c in unified.columns if "rank" in c]
    hit_cols = [c for c in unified.columns if "hit" in c]
    
    unified[score_cols] = unified[score_cols].fillna(0.0)
    unified[rank_cols] = unified[rank_cols].fillna(999)
    unified[hit_cols] = unified[hit_cols].fillna(0.0)

    # Hard cap (heuristic: sum of inverted ranks, or just random if no scores)
    # Better heuristic: min rank across any blocker
    if k_max > 0 and len(unified) > 0:
        if rank_cols:
            unified["_min_rank"] = unified[rank_cols].min(axis=1)
        else:
            unified["_min_rank"] = 1
            
        unified = unified.sort_values(["source1_entity_id", "_min_rank"])
        unified = unified.groupby("source1_entity_id").head(k_max).reset_index(drop=True)
        unified = unified.drop(columns=["_min_rank"])

    return unified
