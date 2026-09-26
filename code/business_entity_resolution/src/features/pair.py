"""
src/features/pair.py — Pairwise string and numeric features.

No leakage allowed. Only relies on the pair of normalized text records.
Features:
- Token overlap (Jaccard, Dice, overlap_coeff)
- Discriminative-token IDF sum (new in v2)
- Skeleton Jaccard (new in v2)
- RapidFuzz distances (Jaro-Winkler, Levenshtein, Indel)
- Monge-Elkan (new in v2)
- Exact flag matches (digits, house_no, postcode, flags)
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def _jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def _dice(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return 2 * len(set1 & set2) / (len(set1) + len(set2))

def _overlap(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))

def _monge_elkan(tokens1: List[str], tokens2: List[str]) -> float:
    """Monge-Elkan using Jaro-Winkler."""
    try:
        from rapidfuzz.distance import JaroWinkler
    except ImportError:
        return 0.0
        
    if not tokens1 or not tokens2:
        return 0.0
        
    total_score = 0.0
    for t1 in tokens1:
        best = max([JaroWinkler.normalized_similarity(t1, t2) for t2 in tokens2])
        total_score += best
    return total_score / len(tokens1)


def build_pair_features(
    pairs_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    idf_map: Dict[str, float] = None
) -> pd.DataFrame:
    """Compute string similarity features for a set of pairs."""
    try:
        from rapidfuzz.distance import JaroWinkler, Levenshtein, Indel
    except ImportError:
        JaroWinkler = None

    if idf_map is None:
        idf_map = {}

    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s23_dict = s23_df.set_index("entity_id").to_dict("index")

    rows = []
    for _, row in pairs_df.iterrows():
        s1_id = row["source1_entity_id"]
        c_id = row["candidate_entity_id"]
        
        s1_rec = s1_dict.get(s1_id, {})
        c_rec = s23_dict.get(c_id, {})
        
        n1 = str(s1_rec.get("name_norm", ""))
        n2 = str(c_rec.get("name_norm", ""))
        a1 = str(s1_rec.get("addr_norm", ""))
        a2 = str(c_rec.get("addr_norm", ""))
        
        nt1 = set(str(s1_rec.get("name_tokens", "")).split())
        nt2 = set(str(c_rec.get("name_tokens", "")).split())
        at1 = set(str(s1_rec.get("addr_tokens", "")).split())
        at2 = set(str(c_rec.get("addr_tokens", "")).split())
        
        sk1 = set(str(s1_rec.get("name_skel", "")).split())
        sk2 = set(str(c_rec.get("name_skel", "")).split())

        feats = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": c_id,
        }
        
        # Jaccard/Dice/Overlap
        feats["name_tok_jaccard"] = _jaccard(nt1, nt2)
        feats["name_tok_dice"] = _dice(nt1, nt2)
        feats["name_tok_overlap"] = _overlap(nt1, nt2)
        feats["addr_tok_jaccard"] = _jaccard(at1, at2)
        feats["name_skel_jaccard"] = _jaccard(sk1, sk2)
        
        # IDF sum (discriminative token match)
        shared_nt = nt1 & nt2
        feats["name_shared_idf"] = sum(idf_map.get(t, 1.0) for t in shared_nt) if shared_nt else 0.0
        
        # RapidFuzz
        if JaroWinkler:
            feats["name_jw"] = JaroWinkler.normalized_similarity(n1, n2)
            feats["addr_jw"] = JaroWinkler.normalized_similarity(a1, a2)
            feats["name_lev_sim"] = Levenshtein.normalized_similarity(n1, n2)
            feats["addr_lev_sim"] = Levenshtein.normalized_similarity(a1, a2)
        
        # Monge-Elkan
        feats["name_me"] = _monge_elkan(list(nt1), list(nt2))
        
        # Exact flag matches
        feats["exact_postcode"] = int(str(s1_rec.get("postcode_like", "")) == str(c_rec.get("postcode_like", "")) and bool(s1_rec.get("postcode_like")))
        feats["exact_house_no"] = int(str(s1_rec.get("house_no", "")) == str(c_rec.get("house_no", "")) and bool(s1_rec.get("house_no")))
        
        rows.append(feats)
        
    feat_df = pd.DataFrame(rows)
    # Merge back any blocking scores (B1-B8 ranks/scores, etc) from pairs_df
    # We drop source/cand IDs to avoid duplication in merge
    merged = pd.merge(pairs_df, feat_df, on=["source1_entity_id", "candidate_entity_id"])
    return merged
