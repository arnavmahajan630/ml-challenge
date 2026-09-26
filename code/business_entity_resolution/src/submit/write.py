"""
src/submit/write.py — Submission file generation.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from ..data.io import format_matched_ids

def write_submission(
    predictions_map: dict[str, list[str]],
    pairs_df: pd.DataFrame,
    output_dir: str | Path
):
    """Write matching_results.tsv and candidate_pairs.tsv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # 1. matching_results.tsv
    rows = []
    # Ensure every S1 in test is present, even if empty
    s1_ids = sorted(list(predictions_map.keys()))
    for s1 in s1_ids:
        rows.append({
            "source1_entity_id": s1,
            "matched_entity_ids": format_matched_ids(predictions_map[s1])
        })
    match_df = pd.DataFrame(rows)
    match_df.to_csv(out / "matching_results.tsv", sep="\t", index=False)
    
    # 2. candidate_pairs.tsv
    # Spec: exactly the set of pairs the final scoring model ran inference on.
    # pairs_df must contain source1_entity_id, candidate_entity_id
    cand_groups = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()
    
    cand_rows = []
    for s1 in s1_ids:
        cands = cand_groups.get(s1, [])
        cand_rows.append({
            "source1_entity_id": s1,
            "candidate_entity_ids": format_matched_ids(cands)
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(out / "candidate_pairs.tsv", sep="\t", index=False)
