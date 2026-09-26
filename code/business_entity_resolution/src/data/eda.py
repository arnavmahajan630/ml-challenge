"""
src/data/eda.py — Exploratory Data Analysis.

Generates S0 EDA reports and sanity checks on sizes and overlaps.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

def generate_eda_report(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    out_path: str | Path
):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    n_s1 = len(s1_df)
    n_s2 = len(s2_df)
    n_s3 = len(s3_df)
    
    matched_lists = gt_df["matched_ids_list"].tolist()
    flat_matches = [x for lst in matched_lists for x in lst]
    
    singletons = sum(1 for lst in matched_lists if not lst)
    multi_match = sum(1 for lst in matched_lists if len(lst) > 1)
    
    # Are S2/S3 IDs assigned to multiple S1s?
    match_counts = pd.Series(flat_matches).value_counts()
    multi_assigned = (match_counts > 1).sum()
    
    report = f"""# Data EDA Report

- S1 records: {n_s1}
- S2 records: {n_s2}
- S3 records: {n_s3}
- Total S2+S3: {n_s2 + n_s3}

- S1 singletons (no match): {singletons} ({(singletons/n_s1)*100:.1f}%)
- S1 with >1 match: {multi_match} ({(multi_match/n_s1)*100:.1f}%)
- S2/S3 records assigned to multiple S1s: {multi_assigned}

## Distribution of Matches per S1
"""
    sizes = pd.Series([len(lst) for lst in matched_lists]).value_counts().sort_index()
    for size, count in sizes.items():
        report += f"- {size} matches: {count}\n"
        
    with open(out, "w") as f:
        f.write(report)
