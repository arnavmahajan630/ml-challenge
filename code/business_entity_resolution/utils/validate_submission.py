#!/usr/8bin/env python3
"""
utils/validate_submission.py

Validates the format and constraints of matching_results.tsv and candidate_pairs.tsv.
Must run and print PASS.
"""
import sys
import pandas as pd
from pathlib import Path

def validate(output_dir: str):
    p = Path(output_dir)
    res_path = p / "matching_results.tsv"
    cand_path = p / "candidate_pairs.tsv"
    
    if not res_path.exists():
        print(f"FAIL: {res_path} not found")
        sys.exit(1)
        
    if not cand_path.exists():
        print(f"FAIL: {cand_path} not found")
        sys.exit(1)
        
    try:
        res = pd.read_csv(res_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
        cand = pd.read_csv(cand_path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
    except Exception as e:
        print(f"FAIL: could not parse TSVs: {e}")
        sys.exit(1)
        
    if "source1_entity_id" not in res.columns or "matched_entity_ids" not in res.columns:
        print("FAIL: missing columns in matching_results.tsv")
        sys.exit(1)
        
    if "source1_entity_id" not in cand.columns or "candidate_entity_ids" not in cand.columns:
        print("FAIL: missing columns in candidate_pairs.tsv")
        sys.exit(1)
        
    # Check that matches are a subset of candidates
    res_dict = res.set_index("source1_entity_id")["matched_entity_ids"].to_dict()
    cand_dict = cand.set_index("source1_entity_id")["candidate_entity_ids"].to_dict()
    
    for s1, m_str in res_dict.items():
        m_list = set(x.strip() for x in m_str.split(",") if x.strip())
        c_str = cand_dict.get(s1, "")
        c_list = set(x.strip() for x in c_str.split(",") if x.strip())
        
        if not m_list.issubset(c_list):
            print(f"FAIL: For {s1}, matched IDs {m_list - c_list} are not in candidate pairs.")
            sys.exit(1)
            
    print("PASS")
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        validate(sys.argv[1])
    else:
        validate("output")
