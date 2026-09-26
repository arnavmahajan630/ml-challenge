"""
src/eval/report.py — Utilities for writing the evaluation reports.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def append_experiment_result(
    csv_path: str | Path,
    exp_id: str,
    desc: str,
    oof_f05: float,
    loco_f05: float,
    delta: float,
    ci_low: float,
    ci_high: float,
    kept: str,
    notes: str = ""
):
    path = Path(csv_path)
    header = not path.exists()
    
    with open(path, "a") as f:
        if header:
            f.write("timestamp,exp_id,desc,oof_f05,loco_f05,delta,ci_low,ci_high,kept,notes\n")
            
        ts = datetime.now(tz=timezone.utc).isoformat()
        row = f"{ts},{exp_id},{desc},{oof_f05:.4f},{loco_f05:.4f},{delta:.4f},{ci_low:.4f},{ci_high:.4f},{kept},{notes}\n"
        f.write(row)
        
def write_status_md(path: str | Path, content: str):
    with open(path, "w") as f:
        f.write(content)
