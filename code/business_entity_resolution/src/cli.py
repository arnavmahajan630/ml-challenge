"""
src/cli.py — Main CLI entrypoint for the BER pipeline.

Usage:
    python -m src.cli [stage] --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path
from src import config

def main():
    parser = argparse.ArgumentParser(description="BER Pipeline")
    parser.add_argument("stage", choices=["s0", "all"], help="Stage to run")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file path")
    
    args = parser.parse_args()
    cfg = config.load(args.config)
    
    print(f"Loaded config. Hash: {cfg.config_hash}")
    
    if args.stage == "s0":
        run_s0(cfg)
    elif args.stage == "all":
        print("Pipeline scaffolded successfully. Data needed to run.")

def run_s0(cfg: config.Cfg):
    print("Running S0 EDA...")
    from src.data import io, eda
    try:
        s1 = io.read_tsv(Path(cfg.data.train_dir) / cfg.data.train_s1)
        s2 = io.read_tsv(Path(cfg.data.train_dir) / cfg.data.train_s2)
        s3 = io.read_tsv(Path(cfg.data.train_dir) / cfg.data.train_s3)
        gt = io.read_gt(Path(cfg.data.train_dir) / cfg.data.train_gt)
        eda.generate_eda_report(s1, s2, s3, gt, Path(cfg.reports_dir) / "eda.md")
        print("S0 EDA complete.")
    except FileNotFoundError as e:
        print(f"Data not found: {e}")
        
if __name__ == "__main__":
    main()
