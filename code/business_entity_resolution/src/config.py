"""
src/config.py — Configuration loading, hashing, and global setup.

Usage:
    cfg = config.load("configs/base.yaml")
    # cfg.seed, cfg.data.train_dir, cfg.blocking.k_final, …

All YAML keys are accessible as attribute-style nested dataclasses (via
omegaconf's DictConfig wrapped in a thin typed Cfg class).  A sha256 hash
of the serialised YAML is computed and stored as cfg.config_hash; every
artifact in the cache is keyed by this hash.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Dataclasses that mirror configs/base.yaml structure
# ---------------------------------------------------------------------------

@dataclass
class DataCfg:
    train_dir: str = "dataset/train"
    test_dir: str = "dataset/test"
    train_s1: str = "train_source1.tsv"
    train_s2: str = "train_source2.tsv"
    train_s3: str = "train_source3.tsv"
    train_gt: str = "train_ground_truth.tsv"
    test_s1: str = "test_source1.tsv"
    test_s2: str = "test_source2.tsv"
    test_s3: str = "test_source3.tsv"


@dataclass
class BlockingCfg:
    k_tfidf: int = 30
    k_dense: int = 30
    k_phonetic: int = 20
    k_addr_anchor: int = 20
    k_union: int = 100
    k_final: int = 20
    k_max: int = 50
    prefilter_min_prob: float = 0.01
    country_partition: str = "auto"


@dataclass
class BiEncoderCfg:
    base: str = "intfloat/multilingual-e5-base"
    revision: str = ""
    max_len: int = 96
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 64
    label_free_frac: float = 0.20


@dataclass
class GBMCfg:
    objective: str = "binary"
    num_leaves: int = 63
    n_estimators: int = 1000
    seed_bag: List[int] = field(default_factory=lambda: [42, 123, 777])
    deterministic: bool = True
    force_col_wise: bool = True
    num_threads: int = 4


@dataclass
class CrossEncoderCfg:
    base: str = "xlm-roberta-base"
    revision: str = ""
    max_len: int = 160
    lr: float = 2e-5
    epochs: int = 3
    warmup_frac: float = 0.06
    neg_pos_ratio: int = 6
    focal_gamma: Optional[float] = None


@dataclass
class Rec2RecCfg:
    enabled: bool = False
    num_leaves: int = 31


@dataclass
class EntityModelCfg:
    enabled: bool = False
    num_leaves: int = 31


@dataclass
class LLMRerankerCfg:
    base: str = "microsoft/Phi-4-mini-instruct"
    revision: str = ""
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 1e-4
    epochs: int = 2
    uncertain_lo: float = 0.02
    uncertain_hi: float = 0.98
    use_qlora: bool = False


@dataclass
class ModelCfg:
    biencoder: BiEncoderCfg = field(default_factory=BiEncoderCfg)
    gbm: GBMCfg = field(default_factory=GBMCfg)
    cross_encoder: CrossEncoderCfg = field(default_factory=CrossEncoderCfg)
    rec2rec: Rec2RecCfg = field(default_factory=Rec2RecCfg)
    entity_model: EntityModelCfg = field(default_factory=EntityModelCfg)
    llm_reranker: LLMRerankerCfg = field(default_factory=LLMRerankerCfg)


@dataclass
class SageMakerCfg:
    instance_type: str = "ml.g6e.xlarge"
    role_arn: str = ""


@dataclass
class TrainingCfg:
    backend: str = "local"
    sagemaker: SageMakerCfg = field(default_factory=SageMakerCfg)


@dataclass
class DecodeCfg:
    mode: str = "threshold_relative"
    threshold: float = 0.5
    alpha: float = 0.5
    singleton_gate: float = 0.3
    prob_power_grid: List[float] = field(
        default_factory=lambda: [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
    )
    empty_bonus_grid: List[float] = field(
        default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    )
    lambda_miss: float = 0.0
    entity_model_variant: str = "none"
    conflict_mode: str = "hard"
    conflict_delta: float = 0.2
    soft_lambda0: float = 0.1


@dataclass
class S3Cfg:
    bucket: str = ""
    prefix: str = "ber"


@dataclass
class Cfg:
    """Top-level configuration dataclass."""

    seed: int = 42
    data: DataCfg = field(default_factory=DataCfg)
    blocking: BlockingCfg = field(default_factory=BlockingCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    decode: DecodeCfg = field(default_factory=DecodeCfg)
    s3: S3Cfg = field(default_factory=S3Cfg)
    hf_hub_offline: bool = False
    artifacts_dir: str = "artifacts"
    output_dir: str = "output"
    reports_dir: str = "reports"

    # Computed at load time — not serialised
    config_hash: str = field(default="", init=False, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base* (in-place)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_cfg(d: Dict[str, Any]) -> Cfg:
    """Convert a raw YAML dict to a typed Cfg dataclass (best-effort)."""

    def _make(cls, src):
        if not isinstance(src, dict):
            return src
        kwargs: Dict[str, Any] = {}
        hints = cls.__dataclass_fields__  # type: ignore[attr-defined]
        for fname, finfo in hints.items():
            if fname in ("config_hash",):
                continue
            val = src.get(fname, finfo.default if finfo.default is not finfo.default_factory else None)  # type: ignore
            if val is None and finfo.default_factory is not None:  # type: ignore
                val = finfo.default_factory()  # type: ignore
            ft = finfo.type  # type: ignore
            # Recursively handle nested dataclasses
            sub_cls = {
                "DataCfg": DataCfg,
                "BlockingCfg": BlockingCfg,
                "ModelCfg": ModelCfg,
                "BiEncoderCfg": BiEncoderCfg,
                "GBMCfg": GBMCfg,
                "CrossEncoderCfg": CrossEncoderCfg,
                "Rec2RecCfg": Rec2RecCfg,
                "EntityModelCfg": EntityModelCfg,
                "LLMRerankerCfg": LLMRerankerCfg,
                "TrainingCfg": TrainingCfg,
                "SageMakerCfg": SageMakerCfg,
                "DecodeCfg": DecodeCfg,
                "S3Cfg": S3Cfg,
            }.get(ft)
            if sub_cls and isinstance(val, dict):
                val = _make(sub_cls, val)
            elif sub_cls and isinstance(src.get(fname), dict):
                val = _make(sub_cls, src[fname])
            kwargs[fname] = val
        return cls(**kwargs)

    return _make(Cfg, d)


def _compute_hash(raw_yaml: str) -> str:
    return hashlib.sha256(raw_yaml.encode()).hexdigest()[:8]


def load(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Cfg:
    """Load YAML config, compute hash, seed everything, init cache + S3.

    Args:
        path: Path to a YAML config file (e.g. ``configs/base.yaml``).
        overrides: Optional dict of key-value pairs that override the YAML.

    Returns:
        Populated :class:`Cfg` instance with ``config_hash`` set.
    """
    path = Path(path)
    raw_yaml = path.read_text(encoding="utf-8")
    raw_dict: Dict[str, Any] = yaml.safe_load(raw_yaml) or {}

    if overrides:
        _deep_update(raw_dict, overrides)
        raw_yaml = yaml.dump(raw_dict, sort_keys=True)

    cfg = _dict_to_cfg(raw_dict)
    cfg.config_hash = _compute_hash(yaml.dump(raw_dict, sort_keys=True))

    # Global seed
    from .utils.seed import seed_everything
    seed_everything(cfg.seed)

    # HF offline mode
    if cfg.hf_hub_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # Init cache
    from .utils import cache as _cache
    _cache.init(cfg.artifacts_dir)

    # Init S3
    from .utils import s3 as _s3
    _s3.init(cfg.s3.bucket, cfg.s3.prefix)

    return cfg


def get_git_hash() -> str:
    """Return the short HEAD commit hash, or 'nogit' / 'dirty' if unavailable."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).strip().decode()
        # Check for dirty working tree
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        )
        return h + ("-dirty" if dirty != 0 else "")
    except Exception:
        return "nogit"
