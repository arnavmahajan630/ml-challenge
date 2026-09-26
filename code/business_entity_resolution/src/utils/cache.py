"""
src/utils/cache.py — Parquet-based artifact cache keyed by (stage, config_hash).

Every pipeline stage writes its outputs here; downstream stages read from cache
so any stage can be re-run in isolation.

Layout:  artifacts/<stage>/<config_hash>/<name>.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

_ARTIFACTS_DIR: Optional[Path] = None


def init(artifacts_dir: str | Path) -> None:
    """Set the root artifacts directory (called once from config.load)."""
    global _ARTIFACTS_DIR
    _ARTIFACTS_DIR = Path(artifacts_dir)
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _root() -> Path:
    if _ARTIFACTS_DIR is None:
        raise RuntimeError("cache.init() must be called before using the cache.")
    return _ARTIFACTS_DIR


def artifact_path(stage: str, cfg_hash: str, name: str = "data") -> Path:
    """Return the parquet path for a given stage + config hash + file name."""
    return _root() / stage / cfg_hash / f"{name}.parquet"


def save(
    df: pd.DataFrame,
    stage: str,
    cfg_hash: str,
    name: str = "data",
) -> Path:
    """Save *df* as parquet and return the written path.

    Creates parent directories as needed.
    """
    path = artifact_path(stage, cfg_hash, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load(
    stage: str,
    cfg_hash: str,
    name: str = "data",
) -> pd.DataFrame:
    """Load and return the cached parquet DataFrame."""
    path = artifact_path(stage, cfg_hash, name)
    if not path.exists():
        raise FileNotFoundError(
            f"Cache miss: {path}. Run stage '{stage}' first."
        )
    return pd.read_parquet(path)


def exists(stage: str, cfg_hash: str, name: str = "data") -> bool:
    """Return True if the artifact exists on disk."""
    return artifact_path(stage, cfg_hash, name).exists()


def save_object(obj: Any, stage: str, cfg_hash: str, name: str) -> Path:
    """Pickle-save a non-DataFrame artifact (e.g. sklearn calibrator, list)."""
    import joblib  # type: ignore

    path = _root() / stage / cfg_hash / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_object(stage: str, cfg_hash: str, name: str) -> Any:
    """Load a pickled artifact."""
    import joblib  # type: ignore

    path = _root() / stage / cfg_hash / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cache miss: {path}")
    return joblib.load(path)


def object_exists(stage: str, cfg_hash: str, name: str) -> bool:
    return (_root() / stage / cfg_hash / f"{name}.pkl").exists()
