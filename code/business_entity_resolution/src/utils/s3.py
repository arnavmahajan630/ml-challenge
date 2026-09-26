"""
src/utils/s3.py — Optional S3 artifact sync via boto3.

Falls back to a no-op (with a logged warning) when S3_BUCKET is not set
in the environment. This keeps the pipeline fully local-capable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .log import get_logger

logger = get_logger(__name__)

_BUCKET: Optional[str] = None
_PREFIX: str = "ber"


def init(bucket: str, prefix: str = "ber") -> None:
    """Configure S3 sync target. Called from config.load()."""
    global _BUCKET, _PREFIX
    env_bucket = os.environ.get("S3_BUCKET", bucket).strip()
    _BUCKET = env_bucket if env_bucket else None
    _PREFIX = prefix


def _enabled() -> bool:
    return bool(_BUCKET)


def _s3_key(local_path: Path, exp_id: str) -> str:
    return f"{_PREFIX}/{exp_id}/{local_path.name}"


def sync_up(local_path: str | Path, exp_id: str) -> None:
    """Upload *local_path* to S3 under ``<prefix>/<exp_id>/``.

    No-op if S3_BUCKET is not configured.
    """
    if not _enabled():
        logger.warning("S3 sync disabled (S3_BUCKET not set). Skipping upload.")
        return

    import boto3  # type: ignore

    local_path = Path(local_path)
    if not local_path.exists():
        logger.warning("sync_up: %s does not exist; skipping.", local_path)
        return

    s3 = boto3.client("s3")
    key = _s3_key(local_path, exp_id)
    logger.info("Uploading %s → s3://%s/%s", local_path, _BUCKET, key)
    s3.upload_file(str(local_path), _BUCKET, key)


def sync_down(local_path: str | Path, exp_id: str) -> None:
    """Download from S3 into *local_path*.

    No-op if S3_BUCKET is not configured.
    """
    if not _enabled():
        logger.warning("S3 sync disabled (S3_BUCKET not set). Skipping download.")
        return

    import boto3  # type: ignore

    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    key = _s3_key(local_path, exp_id)
    logger.info("Downloading s3://%s/%s → %s", _BUCKET, key, local_path)
    s3.download_file(_BUCKET, key, str(local_path))


def sync_dir_up(local_dir: str | Path, exp_id: str) -> None:
    """Recursively upload a local directory to S3."""
    if not _enabled():
        logger.warning("S3 sync disabled. Skipping directory upload.")
        return

    import boto3  # type: ignore

    local_dir = Path(local_dir)
    s3 = boto3.client("s3")
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            key = f"{_PREFIX}/{exp_id}/{rel}"
            s3.upload_file(str(p), _BUCKET, key)
            logger.info("Uploaded %s → s3://%s/%s", p, _BUCKET, key)
