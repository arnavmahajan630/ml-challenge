"""
src/data/io.py — Safe TSV read/write for the BER pipeline.

All files in the challenge are tab-separated.  Commas appear inside fields
(comma-separated ID lists), so we must NEVER use the default CSV parser.

Non-negotiable spec (PLAN.md §1.5):
    pd.read_csv(path, sep="\\t", dtype=str, keep_default_na=False,
                na_filter=False, quoting=csv.QUOTE_NONE)
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


def read_tsv(path: PathLike) -> pd.DataFrame:
    """Read a TSV file, treating every column as a string.

    - ``dtype=str``: all values are kept as strings; numeric IDs are NOT cast.
    - ``keep_default_na=False, na_filter=False``: empty cells become ``""``
      instead of ``NaN``.
    - ``quoting=csv.QUOTE_NONE``: prevents misinterpretation of quoted fields.

    Args:
        path: Path to the ``.tsv`` file.

    Returns:
        :class:`pd.DataFrame` with all columns as ``object`` (str) dtype.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TSV not found: {path}")
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_NONE,
    )


def write_tsv(df: pd.DataFrame, path: PathLike) -> None:
    """Write a DataFrame to a TSV file (no index, no quoting).

    Ensures the parent directory exists before writing.

    Args:
        df: DataFrame to serialise.
        path: Destination file path (``.tsv`` extension expected).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        sep="\t",
        index=False,
        quoting=csv.QUOTE_NONE,
    )


def read_gt(path: PathLike) -> pd.DataFrame:
    """Read the ground-truth file and normalise the ``matched_entity_ids`` column.

    The GT file has two columns:
        ``source1_entity_id``  |  ``matched_entity_ids``

    ``matched_entity_ids`` is a comma-separated list of S2-/S3- IDs, or empty
    string for singletons.

    Args:
        path: Path to ``train_ground_truth.tsv``.

    Returns:
        DataFrame with an added ``matched_ids_list`` column (list of str).
    """
    df = read_tsv(path)
    assert "source1_entity_id" in df.columns, "GT file missing 'source1_entity_id'"
    assert "matched_entity_ids" in df.columns, "GT file missing 'matched_entity_ids'"

    df["matched_ids_list"] = df["matched_entity_ids"].apply(
        lambda s: [x.strip() for x in s.split(",") if x.strip()] if s.strip() else []
    )
    return df


def format_matched_ids(ids: list[str]) -> str:
    """Serialise a list of matched IDs back to the submission comma-separated format.

    Deduplicates while preserving order.  Empty list → empty string.
    """
    seen: dict[str, None] = {}
    for i in ids:
        seen[i] = None
    return ",".join(seen.keys())
