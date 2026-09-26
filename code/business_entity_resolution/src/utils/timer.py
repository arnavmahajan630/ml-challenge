"""
src/utils/timer.py — Wall-clock timing context manager.

Usage:
    with Timer("blocking") as t:
        run_blocking(...)
    print(t.elapsed_minutes)  # float
"""
from __future__ import annotations

import time
from typing import Optional


class Timer:
    """Context manager that records wall-clock elapsed time.

    Args:
        name: Human-readable label for the timed operation.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed seconds; 0 if not yet completed."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    @property
    def elapsed_minutes(self) -> float:
        """Elapsed minutes (float)."""
        return self.elapsed_seconds / 60.0

    def __repr__(self) -> str:
        return f"Timer({self.name!r}, elapsed={self.elapsed_seconds:.2f}s)"
