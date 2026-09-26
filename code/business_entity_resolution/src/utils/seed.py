"""
src/utils/seed.py — Global seed management for full reproducibility.

Sets seeds for: Python random, NumPy, PyTorch (CPU+CUDA), LightGBM, scikit-learn,
FAISS, and environment variables for deterministic CUDA operations.
"""
from __future__ import annotations

import os
import random


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs used in the pipeline for exact reproducibility.

    Must be called once at process start (via config.load()).
    After calling, also set CUBLAS_WORKSPACE_CONFIG in environment.

    Args:
        seed: Global integer seed. Default 42 per spec.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    except ImportError:
        pass

    # LightGBM — determinism via config, not a global seed function;
    # individual trainers set num_threads and deterministic=True in their params.

    # scikit-learn — uses numpy RNG, already seeded above.

    # FAISS — set thread count for determinism
    try:
        import faiss  # type: ignore
        faiss.omp_set_num_threads(1)
    except ImportError:
        pass
