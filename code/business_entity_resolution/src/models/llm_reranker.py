"""
src/models/llm_reranker.py — LoRA fine-tuned pairwise classifier (Optional E14).

Uses a small instruction-tuned model (e.g. Phi-4-mini-instruct) trained with LoRA
to classify (S1, S2) pairs as match/no-match. Used only for uncertain pairs.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

def train_llm_reranker(*args, **kwargs):
    """Stub for LLM Reranker (E14 gated)."""
    pass

def predict_llm_reranker(*args, **kwargs) -> np.ndarray:
    """Stub for LLM Reranker."""
    return np.array([])
