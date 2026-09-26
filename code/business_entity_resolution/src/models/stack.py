"""
src/models/stack.py — Level-2 Stacker (E11).

Blends Pair GBM and CrossEncoder probabilities (both calibrated).
Can be a simple LogisticRegression or another shallow LightGBM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

class Stacker:
    def __init__(self):
        self.model = LogisticRegression(penalty='none') # or small L2
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "Stacker":
        # X contains e.g., ["gbm_prob", "ce_prob", "context_features..."]
        self.feature_names = list(X.columns)
        self.model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_names])[:, 1]
