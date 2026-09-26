"""
src/models/entity.py — Has-match classifier and cardinality model.

Uses S1-level aggregate features to predict:
1. P(G >= 1) — Does this S1 have any matches?
2. E[G] — (Optional) Cardinality prediction.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Any


class EntityModel:
    """Predicts entity-level properties like has_match (P(G >= 1))."""
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "EntityModel":
        self.feature_names = list(X.columns)
        dtrain = lgb.Dataset(X, label=y)
        
        p = self.params.copy()
        n_estimators = p.pop("n_estimators", 100)
        p["objective"] = "binary"
        
        self.model = lgb.train(p, dtrain, num_boost_round=n_estimators)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.feature_names])
