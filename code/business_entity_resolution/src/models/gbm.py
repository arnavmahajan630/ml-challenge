"""
src/models/gbm.py — LightGBM trainer and predictor.

Used for the main Pair model and the Prefilter.
"""
from __future__ import annotations

import gc
from typing import Dict, List, Optional, Tuple, Any

import lightgbm as lgb
import numpy as np
import pandas as pd


class GBMPairModel:
    def __init__(self, params: Dict[str, Any], seed_bag: List[int] = [42, 123, 777]):
        self.params = params
        self.seed_bag = seed_bag
        self.models = []
        self.feature_names = []
        
    def fit(self, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> "GBMPairModel":
        self.feature_names = list(X.columns)
        self.models = []
        
        for seed in self.seed_bag:
            p = self.params.copy()
            p["seed"] = seed
            p["deterministic"] = True
            
            dtrain = lgb.Dataset(X, label=y, group=groups)
            # Just simple training. In full pipeline we use early stopping with val set if needed,
            # but spec says fit on N-1 folds for OOF.
            n_estimators = p.pop("n_estimators", 1000)
            
            gbm = lgb.train(
                p,
                dtrain,
                num_boost_round=n_estimators
            )
            self.models.append(gbm)
            
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.zeros(len(X), dtype=np.float64)
        for gbm in self.models:
            preds += gbm.predict(X[self.feature_names])
        return preds / len(self.models)
        
    def feature_importance(self) -> pd.DataFrame:
        if not self.models:
            return pd.DataFrame()
            
        imps = np.zeros(len(self.feature_names))
        for gbm in self.models:
            imps += gbm.feature_importance(importance_type='gain')
        imps /= len(self.models)
        
        df = pd.DataFrame({"feature": self.feature_names, "importance": imps})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
