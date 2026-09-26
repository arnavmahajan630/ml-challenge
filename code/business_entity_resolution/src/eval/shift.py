"""
src/eval/shift.py — Adversarial validation for covariate shift (France vs Train).

Trains a model to distinguish test records from train records.
Extracts top shifted features.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def check_covariate_shift(train_features: pd.DataFrame, test_features: pd.DataFrame) -> dict:
    """Train a classifier to distinguish train from test."""
    train_features = train_features.copy()
    test_features = test_features.copy()
    
    train_features["_is_test"] = 0
    test_features["_is_test"] = 1
    
    df = pd.concat([train_features, test_features], ignore_index=True)
    
    # Drop ID columns and target/leakage columns
    drop_cols = [c for c in df.columns if "id" in c or "label" in c or "prob" in c or c.startswith("_")]
    features = [c for c in df.columns if c not in drop_cols]
    
    X = df[features]
    y = df["_is_test"].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    
    imps = clf.feature_importances_
    feat_imps = pd.DataFrame({"feature": features, "importance": imps})
    feat_imps = feat_imps.sort_values("importance", ascending=False).head(20)
    
    return {
        "auc": auc,
        "top_features": feat_imps.to_dict("records")
    }
