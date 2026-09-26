"""
src/models/calibrate.py — Isotonic Regression calibration.

Ensures that output probabilities from tree models and NNs reflect true
expected match rates, which is required for the DP decoder.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

class Calibrator:
    def __init__(self):
        self.ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        
    def fit(self, probs: np.ndarray, labels: np.ndarray) -> "Calibrator":
        self.ir.fit(probs, labels)
        return self
        
    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self.ir.transform(probs)
