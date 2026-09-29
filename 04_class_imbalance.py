"""
04_class_imbalance.py
---------------------
Inverse-frequency class weighting (Methods 3.5).

The four food-security classes are imbalanced (roughly 4.5 : 1 between
the largest and smallest). The framework handles this with inverse-
frequency class weighting alone: each class receives a weight inversely
proportional to its frequency, so the minority classes contribute
proportionally more to the loss. This is applied as `sample_weight` for
the gradient-boosted model and as `class_weight="balanced"` for the
Random Forest and SVM baselines.

An ablation (script 07) compared this against SMOTE-ENN oversampling and
against Nelder-Mead threshold optimisation; class weighting alone was
best, so no resampling or threshold tuning is used in the main pipeline.

This module exposes the weighting helpers used by the modelling,
equal-N, ablation, and calibration scripts, and prints the weights for
each horizon when run directly.
"""

import argparse
import os

import numpy as np
import pandas as pd

HORIZON_MIN_MONTHS = {"H1": 1, "H2": 3, "H3": 6}
TARGET = "adltfscat"


def class_weights(y: np.ndarray) -> dict:
    """Inverse-frequency weights, normalised so the mean weight is 1.

    w_c = N / (K * n_c)  for class c, with K classes and n_c its count.
    """
    classes, counts = np.unique(y, return_counts=True)
    w = counts.sum() / (len(counts) * counts)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


def sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-observation weights for models that take `sample_weight`
    (e.g. XGBoost). Computed from the training labels only."""
    wmap = class_weights(y)
    return np.array([wmap[int(v)] for v in y], dtype=float)


def main(data_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    for h, m in HORIZON_MIN_MONTHS.items():
        y = df.loc[df["startmon"] >= m, TARGET].values
        classes, counts = np.unique(y, return_counts=True)
        imbalance = counts.max() / counts.min()
        print(f"\n{h}: n = {len(y)}, imbalance ratio = {imbalance:.2f}:1")
        for c, cnt in zip(classes, counts):
            print(f"  class {int(c)}: n = {cnt:5d}  "
                  f"weight = {class_weights(y)[int(c)]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    a = ap.parse_args()
    main(a.data)
