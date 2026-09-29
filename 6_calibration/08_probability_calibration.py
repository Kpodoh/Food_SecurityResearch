"""
08_probability_calibration.py
-----------------------------
Probability calibration and skill (Results 4.5).

Accuracy and F1 say whether the top predicted class is right; they say
nothing about whether the predicted probabilities are trustworthy. For a
food-security early-warning use, the probabilities are the product, so
they are assessed directly on the out-of-fold predictions saved by
script 05:

* **Multiclass Brier score** — mean squared error between the predicted
  probability vector and the one-hot outcome. Lower is better.
* **Climatological skill score** — the Brier score relative to a
  no-skill baseline that always predicts the marginal class frequencies.
  Positive means the model beats climatology; zero or negative means it
  does not.
* **Expected calibration error (ECE)** — the average gap between
  confidence and accuracy across probability bins. Lower is better.

Reported finding: XGB, RF, and SVM all beat the climatological baseline
at every horizon; Gaussian Naive Bayes does not, consistent with its
poor macro F1.

Input : outputs/oofA_{H}_{model}.npy, outputs/oof_index_{H}.npy,
        data/faps_final.csv
Output: outputs/calibration_metrics.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

TARGET = "adltfscat"
HORIZONS = ["H1", "H2", "H3"]
MODELS = ["xgb", "rf", "svm", "nb"]
N_CLASSES = 4
N_BINS = 10


def onehot(y, k=N_CLASSES):
    Y = np.zeros((len(y), k))
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def brier_multiclass(proba, y):
    return float(np.mean(np.sum((proba - onehot(y)) ** 2, axis=1)))


def climatological_brier(y):
    """Brier of the constant baseline that predicts marginal frequencies."""
    freq = np.bincount(y.astype(int), minlength=N_CLASSES) / len(y)
    proba = np.tile(freq, (len(y), 1))
    return brier_multiclass(proba, y)


def expected_calibration_error(proba, y, n_bins=N_BINS):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def main(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    y_all = df[TARGET].values

    rows = []
    for h in HORIZONS:
        idx = np.load(os.path.join(out_dir, f"oof_index_{h}.npy"))
        y = y_all[idx]
        bs_clim = climatological_brier(y)
        for model in MODELS:
            fp = os.path.join(out_dir, f"oofA_{h}_{model}.npy")
            if not os.path.exists(fp):
                print(f"  missing {fp} — run script 05 first; skipping")
                continue
            proba = np.load(fp)
            bs = brier_multiclass(proba, y)
            skill = 1.0 - bs / bs_clim
            ece = expected_calibration_error(proba, y)
            rows.append({"horizon": h, "model": model,
                         "brier": round(bs, 4),
                         "brier_climatology": round(bs_clim, 4),
                         "skill_score": round(skill, 4),
                         "ece": round(ece, 4)})
            print(f"{h} {model:4s}  Brier={bs:.4f}  "
                  f"skill={skill:+.4f}  ECE={ece:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "calibration_metrics.csv")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
