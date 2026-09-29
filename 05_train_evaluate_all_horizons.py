"""
05_train_evaluate_all_horizons.py
---------------------------------
Train and evaluate MH-XGB and the baselines at all three horizons using
leak-free nested cross-validation (Results 4.1).

This is the core modelling script and the source of every headline
number in the paper. Design points that make the measurement honest:

* **Locked feature set.** Only the 59 validated features are used
  (data/faps_selected_features_59.csv). The target and its derivatives
  are excluded. Selecting features here from the full column set would
  leak the outcome into the model — an earlier version of this pipeline
  did exactly that and scored a spurious 1.0000.

* **Nested cross-validation.** An outer 5-fold stratified loop measures
  generalisation; an inner 3-fold loop tunes hyperparameters on the
  outer training portion only. There is no separate grid-search step,
  because tuning on the full dataset before cross-validation would leak
  validation information into the reported scores.

* **Fold-contained preprocessing.** Label encoders are fit globally on
  the full column (categories are fixed and known, so this leaks
  nothing), but scaling (SVM only) and class weights are computed inside
  each outer training fold. Class weights come from the training labels,
  never the validation labels.

* **Class weighting only.** No resampling, no threshold optimisation.
  Predictions are plain argmax of the predicted class probabilities.

Out-of-fold predicted probabilities are saved so that calibration
(script 08) and residence risk (script 09) are computed on genuinely
held-out predictions.

Input : data/faps_final.csv, data/faps_selected_features_59.csv
Output: outputs/results_main.csv
        outputs/oofA_{H}_{model}.npy, outputs/oof_index_{H}.npy
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

SEED = 42
TARGET = "adltfscat"
N_OUTER, N_INNER = 5, 3
HORIZON_MIN_MONTHS = {"H1": 1, "H2": 3, "H3": 6}

EXCLUDE = [
    "adltfscat", "adltfsraw", "foodsecureq1", "foodsecureq2",
    "foodsecureq3", "foodsufficient", "startmon",
]
OBJECT_COLS = ["primstoresnaptype", "altstoresnaptype"]

# Hyperparameter grids searched in the inner loop.
GRIDS = {
    "xgb": {"n_estimators": [100, 200], "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1]},
    "rf":  {"n_estimators": [100, 200], "max_depth": [None, 10]},
    "svm": {"C": [0.1, 1.0, 10.0], "gamma": ["scale"]},
    "nb":  {},   # GaussianNB has nothing to tune here
}


# ---------------------------------------------------------------- helpers
def inv_freq_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    w = counts.sum() / (len(counts) * counts)
    wmap = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([wmap[int(v)] for v in y], dtype=float)


def make_model(name: str, params: dict):
    if name == "xgb":
        return XGBClassifier(
            objective="multi:softprob", num_class=4, eval_metric="mlogloss",
            tree_method="hist", random_state=SEED, n_jobs=-1, **params)
    if name == "rf":
        return RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=SEED, **params)
    if name == "svm":
        return SVC(class_weight="balanced", probability=True,
                   random_state=SEED, **params)
    if name == "nb":
        return GaussianNB()
    raise ValueError(name)


def fit_model(name, params, Xtr, ytr, scaler_needed):
    """Fit with the right imbalance handling for each model family."""
    model = make_model(name, params)
    if name == "xgb":
        model.fit(Xtr, ytr, sample_weight=inv_freq_sample_weights(ytr))
    elif name == "nb":
        # GaussianNB: pass priors implicitly via sample_weight for balance.
        model.fit(Xtr, ytr, sample_weight=inv_freq_sample_weights(ytr))
    else:
        model.fit(Xtr, ytr)   # rf / svm carry class_weight="balanced"
    return model


def encode_global(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Fit label encoders on the full column (fixed category domains) so
    no validation fold meets an unseen label."""
    X = df[features].copy()
    for col in X.columns:
        if X[col].dtype == object or col in OBJECT_COLS:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(0)


def inner_select(name, Xtr, ytr, scaler_needed):
    """Pick hyperparameters by inner-fold macro F1 on the training portion."""
    grid = list(ParameterGrid(GRIDS[name])) or [{}]
    if len(grid) == 1:
        return grid[0]
    skf = StratifiedKFold(N_INNER, shuffle=True, random_state=SEED)
    best, best_score = grid[0], -np.inf
    for params in grid:
        scores = []
        for itr, iva in skf.split(Xtr, ytr):
            Xi, Xv = Xtr[itr], Xtr[iva]
            if scaler_needed:
                sc = StandardScaler().fit(Xi)
                Xi, Xv = sc.transform(Xi), sc.transform(Xv)
            m = fit_model(name, params, Xi, ytr[itr], scaler_needed)
            scores.append(f1_score(ytr[iva], m.predict(Xv), average="macro"))
        mean = float(np.mean(scores))
        if mean > best_score:
            best, best_score = params, mean
    return best


# ------------------------------------------------------------------- main
def run_horizon(name, X, y, scaler_needed):
    """Outer nested CV for one model at one horizon.
    Returns (macro_f1, weighted_f1, accuracy, oof_proba)."""
    skf = StratifiedKFold(N_OUTER, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y), 4), dtype=float)
    for tr, va in skf.split(X, y):
        Xtr, Xva = X[tr], X[va]
        best = inner_select(name, Xtr, y[tr], scaler_needed)
        if scaler_needed:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
        model = fit_model(name, best, Xtr, y[tr], scaler_needed)
        oof[va] = model.predict_proba(Xva)
    pred = oof.argmax(axis=1)          # plain argmax — no threshold tuning
    return (f1_score(y, pred, average="macro"),
            f1_score(y, pred, average="weighted"),
            accuracy_score(y, pred),
            oof)


def main(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    features = pd.read_csv(
        os.path.join(data_dir, "faps_selected_features_59.csv")
    )["feature"].tolist()
    features = [f for f in features if f not in EXCLUDE]

    Xfull = encode_global(df, features)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for h, m in HORIZON_MIN_MONTHS.items():
        mask = (df["startmon"] >= m).values
        idx = np.where(mask)[0]
        X = Xfull.values[mask]
        y = df[TARGET].values[mask]
        np.save(os.path.join(out_dir, f"oof_index_{h}.npy"), idx)
        print(f"\n=== {h} (n = {len(y)}) ===")
        for name in ["xgb", "rf", "svm", "nb"]:
            mf1, wf1, acc, oof = run_horizon(
                name, X, y, scaler_needed=(name == "svm"))
            np.save(os.path.join(out_dir, f"oofA_{h}_{name}.npy"), oof)
            rows.append({"horizon": h, "model": name,
                         "macro_f1": round(mf1, 4),
                         "weighted_f1": round(wf1, 4),
                         "accuracy": round(acc, 4)})
            print(f"  {name:4s}  macroF1={mf1:.4f}  "
                  f"weightedF1={wf1:.4f}  acc={acc:.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "results_main.csv")
    out.to_csv(path, index=False)
    print(f"\nWritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
