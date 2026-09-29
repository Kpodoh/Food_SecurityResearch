"""
06_equal_n_controlled_experiment.py
------------------------------------
Equal-N controlled experiment (Results 4.2).

Because the horizons are nested, H1 has far more households than H3
(4826 vs 1345). A gradient in performance across horizons could
therefore reflect sample size rather than the horizon itself. This
experiment removes that confound: H1 and H2 are randomly subsampled down
to n = 1345 (the H3 size), stratified on the target, and the model is
re-evaluated. The subsample is repeated 20 times with different seeds and
the macro F1 is averaged, so the estimate does not hinge on one draw.

If the horizon gradient survives at equal N, it is a property of the
prediction problem, not of how much data each horizon happens to have.

Uses the same leak-free single-loop CV and class-weight-only design as
the main script (an inner tuning loop is omitted here for speed, with
fixed sensible hyperparameters, because the question is the horizon
contrast under matched N, not the absolute best score).

Input : data/faps_final.csv, data/faps_selected_features_59.csv
Output: outputs/results_equal_n.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SEED = 42
TARGET = "adltfscat"
N_EQUAL = 1345         # the H3 sample size
N_REPS = 20
N_FOLDS = 5
HORIZON_MIN_MONTHS = {"H1": 1, "H2": 3, "H3": 6}

EXCLUDE = [
    "adltfscat", "adltfsraw", "foodsecureq1", "foodsecureq2",
    "foodsecureq3", "foodsufficient", "startmon",
]
OBJECT_COLS = ["primstoresnaptype", "altstoresnaptype"]

XGB_PARAMS = dict(objective="multi:softprob", num_class=4,
                  eval_metric="mlogloss", tree_method="hist",
                  n_estimators=200, max_depth=5, learning_rate=0.1,
                  random_state=SEED, n_jobs=-1)


def inv_freq_sample_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    w = counts.sum() / (len(counts) * counts)
    wmap = {int(c): float(wi) for c, wi in zip(classes, w)}
    return np.array([wmap[int(v)] for v in y], dtype=float)


def encode_global(df, features):
    X = df[features].copy()
    for col in X.columns:
        if X[col].dtype == object or col in OBJECT_COLS:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X.fillna(0)


def cv_macro_f1(X, y, seed):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), 4))
    for tr, va in skf.split(X, y):
        m = XGBClassifier(**XGB_PARAMS)
        m.fit(X[tr], y[tr], sample_weight=inv_freq_sample_weights(y[tr]))
        oof[va] = m.predict_proba(X[va])
    return f1_score(y, oof.argmax(1), average="macro")


def main(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    features = pd.read_csv(
        os.path.join(data_dir, "faps_selected_features_59.csv")
    )["feature"].tolist()
    features = [f for f in features if f not in EXCLUDE]
    Xfull = encode_global(df, features)

    rows = []
    for h, m in HORIZON_MIN_MONTHS.items():
        mask = (df["startmon"] >= m).values
        X_h = Xfull.values[mask]
        y_h = df[TARGET].values[mask]

        scores = []
        for r in range(N_REPS):
            if len(y_h) > N_EQUAL:
                # stratified subsample to the H3 size
                Xs, _, ys, _ = train_test_split(
                    X_h, y_h, train_size=N_EQUAL,
                    stratify=y_h, random_state=SEED + r)
            else:
                Xs, ys = X_h, y_h   # H3 already at N_EQUAL
            scores.append(cv_macro_f1(Xs, ys, seed=SEED + r))

        rows.append({"horizon": h, "n_equal": N_EQUAL, "reps": N_REPS,
                     "macro_f1_mean": round(float(np.mean(scores)), 4),
                     "macro_f1_std": round(float(np.std(scores)), 4)})
        print(f"{h}: equal-N macro F1 = {np.mean(scores):.4f} "
              f"± {np.std(scores):.4f}")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "results_equal_n.csv")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Written: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
