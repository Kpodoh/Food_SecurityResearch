"""
02_rf_permutation_importance.py
-------------------------------
Primary feature selection by Random Forest permutation importance,
validated against mutual information (Methods 3.3).

A Random Forest is fit on the full candidate set (target derivatives
excluded), and permutation importance is measured on held-out data:
each feature's values are shuffled and the drop in balanced accuracy is
recorded. Permutation importance is model-agnostic in interpretation and
robust to the scale and cardinality inflation that affects impurity-based
importance, which matters here because the candidates mix binary,
categorical, and continuous variables.

Mutual information is computed alongside as an independent check; it is
not used to select, only to confirm that the permutation ranking is not
an artefact of the Random Forest.

The top-ranked features are retained. Two policy-relevant variables
(`nonmetro`, `region`) are force-retained even if they fall below the
cut, because the study is residence-based and these must be present.

Input : data/faps_final.csv
Output: outputs/permutation_importance_59.csv   (ranked features + scores)
        data/faps_selected_features_59.csv       (the locked feature set)
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
N_KEEP = 59            # size of the locked feature set
TARGET = "adltfscat"

EXCLUDE = [
    "adltfscat", "adltfsraw", "foodsecureq1", "foodsecureq2",
    "foodsecureq3", "foodsufficient", "startmon",
]

FORCE_RETAIN = ["nonmetro", "region"]

# Known object columns that need encoding before tree fitting.
OBJECT_COLS = ["primstoresnaptype", "altstoresnaptype"]


def encode_objects(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object columns using a global encoder fit on the full
    column, so that no category is unseen in any downstream split."""
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == object or col in OBJECT_COLS:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    return X


def main(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))

    y = df[TARGET].values
    candidates = [c for c in df.columns if c not in EXCLUDE]
    X = encode_objects(df[candidates]).fillna(0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=SEED,
    )
    rf.fit(X_tr, y_tr)

    perm = permutation_importance(
        rf, X_te, y_te, n_repeats=20, n_jobs=-1,
        random_state=SEED, scoring="balanced_accuracy",
    )

    # Independent validation ranking (not used to select).
    mi = mutual_info_classif(X_tr, y_tr, random_state=SEED)

    ranked = (
        pd.DataFrame({
            "feature": candidates,
            "perm_importance": perm.importances_mean,
            "perm_std": perm.importances_std,
            "mutual_info": mi,
        })
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )
    ranked["rank"] = ranked.index + 1

    # Retain top N_KEEP, then force-retain the policy variables.
    kept = list(ranked.head(N_KEEP)["feature"])
    for f in FORCE_RETAIN:
        if f in candidates and f not in kept:
            kept.append(f)
            print(f"Force-retained below cut: {f} "
                  f"(rank {int(ranked.loc[ranked.feature == f, 'rank'].iloc[0])})")

    os.makedirs(out_dir, exist_ok=True)
    ranked.to_csv(os.path.join(out_dir, "permutation_importance_59.csv"),
                  index=False)
    pd.DataFrame({"feature": kept}).to_csv(
        os.path.join(data_dir, "faps_selected_features_59.csv"), index=False
    )

    print(f"\nTop 5 by permutation importance:")
    print(ranked.head(5)[["rank", "feature", "perm_importance"]].to_string(index=False))
    print(f"\nRetained {len(kept)} features -> "
          f"{os.path.join(data_dir, 'faps_selected_features_59.csv')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
