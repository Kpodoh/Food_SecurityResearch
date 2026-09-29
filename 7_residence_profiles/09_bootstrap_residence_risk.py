"""
09_bootstrap_residence_risk.py
------------------------------
Residence-based risk profiles with bootstrap confidence intervals
(Results 4.4).

The study is residence-based, so the practical output is the predicted
risk of food insecurity by place of residence (metro vs nonmetro). Risk
here is the model's predicted probability of the two insecure classes,
read from the out-of-fold probabilities saved by script 05 — so the
figures are held-out, not in-sample.

A point estimate is not enough for a policy claim, especially at H3 where
the nonmetro subgroup is small. Each residence group's mean risk is
therefore bootstrapped (2000 resamples of the out-of-fold predictions)
to give a 95% percentile confidence interval.

Reported finding: at H1 and H2 the nonmetro group has clearly higher
predicted risk than the metro group, with non-overlapping intervals. At
H3 the nonmetro subgroup is small (n = 43) and the intervals overlap, so
the residence gap is reported as suggestive rather than established at
the six-month horizon.

The insecure classes are taken to be the upper two of the four ordered
categories (indices 2 and 3); adjust `INSECURE_CLASSES` if your target
coding differs.

Input : outputs/oofA_{H}_xgb.npy, outputs/oof_index_{H}.npy,
        data/faps_final.csv
Output: outputs/residence_risk_ci.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

SEED = 42
TARGET = "adltfscat"
RESIDENCE_COL = "nonmetro"     # 1 = nonmetro, 0 = metro
HORIZONS = ["H1", "H2", "H3"]
INSECURE_CLASSES = [2, 3]      # upper two of the four ordered categories
N_BOOT = 2000
MODEL = "xgb"


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        means[b] = sample.mean()
    return (float(values.mean()),
            float(np.percentile(means, 2.5)),
            float(np.percentile(means, 97.5)))


def main(data_dir, out_dir):
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    residence_all = df[RESIDENCE_COL].values

    rows = []
    for h in HORIZONS:
        idx = np.load(os.path.join(out_dir, f"oof_index_{h}.npy"))
        proba = np.load(os.path.join(out_dir, f"oofA_{h}_{MODEL}.npy"))
        risk = proba[:, INSECURE_CLASSES].sum(axis=1)   # P(insecure)
        residence = residence_all[idx]

        for label, code in [("metro", 0), ("nonmetro", 1)]:
            grp = risk[residence == code]
            if len(grp) == 0:
                continue
            mean, lo, hi = bootstrap_ci(grp)
            rows.append({"horizon": h, "residence": label, "n": int(len(grp)),
                         "mean_risk": round(mean, 4),
                         "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4)})
            print(f"{h} {label:8s} n={len(grp):4d}  "
                  f"risk={mean:.4f} [{lo:.4f}, {hi:.4f}]")

    out = pd.DataFrame(rows)
    path = os.path.join(out_dir, "residence_risk_ci.csv")
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(path, index=False)

    # Flag horizons where the metro/nonmetro intervals overlap.
    for h in HORIZONS:
        sub = out[out["horizon"] == h].set_index("residence")
        if {"metro", "nonmetro"} <= set(sub.index):
            overlap = not (sub.loc["nonmetro", "ci_low"]
                           > sub.loc["metro", "ci_high"])
            note = ("intervals overlap — gap suggestive only"
                    if overlap else "intervals separated — gap established")
            print(f"{h}: {note}")

    print(f"\nWritten: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
