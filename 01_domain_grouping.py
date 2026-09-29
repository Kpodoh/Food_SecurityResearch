"""
01_domain_grouping.py
----------------------
Organise the FAPS candidate variables into conceptual domains
(Methods 3.3).

This is the first, descriptive step of feature selection. Before any
importance is measured, the candidate predictors are grouped into
interpretable domains so that the retained feature set can be read as
covering the substantive drivers of household food security rather than
an arbitrary list of columns. The grouping is also used to force-retain
a small number of policy-relevant variables (residence and region) that
must appear in the model regardless of their permutation rank.

Input : data/faps_final.csv
Output: outputs/feature_domains.csv   (feature -> domain map)

The target and its derivatives are never treated as candidates.
"""

import argparse
import os

import pandas as pd

SEED = 42

# Columns derived from the outcome — never candidates, never predictors.
EXCLUDE = [
    "adltfscat",       # the four-class target
    "adltfsraw",       # raw CFSM score the target is cut from (~0.95 corr)
    "foodsecureq1",
    "foodsecureq2",
    "foodsecureq3",
    "foodsufficient",
    "startmon",        # horizon key, used for partitioning only
]

# Keyword patterns used to assign a candidate to a domain. First match wins;
# anything unmatched falls into "other". Patterns are matched case-insensitively
# against the column name.
DOMAIN_PATTERNS = {
    "demographics": ["age", "sex", "male", "female", "hhsize", "adult",
                     "child", "kids", "marital", "educ", "race", "hisp",
                     "employ", "work", "disab"],
    "economic": ["inc", "income", "pov", "poverty", "fincond", "wealth",
                 "asset", "snap", "wic", "benefit", "assist", "save"],
    "geography": ["metro", "nonmetro", "rural", "urban", "region", "state",
                  "division", "census"],
    "food_access": ["store", "grocery", "market", "distance", "dist",
                    "travel", "vehicle", "car", "shop", "access"],
    "food_spending": ["exp", "spend", "cost", "price", "dollar", "spent",
                      "purchase", "acq"],
    "housing": ["house", "home", "rent", "own", "tenure", "dwell"],
}

# Variables that must be retained regardless of importance rank (policy relevance).
FORCE_RETAIN = ["nonmetro", "region"]


def assign_domain(name: str) -> str:
    low = name.lower()
    for domain, keys in DOMAIN_PATTERNS.items():
        if any(k in low for k in keys):
            return domain
    return "other"


def main(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))

    candidates = [c for c in df.columns if c not in EXCLUDE]
    rows = [{"feature": c,
             "domain": assign_domain(c),
             "force_retain": c in FORCE_RETAIN}
            for c in candidates]

    out = pd.DataFrame(rows).sort_values(["domain", "feature"])
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feature_domains.csv")
    out.to_csv(path, index=False)

    print(f"{len(candidates)} candidate variables grouped into "
          f"{out['domain'].nunique()} domains.")
    print(out["domain"].value_counts().to_string())
    print(f"Force-retained: {FORCE_RETAIN}")
    print(f"Written: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
