"""
03_horizon_splits.py
--------------------
Partition households into the three prediction horizons (Methods 3.4).

The prediction horizon is defined by `startmon`, the survey start month
of a household's food-acquisition observation window. Households observed
over a longer window support prediction further into the future. The
three horizons are nested:

    H1 (1 month)  : all households                       n = 4826
    H2 (3 months) : households with >= 3 months of data  n = 3518
    H3 (6 months) : households with >= 6 months of data  n = 1345

so that D_H3 subset of D_H2 subset of D_H1. The nesting is deliberate:
it lets the horizons be compared on progressively longer-window
households without introducing households that differ on anything other
than window length. The equal-N experiment (script 06) removes even the
sample-size difference.

This script derives the horizon membership masks and reports the counts;
downstream scripts recompute the masks from the same rule so each is
standalone.

Input : data/faps_final.csv
Output: outputs/horizon_counts.csv
"""

import argparse
import os

import pandas as pd

# Minimum months of observation required for each horizon.
HORIZON_MIN_MONTHS = {"H1": 1, "H2": 3, "H3": 6}


def horizon_masks(df: pd.DataFrame) -> dict:
    """Return a boolean mask per horizon. `startmon` counts the months of
    the observation window; nesting is enforced by the >= comparison."""
    return {h: (df["startmon"] >= m) for h, m in HORIZON_MIN_MONTHS.items()}


def main(data_dir: str, out_dir: str) -> None:
    df = pd.read_csv(os.path.join(data_dir, "faps_final.csv"))
    masks = horizon_masks(df)

    rows = []
    for h, mask in masks.items():
        n = int(mask.sum())
        rows.append({"horizon": h,
                     "min_months": HORIZON_MIN_MONTHS[h],
                     "n": n})
        print(f"{h}: n = {n}")

    # Confirm the nesting D_H3 ⊆ D_H2 ⊆ D_H1.
    assert (masks["H3"] <= masks["H2"]).all(), "H3 not nested in H2"
    assert (masks["H2"] <= masks["H1"]).all(), "H2 not nested in H1"
    print("Nesting confirmed: D_H3 ⊆ D_H2 ⊆ D_H1")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "horizon_counts.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Written: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./outputs")
    a = ap.parse_args()
    main(a.data, a.out)
