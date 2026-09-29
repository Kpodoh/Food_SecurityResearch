# Data

This folder holds the input data. **No data files are committed** — the
USDA National Household Food Acquisition and Purchase Survey (FAPS) is
distributed by USDA under its own terms and cannot be redistributed here.

## What the scripts expect

Place two files in this folder:

| File | What it is |
|---|---|
| `faps_final.csv` | The analysis file: one row per household (n = 4,826), the target `adltfscat`, the horizon key `startmon`, and all candidate features. |
| `faps_selected_features_59.csv` | The locked feature set — one column, `feature`, listing the 59 predictors retained after selection. Produced by `1_feature_selection/02_rf_permutation_importance.py`. |

If `faps_selected_features_59.csv` is absent, run the feature-selection
script first; it writes the file here.

## Obtaining FAPS

The FAPS public-use data is available from the USDA Economic Research
Service:

- https://www.ers.usda.gov/data-products/foodaps-national-household-food-acquisition-and-purchase-survey/

Follow the ERS access instructions, then derive `faps_final.csv` from the
public-use files. The variable construction (target coding of the USDA
Core Food Security Module into the four categories `adltfscat`, and the
`startmon` horizon key) is described in Section 3 of the paper.

## Target and leakage note

`adltfscat` is the four-class food security outcome. The following
columns are **derived from the outcome** and must never be used as
predictors — every script excludes them explicitly:

```
adltfscat, adltfsraw, foodsecureq1, foodsecureq2, foodsecureq3,
foodsufficient, startmon
```

`adltfsraw` in particular is the raw CFSM score the target is cut from
(correlation ≈ 0.95 with the label); leaving it in produces a spurious
perfect classifier.
