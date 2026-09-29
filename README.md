# MH-XGB — Multi-Horizon Food Security Prediction (FAPS)

Reproducible pipeline for **"Multi-Horizon Prediction of Household Food
Security Using a Gradient Boosting Framework"** (Paper 1).

The framework, **MH-XGB**, classifies households into the four ordered
food security categories of the USDA Core Food Security Module at one-,
three-, and six-month prediction horizons, using the USDA Food
Acquisition and Purchase Survey (FAPS). Class imbalance is handled with
inverse-frequency class weighting; every preprocessing step runs inside
the training portion of a nested cross-validation loop, so no
information from a validation fold reaches training.

> The horizon-specific TreeSHAP explainability analysis is **not** part
> of this repository. It belongs to the companion paper (EMH-XGB) and
> lives in its own repository.

## Pipeline stages

| Folder | Script | Paper section |
|---|---|---|
| `1_feature_selection/` | `01_domain_grouping.py` | Methods 3.3 |
| | `02_rf_permutation_importance.py` | Methods 3.3 |
| `2_horizon_partitioning/` | `03_horizon_splits.py` | Methods 3.4 |
| `3_modelling/` | `04_class_imbalance.py` | Methods 3.5 |
| | `05_train_evaluate_all_horizons.py` | Results 4.1 |
| `4_equal_sample_experiment/` | `06_equal_n_controlled_experiment.py` | Results 4.2 |
| `5_ablation/` | `07_ablation_imbalance_strategies.py` | Results 4.3 |
| `6_calibration/` | `08_probability_calibration.py` | Results 4.5 |
| `7_residence_profiles/` | `09_bootstrap_residence_risk.py` | Results 4.4 |

`05_train_evaluate_all_horizons.py` performs hyperparameter tuning
**inside** each outer training fold (an inner 3-fold loop) — there is no
separate grid-search script, because tuning on the full dataset before
cross-validation would leak information into the reported scores.

## What was deliberately left out

Two things that appeared in earlier drafts are **not** in the final
pipeline, because an ablation (`07_ablation_imbalance_strategies.py`)
showed they did not help:

- **SMOTE-ENN synthetic oversampling** — reduced macro F1 at every
  horizon, worst at H3 where the minority class is smallest.
- **Nelder-Mead probability threshold optimisation** — added nothing
  once inverse-frequency class weighting was applied (the optimiser
  converged to uniform weights).

The ablation script still runs all five configurations so the result is
reproducible, but the modelling scripts use class weighting alone.

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for pinned versions. The results in the paper were
produced with Python 3.13, scikit-learn 1.7.2, xgboost 3.3.0.

## Data

The FAPS public-use file is **not** included (USDA distribution terms).
See `data/README.md` for how to obtain it and where to place it.

## Running

Each script is standalone and reads its data path from the `--data`
argument (default `./data`). Run them in order:

```bash
python 1_feature_selection/02_rf_permutation_importance.py --data ./data
python 2_horizon_partitioning/03_horizon_splits.py         --data ./data
python 3_modelling/05_train_evaluate_all_horizons.py       --data ./data
python 4_equal_sample_experiment/06_equal_n_controlled_experiment.py --data ./data
python 5_ablation/07_ablation_imbalance_strategies.py      --data ./data
python 6_calibration/08_probability_calibration.py         --data ./data
python 7_residence_profiles/09_bootstrap_residence_risk.py --data ./data
```

Tables (CSV) and out-of-fold probability arrays (`.npy`) are written to
`outputs/`. Scripts 08 and 09 consume the out-of-fold arrays written by
script 05, so run 05 first.

## Reproducibility

All randomness is seeded (`SEED = 42`). Nested five-fold stratified
cross-validation with fold-contained preprocessing is used throughout.

## Citation

Kpodoh, H. E. (2026). *Multi-Horizon Prediction of Household Food
Security Using a Gradient Boosting Framework.* MSc thesis, Pan African
University Institute for Basic Sciences, Technology and Innovation
(PAUSTI).

## License

Code released under the MIT License. The FAPS data is subject to USDA
terms and is not redistributed here.
