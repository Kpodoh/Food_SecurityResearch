# Outputs

Generated tables and arrays are written here. The folder is committed but
its generated contents are not (see `.gitignore` at the repo root if you
add one). Re-running the scripts recreates everything below.

## Files produced

| File | Written by | Contents |
|---|---|---|
| `permutation_importance_59.csv` | `02_rf_permutation_importance.py` | Ranked features with permutation-importance scores. |
| `horizon_counts.csv` | `03_horizon_splits.py` | Sample sizes per horizon (H1/H2/H3). |
| `results_main.csv` | `05_train_evaluate_all_horizons.py` | Nested-CV macro/weighted F1, accuracy per model × horizon. |
| `oofA_{H}_{model}.npy` | `05_train_evaluate_all_horizons.py` | Out-of-fold predicted probabilities (rows = households, cols = 4 classes). Consumed by scripts 08 and 09. |
| `oof_index_{H}.npy` | `05_train_evaluate_all_horizons.py` | Row indices aligning the OOF arrays back to `faps_final.csv`. |
| `results_equal_n.csv` | `06_equal_n_controlled_experiment.py` | Macro F1 at equal sample size (n = 1,345), 20 repetitions. |
| `results_ablation.csv` | `07_ablation_imbalance_strategies.py` | Five imbalance strategies × horizon. |
| `calibration_metrics.csv` | `08_probability_calibration.py` | Brier score, ECE, climatological skill score per model × horizon. |
| `residence_risk_ci.csv` | `09_bootstrap_residence_risk.py` | Bootstrap mean risk and 95% CI by residence (metro / nonmetro) × horizon. |

`{H}` is `H1`, `H2`, or `H3`; `{model}` is `xgb`, `rf`, `svm`, or `nb`.
