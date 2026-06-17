# Experiment Tracker

## EXP-001 (Invalidated Baseline)

- **Date:** 2026-06-14
- **Model:** LightGBM (Default Parameters)
- **Compute Backend:** Host CPU
- **CV Strategy:** GroupKFold (n=5) grouped by `well_id`
- **Local OOF RMSE:** 16.9778 _(Invalid)_
- **Notes:** Superficially low error driven by extreme target leakage. The feature pipeline included `well_mean_TVT`, which allowed the model to cheat by peaking at the average target value across the evaluation validation fold.

---

## EXP-002 (Sanitized Geostatistical Baseline)

- **Date:** 2026-06-14
- **Model:** LightGBM (Default Parameters)
- **Compute Backend:** OpenCL GPU Framework (NVIDIA GeForce RTX 5080 Laptop GPU)
- **CV Strategy:** GroupKFold (n=5) grouped by `well_id`
- **Local OOF RMSE:** 117.6606 _(True Operational Baseline)_
- **Configuration Changes**:
  - Completely purged the `well_mean_TVT` leakage source from `src/features.py`.
  - Introduced physical geometry proxies: `Z_delta_well`, `MD_total_well`, and `cum_well_distance`.
  - Switched execution to the local RTX 5080 GPU via `device="gpu"`, achieving ultra-low latency tensor transfers (~0.04s data load times per split).
- **Analysis:** Removing the target lookup forced the model to rely natively on 3D spatial patterns. `X`, `Y`, and `MD_total_well` took over split dominance. This score is an honest representation of the baseline model's capacity to generalize to completely unseen wells.

---

## EXP-003 (Hyperparameter Optimization)

- **Date:** 2026-06-14
- **Model:** LightGBM + Optuna Tree Architecture Tuning
- **Compute Backend:** OpenCL GPU Framework (NVIDIA GeForce RTX 5080 Laptop GPU)
- **CV Strategy:** GroupKFold (n=5) grouped by `well_id`
- **Local OOF RMSE:** 111.3031
- **Configuration Changes**:
  - `learning_rate`: 0.13394
  - `num_leaves`: 166
  - `max_depth`: 10
  - `feature_fraction`: 0.6082
  - `min_data_in_leaf`: 195
  - `lambda_l1`: 0.3104
  - `lambda_l2`: 8.148e-07
- **Analysis:** Slashed RMSE by 6.36 points from the un-tuned baseline. The lower `feature_fraction` combined with high leaf constraints effectively regularized the model against aggressive local overfitting within individual well clusters. Total runtime took ~13.5 minutes on the RTX 5080 (~54.6s per 5-fold trial matrix).

---

## EXP-004 (Standalone Cloud Deployment Model)

- **Date:** 2026-06-17
- **Model Framework:** 5-Fold Blended GBDT Ensemble (`WellboreModelLGBM`)
- **Compute Backend:** Standalone Kaggle Notebook Run Environment (P100 Cloud GPU Infrastructure)
- **CV Strategy:** GroupKFold ($n=5$) grouped by `well_id`
- **Local OOF RMSE:** 111.3031
- **Public Leaderboard Score:** 393.423 `[Rank: 3040]`
- **Configuration Modifications**:
- Extracted local algorithmic implementations and re-compiled them into a zero-dependency, single-cell execution script.
- Implemented an automated cross-mount global data scanner using Polars (`global_input_root.rglob()`) to remain resilient against changing hidden private directory structures.
- Added a defensive dynamic alignment matrix block to handle structural feature missingness between training and test sets.

- **Analysis:** The script successfully evaluated all **14,151 evaluation rows**, generated clean predictions, and passed the Kaggle validation checker without any structural or data type anomalies. The divergence between the Local OOF RMSE (111.3) and the Public Leaderboard (393.4) establishes that the model is underfitting macro regional trends. Because GBDTs cannot extrapolate trends outside their known training coordinate bounds, the model relies too heavily on local spatial points ($X, Y, Z$) instead of relative geological markers, leading to out-of-distribution performance drops on unseen evaluation wells.

---
