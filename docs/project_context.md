# ROGII Wellbore Geology Prediction - Project Context

## 1. Problem Statement

Drilling horizontal wells relies heavily on manual expert interpretation of limited subsurface data (sensor, seismic, logging tools). Small deviations from the target geological zone result in severe operational inefficiencies, resource waste, and environmental hazards.

**Objective:** Develop machine learning models to predict the precise geology (`tvt` - True Vertical Thickness) encountered along a horizontal wellbore to automate and improve drilling operations.

## 2. Evaluation & Targets

- **Target Variable:** `tvt` (Float) - Represents a spatial/geological metric.
- **Evaluation Metric:** Root Mean Squared Error (RMSE) between predicted `tvt` and actual `tvt`.
- **Submission Format:** A CSV containing `id` and predicted `tvt` for each row in the test set.

## 3. Operational Constraints (Kaggle Rules)

- **Environment:** Submissions run via Kaggle Notebooks.
- **Runtime:** Maximum 9 hours (CPU or GPU).
- **Network:** Internet access must be **DISABLED** during inference.
- **Dependencies:** All external data, custom pre-trained models, and non-standard Python wheels must be packaged as offline Kaggle Datasets (stored locally in `data/external/`).

## 4. Modeling Strategy & Pitfalls

- **Spatial Leakage:** Well log data exhibits high spatial autocorrelation. Standard random K-Fold cross-validation will cause severe data leakage and false-positive performance.
- **Validation:** Must utilize `GroupKFold` grouped by well identifiers (`well_id` or equivalent spatial grouping) to ensure models generalize to unseen wellbores.
- **Hardware:** Local training runs on an Ubuntu 26.04 machine equipped with an Nvidia RTX 5080. Frameworks (`polars`, `xgboost`, `pytorch`) must be explicitly configured to leverage GPU acceleration and manage VRAM efficiently.
