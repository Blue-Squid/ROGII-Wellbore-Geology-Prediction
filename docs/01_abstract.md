# ROGII Wellbore Geology Prediction: Model Architecture & Findings

## 1. Abstract

This document chronicles the design, implementation, and deployment of a self-contained, high-performance geostatistical inference pipeline targeting True Vertical Thickness (`tvt`) predictions across un-drilled stratigraphic horizons. Our architecture maps raw horizontal wellbore trajectory components ($X, Y, Z, MD$) and Gamma Ray ($GR$) sensor streams into a leakage-insulated, multi-fold tree ensemble.

Using a 5-fold cross-validation scheme balanced via `GroupKFold` on individual well identifiers, the model achieved a local Out-Of-Fold (OOF) RMSE baseline of **111.3031**. Deployment of this model via a standalone, zero-dependency cloud execution script on the Kaggle live evaluation engine yielded an official Leaderboard Ranking of **3040** with a metric score of **393.423**. The significant variance between local evaluation metrics and live public test sets isolates an operational domain gap: the feature space under-represents long-range geological macros, deep spatial trends, and regional formation dips, causing out-of-distribution tree degradation when predicting structural horizons on completely blind wells.

## 2. Infrastructure

- **Compute Setup:** Ubuntu 26.04 LTS / NVIDIA GeForce RTX 5080 Laptop GPU (16GB VRAM, Ada Lovelace Next Architecture / CUDA 12.x Core Runtime)
- **Data Engineering Engine:** `polars` (v1.x.x multi-threaded eager/lazy execution), `pyarrow` IPC memory mapping
- **Modeling Stack:** `LightGBM` (NVIDIA OpenCL/CUDA tree-building backends), `XGBoost`, `CatBoost`, `PyTorch`
- **Validation Protocol:** Strict, non-overlapping `GroupKFold` ($n=5$) anchored entirely on `well_id` to prevent the catastrophic leakage of spatial autocorrelation across contiguous sequence rows.
