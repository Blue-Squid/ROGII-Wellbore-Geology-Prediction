"""High-performance cross-validation training execution script.

Processes geostatistical telemetry datasets using Polars, preserves spatial validation
boundaries via GroupKFold isolation, and serializes fold weights for inference pipelines.
"""

import gc
from pathlib import Path
from typing import List
import numpy as np
import numpy.typing as npt
import polars as pl

# Ingest robust, insulated sub-modules from the project architecture
from src.data_loader import load_raw_data, optimize_memory_usage
from src.features import generate_features
from src.model import WellboreModelLGBM
from src.validation import get_cv_splits, evaluate_model


def main() -> None:
    """Executes the complete cross-validation loop and serializes fold weights."""
    
    # 1. High-Performance IO Data Fetching
    # Load raw text records across multiple CPU cores in parallel
    raw_train_df: pl.DataFrame = load_raw_data()

    # 2. Sequential Feature Generation
    # Enforces strict sorting metrics internally by sorting on ['well_id', 'MD']
    train_df: pl.DataFrame = generate_features(raw_train_df)
    
    # Run memory footprint downcasting optimizations right after feature generation
    # to protect local RAM spaces without degrading coordinates or deep depth vectors.
    train_df = optimize_memory_usage(train_df)

    # 3. Dynamic Structural Column Exclusion Boundary Rules
    # We dynamically scan columns to match casing variations for the target column ('TVT' vs 'tvt')
    target_matches = [c for c in train_df.columns if c.lower() == "tvt"]
    if not target_matches:
        raise KeyError("CRITICAL ENFORCEMENT ERROR: Target target log column 'TVT' or 'tvt' not found in dataset schema.")
    target_col: str = target_matches[0]

    # Explicitly compile an aggressive drop list to eliminate target variants and grouping tokens
    # This prevents your LightGBM models from experiencing accidental geostatistical target leakage.
    ignore_list: List[str] = ["well_id", "id", "tvt_input", "tvt", "target"]
    
    cols_to_drop: List[str] = [
        col for col in train_df.columns 
        if col.lower() in ignore_list or col == target_col
    ]
    
    feature_df: pl.DataFrame = train_df.drop(cols_to_drop)
    feature_names: List[str] = feature_df.columns
    print(f"🚀 Feature space locked down. Training on {len(feature_names)} structured geostatistical vectors.")

    # 4. Strict Typed Array Generation for Memory Contiguity
    X: npt.NDArray[np.float64] = feature_df.to_numpy().astype(np.float64)
    y: npt.NDArray[np.float64] = train_df[target_col].to_numpy().astype(np.float64)

    # 5. Ensure Serialization Model Directory Structures Stand Ready
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    # 6. Geostatistical Isolation Cross-Validation
    # This invokes our GroupKFold tracking split engine, which asserts sorting conformity
    kf = get_cv_splits(train_df, n_splits=5)
    oof_rmse_scores: List[float] = []

    for fold_idx, (train_index, val_index) in enumerate(kf):
        print(f"\n=== Fold {fold_idx + 1}/{len(kf)} ===")

        # Extract isolated training and evaluation index slices safely
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Initialize the LightGBM wrapper structure
        model = WellboreModelLGBM()
        
        # Pass feature names directly down to maintain structural transparency during logging
        model.train(
            X_train, y_train, X_val, y_val, feature_names=feature_names
        )

        # 7. Out-Of-Fold Validation Performance Tracking
        y_pred: npt.NDArray[np.float64] = model.predict(X_val)
        
        # Enforce usage of centralized evaluation function to capture NaN/Inf safeguards
        rmse_score: float = evaluate_model(predictions=y_pred, actuals=y_val)
        oof_rmse_scores.append(rmse_score)
        print(f"Fold {fold_idx + 1} Target Log RMSE: {rmse_score:.4f}")

        # Serialize trained fold parameters out securely to persistent disk storage
        model.save_model(model_dir / f"lgbm_fold_{fold_idx}.txt")
        model.log_feature_importances()

        # 8. Aggressive VRAM/RAM Memory Disposal Protection
        # Wipe allocated array matrix spaces out of active threads to prevent page swapping
        del model, X_train, X_val, y_train, y_val, y_pred
        gc.collect()

    # 9. Complete Pipeline Metric Summary Reporting
    mean_oof_rmse: float = float(np.mean(oof_rmse_scores))
    print(f"\n====================================")
    print(f"🎯 Verified Final Mean OOF RMSE: {mean_oof_rmse:.4f}")
    print(f"====================================")


if __name__ == "__main__":
    main()