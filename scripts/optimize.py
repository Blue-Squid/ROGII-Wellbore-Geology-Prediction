"""Optuna hyperparameter optimization script for the LightGBM Wellbore model.

Searches for optimal tree architectural constraints while strictly honoring 
GroupKFold spatial separation on the local RTX 5080 GPU using pre-computed,
memory-contiguous features to minimize IO overhead.
"""

import gc
from typing import Any, Dict, List, Tuple
import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import optuna
import polars as pl

# Ingest robust sub-modules from the project architecture
from src.data_loader import load_raw_data, optimize_memory_usage
from src.features import generate_features
from src.model import WellboreModelLGBM
from src.validation import get_cv_splits, evaluate_model

# Silence Optuna logging chatter to keep console outputs scannable
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
    trial: optuna.Trial,
    X: npt.NDArray[np.float32],  # Changed to float32 to align with hardware matrices
    y: npt.NDArray[np.float64],
    feature_names: List[str],
    kf: List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> float:
    """Optuna objective function maximizing spatial prediction accuracy.

    Args:
        trial (optuna.Trial): Current evaluation trial instance.
        X (npt.NDArray[np.float32]): Pre-computed training feature matrix.
        y (npt.NDArray[np.float64]): Pre-computed training target vector.
        feature_names (List[str]): Extracted feature column names.
        kf (List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]): GroupKFold 
            index splits configuration mapping array.

    Returns:
        float: Calculated mean out-of-fold RMSE across all validation slices.
    """
    # 1. Hyperparameter Search Space Setup
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    # 2. Structural Cross-Validation Boundary Extraction Loop
    trial_rmse_scores: List[float] = []

    for train_index, val_index in kf:
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Instantiates model with GPU parameters merged dynamically
        model = WellboreModelLGBM(params=params)
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)

        # 3. Out-Of-Fold Evaluation Performance Tracking
        y_pred: npt.NDArray[np.float64] = model.predict(X_val)
        
        # Centralize math inside validation module to enforce array cleaning protections
        rmse: float = evaluate_model(predictions=y_pred, actuals=y_val)
        trial_rmse_scores.append(rmse)

        # Clear active memory allocations per fold split execution loop
        del model, X_train, X_val, y_train, y_val, y_pred
        gc.collect()

    mean_rmse: float = float(np.mean(trial_rmse_scores))
    return mean_rmse


def main() -> None:
    """Runs the hyperparameter optimization study with pre-computed data caching."""
    print("Initializing High-Performance Feature Pipeline Caching Block...")
    
    # 1. Execute High-Performance IO & Feature Compute Once
    raw_df: pl.DataFrame = load_raw_data()
    train_df: pl.DataFrame = generate_features(raw_df)
    
    # Apply memory downcasting optimizations right after feature generation
    train_df = optimize_memory_usage(train_df)

    # 2. Case-Agnostic Target Column Extraction
    target_matches = [c for c in train_df.columns if c.lower() == "tvt"]
    if not target_matches:
        raise KeyError("CRITICAL ENFORCEMENT ERROR: Target log column 'TVT' or 'tvt' not found in dataset schema.")
    target_col: str = target_matches[0]

    # Explicitly compile the drop list to avoid target variants leaking into optimization trials
    ignore_list: List[str] = ["well_id", "id", "tvt_input", "tvt", "target"]
    cols_to_drop: List[str] = [
        col for col in train_df.columns 
        if col.lower() in ignore_list or col == target_col
    ]
    
    feature_df: pl.DataFrame = train_df.drop(cols_to_drop)
    feature_names: List[str] = feature_df.columns

    # 3. Extract Strict Typed Contiguous Arrays directly into optimal target types
    # Downcasting X to float32 at this step eliminates the overhead of converting
    # data types during every cross-validation step.
    X: npt.NDArray[np.float32] = feature_df.to_numpy().astype(np.float32)
    y: npt.NDArray[np.float64] = train_df[target_col].to_numpy().astype(np.float64)
    
    # Pre-calculate spatial split targets securely before clearing tracking frames
    kf = get_cv_splits(train_df, n_splits=5)

    # 4. Aggressive Intermediary Dataframe Collection Disposals
    del raw_df, train_df, feature_df
    gc.collect()

    print("Initializing Study Loop... (Running on NVIDIA GeForce RTX 5080 Laptop GPU)")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # 5. Use Lambda Context to Pass Pre-Computed Assets Straight to Worker Threads
    study.optimize(
        lambda trial: objective(trial, X, y, feature_names, kf), 
        n_trials=15, 
        show_progress_bar=True
    )

    print("\n================ Optimization Summary ================")
    print(f"Best Real Baseline Mean OOF RMSE: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("=======================================================")

    # Final cleanup tracking protection
    del X, y, kf
    gc.collect()


if __name__ == "__main__":
    main()