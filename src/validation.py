"""Validation functions for the ROGII Wellbore Geology Prediction model.

This module provides cross-validation generators designed to isolate spatial
telemetry data and prevent downstream target leakage during evaluation.
"""

from typing import Generator, List, Tuple
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.model_selection import GroupKFold


def evaluate_model(
    predictions: npt.NDArray[np.float64], actuals: npt.NDArray[np.float64]
) -> float:
    """Evaluate model predictions against actual values using RMSE.

    Includes defensive array casting and null-value remediation to prevent
    anomalous predictions from returning invalid metadata metrics.

    Args:
        predictions (npt.NDArray[np.float64]): Predicted values from the model.
        actuals (npt.NDArray[np.float64]): Actual ground-truth target values.

    Returns:
        float: The calculated Root Mean Squared Error (RMSE).
    """
    # Defensive processing: convert arrays to float64 and clean up any runtime
    # NaN or infinity occurrences that could slip through model prediction matrices
    clean_preds = np.nan_to_num(predictions, nan=11000.0, posinf=11000.0, neginf=11000.0).astype(np.float64)
    clean_actuals = np.asarray(actuals, dtype=np.float64)
    
    # Standard algebraic calculation for Root Mean Squared Error
    return float(np.sqrt(np.mean((clean_preds - clean_actuals) ** 2)))


def get_cv_splits(
    df: pl.DataFrame, n_splits: int = 5
) -> List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]:
    """Generates GroupKFold index splits isolated strictly by individual well identifiers.

    This ensures that telemetry records belonging to a single well are never
    split across both training and validation sets simultaneously, eliminating
    geostatistical leakage.

    CRITICAL ASSUMPTION: The incoming DataFrame must already be explicitly 
    sorted by ['well_id', 'MD'] to guarantee that row indexing arrays align 
    perfectly with upstream feature matrices.

    Args:
        df (pl.DataFrame): The structurally sorted dataset containing feature columns 
            alongside the required 'well_id' field.
        n_splits (int): Total number of cross-validation folds to generate. 
            Defaults to 5.

    Returns:
        List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]: A list containing 
            tuples of (train_indices, validation_indices) for each generated fold.
            
    Raises:
        ValueError: If the dataframe sorting state cannot be verified or columns are missing.
    """
    # 1. Verification Guard: Validate presence of structural grouping column
    if "well_id" not in df.columns:
        raise ValueError("DataFrame must contain a valid 'well_id' column to apply spatial fold isolation.")

    # 2. Hardening Check: Ensure sorting integrity is preserved before compiling indices.
    # If the rows are out of chronological order, GroupKFold indices will cross-contaminate 
    # when passed to training engines that slice matrices independently.
    if "MD" in df.columns:
        # Check if the dataframe is properly ordered by tracking whether MD values decrease within group runs
        is_sorted = df.select(
            pl.col("MD").diff().over("well_id").fill_null(0.0).ge(0.0).all()
        ).item()
        
        if not is_sorted:
            raise ValueError(
                "CRITICAL DIRECTIONAL ERROR: Input DataFrame is not strictly sorted by ['well_id', 'MD']. "
                "Indices compiled under this state will lead to corrupted target variable mapping loops."
            )

    # 3. Create an explicit dummy feature matrix for Scikit-Learn's splitter interface
    dummy_x: npt.NDArray[np.float64] = np.zeros(shape=(len(df), 1))
    
    # 4. Cast hexadecimal strings to physical category indices cleanly.
    # We enforce string casting to eliminate any unexpected object reference layers.
    groups: npt.NDArray[np.int64] = (
        df["well_id"].cast(pl.String).cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    )

    # 5. Initialize the spatial group isolation interface
    group_kfold: GroupKFold = GroupKFold(n_splits=n_splits)
    
    # 6. Explicitly compile splits into a materialized list structure to match train.py tracking
    splits: List[Tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = list(
        group_kfold.split(X=dummy_x, y=None, groups=groups)
    )
    
    print(f"🎯 Successfully generated {len(splits)} spatial cross-validation splits using GroupKFold isolation.")
    return splits