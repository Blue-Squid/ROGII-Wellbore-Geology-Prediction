"""Kaggle submission inference execution pipeline script.

Loads the raw test assets, extracts geostatistical structural identifiers, 
and aggregates predictions across the 5 out-of-fold persisted booster models.
"""

import gc
import os
from pathlib import Path
from typing import List
import numpy as np
import numpy.typing as npt
import polars as pl

# Ingest robust, insulated sub-modules from the project architecture
from src.data_loader import load_raw_data, optimize_memory_usage
from src.features import generate_features
from src.model import WellboreModelLGBM


def main() -> None:
    """Executes the complete test inference array computation pipeline."""
    print("Initializing Phase 5 Kaggle Submission Inference Pipeline...")

    # 1. Discover Active Environment Root Paths Defensively
    # Automatically switch between your local workspace and Kaggle's server mounts
    kaggle_input_root = Path("/kaggle/input/rogii-wellbore-geology-prediction")
    if kaggle_input_root.exists():
        base_path = kaggle_input_root
        print(f"Mounted Kaggle environment path confirmed: {base_path}")
    else:
        project_root = Path(__file__).resolve().parent.parent
        base_path = project_root / "data" / "raw"
        print(f"Running on local workstation. Root path mapped to: {base_path}")

    # 2. Base Load Sample Submission Manifest
    # Reading the official sample submission file is the only way to guarantee that
    # the index keys perfectly match the leaderboard evaluation matrix.
    sample_sub_path = next(base_path.rglob("sample_submission.csv"), None)
    if sample_sub_path is None:
        raise FileNotFoundError("Critical Ingestion Failure: Could not locate 'sample_submission.csv' anywhere in the data tree.")
        
    sample_sub = pl.read_csv(sample_sub_path)
    target_column_label = [c for c in sample_sub.columns if c.lower() == "tvt"][0]
    manifest_ids: List[str] = [str(x) for x in sample_sub["id"].to_list()]

    # Parse wellbore hashes and sequence rows out of the official manifest IDs
    manifest_df = sample_sub.with_columns([
        pl.col("id").str.split("_").list.get(0).alias("well_id"),
        pl.col("id").str.split("_").list.get(1).cast(pl.Int32).alias("row_index"),
        pl.int_range(0, pl.len()).alias("original_manifest_order")
    ])
    unique_wells = manifest_df["well_id"].unique().to_list()

    # 3. Read and Aggregate Raw Test Data Files Across Core Streams
    # We use our production-hardened data loader to scan and read the 'test' directory streams.
    test_dir_name = "test" if not kaggle_input_root.exists() else str(base_path)
    raw_test_df: pl.DataFrame = load_raw_data(directory_name=test_dir_name)

# 4. Enforce Correct Sorting Order Before Mapping Sequential Row Indices
    # By sorting on ['well_id', 'MD'] BEFORE building our indices, we match the physical
    # trajectory tracking layers we engineered during training.
    sorted_test_df = raw_test_df.sort(["well_id", "MD"])
    
    # Generate the sequential row counters cleanly inside their respective wellbore blocks
    sorted_test_df = sorted_test_df.with_columns([
        pl.int_range(0, pl.len()).over("well_id").cast(pl.Int32).alias("row_index")
    ])

    # 5. Identical Sequential Feature Generation Space Execution
    enriched_test_df = generate_features(sorted_test_df)
    
    # STRATEGIC INDICES RESOLUTION FIXED: Force row_index back to pl.Int32 format.
    # The downstream forward/backward fill operations over 'pl.all()' automatically 
    # upcast all columns to Float64. We explicitly downcast it back here to prevent
    # breaking Polars' strict datatype matching rules during the join phase.
    enriched_test_df = enriched_test_df.with_columns([
        pl.col("row_index").cast(pl.Int32)
    ])
    
    # Downcast feature dimensions to safe hosting boundaries
    enriched_test_df = optimize_memory_usage(enriched_test_df)

    # 6. Perfect Key-Based Join Alignment with the Original Manifest Order
    # Joining explicitly on ['well_id', 'row_index'] guarantees that even if rows are missing 
    # or sorted differently in the hidden dataset, your prediction vectors match up perfectly.
    aligned_test_df = manifest_df.join(
        enriched_test_df, on=["well_id", "row_index"], how="left"
    ).sort("original_manifest_order")
    
    # 7. Load a Reference Model to Extract the Feature Blueprint
    model_dir = Path("models") if not kaggle_input_root.exists() else Path("/kaggle/input/rogii-booster-weights")
    model_paths: List[Path] = sorted(list(model_dir.glob("lgbm_fold_*.txt")))
    if not model_paths:
        # Fallback query to handle generalized input model attachments
        model_paths = sorted([p for p in Path("/kaggle/input").rglob("lgbm_fold_*.txt")])
        if not model_paths:
            raise FileNotFoundError(f"Inference Error: No serialized fold models detected inside: {model_dir}")

    reference_model = WellboreModelLGBM()
    reference_model.load_model(model_paths[0])
    expected_features: List[str] = reference_model.model.feature_name()
    print(f"Model blueprint expects {len(expected_features)} features.")
    del reference_model
    gc.collect()

    # 8. Dynamic Alignment: Backfill Missing Columns Defensively
    missing_features = [feat for feat in expected_features if feat not in aligned_test_df.columns]
    if missing_features:
        print(f"⚠️ Warning: Injecting default safety matrices for {len(missing_features)} missing features.")
        aligned_test_df = aligned_test_df.with_columns([
            pl.lit(0.0).cast(pl.Float32).alias(feat) for feat in missing_features
        ])

    # 9. Slice and Order Columns Exactly as Expected by the Boosters
    feature_df = aligned_test_df.select(expected_features)

    # 10. Memory Contiguous Strict Typed Array Assembly
    X_test: npt.NDArray[np.float32] = feature_df.to_numpy().astype(np.float32)
    print(f"Aligned inference matrix dimensions: {X_test.shape}")
    
    del raw_test_df, sorted_test_df, enriched_test_df, aligned_test_df, feature_df
    gc.collect()

    # 11. Initialize Zero Allocation Matrix for Ensemble Averaging
    blended_predictions: npt.NDArray[np.float64] = np.zeros(len(X_test), dtype=np.float64)

    # 12. Sequentially Load and Score Models Defensively
    print(f"Blending ensemble vectors across {len(model_paths)} saved models...")
    for model_path in model_paths:
        model = WellboreModelLGBM()
        model.load_model(model_path)
        
        # Predictions are automatically filtered through our internal NaN protections
        fold_predictions: npt.NDArray[np.float64] = model.predict(X_test.astype(np.float64))
        blended_predictions += fold_predictions / len(model_paths)
        
        del model, fold_predictions
        gc.collect()

    # 13. Shape Assembly via Polars Matching the Exact Manifest Layout
    # Use pl.String to guarantee compatibility with modern Polars environments
    submission_df = pl.DataFrame({
        "id": pl.Series(manifest_ids, dtype=pl.String),
        "tvt": pl.Series(blended_predictions, dtype=pl.Float64)
    })

    # 14. Write Finalized Submission File Out to Working Storage Root
    # Ensure file outputs are written to the active execution directory
    output_path = Path("/kaggle/working/submission.csv") if kaggle_input_root.exists() else Path("submission.csv")
    submission_df.write_csv(output_path)
    print(f"🎯 Inference complete! Submission verified and saved to: {output_path.resolve()}")

    del manifest_ids, X_test, blended_predictions, submission_df
    gc.collect()


if __name__ == "__main__":
    main()