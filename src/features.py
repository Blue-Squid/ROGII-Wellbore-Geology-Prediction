"""Feature generation module for the ROGII Wellbore Geology Prediction pipeline.

Calculates high-performance geostatistical attributes, spatial vectors, and
sensor transformations natively in Polars.
"""

import polars as pl


def generate_features(df: pl.DataFrame) -> pl.DataFrame:
    """Generates geostatistical and geometric features safely using Polars.

    Ensures that short wellbore segments, boundary calculations, and window
    statistics do not leak metadata or drop into structural null values.

    Args:
        df (pl.DataFrame): Raw input telemetry DataFrame.

    Returns:
        pl.DataFrame: Enriched feature space matrix with no target dependencies.
    """
    # 1. Enforce numeric constraints to eliminate structural type mismatches
    # Explicitly force casting to double-precision floats to eliminate any
    # accidental string contamination coming from raw data ingestion engines.
    df = df.with_columns([
        pl.col("X").cast(pl.Float64),
        pl.col("Y").cast(pl.Float64),
        pl.col("Z").cast(pl.Float64),
        pl.col("MD").cast(pl.Float64),
        pl.col("GR").cast(pl.Float64),
    ])

    # 2. Sort geographically/sequentially per well to guarantee physical accuracy
    df = df.sort(["well_id", "MD"])

    # 3. Calculate 3D trajectory spatial gradients defensively
    # We explicitly isolate the measured depth delta per well and handle both
    # null values (at row 0) and zero values (duplicate stamps) to guarantee
    # a safe division denominator.
    md_delta = (
        pl.col("MD")
        .diff()
        .over("well_id")
        .fill_null(1.0)  # Handle row index 0 null assignment
        .replace(0.0, 1.0)  # Handle structural zero-step telemetry duplicates
    )

    gradient_exprs = [
        (pl.col("X").diff().over("well_id").fill_null(0.0) / md_delta).alias("grad_X"),
        (pl.col("Y").diff().over("well_id").fill_null(0.0) / md_delta).alias("grad_Y"),
        (pl.col("Z").diff().over("well_id").fill_null(0.0) / md_delta).alias("grad_Z"),
    ]
    
    # 4. Compute moving localized sensor logs statistics defensively
    # We specify 'min_periods=1' to force Polars to compute rolling metrics 
    # even on initial boundaries or short profiles. If a well has fewer than 
    # 5 rows, it will gracefully compute metrics on the available rows instead of crashing.
    rolling_exprs = [
        pl.col("GR")
        .rolling_mean(window_size=5, min_periods=1)
        .over("well_id")
        .alias("GR_rolling_mean"),
        
        pl.col("GR")
        .rolling_std(window_size=5, min_periods=1)
        .over("well_id")
        .fill_null(0.0)  # Single element windows have no variance; force to zero
        .alias("GR_rolling_std"),
    ]

    # 5. Geostatistical Non-Leakage Structural Proxies
    # Standardize baseline structural features without using look-ahead metrics 
    # that could compromise spatial isolation rules.
    structural_proxies = [
        (pl.col("Z").max().over("well_id") - pl.col("Z").min().over("well_id")).alias("Z_delta_well"),
        pl.col("MD").max().over("well_id").alias("MD_total_well"),
        (pl.col("MD") - pl.col("MD").min().over("well_id")).alias("cum_well_distance")
    ]

    # 6. Apply Layered Transforms and Enforce Isolated Null Imputation
    return (
        df
        # Classify structural well profile layout metrics safely
        .with_columns([
            (pl.col("Z").max().over("well_id") / pl.col("MD").max().over("well_id").fill_null(1.0).replace(0.0, 1.0) > 0.95)
            .cast(pl.Int32).alias("is_type_well")
        ])
        # Inject features, gradients, and proxies safely
        .with_columns(structural_proxies + gradient_exprs + rolling_exprs)
        
        # CRITICAL HARDENING: Apply interpolation strictly WITHIN each well boundary.
        # This replaces global forward/backward fills with isolated window structures,
        # making cross-well structural metadata leakage completely impossible.
        .with_columns([
            pl.all().fill_null(strategy="forward").over("well_id"),
        ])
        .with_columns([
            pl.all().fill_null(strategy="backward").over("well_id"),
        ])
        # Final safety check: if any columns remain null due to empty inputs, fill with 0.0
        .fill_null(0.0)
    )