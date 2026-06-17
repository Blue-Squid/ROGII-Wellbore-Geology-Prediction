import concurrent.futures
from pathlib import Path
import polars as pl


def _load_single_well(file_path: Path) -> pl.DataFrame:
    """Reads a single well file, forces numeric schema types, and injects well_id.

    Args:
        file_path (Path): The path to the target wellbore CSV log file.

    Returns:
        pl.DataFrame: A collected Polars DataFrame with guaranteed numeric typing.
    """
    well_id: str = file_path.name.split("__")[0]

    # Explicitly enforce core schema structure to override erratic CSV text artifacts
    core_schema = {
        "MD": pl.Float64,
        "X": pl.Float64,
        "Y": pl.Float64,
        "Z": pl.Float64,
        "GR": pl.Float64,
        "TVT": pl.Float32,
        "TVT_input": pl.Float32
    }

    try:
        # Scan lazily but override inference bounds completely
        q = pl.scan_csv(
            file_path,
            null_values=["NaN", "nan", "null", "None", "NA", "", "-", "-999.25", "-999", "9999"],
            infer_schema_length=20000, # Deep scan to accurately detect column layouts
        )

        # FIX PERFORMANCE WARNING: Use collect_schema() to avoid resolving full frames
        current_schema = q.collect_schema()
        existing_columns = current_schema.names()

        cast_exprs = []
        
        # Loop through all columns present in the file dynamically
        for col in existing_columns:
            if col in core_schema:
                # Force core keys to their exact target type representations
                cast_exprs.append(pl.col(col).cast(core_schema[col], strict=False))
            elif col != "well_id":
                # STRATEGIC HARDENING: If an unknown log column (like ANCC) is parsed 
                # as String due to corrupt text rows, force it to Float64 so it can 
                # concatenate with other clean numeric log tables.
                if current_schema[col] == pl.String:
                    cast_exprs.append(pl.col(col).cast(pl.Float64, strict=False))

        return (
            q.with_columns(cast_exprs)
            .with_columns(pl.lit(well_id).alias("well_id"))
            .collect()
        )
        
    except Exception as e:
        print(f"⚠️ Warning: Structural parse failure on file {file_path.name}. Falling back to broad string parse. Error: {e}")
        df = pl.read_csv(file_path, infer_schema_length=0)
        
        for col, dtype in core_schema.items():
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(dtype, strict=False))
        return df.with_columns(pl.lit(well_id).alias("well_id"))

def load_raw_data(directory_name: str = "train") -> pl.DataFrame:
    """Loads raw data from the specified directory and aggregates into a monolithic frame.

    Args:
        directory_name (str): The target subdirectory to parse ("train" or "test").

    Returns:
        pl.DataFrame: A unified stacked collection of clean well profiles.
    """
    project_root: Path = Path(__file__).resolve().parent.parent
    base_path: Path = project_root / "data" / "raw"

    target_dir: Path = base_path / directory_name
    if not target_dir.exists():
        nested_matches: list[Path] = list(base_path.rglob(directory_name))
        if not nested_matches:
            raise FileNotFoundError(
                f"Could not locate a folder named '{directory_name}' within path: {base_path}"
            )
        target_dir = nested_matches[0]

    horizontal_files: list[Path] = list(target_dir.glob("*__horizontal_well.csv"))
    if not horizontal_files:
        raise FileNotFoundError(
            f"No '*__horizontal_well.csv' files found inside {target_dir.absolute()}"
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        well_dfs: list[pl.DataFrame] = list(
            executor.map(_load_single_well, horizontal_files)
        )

    # Now we use 'vertical' instead of 'vertical_relaxed'. 
    # Because we cleaned types at loading, this guarantees strict schema alignment.
    full_df: pl.DataFrame = pl.concat(well_dfs, how="vertical")
    return full_df


def optimize_memory_usage(df: pl.DataFrame) -> pl.DataFrame:
    """Optimizes memory footprint of the Polars DataFrame by safely downcasting types."""
    schema_updates: list[pl.Expr] = []

    # Keep structural telemetry keys at full precision, downcast logging metrics safely
    telemetry_safeguards = ["x", "y", "z", "lat", "lon", "easting", "northing", "md", "tvd"]

    for col, dtype in df.schema.items():
        if dtype == pl.Float64:
            if not any(coord in col.lower() for coord in telemetry_safeguards):
                schema_updates.append(pl.col(col).cast(pl.Float32))
        elif dtype == pl.Int64:
            # Defensively safeguard row matching keys and IDs from wrapping errors
            if not any(idx in col.lower() for idx in ["id", "index", "row_index"]):
                schema_updates.append(pl.col(col).cast(pl.Int32))

    return df.with_columns(schema_updates)