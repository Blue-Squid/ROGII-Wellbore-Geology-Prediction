import pytest
import polars as pl
from src.validation import get_cv_splits

def test_get_cv_splits():
    # Create a sample dataset
    data = {
        "well_id": ["A", "A", "B", "B", "C", "C"],
        "TVT": [10, 20, 30, 40, 50, 60]
    }
    df = pl.DataFrame(data)

    # Generate cross-validation splits
    cv_splits = get_cv_splits(df, n_splits=2)

    # Check that the number of splits is correct
    assert len(cv_splits) == 2

    # Check that each split contains valid indices
    for train_idx, val_idx in cv_splits:
        assert set(train_idx).union(set(val_idx)) == set(range(len(df)))
        assert len(set(train_idx).intersection(set(val_idx))) == 0
