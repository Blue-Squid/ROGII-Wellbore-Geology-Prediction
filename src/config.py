import os
import random
from pathlib import Path


def seed_everything(seed: int = 42) -> None:
    """Ensure global determinism across all numerical and machine learning operations.

    Args:
        seed (int): The random seed value to set globally.
    """
    """Seed everything to enforce determinism across different libraries.

    Args:
        seed (int): The random seed value.
    """
    import numpy as np
    import torch

    # Enforce standard Python environment determinism
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Instruct Polars' execution engine to use a fixed random seed
    os.environ["POLARS_RANDOM_SEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Fix CUDNN behavior for stable deep sequence training iterations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Define paths using cross-platform safe Path objects
DATA_RAW_PATH = Path(os.getenv("DATA_RAW_PATH", "data/raw"))
DATA_PROCESSED_PATH = Path(os.getenv("DATA_PROCESSED_PATH", "data/processed"))
DATA_EXTERNAL_PATH = Path(os.getenv("DATA_EXTERNAL_PATH", "data/external"))
DOCS_PATH = Path(os.getenv("DOCS_PATH", "docs"))