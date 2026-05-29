# tests/conftest.py
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_raw_data():
    """Provides a deterministic, minimal dataframe representing raw input data."""
    np.random.seed(42)
    data = {
        "timestamp": pd.date_range(start="2026-01-01", periods=5, freq="H"),
        "feature_numeric_1": [10.5, 20.0, np.nan, 40.2, 50.0],
        "feature_numeric_2": np.random.uniform(100, 200, 5),
        "target_label": [0, 1, 0, 0, 1],
    }
    return pd.DataFrame(data)
