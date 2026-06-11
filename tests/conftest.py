import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_raw_data():
    """realistic minimal dataframe matching the training data schema."""
    data = {
        "UDI": [1, 2, 3, 4, 5],
        "Product ID": ["M14860", "L47181", "E11234", "M14863", "L47185"],
        "Type": ["M", "L", "H", "M", "L"],
        "Air temperature [K]": [298.1, 298.2, 298.1, 298.5, 300.0],
        "Process temperature [K]": [308.6, 308.7, 308.5, 308.9, 310.2],
        "Rotational speed [rpm]": [1551, 1408, 1498, 1432, 2825],
        "Torque [Nm]": [42.8, 46.3, 49.4, 40.2, 10.0],
        "Tool wear [min]": [0, 3, 5, 205, 12],
        "TWF": [0, 0, 0, 1, 0],  # Tool Wear Failure
        "HDF": [0, 0, 0, 0, 1],  # Heat Dissipation Failure
        "PWF": [0, 0, 0, 0, 0],  # Power Failure
        "OSF": [0, 0, 0, 0, 0],  # Overstrain Failure
        "RNF": [0, 0, 0, 0, 0],  # Random Failure
    }
    return pd.DataFrame(data)
