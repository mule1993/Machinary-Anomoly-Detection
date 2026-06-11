import pathlib

import pandas as pd
import pandera as pa
import pytest

import src.data.ingest


@pytest.mark.data
def test_pandera_valid_data_passes(
    sample_raw_data: pd.DataFrame, tmp_path: pathlib.Path
):
    """Verify that a standard clean dataset passes the Pandera gatekeeper."""
    # Create a temporary CSV file using pytest's built-in tmp_path fixture
    temp_csv = tmp_path / "clean_data.csv"
    sample_raw_data.to_csv(temp_csv, index=False)

    # Act & Assert: This should run smoothly and return a valid DataFrame
    try:
        validated_df = src.data.ingest.load_csv_from_s3_and_validate_training_data(
            str(temp_csv)
        )
        assert isinstance(validated_df, pd.DataFrame)
    except pa.errors.SchemaError:
        pytest.fail("Valid production dataset unexpectedly failed Pandera validation!")


@pytest.mark.data
def test_pandera_invalid_temperature_fails(
    sample_raw_data: pd.DataFrame, tmp_path: pathlib.Path
):
    """Verify that anomalous out-of-bounds telemetry gets caught instantly."""
    corrupt_df = sample_raw_data.copy()
    # Physics violation: Air temperature set to 999 Kelvin
    corrupt_df.loc[0, "Air temperature [K]"] = 999.0

    temp_csv = tmp_path / "corrupt_data.csv"
    corrupt_df.to_csv(temp_csv, index=False)

    # Act & Assert: Verify that Pandera raises a SchemaError
    with pytest.raises(pa.errors.SchemaError) as exc_info:
        src.data.ingest.load_csv_from_s3_and_validate_training_data(str(temp_csv))

    # Pandera errors explicitly detail which column failed validation
    assert "Air temperature [K]" in str(exc_info.value)
