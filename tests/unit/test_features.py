'''
# tests/unit/test_features.py
import pytest

# Assuming your source logic is structured like this:
# from src.features import missing_imputer


def missing_imputer(df, column, strategy="mean"):
    """Example production code function to be tested."""
    fill_value = df[column].mean() if strategy == "mean" else 0
    df_copy = df.copy()
    df_copy[column] = df_copy[column].fillna(fill_value)
    return df_copy


@pytest.mark.unit
def test_missing_imputer_calculates_mean(sample_raw_data):
    # Arrange: Use the fixture data
    target_col = "feature_numeric_1"

    # Act: Run the processing logic
    processed_df = missing_imputer(sample_raw_data, column=target_col, strategy="mean")

    # Assert: Verify no missing values remain and the mean calculation was correct
    assert processed_df[target_col].isnull().sum() == 0

    # Expected mean of [10.5, 20.0, 40.2, 50.0] is 30.175
    assert processed_df.loc[2, target_col] == pytest.approx(30.175, rel=1e-3)


@pytest.mark.unit
def test_calculate_temperature_differential_math(sample_raw_data):
    # Act: Pass your sample fixture through the production function
    processed_df = calculate_temperature_differential(sample_raw_data)

    # Assert: Verify the new column exists
    assert "Temperature_Delta_K" in processed_df.columns

    # Verify the math on the first row: 308.6K - 298.1K = 10.5K
    assert processed_df.loc[0, "Temperature_Delta_K"] == pytest.approx(10.5)
'''
