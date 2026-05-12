import pandas as pd
from ingest import load_csv_from_s3, upload_csv_to_s3

# Load data from S3
df = load_csv_from_s3()


def kelvin_to_celsius(df):
    """
    Convert air and process temperature columns from Kelvin to Celsius.

    Args:
        df: DataFrame with temperature columns

    Returns:
        DataFrame with converted temperature values
    """
    if "Air temperature [K]" in df.columns:
        df["Air temperature [K]"] = df["Air temperature [K]"] - 273.15
        df.rename(columns={"Air temperature [K]": "Air temperature [C]"}, inplace=True)
    if "Process temperature [K]" in df.columns:
        df["Process temperature [K]"] = df["Process temperature [K]"] - 273.15
        df.rename(
            columns={"Process temperature [K]": "Process temperature [C]"}, inplace=True
        )
    return df


def drop_rows_with_negatives(df):
    """
    Drop rows that contain negative numbers in numerical columns.

    Args:
        df: DataFrame

    Returns:
        DataFrame with rows containing negatives dropped
    """
    numerical_cols = df.select_dtypes(include="number").columns
    mask = (df[numerical_cols] < 0).any(axis=1)
    return df[~mask]


def speed_torque_to_power(df):
    """
    Convert Rotational speed [rpm]	Torque [Nm] to Power [KW]
    Args:
        df: DataFrame with Rotational speed [rpm]	Torque [Nm] columns

    Returns:
        DataFrame with converted Power values
    """
    if "Rotational speed [rpm]" in df.columns and "Torque [Nm]" in df.columns:
        df.insert(
            5, "Power [KW]", df["Rotational speed [rpm]"] * df["Torque [Nm]"] / 9550
        )
        df.drop(columns=["Rotational speed [rpm]", "Torque [Nm]"], inplace=True)
    return df


def drop_unwanted_columns(df):
    """
    Drop columns that are not needed for modeling.

    Args:
        df: DataFrame with columns to drop ('Product ID', 'Failure Type')

    Returns:
        DataFrame with unwanted columns dropped
    """
    columns_to_drop = ["Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    return df.drop(columns=columns_to_drop, errors="ignore")


def one_hot_encode_types(df):
    """
    One-hot encode the 'Type' column.

    Args:
        df: DataFrame with 'Type' column
    Returns:
        DataFrame with one-hot encoded 'Type' column
    """
    if "Type" in df.columns:
        # 1. Get the integer position of the 'Type' column
        idx = df.columns.get_loc("Type")
        # 2. Create the dummies
        type_dummies = pd.get_dummies(df["Type"], prefix="Type", dtype=int)
        # 3. Slice the dataframe and sandwich the dummies in the middle
        df = pd.concat([df.iloc[:, :idx], type_dummies, df.iloc[:, idx + 1 :]], axis=1)
    return df


def preprocess_data():
    """
    Preprocess the data by applying all transformations.

    Args:
        df: Raw DataFrame

    Returns:
        Preprocessed DataFrame
    """
    df = load_csv_from_s3()
    df = kelvin_to_celsius(df)
    df = drop_rows_with_negatives(df)
    df = speed_torque_to_power(df)
    df = drop_unwanted_columns(df)
    df = one_hot_encode_types(df)
    return df


if __name__ == "__main__":
    print(df.head())
    celsius_df = kelvin_to_celsius(df)
    print(celsius_df.head())
    # print("\nAfter dropping rows with negative values:\n")
    positive_df = drop_rows_with_negatives(celsius_df)
    print(positive_df.head())
    power_df = speed_torque_to_power(positive_df)
    print(power_df.head())
    final_df = drop_unwanted_columns(power_df)
    print(final_df.head())
    encoded_df = one_hot_encode_types(final_df)
    print(encoded_df.head())
    print(f"Final DataFrame shape: {encoded_df.shape}")
    upload_csv_to_s3(encoded_df)
