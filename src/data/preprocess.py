from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def convert_kelvin_to_celsius(temp_k):
    """Convert temperature from Kelvin to Celsius."""
    return temp_k - 273.15


def calculate_power(X):
    """Calculate power in Watts from RPM and Torque."""
    if hasattr(X, "values"):
        X = X.values
    return (X[:, [0]] * X[:, [1]]) / 9.5488


def get_power_feature_names(transformer, input_features):
    """Get feature names for the power calculator."""
    return ["Power"]


temperature_converter = FunctionTransformer(
    convert_kelvin_to_celsius, feature_names_out="one-to-one"
)

power_calculator = FunctionTransformer(
    calculate_power, feature_names_out=get_power_feature_names
)


def build_preprocessor(X):
    """Create a scikit-learn preprocessing pipeline.

    Args:
        X (pandas.DataFrame): Input DataFrame with features.

    Returns:
        ColumnTransformer: A preprocessing transformer for numeric and categorical data.
    """
    X = X.drop(
        columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"],
        errors="ignore",
    )
    numeric_features_temprature = ["Air temperature [K]", "Process temperature [K]"]
    numeric_features_power = ["Rotational speed [rpm]", "Torque [Nm]"]
    numeric_features = ["Tool wear [min]"]
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    temperature_numeric_transformer = Pipeline(
        [
            ("temp_converter", temperature_converter),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    power_numeric_transformer = Pipeline(
        [
            ("combine", power_calculator),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("temp", temperature_numeric_transformer, numeric_features_temprature),
            ("power", power_numeric_transformer, numeric_features_power),
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor


"""if __name__ == "__main__":
    df = load_csv_from_s3()
    print("Sample data:")
    print(df.head())
    # print(df.info())
    preprocessor = build_preprocessor(df)
    X_train = preprocessor.fit_transform(df)
    print("Preprocessor created successfully.")
    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(X_train, columns=feature_names)
    # find columns that came from the categorical pipe and convert them
    # cat_cols = [col for col in df_processed.columns if col.startswith("cat__")]
    # df_processed[cat_cols] = df_processed[cat_cols].astype(bool)
    print("\nProcessed DataFrame preview:")
    print(df_processed.head(2))
    print(df_processed.info())"""

"""
# Load and preprocess data
df = load_csv_from_s3()
target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
preprocessor = build_preprocessor(X_train)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)"""
