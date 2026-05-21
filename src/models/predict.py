import os
import sys

import mlflow

from src.data.ingest import load_csv_from_s3
from src.data.preprocess import build_preprocessor


def load_production_model(model_uri: str = None):
    """
    Loads the MLflow pyfunc model from the specified URI.
    Defaults to the environment variable if not explicitly passed.
    """
    model_uri = os.getenv("MODEL_URI")

    try:
        # MLflow handles downloading artifacts from S3 and resolving the wrapper class
        model = mlflow.pyfunc.load_model(model_uri)
        print("[+] Model loaded successfully.")
        return model
    except Exception as e:
        print(f"[-] Failed to load model: {e}")
        sys.exit(1)


def make_prediction(model, input_data):
    """
    Passes data through the standardized pyfunc interface.
    """
    print(f"[*] Running inference on {len(input_data)} samples...")
    predictions = model.predict(input_data)

    # Convert predictions to a clean DataFrame output
    output_df = input_data.copy()
    output_df["prediction"] = predictions
    return output_df


if __name__ == "__main__":
    # 1. Load data
    ml_model = load_production_model(os.getenv("MODEL_URI"))
    df_input = load_csv_from_s3(bucket_key=os.getenv("TEST_DATA_PATH"))

    print("df_input before preprocessing:")
    print(df_input)
    preprocessor = build_preprocessor(df_input)
    print("Preprocessor built successfully.")
    df_input = preprocessor.fit_transform(df_input)
    print("Preprocessor applied successfully.")
    # df_input = preprocessor.transform(X_test)
    print("df_input after preprocessing:")
    print(df_input)

    # 3. Generate predictions
# df_results = make_prediction(ml_model, df_input)

# 4. Save results
# df_results.to_csv("data/results.csv", index=False)
# print("[+] Inference complete! Results saved to: data/results.csv")
