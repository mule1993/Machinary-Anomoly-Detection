import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient

app = FastAPI()

# 1. Setup MLflow Connection
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
model_name = os.getenv("MODEL_NAME")
model_alias = os.getenv("MODEL_ALIAS")
# 1. Path to your model in S3 (Use the folder path from your MLflow UI)
MODEL_URI = f"models:/{model_name}@{model_alias}"
# PREPROCESSOR_URI = os.getenv("PREPROCESSOR_URI")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
# MODEL_URI = "s3://machinery-mlops-muluneh-2026/mlflow-artifacts/models/m-7c13b7532ad74f6f95d776a105505dda/artifacts/"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
# 2. Load the model globally so it's fast
print(f"Loading model from: {MODEL_URI}")
pipeline = mlflow.pyfunc.load_model(MODEL_URI)
# preprocessor = mlflow.sklearn.load_model(PREPROCESSOR_URI)


def get_model_info():
    try:
        # Resolve the alias to get the specific version metadata
        model_data = client.get_model_version_by_alias(model_name, MODEL_ALIAS)
        return {
            "version": model_data.version,
            "run_id": model_data.run_id,
            "status": "READY",
        }
    except Exception as e:
        return {"version": "unknown", "status": "error", "message": str(e)}


@app.get("/status")
def status():
    info = get_model_info()
    return {
        "service": "ARISE Prediction API",
        "model_name": model_name,
        "model_version": info["version"],  # <--- This shows '1', '2', etc.
        "model_alias": MODEL_ALIAS,
        "mlflow_run_id": info["run_id"],
        "model_description": client.get_model_version(
            name=model_name, version=info["version"]
        ).description,
    }


@app.get("/")
def home():
    return {"status": "Model API is Running now"}


@app.post("/predict")
def predict(data: dict):
    # 1. Convert incoming JSON dictionary to a baseline DataFrame
    raw_df = pd.DataFrame([data])

    # 2. Extract the exact column sequence your ColumnTransformer requires
    # (Scikit-Learn stores this internal mapping inside 'feature_names_in_')
    expected_columns = pipeline._model_impl.python_model.expected_features

    # 3. Re-index your dataframe to match that exact footprint.
    # Any missing columns (like your training target label) will
    # be safely filled with 0 or NaN
    final_df = raw_df.reindex(columns=expected_columns, fill_value=0)
    # preprocessed_df = preprocessor.transform(final_df)
    prediction = pipeline.predict(final_df)
    return {
        "prediction": prediction.tolist()[0],
        "status": "success",
        "type": str(type(prediction)),
    }


if __name__ == "__main__":
    print("Starting ARISE Prediction API...")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
