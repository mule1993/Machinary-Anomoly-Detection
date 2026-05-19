import os

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from mlflow import MlflowClient

app = FastAPI()

# 1. Setup MLflow Connection
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://172.17.0.1:5000")
model_name = "failure-prediction-model"

# 1. Path to your model in S3 (Use the folder path from your MLflow UI)
MODEL_URI = os.getenv("MODEL_URI")
MODEL_ALIAS = os.getenv("MODEL_ALIAS")
# MODEL_URI = "s3://machinery-mlops-muluneh-2026/mlflow-artifacts/models/m-7c13b7532ad74f6f95d776a105505dda/artifacts/"

mlflow.set_tracking_uri(MLFLOW_URI)
client = MlflowClient()
# 2. Load the model globally so it's fast
print(f"Loading model from: {MODEL_URI}")
model = mlflow.sklearn.load_model(MODEL_URI)


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
    }


@app.get("/")
def home():
    return {"status": "Model API is Running"}


@app.post("/predict")
def predict(data: dict):
    # Expecting JSON like: {"features": [[298.1, 308.6, 1500, 40.5, 0]]}
    df = pd.DataFrame(data["features"])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
