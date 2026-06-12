import os
import uuid
from typing import List

import mlflow.sklearn
import pandas as pd
import pandera as pa
from fastapi import BackgroundTasks, FastAPI, HTTPException
from mlflow import MlflowClient

from src.data.ingest import upload_payload_to_s3
from src.data.schemas import MachineFeaturesSchema, MachineInferencePayload

app = FastAPI()

# 1. Setup MLflow Connection
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI")
model_name = os.getenv("MODEL_NAME")
model_alias = os.getenv("MODEL_ALIAS")
# 1. Path to your model in S3 (Use the folder path from your MLflow UI)
MODEL_URI = f"models:/{model_name}@{model_alias}"
# PREPROCESSOR_URI = os.getenv("PREPROCESSOR_URI")
# MODEL_ALIAS = os.getenv("MODEL_ALIAS")
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
        model_data = client.get_model_version_by_alias(model_name, model_alias)
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
        "model_alias": model_alias,
        "mlflow_run_id": info["run_id"],
        "model_description": client.get_model_version(
            name=model_name, version=info["version"]
        ).description,
    }


@app.get("/")
def home():
    return {"status": "Model API is Running now"}


@app.post("/predict")
async def predict_maintenance(
    payload: List[MachineInferencePayload], background_tasks: BackgroundTasks
):
    prediction_id = str(uuid.uuid4())
    # timestamp = datetime.now(datetime.timezone.utc).isoformat() + "Z"
    # 1. Structural Validation (FastAPI + Pydantic step completed implicitly on entry)
    # Convert list of Pydantic models back to standard dictionaries maintaining
    #  JSON naming
    raw_records = [record.model_dump(by_alias=True) for record in payload]
    df = pd.DataFrame(raw_records)

    # 2. Value Range and Feature Integrity Validation (Pandera Step)
    try:
        validated_df = MachineFeaturesSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Data values failed operational validation bounds.",
                "failures": exc.failure_cases[
                    ["column", "check", "failure_case"]
                ].to_dict(orient="records"),
            },
        )

    # 3. Model Inference
    prediction = pipeline.predict(validated_df)

    # Convert numpy array / pandas series safely to native Python types
    if hasattr(prediction, "tolist"):
        serializable_prediction = prediction.tolist()
    else:
        serializable_prediction = list(prediction)

    # 4. Asynchronously log the inference payload and prediction to S3
    info = get_model_info()
    logging_document = {
        "prediction_id": prediction_id,
        # "timestamp": timestamp,
        "model_version": info["version"],
        "features": validated_df,
        "prediction": serializable_prediction,
    }

    # Schedule the S3 upload to run in the background after the response is sent
    background_tasks.add_task(upload_payload_to_s3, logging_document, prediction_id)

    return {
        "prediction_id": prediction_id,
        "prediction": serializable_prediction,
        "status": "success",
        "type": str(type(prediction)),
    }


if __name__ == "__main__":
    print("Starting ARISE Prediction API...")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
