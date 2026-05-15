import os  # noqa: I001

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

from src.models.model_classes import DummyFailureModel  # noqa: F401

app = FastAPI()

# 1. Path to your model in S3 (Use the folder path from your MLflow UI)
MODEL_URI = os.getenv("MODEL_URI")

# 2. Load the model globally so it's fast
print(f"Loading model from: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/")
def home():
    return {"status": "Model API is Running"}


@app.post("/predict")
def predict(data: dict):
    # Expecting JSON like: {"features": [[298.1, 308.6, 1500, 40.5, 0]]}
    df = pd.DataFrame(data["features"])
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
