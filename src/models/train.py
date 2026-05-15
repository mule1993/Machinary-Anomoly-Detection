import logging
import os

import mlflow
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3
from src.models.model_classes import DummyFailureModel

# ================= CONFIGURATION =================
# BUCKET_NAME = os.environ["BUCKET_NAME"]
REGION = os.environ["AWS_REGION"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
ARTIFACT_PATH = os.environ["ARTIFACT_PATH"]
# =================================================

# Load and preprocess data
df = load_csv_from_s3()
target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =================================================


# Silence the specific sklearn flavor logger warning
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

# requirements.txt dependencies warning workaround for MLflow's conda environment
conda_env = {
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.10",
        "scikit-learn=1.8.0",
        {"pip": ["mlflow", "boto3", "s3fs"]},
    ],
    "name": "mlflow_env",
}


# =================================================
try:
    mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_PATH)
except Exception:
    pass

mlflow.set_experiment(EXPERIMENT_NAME)


# 4. The Run
with mlflow.start_run():
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")

    # Log tags/params
    mlflow.set_tag("model_type", "dummy")
    mlflow.log_param("data_source", "manual_test")

    # Initialize model
    model = DummyFailureModel()

    # log model to S3
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model_output",
        serialization_format="pickle",
        conda_env=conda_env,
    )
    print("✅ Success! Model logged to S3.")


# =================================================
# Save the model locally
# BASE_DIR = Path(__file__).resolve().parent.parent.parent
# MODEL_DIR = BASE_DIR / "models"
# model_path = MODEL_DIR / "DummyFailureModel.joblib"
# joblib.dump(model, model_path)

# =================================================
# Predict on the test set
y_pred = model.predict(X_test)

# Log evaluation metrics to MLflow
with mlflow.start_run(run_name="Evaluation_Step", nested=True):
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

# Evaluate the model
print(classification_report(y_test, y_pred, zero_division=0))
print(
    f"Precision: {
        precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
)
print(f"Recall: {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

# =================================================
