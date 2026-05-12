from pathlib import Path
from sys import meta_path

import joblib
import mlflow
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load data from S3
df = load_csv_from_s3()

# Prepare data for training
target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train a dummy model
class DummyFailureModel:
    """A model that always predicts 0 (No Failure)."""

    def predict(self, X):
        return [0] * len(X)


# Configure MLflow to use a local database
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
# AWS: Amazon RDS (Relational Database Service) running PostgreSQL or MySQL.
# Pmlflow.set_tracking_uri("http://internal-mlflow-load-balancer.aws.com:5000")
mlflow.set_experiment("Second_Dummy_Model_Experiment")

# Save the model and log it with MLflow
with mlflow.start_run():
    mlflow.set_tag("model_type", "dummy")
    mlflow.log_param("data_source", meta_path)

    model = DummyFailureModel()
    model_path = MODEL_DIR / "DummyFailureModel.joblib"
    # Save the model artifact to S3 and log it with MLflow
    mlflow.log_artifact(str(model_path))
    # Save the model package using MLflow's sklearn flavor on S3
    mlflow.sklearn.log_model(
        sk_model=model, name="model_output", serialization_format="pickle"
    )

    # Save the model locally
    joblib.dump(model, model_path)


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
