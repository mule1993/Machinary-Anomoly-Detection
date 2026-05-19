import logging
import os

import mlflow
import xgboost as xgb
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3
from src.data.preprocess import build_preprocessor

# ================= CONFIGURATION =================
# BUCKET_NAME = os.environ["BUCKET_NAME"]
REGION = os.environ["AWS_REGION"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
ARTIFACT_PATH = os.environ["ARTIFACT_PATH"]
alias = os.getenv("MODEL_ALIAS")
# Define model name in the registry
model_name = os.getenv("MODEL_NAME")
# =================================================

# Load and preprocess data
df = load_csv_from_s3(RAW_DATA_PATH=os.getenv("RAW_DATA_PATH"))
target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
preprocessor = build_preprocessor(X_train)
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


# =================================================
# Initialize model
# model = DummyFailureModel()
#  Model training
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

# Autolog captures everything during this .fit() call
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
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
# Set the experiment
try:
    mlflow.create_experiment(name=EXPERIMENT_NAME, artifact_location=ARTIFACT_PATH)
except Exception:
    pass
mlflow.set_experiment(EXPERIMENT_NAME)


# 4. The Run
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Artifact URI: {mlflow.get_artifact_uri()}")

    # 1. Standard Tagging/Params
    mlflow.set_tag("model_type", "dummy_classifier")
    mlflow.log_param("data_source", "manual_test")

    # 2. Log model to S3
    # 'artifact_path' creates the folder inside the S3 run directory
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model_output",
        serialization_format="pickle",
        conda_env=conda_env,
        registered_model_name=model_name,  # This automatically registers it
    )
    # 3. Industry Standard: Assign an Alias (e.g., "champion" or "production")
    client = mlflow.tracking.MlflowClient()

    # Get the latest version we just registered
    model_version_details = client.get_latest_versions(model_name, stages=["None"])[0]

    # Set the alias so your Docker container knows which version to pull
    client.set_registered_model_alias(
        name=model_name, alias=alias, version=model_version_details.version
    )
    print(f"✅ Successfully logged run {run_id} and tagged as '{alias}'")


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
