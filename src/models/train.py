import os

import hydra
import mlflow
import xgboost as xgb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3
from src.data.preprocess import build_preprocessor
from src.models.StreamedPipelineWrapper import StreamedPipelineWrapper

# ================= CONFIGURATION =================
load_dotenv()  # Load environment variables from .env file
# BUCKET_NAME = os.environ["BUCKET_NAME"]
REGION = os.environ["AWS_REGION"]
EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
# alias = os.getenv("MODEL_ALIAS")
# Define model name in the registry
# model_name = os.getenv("MODEL_NAME")

# ====================Silence======================
"""
# Silence the specific sklearn flavor logger warning
logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)

# requirements.txt dependencies warning workaround for MLflow's conda environment
conda_env = {
   "channels": ["conda-forge"],
   "dependencies": [
      "python=3.10",
    {"pip": ["mlflow", "boto3", "s3fs"]},
 ],
 "name": "mlflow_env",
 }"""

# =================================================


# 1. Load the external configuration layer
# 2. Tell Hydra to read config.yaml from the current directory (".")
@hydra.main(version_base=None, config_path="../../config/", config_name="config")
def train_pipeline(config: DictConfig):
    # Set the experiment
    try:
        mlflow.create_experiment(name=config.experiment_name)
    except Exception:
        pass
    mlflow.set_experiment(config.experiment_name)

    # 4. The Run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

        # 2. LOG THE HYPERPARAMETERS AS METADATA
        # This records your entire config block into the MLflow database ledger
        mlflow.log_params(config["hyperparameters"])
        mlflow.log_param("test_size", config["data"]["test_size"])

        # Load and preprocess data
        df = load_csv_from_s3(bucket_key=os.getenv("RAW_DATA_PATH"))
        target_column = "Machine failure"
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, **config["data"])
        # print("X_test before preprocessing:")
        # print(X_test)
        preprocessor = build_preprocessor(X_train)
        # raw_input_example = X_test.iloc[[0]].copy()
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        # print("X_test after preprocessing:")
        # print(X_test)

        # =================================================
        # Initialize model
        # model = DummyFailureModel()
        #  Model training
        model = xgb.XGBClassifier(**config["hyperparameters"])

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # =================================================
        """# log the preprocessor separately
        mlflow.sklearn.log_model(
        sk_model=preprocessor,
        name="preprocessor_module",
        input_example=raw_input_example,
        code_path=["src/models/custom_transformers.py"],
        registered_model_name=model_name,
        )

        #  Log model to S3 separetely so we can reuse it in API without the wrapper overhead
        mlflow.sklearn.log_model(
        sk_model=model,
        name="model_module",
        serialization_format="pickle",
        conda_env=conda_env,
        registered_model_name=model_name,  # This automatically registers it
        )"""

        # 3. Stream the objects directly into your custom wrapper in-memory
        unified_pipeline = StreamedPipelineWrapper(
            preprocessor=preprocessor, model=model
        )

        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=unified_pipeline,
            registered_model_name=config.model_name,
        )
        # Assigning an Alias ("champion" or "production")
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(
            name=config.model_name,
            alias=config.alias,
            version=model_info.registered_model_version,
        )

        print(f"✅ Successfully logged run {run_id} and tagged as '{config.alias}'")

        # Predict on the test set

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    # Logging evaluation metrics to MLflow
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    # =================================================
    # Evaluate the model
    print(classification_report(y_test, y_pred, zero_division=0))
    print(
        f"Precision: {
            precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Recall: {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )

    # =================================================


if __name__ == "__main__":
    train_pipeline()
