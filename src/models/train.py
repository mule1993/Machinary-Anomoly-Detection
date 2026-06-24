import hydra
import mlflow
import xgboost as xgb
from dotenv import load_dotenv
from omegaconf import DictConfig
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3_and_validate
from src.data.preprocess import build_preprocessor
from src.data.schemas import MachineFeaturesSchema
from src.models.StreamedPipelineWrapper import StreamedPipelineWrapper
import os
load_dotenv()
# ================= CONFIGURATION =================
# Load environment variables from .env file
# BUCKET_NAME = os.environ["BUCKET_NAME"]

# HYPERPARAMS = config["hyperparameters"]
# REGION = os.environ["AWS_REGION"]
# EXPERIMENT_NAME = os.environ["EXPERIMENT_NAME"]
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


# Load and split data
def load_and_split(config: DictConfig):
    df = load_csv_from_s3_and_validate(bucket_key=config.data_path)
    print(f"Dataset shape: {df.shape}")
    target_column = "Machine failure"
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test_, y_train, y_test = train_test_split(X, y, **config["data"])
    return X_train, X_test_, y_train, y_test


# Predict on the test set and calculate metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    # =================================================
    """# Evaluate the model
    print(classification_report(y_test, y_pred, zero_division=0))
    print(
        f"Precision: {
            precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(
        f"Recall: {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}"
    )
    print(f"F1-Score: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
"""
    # =================================================
    return precision, recall, f1


# Initialize MLflow experiment, train the model and log everything
def train_pipeline(config: DictConfig):
    print("\n--pipeline started ---")
    print(f"MLflow Tracking URI: {os.getenv('MLFLOW_TRACKING_URI')}")
    # Set the experiment
    try:
        mlflow.create_experiment(name=config.experiment_name)
    except Exception:
        pass
    mlflow.set_experiment(config.experiment_name)

    # The Run
    print("\n---mlflow found---")
    with mlflow.start_run(run_name=config.model_run_name) as run:
        run_id = run.info.run_id
        print(f"Tracking URI: {mlflow.get_tracking_uri()}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

        # LOG THE HYPERPARAMETERS AS METADATA
        mlflow.log_params(config["hyperparameters"])
        mlflow.log_param("test_size", config["data"]["test_size"])
        # load data and build pipeline
        X_train, X_test_, y_train, y_test = load_and_split(config)
        preprocessor = build_preprocessor(X_train)
        X_train = preprocessor.fit_transform(X_train)
        # Enforce vectorized schema check on evaluation split
        X_test = MachineFeaturesSchema.validate(X_test_, lazy=True)
        X_test = preprocessor.transform(X_test)

        # =================================================
        # Dummy Model
        # model = DummyFailureModel()
        # =================================================
        # Train the model
        model = xgb.XGBClassifier(**config["hyperparameters"])
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        """# log the preprocessor separately
        mlflow.sklearn.log_model(
        sk_model=preprocessor,
        name="preprocessor_module",
        input_example=raw_input_example,
        code_path=["src/models/custom_transformers.py"],
        registered_model_name=model_name,
        )

        #  Log model to S3 separetely
        mlflow.sklearn.log_model(
        sk_model=model,
        name="model_module",
        serialization_format="pickle",
        conda_env=conda_env,
        registered_model_name=model_name,
        )"""

        # Predict on the test set
        precision, recall, f1 = evaluate_model(model, X_test, y_test)
        # Logging evaluation metrics to MLflow
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Stream the objects directly into custom wrapper
        unified_pipeline = StreamedPipelineWrapper(
            preprocessor=preprocessor, model=model
        )
        model_info = mlflow.pyfunc.log_model(
            name="model",
            python_model=unified_pipeline,
            registered_model_name=config.model_name,
        )
        client = mlflow.tracking.MlflowClient()
        # Adding a metadata description tag
        client.update_model_version(
            name=config.model_name,
            version=model_info.registered_model_version,
            description=config.model_description,
        )
        client.set_registered_model_alias(
            name=config.model_name,
            alias=config.alias,
            version=model_info.registered_model_version,
        )

        print(f"✅ Successfully logged run {run_id}")


# Load the external configuration layer
@hydra.main(version_base=None, config_path="../../config/", config_name="config")
def training(config: DictConfig):
    print("\n---training pipeline ---")
    train_pipeline(config)


if __name__ == "__main__":
    training()
