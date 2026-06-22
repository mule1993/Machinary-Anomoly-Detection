from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.amazon.aws.hooks.base_aws import AwsGenericHook
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "machinery_failure_retraining_pipeline",
    default_args=default_args,
    description="Automated orchestration loop for Machinery Failure XGBClassifier",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    # 1. Fetch AWS credentials dynamically from Airflow's secure Connection manager
    try:
        aws_hook = AwsGenericHook(aws_conn_id="aws_default")
        creds = aws_hook.get_credentials()
        aws_access_key = creds.access_key
        aws_secret_key = creds.secret_key
    except Exception:
        # Fallback placeholders for initialization phase
        aws_access_key = "PLACEHOLDER"
        aws_secret_key = "PLACEHOLDER"

    # 2. Build out runtime environment variables to pass into your worker container
    container_env = {
        "AWS_ACCESS_KEY_ID": aws_access_key,
        "AWS_SECRET_ACCESS_KEY": aws_secret_key,
        "AWS_DEFAULT_REGION": "us-east-1",
        "MLFLOW_TRACKING_URI": "http://host.docker.internal:5000",  # Maps safely to local machine localhost
        "BUCKET_NAME": "machinery-mlops-muluneh-2026",
        "MODEL_NAME": "machinery_failure_anomaly_detector",
        "MODEL_ALIAS": "production",
        "ALIAS1": "challenger",
        "TARGET_METRIC": "f1",
        "EXPERIMENT_NAME": "machinery-failure.anomaly-detection",
        "RAW_DATA_PATH": "data/raw/ai4i2020.csv",
        "MODEL_DESCRIPTION": "An XGBoost model for machinery failure anomaly detection",
        "MODEL_RUN_NAME": "xgboost_baseline",
        "HYDRA_FULL_ERROR": "1",
    }

    # 3. Mount configurations
    # Replace this with the exact absolute path to your config directory on your Ubuntu machine
    config_mount = Mount(
        target="/app/config",
        source="/home/muluneh/Machinary-Anomoly-Detection/config",  # <--- Update this to your absolute host path
        type="bind",
    )

    # Old local pointer:
    # IMAGE_URI = "machinery-prod:latest"

    # New remote cloud pointer (Replace with your actual AWS account ID):
    IMAGE_URI = "457736182931.dkr.ecr.us-east-1.amazonaws.com/machinery-prod:latest"

    # Task 1: Simulate Incoming Data/Labels (TEMPORARY DEBUG OVERRIDE)
    simulate_labels = DockerOperator(
        task_id="docker_simulate_labels",
        image=IMAGE_URI,
        container_name="airflow_simulate_labels",
        auto_remove="success",
        mount_tmp_dir=False,
        working_dir="/app",
        environment=container_env,
        mounts=[config_mount],
        command=["find", "."],  # <--- Change this from python to ['find', '.']
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    # Task 2: Evaluate Performance
    evaluate_inference = DockerOperator(
        task_id="docker_evaluate_inference",
        image=IMAGE_URI,
        container_name="airflow_evaluate_inference",
        auto_remove="success",
        mount_tmp_dir=False,
        working_dir="/app",  # <--- Explicitly sets the context execution folder
        environment=container_env,
        mounts=[config_mount],
        command=[
            "python",
            "src/evaluate_inference_performance.py",
        ],  # <--- Array formatting
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    # Task 3: Retrain Model
    retrain_model = DockerOperator(
        task_id="docker_retrain_model",
        image=IMAGE_URI,
        container_name="airflow_retrain_model",
        auto_remove="success",
        mount_tmp_dir=False,
        working_dir="/app",
        environment=container_env,
        mounts=[config_mount],
        command=[
            "python",
            "src/models/retraining.py",
        ],  # <--- Matches your subfolder layout
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    # Task 4: Evaluate Promotion Gating Logic
    evaluate_promotion = DockerOperator(
        task_id="docker_evaluate_promotion",
        image=IMAGE_URI,
        container_name="airflow_evaluate_promotion",
        auto_remove="success",
        mount_tmp_dir=False,
        working_dir="/app",
        environment=container_env,
        mounts=[config_mount],
        command=[
            "python",
            "src/models/evaluate_models_for_promotion.py",
        ],  # <--- Matches your subfolder layout
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
    )

    # Set up our linear orchestration dependencies
    simulate_labels >> evaluate_inference >> retrain_model >> evaluate_promotion
