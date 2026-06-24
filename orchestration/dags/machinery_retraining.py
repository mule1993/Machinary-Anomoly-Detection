import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.models.connection import Connection
from airflow.providers.amazon.aws.hooks.base_aws import AwsGenericHook
from airflow.providers.amazon.aws.hooks.ecr import EcrHook
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.session import create_session
from docker.types import Mount

# 1. Grab the host repository root from the environment variable if present
HOST_REPO_ROOT = os.getenv("HOST_REPO_ROOT")

if HOST_REPO_ROOT:
    # Running on EC2 production instance
    CONFIG_HOST_PATH = os.path.join(HOST_REPO_ROOT, "config")
else:
    # Fallback configuration for your local laptop environment
    DAG_DIR = os.path.dirname(os.path.abspath(__file__))
    REPO_ROOT = os.path.abspath(os.path.join(DAG_DIR, "../.."))
    CONFIG_HOST_PATH = os.path.join(REPO_ROOT, "config")

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
    # 1. Runtime task that safely logs into ECR and writes/refreshes an Airflow Connection
    @task
    def sync_ecr_connection():
        try:
            # Grab base AWS credentials for container environments
            aws_hook = AwsGenericHook(aws_conn_id="aws_default")
            creds = aws_hook.get_credentials()

            # Fetch dynamic temporary ECR password
            ecr_hook = EcrHook(aws_conn_id="aws_default")
            ecr_creds = ecr_hook.get_temporary_credentials()[0]

            # Upsert a dedicated Docker Registry Connection inside Airflow's metadata database
            conn_id = "ecr_docker_registry"
            with create_session() as session:
                query = session.query(Connection).filter(Connection.conn_id == conn_id)
                conn = query.first()
                if not conn:
                    conn = Connection(conn_id=conn_id, conn_type="docker")

                conn.host = "457736182931.dkr.ecr.us-east-1.amazonaws.com"
                conn.login = ecr_creds.username  # Will always be "AWS"
                conn.set_password(ecr_creds.password)

                session.add(conn)
                session.commit()

            return {
                "aws_access_key": creds.access_key,
                "aws_secret_key": creds.secret_key,
            }
        except Exception as e:
            raise RuntimeError(
                f"Failed to synchronize ECR registry authentication: {str(e)}"
            )

    auth_data = sync_ecr_connection()

    # 2. Map container environments back to runtime templates
    container_env = {
        "AWS_ACCESS_KEY_ID": "{{ ti.xcom_pull(task_ids='sync_ecr_connection')['aws_access_key'] }}",
        "AWS_SECRET_ACCESS_KEY": "{{ ti.xcom_pull(task_ids='sync_ecr_connection')['aws_secret_key'] }}",
        "AWS_DEFAULT_REGION": "us-east-1",
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI"),
        "BUCKET_NAME": os.getenv("BUCKET_NAME"),
        "MODEL_NAME": os.getenv("MODEL_NAME"),
        "MODEL_ALIAS": os.getenv("MODEL_ALIAS"),
        "ALIAS1": os.getenv("ALIAS1"),
        "TARGET_METRIC": os.getenv("TARGET_METRIC"),
        "EXPERIMENT_NAME": os.getenv("EXPERIMENT_NAME"),
        "RAW_DATA_PATH": os.getenv("RAW_DATA_PATH"),
        "MODEL_DESCRIPTION": os.getenv("MODEL_DESCRIPTION"),
        "MODEL_RUN_NAME": os.getenv("MODEL_RUN_NAME"),
        "DATA_PATH": os.getenv("RAW_DATA_PATH"),
        "HYDRA_FULL_ERROR": "1",
    }

    config_mount = Mount(
        target="/app/config",
        source="/home/muluneh/Machinary-Anomoly-Detection/config",
        type="bind",
    )

    IMAGE_URI = "457736182931.dkr.ecr.us-east-1.amazonaws.com/machinery-prod:latest"

    # Task 1: Simulate Incoming Data/Labels
    simulate_labels = DockerOperator(
        task_id="docker_simulate_labels",
        image=IMAGE_URI,
        container_name="airflow_simulate_labels",
        auto_remove=True,
        mount_tmp_dir=False,
        force_pull=True,
        working_dir="/app",
        environment=container_env,  # <--- ADD THIS LINE HERE
        command=["python", "src/simulate_labels.py"],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        docker_conn_id="ecr_docker_registry",
        api_version="auto",
    )

    # Task 2: Evaluate Performance
    evaluate_inference = DockerOperator(
        task_id="docker_evaluate_inference",
        image=IMAGE_URI,
        container_name="airflow_evaluate_inference",
        auto_remove=True,
        mount_tmp_dir=False,
        force_pull=True,
        working_dir="/app",
        environment=container_env,
        mounts=[
            Mount(
                # Update this to your verified host tracking directory configuration
                source="/opt/machinery-anomaly-detection/config",
                target="/app/config",
                type="bind",
            )
        ],
        command=["python", "src/evaluate_inference_performance.py"],
        docker_url="unix://var/run/docker.sock",
        network_mode="bridge",
        docker_conn_id="ecr_docker_registry",
        extra_hosts={"host.docker.internal": "host-gateway"},  # <--- ADD THIS
    )

    # Task 3: Retrain Model
    retrain_model = DockerOperator(
        task_id="docker_retrain_model",
        image=IMAGE_URI,
        container_name="airflow_retrain_model",
        auto_remove=True,
        force_pull=True,
        mount_tmp_dir=False,
        working_dir="/app",
        environment=container_env,
        mounts=[
            Mount(
                # Update this to your verified host tracking directory configuration
                source=CONFIG_HOST_PATH,
                target="/app/config",
                type="bind",
            )
        ],
        # Dynamically format the tracking URI into the command line string for Hydra
        # Force Hydra to run with a clean runtime directory structure so it never caches old tracking IPs
        command="python src/models/retraining.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        docker_conn_id="ecr_docker_registry",
        extra_hosts={"host.docker.internal": "host-gateway"},  # <--- ADD THIS
    )

    # Task 4: Evaluate Promotion Gating Logic
    evaluate_promotion = DockerOperator(
        task_id="docker_evaluate_promotion",
        image=IMAGE_URI,
        container_name="airflow_evaluate_promotion",
        auto_remove=True,
        force_pull=True,
        mount_tmp_dir=False,
        working_dir="/app",
        environment=container_env,
        mounts=[
            Mount(
                # Update this to your verified host tracking directory configuration
                source=CONFIG_HOST_PATH,
                target="/app/config",
                type="bind",
            )
        ],
        command="python src/models/evaluate_models_for_promotion.py",
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        docker_conn_id="ecr_docker_registry",
        extra_hosts={"host.docker.internal": "host-gateway"},  # <--- ADD THIS
    )

    (
        auth_data
        >> simulate_labels
        >> evaluate_inference
        >> retrain_model
        >> evaluate_promotion
    )
