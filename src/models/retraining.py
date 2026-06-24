import os
import sys
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report
import os
import mlflow

# Import the bulletproof extraction and join functions from your previous script
# 1. Dynamically locate the absolute path of the project root
# (Steps up two levels from src/models/retraining.py to reach the repo root)
project_root = str(Path(__file__).resolve().parents[2])

# 2. Inject it into the Python path if it isn't already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now your existing absolute imports will work flawlessly anywhere!
from src.evaluate_inference_performance import (  # noqa: E402
    extract_scalar_value,
    fetch_and_join_datasets,
)
from src.models.train import train_pipeline  # noqa: E402

# =================== RETRAINING CONFIGURATION =================

load_dotenv()
# Define thresholds for your system
MINIMUM_SUPPORT_TO_RETRAIN = 10  # Don't retrain on tiny samples
F1_THRESHOLD = 0.81  # Retrain if model performance falls below this
DATA_PATH = "data/year=2026/month=06/clean_training_set.csv"
dynamic_scale_weight = 1
# ==============================================================


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def retraining(config: DictConfig):
    OmegaConf.set_readonly(config, False)

    # Overrides for retraining,
    # data balance weight hyeperparameters change not architectural hyperparameters
    config.data_path = DATA_PATH
    config.hyperparameters.scale_pos_weight = dynamic_scale_weight
    print(f"INFO: retraining with scale_pos_weight={dynamic_scale_weight}")
    print(f"and data_path={DATA_PATH}")

#    config.alias = alias1
    print("pipeline start")
    train_pipeline(config)


def run_automated_maintenance_loop():
    print("Starting automated system performance evaluation...")

    # 1. Fetch and join S3 data windows
    df_metrics = fetch_and_join_datasets()

    if df_metrics is None or len(df_metrics) == 0:
        print("Aborting loop: No production data available.")
        return

    # 2. Extract and clean the targets using your bulletproof logic
    y_true = (
        df_metrics["Machine failure"]
        .apply(extract_scalar_value)
        .dropna()
        .astype(int)
        .values
    )
    y_pred = (
        df_metrics["model_guess"]
        .apply(extract_scalar_value)
        .dropna()
        .astype(int)
        .values
    )

    total_samples = len(y_true)

    # 3. Compute metrics dictionary dynamically
    report_dict = classification_report(y_true, y_pred, output_dict=True)

    # Extract the anomaly F1-score safely (handle string/int key variations)
    anomaly_f1 = report_dict.get("1", report_dict.get("Anomaly (1)", {})).get(
        "f1-score", 1.0
    )

    print("\n--- Evaluation Diagnostics ---")
    print(f"Total Logged Production Samples: {total_samples}")
    print(f"Current Production Anomaly F1-Score: {anomaly_f1:.2f}")
    print("--------------------------------")

    # 4. The Decision Gate
    if total_samples < MINIMUM_SUPPORT_TO_RETRAIN:
        print(
            f"Decision: STAND DOWN. Data pool ({total_samples}) is too small to trigger retraining safely."
        )
        return

    if anomaly_f1 >= F1_THRESHOLD:
        print(
            f"Decision: STAND DOWN. Production model is performing optimally (F1 >= {F1_THRESHOLD})."
        )
        return

    # 5. RETRAINING TRIGGERED
    print(
        f"🚨 ALERT: Performance dropped below threshold! (F1: {anomaly_f1:.2f} < {F1_THRESHOLD})"
    )
    print("Initiating automated pipeline retraining sequence via MLflow...")

    # 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨RETRAINING TO BE FIXED
    retraining()

    print(
        "🎉 SUCCESS: Automated retraining run completed and registered as a Challenger model in MLflow!"
    )


if __name__ == "__main__":
    run_automated_maintenance_loop()
