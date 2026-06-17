import os

import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
MODEL_NAME = os.environ["MODEL_NAME"]
TARGET_METRIC = os.environ["TARGET_METRIC"]
ALIAS1 = os.environ["ALIAS1"]  # Alias for the latest trained model (Challenger)
ALIAS2 = os.environ["MODEL_ALIAS"]  # Alias for the current reigning model (production)
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()


def get_run_metric(model_version_instance, metric_name):
    """Helper function to fetch a metric value from a model version's source run."""
    run_id = model_version_instance.run_id
    run = client.get_run(run_id)
    return run.data.metrics.get(metric_name, 0.0)


print("--- Starting Automated Model Evaluation Gate ---")

"""
# 1. Fetch the latest registered version (freshly trained Challenger)
latest_versions = client.get_latest_versions(MODEL_NAME)
if not latest_versions:
    print(
        f"No registered models found for {MODEL_NAME}. Please run a training script first."
    )
    exit()

# The highest version number represents the newest trained model
# challenger_version = max(latest_versions, key=lambda v: int(v.version))
"""
challenger_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS1)
challenger_f1 = get_run_metric(challenger_version, TARGET_METRIC)

print(
    f"Challenger: Version {challenger_version.version} | F1-Score: {challenger_f1:.4f}"
)

# 2. Fetch the current reigning Production Champion
try:
    production_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS2)
    champion_f1 = get_run_metric(production_version, TARGET_METRIC)
    print(
        f"Current Champion: Version {production_version.version} | F1-Score: {champion_f1:.4f}"
    )
    has_champion = True
except mlflow.exceptions.MlflowException:
    print(
        f"No current {ALIAS2} alias found in the registry. First model will be promoted automatically."
    )
    has_champion = False

# 3. Gating Logic: Compare and Swap
if not has_champion:
    # If there is no champion yet, promote the challenger immediately
    client.set_registered_model_alias(
        MODEL_NAME, ALIAS2, str(challenger_version.version)
    )
    print(
        f"🏆 Initialized Registry! Version {challenger_version.version} is now the @{ALIAS2}."
    )

elif challenger_f1 > champion_f1:
    # If the challenger strictly beats the champion, swap the alias
    print(f"📈 Challenger beat the Champion by {challenger_f1 - champion_f1:.4f}!")
    client.set_registered_model_alias(
        MODEL_NAME, ALIAS2, str(challenger_version.version)
    )
    print(
        f"🔄 Swapped! Version {challenger_version.version} has been promoted to @{ALIAS2}."
    )

    # Optional: Tag the old champion as a fallback/deprecating asset
    client.set_registered_model_alias(
        MODEL_NAME, "fallback", str(production_version.version)
    )

else:
    # If the challenger underperforms, leave the champion alone
    print(
        f"❌ Gate Rejected: Challenger (F1: {challenger_f1:.4f}) did not beat Champion (F1: {champion_f1:.4f})."
    )
    print("Reigning @champion remains unchanged.")
