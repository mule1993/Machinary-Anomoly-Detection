"""Fetch the version assigned to your alias and print the S3 path for verification.

import os

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = os.getenv("MODEL_NAME")  # Put your exact $MODEL_NAME here
alias_name = os.getenv("MODEL_ALIAS")  # e.g., "production"
# Fetch the version assigned to your alias
version_details = client.get_model_version_by_alias(model_name, alias_name)

print(f"[*] Alias '@{alias_name}' is pointing to Version: {version_details.version}")
print(f"[*] Source S3 Path: {version_details.source}")"""

# This script is a utility to clean up all registered models and experiments from
# an MLflow tracking server
import mlflow
from mlflow.tracking import MlflowClient

# 1. Point to your tracking server URI
TRACKING_URI = "http://localhost:5000"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

print("---Starting MLflow Deep Clean---")

# ==========================================
# 1. DELETE ALL REGISTERED MODELS
# ==========================================
print("\nCleaning Model Registry...")
try:
    registered_models = client.search_registered_models()
    for rm in registered_models:
        print(f"Deleting registered model: {rm.name}")
        # We must delete the registered model container itself
        client.delete_registered_model(name=rm.name)
    print("✓ Model Registry cleared.")
except Exception as e:
    print(f"Error cleaning Registry: {e}")

# ==========================================
# 2. DELETE ALL EXPERIMENTS
# ==========================================
print("\nCleaning Experiments...")
try:
    experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    for exp in experiments:
        # Do not delete the default experiment (ID '0'), it cannot be fully removed
        if exp.experiment_id == "0":
            print("Skipping default experiment '0' (will reset runs inside it instead)")
            continue

        print(f"Deleting experiment: {exp.name} (ID: {exp.experiment_id})")
        client.delete_experiment(experiment_id=exp.experiment_id)
    print("✓ Custom experiments soft-deleted.")
except Exception as e:
    print(f"Error cleaning Experiments: {e}")
