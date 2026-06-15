"""Fetch the version assigned to your alias and print the S3 path for verification.

import os

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = os.getenv("MODEL_NAME")  # Put your exact $MODEL_NAME here
alias_name = os.getenv("MODEL_ALIAS")  # e.g., "production"
# Fetch the version assigned to your alias
version_details = client.get_model_version_by_alias(model_name, alias_name)

print(f"[*] Alias '@{alias_name}' is pointing to Version: {version_details.version}")
print(f"[*] Source S3 Path: {version_details.source}")

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
    print(f"Error cleaning Experiments: {e}")"""

# CSV to JSON conversion script
import csv
import json

csv_file_path = "/home/muluneh/Downloads/MLOps/ai4i+2020+predictive+maintenance+dataset/ai4i2020.csv"
json_file_path = (
    "/home/muluneh/Downloads/MLOps/ai4i+2020+predictive+maintenance+dataset/data.json"
)
# Define which columns should be integers or floats
int_columns = {
    "UDI",
    "Rotational speed [rpm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
}
float_columns = {"Air temperature [K]", "Process temperature [K]", "Torque [Nm]"}

data = []

with open(csv_file_path, mode="r", encoding="utf-8") as csv_file:
    # Added delimiter='\t' to properly separate the tabbed data
    csv_reader = csv.DictReader(csv_file, delimiter="\t")

    for row in csv_reader:
        converted_row = {}
        for key, value in row.items():
            # Clean up any accidental whitespace around keys/values
            key = key.strip()
            value = value.strip()

            # Convert to integer if applicable
            if key in int_columns:
                converted_row[key] = int(value)
            # Convert to float if applicable
            elif key in float_columns:
                converted_row[key] = float(value)
            # Keep as string for "Product ID" and "Type"
            else:
                converted_row[key] = value

        data.append(converted_row)

# Save to JSON file
with open(json_file_path, mode="w", encoding="utf-8") as json_file:
    json.dump(data, json_file, indent=2)

print("Conversion complete! Tab-separated values successfully parsed.")
