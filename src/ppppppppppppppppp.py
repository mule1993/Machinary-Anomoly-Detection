import os

from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = os.getenv("MODEL_NAME")  # Put your exact $MODEL_NAME here
alias_name = os.getenv("MODEL_ALIAS")  # e.g., "production"
# Fetch the version assigned to your alias
version_details = client.get_model_version_by_alias(model_name, alias_name)

print(f"[*] Alias '@{alias_name}' is pointing to Version: {version_details.version}")
print(f"[*] Source S3 Path: {version_details.source}")
