from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "xgboost-failure-prediction-model"  # Put your exact $MODEL_NAME here

# Fetch the version assigned to your alias
version_details = client.get_model_version_by_alias(model_name, "production")

print(f"[*] Alias '@production' is pointing to Version: {version_details.version}")
print(f"[*] Source S3 Path: {version_details.source}")
