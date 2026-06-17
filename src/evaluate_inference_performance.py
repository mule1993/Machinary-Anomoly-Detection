import json
import os
import re
from io import StringIO

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()

S3_BUCKET_NAME = os.getenv("BUCKET_NAME")
s3_client = boto3.client("s3")
year = os.getenv("EVAL_YEAR")
month = os.getenv("EVAL_MONTH")


def fetch_and_join_datasets(year="2026", month="06"):
    """
    Downloads raw feature JSONs and the flat ground-truth CSV from S3,
    joins them uniquely using prediction_id + sub_index, and returns the gold DataFrame.
    """
    # 1. Download the ground truth CSV file from S3
    gt_key = f"ground_truth/year={year}/month={month}/maintenance_logs.csv"
    print(f"Fetching ground truth logs from: s3://{S3_BUCKET_NAME}/{gt_key}")

    try:
        gt_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=gt_key)
        df_ground_truth = pd.read_csv(StringIO(gt_obj["Body"].read().decode("utf-8")))

        # Add a sequential sub_index group counter per prediction_id to prevent many-to-many explosions
        df_ground_truth["sub_index"] = df_ground_truth.groupby(
            "prediction_id"
        ).cumcount()
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return None

    # 2. Find and download all corresponding inference log JSON files
    inference_prefix = f"inference_logs/year={year}/month={month}/"
    print(
        f"Scanning inference payloads under: s3://{S3_BUCKET_NAME}/{inference_prefix}"
    )

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=inference_prefix)

    inference_records = []

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            log_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=obj["Key"])
            payload = json.loads(log_obj["Body"].read().decode("utf-8"))

            features_data = payload["features"]
            prediction_id = payload["prediction_id"]
            model_guess = payload["prediction"]

            # Scenario A: If features_data is a batch list of multiple inferences
            if isinstance(features_data, list):
                for i, item in enumerate(features_data):
                    row = item.copy()
                    row["prediction_id"] = prediction_id
                    row["sub_index"] = i  # Match positional sequence explicitly

                    if isinstance(model_guess, list) and len(model_guess) == len(
                        features_data
                    ):
                        row["model_guess"] = int(model_guess[i])
                    else:
                        row["model_guess"] = int(model_guess)

                    inference_records.append(row)

            # Scenario B: If features_data is a single direct inference dictionary
            elif isinstance(features_data, dict):
                row = features_data.copy()
                row["prediction_id"] = prediction_id
                row["sub_index"] = 0  # Single items default to sub-index 0

                # Unpack scalar value cleanly
                row["model_guess"] = (
                    int(model_guess[0])
                    if isinstance(model_guess, list)
                    else int(model_guess)
                )
                inference_records.append(row)

            # --- THE CRITICAL FIX: REMOVED DUPLICATE LEAKY CODE OUTSIDE THE BLOCKS ---

    df_features = pd.DataFrame(inference_records)

    if df_features.empty:
        print("No matching inference logs found.")
        return None

    # 3. --- THE JOIN CORE ---
    # Merge using BOTH the orchestration UUID and the positional batch index to force a clean 1:1 join
    df_joined = pd.merge(
        df_features, df_ground_truth, on=["prediction_id", "sub_index"]
    )

    # Drop the operational helper index before sending to S3/MLflow
    df_joined = df_joined.drop(columns=["sub_index"])

    print(f"✅ Successfully performed 1:1 join. Final count: {len(df_joined)} records.")

    # 4. --- THE S3 LOGGING ARCHIVE ---
    try:
        csv_buffer = StringIO()
        df_joined.to_csv(csv_buffer, index=False)

        joined_s3_key = (
            f"joined_datasets/year={year}/month={month}/evaluation_training_pool.csv"
        )
        print(
            f"📦 Archiving joined training pool to S3: s3://{S3_BUCKET_NAME}/{joined_s3_key}"
        )

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=joined_s3_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )
    except Exception as e:
        print(f"⚠️ Warning: Failed to archive df_joined to S3: {e}")

    return df_joined


def extract_scalar_value(val):
    """
    Completely bulletproof extraction that strips all layers of nested brackets,
    quotes, and formatting to return a clean native integer.
    """
    # 1. Handle None or empty values immediately
    if val is None:
        return None

    # 2. If it's a native number or numpy scalar, cast it straight to int
    if isinstance(val, (int, float, np.number)):
        return int(val)

    # 3. Convert everything else to a string to clean up string-wrapped lists
    val_str = str(val).strip()

    # Use Regex to extract only the digits.
    # This strips away '[', ']', ' ', and any quotes automatically.
    match = re.search(r"-?\d+", val_str)

    if match:
        return int(match.group())

    return None


def evaluate_production_health():
    # Execute the join for the current data window
    df_metrics = fetch_and_join_datasets()

    if df_metrics is None or len(df_metrics) == 0:
        return

    # -------------------------------

    print("\n" + "=" * 50)
    print("      LIVE PRODUCTION MODEL PERFORMANCE REPORT      ")
    print("=" * 50)

    # --- THE FIX ---
    # Apply our scalar extraction function to every single row to normalize the shapes
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

    # Ensure our targets line up perfectly
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    # Print a quick sanity check to console
    print(f"Total processed samples matching shape: {len(y_pred)}")

    # Generate the report safely
    report = classification_report(
        y_true, y_pred, target_names=["Healthy (0)", "Anomaly (1)"]
    )
    print(report)
    print("=" * 50)


if __name__ == "__main__":
    evaluate_production_health()
