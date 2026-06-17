import json
import os
import random
from datetime import datetime, timezone
from io import StringIO

import boto3
import pandas as pd

S3_BUCKET_NAME = os.getenv("BUCKET_NAME")  # Ensure this is set in your .env file
s3_client = boto3.client("s3")


def generate_and_upload_ground_truth(
    target_date_prefix: str = "inference_logs/year=2026/month=06",
):
    """
    Scans S3 for logged inferences, handles single/batch arrays correctly,
    simulates outcomes item-by-item, and uploads flat ground-truth records back to S3.
    """
    print(f"Scanning S3 logs under: {target_date_prefix}...")

    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET_NAME, Prefix=target_date_prefix
    )

    if "Contents" not in response:
        print("No logged payloads found in S3 for this partition window.")
        return

    maintenance_records = []
    now = datetime.now(timezone.utc)
    record_timestamp = now.isoformat().replace("+00:00", "Z")

    # 1. Gather logs and simulate actual operational outcomes row-by-row
    for obj in response["Contents"]:
        file_key = obj["Key"]

        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_key)
        payload_data = json.loads(s3_object["Body"].read().decode("utf-8"))

        predictions = payload_data["prediction"]
        prediction_id = payload_data["prediction_id"]
        inference_ts = payload_data["timestamp"]

        # --- THE FIX: Handle Batches vs. Singles explicitly ---
        # Scenario A: It's a genuine multi-inference batch log
        if isinstance(predictions, list) and len(predictions) > 1:
            for pred in predictions:
                # Run the 10% noise simulation logic on the individual scalar integer
                raw_pred = int(pred)
                actual_outcome = raw_pred if random.random() > 0.10 else (1 - raw_pred)

                maintenance_records.append(
                    {
                        "prediction_id": prediction_id,  # Keeps the shared lineage ID
                        "Machine failure": int(actual_outcome),  # Strict native integer
                        "maintenance_logged_at": record_timestamp,
                        "inference_logged_at": inference_ts,
                    }
                )

        # Scenario B: It's a single inference (either a scalar or wrapped in a 1-item list)
        else:
            # Safely unpack if it's a single-item list like [1], otherwise keep as is
            raw_pred = (
                int(predictions[0])
                if isinstance(predictions, list)
                else int(predictions)
            )

            actual_outcome = raw_pred if random.random() > 0.10 else (1 - raw_pred)

            maintenance_records.append(
                {
                    "prediction_id": prediction_id,
                    "Machine failure": int(actual_outcome),  # Strict native integer
                    "maintenance_logged_at": record_timestamp,
                    "inference_logged_at": inference_ts,
                }
            )
        # -----------------------------------------------------

    # 2. Convert collected records to a structured DataFrame
    df_maintenance = pd.DataFrame(maintenance_records)

    # 3. Serialize DataFrame straight to an in-memory string buffer
    csv_buffer = StringIO()
    df_maintenance.to_csv(csv_buffer, index=False)

    # 4. Construct a time-partitioned S3 key
    s3_dest_key = (
        f"ground_truth/"
        f"year={now.strftime('%Y')}/"
        f"month={now.strftime('%m')}/"
        f"maintenance_logs.csv"
    )

    # 5. Upload directly to S3
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_dest_key,
            Body=csv_buffer.getvalue(),
            ContentType="text/csv",
        )
        print(
            f"\nSUCCESS: Generated {len(df_maintenance)} flat ground-truth maintenance logs."
        )
        print(f"Uploaded to S3 path: s3://{S3_BUCKET_NAME}/{s3_dest_key}")
    except Exception as e:
        print(f"ERROR: Failed to save ground truth dataset to S3: {str(e)}")


if __name__ == "__main__":
    generate_and_upload_ground_truth()
