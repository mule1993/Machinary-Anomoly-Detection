import io
import json
import os
from datetime import datetime, timezone

import boto3
import pandas as pd
import pandera as pa
import pandera.typing as pat

from src.data.data_validation import IngestedMachineSchema

# from src.data.preprocess import build_preprocessor


# The decorator intercepts the return value and validates it against the type hint
@pa.check_types(lazy=True)
def load_csv_from_s3_and_validate(
    aws_access_key_id=None, aws_secret_access_key=None, bucket_key=None
) -> pat.DataFrame[IngestedMachineSchema]:

    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=os.getenv("BUCKET_NAME"), Key=bucket_key)

    # Read the TSV/CSV stream
    return pd.read_csv(
        io.StringIO(obj["Body"].read().decode("utf-8")), sep=None, engine="python"
    )


# Upload processed data to S3 bucket as CSV
def upload_csv_to_s3(df, key=None, aws_access_key_id=None, aws_secret_access_key=None):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_REGION")),
    )
    s3 = session.client("s3")

    # Convert DataFrame to CSV string
    csv_buffer = df.to_csv(index=False, sep="\t")

    # Upload to S3
    s3.put_object(
        Bucket=os.getenv("PROCESSED_DATA_BUCKET"),
        Key=key or os.getenv("PROCESSED_BUCKET_PATH"),
        Body=csv_buffer,
    )

    # Example usage
    # if __name__ == "__main__":
    # df = load_csv_from_s3()
    # print(df.head())
    # preprocessor = build_preprocessor(df)
    # X_train = preprocessor.fit_transform(df)
    # print("Preprocessor created successfully.")
    # feature_names = preprocessor.get_feature_names_out()
    # df_processed = pd.DataFrame(X_train, columns=feature_names)
    # upload_csv_to_s3(df_processed)


def upload_payload_to_s3(
    payload_dict: dict,
    prediction_id: str,
    key=None,
    aws_access_key_id=None,
    aws_secret_access_key=None,
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_REGION")),
    )
    """
    Asynchronously uploads the inference and prediction payload to S3.
    """
    # 1. Generate the authoritative current UTC time (Python 3.12+ standard)
    now = datetime.now(timezone.utc)

    # Convert to ISO 8601 string format (e.g., "2026-06-15T12:15:30.123456+00:00")
    # We strip the "+00:00" and append "Z" to keep the naming clean and standardized
    timestamp_str = now.isoformat().replace("+00:00", "Z")

    # 2. Inject the timestamp directly into the payload body
    payload_dict["timestamp"] = timestamp_str

    s3 = session.client("s3")
    # Construct a time-partitioned S3 key prefix
    # Format: inference_logs/year=YYYY/month=MM/day=DD/pred_UUID.json
    s3_key = (
        f"inference_logs/"
        f"year={now.strftime('%Y')}/"
        f"month={now.strftime('%m')}/"
        f"day={now.strftime('%d')}/"
        f"pred_{prediction_id}.json"
    )

    try:
        # --- SANITIZATION STEP ---
        # If 'features' is a pandas DataFrame, convert it to a serializable list of dicts
        if isinstance(payload_dict.get("features"), pd.DataFrame):
            # orient="records" turns the rows into a clean list of dictionaries
            payload_dict["features"] = payload_dict["features"].to_dict(
                orient="records"
            )

        # If your prediction or probability are NumPy types (e.g., np.int64 or np.float32),
        # casting the entire dict values via a safe encoder or explicit conversion is necessary:
        if hasattr(payload_dict.get("prediction"), "item"):
            payload_dict["prediction"] = payload_dict["prediction"].item()
        if hasattr(payload_dict.get("probability"), "item"):
            payload_dict["probability"] = payload_dict["probability"].item()
        # Convert dictionary to stringified JSON
        json_data = json.dumps(payload_dict)

        # Upload object to S3
        s3.put_object(
            Bucket=os.getenv("BUCKET_NAME"),
            Key=s3_key,
            Body=json_data,
            ContentType="application/json",
        )
    except Exception as e:
        # Crucial: Log failures to stdout/stderr or CloudWatch logs.
        # Do not raise an HTTP exception here, as
        #  the user already received their response!
        print(f"ERROR: Failed to log payload {prediction_id} to S3: {str(e)}")
