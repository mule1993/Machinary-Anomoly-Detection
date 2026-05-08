import os

import boto3
import pandas as pd

aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = "us - east - 1"
bucket = "machinery-mlops-muluneh-2026"
key = "data/raw/ai4i2020.csv"


def load_csv_from_s3(
    bucket, key, aws_access_key_id=None, aws_secret_access_key=None, region_name=None
):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=region_name or os.getenv("AWS_DEFAULT_REGION"),
    )
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


if __name__ == "__main__":
    df = load_csv_from_s3(bucket, key)
    print(df.head())
