import os

import boto3
import pandas as pd


# Load CSV data from S3 bucket
def load_csv_from_s3(aws_access_key_id=None, aws_secret_access_key=None):
    session = boto3.Session(
        aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=aws_secret_access_key
        or os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=os.getenv("BUCKET_NAME"), Key=os.getenv("RAW_DATA_PATH"))  # noqa: E501
    # Read the CSV file directly from the S3 object(tab-separated value)
    return pd.read_csv(obj["Body"], sep="\t")


# Example usage
if __name__ == "__main__":
    df = load_csv_from_s3()
    print(df.head())
