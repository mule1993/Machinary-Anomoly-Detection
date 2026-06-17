import io

import boto3
import pandas as pd

S3_BUCKET_NAME = "machinery-mlops-muluneh-2026"
s3_client = boto3.client("s3")

# Define columns to isolate
METADATA_COLUMNS = [
    "prediction_id",
    "model_guess",
    "maintenance_logged_at",
    "inference_logged_at",
]
TARGET_COLUMN = "Machine failure"
TARGET_DESIRED_INDEX = 8  # 0-indexed position 8 makes it the 9th column


def clean_and_export_training_set(year="2026", month="06"):
    print(f"=== [Data Prep] Processing Partition: year={year}/month={month} ===")

    # 1. Define S3 Paths
    pool_key = f"joined_datasets/year={year}/month={month}/evaluation_training_pool.csv"
    clean_set_key = f"data/year={year}/month={month}/clean_training_set.csv"

    # 2. Download the evaluation training pool
    try:
        print(f"📥 Downloading raw pool from: s3://{S3_BUCKET_NAME}/{pool_key}")
        pool_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=pool_key)
        df_pool = pd.read_csv(io.StringIO(pool_obj["Body"].read().decode("utf-8")))
    except Exception as e:
        print(f"❌ Error loading evaluation pool: {e}")
        return False

    if df_pool.empty:
        print("🛑 Evaluation pool is empty. Aborting.")
        return False

    # 3. Strip Metadata Columns
    df_clean = df_pool.drop(columns=METADATA_COLUMNS, errors="ignore")

    if TARGET_COLUMN not in df_clean.columns:
        print(f"❌ Target column '{TARGET_COLUMN}' missing from pool data. Aborting.")
        return False

    # 4. Extract and Clean Target Series
    target_series = df_clean[TARGET_COLUMN].astype(int)
    df_features = df_clean.drop(columns=[TARGET_COLUMN])

    # 5. --- EXPLICIT POSITION LAYOUT RECONSTRUCTION ---
    # Check if we have enough columns to place it at the 9th position safely
    if len(df_features.columns) >= TARGET_DESIRED_INDEX:
        # Reconstruct: take first 8 columns, insert target, append remaining columns
        cols_before = list(df_features.columns[:TARGET_DESIRED_INDEX])
        cols_after = list(df_features.columns[TARGET_DESIRED_INDEX:])

        df_final = df_features[cols_before].copy()
        df_final[TARGET_COLUMN] = target_series

        # Add the remaining features back
        for col in cols_after:
            df_final[col] = df_features[col]
    else:
        # Fallback: If there are fewer than 8 features total, append target to the end
        print(
            f"⚠️ Warning: Feature count ({len(df_features.columns)}) is less than target index layout requirement."
        )
        df_final = df_features.copy()
        df_final[TARGET_COLUMN] = target_series

    print("📊 Structural Transformation Complete:")
    print(f"   -> Original Pool Shape  : {df_pool.shape}")
    print(f"   -> Final Export Shape    : {df_final.shape}")
    print(f"   -> Target Column Name   : '{TARGET_COLUMN}'")
    print(
        f"   -> Target True Position : Column Index {df_final.columns.get_loc(TARGET_COLUMN)} (Column #{df_final.columns.get_loc(TARGET_COLUMN) + 1})"
    )

    # 6. --- UPLOAD REORDERED CLEAN DATASET BACK TO S3 ---
    try:
        csv_buffer = io.StringIO()
        df_final.to_csv(csv_buffer, index=False, lineterminator="\n")
        csv_string = csv_buffer.getvalue()

        # Strip potential wrapping quotes that confuse Pandera's CSV parser
        if csv_string.startswith('"UDI,') or csv_string.startswith('""UDI,'):
            print(
                "🔧 Detected header formatting anomaly. Auto-cleaning quote encapsulations..."
            )
            csv_string = csv_string.strip('"')

        print(
            f"📤 Uploading schema-aligned training set to: s3://{S3_BUCKET_NAME}/{clean_set_key}"
        )
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=clean_set_key,
            Body=csv_string,
            ContentType="text/csv",
        )
        print(
            "✅ Structured training dataset successfully isolated, reordered, and saved."
        )
        return True

    except Exception as e:
        print(f"❌ Failed to upload clean training set to S3: {e}")
        return False


if __name__ == "__main__":
    clean_and_export_training_set()
