from src.data.ingest import load_csv_from_s3

bucket = "machinery-mlops-muluneh-2026"
key = "data/raw/ai4i2020.csv"

df = load_csv_from_s3(bucket, key)
print(df.head())
