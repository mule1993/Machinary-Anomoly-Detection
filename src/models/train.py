from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3

bucket = "machinery-mlops-muluneh-2026"
key = "data/raw/ai4i2020.csv"
model_path = "src/models/DummyFailureModel.joblib"
df = load_csv_from_s3(bucket, key)
# print(df.head())

target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class DummyFailureModel:
    """A model that always predicts 0 (No Failure)."""

    def predict(self, X):
        return [0] * len(X)


model = DummyFailureModel()
y_pred = model.predict(X_test)

# print(classification_report(y_test, y_pred))
# print(y_test.dtype)
# print(y_pred.dtype)
# joblib.dump(model, model_path)
