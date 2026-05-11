from pathlib import Path

import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.data.ingest import load_csv_from_s3

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "models"

# Load data from S3
df = load_csv_from_s3()

# Prepare data for training
target_column = "Machine failure"
X = df.drop(target_column, axis=1)
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Train a dummy model
class DummyFailureModel:
    """A model that always predicts 0 (No Failure)."""

    def predict(self, X):
        return [0] * len(X)


# Save the model
model = DummyFailureModel()
model_path = MODEL_DIR / "DummyFailureModel.joblib"
joblib.dump(model, model_path)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, zero_division=0))
