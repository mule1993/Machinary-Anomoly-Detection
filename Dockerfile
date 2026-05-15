FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install mlflow boto3 s3fs fsspec
# Copy the entire src folder so imports work correctly
COPY src/ ./src/

EXPOSE 8000

# Tell uvicorn exactly where the app is located
# It's inside src.api.main, and the FastAPI instance is named 'app'
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]