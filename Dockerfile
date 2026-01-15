FROM python:3.10-slim

# Faster + cleaner python logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Make src/ importable inside container
ENV PYTHONPATH=/app/src

# Train model at build time so artifacts.joblib exists in the image
RUN python -m credit_risk.train

# Expose API port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
