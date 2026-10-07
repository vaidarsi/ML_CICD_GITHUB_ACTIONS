FROM python:3.11-slim

# Don't write .pyc files; show logs immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, config and the trained model
COPY app/ ./app/
COPY src/ ./src/
COPY params.yaml .
COPY models/model.joblib ./models/model.joblib

# Runtime configuration (can be overridden with docker run -e ...)
ENV MODEL_PATH=/app/models/model.joblib \
    MODEL_VERSION=local-dev \
    ENVIRONMENT=production \
    API_PORT=8000

EXPOSE 8000

# Run as a non-root user (safer)
RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]