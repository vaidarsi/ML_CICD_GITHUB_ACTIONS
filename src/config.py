"""Central configuration: reads params.yaml and environment variables."""

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Load params.yaml and return it as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PARAMS = load_params()

# Paths (relative to project root)
RAW_DATA_PATH = PROJECT_ROOT / PARAMS["data"]["raw_path"]
PROCESSED_DATA_PATH = PROJECT_ROOT / PARAMS["data"]["processed_path"]
TARGET_COLUMN = PARAMS["data"]["target_column"]

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

# Environment variables (with safe defaults for local use)
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "local-dev")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"file:{PROJECT_ROOT / 'mlruns'}")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
API_PORT = int(os.getenv("API_PORT", "8000"))