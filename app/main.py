"""FastAPI app that serves the trained model and exposes Prometheus metrics."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src import config
from src.predict import FEATURE_NAMES, load_model, predict_one

# ---- Prometheus metrics ----
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds", ["endpoint"]
)
ERROR_COUNT = Counter("http_errors_total", "Total HTTP error responses (4xx/5xx)")
PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions served", ["prediction"]
)
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded, else 0")

# ---- Model holder ----
state: dict = {"model": None}


def try_load_model() -> None:
    """Load the model into memory. Leaves it None if the file is missing."""
    try:
        state["model"] = load_model()
        MODEL_LOADED.set(1)
    except FileNotFoundError:
        state["model"] = None
        MODEL_LOADED.set(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try_load_model()
    yield


app = FastAPI(
    title="Wine Classifier API",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=len(FEATURE_NAMES),
        max_length=len(FEATURE_NAMES),
        description=f"Exactly {len(FEATURE_NAMES)} numeric features, in dataset column order",
    )


class PredictResponse(BaseModel):
    prediction: str
    model_version: str


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        # Use the route path (not the raw URL) to keep label cardinality low
        route = request.scope.get("route")
        endpoint = route.path if route else "unmatched"
        REQUEST_COUNT.labels(request.method, endpoint, str(status)).inc()
        REQUEST_LATENCY.labels(endpoint).observe(time.perf_counter() - start)
        if status >= 400:
            ERROR_COUNT.inc()


@app.get("/")
def root() -> dict:
    return {
        "app": "Wine Classifier API",
        "model_version": config.MODEL_VERSION,
        "environment": config.ENVIRONMENT,
        "endpoints": ["/health", "/predict", "/metrics", "/docs"],
    }


@app.get("/health")
def health() -> dict:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    label = predict_one(state["model"], request.features)
    PREDICTION_COUNT.labels(label).inc()
    return PredictResponse(prediction=label, model_version=config.MODEL_VERSION)


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
