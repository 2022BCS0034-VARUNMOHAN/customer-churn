"""
app/main.py

The FastAPI inference API — Stage 2 (ML-based).

What changed from Stage 1:
  • /predict-risk now uses the trained ML pipeline (model.pkl) instead of rules
  • /predict-risk-rules still available for comparison (keeps the old rule engine)
  • /model-info shows which model version is loaded
  • /metrics exposes Prometheus-style counters (same as before)

How the ML inference works:
  1. Request arrives with customer data (JSON)
  2. extract_features() converts it to a numeric vector  ← SAME code as training!
  3. pipeline.predict_proba() returns churn probability
  4. We threshold at 0.5 → LOW / HIGH
  5. Response returned with risk label + probability
"""

import os
import logging
import json
import time
from datetime import datetime
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.features import extract_features, FEATURE_NAMES
from app.rules import calculate_risk

# ── Logging (JSON format — easy to parse in log aggregators) ─────────────────
class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# ── Load ML model at startup ──────────────────────────────────────────────────
# We load the model ONCE when the server starts (not on every request).
# This is important for performance — model loading takes ~1 second.
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")

_model = None          # will hold the sklearn Pipeline
_model_loaded_at = None

def load_model():
    """Load model from disk. Called once at startup."""
    global _model, _model_loaded_at
    if os.path.exists(MODEL_PATH):
        _model = joblib.load(MODEL_PATH)
        _model_loaded_at = datetime.utcnow().isoformat()
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        logger.warning("model.pkl not found at %s — ML endpoint will return 503", MODEL_PATH)

load_model()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Risk API — Stage 2 (ML-based)",
    description=(
        "ML-powered churn risk engine using a trained RandomForest pipeline.\n\n"
        "Also exposes the legacy rule-based endpoint for comparison."
    ),
    version="2.0.0",
)

# ── In-memory metrics counters ────────────────────────────────────────────────
# In production you'd use Prometheus client library, but this is simpler for now.
_metrics = {
    "total_ml": 0,
    "total_rules": 0,
    "errors": 0,
    "HIGH": 0,
    "LOW": 0,
    "total_latency_ms": 0.0,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic models  (these define what JSON the API accepts)
# ══════════════════════════════════════════════════════════════════════════════

class Ticket(BaseModel):
    type: Optional[str] = Field(None, example="complaint")
    date: Optional[str] = Field(None, example="2026-03-01T10:00:00")
    description: Optional[str] = Field(None, example="Service keeps dropping, very frustrated")


class CustomerData(BaseModel):
    monthly_charges: float = Field(..., example=80.0)
    previous_month_charges: float = Field(..., example=60.0)
    contract_type: str = Field(..., example="Month-to-month")
    tickets: List[Ticket] = Field(default=[])


class RiskResponse(BaseModel):
    risk: str                    # "HIGH" or "LOW"
    engine: str                  # "ml" or "rules"
    churn_probability: Optional[float] = None   # only set for ML predictions
    features_used: Optional[dict] = None        # feature values (for debugging)


# ══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def home():
    return {
        "status": "ok",
        "stage": "2 — ML-based",
        "version": "2.0.0",
        "model_loaded": _model is not None,
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "stage": 2,
        "model_ready": _model is not None,
        "model_loaded_at": _model_loaded_at,
    }


@app.get("/model-info", tags=["Model"])
def model_info():
    """
    Returns information about the currently loaded model.
    Useful to confirm which model version is serving traffic.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Peek inside the pipeline to get classifier details
    clf = _model.named_steps.get("classifier", None)
    return {
        "model_path": MODEL_PATH,
        "model_loaded_at": _model_loaded_at,
        "pipeline_steps": list(_model.named_steps.keys()),
        "feature_names": FEATURE_NAMES,
        "n_estimators": getattr(clf, "n_estimators", None),
        "n_features": getattr(clf, "n_features_in_", None),
    }


@app.post("/predict-risk", response_model=RiskResponse, tags=["Prediction"])
def predict_risk_ml(data: CustomerData):
    """
    ML-based churn risk prediction.

    Steps:
    1. Convert customer JSON → numeric feature vector (using features.py)
    2. Run through trained sklearn pipeline
    3. Return churn probability + risk label
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run: python scripts/train.py"
        )

    t_start = time.time()
    try:
        _metrics["total_ml"] += 1

        # ── Feature extraction ────────────────────────────────────────────────
        customer_dict = data.model_dump()
        feature_vector = extract_features(customer_dict)   # [[f1, f2, ...]]

        # Put it in a DataFrame with column names (pipeline expects this)
        X = pd.DataFrame(feature_vector, columns=FEATURE_NAMES)

        # ── Inference ─────────────────────────────────────────────────────────
        churn_prob = float(_model.predict_proba(X)[0][1])   # probability of churn

        # Threshold: above 0.5 = HIGH risk, below = LOW risk
        risk = "HIGH" if churn_prob >= 0.5 else "LOW"
        _metrics[risk] += 1

        # ── Latency tracking ──────────────────────────────────────────────────
        latency_ms = (time.time() - t_start) * 1000
        _metrics["total_latency_ms"] += latency_ms

        logger.info(
            "ML prediction: risk=%s prob=%.3f contract=%s tickets=%d latency=%.1fms",
            risk, churn_prob, data.contract_type, len(data.tickets), latency_ms,
        )

        # Feature values for debugging (shown in response)
        features_dict = dict(zip(FEATURE_NAMES, feature_vector[0]))

        return RiskResponse(
            risk=risk,
            engine="ml",
            churn_probability=round(churn_prob, 4),
            features_used=features_dict,
        )

    except Exception as e:
        _metrics["errors"] += 1
        logger.error("ML prediction error: %s", e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict-risk-rules", response_model=RiskResponse, tags=["Prediction"])
def predict_risk_rules(data: CustomerData):
    """
    Legacy rule-based prediction (kept for comparison / fallback).
    This is identical to the Stage 1 endpoint.
    """
    try:
        _metrics["total_rules"] += 1
        risk = calculate_risk(data.model_dump())
        logger.info("Rule prediction: risk=%s", risk)
        return RiskResponse(risk=risk, engine="rules")
    except Exception as e:
        _metrics["errors"] += 1
        logger.error("Rule prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
def metrics():
    """
    Prometheus-style metrics endpoint.
    Scraped by Prometheus (configured in prometheus.yml).
    """
    n = _metrics["total_ml"]
    avg_latency = (_metrics["total_latency_ms"] / n) if n > 0 else 0.0

    lines = [
        "# HELP churn_ml_requests_total Total ML predictions",
        f"churn_ml_requests_total {_metrics['total_ml']}",
        "# HELP churn_rules_requests_total Total rule-based predictions",
        f"churn_rules_requests_total {_metrics['total_rules']}",
        "# HELP churn_errors_total Total errors",
        f"churn_errors_total {_metrics['errors']}",
        "# HELP churn_high_total Predictions labelled HIGH",
        f"churn_high_total {_metrics['HIGH']}",
        "# HELP churn_low_total Predictions labelled LOW",
        f"churn_low_total {_metrics['LOW']}",
        "# HELP churn_avg_latency_ms Average inference latency in milliseconds",
        f"churn_avg_latency_ms {avg_latency:.2f}",
    ]
    return "\n".join(lines)