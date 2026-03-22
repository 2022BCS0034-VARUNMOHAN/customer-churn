from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import os
import logging
import json
import time
from datetime import datetime
from app.features import extract_features

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

app = FastAPI(
    title="Churn Risk API — Stage 2 (ML-based)",
    description="ML-based churn prediction using RandomForest. Risk: LOW | MEDIUM | HIGH",
    version="2.0.0",
)

_metrics = {"total": 0, "errors": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

model_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
model = None

try:
    model = joblib.load(model_path)
    logger.info("ML model loaded from %s", model_path)
except FileNotFoundError:
    logger.warning("model.pkl not found — run scripts/train.py first.")
except Exception as e:
    logger.error("Failed to load model: %s", e)


class Ticket(BaseModel):
    type: Optional[str] = Field(None, json_schema_extra={"example": "complaint"})
    date: Optional[str] = Field(None, json_schema_extra={"example": "2026-03-01T10:00:00"})
    description: Optional[str] = None
    subject: Optional[str] = None


class CustomerData(BaseModel):
    monthly_charges: float = Field(..., json_schema_extra={"example": 80.0})
    previous_month_charges: float = Field(..., json_schema_extra={"example": 60.0})
    contract_type: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    tickets: List[Ticket] = Field(default=[])


class RiskResponse(BaseModel):
    risk: str
    engine: str
    probability: Optional[float] = None


@app.get("/", tags=["Health"])
def home():
    return {"status": "ok", "stage": "2 — ML-based", "version": "2.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "stage": 2, "model_loaded": model is not None}


@app.post("/predict-risk", response_model=RiskResponse, tags=["Prediction"])
def predict_risk(data: CustomerData):
    """
    ML-based churn risk prediction using trained RandomForest.
    Probability thresholds: >= 0.65 = HIGH | >= 0.35 = MEDIUM | else = LOW
    Returns 503 if model has not been trained yet.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Run scripts/train.py first.",
        )
    try:
        start = time.time()
        _metrics["total"] += 1
        features = extract_features(data.model_dump())
        prob = model.predict_proba(features)[0][1]

        if prob >= 0.65:
            risk = "HIGH"
        elif prob >= 0.35:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        _metrics[risk] += 1
        elapsed = round((time.time() - start) * 1000, 2)
        logger.info("ML prediction: risk=%s prob=%.4f latency_ms=%s",
                    risk, prob, elapsed)
        return RiskResponse(risk=risk, engine="ml", probability=round(prob, 4))

    except Exception as e:
        _metrics["errors"] += 1
        logger.error("ML prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.")


@app.get("/metrics", response_class=PlainTextResponse, tags=["Observability"])
def metrics():
    lines = [
        "# HELP churn_requests_total Total predictions",
        f"churn_requests_total {_metrics['total']}",
        "# HELP churn_errors_total Total errors",
        f"churn_errors_total {_metrics['errors']}",
        f"churn_high_total {_metrics['HIGH']}",
        f"churn_medium_total {_metrics['MEDIUM']}",
        f"churn_low_total {_metrics['LOW']}",
        "# HELP churn_model_loaded Whether ML model is loaded",
        f"churn_model_loaded {1 if model is not None else 0}",
    ]
    return "\n".join(lines)