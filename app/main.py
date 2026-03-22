from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import logging, json
from datetime import datetime
from app.rules import calculate_risk

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
    title="Churn Risk API — Stage 1 (Rule-based)",
    description="Pure rule-based churn risk engine. No ML. Risk: LOW | MEDIUM | HIGH",
    version="1.0.0",
)

_metrics = {"total": 0, "errors": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}


class Ticket(BaseModel):
    type: Optional[str] = Field(None, json_schema_extra={"example": "complaint"})
    date: Optional[str] = Field(None, json_schema_extra={"example": "2026-03-01T10:00:00"})
    description: Optional[str] = None


class CustomerData(BaseModel):
    monthly_charges: float = Field(..., json_schema_extra={"example": 80.0})
    previous_month_charges: float = Field(..., json_schema_extra={"example": 60.0})
    contract_type: str = Field(..., json_schema_extra={"example": "Month-to-month"})
    tickets: List[Ticket] = Field(default=[])


class RiskResponse(BaseModel):
    risk: str
    engine: str = "rules"


@app.get("/", tags=["Health"])
def home():
    return {"status": "ok", "stage": "1 — Rule-based", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy", "stage": 1}


@app.post("/predict-risk", response_model=RiskResponse, tags=["Prediction"])
def predict_risk(data: CustomerData):
    """
    Rule-based churn risk prediction.
    Rules:
    1. Month-to-month + complaint ticket → HIGH
    2. More than 5 tickets in last 30 days → HIGH
    3. Charge increase + 3 recent tickets → MEDIUM
    4. Default → LOW
    """
    try:
        _metrics["total"] += 1
        risk = calculate_risk(data.model_dump())
        _metrics[risk] += 1
        logger.info("Rule prediction: risk=%s contract=%s tickets=%d",
                    risk, data.contract_type, len(data.tickets))
        return RiskResponse(risk=risk, engine="rules")
    except Exception as e:
        _metrics["errors"] += 1
        logger.error("Prediction error: %s", e)
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
    ]
    return "\n".join(lines)