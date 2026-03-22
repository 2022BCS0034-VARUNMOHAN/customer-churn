"""
tests/test_api.py
Unit tests for Stage 2 ML-based prediction endpoint.
Run: pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app, model

client = TestClient(app)

VALID_RISK_VALUES = {"HIGH", "MEDIUM", "LOW"}


def post_predict(payload: dict) -> dict:
    response = client.post("/predict-risk", json=payload)
    if response.status_code == 503:
        pytest.skip("ML model not loaded — run scripts/train.py first")
    assert response.status_code == 200, (
        f"Unexpected status {response.status_code}: {response.text}"
    )
    body = response.json()
    assert "risk" in body
    assert "engine" in body
    assert "probability" in body
    assert body["risk"] in VALID_RISK_VALUES
    assert body["engine"] == "ml"
    assert 0.0 <= body["probability"] <= 1.0
    return body


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["stage"] == "2 — ML-based"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["stage"] == 2
    assert "model_loaded" in data


def test_health_model_loaded_is_bool():
    assert isinstance(client.get("/health").json()["model_loaded"], bool)


def test_complaint_customer():
    """Month-to-month + complaints."""
    body = post_predict({
        "monthly_charges": 90,
        "previous_month_charges": 50,
        "contract_type": "Month-to-month",
        "tickets": [
            {"type": "complaint", "date": "2026-03-18T10:00:00",
             "description": "service broken terrible awful"},
            {"type": "complaint", "date": "2026-03-17T10:00:00",
             "description": "still broken frustrated angry"},
            {"type": "complaint", "date": "2026-03-16T10:00:00",
             "description": "worst service horrible"},
        ],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_stable_customer():
    """Long-term customer with no tickets."""
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "Two year",
        "tickets": [],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_probability_valid_float():
    body = post_predict({
        "monthly_charges": 70,
        "previous_month_charges": 65,
        "contract_type": "One year",
        "tickets": [
            {"type": "query", "date": "2026-03-10T10:00:00",
             "description": "how do I upgrade my plan"},
        ],
    })
    assert 0.0 <= body["probability"] <= 1.0


def test_many_recent_complaints():
    body = post_predict({
        "monthly_charges": 100,
        "previous_month_charges": 60,
        "contract_type": "Month-to-month",
        "tickets": [
            {"type": "complaint", "date": "2026-03-20T10:00:00",
             "description": "broken useless terrible"},
            {"type": "complaint", "date": "2026-03-19T10:00:00",
             "description": "awful service horrible"},
            {"type": "complaint", "date": "2026-03-18T10:00:00",
             "description": "frustrated angry worst"},
            {"type": "complaint", "date": "2026-03-17T10:00:00",
             "description": "failure bad service"},
            {"type": "complaint", "date": "2026-03-16T10:00:00",
             "description": "useless broken again"},
            {"type": "complaint", "date": "2026-03-15T10:00:00",
             "description": "terrible experience"},
        ],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_empty_tickets():
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "Month-to-month",
        "tickets": [],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_no_description_field():
    """Tickets without description should not crash."""
    body = post_predict({
        "monthly_charges": 60,
        "previous_month_charges": 55,
        "contract_type": "One year",
        "tickets": [
            {"type": "complaint", "date": "2026-03-15T10:00:00"},
            {"type": "query", "date": "2026-03-10T10:00:00"},
        ],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_malformed_date_ignored():
    """Bad dates must not crash the API."""
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": [
            {"type": "complaint", "date": "not-a-date"},
            {"type": "query", "date": "also-bad"},
        ],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_requests_total" in response.text
    assert "churn_model_loaded" in response.text


def test_503_when_model_not_loaded(monkeypatch):
    import app.main as m
    monkeypatch.setattr(m, "model", None)
    response = client.post("/predict-risk", json={
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "Month-to-month",
        "tickets": [],
    })
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"]