"""
tests/test_api.py
Unit tests for Stage 1 rule-based prediction endpoint.
Run: pytest tests/ -v
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_RISK_VALUES = {"HIGH", "MEDIUM", "LOW"}


def post_predict(payload: dict) -> dict:
    response = client.post("/predict-risk", json=payload)
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    body = response.json()
    assert "risk" in body, "Response missing 'risk' field"
    assert "engine" in body, "Response missing 'engine' field"
    assert body["risk"] in VALID_RISK_VALUES, f"Invalid risk value: {body['risk']}"
    return body


# ── Health checks ─────────────────────────────────────────────────────────────
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    # Stage 1 returns status and stage only — no model_loaded field
    assert response.json()["status"] == "healthy"
    assert response.json()["stage"] == 1


# ── HIGH risk cases ───────────────────────────────────────────────────────────
def test_high_risk_month_to_month_with_complaint():
    """Month-to-month + complaint ticket must return HIGH."""
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 20,
        "contract_type": "Month-to-month",
        "tickets": [
            {"type": "complaint", "date": "2026-03-01T10:00:00"}
        ],
    })
    assert body["risk"] == "HIGH"
    assert body["engine"] == "rules"


def test_high_risk_many_recent_tickets():
    """More than 5 tickets in last 30 days must return HIGH."""
    recent_tickets = [
        {"type": "query", "date": "2026-03-18T10:00:00"},
        {"type": "query", "date": "2026-03-17T10:00:00"},
        {"type": "query", "date": "2026-03-16T10:00:00"},
        {"type": "query", "date": "2026-03-15T10:00:00"},
        {"type": "query", "date": "2026-03-14T10:00:00"},
        {"type": "query", "date": "2026-03-13T10:00:00"},
    ]
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": recent_tickets,
    })
    assert body["risk"] == "HIGH"


# ── MEDIUM risk cases ─────────────────────────────────────────────────────────
def test_medium_risk_charge_increase_with_tickets():
    """Charge increase + 3 recent tickets must return MEDIUM."""
    recent_tickets = [
        {"type": "query", "date": "2026-03-10T10:00:00"},
        {"type": "query", "date": "2026-03-11T10:00:00"},
        {"type": "query", "date": "2026-03-12T10:00:00"},
    ]
    body = post_predict({
        "monthly_charges": 80,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": recent_tickets,
    })
    assert body["risk"] == "MEDIUM"


# ── LOW risk cases ────────────────────────────────────────────────────────────
def test_low_risk_no_issues():
    """No complaints, stable charges, annual contract → LOW."""
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": [],
    })
    assert body["risk"] == "LOW"


def test_low_risk_old_tickets_ignored():
    """Tickets older than 30 days should not trigger HIGH or MEDIUM."""
    old_tickets = [
        {"type": "query", "date": "2025-01-01T10:00:00"},
        {"type": "query", "date": "2025-01-02T10:00:00"},
        {"type": "query", "date": "2025-01-03T10:00:00"},
        {"type": "query", "date": "2025-01-04T10:00:00"},
        {"type": "query", "date": "2025-01-05T10:00:00"},
        {"type": "query", "date": "2025-01-06T10:00:00"},
    ]
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": old_tickets,
    })
    assert body["risk"] == "LOW"


# ── Edge cases ────────────────────────────────────────────────────────────────
def test_empty_tickets():
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "Month-to-month",
        "tickets": [],
    })
    assert body["risk"] in VALID_RISK_VALUES


def test_malformed_ticket_date_ignored():
    """Tickets with bad dates should not crash the API."""
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
    """Metrics endpoint must return Prometheus-formatted text."""
    client.post("/predict-risk", json={
        "monthly_charges": 50,
        "previous_month_charges": 20,
        "contract_type": "Month-to-month",
        "tickets": [{"type": "complaint", "date": "2026-03-01T10:00:00"}],
    })
    response = client.get("/metrics")
    assert response.status_code == 200
    # Stage 1 metrics use churn_requests_total
    assert "churn_requests_total" in response.text
    assert "churn_high_total" in response.text