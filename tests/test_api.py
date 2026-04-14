"""
tests/test_api.py
Unit tests for rule-based prediction endpoint (Stage 2 compatible).
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
    response = client.post("/predict-risk-rules", json=payload)
    assert response.status_code == 200, f"Unexpected status: {response.status_code}"
    body = response.json()
    assert "risk" in body
    assert "engine" in body
    assert body["risk"] in VALID_RISK_VALUES
    return body


# ── Health checks ─────────────────────────────────────────────────────────────
def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["stage"] == 2


# ── HIGH risk cases ───────────────────────────────────────────────────────────
def test_high_risk_month_to_month_with_complaint():
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 20,
        "contract_type": "Month-to-month",
        "tickets": [
            {"type": "complaint", "date": "2026-04-10T10:00:00"}
        ],
    })
    assert body["risk"] == "HIGH"
    assert body["engine"] == "rules"


def test_high_risk_many_recent_tickets():
    recent_tickets = [
        {"type": "query", "date": "2026-04-14T10:00:00"},
        {"type": "query", "date": "2026-04-13T10:00:00"},
        {"type": "query", "date": "2026-04-12T10:00:00"},
        {"type": "query", "date": "2026-04-11T10:00:00"},
        {"type": "query", "date": "2026-04-10T10:00:00"},
        {"type": "query", "date": "2026-04-09T10:00:00"},
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
    recent_tickets = [
        {"type": "query", "date": "2026-04-14T10:00:00"},
        {"type": "query", "date": "2026-04-13T10:00:00"},
        {"type": "query", "date": "2026-04-12T10:00:00"},
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
    body = post_predict({
        "monthly_charges": 50,
        "previous_month_charges": 50,
        "contract_type": "One year",
        "tickets": [],
    })
    assert body["risk"] == "LOW"


def test_low_risk_old_tickets_ignored():
    old_tickets = [
        {"type": "query", "date": "2025-01-01T10:00:00"},
        {"type": "query", "date": "2025-01-02T10:00:00"},
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
    client.post("/predict-risk-rules", json={
        "monthly_charges": 50,
        "previous_month_charges": 20,
        "contract_type": "Month-to-month",
        "tickets": [{"type": "complaint", "date": "2026-04-10T10:00:00"}],
    })
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "churn_ml_requests_total" in response.text
    assert "churn_high_total" in response.text