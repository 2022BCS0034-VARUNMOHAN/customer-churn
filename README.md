# Customer Churn Prediction — MLOps Assignment

## Overview

Two-stage MLOps system predicting customer churn risk from support ticket behaviour.

## Stage 1 — Rule-based (branch: stage1)

Pure rule engine, no ML.

Rules:

- Month-to-month contract + complaint ticket → HIGH
- More than 5 tickets in last 30 days → HIGH
- Charge increase + 3 recent tickets → MEDIUM
- Default → LOW

Run:
pip install -r requirements.txt
uvicorn app.main:app --reload

## Stage 2 — ML-based (branch: stage2)

RandomForest classifier replacing the rule engine.

Features: ticket frequency (7d/30d/90d), sentiment score,
time between tickets, charge change, contract type.

Run:
pip install -r requirements.txt
python scripts/prepare_data.py
python scripts/train.py
uvicorn app.main:app --reload

## API Endpoints

| Endpoint      | Method | Description        |
| ------------- | ------ | ------------------ |
| /             | GET    | Health check       |
| /health       | GET    | Liveness probe     |
| /predict-risk | POST   | Predict churn risk |
| /metrics      | GET    | Prometheus metrics |
| /docs         | GET    | Swagger UI         |

## Docker

    docker build -t churn-ml .
    docker run -p 8000:8000 churn-ml

## Docker Compose (with Prometheus)

    docker-compose up --build
