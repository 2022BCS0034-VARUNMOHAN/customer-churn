"""
scripts/train.py
Train a RandomForest churn classifier and save model + evaluation artifacts.
Run from the project root: python scripts/train.py
"""

import json
import sys
import os
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import joblib
import matplotlib.pyplot as plt

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.features import extract_features, FEATURE_NAMES  # single source of truth

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("artifacts", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
data_path = os.path.join("data", "processed_data.json")
logger.info("Loading data from %s", data_path)

with open(data_path) as f:
    data = json.load(f)

# ── Build feature matrix using the same extract_features() as inference ───────
rows = []
for customer in data:
    features_row = extract_features(customer).values[0].tolist()
 # returns [[...]]

    # Derive churn label (same logic as original)
    tickets = customer.get("tickets", [])
    contract = customer.get("contract_type", "")

    from datetime import datetime
    now = datetime.now()
    t30 = 0
    complaint = 0
    for t in tickets:
        if "date" in t:
            try:
                d = datetime.fromisoformat(t["date"])
                if (now - d).days <= 30:
                    t30 += 1
            except Exception:
                pass
        if t.get("type") == "complaint":
            complaint += 1

    churn = 1 if (t30 > 5 or (contract.lower() == "month-to-month" and complaint > 0)) else 0

    rows.append(features_row + [churn])

columns = FEATURE_NAMES + ["churn"]
df = pd.DataFrame(rows, columns=columns)

logger.info("Dataset shape: %s | Churn rate: %.1f%%", df.shape, df["churn"].mean() * 100)

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── Train ─────────────────────────────────────────────────────────────────────
logger.info("Training RandomForestClassifier ...")
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

# ── Metrics ───────────────────────────────────────────────────────────────────
report = classification_report(y_test, preds)
roc_auc = roc_auc_score(y_test, probs)
avg_precision = average_precision_score(y_test, probs)

logger.info("\n%s", report)
logger.info("ROC AUC:          %.4f", roc_auc)
logger.info("Avg Precision:    %.4f", avg_precision)

with open("artifacts/metrics.txt", "w") as f:
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nROC AUC:        {roc_auc:.4f}")
    f.write(f"\nAvg Precision:  {avg_precision:.4f}")

# ── ROC curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/roc_curve.png", dpi=150)
plt.close()
logger.info("Saved artifacts/roc_curve.png")

# ── Precision-Recall curve ────────────────────────────────────────────────────
precision, recall, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f"PR (AP = {avg_precision:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/pr_curve.png", dpi=150)
plt.close()
logger.info("Saved artifacts/pr_curve.png")

# ── Feature importance ────────────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=True)
plt.figure(figsize=(7, 4))
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png", dpi=150)
plt.close()
logger.info("Saved artifacts/feature_importance.png")

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")
logger.info("Model saved as model.pkl")
