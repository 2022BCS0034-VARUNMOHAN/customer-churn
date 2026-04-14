"""
scripts/train.py

What this script does (in plain English):
  1. Loads processed customer data from data/processed_data.json
  2. Extracts ML features using app/features.py  (same code used at inference!)
  3. Trains a RandomForest classifier inside an sklearn Pipeline
  4. Evaluates with F1, ROC-AUC, Precision-Recall
  5. Logs EVERYTHING to MLflow  (params, metrics, model, artifacts)
  6. Registers the model in MLflow Model Registry
  7. Saves model.pkl locally as a fallback

Run from project root:
    python scripts/train.py

MLflow UI (to view results):
    mlflow ui
    Then open http://localhost:5000 in your browser
"""

import json
import sys
import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import joblib
import matplotlib
matplotlib.use("Agg")   # no display needed — works in Docker/servers too
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# ── Allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.features import extract_features, FEATURE_NAMES

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)      # MLflow stores runs here

# ── MLflow configuration ──────────────────────────────────────────────────────
# We use a local file-based MLflow tracking server (no extra server needed)
# All runs are saved in the ./mlruns folder
MLFLOW_TRACKING_URI = "mlruns"            # relative path → ./mlruns
EXPERIMENT_NAME = "churn-prediction"
MODEL_NAME = "churn-classifier"           # name in the Model Registry

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create the experiment if it doesn't exist yet
try:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

mlflow.set_experiment(EXPERIMENT_NAME)

logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
logger.info("Experiment          : %s (id=%s)", EXPERIMENT_NAME, experiment_id)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
data_path = os.path.join("data", "processed_data.json")
logger.info("Loading data from %s", data_path)

with open(data_path) as f:
    data = json.load(f)

logger.info("Loaded %d customer records", len(data))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Feature engineering + label creation
# ══════════════════════════════════════════════════════════════════════════════
# We use the SAME extract_features() that the API uses at inference time.
# This guarantees training and inference always compute features identically.

rows = []
for customer in data:
    # extract_features returns [[f1, f2, ...]] → we take the inner list
    feature_vector = extract_features(customer)[0]

    # ── Derive churn label ────────────────────────────────────────────────────
    # A customer is labelled "churned" if they had > 5 tickets in last 30 days
    # OR they are month-to-month and had at least one complaint.
    # This mirrors the rule engine logic so the ML model learns from the same signal.
    tickets  = customer.get("tickets", [])
    contract = customer.get("contract_type", "")
    now      = datetime.now()

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
    rows.append(feature_vector + [churn])

columns = FEATURE_NAMES + ["churn"]
df = pd.DataFrame(rows, columns=columns)

churn_rate = df["churn"].mean() * 100
logger.info("Dataset shape : %s", df.shape)
logger.info("Churn rate    : %.1f%%", churn_rate)

X = df.drop("churn", axis=1)
y = df["churn"]

# 80% train, 20% test — random_state=42 makes this reproducible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Build sklearn Pipeline
# ══════════════════════════════════════════════════════════════════════════════
# A Pipeline chains preprocessing + model into ONE object.
# Benefits:
#   • Save/load preprocessing + model together → no mismatch
#   • Inference just calls pipeline.predict(raw_features)
#   • Prevents "data leakage" (scaler fitted only on train data)

HYPERPARAMS = {
    "n_estimators"  : 200,    # number of trees in the forest
    "max_depth"     : None,   # None = trees grow until pure leaves
    "random_state"  : 42,     # reproducibility seed
    "class_weight"  : "balanced",  # handles class imbalance automatically
}

pipeline = Pipeline([
    # Step 1: StandardScaler normalises each feature to mean=0, std=1
    # RandomForest doesn't strictly need this, but it helps consistency
    # and makes the pipeline work with other classifiers too
    ("scaler", StandardScaler()),

    # Step 2: The actual classifier
    ("classifier", RandomForestClassifier(**HYPERPARAMS)),
])

logger.info("Training pipeline (StandardScaler → RandomForestClassifier) ...")
pipeline.fit(X_train, y_train)
logger.info("Training complete.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]   # probability of churn=1

f1       = f1_score(y_test, preds, average="weighted")
roc_auc  = roc_auc_score(y_test, probs)
avg_prec = average_precision_score(y_test, probs)
report   = classification_report(y_test, preds)

logger.info("\n%s", report)
logger.info("F1 (weighted)  : %.4f", f1)
logger.info("ROC-AUC        : %.4f", roc_auc)
logger.info("Avg Precision  : %.4f", avg_prec)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Save evaluation plots
# ══════════════════════════════════════════════════════════════════════════════
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, probs)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/roc_curve.png", dpi=150)
plt.close()

# Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
plt.figure(figsize=(6, 5))
plt.plot(recall_vals, precision_vals, label=f"PR (AP={avg_prec:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/pr_curve.png", dpi=150)
plt.close()

# Feature Importance
importances = pd.Series(
    pipeline.named_steps["classifier"].feature_importances_,
    index=FEATURE_NAMES,
).sort_values(ascending=True)

plt.figure(figsize=(7, 4))
importances.plot(kind="barh")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("artifacts/feature_importance.png", dpi=150)
plt.close()

logger.info("Saved evaluation plots to artifacts/")

# Save metrics to text file
with open("artifacts/metrics.txt", "w") as f:
    f.write(f"Run timestamp : {datetime.now().isoformat()}\n")
    f.write(f"Train size    : {len(X_train)}\n")
    f.write(f"Test size     : {len(X_test)}\n")
    f.write(f"Churn rate    : {churn_rate:.1f}%\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write(f"\nF1 (weighted)  : {f1:.4f}\n")
    f.write(f"ROC-AUC        : {roc_auc:.4f}\n")
    f.write(f"Avg Precision  : {avg_prec:.4f}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Log everything to MLflow
# ══════════════════════════════════════════════════════════════════════════════
# MLflow "run" = one training experiment.
# Everything inside `with mlflow.start_run()` is recorded automatically.

logger.info("Logging run to MLflow ...")

with mlflow.start_run(run_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
    run_id = run.info.run_id
    logger.info("MLflow run_id: %s", run_id)

    # ── Log hyperparameters ───────────────────────────────────────────────────
    # These are the "settings" we used to train the model
    mlflow.log_params(HYPERPARAMS)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("features", FEATURE_NAMES)
    mlflow.log_param("data_path", data_path)
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("churn_rate_pct", round(churn_rate, 2))

    # ── Log metrics ───────────────────────────────────────────────────────────
    # These are the "scores" that tell us how good the model is
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("avg_precision", avg_prec)

    # ── Log artifacts (files) ─────────────────────────────────────────────────
    mlflow.log_artifact("artifacts/roc_curve.png")
    mlflow.log_artifact("artifacts/pr_curve.png")
    mlflow.log_artifact("artifacts/feature_importance.png")
    mlflow.log_artifact("artifacts/metrics.txt")
    mlflow.log_artifact("data/processed_data.json")     # track data version too

    # ── Log the model itself ──────────────────────────────────────────────────
    # infer_signature tells MLflow what input/output shapes to expect
    signature = infer_signature(X_train, pipeline.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="model",
        signature=signature,
        registered_model_name=MODEL_NAME,   # this goes to Model Registry!
    )

    logger.info("Model registered in MLflow registry as '%s'", MODEL_NAME)

    # ── Tag the run for easy searching ───────────────────────────────────────
    mlflow.set_tag("stage", "training")
    mlflow.set_tag("model_type", "RandomForest")
    mlflow.set_tag("feature_version", "v1")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Promote model to "Staging" in the registry
# ══════════════════════════════════════════════════════════════════════════════
# Model Registry stages:
#   None     → just registered, not reviewed yet
#   Staging  → ready for QA / integration testing
#   Production → live, serving real traffic
#   Archived → old, no longer used

client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Get the latest version of this model
versions = client.get_latest_versions(MODEL_NAME)
if versions:
    latest_version = versions[-1].version
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=False,   # don't auto-archive old versions
    )
    logger.info("Model v%s promoted to 'Staging'", latest_version)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Save model.pkl locally (fallback for the API)
# ══════════════════════════════════════════════════════════════════════════════
joblib.dump(pipeline, "model.pkl")
logger.info("Local model.pkl saved (pipeline includes scaler + classifier)")

logger.info("=" * 60)
logger.info("Training complete!")
logger.info("View results: run  `mlflow ui`  then open http://localhost:5000")
logger.info("=" * 60)