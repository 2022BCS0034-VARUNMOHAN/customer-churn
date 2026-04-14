"""
scripts/monitor.py

What is monitoring in MLOps?

After you deploy a model, real-world data keeps changing:
  - Customer behaviour shifts  → "concept drift"
  - Input data distribution changes → "feature drift / data drift"
  - Model accuracy decays silently

This script:
  1. Loads the training data (baseline distribution)
  2. Loads recent inference logs (what the model is seeing NOW)
  3. Computes drift scores for each feature
  4. Flags if any feature has drifted beyond a threshold
  5. Saves a monitoring report

Run from project root:
    python scripts/monitor.py

In production this would run as a scheduled job (e.g., every night at 2am).
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.features import extract_features, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("artifacts/monitoring", exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Population Stability Index (PSI)
# ══════════════════════════════════════════════════════════════════════════════
# PSI measures how much a feature's distribution has shifted.
# It compares:
#   - "expected" = training data distribution (baseline)
#   - "actual"   = recent inference data distribution
#
# PSI thresholds:
#   PSI < 0.1   → No significant change    ✅
#   PSI 0.1–0.2 → Moderate change          ⚠️  Monitor
#   PSI > 0.2   → Significant drift        🚨  Retrain!

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Compute Population Stability Index between two distributions.

    Args:
        expected : 1-D array — training/baseline values
        actual   : 1-D array — recent/current values
        buckets  : number of bins to split the distribution into

    Returns:
        PSI value (float). Higher = more drift.
    """
    # Create bins based on the expected (training) distribution
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percents = np.percentile(expected, breakpoints)

    # Remove duplicates (happens when data has many identical values)
    expected_percents = np.unique(expected_percents)

    # Count how many values fall into each bin
    def _bucket_counts(data, breaks):
        counts = np.zeros(len(breaks) - 1)
        for i in range(len(breaks) - 1):
            mask = (data >= breaks[i]) & (data < breaks[i + 1])
            counts[i] = mask.sum()
        # Last bucket: inclusive of max
        counts[-1] += (data == breaks[-1]).sum()
        return counts

    if len(expected_percents) < 2:
        return 0.0   # can't compute PSI with no variation

    exp_counts = _bucket_counts(expected, expected_percents)
    act_counts = _bucket_counts(actual, expected_percents)

    # Convert to proportions (avoid division by zero)
    exp_pct = np.where(exp_counts == 0, 0.0001, exp_counts / len(expected))
    act_pct = np.where(act_counts == 0, 0.0001, act_counts / len(actual))

    # PSI formula: sum( (actual% - expected%) * ln(actual% / expected%) )
    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


# ══════════════════════════════════════════════════════════════════════════════
#  Load baseline (training) features
# ══════════════════════════════════════════════════════════════════════════════
def load_baseline_features(data_path: str = "data/processed_data.json") -> pd.DataFrame:
    """Extract features from the training dataset (our baseline)."""
    logger.info("Loading baseline data from %s", data_path)
    with open(data_path) as f:
        data = json.load(f)

    rows = [extract_features(c)[0] for c in data]
    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    logger.info("Baseline: %d records", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Simulate recent inference data
# ══════════════════════════════════════════════════════════════════════════════
def simulate_recent_data(baseline_df: pd.DataFrame, drift_factor: float = 0.3) -> pd.DataFrame:
    """
    In a real system, you would load this from your API logs or a database.
    Here we simulate "recent" data by adding controlled drift to the baseline.

    drift_factor = 0.3 means features shift by up to 30% of their std dev.
    This simulates real-world distribution shift.
    """
    logger.info("Simulating recent inference data (drift_factor=%.1f) ...", drift_factor)
    recent = baseline_df.copy()
    rng = np.random.RandomState(99)

    for col in FEATURE_NAMES:
        std = baseline_df[col].std()
        # Add random noise + a small directional shift to simulate drift
        noise = rng.normal(loc=drift_factor * std, scale=0.1 * std, size=len(recent))
        recent[col] = (recent[col] + noise).clip(lower=0)

    return recent


# ══════════════════════════════════════════════════════════════════════════════
#  Main monitoring logic
# ══════════════════════════════════════════════════════════════════════════════
def run_monitoring():
    timestamp = datetime.now().isoformat()
    logger.info("=" * 60)
    logger.info("Drift Monitoring Report — %s", timestamp)
    logger.info("=" * 60)

    # Load baseline features
    baseline_df = load_baseline_features()

    # Load (or simulate) recent inference data
    recent_df = simulate_recent_data(baseline_df, drift_factor=0.25)

    # ── Compute PSI for each feature ──────────────────────────────────────────
    PSI_WARN_THRESHOLD  = 0.1    # yellow flag
    PSI_ALERT_THRESHOLD = 0.2    # red flag → trigger retraining

    results = []
    any_alert = False

    for feature in FEATURE_NAMES:
        psi = compute_psi(
            expected=baseline_df[feature].values,
            actual=recent_df[feature].values,
        )

        # Determine status
        if psi >= PSI_ALERT_THRESHOLD:
            status = "🚨 DRIFT ALERT"
            any_alert = True
        elif psi >= PSI_WARN_THRESHOLD:
            status = "⚠️  WARNING"
        else:
            status = "✅ OK"

        results.append({
            "feature": feature,
            "psi": round(psi, 4),
            "status": status,
            "baseline_mean": round(float(baseline_df[feature].mean()), 4),
            "recent_mean": round(float(recent_df[feature].mean()), 4),
            "baseline_std": round(float(baseline_df[feature].std()), 4),
            "recent_std": round(float(recent_df[feature].std()), 4),
        })

        logger.info("%-30s PSI=%.4f  %s", feature, psi, status)

    # ── Build report ──────────────────────────────────────────────────────────
    report = {
        "timestamp": timestamp,
        "baseline_records": len(baseline_df),
        "recent_records": len(recent_df),
        "psi_warn_threshold": PSI_WARN_THRESHOLD,
        "psi_alert_threshold": PSI_ALERT_THRESHOLD,
        "retrain_recommended": any_alert,
        "feature_drift": results,
    }

    # Save JSON report
    report_path = "artifacts/monitoring/drift_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Drift report saved to %s", report_path)

    # Save human-readable summary
    summary_path = "artifacts/monitoring/drift_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Drift Monitoring Report\n")
        f.write(f"Generated: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Feature':<35} {'PSI':>8}  {'Status'}\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"{r['feature']:<35} {r['psi']:>8.4f}  {r['status']}\n")
        f.write("\n")
        if any_alert:
            f.write("⚠️  RECOMMENDATION: Retrain the model — significant drift detected.\n")
        else:
            f.write("✅  Model is healthy — no significant drift detected.\n")

    logger.info("Summary saved to %s", summary_path)

    # ── Exit code signals to CI/CD whether to trigger retraining ─────────────
    if any_alert:
        logger.warning("DRIFT ALERT: Retraining is recommended!")
        return 1   # non-zero exit = alert
    else:
        logger.info("No significant drift. Model is healthy.")
        return 0


if __name__ == "__main__":
    exit_code = run_monitoring()
    sys.exit(exit_code)