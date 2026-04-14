"""
scripts/retrain.py

Automated Retraining Script

What problem does this solve?
  Customer behaviour changes over time. A model trained 3 months ago may no
  longer be accurate because:
    - New customers have different patterns
    - Pricing has changed
    - Seasonal effects

  This script checks for drift, and if found, retrains the model automatically.

How to use:
  Manual     : python scripts/retrain.py
  Scheduled  : Add to Windows Task Scheduler or Linux cron
  CI/CD      : Called by GitHub Actions on a schedule (see .github/workflows/ct.yml)

What it does:
  1. Run drift monitoring (monitor.py)
  2. If drift detected (or --force flag used): retrain
  3. After retraining: promote new model to Staging in MLflow registry
  4. Write a retrain log
"""

import subprocess
import sys
import os
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("artifacts/monitoring", exist_ok=True)

RETRAIN_LOG = "artifacts/monitoring/retrain_log.json"


def load_retrain_log() -> list:
    """Load history of past retraining runs."""
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            return json.load(f)
    return []


def save_retrain_log(log: list):
    """Save retrain history."""
    with open(RETRAIN_LOG, "w") as f:
        json.dump(log, f, indent=2)


def run_script(script_path: str) -> int:
    """Run a Python script and return its exit code."""
    logger.info("Running: %s", script_path)
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,   # show output in terminal
    )
    return result.returncode


def check_drift() -> bool:
    """
    Run drift monitoring and return True if retraining is needed.
    monitor.py exits with code 1 if drift is detected.
    """
    exit_code = run_script("scripts/monitor.py")
    drift_detected = exit_code != 0
    logger.info("Drift check result: %s", "DRIFT DETECTED" if drift_detected else "No drift")
    return drift_detected


def retrain_model():
    """Run the full training pipeline."""
    logger.info("Starting model retraining ...")
    exit_code = run_script("scripts/train.py")
    if exit_code != 0:
        raise RuntimeError(f"Training script failed with exit code {exit_code}")
    logger.info("Retraining complete.")


def main():
    # Check if --force flag was passed (bypasses drift check)
    force_retrain = "--force" in sys.argv
    timestamp = datetime.now().isoformat()

    logger.info("=" * 60)
    logger.info("Automated Retraining Check — %s", timestamp)
    logger.info("Force mode: %s", force_retrain)
    logger.info("=" * 60)

    # Decide whether to retrain
    should_retrain = force_retrain or check_drift()

    log_entry = {
        "timestamp": timestamp,
        "force_retrain": force_retrain,
        "drift_detected": not force_retrain and should_retrain,
        "retrained": False,
        "outcome": "skipped",
    }

    if should_retrain:
        try:
            retrain_model()
            log_entry["retrained"] = True
            log_entry["outcome"] = "success"
            logger.info("✅ Retraining succeeded")
        except Exception as e:
            log_entry["outcome"] = f"error: {str(e)}"
            logger.error("❌ Retraining failed: %s", e)
            sys.exit(1)
    else:
        logger.info("No retraining needed — model is healthy.")

    # Save log entry
    log = load_retrain_log()
    log.append(log_entry)
    save_retrain_log(log)
    logger.info("Retrain log updated: %s", RETRAIN_LOG)


if __name__ == "__main__":
    main()