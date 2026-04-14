#rules.py
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def calculate_risk(data: dict) -> str:
    """
    Rule-based churn risk engine.
    Returns: 'HIGH', 'MEDIUM', or 'LOW'
    """
    tickets = data.get("tickets", [])
    contract = data.get("contract_type", "")
    monthly = data.get("monthly_charges", 0)
    previous = data.get("previous_month_charges", 0)

    charge_increase = monthly - previous

    recent_tickets = 0
    for t in tickets:
        if "date" in t:
            try:
                ticket_date = datetime.fromisoformat(t["date"])
                if (datetime.now() - ticket_date).days <= 30:
                    recent_tickets += 1
            except Exception as e:
                logger.warning("Skipping ticket with invalid date: %s", e)
                continue

    # PRIORITY 1: Month-to-month + complaint ticket → HIGH
    if contract.lower() == "month-to-month":
        for t in tickets:
            if t.get("type") == "complaint":
                return "HIGH"

    # PRIORITY 2: More than 5 tickets in last 30 days → HIGH
    if recent_tickets > 5:
        return "HIGH"

    # PRIORITY 3: Charge increase + at least 3 recent tickets → MEDIUM
    if charge_increase > 0 and recent_tickets >= 3:
        return "MEDIUM"

    return "LOW"
