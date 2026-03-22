from datetime import datetime
import logging

logger = logging.getLogger(__name__)

NEGATIVE_KEYWORDS = {"broken", "worst", "terrible", "angry", "frustrated", "useless", "failure", "horrible", "bad", "awful"}
POSITIVE_KEYWORDS = {"resolved", "thanks", "great", "good", "fixed", "helpful", "satisfied", "excellent"}


def compute_sentiment_score(tickets: list) -> float:
    """
    Simple keyword-based sentiment score.
    Returns a value between -1.0 (very negative) and +1.0 (very positive).
    Complaint tickets with negative keywords score lower.
    """
    if not tickets:
        return 0.0

    total_score = 0.0
    for t in tickets:
        description = (t.get("description", "") + " " + t.get("subject", "")).lower()
        words = set(description.split())

        neg_hits = len(words & NEGATIVE_KEYWORDS)
        pos_hits = len(words & POSITIVE_KEYWORDS)

        if t.get("type") == "complaint":
            total_score -= 0.3
        if neg_hits > 0:
            total_score -= 0.2 * neg_hits
        if pos_hits > 0:
            total_score += 0.1 * pos_hits

    score = total_score / len(tickets)
    return round(max(-1.0, min(1.0, score)), 4)


def compute_avg_time_between_tickets(tickets: list) -> float:
    """
    Returns average number of days between consecutive tickets.
    Returns 0.0 if fewer than 2 dated tickets.
    """
    dates = []
    for t in tickets:
        if "date" in t:
            try:
                dates.append(datetime.fromisoformat(t["date"]))
            except Exception as e:
                logger.warning("Skipping ticket with bad date: %s", e)
                continue

    if len(dates) < 2:
        return 0.0

    dates.sort()
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    return round(sum(gaps) / len(gaps), 2)


def extract_features(data: dict) -> list:
    tickets = data.get("tickets", [])
    monthly = data.get("monthly_charges", 0)
    previous = data.get("previous_month_charges", 0)
    contract = data.get("contract_type", "")

    now = datetime.now()

    t7 = t30 = t90 = 0
    complaint = 0
    query = 0

    for t in tickets:
        if "date" in t:
            try:
                d = datetime.fromisoformat(t["date"])
                days = (now - d).days

                if days <= 7:
                    t7 += 1
                if days <= 30:
                    t30 += 1
                if days <= 90:
                    t90 += 1
            except Exception as e:
                logger.warning("Skipping ticket with invalid date: %s", e)
                continue

        if t.get("type") == "complaint":
            complaint += 1
        if t.get("type") == "query":
            query += 1

    charge_change = monthly - previous
    sentiment = compute_sentiment_score(tickets)
    avg_gap = compute_avg_time_between_tickets(tickets)

    return [[
        t7,
        t30,
        t90,
        complaint,
        query,
        charge_change,
        1 if contract.lower() == "month-to-month" else 0,
        sentiment,
        avg_gap,
    ]]


FEATURE_NAMES = [
    "t7",
    "t30",
    "t90",
    "complaint",
    "query",
    "charge_change",
    "contract_monthly",
    "sentiment_score",
    "avg_days_between_tickets",
]
