"""
app/features.py

THIS IS THE MOST IMPORTANT FILE IN THE ML SYSTEM.

Why? Because the SAME feature logic must run during:
  1. Training  (scripts/train.py)
  2. Inference (app/main.py  → /predict-risk endpoint)

If these two ever differ, the model gets garbage inputs at inference time.
This file is the single source of truth — both scripts import from here.

Features we build per customer:
  - ticket_7d        : how many tickets in the last 7 days
  - ticket_30d       : how many tickets in the last 30 days
  - ticket_90d       : how many tickets in the last 90 days
  - sentiment_score  : average negativity of ticket descriptions (-1=very negative, +1=very positive)
  - complaint_count  : total complaint-type tickets ever
  - query_count      : total query-type tickets ever
  - avg_days_between : average number of days between consecutive tickets (0 if <2 tickets)
  - charge_change    : monthly_charges minus previous_month_charges
"""

from datetime import datetime

# ── Tiny sentiment word lists ─────────────────────────────────────────────────
# We don't use a heavy NLP library (no transformers needed).
# Instead we check for negative/positive keywords in ticket descriptions.
# Score per ticket: -1 (all negative words), 0 (neutral), +1 (all positive words)

NEGATIVE_WORDS = {
    "terrible", "awful", "broken", "frustrated", "angry", "useless",
    "horrible", "wrong", "worst", "disappointed", "fails", "fail",
    "broken", "bad", "hate", "angry", "outrage",
}

POSITIVE_WORDS = {
    "great", "good", "happy", "help", "upgrade", "thanks",
    "discount", "easy", "nice", "excellent", "smooth",
}


def _sentiment(text: str) -> float:
    """
    Returns a sentiment score for one piece of text.
    Score = (positive_hits - negative_hits) / total_words
    Clamped to [-1, +1]. Returns 0.0 for empty text.
    """
    if not text:
        return 0.0
    words = text.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    score = (pos - neg) / len(words)
    return max(-1.0, min(1.0, score))   # clamp


# This list defines the ORDER of features.
# The model learns on these names, and inference must supply them in the same order.
FEATURE_NAMES = [
    "ticket_7d",
    "ticket_30d",
    "ticket_90d",
    "sentiment_score",
    "complaint_count",
    "query_count",
    "avg_days_between_tickets",
    "charge_change",
]


def extract_features(customer: dict) -> list:
    """
    Turn one customer dict into a list of numeric features.

    Args:
        customer: dict with keys:
            - tickets              : list of {type, date, description}
            - monthly_charges      : float
            - previous_month_charges: float

    Returns:
        [[f1, f2, ..., f8]]   ← a list containing ONE inner list
        (sklearn expects a 2-D structure even for a single row)
    """
    now = datetime.now()
    tickets = customer.get("tickets", [])
    monthly = float(customer.get("monthly_charges", 0))
    previous = float(customer.get("previous_month_charges", 0))

    # ── Ticket window counts ──────────────────────────────────────────────────
    ticket_7d = ticket_30d = ticket_90d = 0
    complaint_count = query_count = 0
    sentiment_scores = []
    ticket_dates = []

    for t in tickets:
        # Count complaint / query types
        ttype = t.get("type", "")
        if ttype == "complaint":
            complaint_count += 1
        elif ttype == "query":
            query_count += 1

        # Sentiment of the description
        desc = t.get("description", "")
        sentiment_scores.append(_sentiment(desc))

        # Date-based window counts
        raw_date = t.get("date", "")
        if raw_date:
            try:
                d = datetime.fromisoformat(raw_date)
                ticket_dates.append(d)
                days_ago = (now - d).days
                if days_ago <= 7:
                    ticket_7d += 1
                if days_ago <= 30:
                    ticket_30d += 1
                if days_ago <= 90:
                    ticket_90d += 1
            except ValueError:
                pass   # skip bad dates

    # ── Average sentiment ─────────────────────────────────────────────────────
    sentiment_score = (
        sum(sentiment_scores) / len(sentiment_scores)
        if sentiment_scores else 0.0
    )

    # ── Average days between consecutive tickets ──────────────────────────────
    if len(ticket_dates) >= 2:
        ticket_dates_sorted = sorted(ticket_dates)
        gaps = [
            (ticket_dates_sorted[i + 1] - ticket_dates_sorted[i]).days
            for i in range(len(ticket_dates_sorted) - 1)
        ]
        avg_days_between = sum(gaps) / len(gaps)
    else:
        avg_days_between = 0.0

    # ── Charge change ─────────────────────────────────────────────────────────
    charge_change = monthly - previous

    # ── Assemble feature vector ───────────────────────────────────────────────
    features = [
        ticket_7d,
        ticket_30d,
        ticket_90d,
        round(sentiment_score, 4),
        complaint_count,
        query_count,
        round(avg_days_between, 2),
        round(charge_change, 2),
    ]

    # Return as [[...]] so sklearn is happy (it expects 2-D input)
    return [features]