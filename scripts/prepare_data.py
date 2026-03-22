"""
scripts/prepare_data.py

Converts the Kaggle Telco Customer Churn CSV into the JSON format
expected by scripts/train.py and the API.

Run from project root:
    python scripts/prepare_data.py

Requires:
    data/Telco-Customer-Churn.csv   (download from Kaggle)
"""

import pandas as pd
import json
import random
from datetime import datetime, timedelta

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)

# ── Sample ticket descriptions so sentiment feature has real signal ───────────
COMPLAINT_DESCRIPTIONS = [
    "Service keeps dropping, absolutely terrible",
    "Billing is wrong again, very frustrated",
    "Internet is broken and support is useless",
    "Worst service I have ever had, horrible experience",
    "Speed is awful, nothing works, angry customer",
    "Charged twice this month, awful billing system",
    "Connection fails every night, completely broken",
    "Support did not help at all, very disappointed",
]

QUERY_DESCRIPTIONS = [
    "How do I upgrade my plan",
    "What channels are included in my package",
    "Can I add an extra line",
    "When is my next billing date",
    "How do I set up autopay",
    "I need help resetting my router",
    "What is the process to cancel",
    "Is there a family discount available",
]

# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading Telco-Customer-Churn.csv ...")
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print(f"Loaded {len(df)} customers.")

processed = []

for _, row in df.iterrows():
    monthly = float(row["MonthlyCharges"])

    # Previous charges: vary by up to 20 but never go below 5
    delta = random.uniform(-20, 20)
    previous = round(max(5.0, monthly + delta), 2)

    # Number of tickets: churned customers tend to have more
    churned = str(row.get("Churn", "No")).strip() == "Yes"
    if churned:
        num_tickets = random.randint(1, 7)
    else:
        num_tickets = random.randint(0, 3)

    tickets = []
    for _ in range(num_tickets):
        ticket_type = random.choice(["complaint", "query"])

        # Spread dates across last 90 days so t7 / t30 / t90 all vary
        days_ago = random.randint(1, 90)
        ticket_date = (datetime.now() - timedelta(days=days_ago)).isoformat()

        # Pick a matching description so sentiment scoring has real signal
        if ticket_type == "complaint":
            description = random.choice(COMPLAINT_DESCRIPTIONS)
        else:
            description = random.choice(QUERY_DESCRIPTIONS)

        tickets.append({
            "type": ticket_type,
            "date": ticket_date,
            "description": description,
        })

    processed.append({
        "customerID": row["customerID"],
        "monthly_charges": monthly,
        "previous_month_charges": previous,
        "contract_type": row["Contract"],
        "tickets": tickets,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "data/processed_data.json"
with open(output_path, "w") as f:
    json.dump(processed, f, indent=2)

print(f"Saved {len(processed)} customers to {output_path}")

# ── Quick sanity check ────────────────────────────────────────────────────────
total_tickets = sum(len(c["tickets"]) for c in processed)
churned_count = sum(1 for c in processed if any(t["type"] == "complaint" for t in c["tickets"]))
print(f"Total tickets generated:     {total_tickets}")
print(f"Customers with complaints:   {churned_count}")
print(f"Avg tickets per customer:    {total_tickets / len(processed):.1f}")