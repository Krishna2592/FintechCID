"""
FintechCID — Synthetic Transaction Data Engine
==================================================
Simulates a scalable data lake ingestion process by generating
10,000 synthetic financial transaction records with a strict 5%
fraud rate and logically injected fraud patterns.

Architecture note
-----------------
This script is written with a PySpark-ready design (explicit schema,
columnar parquet output, partition-friendly patterns).  To run on a
real Spark cluster simply wrap the DataFrame construction in a
SparkSession and replace ``pd.DataFrame`` with ``spark.createDataFrame``.
The current implementation uses pandas + pyarrow because the target
environment has Java 11, while PySpark ≥ 3.5 requires Java 17.
Upgrading Java to 17+ will make the PySpark switch seamless.

Output: data/raw/transactions.parquet
"""

import uuid
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_RECORDS = 1_000_000
FRAUD_RATE  = 0.05          # Strict 5% fraud rate → 500 fraud rows
NUM_USERS   = 500           # Pool of user IDs

MERCHANT_CATEGORIES = [
    "grocery", "restaurant", "online_retail", "travel",
    "gas_station", "electronics", "healthcare", "entertainment",
]

RANDOM_SEED = 42
OUTPUT_PATH = "data/raw/transactions.parquet"

random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Row generation helpers
# ---------------------------------------------------------------------------

def _random_timestamp() -> datetime:
    """Return a timezone-aware UTC datetime within the last 90 days."""
    now = datetime.now(timezone.utc)
    offset = timedelta(
        days=random.randint(0, 90),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    return now - offset


def _build_record(is_fraud: int) -> dict:
    """
    Build one synthetic transaction record.

    Fraud pattern logic
    -------------------
    Fraudulent transactions are biased toward:
      • High velocity_24h      (≥ 8 transactions in the past 24 h)
      • Large distance_from_home_km  (≥ 150 km away)
      • High amount            (≥ $300)
      • Remote / travel merchant categories

    Legitimate transactions use much lower ranges for all three
    dimensions, which trains the downstream fraud model to learn
    meaningful feature interactions.
    """
    user_id = f"user_{random.randint(1, NUM_USERS):04d}"

    if is_fraud:
        amount   = round(float(rng.uniform(300,  5_000)), 2)
        distance = round(float(rng.uniform(150,  3_000)), 2)
        velocity = int(rng.integers(8, 31))
        merchant = random.choice(["online_retail", "travel", "electronics"])
    else:
        amount   = round(float(rng.uniform(1,    400)), 2)
        distance = round(float(rng.uniform(0,    150)), 2)
        velocity = int(rng.integers(1, 8))
        merchant = random.choice(MERCHANT_CATEGORIES)

    return {
        "transaction_id":        str(uuid.uuid4()),
        "user_id":               user_id,
        "timestamp":             _random_timestamp(),
        "amount":                amount,
        "merchant_category":     merchant,
        "distance_from_home_km": distance,
        "velocity_24h":          velocity,
        "is_fraud":              is_fraud,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Build label list with strict 5% fraud rate -----------------------
    num_fraud = int(NUM_RECORDS * FRAUD_RATE)   # 500
    num_legit = NUM_RECORDS - num_fraud          # 9 500
    labels = [1] * num_fraud + [0] * num_legit
    random.shuffle(labels)

    # --- Generate records --------------------------------------------------
    print("[Data Engine] Generating records …")
    records = [_build_record(label) for label in labels]

    # --- Build DataFrame with explicit dtypes (mirrors Spark schema) -------
    df = pd.DataFrame(records).astype({
        "transaction_id":        "string",
        "user_id":               "string",
        "amount":                "float32",
        "merchant_category":     "string",
        "distance_from_home_km": "float32",
        "velocity_24h":          "int32",
        "is_fraud":              "int32",
    })
    # timestamp column is already datetime[ns, UTC] from _random_timestamp()

    # --- Verify fraud rate -------------------------------------------------
    fraud_count  = int(df["is_fraud"].sum())
    total_count  = len(df)
    actual_rate  = fraud_count / total_count
    print(f"[Data Engine] Total records : {total_count:,}")
    print(f"[Data Engine] Fraud records : {fraud_count:,}  ({actual_rate:.1%})")

    # --- Write parquet (snappy-compressed, overwrite-safe) ----------------
    df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
    print(f"[Data Engine] Parquet written -> {OUTPUT_PATH}")
    print("[Data Engine] Done.")


if __name__ == "__main__":
    main()
