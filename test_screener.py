"""
FintechCID -- Screener + Document Forensics Integration Test
===============================================================
Three test cases:
  TEST-A  Fast path: clean transaction  --> APPROVED (no forensics)
  TEST-B  Full path: high-risk mock     --> forensics on a known fraud document
  TEST-C  Known fraud transaction_id    --> full trace through all nodes

Usage:
    python test_screener.py          (from project root: FintechCID/)
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from agents.graph import build_graph
from agents.state import AgentState

# ---------------------------------------------------------------------------
# Load the first fraud transaction_id from the generated index
# ---------------------------------------------------------------------------
_INDEX_PATH = pathlib.Path(__file__).parent / "data" / "sample_docs" / "generated_index.json"
with open(_INDEX_PATH, encoding="utf-8") as _f:
    _INDEX = json.load(_f)
KNOWN_FRAUD_TID = _INDEX["fraud_tids"][0]

# Also load the actual transaction data from parquet for that ID
import pandas as pd
_DF = pd.read_parquet(
    pathlib.Path(__file__).parent / "data" / "raw" / "transactions.parquet"
)
_FRAUD_ROW = _DF[_DF["transaction_id"] == KNOWN_FRAUD_TID].iloc[0]

# ---------------------------------------------------------------------------
# Build graph (model + LLM lazy-loaded on first node execution)
# ---------------------------------------------------------------------------
graph = build_graph()


def run_test(label: str, tx: dict) -> None:
    initial_state: AgentState = {
        "transaction_data":  tx,
        "ml_risk_score":     0.0,
        "suspicion_level":   "",
        "document_evidence": [],
        "final_decision":    "",
        "audit_log":         [],
    }

    result = graph.invoke(initial_state)

    print("=" * 70)
    print(f"  TEST: {label}")
    print("=" * 70)
    print(f"  Transaction ID   : {tx.get('transaction_id')}")
    print(f"  Amount           : ${tx.get('amount', 0):>10,.2f}")
    print(f"  Distance (km)    : {tx.get('distance_from_home_km', 0):>10.1f}")
    print(f"  Velocity 24h     : {tx.get('velocity_24h', 0):>10}")
    print(f"  Merchant         : {tx.get('merchant_category')}")
    print("-" * 70)
    score = result["ml_risk_score"]
    print(f"  ML Risk Score    : {score:.6f}" if score >= 0 else "  ML Risk Score    : N/A (error)")
    print(f"  Suspicion Level  : {result['suspicion_level']}")
    print(f"  Final Decision   : {result['final_decision'] or 'PENDING REVIEW'}")

    if result["document_evidence"]:
        print()
        print("  Document Evidence:")
        for ev in result["document_evidence"]:
            print(f"    amount_match   : {ev.get('amount_match')}")
            print(f"    merchant_match : {ev.get('merchant_match')}")
            print(f"    forensic_notes : {ev.get('forensic_notes')}")

    print()
    print("  Audit Log:")
    for entry in result["audit_log"]:
        print(f"    > {entry}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# TEST-A: Clean low-risk transaction (fast path, no forensics)
# ---------------------------------------------------------------------------
run_test(
    label="TEST-A | LOW-RISK: Morning grocery run near home",
    tx={
        "transaction_id":        "TEST-A-LOW",
        "user_id":               "user_0007",
        "timestamp":             "2026-03-29T09:30:00+00:00",
        "amount":                42.75,
        "merchant_category":     "grocery",
        "distance_from_home_km": 3.2,
        "velocity_24h":          2,
    },
)

# ---------------------------------------------------------------------------
# TEST-B: High-risk mock transaction (forensics with known fraud document)
# ---------------------------------------------------------------------------
run_test(
    label="TEST-B | HIGH-RISK: Late-night overseas purchase (mock, known fraud doc)",
    tx={
        "transaction_id":        KNOWN_FRAUD_TID,
        "user_id":               str(_FRAUD_ROW["user_id"]),
        "timestamp":             str(_FRAUD_ROW["timestamp"]),
        "amount":                float(_FRAUD_ROW["amount"]),
        "merchant_category":     str(_FRAUD_ROW["merchant_category"]),
        "distance_from_home_km": float(_FRAUD_ROW["distance_from_home_km"]),
        "velocity_24h":          int(_FRAUD_ROW["velocity_24h"]),
    },
)
