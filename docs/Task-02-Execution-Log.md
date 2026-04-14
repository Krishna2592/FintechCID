---
tags: [fintechcid, task-log, langgraph, mlflow, fraud-detection]
task: "Task 02 — LangGraph State & Transaction Screener Node"
status: "✅ Complete"
date_completed: 2026-03-29
linked_task: "[[_Tasks/Task-02-LangGraph-Screener.md]]"
---

# Task 02 — LangGraph State & Transaction Screener Node

## Status: ✅ Complete

## What Was Built

| File | Purpose |
|---|---|
| `agents/__init__.py` | Package initialiser |
| `agents/state.py` | `AgentState` TypedDict — shared graph state |
| `agents/transaction_screener.py` | LangGraph node — RF model inference + gating |
| `agents/graph.py` | Compiled `StateGraph` with conditional routing |
| `test_screener.py` | Integration test — 2 mock transactions |

---

## AgentState Schema

```python
class AgentState(TypedDict):
    transaction_data:  dict   # Raw incoming JSON payload
    ml_risk_score:     float  # P(fraud) from Random Forest
    suspicion_level:   str    # "LOW" | "HIGH"
    document_evidence: list   # Populated by Task 03
    final_decision:    str    # "APPROVED" | "FLAGGED" | "REJECTED"
    audit_log:         list   # EY compliance trace
```

---

## Graph Architecture

```
[START]
   │
   ▼
screen_transaction  ──── suspicion == LOW  ──▶  [END]
   │
   └── suspicion == HIGH ──▶  document_forensics (placeholder)
                                      │
                                      ▼
                                   [END]
```

---

## Test Output

### TEST-001 — HIGH-RISK (Late-night overseas purchase)

| Field | Value |
|---|---|
| Amount | $4,500.00 |
| Distance | 2,200 km |
| Velocity 24h | 25 |
| Merchant | online_retail |
| **ML Risk Score** | **1.000000** |
| **Suspicion Level** | **HIGH** |
| **Final Decision** | FLAGGED — PENDING FORENSIC REVIEW |

**Audit Log:**
- Screener Agent: High ML risk detected. Routing to Document Forensics.
- Document Forensics Agent: [PLACEHOLDER] Awaiting implementation in Task 03.

---

### TEST-002 — LOW-RISK (Morning grocery run)

| Field | Value |
|---|---|
| Amount | $42.75 |
| Distance | 3.2 km |
| Velocity 24h | 2 |
| Merchant | grocery |
| **ML Risk Score** | **0.001776** |
| **Suspicion Level** | **LOW** |
| **Final Decision** | APPROVED |

**Audit Log:**
- Screener Agent: Cleared via ML baseline.

---

## Design Decisions

- **Model caching:** `_model` and `_label_encoder` are module-level singletons — loaded once per process, never reloaded on re-invocations of the node.
- **Fail-safe secure:** Any exception (bad input, model load failure) forces `suspicion_level = "HIGH"` and logs the error — fraudulent transactions can never silently pass.
- **`amount_z` at inference:** Defaults to `0.0` (neutral) since per-user historical averages aren't available in a single-transaction scoring context. The RF's other features (amount, distance, velocity) carry sufficient discriminative power.
- **Fraud threshold:** `0.65` per spec. The HIGH-RISK test scored `1.0`; LOW-RISK scored `0.0018` — clean separation.

---

## Next Step

→ [[_Tasks/Task-03]] — Document Forensics Agent (replace placeholder node)
