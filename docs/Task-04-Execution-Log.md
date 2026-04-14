---
tags: [fintechcid, task-log, langgraph, hitl, streamlit, fastapi, compliance]
task: "Task 04 -- HITL Auditor Dashboard"
status: "Complete"
date_completed: 2026-03-29
linked_task: "[[_Tasks/Task-04-HITL-Auditor-Dashboard.md]]"
---

# Task 04 -- Human-in-the-Loop (HITL) Auditor Dashboard

## Status: Complete

## What Was Built

| File | Purpose |
|---|---|
| `agents/state.py` | Updated -- added `auditor_decision`, `auditor_comments` fields |
| `agents/compliance_arbitrator.py` | Final ruling node -- generates signed Audit Package |
| `agents/graph.py` | Updated -- MemorySaver checkpointer + `interrupt_before=["compliance_arbitrator"]` |
| `api/main.py` | FastAPI backend -- manages graph threads + HITL lifecycle |
| `frontend/app.py` | Streamlit auditor dashboard -- case queue, PDF viewer, decision UI |
| `run_all.sh` | One-command startup for all three services |

---

## Graph Architecture (final)

```
[START]
   |
   v
screen_transaction ---- LOW ----> [END]
   |
   +-- HIGH ----> document_forensics
                       |
                       v
              [HITL BREAKPOINT] <-- MemorySaver pauses here
                       |
                       v
               compliance_arbitrator  <-- runs after auditor input
                       |
                       v
                    [END]
```

---

## FastAPI Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/transactions/submit` | Submit transaction; runs to breakpoint |
| GET | `/api/transactions/pending` | List cases awaiting review |
| GET | `/api/transactions/{thread_id}` | Get full AgentState snapshot |
| POST | `/api/transactions/{thread_id}/resume` | Inject auditor decision; resume graph |
| GET | `/api/resolved` | List completed cases |
| GET | `/health` | Service health check |

---

## Streamlit Dashboard Features

- **Case Queue** (sidebar): Live list of pending threads with amount, merchant, risk score
- **Transaction Panel**: Full tx data + ML risk badge + metric
- **Forensic Evidence**: amount_match / merchant_match metrics + LLM notes
- **PDF Viewer**: Base64-embedded invoice PDF iframe with download button
- **Audit Log**: Full append-only trace from all agents
- **Decision Interface**: Approve / Reject buttons + comments textarea
- **Resume Logic**: Calls `POST /api/transactions/{thread_id}/resume` → shows final ruling + balloons

---

## Live Test Output (curl)

### Submit → PAUSED_AT_BREAKPOINT

```json
{
  "thread_id": "e6bfb3b3-4812-4eb6-9d90-6c2d8f1ae57e",
  "status": "PAUSED_AT_BREAKPOINT",
  "suspicion_level": "HIGH",
  "ml_risk_score": 0.999998
}
```

### Pending Queue

```json
{
  "pending_cases": [{
    "transaction_id": "ee6a81b7-...",
    "amount": 1308.55,
    "merchant": "travel",
    "ml_risk_score": 0.999998,
    "next_node": ["compliance_arbitrator"]
  }]
}
```

### Resume → REJECTED BY AUDITOR

```json
{
  "status": "RESOLVED",
  "final_decision": "REJECTED BY AUDITOR",
  "audit_log": [
    "Screener Agent: High ML risk detected. Routing to Document Forensics.",
    "Document Forensics: ... amount_match=False, merchant_match=True.",
    "Compliance Arbitrator: Auditor REJECTED transaction ee6a81b7-...",
    "Compliance Arbitrator: Audit Package generated. Final ruling: REJECTED BY AUDITOR."
  ]
}
```

---

## Dashboard Description

The Streamlit app at `http://localhost:8501` presents:

- **Left sidebar:** A form to submit transactions (pre-filled with known fraud case) and a radio list of pending cases showing truncated TID, amount, and risk score.
- **Top header:** Transaction ID + colour-coded risk badge (red=HIGH, green=LOW) + ML score metric.
- **Left column:** Transaction data table, forensic evidence metrics (PASS/FAIL tiles for amount/merchant match), LLM forensic notes in a warning box, and the full audit log numbered list.
- **Right column:** The synthetic invoice PDF embedded as a scrollable iframe with a Download button underneath. Below the PDF, the decision interface with a comments textarea and APPROVE/REJECT buttons.
- **Post-decision:** Final ruling displayed in a green/red banner, updated audit log, and Streamlit balloons animation.

---

## How to Start All Services

```bash
# From project root: FintechCID/
bash run_all.sh
```

| Service | URL |
|---|---|
| FastAPI REST API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## Next Step

-> Task 05 (if defined) -- or production hardening / database-backed checkpointer
