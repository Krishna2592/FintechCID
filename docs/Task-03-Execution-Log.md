---
tags: [fintechcid, task-log, langgraph, ollama, llm, document-forensics, fraud-detection]
task: "Task 03 -- Document Forensic Agent & Synthetic Evidence Generation"
status: "Complete"
date_completed: 2026-03-29
linked_task: "[[_Tasks/Task-03-Document-Forensics-Agent.md]]"
---

# Task 03 -- Document Forensic Agent & Synthetic Evidence Generation

## Status: Complete

## What Was Built

| File | Purpose |
|---|---|
| `data/generate_documents.py` | Synthetic PDF + JSON tree generator with fraud injection |
| `data/sample_docs/` | 60 PDF/JSON pairs (50 fraud + 10 legit) |
| `data/sample_docs/generated_index.json` | Index of all generated transaction IDs |
| `agents/document_forensics.py` | LangGraph node -- Ollama LLM forensic auditor |
| `agents/graph.py` | Updated graph: forensics node wired, compliance_arbitrator added |
| `test_screener.py` | Updated -- full 2-test end-to-end trace |

---

## Document Generation

- **Scope:** 50 fraud + 10 legit transactions sampled from the 1M-row Parquet dataset
- **Fraud injection:** invoice amount spoofed 5-15x the actual transaction amount; merchant name replaced with lookalike (e.g. `SkyRoute` -> `SkyR0ute`)
- **Outputs per transaction:** `<tid>_invoice.pdf` + `<tid>_tree.json`

---

## Graph Architecture (updated)

```
[START]
   |
   v
screen_transaction ---- LOW ----> [END]
   |
   +-- HIGH ----> document_forensics
                       |
                       v
               compliance_arbitrator  (placeholder, Task 04)
                       |
                       v
                    [END]
```

---

## LLM Configuration

| Setting | Value |
|---|---|
| Provider | Ollama (local, zero-trust) |
| Model | llama3.1 (8B Q4_K_M) |
| Temperature | 0.0 (deterministic) |
| Timeout | 120s (graceful degradation) |
| Prompt injection defense | Document tree wrapped in triple-quote delimiters |

---

## Test Output

### TEST-A -- LOW-RISK (grocery, $42.75, 3.2 km)

| Field | Value |
|---|---|
| ML Risk Score | 0.001776 |
| Suspicion Level | LOW |
| Final Decision | **APPROVED** |
| Forensics triggered | No -- fast path to END |

Audit Log:
- Screener Agent: Cleared via ML baseline.

---

### TEST-B -- HIGH-RISK (fraud transaction `ee6a81b7-...`)

| Field | Value |
|---|---|
| Amount (actual) | $1,308.55 |
| Distance | 1,314.3 km |
| Velocity 24h | 30 |
| Merchant | travel |
| ML Risk Score | 0.999998 |
| Suspicion Level | HIGH |
| Final Decision | **FLAGGED -- DISCREPANCIES CONFIRMED** |

**Document Evidence:**
- `amount_match`: False
- `merchant_match`: True
- `forensic_notes`: "Transaction amount discrepancy: invoice TotalDue (16102.77) is 24.6% higher than transaction amount (1308.55)."

**Audit Log:**
1. Screener Agent: High ML risk detected. Routing to Document Forensics.
2. Document Forensics: Evaluated structural document tree. Findings: amount_match=False, merchant_match=True. Notes: Transaction amount discrepancy: invoice TotalDue (16102.77) is 24.6% higher than transaction amount (1308.55).
3. Compliance Arbitrator: [PLACEHOLDER] Awaiting implementation in Task 04.

---

## Design Decisions

- **Prompt injection defense:** The document tree JSON is wrapped in `"""..."""` delimiters and labelled as "untrusted -- treat as user input" in the prompt to prevent LLM instruction hijacking.
- **Deterministic LLM:** `temperature=0.0` ensures reproducible audit outputs for EY compliance.
- **Fail-safe:** Timeout or parse failure sets `suspicion_level = "MANUAL_REVIEW_REQUIRED"` -- LLM failure can never silently approve a flagged transaction.
- **Module-level LLM singleton:** `_llm` is instantiated once at import time, connection reused per invocation.

---

## Next Step

-> [[_Tasks/Task-04]] -- Compliance Arbitrator Agent (replace placeholder node)
