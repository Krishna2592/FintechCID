"""
FintechCID -- FastAPI Backend
==================================
Manages the LangGraph HITL workflow. Holds the compiled graph and
MemorySaver checkpointer in-process so thread state persists across
HTTP requests.

Endpoints
---------
POST /api/transactions/submit          Submit a transaction; runs to HITL breakpoint
GET  /api/transactions/pending         List cases waiting for auditor review
GET  /api/transactions/{thread_id}     Get full AgentState for a thread
POST /api/transactions/{thread_id}/resume  Inject auditor decision; resume graph
POST /api/predict                      Thin inference endpoint for benchmarking
"""

import io
import json
import os
import pathlib
import re
import sys
import uuid
from typing import Any

import pypdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is importable
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from agents.graph import build_graph
from agents.state import AgentState
from agents.transaction_screener import (
    _build_feature_row,
    _infer_triton,
    _load_artifacts,
    _USE_TRITON,
    FRAUD_THRESHOLD,
)

# ---------------------------------------------------------------------------
# App + graph singletons
# ---------------------------------------------------------------------------
app = FastAPI(title="FintechCID -- Fraud Investigation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single compiled graph with HITL breakpoint -- shared across all requests
_graph = build_graph(with_hitl=True)

# In-memory case registry: thread_id -> summary dict
# Append-only during submission; removed only on final resolution
_pending_cases: dict[str, dict] = {}
_resolved_cases: dict[str, dict] = {}

_DOCS_DIR = _ROOT / "data" / "sample_docs"
_DOCS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class TransactionPayload(BaseModel):
    transaction_id: str | None = None
    user_id: str = "user_unknown"
    timestamp: str = "2026-01-01T00:00:00+00:00"
    amount: float = 0.0
    merchant_category: str = "grocery"
    distance_from_home_km: float = 0.0
    velocity_24h: int = 1
    amount_z: float = 0.0


class AuditorDecision(BaseModel):
    decision: str        # "APPROVE" or "REJECT"
    comments: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_serialize(obj: Any) -> Any:
    """Recursively make a state dict JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _state_to_dict(state_snapshot) -> dict:
    """Extract values from a LangGraph StateSnapshot into a plain dict."""
    return _safe_serialize(dict(state_snapshot.values))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/transactions/submit")
def submit_transaction(payload: TransactionPayload):
    """
    Submit a transaction into the LangGraph pipeline.
    The graph runs until the HITL breakpoint (before compliance_arbitrator).
    Returns the thread_id and the paused state.
    """
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    tx = payload.model_dump()
    if not tx.get("transaction_id"):
        tx["transaction_id"] = thread_id

    initial_state: AgentState = {
        "transaction_data":  tx,
        "ml_risk_score":     0.0,
        "suspicion_level":   "",
        "document_evidence": [],
        "final_decision":    "",
        "audit_log":         [],
        "auditor_decision":  "",
        "auditor_comments":  "",
    }

    try:
        _graph.invoke(initial_state, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    snapshot    = _graph.get_state(config)
    state_dict  = _state_to_dict(snapshot)
    suspicion   = state_dict.get("suspicion_level", "")
    score       = state_dict.get("ml_risk_score", 0.0)

    # Any case paused at the compliance_arbitrator breakpoint needs human review
    # This includes both HIGH and MANUAL_REVIEW_REQUIRED suspicion levels
    paused_at_breakpoint = "compliance_arbitrator" in (snapshot.next or [])
    if paused_at_breakpoint:
        _pending_cases[thread_id] = {
            "thread_id":       thread_id,
            "transaction_id":  tx["transaction_id"],
            "amount":          tx.get("amount"),
            "merchant":        tx.get("merchant_category"),
            "ml_risk_score":   score,
            "suspicion_level": suspicion,
            "next_node":       list(snapshot.next) if snapshot.next else [],
        }
        status = "PAUSED_AT_BREAKPOINT"
    else:
        status = "RESOLVED_AUTOMATICALLY"

    return {
        "thread_id":       thread_id,
        "status":          status,
        "suspicion_level": suspicion,
        "ml_risk_score":   score,
        "state":           state_dict,
    }


@app.get("/api/transactions/pending")
def list_pending():
    """Return all cases waiting for auditor review."""
    return {"pending_cases": list(_pending_cases.values())}


@app.get("/api/transactions/{thread_id}")
def get_thread_state(thread_id: str):
    """Return the full AgentState snapshot for a given thread."""
    config   = {"configurable": {"thread_id": thread_id}}
    snapshot = _graph.get_state(config)
    if not snapshot or not snapshot.values:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id!r} not found.")
    return {
        "thread_id": thread_id,
        "next_node": list(snapshot.next) if snapshot.next else [],
        "state":     _state_to_dict(snapshot),
    }


@app.post("/api/transactions/{thread_id}/resume")
def resume_thread(thread_id: str, body: AuditorDecision):
    """
    Inject auditor decision into the paused thread and resume the graph.
    The compliance_arbitrator node runs and produces the final ruling.
    """
    if thread_id not in _pending_cases:
        raise HTTPException(
            status_code=404,
            detail=f"Thread {thread_id!r} is not in the pending queue.",
        )

    decision = body.decision.strip().upper()
    if decision not in ("APPROVE", "REJECT"):
        raise HTTPException(
            status_code=400,
            detail="decision must be 'APPROVE' or 'REJECT'.",
        )

    config = {"configurable": {"thread_id": thread_id}}

    # Inject auditor input -- audit_log is append-only; new entries added by node
    try:
        _graph.update_state(
            config,
            {
                "auditor_decision": decision,
                "auditor_comments": body.comments or "No comments provided.",
            },
        )
        final_state = _graph.invoke(None, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Move from pending to resolved
    case = _pending_cases.pop(thread_id, {})
    case["final_decision"] = (final_state or {}).get("final_decision", "UNKNOWN")
    _resolved_cases[thread_id] = case

    return {
        "thread_id":      thread_id,
        "status":         "RESOLVED",
        "final_decision": case["final_decision"],
        "state":          _safe_serialize(final_state or {}),
    }


@app.get("/api/resolved")
def list_resolved():
    """Return all resolved cases."""
    return {"resolved_cases": list(_resolved_cases.values())}


@app.get("/health")
def health():
    backend = "triton" if _USE_TRITON else "sklearn"
    return {"status": "ok", "service": "FintechCID API", "inference_backend": backend}


# ---------------------------------------------------------------------------
# Benchmark endpoint -- thin inference only, bypasses LangGraph
# Used by triton_serving/benchmark.py to isolate inference latency
# ---------------------------------------------------------------------------

@app.post("/api/predict")
def predict(payload: TransactionPayload):
    """
    Direct inference endpoint -- skips the full HITL pipeline.
    Returns just the ML risk score and suspicion level.
    Useful for load testing and comparing FastAPI vs Triton throughput.
    """
    try:
        _load_artifacts()
        tx       = payload.model_dump()
        features = _build_feature_row(tx)

        if _USE_TRITON:
            proba = _infer_triton(features)
        else:
            from agents.transaction_screener import _model
            proba = float(_model.predict_proba(features)[0][1])

        return {
            "ml_risk_score":   round(proba, 6),
            "suspicion_level": "HIGH" if proba >= FRAUD_THRESHOLD else "LOW",
            "backend":         "triton" if _USE_TRITON else "sklearn",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Evidence Upload Endpoint (Task 05)
# ---------------------------------------------------------------------------

def _extract_pdf_text(content: bytes) -> str:
    """Extract all text from PDF bytes using pypdf."""
    reader = pypdf.PdfReader(io.BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _build_tree_from_pdf(text: str, transaction_id: str,
                          amount: float, merchant_category: str) -> dict:
    """
    Build a Vectorless RAG JSON tree from extracted PDF text.
    Uses regex to recover invoice fields; falls back to provided form values.
    """
    # Try to extract amount from text (e.g. "TOTAL DUE  $14,909.97")
    doc_amount = amount
    m = re.search(r"TOTAL DUE[:\s]+\$?([\d,]+\.?\d*)", text, re.IGNORECASE)
    if m:
        try:
            doc_amount = float(m.group(1).replace(",", ""))
        except ValueError:
            pass

    # Try to extract merchant display name
    merchant_name = merchant_category
    m = re.search(r"(?:PROJECT AEGIS.*?INVOICE\n)(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        if candidate and len(candidate) < 80:
            merchant_name = candidate

    # Try to extract invoice number
    invoice_no = f"INV-{transaction_id[:8].upper()}"
    m = re.search(r"Invoice No[:\s]+(\S+)", text, re.IGNORECASE)
    if m:
        invoice_no = m.group(1).strip()

    return {
        "DocumentRoot": {
            "Header": {
                "InvoiceNumber":  invoice_no,
                "TransactionID":  transaction_id,
                "DocumentType":   "Commercial Invoice (Uploaded)",
                "ExtractionMethod": "pypdf text extraction",
            },
            "MerchantInfo": {
                "DisplayName": merchant_name,
                "Category":    merchant_category,
            },
            "Body": {
                "LineItems": [{"Description": merchant_name, "Quantity": 1,
                               "UnitPrice": round(doc_amount, 2)}],
                "Subtotal":  round(doc_amount, 2),
                "Tax":       round(doc_amount * 0.08, 2),
                "TotalDue":  round(doc_amount * 1.08, 2),
            },
            "Metadata": {
                "source":              "uploaded_pdf",
                "actual_tx_amount":    amount,
                "discrepancy_injected": False,
            },
        }
    }


@app.post("/api/v1/upload-evidence")
async def upload_evidence(
    file: UploadFile = File(...),
    transaction_id: str = Form(...),
    amount: float = Form(0.0),
    merchant_category: str = Form("grocery"),
    distance_from_home_km: float = Form(0.0),
    velocity_24h: int = Form(1),
    timestamp: str = Form("2026-01-01T00:00:00+00:00"),
):
    """
    Accept a PDF invoice upload, parse it into a JSON tree, and run the
    full LangGraph pipeline against the associated transaction.

    Security controls
    -----------------
    - MIME type must be application/pdf
    - PDF magic bytes (%PDF) validated before processing
    - File contents never executed; only text extracted
    """
    # --- MIME validation ----------------------------------------------------
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Only PDF files are accepted.",
        )

    content = await file.read()

    # --- Magic bytes validation ----------------------------------------------
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="File does not appear to be a valid PDF (missing %PDF header).",
        )

    # --- Extract text + build JSON tree -------------------------------------
    try:
        pdf_text = _extract_pdf_text(content)
        tree     = _build_tree_from_pdf(pdf_text, transaction_id, amount, merchant_category)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {exc}") from exc

    # --- Persist tree so forensics node can load it -------------------------
    tree_path = _DOCS_DIR / f"{transaction_id}_tree.json"
    with open(tree_path, "w", encoding="utf-8") as fh:
        json.dump(tree, fh, indent=2)

    # --- Also save the raw PDF to sample_docs --------------------------------
    pdf_path = _DOCS_DIR / f"{transaction_id}_invoice.pdf"
    with open(pdf_path, "wb") as fh:
        fh.write(content)

    # --- PageIndex vectorless RAG indexing (when USE_PAGEINDEX=1) -----------
    # Index the PDF before running the pipeline so the forensics node can
    # retrieve page-level context via tree traversal instead of the static tree.
    pageindex_doc_id = None
    if os.environ.get("USE_PAGEINDEX", "").strip() in ("1", "true", "yes"):
        try:
            from agents.forensics_pageindex import index_pdf
            pageindex_doc_id = index_pdf(transaction_id, pdf_path)
        except ImportError:
            pass   # PageIndex not installed -- forensics falls back to static tree

    # --- Run pipeline -------------------------------------------------------
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    tx: dict = {
        "transaction_id":        transaction_id,
        "user_id":               "auditor_upload",
        "timestamp":             timestamp,
        "amount":                amount,
        "merchant_category":     merchant_category,
        "distance_from_home_km": distance_from_home_km,
        "velocity_24h":          velocity_24h,
        "amount_z":              0.0,
    }

    initial_state = {
        "transaction_data":  tx,
        "ml_risk_score":     0.0,
        "suspicion_level":   "",
        "document_evidence": [],
        "final_decision":    "",
        "audit_log":         [],
        "auditor_decision":  "",
        "auditor_comments":  "",
    }

    try:
        _graph.invoke(initial_state, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    snapshot   = _graph.get_state(config)
    state_dict = _state_to_dict(snapshot)
    suspicion  = state_dict.get("suspicion_level", "")
    score      = state_dict.get("ml_risk_score", 0.0)

    paused = "compliance_arbitrator" in (snapshot.next or [])
    if paused:
        _pending_cases[thread_id] = {
            "thread_id":       thread_id,
            "transaction_id":  transaction_id,
            "amount":          amount,
            "merchant":        merchant_category,
            "ml_risk_score":   score,
            "suspicion_level": suspicion,
            "next_node":       list(snapshot.next),
            "source":          "pdf_upload",
        }

    return {
        "thread_id":          thread_id,
        "status":             "PAUSED_AT_BREAKPOINT" if paused else "RESOLVED_AUTOMATICALLY",
        "suspicion_level":    suspicion,
        "ml_risk_score":      score,
        "tree_saved":         str(tree_path),
        "pageindex_doc_id":   pageindex_doc_id,
        "retrieval_method":   "pageindex_vectorless_rag" if pageindex_doc_id else "static_json_tree",
        "state":              state_dict,
    }
