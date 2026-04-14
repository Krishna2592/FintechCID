"""
FintechCID -- Compliance Arbitrator Node
============================================
Runs AFTER the human auditor provides a decision via the HITL dashboard.
Combines ML risk, forensic findings, and auditor input into a final
signed Audit Package, then sets final_decision in the state.
"""

import json
from datetime import datetime, timezone

from agents.state import AgentState


def compliance_arbitrator_node(state: AgentState) -> AgentState:
    """
    LangGraph node -- Compliance Arbitrator.

    Expects state to contain:
      - auditor_decision  : "APPROVE" or "REJECT"
      - auditor_comments  : free-text justification
      - ml_risk_score     : float from screener
      - document_evidence : list of forensic findings
      - audit_log         : running trace

    Produces:
      - final_decision    : "APPROVED BY AUDITOR" | "REJECTED BY AUDITOR"
      - audit_log         : appended with final ruling + audit package summary
    """
    state     = dict(state)
    audit_log = list(state.get("audit_log", []))

    decision = (state.get("auditor_decision") or "").strip().upper()
    comments = (state.get("auditor_comments") or "No comments provided.").strip()
    tx       = state.get("transaction_data", {})
    evidence = state.get("document_evidence", [])

    # ---- Build final audit package ----------------------------------------
    audit_package = {
        "audit_package_version": "1.0",
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "transaction_id":        tx.get("transaction_id", "unknown"),
        "ml_risk_score":         state.get("ml_risk_score", -1.0),
        "suspicion_level":       state.get("suspicion_level", "UNKNOWN"),
        "forensic_findings":     evidence,
        "auditor_decision":      decision,
        "auditor_comments":      comments,
        "compliance_standard":   "EY AML / CFT Internal Policy v3.2",
        "ruling_justification": (
            f"Auditor {decision}D this transaction based on ML risk score "
            f"{state.get('ml_risk_score', 0):.4f} and forensic document analysis. "
            f"Auditor note: {comments}"
        ),
    }

    if decision == "APPROVE":
        final_decision = "APPROVED BY AUDITOR"
        audit_log.append(
            f"Compliance Arbitrator: Auditor APPROVED transaction "
            f"{tx.get('transaction_id', 'unknown')}. "
            f"Comments: {comments}"
        )
    elif decision == "REJECT":
        final_decision = "REJECTED BY AUDITOR"
        audit_log.append(
            f"Compliance Arbitrator: Auditor REJECTED transaction "
            f"{tx.get('transaction_id', 'unknown')}. "
            f"Comments: {comments}"
        )
    else:
        # No auditor input -- should not normally reach here
        final_decision = "UNRESOLVED -- NO AUDITOR INPUT"
        audit_log.append(
            "Compliance Arbitrator: WARNING -- No auditor decision received. "
            "Marking as UNRESOLVED."
        )

    audit_log.append(
        f"Compliance Arbitrator: Audit Package generated. "
        f"Final ruling: {final_decision}. "
        f"Package keys: {list(audit_package.keys())}"
    )

    state["final_decision"] = final_decision
    state["audit_log"]      = audit_log
    # Store the serialised audit package in document_evidence for record-keeping
    state["document_evidence"] = evidence + [{"audit_package": audit_package}]
    return state
