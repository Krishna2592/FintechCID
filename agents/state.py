"""
FintechCID — Shared LangGraph State
=======================================
Defines the canonical AgentState TypedDict that flows through
every node in the LangGraph orchestration pipeline.
"""

from typing import TypedDict


class AgentState(TypedDict):
    transaction_data: dict        # Raw incoming JSON payload
    ml_risk_score: float          # P(fraud) from the RF model
    suspicion_level: str          # "LOW" | "HIGH" | "MANUAL_REVIEW_REQUIRED"
    document_evidence: list       # Populated by Document Forensics agent (Task 03)
    final_decision: str           # "APPROVED" | "FLAGGED" | "REJECTED"
    audit_log: list               # Ordered trace of agent actions (EY compliance)
    # HITL fields -- injected by human auditor via dashboard (Task 04)
    auditor_decision: str         # "APPROVE" | "REJECT"
    auditor_comments: str         # Free-text auditor justification
