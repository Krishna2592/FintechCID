"""
FintechCID -- HITL Auditor Dashboard
=========================================
Streamlit front-end for the Human-in-the-Loop fraud review workflow.

Layout
------
Sidebar  : Submit test case | Case queue (pending cases)
Main     : Evidence panel (transaction + forensics) | PDF viewer | Decision interface
"""

import base64
import json
import os
import pathlib

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE  = os.environ.get("API_BASE", "http://localhost:8000")
DOCS_DIR  = pathlib.Path(__file__).parent.parent / "data" / "sample_docs"
INDEX_FILE = DOCS_DIR / "generated_index.json"

st.set_page_config(
    page_title="FintechCID -- Auditor Dashboard",
    page_icon="shield",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error ({path}): {exc}")
        return None


def pdf_iframe(pdf_path: pathlib.Path) -> str:
    """Return an HTML iframe embedding the PDF as base64."""
    if not pdf_path.exists():
        return "<p style='color:red'>PDF not found.</p>"
    b64 = base64.b64encode(pdf_path.read_bytes()).decode()
    return (
        f'<iframe src="data:application/pdf;base64,{b64}" '
        f'width="100%" height="520px" style="border:1px solid #ccc; border-radius:6px;">'
        f"</iframe>"
    )


def risk_badge(level: str) -> str:
    colours = {
        "HIGH":                    "#dc2626",
        "LOW":                     "#16a34a",
        "MANUAL_REVIEW_REQUIRED":  "#d97706",
    }
    c = colours.get(level.upper(), "#6b7280")
    return (
        f'<span style="background:{c};color:white;padding:3px 10px;'
        f'border-radius:12px;font-weight:bold;font-size:0.85rem;">{level}</span>'
    )


# ---------------------------------------------------------------------------
# Sidebar -- Submit & Queue
# ---------------------------------------------------------------------------
st.sidebar.title("FintechCID")
st.sidebar.caption("Financial Crime Investigation & Detection")
st.sidebar.divider()

# -- Load known fraud TIDs from index
known_fraud_tid = None
if INDEX_FILE.exists():
    idx = json.loads(INDEX_FILE.read_text())
    known_fraud_tid = idx.get("fraud_tids", [None])[0]

st.sidebar.subheader("Submit Transaction")
with st.sidebar.form("submit_form"):
    use_known = st.checkbox("Use known fraud case", value=True)
    tid_input = st.text_input(
        "Transaction ID (optional)",
        value=known_fraud_tid if use_known else "",
    )
    amount_input   = st.number_input("Amount ($)", value=1308.55, min_value=0.01)
    dist_input     = st.number_input("Distance from home (km)", value=1314.3, min_value=0.0)
    vel_input      = st.number_input("Velocity 24h", value=30, min_value=1)
    merchant_input = st.selectbox(
        "Merchant category",
        ["travel", "online_retail", "electronics", "grocery",
         "restaurant", "gas_station", "healthcare", "entertainment"],
    )
    submitted = st.form_submit_button("Submit to Pipeline")

if submitted:
    with st.spinner("Running pipeline to breakpoint..."):
        payload = {
            "transaction_id":        tid_input or None,
            "timestamp":             "2026-03-29T02:15:00+00:00",
            "amount":                amount_input,
            "distance_from_home_km": dist_input,
            "velocity_24h":          int(vel_input),
            "merchant_category":     merchant_input,
        }
        result = api_post("/api/transactions/submit", payload)
    if result:
        st.sidebar.success(
            f"Submitted. Status: **{result.get('status')}**  \n"
            f"Thread: `{result.get('thread_id','')[:8]}...`"
        )

st.sidebar.divider()

# -- Upload Evidence PDF -----------------------------------------------------
st.sidebar.subheader("Upload Evidence PDF")
uploaded_pdf = st.sidebar.file_uploader(
    "Upload an invoice PDF for analysis",
    type=["pdf"],
    help="Upload any invoice PDF. The system will parse it and run the full pipeline.",
)
if uploaded_pdf is not None:
    with st.sidebar.form("upload_form"):
        up_tid      = st.text_input("Transaction ID to link", value=uploaded_pdf.name.replace(".pdf","").replace("_invoice",""))
        up_amount   = st.number_input("Declared transaction amount ($)", value=500.0, min_value=0.01)
        up_dist     = st.number_input("Distance from home (km)", value=500.0)
        up_vel      = st.number_input("Velocity 24h", value=10, min_value=1)
        up_merchant = st.selectbox("Merchant category", [
            "travel","online_retail","electronics","grocery",
            "restaurant","gas_station","healthcare","entertainment",
        ])
        up_submit = st.form_submit_button("Analyze Uploaded PDF")

    if up_submit:
        with st.spinner("Uploading and running pipeline..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/api/v1/upload-evidence",
                    files={"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")},
                    data={
                        "transaction_id":        up_tid,
                        "amount":                str(up_amount),
                        "merchant_category":     up_merchant,
                        "distance_from_home_km": str(up_dist),
                        "velocity_24h":          str(int(up_vel)),
                    },
                    timeout=180,
                )
                resp.raise_for_status()
                up_result = resp.json()
                st.sidebar.success(
                    f"Done. Status: **{up_result.get('status')}**  \n"
                    f"Risk: **{up_result.get('ml_risk_score', 0):.4f}**  \n"
                    f"Thread: `{up_result.get('thread_id','')[:8]}...`"
                )
            except Exception as exc:
                st.sidebar.error(f"Upload failed: {exc}")

st.sidebar.divider()
st.sidebar.subheader("Case Queue")

pending_data = api_get("/api/transactions/pending") or {}
pending_list = pending_data.get("pending_cases", [])

if not pending_list:
    st.sidebar.info("No cases pending review.")
    selected_thread = None
else:
    case_options = {
        f"{c['transaction_id'][:12]}... | ${c.get('amount',0):,.0f} | "
        f"score={c.get('ml_risk_score',0):.3f}": c["thread_id"]
        for c in pending_list
    }
    chosen_label  = st.sidebar.radio("Select case to review:", list(case_options.keys()))
    selected_thread = case_options[chosen_label]

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Auditor Review Dashboard")
st.caption("Human-in-the-Loop | Maker-Checker Workflow | EY Compliance")

if not selected_thread:
    st.info(
        "No case selected. Submit a transaction via the sidebar to populate the queue, "
        "then select a case to review."
    )
    st.stop()

# Fetch full state
thread_data = api_get(f"/api/transactions/{selected_thread}") or {}
state       = thread_data.get("state", {})
tx          = state.get("transaction_data", {})
evidence    = state.get("document_evidence", [])
audit_log   = state.get("audit_log", [])
suspicion   = state.get("suspicion_level", "UNKNOWN")
score       = state.get("ml_risk_score", 0.0)
tx_id       = tx.get("transaction_id", selected_thread)

# ---- Header row -----------------------------------------------------------
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.subheader(f"Case: `{tx_id}`")
with hcol2:
    st.markdown(f"**Risk Level:** {risk_badge(suspicion)}", unsafe_allow_html=True)
    st.metric("ML Risk Score", f"{score:.6f}")

st.divider()

# ---- Main two-column layout -----------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Transaction Data")
    tx_display = {
        "Transaction ID":    tx.get("transaction_id", ""),
        "User ID":           tx.get("user_id", ""),
        "Timestamp":         tx.get("timestamp", ""),
        "Amount (USD)":      f"${tx.get('amount', 0):,.2f}",
        "Merchant Category": tx.get("merchant_category", ""),
        "Distance (km)":     f"{tx.get('distance_from_home_km', 0):,.1f}",
        "Velocity 24h":      tx.get("velocity_24h", 0),
    }
    for k, v in tx_display.items():
        c1, c2 = st.columns([2, 3])
        c1.markdown(f"**{k}**")
        c2.markdown(str(v))

    st.divider()
    st.subheader("Forensic Evidence")
    forensic_findings = [e for e in evidence if "forensic_notes" in e]
    if forensic_findings:
        for finding in forensic_findings:
            amount_ok   = finding.get("amount_match", None)
            merchant_ok = finding.get("merchant_match", None)
            notes       = finding.get("forensic_notes", "")
            col_a, col_b = st.columns(2)
            col_a.metric(
                "Amount Match",
                "PASS" if amount_ok else "FAIL",
                delta=None,
                delta_color="normal",
            )
            col_b.metric(
                "Merchant Match",
                "PASS" if merchant_ok else "FAIL",
                delta=None,
                delta_color="normal",
            )
            st.warning(f"**LLM Forensic Notes:** {notes}")
    else:
        st.info("No document evidence available for this case.")

    st.divider()
    st.subheader("Audit Log")
    for i, entry in enumerate(audit_log, 1):
        st.markdown(f"`{i:02d}.` {entry}")

with right:
    st.subheader("Invoice Document (PDF)")
    pdf_path = DOCS_DIR / f"{tx_id}_invoice.pdf"
    if pdf_path.exists():
        st.markdown(pdf_iframe(pdf_path), unsafe_allow_html=True)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name=f"{tx_id}_invoice.pdf",
                mime="application/pdf",
            )
    else:
        st.warning(
            f"No PDF found for `{tx_id}`.  \n"
            "Ensure `generate_documents.py` has been run and the transaction is "
            "in the fraud sample."
        )

    st.divider()
    st.subheader("Auditor Decision")
    st.caption(
        "Your decision will be injected into the LangGraph state and the "
        "Compliance Arbitrator node will finalise the case."
    )

    with st.form("decision_form"):
        comments = st.text_area(
            "Auditor Comments (required for REJECT)",
            placeholder="Summarise your reasoning...",
            height=100,
        )
        btn_col1, btn_col2 = st.columns(2)
        approve_btn = btn_col1.form_submit_button(
            "APPROVE",
            type="secondary",
            use_container_width=True,
        )
        reject_btn = btn_col2.form_submit_button(
            "REJECT",
            type="primary",
            use_container_width=True,
        )

    if approve_btn or reject_btn:
        decision = "APPROVE" if approve_btn else "REJECT"
        with st.spinner(f"Submitting {decision} decision and resuming graph..."):
            result = api_post(
                f"/api/transactions/{selected_thread}/resume",
                {"decision": decision, "comments": comments},
            )
        if result:
            final = result.get("final_decision", "UNKNOWN")
            colour = "#16a34a" if "APPROVED" in final else "#dc2626"
            st.markdown(
                f'<div style="background:{colour};color:white;padding:12px;'
                f'border-radius:8px;font-size:1.1rem;text-align:center;">'
                f'<b>RESOLVED: {final}</b></div>',
                unsafe_allow_html=True,
            )
            # Show final audit log
            final_state = result.get("state", {})
            final_log   = final_state.get("audit_log", [])
            if final_log:
                st.subheader("Final Audit Log")
                for i, entry in enumerate(final_log, 1):
                    st.markdown(f"`{i:02d}.` {entry}")
            st.balloons()
