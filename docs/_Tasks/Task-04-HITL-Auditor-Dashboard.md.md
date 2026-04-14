# Task 04: Human-in-the-Loop (HITL) & Auditor Dashboard

## 🎯 Objective
Implement the "Maker-Checker" audit workflow. Configure LangGraph to pause execution before the final decision, and build a Streamlit dashboard that allows a human auditor to inspect the AI's evidence and resume the workflow.

## 🏗️ Architecture Requirements

### 1. LangGraph Persistence & Breakpoints
- Update `agents/graph.py` to use a **Checkpointer** (use `MemorySaver` for local development).
- Configure the graph to **interrupt** before the `compliance_arbitrator` node.
- Ensure every graph invocation uses a unique `thread_id` so the auditor can resume specific cases.

### 2. The Compliance Arbitrator Node
- Create `agents/compliance_arbitrator.py`.
- This node only runs *after* the human auditor provides input.
- **Logic:** 1. Receive the auditor's decision (`APPROVE` or `REJECT`) and their comments via the state.
  2. Combine the Auditor's input with the findings from the Screener and Forensic agents.
  3. Generate a final "Audit Package" (JSON) that logs the complete end-to-end reasoning.
  4. Set `final_decision` in the state.

### 3. The Auditor Dashboard (Streamlit)
- Create `frontend/app.py`.
- **Features:**
  - **Case Queue:** Display a list of transactions currently "Pending Review" (waiting at a breakpoint).
  - **Evidence Viewer:** Show the `transaction_data` side-by-side with the `document_evidence` (the LLM's forensic notes).
  - **PDF Preview:** Use a Streamlit PDF viewer to show the actual synthetic PDF generated in Task 03.
  - **Decision Interface:** Buttons for "Approve" and "Reject" with a text area for "Auditor Comments."
  - **Resume Logic:** When clicked, use `graph.update_state` to inject the auditor's decision and `graph.invoke(None, config)` to resume the thread.

### 4. API Integration (FastAPI)
- Update `api/main.py` to provide endpoints that the Streamlit app can call to:
  - List pending threads.
  - Get the state of a specific thread.
  - Resume a thread with auditor input.

## 🔒 DevSecOps Constraints
- **Session Security:** Ensure the `thread_id` is tracked securely.
- **Audit Integrity:** The `audit_log` in the State must be read-only for the agents but append-only for the human auditor. No one should be able to delete an entry from the log.

## 🚀 Execution Instructions for Claude Code
1. Read this entire specification.
2. Implement the persistence layer and breakpoint in `agents/graph.py`.
3. Build the `compliance_arbitrator.py` logic.
4. Create the Streamlit `frontend/app.py` and the necessary FastAPI routes.
5. Provide a bash script `run_all.sh` that starts the FastAPI server, the Streamlit app, and the MLflow UI simultaneously.
6. Report back with a screenshot description of the dashboard.