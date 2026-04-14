# Task 02: LangGraph State & Transaction Screener Node

## 🎯 Objective
Initialize the LangGraph orchestration architecture. Define the heavily typed State dictionary that will pass between agents, and implement the `Transaction Screener Node` which loads our MLflow-registered Random Forest model to gatekeep the workflow.
## 🏗️ Architecture Requirements

### 1. The LangGraph State Definition
- Create `agents/state.py`.
- Define a strictly typed `TypedDict` (or Pydantic BaseModel) called `AgentState`.
- It must contain at least the following keys:
  - `transaction_data`: dict (the incoming JSON payload)
  - `ml_risk_score`: float (populated by the RF model)
  - `suspicion_level`: str (e.g., "LOW", "HIGH")
  - `document_evidence`: list (placeholder for the next agent)
  - `final_decision`: str
  - `audit_log`: list of strings (to track which agent did what, crucial for EY compliance)

### 2. The Transaction Screener Node
- Create `agents/transaction_screener.py`.
- Write a function `screen_transaction(state: AgentState) -> AgentState`.
- **Logic:**
  1. Load the `TransactionScreener_v1` Random Forest model from the local MLflow registry (handle the model loading efficiently so it doesn't reload on every single execution).
  2. Parse the `transaction_data` from the state and format it for the Scikit-Learn model.
  3. Execute `.predict_proba()`.
  4. **The Gate:** If the probability of fraud is >= 0.65, set `suspicion_level` to "HIGH" and append to the `audit_log`: "Screener Agent: High ML risk detected. Routing to Document Forensics."
  5. If < 0.65, set `suspicion_level` to "LOW", `final_decision` to "APPROVED", and append to the `audit_log`: "Screener Agent: Cleared via ML baseline."
- Return the updated state.

### 3. The Graph Routing Skeleton
- Create `agents/graph.py`.
- Initialize a `StateGraph(AgentState)`.
- Add the `screen_transaction` node.
- Add a conditional edge from the screener:
  - If `suspicion_level == "HIGH"`, route to a placeholder node called `document_forensics`.
  - If `suspicion_level == "LOW"`, route to `END`.

## 🔒 DevSecOps Constraints
- Use robust `try/except` blocks around the MLflow model loading and prediction steps. If the model fails or the input data is malformed, the system must default to a "HIGH" suspicion level (Fail-Safe Secure) and log the error in the `audit_log`.

## 🚀 Execution Instructions for Claude Code
1. Read this entire specification.
2. Create/update the 3 files: `agents/state.py`, `agents/transaction_screener.py`, and `agents/graph.py`.
3. Write a quick test script in the root directory called `test_screener.py` that manually feeds a mock highly-suspicious transaction into the LangGraph and prints the resulting state.
4. Execute `test_screener.py` and report back the terminal output.