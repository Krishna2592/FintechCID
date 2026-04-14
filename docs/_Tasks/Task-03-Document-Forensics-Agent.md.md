# Task 03: Document Forensic Agent & Synthetic Evidence Generation

## 🎯 Objective
Implement the `Document Forensic Node` in our LangGraph architecture. This agent activates only when the Transaction Screener flags a transaction. It will simulate a Vectorless RAG approach by ingesting hierarchical document structures and using a local LLM via Ollama to verify discrepancies. Crucially, we will also generate safe, synthetic PDF evidence for public demoing without exposing PII.

## 🏗️ Architecture Requirements

### 1. Synthetic Evidence Generation (Safe Demoing)
- Ensure `fpdf` or `reportlab` is added to `requirements.txt` and installed.
- Create `data/generate_documents.py`.
- Write a utility that hooks into the Parquet data generated in Task 00. For every transaction flagged as `is_fraud = 1` (and a small sample of legit transactions), generate two things:
  1. **A Visual PDF:** Save a synthetic Invoice or KYC document to `/data/sample_docs/{transaction_id}_invoice.pdf`. Use fake names, fake addresses, and fake merchant logos/text.
  2. **The Vectorless RAG Tree (JSON):** Generate a corresponding hierarchical JSON file (e.g., `{"DocumentRoot": {"Header": {"Merchant": "X"}, "Body": {"Amount": 500, "Items": [...]}}}`). This simulates the structural extraction of the PDF.
- **Fraud Injection:** Intentionally inject discrepancies into the JSON/PDF for the fraud cases (e.g., the invoice amount reads $5,000, but the transaction amount was $50,000, or the merchant name is spoofed).

### 2. Local LLM Integration (Zero-Trust)
- In `agents/document_forensics.py`, initialize a local LLM connection using `ChatOllama` (model: `llama3` or `mistral`).
- Enforce `temperature=0.0` to guarantee deterministic, highly repeatable audit behavior.

### 3. The Document Forensic Node
- Write the function `document_forensics_node(state: AgentState) -> AgentState` in `agents/document_forensics.py`.
- **Logic:**
  1. Retrieve the `transaction_data` from the current state.
  2. Fetch the corresponding structured JSON document tree based on the `transaction_id`.
  3. **The Prompt:** Construct a strict prompt for the LLM. Pass the structured JSON document tree and the transaction data. Instruct the LLM to act as a forensic auditor and output a JSON response verifying: `amount_match` (boolean), `merchant_match` (boolean), and `forensic_notes` (string).
  4. Parse the LLM's output using LangChain's JSON output parsers.
  5. **State Update:** Append the `forensic_notes` to the `document_evidence` list.
  6. Update the `audit_log` with: "Document Forensics: Evaluated structural document tree. Findings: [Summary of match/mismatch]".
- Return the updated state.

### 4. Graph Update
- In `agents/graph.py`, update the placeholder `document_forensics` node to point to this new function.
- Add an edge from `document_forensics` routing to a new placeholder node: `compliance_arbitrator`.

## 🔒 DevSecOps Constraints
- **Prompt Injection Defense:** Treat the synthetic OCR JSON text as untrusted user input. Ensure it is strictly wrapped in delimiters (like `"""`) within the LangChain prompt template so the LLM does not confuse document text with system instructions.
- **Graceful Degradation:** Wrap the Ollama call in a `try/except` block with a timeout. If the local LLM fails to respond, update the `audit_log` with "LLM Service Timeout" and pass the state forward with `suspicion_level` forced to "MANUAL_REVIEW_REQUIRED".

## 🚀 Execution Instructions for Claude Code
1. Read this entire specification.
2. Update `requirements.txt` and install PDF dependencies.
3. Create `/data/sample_docs/` directory.
4. Implement and run `data/generate_documents.py` to create the mock evidence database and PDFs.
5. Implement the LLM logic and node function in `agents/document_forensics.py`.
6. Update `agents/graph.py`.
7. Update `test_screener.py` to run a full trace through both the Screener and the Forensics node using a known flagged `transaction_id`. 
8. Execute the test script and report the terminal output, explicitly showing the updated `AgentState` with the LLM's forensic notes.