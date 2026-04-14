# Task 05: DevSecOps Visibility & Interactive Evidence Upload

## 🎯 Objective
Finalize the "FintechCID" architecture by adding live, interactive capabilities and transparent DevSecOps reporting. We will add a file upload feature to the Streamlit app for dynamic testing, create a dedicated Security Dashboard page, and wire up the local security scanning tools to feed data to that dashboard.

## 🏗️ Architecture Requirements

### 1. The Interactive File Upload (Streamlit Update)
- Update `frontend/app.py` to include a `st.file_uploader` in the sidebar or main page, accepting PDFs.
- When a PDF is uploaded:
  1. Send it to a new FastAPI endpoint (`/api/v1/upload-evidence`).
  2. The backend must simulate parsing this PDF into the structural JSON tree required by the Vectorless RAG agent.
  3. Trigger the LangGraph pipeline using the uploaded document's data against a selected transaction.

### 2. The DevSecOps Scan Executor (Local Script)
- Create a bash script at the root level called `run_security_audit.sh`.
- **Logic:**
  1. Run `bandit` (Python SAST) on the `/api`, `/agents`, and `/core_logic` directories. Output the results strictly as a JSON file: `frontend/data/bandit_report.json`.
  2. Run a lightweight local dependency check (like `pip-audit` or safety) and output to `frontend/data/dependency_report.json`.
  *(Note: We will simulate the Trivy container scan output locally since running full Docker daemon scans inside a quick local script can be brittle for demo purposes, but the CI/CD pipeline will run the real thing).*

### 3. The Security Dashboard (Streamlit Page 2)
- Streamlit supports multipage apps. Create a folder `frontend/pages/` and add a file `1_🛡️_Security_Audit.py`.
- **Dashboard Features:**
  - Read the `bandit_report.json` and `dependency_report.json` files.
  - Display high-level metrics using `st.metric` (e.g., "High Severity Issues: 0", "Lines of Code Scanned: 1,204").
  - Create simple, clean bar charts or pie charts showing the breakdown of vulnerability severities.
  - Provide a raw data expander for auditors who want to see the exact lines of code flagged by the SAST tool.

### 4. GitHub Actions CI/CD (The Real Pipeline)
- Create `.github/workflows/devsecops.yml`.
- Write a standard GitHub Actions pipeline that triggers on `push`.
- It must execute:
  - Code linting (Flake8).
  - SAST scanning (Bandit).
  - Container scanning (Trivy).
- If any High or Critical vulnerabilities are found, the pipeline must explicitly fail.

## 🔒 DevSecOps Constraints
- **Safe File Handling:** The FastAPI upload endpoint must validate the MIME type of the uploaded file to ensure it is actually a PDF and not a malicious executable masquerading as a document.
- **Fail-Safe Dashboard:** The Streamlit Security page must gracefully handle the absence of the JSON report files (e.g., displaying "Run the security audit script to generate metrics" instead of throwing a Python FileNotFoundError).

## 🚀 Execution Instructions for Claude Code
1. Read this entire specification.
2. Update the FastAPI `api/main.py` with the secure upload endpoint.
3. Overhaul `frontend/app.py` to include the file uploader and integrate it with the backend.
4. Create the `run_security_audit.sh` script and ensure the `frontend/data/` directory exists.
5. Build the multipage Streamlit dashboard `frontend/pages/1_🛡️_Security_Audit.py`.
6. Write the `.github/workflows/devsecops.yml` file.
7. Execute `run_security_audit.sh` to generate the initial local reports, then run the Streamlit app to verify the Security Audit page renders the metrics correctly.