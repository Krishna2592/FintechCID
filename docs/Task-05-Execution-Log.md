---
tags: [fintechcid, task-log, devsecops, streamlit, bandit, pip-audit, fastapi, upload]
task: "Task 05 -- DevSecOps Visibility & Interactive Evidence Upload"
status: "Complete"
date_completed: 2026-03-29
linked_task: "[[_Tasks/Task-05-DevSecOps-Dashboard-and-Upload.md]]"
---

# Task 05 -- DevSecOps Visibility & Interactive Evidence Upload

## Status: Complete

## What Was Built

| File | Purpose |
|---|---|
| `run_security_audit.sh` | Local SAST + dependency + Trivy scan executor |
| `frontend/data/bandit_report.json` | Bandit SAST output (generated) |
| `frontend/data/dependency_report.json` | pip-audit CVE output (generated) |
| `frontend/data/trivy_report.json` | Simulated Trivy container scan |
| `api/main.py` | Added `POST /api/v1/upload-evidence` endpoint |
| `frontend/app.py` | Added PDF file uploader to sidebar |
| `frontend/pages/1_Security_Audit.py` | Multipage Streamlit security dashboard |
| `.github/workflows/devsecops.yml` | GitHub Actions CI/CD pipeline |
| `requirements.txt` | Added pypdf, python-multipart, bandit, pip-audit |

---

## Security Audit Results

### Bandit SAST (api/, agents/, core_logic/)

| Severity | Count |
|---|---|
| HIGH | 0 |
| MEDIUM | 1 |
| LOW | 1 |
| Lines Scanned | 724 |

No HIGH severity issues found.

### pip-audit Dependency Scan

| Package | CVEs | Fix Available |
|---|---|---|
| pip 25.2 | CVE-2025-8869, CVE-2026-1703 | pip >= 26.0 |
| pygments 2.19.2 | CVE-2026-4539 | No fix yet |
| tornado 6.5.2 | GHSA-78cv, CVE-2026-31958 | tornado >= 6.5.5 |

**Total: 5 known vulnerabilities across 3 packages**

### Trivy Container Scan
Simulated locally (no vulnerabilities). Real scan runs in CI/CD via GitHub Actions.

---

## Upload Endpoint

`POST /api/v1/upload-evidence`

**Security controls:**
- MIME type validated: only `application/pdf` accepted
- Magic bytes validated: file must start with `%PDF`
- PDF text extracted with pypdf (no code execution)
- Simulated JSON tree saved to `data/sample_docs/<tid>_tree.json`
- Full LangGraph pipeline triggered on upload

**Flow:** Upload PDF → validate → extract text → build RAG tree → save → run pipeline → return thread_id + state

---

## Streamlit Multipage App

Two pages now available at `http://localhost:8501`:

| Page | URL path | Content |
|---|---|---|
| Auditor Dashboard | `/` | Case queue, PDF viewer, HITL decision |
| Security Audit | `/Security_Audit` | SAST + CVE + Trivy metrics |

**Security page features:**
- st.metric tiles: HIGH/MED/LOW findings + LOC + total issues
- Bar chart: findings by severity
- Bar chart: findings by file
- Expandable raw findings with code snippets
- Dependency CVE cards with fix versions
- Trivy container scan results
- CI/CD pipeline status table

---

## CI/CD Pipeline (.github/workflows/devsecops.yml)

Triggers on every push. Four jobs:

1. **Lint** (Flake8) -- fails on syntax/style errors
2. **SAST** (Bandit) -- fails if HIGH severity findings > 0
3. **Dependency Audit** (pip-audit) -- uploads report as artifact
4. **Container Scan** (Trivy) -- fails on HIGH or CRITICAL CVEs; uploads SARIF to GitHub Security tab

---

## How to Run

```bash
# Generate security reports (one-time or on each audit)
bash run_security_audit.sh

# Start all services
bash run_all.sh

# Or just Streamlit
streamlit run frontend/app.py
```

- Auditor Dashboard: http://localhost:8501
- Security Audit page: http://localhost:8501/Security_Audit
