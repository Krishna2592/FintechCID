#!/usr/bin/env bash
# FintechCID -- Local Security Audit Script
# =============================================
# Runs SAST (Bandit) + dependency audit (pip-audit) and writes
# JSON reports to frontend/data/ for the Security Dashboard.
#
# Usage (from project root: FintechCID/):
#   bash run_security_audit.sh

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$ROOT/frontend/data"
mkdir -p "$DATA_DIR"

echo "========================================================"
echo "  FintechCID -- Security Audit"
echo "  $(date)"
echo "========================================================"

# ---- 1. Bandit SAST --------------------------------------------------------
echo ""
echo "[1/3] Running Bandit SAST on api/, agents/, core_logic/ ..."

python -m bandit \
  -r "$ROOT/api" "$ROOT/agents" "$ROOT/core_logic" \
  -f json \
  --exit-zero \
  -q \
  -o "$DATA_DIR/bandit_report.json" 2>/dev/null

echo "      Bandit report -> $DATA_DIR/bandit_report.json"

# ---- 2. pip-audit dependency scan -----------------------------------------
echo ""
echo "[2/3] Running pip-audit dependency vulnerability scan ..."

python -m pip_audit \
  --format json \
  --progress-spinner off \
  2>/dev/null > "$DATA_DIR/dependency_report.json" || true

echo "      Dependency report -> $DATA_DIR/dependency_report.json"

# ---- 3. Simulated Trivy container scan (CI/CD runs the real scan) ----------
echo ""
echo "[3/3] Generating simulated Trivy container scan report ..."

cat > "$DATA_DIR/trivy_report.json" << 'EOF'
{
  "SchemaVersion": 2,
  "ArtifactName": "fintechcid:local",
  "ArtifactType": "container_image",
  "Metadata": {
    "ImageID": "sha256:simulated-local-build",
    "OS": {"Family": "debian", "Name": "12.0"},
    "ImageConfig": {"Architecture": "amd64"}
  },
  "Results": [
    {
      "Target": "fintechcid:local (debian 12.0)",
      "Class": "os-pkgs",
      "Type": "debian",
      "Vulnerabilities": null
    },
    {
      "Target": "Python (python-pkg)",
      "Class": "lang-pkgs",
      "Type": "python-pkg",
      "Vulnerabilities": null
    }
  ],
  "_note": "SIMULATED -- CI/CD pipeline runs real Trivy scan against built image."
}
EOF
echo "      Trivy report    -> $DATA_DIR/trivy_report.json"

# ---- Summary ---------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Scan complete. Reports written to frontend/data/"
echo ""
BANDIT_HIGH=$(python -c "
import json, sys
try:
    d = json.load(open('$DATA_DIR/bandit_report.json'))
    t = d.get('metrics', {}).get('_totals', {})
    print(f\"  Bandit  -- HIGH: {int(t.get('SEVERITY.HIGH',0))}  MED: {int(t.get('SEVERITY.MEDIUM',0))}  LOW: {int(t.get('SEVERITY.LOW',0))}  LOC: {int(t.get('loc',0))}\")
except Exception as e:
    print(f'  Bandit  -- could not parse: {e}')
" 2>/dev/null)
echo "$BANDIT_HIGH"

VULN_COUNT=$(python -c "
import json
try:
    d = json.load(open('$DATA_DIR/dependency_report.json'))
    deps = d.get('dependencies', [])
    vuln_pkgs = [p for p in deps if p.get('vulns')]
    total_vulns = sum(len(p['vulns']) for p in vuln_pkgs)
    print(f'  pip-audit -- {total_vulns} vulnerabilities across {len(vuln_pkgs)} package(s)')
except Exception as e:
    print(f'  pip-audit -- could not parse: {e}')
" 2>/dev/null)
echo "$VULN_COUNT"
echo "  Trivy   -- Simulated (no vulnerabilities in local scan)"
echo "========================================================"
echo ""
echo "  Open the Security Dashboard to view full details:"
echo "  streamlit run frontend/app.py"
echo ""
