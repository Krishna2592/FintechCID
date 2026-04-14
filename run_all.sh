#!/usr/bin/env bash
# FintechCID -- Start all services
# =====================================
# Starts FastAPI, Streamlit, and MLflow UI simultaneously.
#
# Usage (from project root FintechCID/):
#   bash run_all.sh
#
# Services:
#   FastAPI    http://localhost:8000       (REST API + Swagger docs at /docs)
#   Streamlit  http://localhost:8501       (Auditor Dashboard)
#   MLflow UI  http://localhost:5000       (Experiment tracking)

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[FintechCID] Project root: $ROOT"
echo "[FintechCID] Starting all services..."

# ---- FastAPI ---------------------------------------------------------------
echo "[FintechCID] Starting FastAPI on :8000 ..."
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --app-dir "$ROOT" &
FASTAPI_PID=$!
echo "[FintechCID] FastAPI PID: $FASTAPI_PID"

# Give FastAPI a moment to start (model loading takes a few seconds)
sleep 4

# ---- Streamlit -------------------------------------------------------------
echo "[FintechCID] Starting Streamlit on :8501 ..."
streamlit run "$ROOT/frontend/app.py" \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false &
STREAMLIT_PID=$!
echo "[FintechCID] Streamlit PID: $STREAMLIT_PID"

# ---- MLflow UI -------------------------------------------------------------
echo "[FintechCID] Starting MLflow UI on :5000 ..."
mlflow ui \
  --backend-store-uri "$ROOT/mlruns" \
  --host 0.0.0.0 \
  --port 5000 &
MLFLOW_PID=$!
echo "[FintechCID] MLflow UI PID: $MLFLOW_PID"

# ---- Summary ---------------------------------------------------------------
echo ""
echo "========================================================"
echo "  FintechCID -- All Services Running"
echo "========================================================"
echo "  FastAPI  (REST API)   http://localhost:8000"
echo "  FastAPI  (Swagger)    http://localhost:8000/docs"
echo "  Streamlit (Dashboard) http://localhost:8501"
echo "  MLflow UI             http://localhost:5000"
echo "========================================================"
echo "  Press Ctrl+C to stop all services."
echo ""

# Trap Ctrl+C and kill all background processes
cleanup() {
  echo ""
  echo "[FintechCID] Shutting down all services..."
  kill $FASTAPI_PID $STREAMLIT_PID $MLFLOW_PID 2>/dev/null
  echo "[FintechCID] Done."
  exit 0
}
trap cleanup SIGINT SIGTERM

wait
