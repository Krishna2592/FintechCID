"""
FintechCID — Transaction Screener Node
==========================================
LangGraph node that loads the MLflow-registered Random Forest model
and gates every transaction as LOW or HIGH suspicion.

Fail-safe: any exception during model load or inference defaults to
HIGH suspicion so potentially fraudulent transactions are never silently
passed through.
"""

import os
import pathlib
import pickle

import mlflow.sklearn
import numpy as np
import pandas as pd

from agents.state import AgentState

# ---------------------------------------------------------------------------
# Triton routing -- set USE_TRITON=1 to route inference to Triton Server
# ---------------------------------------------------------------------------
_USE_TRITON  = os.environ.get("USE_TRITON", "").strip() in ("1", "true", "yes")
_TRITON_URL  = os.environ.get("TRITON_URL", "localhost:8001")
_TRITON_MODEL = "transaction_screener"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR  = pathlib.Path(__file__).parent.parent
_MLRUNS    = _BASE_DIR / "mlruns"
_LE_PATH   = (
    _BASE_DIR
    / "mlruns"
    / "990591830491573391"
    / "1fcea895477d43e4a7c405832bcacd3d"
    / "artifacts"
    / "encoders"
    / "label_encoder.pkl"
)

REGISTERED_MODEL_URI = "models:/TransactionScreener_v1/1"
FRAUD_THRESHOLD      = 0.65

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once, reused on every invocation
# ---------------------------------------------------------------------------
_model         = None
_label_encoder = None


def _load_artifacts() -> None:
    """Load model and encoder into module-level singletons (lazy, thread-safe enough for single-process use)."""
    global _model, _label_encoder

    # Label encoder is always needed -- used in feature construction regardless of backend
    if _label_encoder is None:
        with open(_LE_PATH, "rb") as fh:
            _label_encoder = pickle.load(fh)

    # sklearn model only loaded when not routing to Triton
    if not _USE_TRITON and _model is None:
        mlflow.set_tracking_uri(_MLRUNS.as_uri())
        _model = mlflow.sklearn.load_model(REGISTERED_MODEL_URI)


def _infer_triton(features: pd.DataFrame) -> float:
    """
    Call Triton Inference Server via HTTP and return P(fraud).
    Uses the KServe v2 inference protocol that Triton implements.
    """
    import tritonclient.http as httpclient  # lazy -- only imported when USE_TRITON=1

    client = httpclient.InferenceServerClient(url=_TRITON_URL, verbose=False)

    arr = features.values.astype(np.float32)
    inp = httpclient.InferInput("float_input", list(arr.shape), "FP32")
    inp.set_data_from_numpy(arr)

    out      = httpclient.InferRequestedOutput("probabilities")
    response = client.infer(_TRITON_MODEL, inputs=[inp], outputs=[out])
    probs    = response.as_numpy("probabilities")
    return float(probs[0][1])   # column 1 == P(fraud)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _build_feature_row(tx: dict) -> pd.DataFrame:
    """
    Reconstruct the 7 features expected by the trained RandomForest from
    a raw transaction dict.

    Feature mapping
    ---------------
    amount               → direct
    distance_from_home_km → direct
    velocity_24h         → direct
    hour_of_day          → derived from timestamp
    is_weekend           → derived from timestamp
    amount_z             → caller may supply; defaults to 0.0 (neutral)
    merchant_enc         → label-encoded via saved LabelEncoder
    """
    ts_raw = tx.get("timestamp")
    if ts_raw:
        ts          = pd.to_datetime(ts_raw, utc=True)
        hour_of_day = int(ts.hour)
        is_weekend  = int(ts.dayofweek in [5, 6])
    else:
        hour_of_day = 12
        is_weekend  = 0

    merchant = tx.get("merchant_category", "grocery")
    try:
        merchant_enc = int(_label_encoder.transform([merchant])[0])
    except ValueError:
        merchant_enc = 0   # unknown category → treat as first class

    return pd.DataFrame([{
        "amount":                float(tx.get("amount", 0.0)),
        "distance_from_home_km": float(tx.get("distance_from_home_km", 0.0)),
        "velocity_24h":          int(tx.get("velocity_24h", 1)),
        "hour_of_day":           hour_of_day,
        "is_weekend":            is_weekend,
        "amount_z":              float(tx.get("amount_z", 0.0)),
        "merchant_enc":          merchant_enc,
    }])


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def screen_transaction(state: AgentState) -> AgentState:
    """
    LangGraph node — Transaction Screener.

    Reads `transaction_data` from state, runs the RF model, and writes:
      - ml_risk_score   : P(fraud)
      - suspicion_level : "HIGH" | "LOW"
      - final_decision  : set to "APPROVED" only for LOW risk
      - audit_log       : appended with outcome message
    """
    state     = dict(state)
    audit_log = list(state.get("audit_log", []))

    try:
        _load_artifacts()

        tx       = state["transaction_data"]
        features = _build_feature_row(tx)

        if _USE_TRITON:
            proba = _infer_triton(features)
        else:
            proba = float(_model.predict_proba(features)[0][1])

        state["ml_risk_score"] = round(proba, 6)

        if proba >= FRAUD_THRESHOLD:
            state["suspicion_level"] = "HIGH"
            audit_log.append(
                "Screener Agent: High ML risk detected. Routing to Document Forensics."
            )
        else:
            state["suspicion_level"] = "LOW"
            state["final_decision"]  = "APPROVED"
            audit_log.append("Screener Agent: Cleared via ML baseline.")

    except Exception as exc:  # noqa: BLE001
        state["suspicion_level"] = "HIGH"
        state["ml_risk_score"]   = -1.0
        audit_log.append(
            f"Screener Agent: ERROR — defaulting to HIGH (fail-safe). Detail: {exc}"
        )

    state["audit_log"] = audit_log
    return state
