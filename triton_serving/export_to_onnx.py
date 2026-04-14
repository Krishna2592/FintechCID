"""
FintechCID -- ONNX Export for Triton Inference Server
======================================================
Loads the MLflow-registered Random Forest model and converts it to ONNX
format using skl2onnx, then writes the output into the Triton model
repository directory structure.

Run this once after training before starting the Triton container.

Usage (from project root):
    python triton_serving/export_to_onnx.py

Output:
    triton_serving/model_repository/transaction_screener/1/model.onnx
"""

import pathlib
import pickle
import sys

import mlflow.sklearn
import numpy as np
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR    = pathlib.Path(__file__).parent.parent
_MLRUNS      = _BASE_DIR / "mlruns"
_OUTPUT_PATH = (
    pathlib.Path(__file__).parent
    / "model_repository"
    / "transaction_screener"
    / "1"
    / "model.onnx"
)

# Hardcoded because the encoder lives at a run-specific artifact path.
# Update if you retrain and the run ID changes.
_LE_PATH = (
    _BASE_DIR
    / "mlruns"
    / "990591830491573391"
    / "1fcea895477d43e4a7c405832bcacd3d"
    / "artifacts"
    / "encoders"
    / "label_encoder.pkl"
)

REGISTERED_MODEL_URI = "models:/TransactionScreener_v1/1"

# The 7 features in the exact order the RF was trained on
FEATURE_NAMES = [
    "amount",
    "distance_from_home_km",
    "velocity_24h",
    "hour_of_day",
    "is_weekend",
    "amount_z",
    "merchant_enc",
]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export() -> None:
    print(f"[Export] Loading model from MLflow registry: {REGISTERED_MODEL_URI}")
    mlflow.set_tracking_uri(_MLRUNS.as_uri())
    sklearn_model = mlflow.sklearn.load_model(REGISTERED_MODEL_URI)

    # skl2onnx needs to know the input shape up front.
    # All 7 features are float32 — ints were cast during training anyway.
    initial_types = [("float_input", FloatTensorType([None, len(FEATURE_NAMES)]))]

    print("[Export] Converting to ONNX (zipmap=False for clean array outputs) ...")
    onnx_model = convert_sklearn(
        sklearn_model,
        initial_types=initial_types,
        options={type(sklearn_model): {"zipmap": False}},
    )

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT_PATH, "wb") as fh:
        fh.write(onnx_model.SerializeToString())
    print(f"[Export] Saved -> {_OUTPUT_PATH}")

    # ---------------------------------------------------------------------------
    # Sanity check with ONNX Runtime before handing off to Triton
    # ---------------------------------------------------------------------------
    print("[Export] Validating with ONNX Runtime ...")

    sess = rt.InferenceSession(str(_OUTPUT_PATH), providers=["CPUExecutionProvider"])

    with open(_LE_PATH, "rb") as fh:
        le = pickle.load(fh)

    # A synthetic high-risk row: large amount, far from home, late night, travel
    sample = np.array([[
        1308.55,   # amount
        1314.3,    # distance_from_home_km
        30.0,      # velocity_24h
        2.0,       # hour_of_day (2am)
        1.0,       # is_weekend
        4.8,       # amount_z
        float(le.transform(["travel"])[0]),  # merchant_enc
    ]], dtype=np.float32)

    outputs = sess.run(["label", "probabilities"], {"float_input": sample})
    label  = int(outputs[0][0])
    p_fraud = float(outputs[1][0][1])

    print(f"[Export] Validation result  -- label={label}  P(fraud)={p_fraud:.6f}")

    if p_fraud < 0.5:
        print("[Export] WARNING: sample fraud probability is low -- check feature order")
    else:
        print("[Export] Validation passed.")

    print()
    print("  Model inputs  :", [i.name for i in sess.get_inputs()])
    print("  Model outputs :", [o.name for o in sess.get_outputs()])
    print()
    print("[Export] Done. Start the Triton container to serve this model.")
    print("         docker compose up triton")


if __name__ == "__main__":
    sys.path.insert(0, str(_BASE_DIR))
    export()
