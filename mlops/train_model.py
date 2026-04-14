"""
FintechCID -- Random Forest Training with MLflow
====================================================
Reads the 1M-row parquet dataset, engineers features,
tunes hyperparameters via RandomizedSearchCV on a 10%
subsample, trains the final model on the full dataset,
logs everything to MLflow, and registers the model as
"TransactionScreener_v1".

Usage:
    python mlops/train_model.py
"""

import os
import pathlib
import pickle
import tempfile

import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = pathlib.Path(__file__).parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "transactions.parquet"
MLRUNS_DIR = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "FintechCID_FraudDetection"
REGISTERED_NAME = "TransactionScreener_v1"

# ---------------------------------------------------------------------------
# 1. Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Returns feature matrix X and the fitted LabelEncoder for merchant_category.

    Engineered features
    -------------------
    - hour_of_day     : transaction hour (0-23) -- captures time-of-day risk
    - is_weekend      : 1 if Saturday/Sunday, else 0
    - amount_z        : z-score of amount per user (how unusual is this spend?)
    - merchant_enc    : label-encoded merchant_category
    """
    le = LabelEncoder()

    # Datetime features
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df = df.copy()
    df["hour_of_day"] = ts.dt.hour.astype("int32")
    df["is_weekend"]  = ts.dt.dayofweek.isin([5, 6]).astype("int32")

    # Amount z-score per user (robust spend anomaly signal)
    user_stats = df.groupby("user_id")["amount"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9)
    )
    df["amount_z"] = user_stats.astype("float32")

    # Encode merchant category
    df["merchant_enc"] = le.fit_transform(df["merchant_category"]).astype("int32")

    feature_cols = [
        "amount",
        "distance_from_home_km",
        "velocity_24h",
        "hour_of_day",
        "is_weekend",
        "amount_z",
        "merchant_enc",
    ]
    return df[feature_cols], le


# ---------------------------------------------------------------------------
# 2. Hyperparameter search on 10% subsample
# ---------------------------------------------------------------------------

def tune_hyperparams(
    X_train: pd.DataFrame, y_train: pd.Series
) -> dict:
    """
    Runs RandomizedSearchCV on 10% of the training data to keep it fast
    while still covering a meaningful search space.
    Returns the best parameters found.
    """
    sample_size = max(int(len(X_train) * 0.10), 10_000)
    idx = np.random.default_rng(42).choice(len(X_train), size=sample_size, replace=False)
    Xs, ys = X_train.iloc[idx], y_train.iloc[idx]

    param_dist = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth":    [5, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"],   # always use balanced for fraud
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=12,
        scoring="recall",   # maximise recall: catching fraud matters most
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(Xs, ys)
    print(f"[Tuning] Best params : {search.best_params_}")
    print(f"[Tuning] Best recall : {search.best_score_:.4f}")
    return search.best_params_


# ---------------------------------------------------------------------------
# 3. Confusion matrix artifact
# ---------------------------------------------------------------------------

def save_confusion_matrix(y_true, y_pred, artifact_dir: str) -> str:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix -- TransactionScreener_v1")
    path = os.path.join(artifact_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[Train] Loading data from {DATA_PATH} ...")
    df = pd.read_parquet(DATA_PATH)
    print(f"[Train] Loaded {len(df):,} rows  |  fraud rate: {df['is_fraud'].mean():.2%}")

    # Feature engineering
    print("[Train] Engineering features ...")
    X, label_encoder = engineer_features(df)
    y = df["is_fraud"]

    # Train / test split (stratified to preserve 5% fraud in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[Train] Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # MLflow setup
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="RF_tuned_1M"):

        # ---- Hyperparameter search ----------------------------------------
        print("[Train] Starting hyperparameter search (10% subsample) ...")
        best_params = tune_hyperparams(X_train, y_train)
        mlflow.log_params(best_params)

        # ---- Final training on full dataset ---------------------------------
        print("[Train] Training final model on full dataset ...")
        final_model = RandomForestClassifier(
            **best_params,
            random_state=42,
            n_jobs=-1,
        )
        final_model.fit(X_train, y_train)

        # ---- Evaluation -----------------------------------------------------
        y_pred = final_model.predict(X_test)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)

        mlflow.log_metrics({
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "f1_score":  round(f1,        4),
        })

        print("\n[Train] Classification report:")
        print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

        # ---- Artifacts ------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmp:
            # Confusion matrix plot
            cm_path = save_confusion_matrix(y_test, y_pred, tmp)
            mlflow.log_artifact(cm_path, artifact_path="plots")

            # LabelEncoder pickle
            le_path = os.path.join(tmp, "label_encoder.pkl")
            with open(le_path, "wb") as f:
                pickle.dump(label_encoder, f)
            mlflow.log_artifact(le_path, artifact_path="encoders")

        # ---- Log + register model ------------------------------------------
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name=REGISTERED_NAME,
            input_example=X_test.iloc[:5],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n[Train] MLflow run ID : {run_id}")
        print(f"[Train] Precision      : {precision:.4f}")
        print(f"[Train] Recall         : {recall:.4f}")
        print(f"[Train] F1-Score       : {f1:.4f}")
        print(f"[Train] Model registered as '{REGISTERED_NAME}'")
        print(f"\n[Train] Launch UI with:  mlflow ui --backend-store-uri {MLRUNS_DIR}")


if __name__ == "__main__":
    main()
