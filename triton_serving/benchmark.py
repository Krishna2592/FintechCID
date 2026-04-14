"""
FintechCID -- FastAPI vs Triton Inference Benchmark
====================================================
Sends N concurrent inference requests to both backends and prints
a side-by-side latency / throughput comparison.

FastAPI path  : POST /api/predict   (sklearn model loaded in-process)
Triton path   : POST /v2/models/transaction_screener/infer  (ONNX Runtime)

Both services must be running before executing this script.
  FastAPI : http://localhost:8000   (bash run_all.sh  or  docker compose up)
  Triton  : http://localhost:8001   (docker compose up triton)

Usage:
    python triton_serving/benchmark.py
    python triton_serving/benchmark.py --requests 1000 --workers 100
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FASTAPI_URL = "http://localhost:8000/api/predict"
TRITON_URL  = "http://localhost:8001/v2/models/transaction_screener/infer"

# A representative high-risk transaction payload
_SAMPLE_TX = {
    "transaction_id":        "BENCH-001",
    "user_id":               "user_0042",
    "timestamp":             "2026-03-29T02:15:00+00:00",
    "amount":                1308.55,
    "merchant_category":     "travel",
    "distance_from_home_km": 1314.3,
    "velocity_24h":          30,
    "amount_z":              4.8,
}

# Pre-built feature vector matching the same transaction.
# Order must match what export_to_onnx.py and the trained model expect:
# [amount, distance_from_home_km, velocity_24h, hour_of_day, is_weekend, amount_z, merchant_enc]
# merchant_enc for "travel" is 5 (depends on LabelEncoder fit order -- check your encoder if results look wrong)
_SAMPLE_FEATURES = [1308.55, 1314.3, 30.0, 2.0, 1.0, 4.8, 5.0]

_TRITON_BODY = {
    "inputs": [
        {
            "name":     "float_input",
            "shape":    [1, 7],
            "datatype": "FP32",
            "data":     _SAMPLE_FEATURES,
        }
    ],
    "outputs": [{"name": "probabilities"}],
}


# ---------------------------------------------------------------------------
# Single-request helpers
# ---------------------------------------------------------------------------

def _call_fastapi(session: requests.Session) -> float:
    t0 = time.perf_counter()
    r  = session.post(FASTAPI_URL, json=_SAMPLE_TX, timeout=10)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000   # ms


def _call_triton(session: requests.Session) -> float:
    t0 = time.perf_counter()
    r  = session.post(TRITON_URL, json=_TRITON_BODY, timeout=10)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000   # ms


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run(fn, label: str, n_requests: int, n_workers: int) -> list[float]:
    """Fire n_requests calls using a thread pool and collect latencies."""
    latencies: list[float] = []
    errors = 0

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(fn, session) for _ in range(n_requests)]
            for f in as_completed(futures):
                try:
                    latencies.append(f.result())
                except Exception:
                    errors += 1

    if errors:
        print(f"  [{label}] {errors} request(s) failed -- check service is running")

    return latencies


def _stats(label: str, latencies: list[float]) -> dict:
    arr = np.array(latencies)
    return {
        "label":     label,
        "n":         len(arr),
        "p50":       float(np.percentile(arr, 50)),
        "p95":       float(np.percentile(arr, 95)),
        "p99":       float(np.percentile(arr, 99)),
        "mean":      float(arr.mean()),
        "total_s":   float(arr.sum() / 1000),      # wall-clock approximation
    }


# ---------------------------------------------------------------------------
# Connectivity check
# ---------------------------------------------------------------------------

def _check(url: str, label: str) -> bool:
    try:
        requests.get(url.rsplit("/", 2)[0], timeout=3)
        return True
    except Exception:
        print(f"  WARNING: {label} not reachable at {url}")
        print(f"           Skipping {label} benchmark.")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FintechCID inference benchmark")
    parser.add_argument("--requests", type=int, default=500,
                        help="Total number of inference requests per backend (default: 500)")
    parser.add_argument("--workers",  type=int, default=50,
                        help="Concurrent worker threads (default: 50)")
    args = parser.parse_args()

    n  = args.requests
    w  = args.workers

    print()
    print("=" * 65)
    print("  FintechCID -- Inference Benchmark")
    print("=" * 65)
    print(f"  Requests : {n}  |  Workers : {w}")
    print(f"  FastAPI  : {FASTAPI_URL}")
    print(f"  Triton   : {TRITON_URL}")
    print("=" * 65)
    print()

    results = []

    # ---- FastAPI (sklearn) ------------------------------------------------
    fastapi_ok = _check(FASTAPI_URL, "FastAPI")
    if fastapi_ok:
        print(f"  [1/2] Warming up FastAPI ...")
        try:
            _run(_call_fastapi, "warmup", 5, 5)
        except Exception:
            pass

        print(f"  [1/2] Benchmarking FastAPI ({n} requests) ...")
        t_start = time.perf_counter()
        lats = _run(_call_fastapi, "FastAPI", n, w)
        wall = time.perf_counter() - t_start

        if lats:
            s = _stats("FastAPI  (sklearn)", lats)
            s["rps"] = len(lats) / wall
            results.append(s)
            print(f"         Done. {len(lats)} succeeded in {wall:.1f}s")

    # ---- Triton (ONNX Runtime) -------------------------------------------
    triton_ok = _check(TRITON_URL, "Triton")
    if triton_ok:
        print(f"  [2/2] Warming up Triton ...")
        try:
            _run(_call_triton, "warmup", 5, 5)
        except Exception:
            pass

        print(f"  [2/2] Benchmarking Triton ({n} requests) ...")
        t_start = time.perf_counter()
        lats = _run(_call_triton, "Triton", n, w)
        wall = time.perf_counter() - t_start

        if lats:
            s = _stats("Triton   (ONNX RT)", lats)
            s["rps"] = len(lats) / wall
            results.append(s)
            print(f"         Done. {len(lats)} succeeded in {wall:.1f}s")

    if not results:
        print("\n  No results -- are both services running?")
        return

    # ---- Print table -------------------------------------------------------
    print()
    print("=" * 65)
    hdr = f"  {'Backend':<24} | {'p50':>8} | {'p95':>8} | {'p99':>8} | {'RPS':>8}"
    sep = "  " + "-" * 62
    print(hdr)
    print(sep)
    for r in results:
        print(
            f"  {r['label']:<24} | "
            f"{r['p50']:>7.1f}ms | "
            f"{r['p95']:>7.1f}ms | "
            f"{r['p99']:>7.1f}ms | "
            f"{r['rps']:>7.1f}"
        )
    print("=" * 65)

    if len(results) == 2:
        fa, tr = results[0], results[1]
        p95_improvement = ((fa["p95"] - tr["p95"]) / fa["p95"]) * 100
        rps_gain        = tr["rps"] / fa["rps"]
        print()
        print(f"  Throughput gain    : {rps_gain:.2f}x")
        print(f"  p95 latency delta  : {p95_improvement:.1f}% faster via Triton")

    print()


if __name__ == "__main__":
    main()
