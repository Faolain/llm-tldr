#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from tldr.semantic import compute_embedding


@dataclass(frozen=True)
class LatencyStats:
    p50_ms: float
    p95_ms: float
    mean_ms: float


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _latency_stats(samples_ms: list[float]) -> LatencyStats:
    arr = np.asarray(samples_ms, dtype=np.float64)
    return LatencyStats(
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        mean_ms=float(arr.mean()),
    )


def _run_backend(
    *,
    backend: str,
    model: str,
    query: str,
    runs: int,
    device: str | None,
) -> tuple[LatencyStats, list[np.ndarray]]:
    os.environ["TLDR_BACKEND"] = backend
    latencies_ms: list[float] = []
    embeddings: list[np.ndarray] = []

    for _ in range(runs):
        t0 = time.perf_counter()
        emb = compute_embedding(query, model_name=model, device=device)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        embeddings.append(np.asarray(emb, dtype=np.float32))

    return _latency_stats(latencies_ms), embeddings


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark semantic query latency: PyTorch vs ONNX Runtime.")
    ap.add_argument("--model", default="bge-large-en-v1.5", help="Model key or HF model id.")
    ap.add_argument("--runs", type=int, default=10, help="Number of identical queries per backend.")
    ap.add_argument("--query", default="Represent this code search query: find authentication token validation logic")
    ap.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.99,
        help="Minimum cosine similarity expected between PyTorch and ONNX embeddings.",
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("TLDR_DEVICE"),
        help="Optional device override passed to embedding calls (cpu/cuda/mps).",
    )
    args = ap.parse_args()

    try:
        pytorch_stats, pytorch_embeddings = _run_backend(
            backend="pytorch",
            model=args.model,
            query=args.query,
            runs=max(1, int(args.runs)),
            device=args.device,
        )
        onnx_stats, onnx_embeddings = _run_backend(
            backend="onnx",
            model=args.model,
            query=args.query,
            runs=max(1, int(args.runs)),
            device=args.device,
        )
    except ImportError as exc:
        print(f"Missing dependency for ONNX benchmark: {exc}", file=sys.stderr)
        print(
            "Install with: uv pip install onnxruntime optimum[onnxruntime] transformers tokenizers",
            file=sys.stderr,
        )
        return 2

    cosines = [
        _cosine_similarity(p_emb, o_emb)
        for p_emb, o_emb in zip(pytorch_embeddings, onnx_embeddings)
    ]
    min_cos = float(min(cosines))
    mean_cos = float(np.mean(cosines))

    print()
    print("Latency Comparison (ms)")
    print(f"{'Backend':<10} {'p50':>10} {'p95':>10} {'mean':>10}")
    print(f"{'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    print(f"{'pytorch':<10} {pytorch_stats.p50_ms:>10.2f} {pytorch_stats.p95_ms:>10.2f} {pytorch_stats.mean_ms:>10.2f}")
    print(f"{'onnx':<10} {onnx_stats.p50_ms:>10.2f} {onnx_stats.p95_ms:>10.2f} {onnx_stats.mean_ms:>10.2f}")

    speedup = pytorch_stats.mean_ms / onnx_stats.mean_ms if onnx_stats.mean_ms > 0 else 0.0
    print()
    print(f"Mean speedup (PyTorch / ONNX): {speedup:.2f}x")
    print(f"Cosine similarity (min/mean): {min_cos:.6f} / {mean_cos:.6f}")

    if min_cos < float(args.cosine_threshold):
        print(
            f"FAIL: embedding similarity below threshold {args.cosine_threshold:.3f}",
            file=sys.stderr,
        )
        return 1

    print(f"PASS: embedding similarity >= {args.cosine_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
