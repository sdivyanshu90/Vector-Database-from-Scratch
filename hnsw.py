from __future__ import annotations

from hnsw_core import (
    BenchmarkResult,
    Candidate,
    CandidateLike,
    HNSW,
    Node,
    SearchResults,
    benchmark_hnsw,
    cosine_distance,
    exact_search,
    l2_distance,
    mean_recall,
)

__all__ = [
    "BenchmarkResult",
    "Candidate",
    "CandidateLike",
    "HNSW",
    "Node",
    "SearchResults",
    "benchmark_hnsw",
    "cosine_distance",
    "exact_search",
    "l2_distance",
    "mean_recall",
]


if __name__ == "__main__":
    metrics = benchmark_hnsw()
    assert metrics.recall > 0.95, f"Recall below target: {metrics.recall:.4f}"

    print(f"Indexed {metrics.total_vectors} vectors of dimension {metrics.dimensions}")
    print(f"Build time: {metrics.build_seconds:.3f}s ({metrics.build_us_per_vector:.3f} us/vector)")
    print(f"Recall@{metrics.top_k}: {metrics.recall:.4f}")
    print(f"ANN latency: mean={metrics.ann_mean_ms:.3f} ms, p95={metrics.ann_p95_ms:.3f} ms")
    print(f"Exact latency: mean={metrics.exact_mean_ms:.3f} ms, p95={metrics.exact_p95_ms:.3f} ms")