from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

Candidate = Tuple[float, int]
CandidateLike = int | Candidate
SearchResults = List[Candidate]


class Node:
    __slots__ = ["id", "vector", "edges"]

    def __init__(self, node_id: int, vector: np.ndarray) -> None:
        vector_array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
        if vector_array.ndim != 1:
            raise ValueError("node vectors must be 1D")
        if vector_array.size == 0:
            raise ValueError("node vectors must be non-empty")

        self.id = node_id
        self.vector = vector_array
        self.edges: Dict[int, List[int]] = {}


class BenchmarkResult:
    __slots__ = [
        "total_vectors",
        "dimensions",
        "query_count",
        "top_k",
        "build_seconds",
        "build_us_per_vector",
        "recall",
        "ann_mean_ms",
        "ann_p95_ms",
        "exact_mean_ms",
        "exact_p95_ms",
    ]

    def __init__(
        self,
        total_vectors: int,
        dimensions: int,
        query_count: int,
        top_k: int,
        build_seconds: float,
        build_us_per_vector: float,
        recall: float,
        ann_mean_ms: float,
        ann_p95_ms: float,
        exact_mean_ms: float,
        exact_p95_ms: float,
    ) -> None:
        self.total_vectors = total_vectors
        self.dimensions = dimensions
        self.query_count = query_count
        self.top_k = top_k
        self.build_seconds = build_seconds
        self.build_us_per_vector = build_us_per_vector
        self.recall = recall
        self.ann_mean_ms = ann_mean_ms
        self.ann_p95_ms = ann_p95_ms
        self.exact_mean_ms = exact_mean_ms
        self.exact_p95_ms = exact_p95_ms