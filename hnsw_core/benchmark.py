from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np

from .distance import cosine_distance, l2_distance
from .index import HNSW
from .types import BenchmarkResult, SearchResults
from .utils import chunk_ids, now_ns


def exact_search(
    query_vec: np.ndarray,
    vectors: np.ndarray,
    ids: np.ndarray,
    k: int,
    metric: str,
) -> SearchResults:
    if k < 1:
        raise ValueError("k must be positive")
    if metric == "l2":
        distances = l2_distance(query_vec, vectors)
    elif metric == "cosine":
        distances = cosine_distance(query_vec, vectors)
    else:
        raise ValueError("metric must be 'l2' or 'cosine'")

    order = np.lexsort((ids, distances))[:k]
    return [(float(distances[index]), int(ids[index])) for index in order.tolist()]


def mean_recall(
    approximate_results: Sequence[SearchResults],
    exact_results: Sequence[SearchResults],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if len(approximate_results) != len(exact_results):
        raise ValueError("approximate and exact result lists must have equal length")

    recalls = np.empty(len(approximate_results), dtype=np.float32)
    for index, (approximate, exact) in enumerate(zip(approximate_results, exact_results)):
        approximate_ids = {node_id for _, node_id in approximate[:k]}
        exact_ids = {node_id for _, node_id in exact[:k]}
        recalls[index] = len(approximate_ids & exact_ids) / float(k)
    return float(np.mean(recalls))


def measure_chunked_latency_ms(
    query_ids: Sequence[int],
    chunk_size: int,
    repeats: int,
    evaluator,
) -> np.ndarray:
    chunk_latencies_ms = np.empty(len(chunk_ids(query_ids, chunk_size)), dtype=np.float64)
    for chunk_index, chunk in enumerate(chunk_ids(query_ids, chunk_size)):
        start_ns = now_ns()
        for _ in range(repeats):
            for query_id in chunk:
                evaluator(query_id)
        elapsed_ns = now_ns() - start_ns
        chunk_latencies_ms[chunk_index] = elapsed_ns / float(len(chunk) * repeats) / 1_000_000.0
    return chunk_latencies_ms


def benchmark_hnsw(
    total_vectors: int = 10_000,
    dimensions: int = 128,
    query_count: int = 100,
    top_k: int = 1,
    seed: int = 7,
) -> BenchmarkResult:
    random.seed(seed)
    rng = np.random.default_rng(seed)

    dataset = rng.standard_normal((total_vectors, dimensions)).astype(np.float32)
    dataset_ids = np.arange(total_vectors, dtype=np.int64)
    graph_degree = 24
    index = HNSW(
        M=graph_degree,
        M_max=graph_degree,
        M_max0=2 * graph_degree,
        ef_construction=64,
        ef_search=64,
        m_L=1.0 / math.log(graph_degree),
        metric="l2",
        extend_candidates=False,
        keep_pruned_connections=False,
        use_heuristic=True,
    )

    build_start_ns = now_ns()
    for vector_id, vector in enumerate(dataset):
        index.insert(vector, vector_id)
    build_elapsed_ns = now_ns() - build_start_ns

    query_ids = rng.choice(total_vectors, size=query_count, replace=False).tolist()
    approximate_results: list[SearchResults] = []
    exact_results: list[SearchResults] = []
    for query_id in query_ids:
        query_vector = dataset[query_id]
        approximate_results.append(index.knn_search(query_vector, k=top_k))
        exact_results.append(exact_search(query_vector, dataset, dataset_ids, top_k, index.metric))

    recall = mean_recall(approximate_results, exact_results, top_k)

    chunk_size = max(1, query_count // 5)
    repeats = 20
    ann_latency_ms = measure_chunked_latency_ms(
        query_ids,
        chunk_size,
        repeats,
        lambda query_id: index.knn_search(dataset[query_id], k=top_k),
    )
    exact_latency_ms = measure_chunked_latency_ms(
        query_ids,
        chunk_size,
        repeats,
        lambda query_id: exact_search(dataset[query_id], dataset, dataset_ids, top_k, index.metric),
    )

    return BenchmarkResult(
        total_vectors=total_vectors,
        dimensions=dimensions,
        query_count=query_count,
        top_k=top_k,
        build_seconds=build_elapsed_ns / 1_000_000_000.0,
        build_us_per_vector=build_elapsed_ns / total_vectors / 1_000.0,
        recall=recall,
        ann_mean_ms=float(np.mean(ann_latency_ms)),
        ann_p95_ms=float(np.percentile(ann_latency_ms, 95)),
        exact_mean_ms=float(np.mean(exact_latency_ms)),
        exact_p95_ms=float(np.percentile(exact_latency_ms, 95)),
    )