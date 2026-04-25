from __future__ import annotations

import numpy as np


def l2_distance(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray | float:
    query = np.asarray(query_vec, dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_vec must be a 1D float32 vector")

    single_vector = matrix.ndim == 1
    if single_vector:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("vectors must have shape (n, dim) matching query_vec")

    deltas = matrix - query.reshape(1, -1)
    distances = np.sqrt(np.sum(deltas * deltas, axis=1, dtype=np.float32))
    if single_vector:
        return float(distances[0])
    return distances.astype(np.float32, copy=False)


def cosine_distance(query_vec: np.ndarray, vectors: np.ndarray) -> np.ndarray | float:
    query = np.asarray(query_vec, dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_vec must be a 1D float32 vector")

    single_vector = matrix.ndim == 1
    if single_vector:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[1] != query.shape[0]:
        raise ValueError("vectors must have shape (n, dim) matching query_vec")

    query_norm_sq = np.sum(query * query, dtype=np.float32)
    vector_norm_sq = np.sum(matrix * matrix, axis=1, dtype=np.float32)
    numerator = matrix @ query
    denominator = np.sqrt(np.maximum(vector_norm_sq * query_norm_sq, np.float32(0.0)))

    distances = np.ones(matrix.shape[0], dtype=np.float32)
    valid = denominator > 0.0
    distances[valid] = 1.0 - (numerator[valid] / denominator[valid])

    both_zero = (~valid) & (vector_norm_sq == 0.0) & (query_norm_sq == 0.0)
    distances[both_zero] = 0.0

    if single_vector:
        return float(distances[0])
    return distances