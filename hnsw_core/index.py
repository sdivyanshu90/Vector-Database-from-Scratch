from __future__ import annotations

import math
import random
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .distance import cosine_distance, l2_distance
from .types import Candidate, CandidateLike, Node, SearchResults


class HNSW:
    __slots__ = [
        "M",
        "M_max",
        "M_max0",
        "ef_construction",
        "ef_search",
        "m_L",
        "metric",
        "extend_candidates",
        "keep_pruned_connections",
        "use_heuristic",
        "nodes",
        "max_level",
        "enter_point",
        "dim",
        "next_id",
    ]

    def __init__(
        self,
        M: int = 16,
        M_max: Optional[int] = None,
        M_max0: Optional[int] = None,
        ef_construction: int = 200,
        ef_search: int = 128,
        m_L: Optional[float] = None,
        metric: str = "l2",
        extend_candidates: bool = True,
        keep_pruned_connections: bool = True,
        use_heuristic: bool = True,
    ) -> None:
        if M < 2:
            raise ValueError("M must be at least 2")
        if ef_construction < 1:
            raise ValueError("ef_construction must be positive")
        if ef_search < 1:
            raise ValueError("ef_search must be positive")
        if metric not in {"l2", "cosine"}:
            raise ValueError("metric must be 'l2' or 'cosine'")

        resolved_M_max = M if M_max is None else M_max
        resolved_M_max0 = (2 * M) if M_max0 is None else M_max0
        if resolved_M_max < M:
            raise ValueError("M_max must be at least M")
        if resolved_M_max0 < M:
            raise ValueError("M_max0 must be at least M")

        resolved_m_L = (1.0 / math.log(M)) if m_L is None else m_L
        if resolved_m_L <= 0.0:
            raise ValueError("m_L must be positive")

        self.M = M
        self.M_max = resolved_M_max
        self.M_max0 = resolved_M_max0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = resolved_m_L
        self.metric = metric
        self.extend_candidates = extend_candidates
        self.keep_pruned_connections = keep_pruned_connections
        self.use_heuristic = use_heuristic
        self.nodes: Dict[int, Node] = {}
        self.max_level = -1
        self.enter_point: Optional[int] = None
        self.dim: Optional[int] = None
        self.next_id = 0

    def __len__(self) -> int:
        return len(self.nodes)

    def _prepare_vector(self, vector: np.ndarray, allow_new_dim: bool = True) -> np.ndarray:
        vector_array = np.ascontiguousarray(np.asarray(vector, dtype=np.float32))
        if vector_array.ndim != 1:
            raise ValueError("vectors must be 1D")
        if vector_array.size == 0:
            raise ValueError("vectors must be non-empty")

        if self.dim is None:
            if not allow_new_dim:
                raise ValueError("cannot validate query dimensions on an empty index")
            self.dim = int(vector_array.shape[0])
        elif vector_array.shape[0] != self.dim:
            raise ValueError(f"expected vectors with dimension {self.dim}, got {vector_array.shape[0]}")
        return vector_array

    def _distance(self, query_vec: np.ndarray, vector: np.ndarray) -> float:
        if self.metric == "l2":
            return float(l2_distance(query_vec, vector))
        return float(cosine_distance(query_vec, vector))

    def _distance_to_ids(self, query_vec: np.ndarray, node_ids: Sequence[int]) -> np.ndarray:
        if not node_ids:
            return np.empty(0, dtype=np.float32)
        matrix = np.asarray([self.nodes[node_id].vector for node_id in node_ids], dtype=np.float32)
        if self.metric == "l2":
            return l2_distance(query_vec, matrix)
        return cosine_distance(query_vec, matrix)

    def _sample_level(self) -> int:
        value = random.uniform(0.0, 1.0)
        while value <= 0.0:
            value = random.uniform(0.0, 1.0)
        return int(math.floor(-math.log(value) * self.m_L))

    def _allocate_node_id(self, node_id: Optional[int]) -> int:
        if node_id is None:
            node_id = self.next_id
            while node_id in self.nodes:
                node_id += 1
            self.next_id = node_id + 1
            return node_id
        if node_id in self.nodes:
            raise ValueError(f"node id {node_id} already exists")
        self.next_id = max(self.next_id, node_id + 1)
        return node_id

    def _normalize_entry_points(self, entry_points: int | Sequence[int]) -> List[int]:
        if isinstance(entry_points, (int, np.integer)):
            normalized = [int(entry_points)]
        else:
            normalized = []
            seen = set()
            for entry_point in entry_points:
                entry_id = int(entry_point)
                if entry_id in seen:
                    continue
                seen.add(entry_id)
                normalized.append(entry_id)

        if not normalized:
            raise ValueError("entry_points must contain at least one node id")
        for entry_id in normalized:
            if entry_id not in self.nodes:
                raise KeyError(f"unknown entry point node id {entry_id}")
        return normalized

    def _normalize_candidate_ids(
        self,
        candidates: Iterable[CandidateLike],
        exclude_id: Optional[int] = None,
    ) -> List[int]:
        candidate_ids: List[int] = []
        seen = set()
        for candidate in candidates:
            node_id = int(candidate[1]) if isinstance(candidate, tuple) else int(candidate)
            if node_id == exclude_id or node_id in seen:
                continue
            if node_id not in self.nodes:
                raise KeyError(f"unknown candidate node id {node_id}")
            seen.add(node_id)
            candidate_ids.append(node_id)
        return candidate_ids

    def _ordered_candidates(self, query_vec: np.ndarray, candidate_ids: Sequence[int]) -> SearchResults:
        distances = self._distance_to_ids(query_vec, candidate_ids)
        ordered = [
            (float(distance), candidate_id)
            for distance, candidate_id in zip(distances.tolist(), candidate_ids)
        ]
        ordered.sort(key=lambda item: (item[0], item[1]))
        return ordered

    def _nearest_id(self, results: SearchResults) -> int:
        if not results:
            raise ValueError("results must contain at least one element")
        return results[0][1]

    def _result_ids(self, results: SearchResults) -> List[int]:
        return [node_id for _, node_id in results]

    def _max_connections_for_layer(self, layer: int) -> int:
        return self.M_max0 if layer == 0 else self.M_max

    def _neighbors(self, node_id: int, layer: int) -> List[int]:
        return self.nodes[node_id].edges.get(layer, [])

    def _passes_diversity_heuristic(
        self,
        candidate_id: int,
        candidate_distance: float,
        selected_ids: Sequence[int],
    ) -> bool:
        if not selected_ids:
            return True

        candidate_vector = self.nodes[candidate_id].vector
        selected_vectors = np.asarray([self.nodes[selected_id].vector for selected_id in selected_ids], dtype=np.float32)
        if self.metric == "l2":
            distances_to_selected = l2_distance(candidate_vector, selected_vectors)
        else:
            distances_to_selected = cosine_distance(candidate_vector, selected_vectors)
        return bool(np.all(candidate_distance < distances_to_selected))

    def _search_layer(
        self,
        query_vec: np.ndarray,
        entry_points: int | Sequence[int],
        ef: int,
        layer: int,
    ) -> SearchResults:
        if ef < 1:
            raise ValueError("ef must be positive")

        entry_ids = self._normalize_entry_points(entry_points)
        entry_distances = self._distance_to_ids(query_vec, entry_ids)
        candidate_queue: List[Candidate] = []
        top_candidates: List[Candidate] = []
        visited = set(entry_ids)

        for distance, entry_id in zip(entry_distances.tolist(), entry_ids):
            candidate = (float(distance), entry_id)
            heappush(candidate_queue, candidate)
            heappush(top_candidates, (-candidate[0], candidate[1]))

        while candidate_queue:
            current_distance, current_id = heappop(candidate_queue)
            furthest_distance = -top_candidates[0][0]
            if current_distance > furthest_distance:
                break

            neighbor_ids = [neighbor_id for neighbor_id in self._neighbors(current_id, layer) if neighbor_id not in visited]
            if not neighbor_ids:
                continue

            visited.update(neighbor_ids)
            neighbor_distances = self._distance_to_ids(query_vec, neighbor_ids)
            for neighbor_distance, neighbor_id in zip(neighbor_distances.tolist(), neighbor_ids):
                candidate_distance = float(neighbor_distance)
                furthest_distance = -top_candidates[0][0]
                if len(top_candidates) < ef or candidate_distance < furthest_distance:
                    heappush(candidate_queue, (candidate_distance, neighbor_id))
                    heappush(top_candidates, (-candidate_distance, neighbor_id))
                    if len(top_candidates) > ef:
                        heappop(top_candidates)

        results = [(-negative_distance, node_id) for negative_distance, node_id in top_candidates]
        results.sort(key=lambda item: (item[0], item[1]))
        return results

    def _select_neighbors_simple(
        self,
        query_vec: np.ndarray,
        candidates: Iterable[CandidateLike],
        M: int,
        exclude_id: Optional[int] = None,
    ) -> List[int]:
        if M < 1:
            raise ValueError("M must be positive")

        candidate_ids = self._normalize_candidate_ids(candidates, exclude_id=exclude_id)
        ordered_candidates = self._ordered_candidates(query_vec, candidate_ids)
        return [candidate_id for _, candidate_id in ordered_candidates[:M]]

    def _select_neighbors_heuristic(
        self,
        query_vec: np.ndarray,
        candidates: Iterable[CandidateLike],
        M: int,
        layer: int,
        extend_candidates: Optional[bool] = None,
        keep_pruned_connections: Optional[bool] = None,
        exclude_id: Optional[int] = None,
    ) -> List[int]:
        if M < 1:
            raise ValueError("M must be positive")

        extend_candidates_flag = self.extend_candidates if extend_candidates is None else extend_candidates
        keep_pruned_flag = self.keep_pruned_connections if keep_pruned_connections is None else keep_pruned_connections

        candidate_ids = self._normalize_candidate_ids(candidates, exclude_id=exclude_id)
        working_ids = list(candidate_ids)
        working_seen = set(candidate_ids)

        if extend_candidates_flag:
            for candidate_id in candidate_ids:
                for adjacent_id in self._neighbors(candidate_id, layer):
                    if adjacent_id == exclude_id or adjacent_id in working_seen:
                        continue
                    working_seen.add(adjacent_id)
                    working_ids.append(adjacent_id)

        working_queue: List[Candidate] = []
        for candidate in self._ordered_candidates(query_vec, working_ids):
            heappush(working_queue, candidate)

        discarded_queue: List[Candidate] = []
        selected: List[int] = []
        while working_queue and len(selected) < M:
            candidate_distance, candidate_id = heappop(working_queue)
            if self._passes_diversity_heuristic(candidate_id, candidate_distance, selected):
                selected.append(candidate_id)
            else:
                heappush(discarded_queue, (candidate_distance, candidate_id))

        if keep_pruned_flag:
            while discarded_queue and len(selected) < M:
                _, candidate_id = heappop(discarded_queue)
                selected.append(candidate_id)

        return selected

    def _select_neighbors(
        self,
        query_vec: np.ndarray,
        candidates: Iterable[CandidateLike],
        M: int,
        layer: int,
        exclude_id: Optional[int] = None,
    ) -> List[int]:
        if self.use_heuristic:
            return self._select_neighbors_heuristic(
                query_vec,
                candidates,
                M,
                layer,
                exclude_id=exclude_id,
            )
        return self._select_neighbors_simple(query_vec, candidates, M, exclude_id=exclude_id)

    def _shrink_connections(self, node_id: int, layer: int) -> None:
        neighbors = self._neighbors(node_id, layer)
        max_connections = self._max_connections_for_layer(layer)
        if len(neighbors) <= max_connections:
            return

        node = self.nodes[node_id]
        self.nodes[node_id].edges[layer] = self._select_neighbors(
            node.vector,
            neighbors,
            max_connections,
            layer,
            exclude_id=node_id,
        )

    def _greedy_route(
        self,
        query_vec: np.ndarray,
        entry_point: int,
        start_layer: int,
        stop_layer: int,
    ) -> int:
        current_entry = entry_point
        for layer in range(start_layer, stop_layer, -1):
            current_entry = self._nearest_id(self._search_layer(query_vec, current_entry, 1, layer))
        return current_entry

    def _link_bidirectional(self, node_id: int, neighbor_ids: Sequence[int], layer: int) -> None:
        self.nodes[node_id].edges[layer] = list(neighbor_ids)
        for neighbor_id in neighbor_ids:
            neighbor_edges = self.nodes[neighbor_id].edges.setdefault(layer, [])
            if node_id not in neighbor_edges:
                neighbor_edges.append(node_id)
            if len(neighbor_edges) > self._max_connections_for_layer(layer):
                self._shrink_connections(neighbor_id, layer)

    def insert(self, vector: np.ndarray, node_id: Optional[int] = None) -> int:
        vector_array = self._prepare_vector(vector)
        allocated_node_id = self._allocate_node_id(node_id)
        node_level = self._sample_level()
        new_node = Node(allocated_node_id, vector_array)
        self.nodes[allocated_node_id] = new_node

        if self.enter_point is None:
            self.enter_point = allocated_node_id
            self.max_level = node_level
            return allocated_node_id

        current_enter_point = self.enter_point
        current_max_level = self.max_level
        if current_max_level > node_level:
            current_enter_point = self._greedy_route(new_node.vector, current_enter_point, current_max_level, node_level)

        construction_entry_points: int | Sequence[int] = current_enter_point
        for layer in range(min(current_max_level, node_level), -1, -1):
            layer_results = self._search_layer(new_node.vector, construction_entry_points, self.ef_construction, layer)
            selected_neighbors = self._select_neighbors(
                new_node.vector,
                layer_results,
                self.M,
                layer,
                exclude_id=allocated_node_id,
            )
            self._link_bidirectional(allocated_node_id, selected_neighbors, layer)
            construction_entry_points = self._result_ids(layer_results)

        if node_level > current_max_level:
            self.enter_point = allocated_node_id
            self.max_level = node_level

        return allocated_node_id

    def add_item(self, vector: np.ndarray, node_id: Optional[int] = None) -> int:
        return self.insert(vector, node_id=node_id)

    def knn_search(self, query_vec: np.ndarray, k: int = 10, ef: Optional[int] = None) -> SearchResults:
        if k < 1:
            raise ValueError("k must be positive")
        if self.enter_point is None:
            return []

        query = self._prepare_vector(query_vec, allow_new_dim=False)
        current_entry = self.enter_point
        if self.max_level > 0:
            current_entry = self._greedy_route(query, current_entry, self.max_level, 0)

        search_ef = max(k, self.ef_search if ef is None else ef)
        layer_zero_results = self._search_layer(query, current_entry, search_ef, 0)
        return layer_zero_results[: min(k, len(layer_zero_results))]

    def search(self, query_vec: np.ndarray, k: int = 10, ef: Optional[int] = None) -> SearchResults:
        return self.knn_search(query_vec, k=k, ef=ef)