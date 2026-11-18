"""Road network representation, CSR adjacency, and fast bounded shortest-path routing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.geo.geometry import compute_bearings_deg
from src.geo.projection import TripProjector


@dataclass
class RoadNetwork:
    """Directed road network in projected metric coordinates (meters)."""

    nodes_lonlat: np.ndarray  # (N, 2) [lon, lat]
    arcs: np.ndarray  # (M, 2) [from_node, to_node]
    projector: TripProjector

    node_x: np.ndarray = field(init=False)
    node_y: np.ndarray = field(init=False)
    from_nodes: np.ndarray = field(init=False)
    to_nodes: np.ndarray = field(init=False)
    x1: np.ndarray = field(init=False)
    y1: np.ndarray = field(init=False)
    x2: np.ndarray = field(init=False)
    y2: np.ndarray = field(init=False)
    arc_lengths: np.ndarray = field(init=False)
    arc_bearings: np.ndarray = field(init=False)

    # CSR outgoing adjacency: for node u, edges are indptr[u]:indptr[u+1]
    out_indptr: np.ndarray = field(init=False)
    out_to_nodes: np.ndarray = field(init=False)
    out_arc_ids: np.ndarray = field(init=False)
    out_lengths: np.ndarray = field(init=False)

    # Topological node/arc properties
    out_degree: np.ndarray = field(init=False)
    in_degree: np.ndarray = field(init=False)
    has_reverse_arc: np.ndarray = field(init=False)

    # Per-network Dijkstra distance cache: (source_node, cutoff_bucket) -> Dict[node, dist_m]
    _dist_cache: Dict[Tuple[int, int], Dict[int, float]] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.node_x, self.node_y = self.projector.to_xy(
            self.nodes_lonlat[:, 0], self.nodes_lonlat[:, 1]
        )
        n_nodes = len(self.nodes_lonlat)
        n_arcs = len(self.arcs)

        self.from_nodes = self.arcs[:, 0].astype(np.int32, copy=False)
        self.to_nodes = self.arcs[:, 1].astype(np.int32, copy=False)

        self.x1 = self.node_x[self.from_nodes]
        self.y1 = self.node_y[self.from_nodes]
        self.x2 = self.node_x[self.to_nodes]
        self.y2 = self.node_y[self.to_nodes]

        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        self.arc_lengths = np.hypot(dx, dy)
        self.arc_bearings = compute_bearings_deg(dx, dy)

        # Degrees
        self.out_degree = np.bincount(self.from_nodes, minlength=n_nodes).astype(np.int32)
        self.in_degree = np.bincount(self.to_nodes, minlength=n_nodes).astype(np.int32)

        # Build CSR outgoing adjacency sorted by from_node
        order = np.argsort(self.from_nodes, kind="mergesort")
        self.out_to_nodes = self.to_nodes[order]
        self.out_arc_ids = order.astype(np.int32)
        self.out_lengths = self.arc_lengths[order]

        self.out_indptr = np.zeros(n_nodes + 1, dtype=np.int32)
        np.cumsum(self.out_degree, out=self.out_indptr[1:])

        # Fast vectorized check for reverse arc (to_node -> from_node)
        # Pack (u, v) into int64 key: u * n_nodes + v
        n64 = np.int64(max(n_nodes, 1))
        fwd_keys = self.from_nodes.astype(np.int64) * n64 + self.to_nodes.astype(np.int64)
        rev_keys = self.to_nodes.astype(np.int64) * n64 + self.from_nodes.astype(np.int64)
        fwd_sorted = np.sort(fwd_keys)
        idx = np.searchsorted(fwd_sorted, rev_keys)
        idx_clipped = np.minimum(idx, max(n_arcs - 1, 0))
        if n_arcs > 0:
            self.has_reverse_arc = (fwd_sorted[idx_clipped] == rev_keys).astype(np.int8)
        else:
            self.has_reverse_arc = np.zeros(0, dtype=np.int8)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes_lonlat)

    @property
    def num_arcs(self) -> int:
        return len(self.arcs)

    def clear_cache(self) -> None:
        self._dist_cache.clear()

    def graph_connectivity_stats(self) -> Dict[str, int]:
        """Compute isolated nodes, weakly and strongly connected component counts."""
        n = self.num_nodes
        isolated = int(np.sum((self.out_degree == 0) & (self.in_degree == 0)))
        if n == 0 or self.num_arcs == 0:
            return {
                "isolated_nodes": isolated,
                "weakly_connected_components": n,
                "strongly_connected_components": n,
                "largest_wcc_nodes": 1 if n > 0 else 0,
                "largest_scc_nodes": 1 if n > 0 else 0,
            }
        mat = csr_matrix(
            (np.ones(self.num_arcs, dtype=np.int8), (self.from_nodes, self.to_nodes)),
            shape=(n, n),
        )
        n_wcc, wcc_labels = connected_components(mat, directed=True, connection="weak")
        n_scc, scc_labels = connected_components(mat, directed=True, connection="strong")
        largest_wcc = int(np.bincount(wcc_labels).max()) if n_wcc > 0 else 0
        largest_scc = int(np.bincount(scc_labels).max()) if n_scc > 0 else 0
        return {
            "isolated_nodes": isolated,
            "weakly_connected_components": int(n_wcc),
            "strongly_connected_components": int(n_scc),
            "largest_wcc_nodes": largest_wcc,
            "largest_scc_nodes": largest_scc,
        }

    def is_directly_connected(self, arc_u: int, arc_v: int) -> bool:
        """Return True if arc_v starts at arc_u's end node or is the same arc."""
        if arc_u == arc_v:
            return True
        return int(self.to_nodes[arc_u]) == int(self.from_nodes[arc_v])

    def bounded_dijkstra_from_node(
        self,
        source_node: int,
        max_dist_m: float,
        target_nodes: Optional[Set[int]] = None,
    ) -> Dict[int, float]:
        """Run bounded Dijkstra from `source_node` up to `max_dist_m`.

        Caches full-cutoff results (when target_nodes is None or bucketed) for rapid reuse
        across consecutive Viterbi transitions.
        """
        bucket = int(np.ceil(max_dist_m / 50.0) * 50)
        cache_key = (int(source_node), bucket)
        cached = self._dist_cache.get(cache_key)
        if cached is not None:
            return cached

        # Check if we already have a larger cutoff cached for this source_node
        if target_nodes is None:
            for (s_node, b_dist), dist_map in self._dist_cache.items():
                if s_node == source_node and b_dist >= bucket:
                    return dist_map

        indptr = self.out_indptr
        to_nodes = self.out_to_nodes
        lengths = self.out_lengths

        dist: Dict[int, float] = {int(source_node): 0.0}
        heap: List[Tuple[float, int]] = [(0.0, int(source_node))]
        limit = float(bucket)

        remaining_targets = set(target_nodes) if target_nodes is not None else None
        if remaining_targets is not None and source_node in remaining_targets:
            remaining_targets.discard(source_node)

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, float("inf")):
                continue
            if d > limit:
                break
            if remaining_targets is not None:
                remaining_targets.discard(u)
                if not remaining_targets:
                    break

            start = int(indptr[u])
            end = int(indptr[u + 1])
            for idx in range(start, end):
                v = int(to_nodes[idx])
                nd = d + float(lengths[idx])
                if nd <= limit and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))

        if target_nodes is None:
            # Guard cache memory for very long trajectories
            if len(self._dist_cache) > 8000:
                self._dist_cache.clear()
            self._dist_cache[cache_key] = dist
        return dist

    def shortest_node_distance(
        self,
        source_node: int,
        target_node: int,
        max_dist_m: float,
    ) -> float:
        """Return directed shortest-path distance (m) from source_node to target_node, or inf."""
        if source_node == target_node:
            return 0.0
        dist_map = self.bounded_dijkstra_from_node(source_node, max_dist_m)
        return dist_map.get(int(target_node), float("inf"))

    def shortest_arc_path(
        self,
        arc_u: int,
        arc_v: int,
        max_dist_m: float = 2000.0,
    ) -> Optional[List[int]]:
        """Reconstruct directed sequence of intermediate arc IDs strictly between arc_u and arc_v.

        Returns:
            [] if arc_u == arc_v or to_node(arc_u) == from_node(arc_v) (directly connected),
            [a1, a2, ...] if reachable within max_dist_m,
            None if unreachable within max_dist_m.
        """
        if arc_u == arc_v:
            return []
        src = int(self.to_nodes[arc_u])
        dst = int(self.from_nodes[arc_v])
        if src == dst:
            return []

        indptr = self.out_indptr
        to_nodes = self.out_to_nodes
        arc_ids = self.out_arc_ids
        lengths = self.out_lengths

        dist: Dict[int, float] = {src: 0.0}
        prev_node: Dict[int, int] = {}
        prev_arc: Dict[int, int] = {}
        heap: List[Tuple[float, int]] = [(0.0, src)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                path: List[int] = []
                curr = dst
                while curr != src:
                    path.append(prev_arc[curr])
                    curr = prev_node[curr]
                path.reverse()
                return path
            if d > dist.get(u, float("inf")) or d > max_dist_m:
                continue

            start = int(indptr[u])
            end = int(indptr[u + 1])
            for idx in range(start, end):
                v = int(to_nodes[idx])
                nd = d + float(lengths[idx])
                if nd <= max_dist_m and nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev_node[v] = u
                    prev_arc[v] = int(arc_ids[idx])
                    heapq.heappush(heap, (nd, v))
        return None

    def route_distance_between_projections(
        self,
        arc_u: int,
        frac_u: float,
        arc_v: int,
        frac_v: float,
        max_dist_m: float,
    ) -> float:
        """Directed road-network distance between two projected positions (arc_u, frac_u) -> (arc_v, frac_v).

        Handles same-arc forward/backward motion, reverse-arc U-turns, and multi-hop shortest paths.
        """
        len_u = float(self.arc_lengths[arc_u])
        len_v = float(self.arc_lengths[arc_v])

        if arc_u == arc_v:
            delta = (frac_v - frac_u) * len_u
            if delta >= -15.0:
                # Allow slight GPS jitter backwards on the same arc (up to 15m)
                return max(0.0, delta)
            # Significant backward jump on same arc requires looping around the network
            rem_u = (1.0 - frac_u) * len_u
            pref_v = frac_v * len_v
            node_d = self.shortest_node_distance(
                int(self.to_nodes[arc_u]),
                int(self.from_nodes[arc_v]),
                max(0.0, max_dist_m - rem_u - pref_v),
            )
            return rem_u + node_d + pref_v

        rem_u = (1.0 - frac_u) * len_u
        pref_v = frac_v * len_v
        budget = max_dist_m - rem_u - pref_v
        if budget < 0.0:
            # Still allow direct connection if total <= max_dist_m + 50m slack
            if int(self.to_nodes[arc_u]) == int(self.from_nodes[arc_v]):
                return rem_u + pref_v
            return float("inf")

        node_d = self.shortest_node_distance(
            int(self.to_nodes[arc_u]),
            int(self.from_nodes[arc_v]),
            budget,
        )
        if np.isinf(node_d):
            return float("inf")
        return rem_u + node_d + pref_v

    def get_arc_lonlat_coords(self, arc_ids: Iterable[int]) -> List[List[Tuple[float, float]]]:
        """Return list of [(lat, lon), (lat, lon)] polylines for Folium/Leaflet rendering."""
        coords: List[List[Tuple[float, float]]] = []
        for aid in arc_ids:
            u = int(self.from_nodes[aid])
            v = int(self.to_nodes[aid])
            coords.append(
                [
                    (float(self.nodes_lonlat[u, 1]), float(self.nodes_lonlat[u, 0])),
                    (float(self.nodes_lonlat[v, 1]), float(self.nodes_lonlat[v, 0])),
                ]
            )
        return coords
