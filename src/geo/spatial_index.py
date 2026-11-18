"""Spatial indexing over road segments using Shapely 2.0 vectorized STRtree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import shapely
from shapely import STRtree, box

from src.geo.geometry import point_to_segments_projection
from src.geo.road_graph import RoadNetwork


@dataclass
class CandidateSet:
    """Candidate road segments for a single GPS observation."""

    obs_idx: int
    arc_ids: np.ndarray  # (K,) int32
    dist_m: np.ndarray  # (K,) float64 perpendicular distance
    frac: np.ndarray  # (K,) float64 clamped projection along arc [0, 1]
    raw_frac: np.ndarray  # (K,) float64 unclamped projection
    proj_x: np.ndarray  # (K,) float64
    proj_y: np.ndarray  # (K,) float64


class RoadSpatialIndex:
    """Fast R-tree spatial index over metric road segment LineStrings."""

    def __init__(self, network: RoadNetwork) -> None:
        self.network = network
        n_arcs = network.num_arcs
        if n_arcs > 0:
            # Vectorized construction of (M, 2, 2) coordinate array -> Shapely LineStrings
            coords = np.empty((n_arcs, 2, 2), dtype=np.float64)
            coords[:, 0, 0] = network.x1
            coords[:, 0, 1] = network.y1
            coords[:, 1, 0] = network.x2
            coords[:, 1, 1] = network.y2
            self._geoms = shapely.linestrings(coords)
            self._tree = STRtree(self._geoms)
        else:
            self._geoms = np.array([], dtype=object)
            self._tree = None

    def query_point(
        self,
        obs_idx: int,
        px: float,
        py: float,
        radius_m: float = 100.0,
        fallback_radius_m: float = 300.0,
        top_k: int = 10,
    ) -> CandidateSet:
        """Find up to `top_k` candidate directed arcs near (px, py) within `radius_m`."""
        if self._tree is None or self.network.num_arcs == 0:
            empty_i = np.zeros(0, dtype=np.int32)
            empty_f = np.zeros(0, dtype=np.float64)
            return CandidateSet(obs_idx, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f)

        net = self.network

        def _search_radius(r: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            search_box = box(px - r, py - r, px + r, py + r)
            cand_indices = self._tree.query(search_box)
            if len(cand_indices) == 0:
                empty_i = np.zeros(0, dtype=np.int32)
                empty_f = np.zeros(0, dtype=np.float64)
                return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f

            cand_indices = np.asarray(cand_indices, dtype=np.int32)
            dist_m, frac, raw_frac, proj_x, proj_y = point_to_segments_projection(
                px,
                py,
                net.x1[cand_indices],
                net.y1[cand_indices],
                net.x2[cand_indices],
                net.y2[cand_indices],
            )
            mask = dist_m <= r
            return (
                cand_indices[mask],
                dist_m[mask],
                frac[mask],
                raw_frac[mask],
                proj_x[mask],
                proj_y[mask],
            )

        arc_ids, dist_m, frac, raw_frac, proj_x, proj_y = _search_radius(radius_m)

        # Fallback to wider radius if no candidates found inside primary radius
        if len(arc_ids) == 0 and fallback_radius_m > radius_m:
            arc_ids, dist_m, frac, raw_frac, proj_x, proj_y = _search_radius(fallback_radius_m)

        # Final fallback: single nearest segment in STRtree (guarantees non-empty unless graph empty)
        if len(arc_ids) == 0:
            nearest_idx = int(self._tree.nearest(shapely.points(px, py)))
            cand_indices = np.array([nearest_idx], dtype=np.int32)
            dist_m, frac, raw_frac, proj_x, proj_y = point_to_segments_projection(
                px,
                py,
                net.x1[cand_indices],
                net.y1[cand_indices],
                net.x2[cand_indices],
                net.y2[cand_indices],
            )
            arc_ids = cand_indices

        # Sort by perpendicular distance ascending and retain top_k
        order = np.argsort(dist_m, kind="mergesort")[:top_k]
        return CandidateSet(
            obs_idx=obs_idx,
            arc_ids=arc_ids[order],
            dist_m=dist_m[order],
            frac=frac[order],
            raw_frac=raw_frac[order],
            proj_x=proj_x[order],
            proj_y=proj_y[order],
        )

    def query_trajectory(
        self,
        gps_x: np.ndarray,
        gps_y: np.ndarray,
        radius_m: float = 100.0,
        fallback_radius_m: float = 300.0,
        top_k: int = 10,
    ) -> List[CandidateSet]:
        """Generate candidate sets for all GPS points along a trajectory."""
        n_obs = len(gps_x)
        if self._tree is None or self.network.num_arcs == 0:
            return [
                CandidateSet(
                    i,
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                    np.zeros(0, dtype=np.float64),
                )
                for i in range(n_obs)
            ]

        # Batch R-tree query for all points using primary radius boxes
        boxes = shapely.box(
            gps_x - radius_m,
            gps_y - radius_m,
            gps_x + radius_m,
            gps_y + radius_m,
        )
        # pairs has shape (2, P): row 0 = query_idx, row 1 = tree_geom_idx
        pairs = self._tree.query(boxes)

        # Group tree indices by obs_idx efficiently via split
        if pairs.shape[1] > 0:
            q_idx = pairs[0]
            g_idx = pairs[1].astype(np.int32)
            # STRtree.query returns pairs ordered by q_idx, ensure sorted
            if not np.all(q_idx[:-1] <= q_idx[1:]):
                ord_p = np.argsort(q_idx, kind="mergesort")
                q_idx = q_idx[ord_p]
                g_idx = g_idx[ord_p]
            counts = np.bincount(q_idx, minlength=n_obs)
            splits = np.cumsum(counts)[:-1]
            grouped = np.split(g_idx, splits)
        else:
            grouped = [np.zeros(0, dtype=np.int32) for _ in range(n_obs)]

        net = self.network
        results: List[CandidateSet] = []
        for i in range(n_obs):
            px = float(gps_x[i])
            py = float(gps_y[i])
            cand_indices = grouped[i]
            if len(cand_indices) > 0:
                dist_m, frac, raw_frac, proj_x, proj_y = point_to_segments_projection(
                    px,
                    py,
                    net.x1[cand_indices],
                    net.y1[cand_indices],
                    net.x2[cand_indices],
                    net.y2[cand_indices],
                )
                mask = dist_m <= radius_m
                if np.any(mask):
                    cand_indices = cand_indices[mask]
                    dist_m = dist_m[mask]
                    frac = frac[mask]
                    raw_frac = raw_frac[mask]
                    proj_x = proj_x[mask]
                    proj_y = proj_y[mask]
                    order = np.argsort(dist_m, kind="mergesort")[:top_k]
                    results.append(
                        CandidateSet(
                            obs_idx=i,
                            arc_ids=cand_indices[order],
                            dist_m=dist_m[order],
                            frac=frac[order],
                            raw_frac=raw_frac[order],
                            proj_x=proj_x[order],
                            proj_y=proj_y[order],
                        )
                    )
                    continue

            # Fallback for sparse points
            results.append(
                self.query_point(
                    obs_idx=i,
                    px=px,
                    py=py,
                    radius_m=radius_m,
                    fallback_radius_m=fallback_radius_m,
                    top_k=top_k,
                )
            )
        return results

    def query_arcs_in_bbox(
        self, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> np.ndarray:
        """Return arc IDs whose bounding boxes intersect the given metric bounding box."""
        if self._tree is None:
            return np.zeros(0, dtype=np.int32)
        return np.asarray(self._tree.query(box(min_x, min_y, max_x, max_y)), dtype=np.int32)
