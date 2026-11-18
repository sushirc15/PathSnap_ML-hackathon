"""Monotonic dynamic-programming alignment between GPS observations and `.route` arc sequences.

IMPORTANT:
- This module is used ONLY for generating training labels and offline/server-side evaluation labels.
- It MUST NEVER be called during prediction-time feature extraction or model inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.data.preprocessing import PreparedTrip
from src.geo.geometry import angular_diff_deg, point_to_segments_projection
from src.geo.spatial_index import CandidateSet


@dataclass
class RouteAlignmentResult:
    """Aligned observation-to-reference-arc mapping with quality diagnostics."""

    trip_id: str
    aligned_route_pos: np.ndarray  # (T,) index into `.route` array
    aligned_arc_ids: np.ndarray  # (T,) arc_id = route[aligned_route_pos]
    aligned_dist_m: np.ndarray  # (T,) perpendicular distance to aligned reference arc
    aligned_heading_diff_deg: np.ndarray  # (T,) heading diff to aligned reference arc
    is_low_confidence: np.ndarray  # (T,) bool: high distance or ambiguous alignment
    candidate_hit: np.ndarray  # (T,) bool: whether aligned_arc_id is in generated candidate set

    @property
    def candidate_miss_rate(self) -> float:
        if len(self.candidate_hit) == 0:
            return 0.0
        return float(1.0 - np.mean(self.candidate_hit))

    @property
    def low_confidence_rate(self) -> float:
        if len(self.is_low_confidence) == 0:
            return 0.0
        return float(np.mean(self.is_low_confidence))


def align_gps_to_reference_route(
    trip: PreparedTrip,
    route_arcs: np.ndarray,
    candidates: List[CandidateSet] | None = None,
    sigma_dist_m: float = 20.0,
    heading_weight: float = 1.2,
    progression_weight: float = 0.25,
    max_step_window: int = 80,
    low_conf_dist_m: float = 65.0,
) -> RouteAlignmentResult:
    """Align T GPS observations monotonically to R chronological reference-route arcs.

    Uses banded dynamic programming over route positions r in [0, R-1] so that:
    - Repeated road segments (loops/revisits) are distinguished by their chronological route index r.
    - Every observation t maps to a route position r_t >= r_{t-1} (monotonic progression).
    - Emission cost penalizes perpendicular distance and heading mismatch.
    - Transition cost penalizes discrepancies between route-length advance and observed GPS displacement,
      while widening the search window automatically across GPS sampling gaps.
    """
    n_obs = len(trip.gps_x)
    n_route = len(route_arcs)

    if n_obs == 0 or n_route == 0:
        return RouteAlignmentResult(
            trip_id=trip.trip_id,
            aligned_route_pos=np.zeros(n_obs, dtype=np.int32),
            aligned_arc_ids=np.full(n_obs, -1, dtype=np.int32),
            aligned_dist_m=np.full(n_obs, np.nan, dtype=np.float64),
            aligned_heading_diff_deg=np.full(n_obs, np.nan, dtype=np.float64),
            is_low_confidence=np.ones(n_obs, dtype=bool),
            candidate_hit=np.zeros(n_obs, dtype=bool),
        )

    net = trip.network
    r_arcs = np.asarray(route_arcs, dtype=np.int32)
    rx1 = net.x1[r_arcs]
    ry1 = net.y1[r_arcs]
    rx2 = net.x2[r_arcs]
    ry2 = net.y2[r_arcs]
    r_lens = net.arc_lengths[r_arcs]
    r_bearings = net.arc_bearings[r_arcs]

    # Cumulative distance along the reference route up to the start of each route entry
    cum_len = np.zeros(n_route, dtype=np.float64)
    if n_route > 1:
        cum_len[1:] = np.cumsum(r_lens[:-1])

    # Precompute emission cost matrix (T, R), distances (T, R), and along-route positions (T, R)
    dist_mat = np.empty((n_obs, n_route), dtype=np.float64)
    route_pos_m = np.empty((n_obs, n_route), dtype=np.float64)
    hdiff_mat = np.empty((n_obs, n_route), dtype=np.float64)

    for t in range(n_obs):
        d_m, frac, _, _, _ = point_to_segments_projection(
            float(trip.gps_x[t]), float(trip.gps_y[t]), rx1, ry1, rx2, ry2
        )
        dist_mat[t] = d_m
        route_pos_m[t] = cum_len + frac * r_lens
        hdiff_mat[t] = angular_diff_deg(float(trip.heading_deg[t]), r_bearings)

    # Emission cost: squared distance normalized by sigma_dist_m + heading mismatch
    # Cap extreme outlier distances at 250m so one GPS glitch cannot corrupt the whole DP
    d_capped = np.minimum(dist_mat, 250.0)
    emit_cost = 0.5 * (d_capped / sigma_dist_m) ** 2 + (
        heading_weight
        * trip.heading_valid[:, None]
        * (hdiff_mat / 45.0) ** 2
    )

    # Forward monotonic DP
    dp = np.full((n_obs, n_route), np.inf, dtype=np.float64)
    backptr = np.zeros((n_obs, n_route), dtype=np.int32)
    dp[0] = emit_cost[0]

    r_idx = np.arange(n_route, dtype=np.int32)

    for t in range(1, n_obs):
        disp = float(trip.disp_prev[t])
        dt = float(trip.dt_prev[t])
        # Expand allowed route-index jump window when GPS has a large spatial/temporal gap
        if dt > 5.0 or disp > 100.0:
            win = n_route
        else:
            win = min(n_route, max_step_window)

        prev_dp = dp[t - 1]
        prev_pos = route_pos_m[t - 1]
        curr_pos = route_pos_m[t]
        scale = max(disp, 15.0)

        # Vectorized banded DP update across step offsets d in [0, win]
        # Eliminates O(T * R) Python inner loops (~20-30x faster on long trips)
        best_prev_cost = np.full(n_route, np.inf, dtype=np.float64)
        best_prev_idx = np.zeros(n_route, dtype=np.int32)

        for step in range(win + 1):
            if step >= n_route:
                break
            p_pos = prev_pos[: n_route - step]
            c_pos = curr_pos[step:]
            prog = c_pos - p_pos
            neg_pen = np.where(prog < -15.0, (-prog - 15.0) / 10.0, 0.0)
            prog_clamped = np.maximum(prog, 0.0)
            trans_pen = progression_weight * np.abs(prog_clamped - disp) / scale + neg_pen + 0.005 * step
            cand_scores = prev_dp[: n_route - step] + trans_pen

            better = cand_scores < best_prev_cost[step:]
            if np.any(better):
                target_cost = best_prev_cost[step:]
                target_idx = best_prev_idx[step:]
                target_cost[better] = cand_scores[better]
                target_idx[better] = r_idx[: n_route - step][better]

        dp[t] = emit_cost[t] + best_prev_cost
        backptr[t] = best_prev_idx

    # Backtrack optimal monotonic alignment
    aligned_pos = np.zeros(n_obs, dtype=np.int32)
    aligned_pos[-1] = int(np.argmin(dp[-1]))
    for t in range(n_obs - 2, -1, -1):
        aligned_pos[t] = backptr[t + 1, aligned_pos[t + 1]]

    aligned_arcs = r_arcs[aligned_pos]
    aligned_dist = dist_mat[np.arange(n_obs), aligned_pos]
    aligned_hdiff = hdiff_mat[np.arange(n_obs), aligned_pos]

    # Identify low-confidence / ambiguous alignments
    is_low_conf = aligned_dist > low_conf_dist_m
    if n_route > 2:
        # Also mark ambiguous if a distant route index (|r - r*| > 3) has near-identical emission cost
        for t in range(n_obs):
            r_star = int(aligned_pos[t])
            far_mask = np.abs(r_idx - r_star) > 3
            if np.any(far_mask):
                best_far = float(np.min(emit_cost[t, far_mask]))
                if best_far <= emit_cost[t, r_star] + 0.25 and aligned_dist[t] > 20.0:
                    is_low_conf[t] = True

    cand_hit = np.zeros(n_obs, dtype=bool)
    if candidates is not None:
        for t, cset in enumerate(candidates):
            if len(cset.arc_ids) > 0 and aligned_arcs[t] in cset.arc_ids:
                cand_hit[t] = True

    return RouteAlignmentResult(
        trip_id=trip.trip_id,
        aligned_route_pos=aligned_pos,
        aligned_arc_ids=aligned_arcs,
        aligned_dist_m=aligned_dist,
        aligned_heading_diff_deg=aligned_hdiff,
        is_low_confidence=is_low_conf,
        candidate_hit=cand_hit,
    )
