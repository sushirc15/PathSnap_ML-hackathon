"""Reusable prediction-time candidate feature extraction pipeline.

STRICT LEAKAGE GUARANTEE:
- Uses ONLY `PreparedTrip` (derived from `.track`, `.nodes`, `.arcs`) and `CandidateSet`s.
- Never accepts `.route`, arc IDs, or trip IDs as model features.
- Identical function runs for training, cross-validation, student testing, and live API inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.preprocessing import PreparedTrip
from src.geo.geometry import angular_diff_deg, point_to_segments_projection
from src.geo.spatial_index import CandidateSet

FEATURE_COLUMNS: List[str] = [
    # Spatial emission geometry
    "dist_m",
    "log_dist_m",
    "frac",
    "off_segment_m",
    "arc_len_m",
    "log_arc_len_m",
    # Relative candidate ranking within the current GPS observation
    "dist_rank",
    "dist_diff_min",
    "dist_ratio_min",
    "dist_zscore",
    "n_candidates",
    # Heading & vehicle kinematics
    "heading_diff_deg",
    "cos_heading_diff",
    "heading_diff_weighted",
    "heading_diff_rank",
    "heading_diff_from_min",
    "heading_valid",
    "speed_smooth_mps",
    "disp_prev_m",
    "disp_next_m",
    "dt_prev_s",
    "dt_next_s",
    "turn_rate_deg",
    # Local road-network topology (ID-free)
    "has_reverse_arc",
    "from_in_degree",
    "from_out_degree",
    "to_in_degree",
    "to_out_degree",
    # Temporal & local connectivity context across consecutive observations
    "prev_same_arc",
    "next_same_arc",
    "prev_direct_conn",
    "next_direct_conn",
    "dist_prev_obs_to_arc",
    "dist_next_obs_to_arc",
    "dist_temporal_avg",
    "progress_along_arc_m",
]


def extract_candidate_features(
    trip: PreparedTrip,
    candidates: List[CandidateSet],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (N_cand, F) feature matrix and corresponding (obs_idx, arc_id) index arrays.

    Returns:
        X: float32 array of shape (total_candidates, len(FEATURE_COLUMNS))
        obs_indices: int32 array of shape (total_candidates,)
        arc_ids: int32 array of shape (total_candidates,)
    """
    n_obs = len(candidates)
    total_cands = sum(len(c.arc_ids) for c in candidates)
    n_feats = len(FEATURE_COLUMNS)

    if total_cands == 0:
        return (
            np.zeros((0, n_feats), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )

    X = np.empty((total_cands, n_feats), dtype=np.float32)
    obs_indices = np.empty(total_cands, dtype=np.int32)
    arc_ids_out = np.empty(total_cands, dtype=np.int32)

    net = trip.network

    # Precompute candidate arc sets and endpoint node sets per observation for O(1) temporal context
    cand_arc_sets = [set(int(a) for a in c.arc_ids) for c in candidates]
    cand_to_nodes = [
        set(int(net.to_nodes[a]) for a in c.arc_ids) for c in candidates
    ]
    cand_from_nodes = [
        set(int(net.from_nodes[a]) for a in c.arc_ids) for c in candidates
    ]

    row_cursor = 0
    for t, cset in enumerate(candidates):
        k = len(cset.arc_ids)
        if k == 0:
            continue

        aids = cset.arc_ids
        dist_m = cset.dist_m
        frac = cset.frac
        raw_frac = cset.raw_frac
        arc_len = net.arc_lengths[aids]
        off_seg_m = np.abs(raw_frac - frac) * arc_len

        # Relative distance competition within obs t
        min_d = float(np.min(dist_m))
        std_d = float(np.std(dist_m))
        dist_diff_min = dist_m - min_d
        dist_ratio_min = dist_m / (min_d + 1.0)
        dist_zscore = (dist_m - float(np.mean(dist_m))) / (std_d + 1.0)
        # candidates are already sorted by dist_m ascending
        dist_rank = np.arange(k, dtype=np.float64)

        # Heading features
        arc_bearings = net.arc_bearings[aids]
        h_valid = float(trip.heading_valid[t])
        h_diff = angular_diff_deg(float(trip.heading_deg[t]), arc_bearings)
        cos_h = np.cos(np.radians(h_diff))
        h_diff_weighted = h_diff * h_valid
        h_min = float(np.min(h_diff))
        h_diff_from_min = h_diff - h_min
        # Rank by heading diff (0 = best aligned)
        h_order = np.argsort(h_diff, kind="mergesort")
        h_rank = np.empty(k, dtype=np.float64)
        h_rank[h_order] = np.arange(k, dtype=np.float64)

        # Road topology features
        from_n = net.from_nodes[aids]
        to_n = net.to_nodes[aids]
        has_rev = net.has_reverse_arc[aids].astype(np.float64)
        f_in = net.in_degree[from_n].astype(np.float64)
        f_out = net.out_degree[from_n].astype(np.float64)
        t_in = net.in_degree[to_n].astype(np.float64)
        t_out = net.out_degree[to_n].astype(np.float64)

        # Temporal & topological neighbor context
        prev_arcs = cand_arc_sets[t - 1] if t > 0 else set()
        next_arcs = cand_arc_sets[t + 1] if t < n_obs - 1 else set()
        prev_to = cand_to_nodes[t - 1] if t > 0 else set()
        next_from = cand_from_nodes[t + 1] if t < n_obs - 1 else set()

        prev_same = np.array([1.0 if int(a) in prev_arcs else 0.0 for a in aids], dtype=np.float64)
        next_same = np.array([1.0 if int(a) in next_arcs else 0.0 for a in aids], dtype=np.float64)
        prev_conn = np.array(
            [1.0 if (int(a) in prev_arcs or int(fn) in prev_to) else 0.0 for a, fn in zip(aids, from_n)],
            dtype=np.float64,
        )
        next_conn = np.array(
            [1.0 if (int(a) in next_arcs or int(tn) in next_from) else 0.0 for a, tn in zip(aids, to_n)],
            dtype=np.float64,
        )

        # Temporal corridor distances & signed directional progress along each candidate arc
        t_prev = max(0, t - 1)
        t_next = min(n_obs - 1, t + 1)
        ax1, ay1, ax2, ay2 = net.x1[aids], net.y1[aids], net.x2[aids], net.y2[aids]
        d_prev_obs, _, raw_frac_prev, _, _ = point_to_segments_projection(
            float(trip.gps_x[t_prev]), float(trip.gps_y[t_prev]), ax1, ay1, ax2, ay2
        )
        d_next_obs, _, raw_frac_next, _, _ = point_to_segments_projection(
            float(trip.gps_x[t_next]), float(trip.gps_y[t_next]), ax1, ay1, ax2, ay2
        )
        dist_temp_avg = (dist_m + d_prev_obs + d_next_obs) / 3.0
        # Signed distance moved in the candidate arc's forward direction between t-1 and t+1
        progress_m = (raw_frac_next - raw_frac_prev) * arc_len
        # Clip extreme GPS glitch values
        progress_m = np.clip(progress_m, -150.0, 150.0)

        sl = slice(row_cursor, row_cursor + k)
        obs_indices[sl] = t
        arc_ids_out[sl] = aids

        X[sl, 0] = dist_m
        X[sl, 1] = np.log1p(dist_m)
        X[sl, 2] = frac
        X[sl, 3] = off_seg_m
        X[sl, 4] = arc_len
        X[sl, 5] = np.log1p(arc_len)
        X[sl, 6] = dist_rank
        X[sl, 7] = dist_diff_min
        X[sl, 8] = dist_ratio_min
        X[sl, 9] = dist_zscore
        X[sl, 10] = float(k)
        X[sl, 11] = h_diff
        X[sl, 12] = cos_h
        X[sl, 13] = h_diff_weighted
        X[sl, 14] = h_rank
        X[sl, 15] = h_diff_from_min
        X[sl, 16] = h_valid
        X[sl, 17] = float(trip.speed_smooth[t])
        X[sl, 18] = float(trip.disp_prev[t])
        X[sl, 19] = float(trip.disp_next[t])
        X[sl, 20] = min(float(trip.dt_prev[t]), 60.0)
        X[sl, 21] = min(float(trip.dt_next[t]), 60.0)
        X[sl, 22] = float(trip.turn_rate_deg[t])
        X[sl, 23] = has_rev
        X[sl, 24] = f_in
        X[sl, 25] = f_out
        X[sl, 26] = t_in
        X[sl, 27] = t_out
        X[sl, 28] = prev_same
        X[sl, 29] = next_same
        X[sl, 30] = prev_conn
        X[sl, 31] = next_conn
        X[sl, 32] = d_prev_obs
        X[sl, 33] = d_next_obs
        X[sl, 34] = dist_temp_avg
        X[sl, 35] = progress_m

        row_cursor += k

    # Guard against any rare NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)
    return X, obs_indices, arc_ids_out


def features_to_dataframe(
    X: np.ndarray,
    obs_indices: np.ndarray,
    arc_ids: np.ndarray,
    trip_id: str,
) -> pd.DataFrame:
    """Convert extracted feature array to a labeled DataFrame (for inspection/caching)."""
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    df.insert(0, "arc_id", arc_ids)
    df.insert(0, "obs_idx", obs_indices)
    df.insert(0, "trip_id", trip_id)
    return df
