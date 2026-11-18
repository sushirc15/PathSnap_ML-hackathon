"""Numerically stable log-space Viterbi decoder for PathSnap."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from src.data.preprocessing import PreparedTrip
from src.decoder.transitions import TransitionConfig, compute_transition_log_matrix
from src.geo.spatial_index import CandidateSet


@dataclass
class ViterbiDecodeResult:
    """Decoded observation-level arcs, reconstructed continuous route, and diagnostics."""

    obs_arc_ids: np.ndarray  # (T,) predicted arc ID per GPS observation (-1 if empty candidates)
    reconstructed_route_arcs: List[int]  # Continuous road-network route with bridged intermediate arcs
    segment_breaks: List[int] = field(default_factory=list)  # Observation indices where HMM segmented/recovered
    gap_breaks: List[int] = field(default_factory=list)  # Observation indices with GPS dropout gaps
    n_bridged_arcs: int = 0
    total_log_score: float = 0.0


def viterbi_decode(
    trip: PreparedTrip,
    candidates: List[CandidateSet],
    obs_emission_probs: List[np.ndarray],
    emission_weight: float = 1.0,
    transition_weight: float = 1.0,
    prob_floor: float = 1e-6,
    transition_cfg: TransitionConfig | None = None,
) -> ViterbiDecodeResult:
    """Run numerically stable log-space Viterbi decoding with segmentation recovery.

    Handles:
    - Empty candidate sets (carries forward state or triggers clean segment reset).
    - Disconnected transitions (when every transition hits `unreachable_penalty`, segments
      the trajectory at obs t and restarts Viterbi cleanly rather than propagating garbage).
    - Major GPS dropout gaps (detected via `dt_s` or `gps_disp_m` thresholds).
    - Full intermediate road-network path reconstruction between consecutive matched arcs.
    """
    t_cfg = transition_cfg or TransitionConfig()
    n_obs = len(candidates)

    if n_obs == 0:
        return ViterbiDecodeResult(
            obs_arc_ids=np.zeros(0, dtype=np.int32),
            reconstructed_route_arcs=[],
        )

    # Convert candidate classifier probabilities to clipped log-emission scores
    log_emissions: List[np.ndarray] = []
    for probs in obs_emission_probs:
        if len(probs) == 0:
            log_emissions.append(np.zeros(0, dtype=np.float64))
        else:
            p = np.clip(np.asarray(probs, dtype=np.float64), prob_floor, 1.0 - prob_floor)
            # Log-odds / log-prob emission: log(p) normalized per observation keeps scale crisp
            log_emissions.append(emission_weight * np.log(p))

    dp: List[np.ndarray] = []
    backptrs: List[np.ndarray] = []
    segment_breaks: List[int] = []
    gap_breaks: List[int] = []

    # Track indices of segments [seg_start, seg_end) for clean independent backtracking on hard breaks
    seg_starts: List[int] = [0]
    last_valid_t = -1

    for t in range(n_obs):
        k_curr = len(candidates[t].arc_ids)
        if k_curr == 0:
            dp.append(np.zeros(0, dtype=np.float64))
            backptrs.append(np.zeros(0, dtype=np.int32))
            segment_breaks.append(t)
            continue

        emit_t = log_emissions[t]

        if last_valid_t < 0:
            # First valid observation in trajectory
            dp.append(emit_t.copy())
            backptrs.append(np.full(k_curr, -1, dtype=np.int32))
            last_valid_t = t
            continue

        # Compute road-network transition matrix from last_valid_t -> t
        dt_s = float(trip.timestamp[t] - trip.timestamp[last_valid_t])
        gps_disp_m = float(
            np.hypot(
                trip.gps_x[t] - trip.gps_x[last_valid_t],
                trip.gps_y[t] - trip.gps_y[last_valid_t],
            )
        )

        log_trans, is_gap = compute_transition_log_matrix(
            network=trip.network,
            prev_cands=candidates[last_valid_t],
            curr_cands=candidates[t],
            gps_disp_m=gps_disp_m,
            dt_s=dt_s,
            cfg=t_cfg,
        )
        if is_gap:
            gap_breaks.append(t)

        # Check if ALL transitions are unreachable (disconnected graph / extreme GPS outlier)
        all_unreachable = bool(np.all(log_trans <= t_cfg.unreachable_penalty + 1e-6))

        if all_unreachable:
            # Documented trajectory-segmentation recovery: start a fresh Viterbi segment at t
            # instead of silently forcing an impossible road jump.
            segment_breaks.append(t)
            seg_starts.append(t)
            dp.append(emit_t.copy())
            backptrs.append(np.full(k_curr, -1, dtype=np.int32))
            last_valid_t = t
            continue

        # Viterbi recurrence in log space:
        # dp[t, j] = emit[t, j] + max_i (dp[last_valid_t, i] + w_trans * log_trans[i, j])
        scores = dp[last_valid_t][:, None] + transition_weight * log_trans
        best_prev = np.argmax(scores, axis=0).astype(np.int32)
        best_vals = scores[best_prev, np.arange(k_curr)]

        # Numerical stability: subtract max(best_vals) at every step? No, keep raw or normalize by max
        # Subtracting max avoids float underflow on very long (20,000-point) trips while preserving argmax!
        curr_dp = emit_t + best_vals
        shift = float(np.max(curr_dp))
        if np.isfinite(shift):
            curr_dp = curr_dp - shift

        dp.append(curr_dp)
        backptrs.append(best_prev)
        last_valid_t = t

    # Backtrack independently within each valid segment
    obs_arc_ids = np.full(n_obs, -1, dtype=np.int32)
    seg_boundaries = seg_starts + [n_obs]
    total_score = 0.0

    for s_idx in range(len(seg_starts)):
        s_start = seg_boundaries[s_idx]
        s_end = seg_boundaries[s_idx + 1]
        valid_ts = [t for t in range(s_start, s_end) if len(dp[t]) > 0]
        if not valid_ts:
            continue

        last_t = valid_ts[-1]
        state = int(np.argmax(dp[last_t]))
        total_score += float(dp[last_t][state])
        obs_arc_ids[last_t] = int(candidates[last_t].arc_ids[state])

        for pos in range(len(valid_ts) - 1, 0, -1):
            t_curr = valid_ts[pos]
            t_prev = valid_ts[pos - 1]
            prev_state = int(backptrs[t_curr][state])
            if prev_state < 0 or prev_state >= len(candidates[t_prev].arc_ids):
                prev_state = int(np.argmax(dp[t_prev]))
            state = prev_state
            obs_arc_ids[t_prev] = int(candidates[t_prev].arc_ids[state])

    # Reconstruct complete continuous route sequence by bridging intermediate directed arcs
    reconstructed: List[int] = []
    n_bridged = 0
    for a in obs_arc_ids:
        aid = int(a)
        if aid < 0:
            continue
        if not reconstructed:
            reconstructed.append(aid)
            continue
        prev_aid = reconstructed[-1]
        if aid == prev_aid:
            continue
        if trip.network.is_directly_connected(prev_aid, aid):
            reconstructed.append(aid)
        else:
            bridge = trip.network.shortest_arc_path(prev_aid, aid, max_dist_m=600.0)
            if bridge is not None and len(bridge) > 0:
                reconstructed.extend(bridge)
                n_bridged += len(bridge)
            reconstructed.append(aid)

    return ViterbiDecodeResult(
        obs_arc_ids=obs_arc_ids,
        reconstructed_route_arcs=reconstructed,
        segment_breaks=segment_breaks,
        gap_breaks=gap_breaks,
        n_bridged_arcs=n_bridged,
        total_log_score=total_score,
    )
