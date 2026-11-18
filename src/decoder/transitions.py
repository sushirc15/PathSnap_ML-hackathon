"""Road-network-aware transition scoring for the PathSnap HMM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geo.geometry import angular_diff_deg
from src.geo.road_graph import RoadNetwork
from src.geo.spatial_index import CandidateSet


@dataclass(frozen=True)
class TransitionConfig:
    """Tunable parameters for directed road-network HMM transitions."""

    beta_m: float = 12.0  # Exponential decay scale for |route_dist - gps_dist| (Newson & Krumm)
    max_speed_mps: float = 45.0  # Max plausible vehicle speed (~162 km/h)
    base_search_slack_m: float = 120.0  # Slack added to GPS displacement for Dijkstra cutoff
    max_search_dist_m: float = 1500.0  # Hard cap on per-step shortest-path search
    uturn_penalty: float = 3.5  # Log-penalty for reversing onto the exact reverse arc
    same_arc_bonus: float = 0.25  # Mild continuity preference for staying on current arc
    gap_dt_threshold_s: float = 15.0  # Above this dt, treat step as a GPS dropout gap
    gap_dist_threshold_m: float = 350.0  # Above this GPS jump, relax routing or segment trajectory
    unreachable_penalty: float = -25.0  # Log-penalty when no bounded directed path exists


def compute_transition_log_matrix(
    network: RoadNetwork,
    prev_cands: CandidateSet,
    curr_cands: CandidateSet,
    gps_disp_m: float,
    dt_s: float,
    cfg: TransitionConfig,
) -> tuple[np.ndarray, bool]:
    """Compute (K_prev, K_curr) log-transition score matrix between consecutive candidate sets.

    Returns:
        log_trans: float64 matrix of shape (K_prev, K_curr)
        is_gap_break: True if dt_s or gps_disp_m indicates a major GPS dropout / segment break
                      where hard shortest-path routing is unreliable.
    """
    k_prev = len(prev_cands.arc_ids)
    k_curr = len(curr_cands.arc_ids)
    log_trans = np.full((k_prev, k_curr), cfg.unreachable_penalty, dtype=np.float64)

    if k_prev == 0 or k_curr == 0:
        return log_trans, True

    # Handle major GPS dropout gaps gracefully: allow smooth re-entry via spatial proximity
    # rather than forcing impossible/expensive long-range Dijkstra searches.
    if dt_s > cfg.gap_dt_threshold_s or gps_disp_m > cfg.gap_dist_threshold_m:
        # Mild Euclidean consistency prior across the gap (prevents random jumps)
        dx = prev_cands.proj_x[:, None] - curr_cands.proj_x[None, :]
        dy = prev_cands.proj_y[:, None] - curr_cands.proj_y[None, :]
        proj_d = np.hypot(dx, dy)
        log_trans = -np.abs(proj_d - gps_disp_m) / max(cfg.beta_m * 4.0, 40.0)
        return log_trans, True

    # Adaptive Dijkstra cutoff based on observed GPS step and plausible speed
    max_plausible_dist = min(
        cfg.max_search_dist_m,
        max(gps_disp_m * 2.5 + cfg.base_search_slack_m, dt_s * cfg.max_speed_mps + 80.0),
    )

    # Pre-run bounded Dijkstra once per unique to_node in prev_cands
    prev_arcs = prev_cands.arc_ids
    curr_arcs = curr_cands.arc_ids

    for i in range(k_prev):
        au = int(prev_arcs[i])
        fu = float(prev_cands.frac[i])
        to_u = int(network.to_nodes[au])
        from_u = int(network.from_nodes[au])
        len_u = float(network.arc_lengths[au])
        rem_u = (1.0 - fu) * len_u

        # Only call Dijkstra if there is at least one curr candidate != au and not directly adjacent
        need_dijkstra = False
        for j in range(k_curr):
            av = int(curr_arcs[j])
            if av != au and int(network.from_nodes[av]) != to_u:
                need_dijkstra = True
                break

        dist_map = (
            network.bounded_dijkstra_from_node(to_u, max_plausible_dist)
            if need_dijkstra
            else {}
        )

        for j in range(k_curr):
            av = int(curr_arcs[j])
            fv = float(curr_cands.frac[j])
            len_v = float(network.arc_lengths[av])

            if au == av:
                delta = (fv - fu) * len_u
                if delta >= -15.0:
                    route_d = max(0.0, delta)
                    diff = abs(route_d - gps_disp_m)
                    log_trans[i, j] = -diff / cfg.beta_m + cfg.same_arc_bonus
                    continue
                else:
                    # Significant backwards jump along same arc
                    node_d = dist_map.get(int(network.from_nodes[av]), float("inf"))
                    route_d = rem_u + node_d + fv * len_v
            else:
                from_v = int(network.from_nodes[av])
                pref_v = fv * len_v
                if to_u == from_v:
                    route_d = rem_u + pref_v
                else:
                    node_d = dist_map.get(from_v, float("inf"))
                    if np.isinf(node_d):
                        continue
                    route_d = rem_u + node_d + pref_v

            if not np.isfinite(route_d) or route_d > max_plausible_dist + 100.0:
                continue

            diff = abs(route_d - gps_disp_m)
            score = -diff / cfg.beta_m

            # Penalize immediate U-turn onto exact reverse arc (u -> v followed by v -> u)
            if int(network.from_nodes[av]) == to_u and int(network.to_nodes[av]) == from_u:
                score -= cfg.uturn_penalty
            elif to_u == int(network.from_nodes[av]):
                # Penalize sharp >150 deg hairpin turns at intersections unless GPS actually turned
                turn_ang = float(
                    angular_diff_deg(
                        float(network.arc_bearings[au]), float(network.arc_bearings[av])
                    )
                )
                if turn_ang > 155.0:
                    score -= 0.5 * cfg.uturn_penalty

            # Speed plausibility check
            if dt_s > 0.0 and route_d > dt_s * cfg.max_speed_mps + 60.0:
                excess = route_d - (dt_s * cfg.max_speed_mps + 60.0)
                score -= excess / 20.0

            log_trans[i, j] = score

    return log_trans, False
