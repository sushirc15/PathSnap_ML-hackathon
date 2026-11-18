"""Route-level continuity and sequence-level reconstruction evaluation metrics."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from src.geo.road_graph import RoadNetwork


def compress_consecutive_arcs(arc_seq: Sequence[int]) -> List[int]:
    """Remove consecutive duplicate arc IDs and invalid (<0) entries to form a route sequence."""
    out: List[int] = []
    prev: Optional[int] = None
    for a in arc_seq:
        ai = int(a)
        if ai < 0:
            continue
        if prev is None or ai != prev:
            out.append(ai)
            prev = ai
    return out


def compute_route_continuity(
    network: RoadNetwork,
    obs_arc_sequence: Sequence[int],
    max_bridge_dist_m: float = 300.0,
) -> Dict[str, float]:
    """Compute topological route continuity metrics for a predicted observation-level arc sequence.

    Measures:
    - `direct_step_continuity`: fraction of consecutive observations (t-1 -> t) that are either
      on the same arc or directly adjacent (`to_node(a_{t-1}) == from_node(a_t)`).
    - `network_reachable_continuity`: fraction of consecutive observations connected either
      directly or via a directed shortest path within `max_bridge_dist_m` meters (accounts for
      vehicles passing through short intermediate arcs between 1 Hz observations).
    - `transition_direct_continuity`: fraction of arc-change transitions (ignoring stays on same arc)
      that are directly adjacent (`to_node == from_node`).
    - `transition_reachable_continuity`: fraction of arc-change transitions reachable within `max_bridge_dist_m`.
    """
    valid = [int(a) for a in obs_arc_sequence if int(a) >= 0]
    if len(valid) <= 1:
        return {
            "direct_step_continuity": 1.0,
            "network_reachable_continuity": 1.0,
            "transition_direct_continuity": 1.0,
            "transition_reachable_continuity": 1.0,
            "disconnected_transitions": 0,
            "total_arc_transitions": 0,
        }

    n_steps = len(valid) - 1
    direct_steps = 0
    reachable_steps = 0
    arc_transitions = 0
    direct_trans = 0
    reachable_trans = 0
    disconnected_trans = 0

    for i in range(n_steps):
        u = valid[i]
        v = valid[i + 1]
        if u == v:
            direct_steps += 1
            reachable_steps += 1
            continue

        arc_transitions += 1
        if network.is_directly_connected(u, v):
            direct_steps += 1
            reachable_steps += 1
            direct_trans += 1
            reachable_trans += 1
        else:
            d = network.shortest_node_distance(
                int(network.to_nodes[u]), int(network.from_nodes[v]), max_bridge_dist_m
            )
            if np.isfinite(d):
                reachable_steps += 1
                reachable_trans += 1
            else:
                disconnected_trans += 1

    return {
        "direct_step_continuity": float(direct_steps / n_steps),
        "network_reachable_continuity": float(reachable_steps / n_steps),
        "transition_direct_continuity": float(direct_trans / arc_transitions) if arc_transitions > 0 else 1.0,
        "transition_reachable_continuity": float(reachable_trans / arc_transitions) if arc_transitions > 0 else 1.0,
        "disconnected_transitions": int(disconnected_trans),
        "total_arc_transitions": int(arc_transitions),
    }


def compute_sequence_route_metrics(
    network: RoadNetwork,
    predicted_route_arcs: Sequence[int],
    reference_route_arcs: Sequence[int],
) -> Dict[str, float]:
    """Compare reconstructed route arc sequence against the `.route` reference sequence.

    Computes both:
    - Arc-set precision, recall, F1 (unweighted)
    - Length-weighted Route Match F1 (standard map-matching RMF / Newson-Krumm style metric:
      length of correctly matched arcs vs predicted and reference route lengths).
    """
    pred_seq = compress_consecutive_arcs(predicted_route_arcs)
    ref_seq = compress_consecutive_arcs(reference_route_arcs)

    pred_set = set(pred_seq)
    ref_set = set(ref_seq)
    inter = pred_set & ref_set

    arc_prec = float(len(inter) / len(pred_set)) if pred_set else 0.0
    arc_rec = float(len(inter) / len(ref_set)) if ref_set else 0.0
    arc_f1 = (2.0 * arc_prec * arc_rec / (arc_prec + arc_rec)) if (arc_prec + arc_rec) > 0 else 0.0

    pred_len = float(sum(network.arc_lengths[a] for a in pred_set))
    ref_len = float(sum(network.arc_lengths[a] for a in ref_set))
    inter_len = float(sum(network.arc_lengths[a] for a in inter))

    len_prec = float(inter_len / pred_len) if pred_len > 0 else 0.0
    len_rec = float(inter_len / ref_len) if ref_len > 0 else 0.0
    len_f1 = (2.0 * len_prec * len_rec / (len_prec + len_rec)) if (len_prec + len_rec) > 0 else 0.0

    # Continuity of the reconstructed compressed route itself
    if len(pred_seq) > 1:
        connected = sum(
            1 for i in range(len(pred_seq) - 1) if network.is_directly_connected(pred_seq[i], pred_seq[i + 1])
        )
        reconstructed_direct_cont = float(connected / (len(pred_seq) - 1))
    else:
        reconstructed_direct_cont = 1.0

    return {
        "seq_arc_precision": arc_prec,
        "seq_arc_recall": arc_rec,
        "seq_arc_f1": arc_f1,
        "seq_length_precision": len_prec,
        "seq_length_recall": len_rec,
        "seq_length_f1": len_f1,
        "reconstructed_route_direct_continuity": reconstructed_direct_cont,
        "n_predicted_route_arcs": len(pred_seq),
        "n_reference_route_arcs": len(ref_seq),
    }
