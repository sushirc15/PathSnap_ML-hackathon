"""Reusable model evaluation routines for CV folds and student-testing trips."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.data.loader import load_trip_inputs
from src.data.preprocessing import PreparedTrip, prepare_trip
from src.evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_metrics,
    select_top1_arcs_per_observation,
)
from src.evaluation.route_metrics import compute_route_continuity
from src.models.dataset_cache import PartitionFeatureDataset


def load_prepared_trips(partition_dir: Path, trip_ids: Sequence[str], partition: str) -> Dict[str, PreparedTrip]:
    """Load and prepare a dictionary of PreparedTrips (for route continuity / HMM evaluation)."""
    out: Dict[str, PreparedTrip] = {}
    for tid in trip_ids:
        raw = load_trip_inputs(partition_dir, tid, partition=partition)
        out[tid] = prepare_trip(raw)
    return out


def evaluate_predictions_on_partition(
    ds: PartitionFeatureDataset,
    probs: np.ndarray,
    trip_ids_subset: Optional[Sequence[str]] = None,
    prepared_trips: Optional[Dict[str, PreparedTrip]] = None,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute candidate-level, observation-level Top-1, and optional route-continuity metrics."""
    selected_tids = list(trip_ids_subset) if trip_ids_subset is not None else list(ds.trip_ids)

    row_masks = []
    total_obs = 0
    total_correct_top1 = 0
    total_hits = 0
    total_correct_given_hit = 0

    direct_conts = []
    reach_conts = []
    trans_reach_conts = []
    per_trip_results: Dict[str, Dict[str, float]] = {}

    for tid in selected_tids:
        meta = ds.trip_meta[tid]
        r0, r1 = meta["row_start"], meta["row_end"]
        n_obs = meta["n_obs"]
        row_masks.append(np.arange(r0, r1, dtype=np.int64))

        obs_t = ds.obs_idx[r0:r1]
        arcs_t = ds.arc_ids[r0:r1]
        probs_t = probs[r0:r1]

        pred_arcs = select_top1_arcs_per_observation(obs_t, arcs_t, probs_t, n_obs)
        ref_arcs = meta["aligned_arc_ids"]
        cand_hit = meta["candidate_hit"]

        top1_m = compute_observation_top1_metrics(ref_arcs, pred_arcs, cand_hit)
        correct = (pred_arcs == ref_arcs) & (ref_arcs >= 0)
        total_obs += n_obs
        total_correct_top1 += int(np.sum(correct))
        total_hits += int(np.sum(cand_hit))
        total_correct_given_hit += int(np.sum(correct[cand_hit]))

        trip_res = {
            "n_obs": n_obs,
            "top1_accuracy": top1_m["top1_accuracy"],
            "top1_accuracy_given_hit": top1_m["top1_accuracy_given_hit"],
            "candidate_miss_rate": top1_m["candidate_miss_rate"],
        }

        if prepared_trips is not None and tid in prepared_trips:
            cont = compute_route_continuity(prepared_trips[tid].network, pred_arcs)
            direct_conts.append(cont["direct_step_continuity"])
            reach_conts.append(cont["network_reachable_continuity"])
            trans_reach_conts.append(cont["transition_reachable_continuity"])
            trip_res["direct_step_continuity"] = cont["direct_step_continuity"]
            trip_res["network_reachable_continuity"] = cont["network_reachable_continuity"]
            trip_res["transition_reachable_continuity"] = cont["transition_reachable_continuity"]

        per_trip_results[tid] = trip_res

    all_idx = np.concatenate(row_masks) if row_masks else np.zeros(0, dtype=np.int64)
    cand_m = compute_candidate_classification_metrics(ds.y[all_idx], probs[all_idx], threshold=threshold)

    summary: Dict[str, Any] = {
        **cand_m,
        "top1_accuracy": float(total_correct_top1 / max(total_obs, 1)),
        "top1_accuracy_given_hit": float(total_correct_given_hit / max(total_hits, 1)),
        "candidate_miss_rate": float(1.0 - total_hits / max(total_obs, 1)),
        "n_obs": int(total_obs),
        "n_cands": int(len(all_idx)),
        "per_trip": per_trip_results,
    }

    if direct_conts:
        summary["direct_step_continuity"] = float(np.mean(direct_conts))
        summary["network_reachable_continuity"] = float(np.mean(reach_conts))
        summary["transition_reachable_continuity"] = float(np.mean(trans_reach_conts))
    else:
        summary["direct_step_continuity"] = float("nan")
        summary["network_reachable_continuity"] = float("nan")
        summary["transition_reachable_continuity"] = float("nan")

    return summary
