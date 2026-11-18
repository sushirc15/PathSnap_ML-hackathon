"""Milestone 1 smoke test: end-to-end pipeline verification on representative trips."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import load_trip_inputs, load_trip_route
from src.data.preprocessing import prepare_trip
from src.decoder.hmm import HMMConfig, PathSnapHMM
from src.evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_metrics,
    select_top1_arcs_per_observation,
)
from src.evaluation.route_metrics import compute_route_continuity, compute_sequence_route_metrics
from src.features.candidate_generation import CandidateConfig, generate_candidates
from src.features.feature_extraction import extract_candidate_features
from src.features.label_alignment import align_gps_to_reference_route
from src.visualization.folium_maps import build_trip_folium_map, save_trip_folium_map


def main() -> None:
    cfg = get_config()
    cand_cfg = CandidateConfig(radius_m=100.0, fallback_radius_m=300.0, top_k=10)

    # Test on two training trips (train on 00000001, predict on 00000002)
    t_load = time.perf_counter()
    trip1 = prepare_trip(load_trip_inputs(cfg.train_dir, "00000001", "train"))
    route1 = load_trip_route(cfg.train_dir, "00000001")
    cands1 = generate_candidates(trip1, cand_cfg)
    align1 = align_gps_to_reference_route(trip1, route1, cands1)
    X1, obs1, arcs1 = extract_candidate_features(trip1, cands1)
    y1 = (arcs1 == align1.aligned_arc_ids[obs1]).astype(np.int32)
    print(
        f"Trip 00000001: {len(trip1.gps_x)} obs, {len(X1)} cands, "
        f"miss_rate={align1.candidate_miss_rate:.4f}, "
        f"median_align_dist={np.median(align1.aligned_dist_m):.2f}m, "
        f"low_conf={align1.low_confidence_rate:.4f}, "
        f"time={time.perf_counter() - t_load:.2f}s"
    )

    t2 = time.perf_counter()
    trip2 = prepare_trip(load_trip_inputs(cfg.train_dir, "00000002", "train"))
    route2 = load_trip_route(cfg.train_dir, "00000002")
    cands2 = generate_candidates(trip2, cand_cfg)
    align2 = align_gps_to_reference_route(trip2, route2, cands2)
    X2, obs2, arcs2 = extract_candidate_features(trip2, cands2)
    y2 = (arcs2 == align2.aligned_arc_ids[obs2]).astype(np.int32)
    print(
        f"Trip 00000002: {len(trip2.gps_x)} obs, {len(X2)} cands, "
        f"miss_rate={align2.candidate_miss_rate:.4f}, "
        f"median_align_dist={np.median(align2.aligned_dist_m):.2f}m, "
        f"time={time.perf_counter() - t2:.2f}s"
    )

    # Quick baseline HGB on trip1 -> evaluate on trip2 (Classifier-only vs HMM)
    clf = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    clf.fit(X1, y1)
    probs2 = clf.predict_proba(X2)[:, 1]
    clf_metrics = compute_candidate_classification_metrics(y2, probs2)
    clf_pred_arcs = select_top1_arcs_per_observation(obs2, arcs2, probs2, len(trip2.gps_x))
    clf_top1 = compute_observation_top1_metrics(align2.aligned_arc_ids, clf_pred_arcs, align2.candidate_hit)
    clf_cont = compute_route_continuity(trip2.network, clf_pred_arcs)
    clf_seq = compute_sequence_route_metrics(trip2.network, clf_pred_arcs, route2)

    t_hmm = time.perf_counter()
    hmm = PathSnapHMM(HMMConfig())
    dec = hmm.decode_from_flat_probs(trip2, cands2, obs2, probs2)
    hmm_time = time.perf_counter() - t_hmm
    hmm_top1 = compute_observation_top1_metrics(align2.aligned_arc_ids, dec.obs_arc_ids, align2.candidate_hit)
    hmm_cont = compute_route_continuity(trip2.network, dec.obs_arc_ids)
    hmm_seq = compute_sequence_route_metrics(trip2.network, dec.reconstructed_route_arcs, route2)

    print("\n--- Milestone 1 End-to-End Smoke Test (Train: 00000001 -> Eval: 00000002) ---")
    print(f"Candidate Macro F1: {clf_metrics['cand_macro_f1']:.4f} | LogLoss: {clf_metrics['log_loss']:.4f}")
    print(
        f"Classifier-Only -> Top1: {clf_top1['top1_accuracy']:.4f} | "
        f"TransReachCont: {clf_cont['transition_reachable_continuity']:.4f} | "
        f"SeqLenF1: {clf_seq['seq_length_f1']:.4f}"
    )
    print(
        f"HMM/Viterbi     -> Top1: {hmm_top1['top1_accuracy']:.4f} | "
        f"TransReachCont: {hmm_cont['transition_reachable_continuity']:.4f} | "
        f"SeqLenF1: {hmm_seq['seq_length_f1']:.4f} | "
        f"ReconDirectCont: {hmm_seq['reconstructed_route_direct_continuity']:.4f} | "
        f"Bridged: {dec.n_bridged_arcs} | Breaks: {len(dec.segment_breaks)} | Time: {hmm_time:.2f}s"
    )

    fmap = build_trip_folium_map(
        trip2,
        predicted_arcs=dec.reconstructed_route_arcs,
        classifier_only_arcs=clf_pred_arcs,
        reference_arcs=route2,
        title="Milestone 1 Smoke Test — Trip 00000002 (Ref vs Clf-Only vs HMM/Viterbi)",
    )
    save_trip_folium_map(fmap, cfg.reports_dir / "maps" / "milestone1_smoke_00000002.html")
    print("Saved Milestone 1 Folium map successfully!")


if __name__ == "__main__":
    main()
