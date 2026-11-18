"""Tune HMM/Viterbi parameters and run mandatory Classifier-Only vs. Classifier+HMM experiments."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.data.loader import list_trip_ids, load_trip_inputs, load_trip_route
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
from src.visualization.plots import save_static_route_comparison_figure


def _prepare_eval_bundle(partition_dir: Path, trip_ids: List[str], partition: str, cand_cfg: CandidateConfig, model):
    """Precompute PreparedTrip, candidates, classifier probs, and reference alignment for fast HMM eval."""
    bundle = {}
    for tid in trip_ids:
        t0 = time.perf_counter()
        raw = load_trip_inputs(partition_dir, tid, partition=partition)
        route = load_trip_route(partition_dir, tid)
        prep = prepare_trip(raw)
        cands = generate_candidates(prep, cand_cfg)
        X, obs_idx, arc_ids = extract_candidate_features(prep, cands)
        t_feat = time.perf_counter() - t0

        t1 = time.perf_counter()
        probs = model.predict_proba(X)[:, 1]
        t_clf = time.perf_counter() - t1

        align = align_gps_to_reference_route(prep, route, cands)
        y_true = (arc_ids == align.aligned_arc_ids[obs_idx]).astype(np.int32)

        bundle[tid] = {
            "prep": prep,
            "route": route,
            "cands": cands,
            "obs_idx": obs_idx,
            "arc_ids": arc_ids,
            "probs": probs,
            "align": align,
            "y_true": y_true,
            "feat_time_s": t_feat,
            "clf_time_s": t_clf,
        }
    return bundle


def _eval_hmm_config_on_bundle(bundle: Dict[str, Any], hmm_cfg: HMMConfig) -> Dict[str, Any]:
    """Run HMM/Viterbi with given config across all trips in bundle and return aggregate score."""
    hmm = PathSnapHMM(hmm_cfg)
    tot_obs = 0
    tot_correct = 0
    len_f1s = []
    reach_conts = []

    for tid, item in bundle.items():
        item["prep"].network.clear_cache()
        dec = hmm.decode_from_flat_probs(item["prep"], item["cands"], item["obs_idx"], item["probs"])
        ref = item["align"].aligned_arc_ids
        tot_obs += len(ref)
        tot_correct += int(np.sum((dec.obs_arc_ids == ref) & (ref >= 0)))
        seq_m = compute_sequence_route_metrics(item["prep"].network, dec.reconstructed_route_arcs, item["route"])
        cont_m = compute_route_continuity(item["prep"].network, dec.obs_arc_ids)
        len_f1s.append(seq_m["seq_length_f1"])
        reach_conts.append(cont_m["network_reachable_continuity"])

    return {
        "top1_accuracy": float(tot_correct / max(tot_obs, 1)),
        "mean_seq_length_f1": float(np.mean(len_f1s)),
        "mean_route_continuity": float(np.mean(reach_conts)),
    }


def _plot_hmm_comparison(df: pd.DataFrame, fig_dir: Path) -> None:
    """Plot per-trip and aggregate Classifier-Only vs HMM/Viterbi comparisons."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    tids = df["trip_id"].tolist()
    x = np.arange(len(tids))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=160)

    # Panel 1: Per-trip Top-1 Accuracy
    axes[0].bar(x - w / 2, df["clf_top1_acc"], width=w, color="#94a3b8", label="Classifier-Only")
    axes[0].bar(x + w / 2, df["hmm_top1_acc"], width=w, color="#2563eb", label="Classifier + HMM/Viterbi")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Observation-Level Top-1 Accuracy")
    axes[0].set_ylim(max(0.4, df[["clf_top1_acc", "hmm_top1_acc"]].min().min() - 0.05), 1.02)
    axes[0].set_title("Per-Trip Top-1 Accuracy", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel 2: Per-trip Sequence Length F1
    axes[1].bar(x - w / 2, df["clf_seq_length_f1"], width=w, color="#fb923c", label="Classifier-Only")
    axes[1].bar(x + w / 2, df["hmm_seq_length_f1"], width=w, color="#16a34a", label="Classifier + HMM/Viterbi")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Length-Weighted Route F1")
    axes[1].set_ylim(max(0.4, df[["clf_seq_length_f1", "hmm_seq_length_f1"]].min().min() - 0.05), 1.02)
    axes[1].set_title("Sequence-Level Route Match F1", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8)

    # Panel 3: Transition Direct Continuity & Reconstructed Route Continuity
    axes[2].bar(x - w / 2, df["clf_trans_direct_cont"], width=w, color="#f87171", label="Clf-Only Direct Trans Cont.")
    axes[2].bar(x + w / 2, df["hmm_recon_direct_cont"], width=w, color="#0d9488", label="HMM Reconstructed Cont.")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Direct Topological Continuity")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Topological Route Continuity", fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=8)

    fig.suptitle("Classifier-Only vs. Classifier + HMM/Viterbi Across All 10 Student-Test Trips", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_dir / "hmm_vs_classifier.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    cfg = get_config()
    feat_cfg = json.loads((cfg.artifacts_dir / "preprocessors" / "feature_config.json").read_text())
    cand_cfg = CandidateConfig(**feat_cfg["candidate_config"])
    model = joblib.load(cfg.artifacts_dir / "models" / "final_model.joblib")

    # Step 1: Tune HMM decoder parameters on a deterministic subset of 8 training trips
    # (never touches evaluator_test; keeps student_test strictly for clean evaluation!)
    train_ids = list_trip_ids(cfg.train_dir)
    rng = np.random.default_rng(cfg.random_seed)
    # Pick 8 diverse medium-sized training trips for fast, unbiased decoder parameter tuning
    inv_df = pd.read_csv(cfg.experiments_dir / "dataset_inventory.csv")
    train_inv = inv_df[(inv_df["partition"] == "train") & (inv_df["n_gps"].between(500, 2500))]
    tune_tids = sorted(train_inv["trip_id"].astype(str).str.zfill(8).tolist())
    if len(tune_tids) > 8:
        chosen = rng.choice(len(tune_tids), size=8, replace=False)
        tune_tids = [tune_tids[i] for i in sorted(chosen)]
    print(f"Tuning HMM decoder on {len(tune_tids)} training trips: {tune_tids}")
    tune_bundle = _prepare_eval_bundle(cfg.train_dir, tune_tids, "train", cand_cfg, model)

    # Grid search over key HMM parameters
    grid = [
        {"emission_weight": ew, "transition_weight": tw, "beta_m": bm, "uturn_penalty": up}
        for ew in [1.0, 1.5, 2.5]
        for tw in [0.5, 1.0, 1.5]
        for bm in [8.0, 15.0]
        for up in [2.5, 4.0]
    ]
    tuning_logs = []
    best_score = -1.0
    best_hmm_cfg = HMMConfig()

    for g_idx, g_params in enumerate(grid):
        h_cfg = HMMConfig(**g_params)
        res = _eval_hmm_config_on_bundle(tune_bundle, h_cfg)
        # Composite objective: balance Top-1 accuracy and Sequence Length F1
        score = 0.5 * res["top1_accuracy"] + 0.5 * res["mean_seq_length_f1"]
        tuning_logs.append({**g_params, **res, "composite_score": score})
        if score > best_score:
            best_score = score
            best_hmm_cfg = h_cfg
        if (g_idx + 1) % 9 == 0 or (g_idx + 1) == len(grid):
            print(
                f"  HMM grid {g_idx + 1}/{len(grid)} | best composite={best_score:.4f} "
                f"(ew={best_hmm_cfg.emission_weight}, tw={best_hmm_cfg.transition_weight}, "
                f"beta={best_hmm_cfg.beta_m}, uturn={best_hmm_cfg.uturn_penalty})"
            )

    pd.DataFrame(tuning_logs).to_csv(cfg.experiments_dir / "hmm_tuning_trials.csv", index=False)

    # Save chosen HMM config artifact
    hmm_cfg_path = cfg.artifacts_dir / "configurations" / "hmm_config.json"
    hmm_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    hmm_cfg_path.write_text(json.dumps(best_hmm_cfg.to_dict(), indent=2))
    print(f"Saved optimal HMM config -> {hmm_cfg_path}")

    # Step 2: Run full head-to-head evaluation on ALL 10 student-testing trips
    stest_ids = list_trip_ids(cfg.student_test_dir)
    print(f"\nEvaluating Classifier-Only vs. Classifier + HMM/Viterbi on {len(stest_ids)} student-test trips...")
    stest_bundle = _prepare_eval_bundle(cfg.student_test_dir, stest_ids, "student_test", cand_cfg, model)
    hmm = PathSnapHMM(best_hmm_cfg)

    rows = []
    all_y_true = []
    all_probs = []
    tot_obs = 0
    tot_hits = 0
    clf_tot_correct = 0
    hmm_tot_correct = 0
    hmm_decoded_store = {}

    for tid in stest_ids:
        item = stest_bundle[tid]
        prep = item["prep"]
        route = item["route"]
        cands = item["cands"]
        obs_idx = item["obs_idx"]
        arc_ids = item["arc_ids"]
        probs = item["probs"]
        align = item["align"]

        all_y_true.append(item["y_true"])
        all_probs.append(probs)

        # A. Classifier-Only
        clf_pred_arcs = select_top1_arcs_per_observation(obs_idx, arc_ids, probs, len(prep.gps_x))
        clf_cand_m = compute_candidate_classification_metrics(item["y_true"], probs)
        clf_top1 = compute_observation_top1_metrics(align.aligned_arc_ids, clf_pred_arcs, align.candidate_hit)
        clf_cont = compute_route_continuity(prep.network, clf_pred_arcs)
        clf_seq = compute_sequence_route_metrics(prep.network, clf_pred_arcs, route)

        # B. Classifier + HMM/Viterbi
        prep.network.clear_cache()
        t_h0 = time.perf_counter()
        dec = hmm.decode_from_flat_probs(prep, cands, obs_idx, probs)
        hmm_time_s = time.perf_counter() - t_h0

        hmm_top1 = compute_observation_top1_metrics(align.aligned_arc_ids, dec.obs_arc_ids, align.candidate_hit)
        hmm_cont = compute_route_continuity(prep.network, dec.obs_arc_ids)
        hmm_seq = compute_sequence_route_metrics(prep.network, dec.reconstructed_route_arcs, route)

        # Observation-level binary match stats for totals
        tot_obs += len(prep.gps_x)
        tot_hits += int(np.sum(align.candidate_hit))
        clf_tot_correct += int(np.sum((clf_pred_arcs == align.aligned_arc_ids) & (align.aligned_arc_ids >= 0)))
        hmm_tot_correct += int(np.sum((dec.obs_arc_ids == align.aligned_arc_ids) & (align.aligned_arc_ids >= 0)))

        hmm_decoded_store[tid] = (prep, route, clf_pred_arcs, dec)

        rows.append(
            {
                "trip_id": tid,
                "n_obs": len(prep.gps_x),
                "n_route_arcs": len(route),
                "candidate_miss_rate": clf_top1["candidate_miss_rate"],
                "cand_macro_f1": clf_cand_m["cand_macro_f1"],
                "cand_log_loss": clf_cand_m["log_loss"],
                # Classifier-only
                "clf_top1_acc": clf_top1["top1_accuracy"],
                "clf_top1_given_hit": clf_top1["top1_accuracy_given_hit"],
                "clf_direct_step_cont": clf_cont["direct_step_continuity"],
                "clf_reach_step_cont": clf_cont["network_reachable_continuity"],
                "clf_trans_direct_cont": clf_cont["transition_direct_continuity"],
                "clf_trans_reach_cont": clf_cont["transition_reachable_continuity"],
                "clf_disconnected_trans": clf_cont["disconnected_transitions"],
                "clf_seq_arc_f1": clf_seq["seq_arc_f1"],
                "clf_seq_length_f1": clf_seq["seq_length_f1"],
                "clf_recon_direct_cont": clf_seq["reconstructed_route_direct_continuity"],
                # HMM/Viterbi
                "hmm_top1_acc": hmm_top1["top1_accuracy"],
                "hmm_top1_given_hit": hmm_top1["top1_accuracy_given_hit"],
                "hmm_direct_step_cont": hmm_cont["direct_step_continuity"],
                "hmm_reach_step_cont": hmm_cont["network_reachable_continuity"],
                "hmm_trans_direct_cont": hmm_cont["transition_direct_continuity"],
                "hmm_trans_reach_cont": hmm_cont["transition_reachable_continuity"],
                "hmm_disconnected_trans": hmm_cont["disconnected_transitions"],
                "hmm_seq_arc_f1": hmm_seq["seq_arc_f1"],
                "hmm_seq_length_f1": hmm_seq["seq_length_f1"],
                "hmm_recon_direct_cont": hmm_seq["reconstructed_route_direct_continuity"],
                "hmm_bridged_arcs": dec.n_bridged_arcs,
                "hmm_segment_breaks": len(dec.segment_breaks),
                "hmm_gap_breaks": len(dec.gap_breaks),
                # Improvements
                "delta_top1_acc": hmm_top1["top1_accuracy"] - clf_top1["top1_accuracy"],
                "delta_seq_length_f1": hmm_seq["seq_length_f1"] - clf_seq["seq_length_f1"],
                # Timing
                "feat_time_s": item["feat_time_s"],
                "clf_time_s": item["clf_time_s"],
                "hmm_time_s": hmm_time_s,
                "total_infer_time_s": item["feat_time_s"] + item["clf_time_s"] + hmm_time_s,
            }
        )
        print(
            f"  Trip {tid} (obs={len(prep.gps_x):4d}): "
            f"Top1 {clf_top1['top1_accuracy']:.4f} -> {hmm_top1['top1_accuracy']:.4f} ({hmm_top1['top1_accuracy'] - clf_top1['top1_accuracy']:+.4f}) | "
            f"SeqLenF1 {clf_seq['seq_length_f1']:.4f} -> {hmm_seq['seq_length_f1']:.4f} | "
            f"ReconCont {clf_seq['reconstructed_route_direct_continuity']:.4f} -> {hmm_seq['reconstructed_route_direct_continuity']:.4f} | "
            f"HMM {hmm_time_s*1000:.0f}ms"
        )

    res_df = pd.DataFrame(rows)
    res_df.to_csv(cfg.experiments_dir / "hmm_vs_classifier.csv", index=False)
    res_df.to_csv(cfg.experiments_dir / "evaluation_results.csv", index=False)

    agg_cand = compute_candidate_classification_metrics(np.concatenate(all_y_true), np.concatenate(all_probs))
    summary = {
        "hmm_config": best_hmm_cfg.to_dict(),
        "tuning_trips": tune_tids,
        "n_student_test_trips": len(stest_ids),
        "total_observations": int(tot_obs),
        "candidate_miss_rate": float(1.0 - tot_hits / max(tot_obs, 1)),
        "classifier_cand_macro_f1": agg_cand["cand_macro_f1"],
        "classifier_log_loss": agg_cand["log_loss"],
        "classifier_only": {
            "weighted_top1_acc": float(clf_tot_correct / max(tot_obs, 1)),
            "mean_trip_top1_acc": float(res_df["clf_top1_acc"].mean()),
            "mean_direct_step_cont": float(res_df["clf_direct_step_cont"].mean()),
            "mean_reach_step_cont": float(res_df["clf_reach_step_cont"].mean()),
            "mean_trans_direct_cont": float(res_df["clf_trans_direct_cont"].mean()),
            "mean_trans_reach_cont": float(res_df["clf_trans_reach_cont"].mean()),
            "total_disconnected_trans": int(res_df["clf_disconnected_trans"].sum()),
            "mean_seq_arc_f1": float(res_df["clf_seq_arc_f1"].mean()),
            "mean_seq_length_f1": float(res_df["clf_seq_length_f1"].mean()),
            "mean_recon_direct_cont": float(res_df["clf_recon_direct_cont"].mean()),
            "mean_clf_infer_ms": float(res_df["clf_time_s"].mean() * 1000.0),
        },
        "classifier_plus_hmm": {
            "weighted_top1_acc": float(hmm_tot_correct / max(tot_obs, 1)),
            "mean_trip_top1_acc": float(res_df["hmm_top1_acc"].mean()),
            "mean_direct_step_cont": float(res_df["hmm_direct_step_cont"].mean()),
            "mean_reach_step_cont": float(res_df["hmm_reach_step_cont"].mean()),
            "mean_trans_direct_cont": float(res_df["hmm_trans_direct_cont"].mean()),
            "mean_trans_reach_cont": float(res_df["hmm_trans_reach_cont"].mean()),
            "total_disconnected_trans": int(res_df["hmm_disconnected_trans"].sum()),
            "mean_seq_arc_f1": float(res_df["hmm_seq_arc_f1"].mean()),
            "mean_seq_length_f1": float(res_df["hmm_seq_length_f1"].mean()),
            "mean_recon_direct_cont": float(res_df["hmm_recon_direct_cont"].mean()),
            "total_bridged_arcs": int(res_df["hmm_bridged_arcs"].sum()),
            "total_segment_breaks": int(res_df["hmm_segment_breaks"].sum()),
            "total_gap_breaks": int(res_df["hmm_gap_breaks"].sum()),
            "mean_hmm_decode_ms": float(res_df["hmm_time_s"].mean() * 1000.0),
            "mean_total_pipeline_ms": float(res_df["total_infer_time_s"].mean() * 1000.0),
        },
    }
    (cfg.experiments_dir / "hmm_summary.json").write_text(json.dumps(summary, indent=2))
    _plot_hmm_comparison(res_df, cfg.reports_dir / "figures")

    # Step 3: Generate representative Folium comparison maps (best lift, median/cleanest, hardest)
    sorted_by_delta = res_df.sort_values("delta_top1_acc", ascending=False)["trip_id"].tolist()
    sorted_by_acc = res_df.sort_values("hmm_top1_acc", ascending=False)["trip_id"].tolist()
    rep_tids = []
    for cand_tid in [sorted_by_delta[0], sorted_by_acc[0], sorted_by_acc[-1], stest_ids[0]]:
        if cand_tid not in rep_tids:
            rep_tids.append(cand_tid)
    rep_tids = rep_tids[:4]

    for tid in rep_tids:
        prep, route, clf_arcs, dec = hmm_decoded_store[tid]
        row = res_df[res_df["trip_id"] == tid].iloc[0]
        fmap = build_trip_folium_map(
            prep,
            predicted_arcs=dec.reconstructed_route_arcs,
            classifier_only_arcs=clf_arcs,
            reference_arcs=route,
            title=(
                f"PathSnap Student-Test Trip {tid} | "
                f"Clf Top-1: {row['clf_top1_acc']*100:.1f}% -> HMM Top-1: {row['hmm_top1_acc']*100:.1f}% | "
                f"Route F1: {row['hmm_seq_length_f1']*100:.1f}%"
            ),
            show_road_network=True,
        )
        m_path = save_trip_folium_map(fmap, cfg.reports_dir / "maps" / f"student_test_{tid}_comparison.html")
        fig_path = save_static_route_comparison_figure(
            prep,
            predicted_arcs=dec.reconstructed_route_arcs,
            classifier_only_arcs=clf_arcs,
            reference_arcs=route,
            title=(
                f"Student-Test Trip {tid} | "
                f"Clf Top-1: {row['clf_top1_acc']*100:.1f}% -> HMM Top-1: {row['hmm_top1_acc']*100:.1f}% | "
                f"Route F1: {row['hmm_seq_length_f1']*100:.1f}%"
            ),
            output_path=cfg.reports_dir / "figures" / f"map_student_test_{tid}.png",
        )
        print(f"Saved Folium map -> {m_path} and static figure -> {fig_path}")

    print("\n=== Aggregate Student-Test Results (10 Trips) ===")
    print(json.dumps(summary["classifier_only"], indent=2))
    print(json.dumps(summary["classifier_plus_hmm"], indent=2))
    print(f"HMM experiments completed in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
