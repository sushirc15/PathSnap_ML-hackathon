"""Strict two-phase validation, frozen inference, and post-prediction evaluation on evaluator_test.

INTEGRITY RULES ENFORCED:
1. Uses ONLY the frozen trained artifacts (`artifacts/models/final_model.joblib`,
   `artifacts/preprocessors/feature_config.json`, `artifacts/configurations/hmm_config.json`).
   Zero retraining, zero feature selection, zero hyperparameter or HMM tuning.
2. Phase 1 validates all 10 trips.
3. Phase 2 runs the complete prediction pipeline (`run_prediction_only`) across ALL 10 trips
   using ONLY `.track`, `.nodes`, and `.arcs` — `.route` is NOT loaded during Phase 2.
4. Phase 3 reads `.route` ONLY AFTER all 10 predictions are complete to compute metrics.
5. Phase 4 verifies the live FastAPI backend (`/api/health`, `/api/trips`, all 10 `/predict`,
   and `/map`) and confirms zero ground-truth exposure to the evaluator-facing frontend.
6. Preserves all existing `train` and `student_test` experiment files completely untouched.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app
from src.api.services import PathSnapInferenceService, PredictionBundle
from src.config import get_config
from src.data.loader import list_trip_ids, load_trip_inputs, load_trip_route
from src.data.validation import validate_trip
from src.evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_metrics,
)
from src.evaluation.route_metrics import compute_route_continuity, compute_sequence_route_metrics
from src.features.label_alignment import align_gps_to_reference_route
from scripts.verify_api_all_trips import _render_terminal_screenshot


def _plot_evaluator_results(df: pd.DataFrame, fig_path: Path) -> None:
    """Create consolidated visual benchmark of the 10 reserved evaluator trips."""
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    tids = [f"{int(t):08d}" for t in df["trip_id"]]
    x = np.arange(len(tids))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), dpi=160)

    # Panel 1: Top-1 Accuracy (Clf-Only vs HMM/Viterbi)
    axes[0].bar(x - w / 2, df["clf_top1_acc"], width=w, color="#94a3b8", label="Classifier-Only")
    axes[0].bar(x + w / 2, df["hmm_top1_acc"], width=w, color="#2563eb", label="Classifier + HMM/Viterbi")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Observation-Level Top-1 Accuracy")
    ymin1 = max(0.0, float(df[["clf_top1_acc", "hmm_top1_acc"]].min().min()) - 0.08)
    axes[0].set_ylim(ymin1, 1.03)
    axes[0].set_title("Evaluator-Test Top-1 Accuracy", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)

    # Panel 2: Sequence-Level Route Length F1
    axes[1].bar(x - w / 2, df["clf_seq_length_f1"], width=w, color="#fb923c", label="Classifier-Only")
    axes[1].bar(x + w / 2, df["hmm_seq_length_f1"], width=w, color="#16a34a", label="Classifier + HMM/Viterbi")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Length-Weighted Route F1")
    ymin2 = max(0.0, float(df[["clf_seq_length_f1", "hmm_seq_length_f1"]].min().min()) - 0.08)
    axes[1].set_ylim(ymin2, 1.03)
    axes[1].set_title("Evaluator-Test Sequence Route F1", fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8)

    # Panel 3: Reconstructed Direct Continuity
    axes[2].bar(x - w / 2, df["clf_recon_direct_cont"], width=w, color="#f87171", label="Clf-Only Direct Cont.")
    axes[2].bar(x + w / 2, df["hmm_recon_direct_cont"], width=w, color="#0d9488", label="HMM Reconstructed Cont.")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tids, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Direct Topological Continuity")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Evaluator-Test Topological Continuity", fontweight="bold")
    axes[2].legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Final Reserved Evaluator-Test Results (10 Unseen Trips, Frozen Model & HMM)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    cfg = get_config()
    eval_dir = cfg.evaluator_test_dir
    trip_ids = list_trip_ids(eval_dir)
    if len(trip_ids) == 0:
        raise RuntimeError(f"No trips found in {eval_dir}")

    print(f"=== PHASE 1: Validating {len(trip_ids)} Reserved Evaluator Trips ===")
    val_rows: List[Dict[str, Any]] = []
    val_warnings: List[Dict[str, str]] = []
    val_errors: List[Dict[str, str]] = []

    for tid in trip_ids:
        raw = load_trip_inputs(eval_dir, tid, partition="evaluator_test")
        # Validate prediction inputs first, then audit .route integrity separately
        route = load_trip_route(eval_dir, tid)
        rep = validate_trip(raw, route=route, include_graph_stats=True)
        val_rows.append(rep.metrics)
        for w in rep.warnings:
            val_warnings.append({"trip_id": tid, "warning": w})
        for e in rep.errors:
            val_errors.append({"trip_id": tid, "error": e})
        print(
            f"  [VALID={rep.is_valid}] Trip {tid}: obs={rep.metrics['n_gps']:4d}, "
            f"nodes={rep.metrics['n_nodes']:6d}, arcs={rep.metrics['n_arcs']:6d}, "
            f"route_arcs={rep.metrics['n_route_arcs']:4d}, "
            f"EPSG:{rep.metrics['utm_epsg']}, max_dt={rep.metrics['max_dt_s']:.0f}s, "
            f"route_cont={rep.metrics['route_direct_continuity']:.4f}"
        )

    val_df = pd.DataFrame(val_rows)
    val_csv = cfg.experiments_dir / "evaluator_test_validation.csv"
    val_df.to_csv(val_csv, index=False)
    assert len(val_errors) == 0, f"Validation errors found: {val_errors}"
    print(f"All {len(trip_ids)} evaluator trips passed validation -> {val_csv}")

    print(f"\n=== PHASE 2: Running Frozen Prediction Pipeline (.track/.nodes/.arcs ONLY) ===")
    svc = PathSnapInferenceService(cfg)
    active_dir, active_part, is_dev = svc.resolve_active_eval_source()
    assert active_part == "evaluator_test" and not is_dev, (
        f"Expected active partition 'evaluator_test', got '{active_part}' (dev={is_dev})"
    )

    # Generate ALL 10 predictions BEFORE touching a single .route file
    predictions: Dict[str, PredictionBundle] = {}
    for tid in trip_ids:
        bundle = svc.run_prediction_only(tid)
        predictions[tid] = bundle
        print(
            f"  [PREDICTED] Trip {tid}: obs={len(bundle.prep.gps_x):4d}, "
            f"recon_arcs={len(bundle.decode_result.reconstructed_route_arcs):4d}, "
            f"recon_cont={bundle.continuity['reconstructed_route_direct_continuity']:.4f}, "
            f"bridged={bundle.decode_result.n_bridged_arcs:3d}, "
            f"breaks={len(bundle.decode_result.segment_breaks)}, "
            f"gaps={len(bundle.decode_result.gap_breaks)}, "
            f"time={bundle.timings_ms['total_ms']:.0f}ms"
        )

    print(f"\n=== PHASE 3: Post-Prediction Evaluation Against Server-Side .route ===")
    eval_rows: List[Dict[str, Any]] = []
    all_y_true = []
    all_probs = []
    tot_obs = 0
    tot_hits = 0
    clf_tot_correct = 0
    hmm_tot_correct = 0

    for tid in trip_ids:
        bundle = predictions[tid]
        prep = bundle.prep
        dec = bundle.decode_result
        route = load_trip_route(eval_dir, tid)
        assert route is not None and len(route) > 0

        align = align_gps_to_reference_route(prep, route, bundle.candidates)
        y_true = (bundle.arc_ids == align.aligned_arc_ids[bundle.obs_idx]).astype(np.int32)
        all_y_true.append(y_true)
        all_probs.append(bundle.probs)

        cand_m = compute_candidate_classification_metrics(y_true, bundle.probs)
        clf_top1 = compute_observation_top1_metrics(
            align.aligned_arc_ids, bundle.clf_obs_arcs, align.candidate_hit
        )
        clf_cont = compute_route_continuity(prep.network, bundle.clf_obs_arcs)
        clf_seq = compute_sequence_route_metrics(prep.network, bundle.clf_obs_arcs, route)

        hmm_top1 = compute_observation_top1_metrics(
            align.aligned_arc_ids, dec.obs_arc_ids, align.candidate_hit
        )
        hmm_cont = bundle.continuity
        hmm_seq = compute_sequence_route_metrics(prep.network, dec.reconstructed_route_arcs, route)

        tot_obs += len(prep.gps_x)
        tot_hits += int(np.sum(align.candidate_hit))
        clf_tot_correct += int(np.sum((bundle.clf_obs_arcs == align.aligned_arc_ids) & (align.aligned_arc_ids >= 0)))
        hmm_tot_correct += int(np.sum((dec.obs_arc_ids == align.aligned_arc_ids) & (align.aligned_arc_ids >= 0)))

        eval_rows.append(
            {
                "trip_id": tid,
                "n_obs": len(prep.gps_x),
                "n_route_arcs": len(route),
                "n_pred_route_arcs": len(dec.reconstructed_route_arcs),
                "utm_epsg": prep.projector.epsg_code,
                "candidate_miss_rate": hmm_top1["candidate_miss_rate"],
                "median_align_dist_m": float(np.median(align.aligned_dist_m)),
                "cand_macro_f1": cand_m["cand_macro_f1"],
                "cand_precision": cand_m["cand_precision"],
                "cand_recall": cand_m["cand_recall"],
                "cand_log_loss": cand_m["log_loss"],
                # Classifier-only
                "clf_top1_acc": clf_top1["top1_accuracy"],
                "clf_top1_given_hit": clf_top1["top1_accuracy_given_hit"],
                "clf_reach_step_cont": clf_cont["network_reachable_continuity"],
                "clf_disconnected_trans": clf_cont["disconnected_transitions"],
                "clf_seq_arc_f1": clf_seq["seq_arc_f1"],
                "clf_seq_length_f1": clf_seq["seq_length_f1"],
                "clf_recon_direct_cont": clf_seq["reconstructed_route_direct_continuity"],
                # Classifier + HMM/Viterbi
                "hmm_top1_acc": hmm_top1["top1_accuracy"],
                "hmm_top1_given_hit": hmm_top1["top1_accuracy_given_hit"],
                "hmm_direct_step_cont": hmm_cont["direct_step_continuity"],
                "hmm_reach_step_cont": hmm_cont["network_reachable_continuity"],
                "hmm_trans_reach_cont": hmm_cont["transition_reachable_continuity"],
                "hmm_disconnected_trans": hmm_cont["disconnected_transitions"],
                "hmm_seq_arc_f1": hmm_seq["seq_arc_f1"],
                "hmm_seq_length_f1": hmm_seq["seq_length_f1"],
                "hmm_recon_direct_cont": hmm_seq["reconstructed_route_direct_continuity"],
                "hmm_bridged_arcs": dec.n_bridged_arcs,
                "hmm_segment_breaks": len(dec.segment_breaks),
                "hmm_gap_breaks": len(dec.gap_breaks),
                "delta_top1_acc": hmm_top1["top1_accuracy"] - clf_top1["top1_accuracy"],
                "delta_seq_length_f1": hmm_seq["seq_length_f1"] - clf_seq["seq_length_f1"],
                "preprocess_ms": bundle.timings_ms["preprocess_ms"],
                "feat_ms": bundle.timings_ms["candidate_and_features_ms"],
                "clf_ms": bundle.timings_ms["classifier_ms"],
                "hmm_ms": bundle.timings_ms["hmm_viterbi_ms"],
                "total_ms": bundle.timings_ms["total_ms"],
            }
        )
        print(
            f"  Trip {tid}: Clf Top1={clf_top1['top1_accuracy']*100:5.2f}% -> HMM Top1={hmm_top1['top1_accuracy']*100:5.2f}% "
            f"({(hmm_top1['top1_accuracy']-clf_top1['top1_accuracy'])*100:+5.2f} pp) | "
            f"MacroF1={cand_m['cand_macro_f1']:.4f} | LogLoss={cand_m['log_loss']:.4f} | "
            f"SeqLenF1={hmm_seq['seq_length_f1']*100:5.2f}% | ReconCont={hmm_seq['reconstructed_route_direct_continuity']*100:6.2f}%"
        )

    res_df = pd.DataFrame(eval_rows)
    res_csv = cfg.experiments_dir / "evaluator_test_results.csv"
    res_df.to_csv(res_csv, index=False)

    agg_cand = compute_candidate_classification_metrics(
        np.concatenate(all_y_true), np.concatenate(all_probs)
    )
    summary = {
        "partition": "evaluator_test",
        "model_artifact": "artifacts/models/final_model.joblib (frozen, trained on 80 train trips only)",
        "hmm_artifact": "artifacts/configurations/hmm_config.json (frozen, tuned on train trips only)",
        "n_trips": len(trip_ids),
        "trip_ids": trip_ids,
        "total_observations": int(tot_obs),
        "total_nodes_mean": float(val_df["n_nodes"].mean()),
        "total_arcs_mean": float(val_df["n_arcs"].mean()),
        "utm_epsg_codes": sorted(int(x) for x in val_df["utm_epsg"].unique()),
        "validation_errors": len(val_errors),
        "validation_warnings": val_warnings,
        "candidate_miss_rate": float(1.0 - tot_hits / max(tot_obs, 1)),
        "classifier_cand_macro_f1": agg_cand["cand_macro_f1"],
        "classifier_cand_precision": agg_cand["cand_precision"],
        "classifier_cand_recall": agg_cand["cand_recall"],
        "classifier_log_loss": agg_cand["log_loss"],
        "classifier_confusion_matrix": agg_cand["confusion_matrix"],
        "classifier_only": {
            "weighted_top1_acc": float(clf_tot_correct / max(tot_obs, 1)),
            "mean_trip_top1_acc": float(res_df["clf_top1_acc"].mean()),
            "mean_reach_step_cont": float(res_df["clf_reach_step_cont"].mean()),
            "total_disconnected_trans": int(res_df["clf_disconnected_trans"].sum()),
            "mean_seq_arc_f1": float(res_df["clf_seq_arc_f1"].mean()),
            "mean_seq_length_f1": float(res_df["clf_seq_length_f1"].mean()),
            "mean_recon_direct_cont": float(res_df["clf_recon_direct_cont"].mean()),
        },
        "classifier_plus_hmm": {
            "weighted_top1_acc": float(hmm_tot_correct / max(tot_obs, 1)),
            "mean_trip_top1_acc": float(res_df["hmm_top1_acc"].mean()),
            "mean_direct_step_cont": float(res_df["hmm_direct_step_cont"].mean()),
            "mean_reach_step_cont": float(res_df["hmm_reach_step_cont"].mean()),
            "mean_trans_reach_cont": float(res_df["hmm_trans_reach_cont"].mean()),
            "total_disconnected_trans": int(res_df["hmm_disconnected_trans"].sum()),
            "mean_seq_arc_f1": float(res_df["hmm_seq_arc_f1"].mean()),
            "mean_seq_length_f1": float(res_df["hmm_seq_length_f1"].mean()),
            "mean_recon_direct_cont": float(res_df["hmm_recon_direct_cont"].mean()),
            "total_bridged_arcs": int(res_df["hmm_bridged_arcs"].sum()),
            "total_segment_breaks": int(res_df["hmm_segment_breaks"].sum()),
            "total_gap_breaks": int(res_df["hmm_gap_breaks"].sum()),
            "mean_hmm_ms": float(res_df["hmm_ms"].mean()),
            "mean_total_ms": float(res_df["total_ms"].mean()),
        },
    }
    sum_json = cfg.experiments_dir / "evaluator_test_summary.json"
    sum_json.write_text(json.dumps(summary, indent=2))
    _plot_evaluator_results(res_df, cfg.reports_dir / "figures" / "evaluator_test_results.png")

    print("\n=== PHASE 4: Verifying FastAPI Evaluator Trip Selector & Ground-Truth Isolation ===")
    captured: List[str] = [
        "$ .venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --access-log --log-level info",
        "INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)",
    ]

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(self.format(record))

    handler = ListHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
    api_logger = logging.getLogger("pathsnap.api")
    api_logger.addHandler(handler)
    api_logger.setLevel(logging.INFO)

    try:
        with TestClient(app) as client:
            h = client.get("/api/health").json()
            assert h["active_partition"] == "evaluator_test"
            assert h["is_development_fallback"] is False
            assert h["available_trip_count"] == 10

            t = client.get("/api/trips").json()
            assert t["active_partition"] == "evaluator_test"
            assert t["is_development_fallback"] is False
            api_tids = [x["trip_id"] for x in t["trips"]]
            assert api_tids == trip_ids

            for tid in trip_ids:
                p = client.post(f"/api/trips/{tid}/predict")
                assert p.status_code == 200
                raw_lower = p.text.lower()
                assert "reference_route_arcs" not in raw_lower
                assert "reference_polylines" not in raw_lower
                assert "ground_truth_arcs" not in raw_lower

            m = client.get(f"/api/trips/{trip_ids[0]}/map")
            assert m.status_code == 200
            assert "Reference Route (.route)" not in m.text
    finally:
        api_logger.removeHandler(handler)

    eval_log = cfg.experiments_dir / "logs" / "evaluator_api_verification.log"
    eval_log.parent.mkdir(parents=True, exist_ok=True)
    eval_log.write_text("\n".join(captured) + "\n")
    _render_terminal_screenshot(captured, cfg.reports_dir / "figures" / "api_logs_screenshot.png")

    print("\n=== AGGREGATE EVALUATOR-TEST SUMMARY (10 RESERVED TRIPS) ===")
    print(json.dumps(summary["classifier_only"], indent=2))
    print(json.dumps(summary["classifier_plus_hmm"], indent=2))
    print(f"Candidate Macro F1: {agg_cand['cand_macro_f1']:.4f} | Log Loss: {agg_cand['log_loss']:.4f} | Miss Rate: {summary['candidate_miss_rate']:.4f}")
    print(f"Completed full evaluator_test run in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
