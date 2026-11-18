"""Train and compare six distinct ML model families using GroupKFold CV and student_test."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.candidate_generation import CandidateConfig
from src.models.dataset_cache import PartitionFeatureDataset, build_or_load_partition_cache
from src.models.evaluate import evaluate_predictions_on_partition, load_prepared_trips
from src.models.registry import ModelSpec, get_model_specs


def _filter_aligned_train_rows(train_ds: PartitionFeatureDataset, indices: np.ndarray) -> np.ndarray:
    """Keep training-fold candidate rows whose observation has a positive aligned candidate.

    Avoids teaching the classifier contradictory y=0 labels on off-route GPS segments
    (e.g., trips 00000054/00000033 where GPS drifts >100m away from the reference .route).
    Applied STRICTLY to training folds — never to validation folds or student_test.
    """
    keep_mask = np.zeros(len(indices), dtype=bool)
    # Build (trip_group, obs_idx) -> hit lookup via precomputed per-trip candidate_hit
    hit_by_group = {
        meta["group_idx"]: meta["candidate_hit"] for meta in train_ds.trip_meta.values()
    }
    grps = train_ds.trip_group[indices]
    obs = train_ds.obs_idx[indices]
    for g in np.unique(grps):
        g_mask = grps == g
        hit_arr = hit_by_group[int(g)]
        keep_mask[g_mask] = hit_arr[obs[g_mask]]
    return indices[keep_mask]


def run_group_cv(
    spec: ModelSpec,
    train_ds: PartitionFeatureDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run GroupKFold cross-validation grouped strictly by trip_id."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(train_ds.y), dtype=np.float64)

    fold_macro_f1: List[float] = []
    fold_top1: List[float] = []
    fold_logloss: List[float] = []
    fold_fit_times: List[float] = []
    fold_pred_times: List[float] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(train_ds.X, train_ds.y, groups=train_ds.trip_group)
    ):
        tr_clean = _filter_aligned_train_rows(train_ds, tr_idx)
        model = spec.factory(seed + fold_idx)
        t0 = time.perf_counter()
        model.fit(train_ds.X[tr_clean], train_ds.y[tr_clean])
        fit_t = time.perf_counter() - t0

        t1 = time.perf_counter()
        va_probs = model.predict_proba(train_ds.X[va_idx])[:, 1]
        pred_t = time.perf_counter() - t1

        oof_probs[va_idx] = va_probs
        fold_fit_times.append(fit_t)
        fold_pred_times.append(pred_t)

        # Evaluate this fold's validation trips
        val_groups = set(int(g) for g in np.unique(train_ds.trip_group[va_idx]))
        val_tids = [train_ds.trip_ids[g] for g in sorted(val_groups)]
        fold_m = evaluate_predictions_on_partition(train_ds, oof_probs, trip_ids_subset=val_tids)
        fold_macro_f1.append(fold_m["cand_macro_f1"])
        fold_top1.append(fold_m["top1_accuracy"])
        fold_logloss.append(fold_m["log_loss"])

    oof_overall = evaluate_predictions_on_partition(train_ds, oof_probs)
    return {
        "cv_macro_f1_mean": float(np.mean(fold_macro_f1)),
        "cv_macro_f1_std": float(np.std(fold_macro_f1)),
        "cv_top1_mean": float(np.mean(fold_top1)),
        "cv_top1_std": float(np.std(fold_top1)),
        "cv_log_loss_mean": float(np.mean(fold_logloss)),
        "cv_log_loss_std": float(np.std(fold_logloss)),
        "cv_oof_macro_f1": oof_overall["cand_macro_f1"],
        "cv_oof_top1": oof_overall["top1_accuracy"],
        "cv_oof_log_loss": oof_overall["log_loss"],
        "cv_mean_fit_time_s": float(np.mean(fold_fit_times)),
        "cv_mean_pred_time_s": float(np.mean(fold_pred_times)),
        "fold_macro_f1": fold_macro_f1,
        "fold_top1": fold_top1,
        "fold_logloss": fold_logloss,
    }


def _generate_comparison_plots(
    comp_df: pd.DataFrame,
    details: Dict[str, Any],
    fig_dir: Path,
) -> None:
    """Generate actual experimental comparison charts for Section 3.3 of the report."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    names = comp_df["model_name"].tolist()
    short_names = [n.replace(" Classifier", "").replace(" Neural Network", "") for n in names]
    x = np.arange(len(names))
    colors = ["#64748b", "#0ea5e9", "#f59e0b", "#10b981", "#2563eb", "#8b5cf6"]

    # 1. Cross-Validation Macro F1-score with fold error bars
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    bars = ax.bar(
        x,
        comp_df["cv_macro_f1_mean"],
        yerr=comp_df["cv_macro_f1_std"],
        capsize=5,
        color=colors[: len(names)],
        alpha=0.9,
        edgecolor="#0f172a",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("5-Fold GroupKFold CV Macro F1-score")
    ax.set_title("Cross-Validation Macro F1-score Across Six Model Families (80 Train Trips)", fontweight="bold")
    ax.set_ylim(max(0.0, comp_df["cv_macro_f1_mean"].min() - 0.08), min(1.0, comp_df["cv_macro_f1_mean"].max() + 0.05))
    for bar, val, std in zip(bars, comp_df["cv_macro_f1_mean"], comp_df["cv_macro_f1_std"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + std + 0.006,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    plt.tight_layout()
    fig.savefig(fig_dir / "model_cv_macro_f1.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Student-Testing Top-1 Accuracy, Macro F1, and Route Continuity
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    w = 0.26
    b1 = ax.bar(x - w, comp_df["student_test_top1_acc"], width=w, label="Obs-Level Top-1 Accuracy", color="#2563eb")
    b2 = ax.bar(x, comp_df["student_test_macro_f1"], width=w, label="Candidate Macro F1", color="#10b981")
    b3 = ax.bar(x + w, comp_df["student_test_route_continuity"], width=w, label="Network Route Continuity", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("Student-Testing Performance Comparison (10 Held-Out Student-Test Trips)", fontweight="bold")
    ax.set_ylim(0.5, 1.04)
    ax.legend(loc="lower right", frameon=True)
    for bars_grp in (b1, b2, b3):
        for bar in bars_grp:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    fig.savefig(fig_dir / "model_student_test_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Log Loss & Inference Time Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160)
    b_ll = ax1.bar(x, comp_df["student_test_log_loss"], color=colors[: len(names)], alpha=0.9, edgecolor="#0f172a", linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=18, ha="right", fontsize=9)
    ax1.set_ylabel("Binary Log Loss (Lower is Better)")
    ax1.set_title("Probability Quality: Student-Test Log Loss", fontweight="bold")
    for bar, val in zip(b_ll, comp_df["student_test_log_loss"]):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, val + 0.005, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5)

    b_rt = ax2.bar(x, comp_df["student_test_infer_time_s"] * 1000.0 / 10.0, color=colors[: len(names)], alpha=0.9, edgecolor="#0f172a", linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=18, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Inference Time per Trip (ms)")
    ax2.set_title("Inference Latency per Student-Test Trip", fontweight="bold")
    ax2.set_yscale("log")
    for bar, val in zip(b_rt, comp_df["student_test_infer_time_s"] * 1000.0 / 10.0):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, val * 1.12, f"{val:.1f}ms", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    fig.savefig(fig_dir / "model_log_loss_and_runtime.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Confusion Matrices grid (2 x 3)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), dpi=160)
    for ax, (_, row) in zip(axes.flat, comp_df.iterrows()):
        key = row["model_key"]
        cm = np.array(details[key]["student_test"]["confusion_matrix"], dtype=np.int64)
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Non-Match", "Pred Match"], fontsize=8.5)
        ax.set_yticklabels(["True Non-Match", "True Match"], fontsize=8.5)
        ax.set_title(row["model_name"], fontsize=10, fontweight="bold")
        for i in range(2):
            for j in range(2):
                txt_col = "white" if cm_norm[i, j] > 0.55 else "#0f172a"
                ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color=txt_col, fontsize=8.5)
    fig.suptitle("Candidate-Level Confusion Matrices on Student-Testing Partition (10 Trips)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_dir / "model_confusion_matrices.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Actual Epoch-wise Training & Validation Curve for MLP (only for iterative neural model)
    mlp_info = details.get("mlp", {})
    loss_curve = mlp_info.get("mlp_loss_curve")
    val_scores = mlp_info.get("mlp_validation_scores")
    if loss_curve:
        epochs = np.arange(1, len(loss_curve) + 1)
        fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=160)
        l1 = ax1.plot(epochs, loss_curve, marker="o", color="#2563eb", linewidth=2, label="Training Log Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Cross-Entropy Loss", color="#2563eb")
        ax1.tick_params(axis="y", labelcolor="#2563eb")
        ax1.set_title("MLP Neural Network: Actual Epoch-Wise Training Loss & Validation Accuracy", fontweight="bold")
        lines = l1
        if val_scores:
            ax2 = ax1.twinx()
            l2 = ax2.plot(epochs, val_scores, marker="s", color="#10b981", linewidth=2, linestyle="--", label="Internal Validation Accuracy")
            ax2.set_ylabel("Validation Accuracy (10% holdout)", color="#10b981")
            ax2.tick_params(axis="y", labelcolor="#10b981")
            lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")
        plt.tight_layout()
        fig.savefig(fig_dir / "mlp_training_curve.png", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    t_all = time.perf_counter()
    cfg = get_config()
    cand_cfg = CandidateConfig(
        radius_m=cfg.candidate_radius_m,
        fallback_radius_m=cfg.candidate_fallback_radius_m,
        top_k=cfg.candidate_top_k,
    )

    print("Loading cached train & student_test datasets...")
    train_ds = build_or_load_partition_cache(cfg, "train", cfg.train_dir, cand_cfg, force_rebuild=False)
    stest_ds = build_or_load_partition_cache(cfg, "student_test", cfg.student_test_dir, cand_cfg, force_rebuild=False)
    stest_prepared = load_prepared_trips(cfg.student_test_dir, stest_ds.trip_ids, "student_test")

    specs = get_model_specs()
    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}

    for spec in specs:
        print(f"\n=== Evaluating {spec.name} ({spec.family}) ===")
        cv_res = run_group_cv(spec, train_ds, n_splits=5, seed=cfg.random_seed)

        # Fit final untuned model on all 80 permitted training trips ONLY (aligned training observations)
        all_tr_clean = _filter_aligned_train_rows(train_ds, np.arange(len(train_ds.y)))
        model = spec.factory(cfg.random_seed)
        t_fit0 = time.perf_counter()
        model.fit(train_ds.X[all_tr_clean], train_ds.y[all_tr_clean])
        full_fit_time = time.perf_counter() - t_fit0

        # Evaluate on a consistent 10-trip training diagnostic slice (for overfitting/underfitting diagnostics)
        diag_tids = train_ds.trip_ids[:10]
        diag_end = train_ds.trip_meta[diag_tids[-1]]["row_end"]
        train_probs_diag = np.zeros(len(train_ds.y), dtype=np.float64)
        train_probs_diag[:diag_end] = model.predict_proba(train_ds.X[:diag_end])[:, 1]
        train_eval = evaluate_predictions_on_partition(
            train_ds, train_probs_diag, trip_ids_subset=diag_tids
        )

        # Evaluate on 10 student-testing trips
        t_inf0 = time.perf_counter()
        stest_probs = model.predict_proba(stest_ds.X)[:, 1]
        stest_infer_time = time.perf_counter() - t_inf0
        stest_eval = evaluate_predictions_on_partition(
            stest_ds, stest_probs, prepared_trips=stest_prepared
        )

        row = {
            "model_key": spec.key,
            "model_name": spec.name,
            "algorithm_family": spec.family,
            "key_configuration": spec.config_summary,
            "loss_objective": spec.loss_objective,
            "train_macro_f1": train_eval["cand_macro_f1"],
            "cv_macro_f1_mean": cv_res["cv_macro_f1_mean"],
            "cv_macro_f1_std": cv_res["cv_macro_f1_std"],
            "cv_top1_mean": cv_res["cv_top1_mean"],
            "cv_log_loss_mean": cv_res["cv_log_loss_mean"],
            "student_test_top1_acc": stest_eval["top1_accuracy"],
            "student_test_top1_given_hit": stest_eval["top1_accuracy_given_hit"],
            "student_test_miss_rate": stest_eval["candidate_miss_rate"],
            "student_test_macro_f1": stest_eval["cand_macro_f1"],
            "student_test_pos_f1": stest_eval["cand_pos_f1"],
            "student_test_precision": stest_eval["cand_precision"],
            "student_test_recall": stest_eval["cand_recall"],
            "student_test_accuracy": stest_eval["cand_accuracy"],
            "student_test_log_loss": stest_eval["log_loss"],
            "student_test_route_continuity": stest_eval["network_reachable_continuity"],
            "student_test_direct_continuity": stest_eval["direct_step_continuity"],
            "student_test_trans_continuity": stest_eval["transition_reachable_continuity"],
            "train_time_s": full_fit_time,
            "student_test_infer_time_s": stest_infer_time,
        }
        rows.append(row)

        model_detail: Dict[str, Any] = {
            "spec": {
                "key": spec.key,
                "name": spec.name,
                "family": spec.family,
                "config_summary": spec.config_summary,
                "loss_objective": spec.loss_objective,
            },
            "cv": cv_res,
            "train_diag": {
                "cand_macro_f1": train_eval["cand_macro_f1"],
                "top1_accuracy": train_eval["top1_accuracy"],
                "log_loss": train_eval["log_loss"],
            },
            "student_test": stest_eval,
            "train_time_s": full_fit_time,
            "student_test_infer_time_s": stest_infer_time,
        }

        # Capture real MLP epoch-wise curves
        if spec.key == "mlp":
            mlp_clf = model.named_steps["clf"]
            model_detail["mlp_loss_curve"] = [float(v) for v in getattr(mlp_clf, "loss_curve_", [])]
            model_detail["mlp_validation_scores"] = [float(v) for v in getattr(mlp_clf, "validation_scores_", [])]

        details[spec.key] = model_detail

        print(
            f"  CV Macro F1: {cv_res['cv_macro_f1_mean']:.4f} ± {cv_res['cv_macro_f1_std']:.4f} | "
            f"Train F1: {train_eval['cand_macro_f1']:.4f} | "
            f"STest Macro F1: {stest_eval['cand_macro_f1']:.4f} | "
            f"STest Top-1: {stest_eval['top1_accuracy']:.4f} | "
            f"LogLoss: {stest_eval['log_loss']:.4f} | "
            f"RouteCont: {stest_eval['network_reachable_continuity']:.4f} | "
            f"Fit: {full_fit_time:.2f}s | Inf: {stest_infer_time*1000:.1f}ms"
        )

    comp_df = pd.DataFrame(rows)
    out_csv = cfg.experiments_dir / "model_comparison.csv"
    comp_df.to_csv(out_csv, index=False)

    out_json = cfg.experiments_dir / "model_comparison_details.json"
    out_json.write_text(json.dumps(details, indent=2))

    _generate_comparison_plots(comp_df, details, cfg.reports_dir / "figures")

    print(f"\nSaved comparison CSV -> {out_csv}")
    print(f"Saved detailed JSON -> {out_json}")
    print(f"Completed 6-model comparison in {time.perf_counter() - t_all:.1f}s")


if __name__ == "__main__":
    main()
