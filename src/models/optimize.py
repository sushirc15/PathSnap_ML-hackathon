"""Bayesian hyperparameter optimization (Optuna TPE) for the winning PathSnap classifier."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.candidate_generation import CandidateConfig
from src.features.feature_extraction import FEATURE_COLUMNS
from src.models.compare import _filter_aligned_train_rows
from src.models.dataset_cache import PartitionFeatureDataset, build_or_load_partition_cache
from src.models.evaluate import evaluate_predictions_on_partition, load_prepared_trips

optuna.logging.set_verbosity(optuna.logging.WARNING)


def evaluate_hgb_cv(
    params: Dict[str, Any],
    train_ds: PartitionFeatureDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate a HistGradientBoosting configuration using 5-fold GroupKFold CV."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros(len(train_ds.y), dtype=np.float64)
    fold_f1 = []
    fold_top1 = []
    fold_ll = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        gkf.split(train_ds.X, train_ds.y, groups=train_ds.trip_group)
    ):
        tr_clean = _filter_aligned_train_rows(train_ds, tr_idx)
        clf = HistGradientBoostingClassifier(
            loss="log_loss",
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed + fold_idx,
            **params,
        )
        clf.fit(train_ds.X[tr_clean], train_ds.y[tr_clean])
        va_probs = clf.predict_proba(train_ds.X[va_idx])[:, 1]
        oof_probs[va_idx] = va_probs

        val_groups = set(int(g) for g in np.unique(train_ds.trip_group[va_idx]))
        val_tids = [train_ds.trip_ids[g] for g in sorted(val_groups)]
        fm = evaluate_predictions_on_partition(train_ds, oof_probs, trip_ids_subset=val_tids)
        fold_f1.append(fm["cand_macro_f1"])
        fold_top1.append(fm["top1_accuracy"])
        fold_ll.append(fm["log_loss"])

    return {
        "cv_macro_f1_mean": float(np.mean(fold_f1)),
        "cv_macro_f1_std": float(np.std(fold_f1)),
        "cv_top1_mean": float(np.mean(fold_top1)),
        "cv_log_loss_mean": float(np.mean(fold_ll)),
    }


def _plot_optuna_results(trials_df: pd.DataFrame, comparison: Dict[str, Any], fig_dir: Path) -> None:
    """Plot Optuna optimization progress and untuned vs tuned performance."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=160)

    # Left: Trial-by-trial CV Macro F1 & best-so-far envelope
    t_nums = trials_df["trial_number"].to_numpy()
    vals = trials_df["cv_macro_f1_mean"].to_numpy()
    best_so_far = np.maximum.accumulate(vals)

    ax1.scatter(t_nums, vals, c="#3b82f6", alpha=0.75, s=42, label="Trial CV Macro F1")
    ax1.plot(t_nums, best_so_far, color="#dc2626", linewidth=2.2, label="Best So Far")
    ax1.axhline(
        comparison["untuned"]["cv_macro_f1_mean"],
        color="#64748b",
        linestyle="--",
        linewidth=1.5,
        label=f"Untuned Baseline ({comparison['untuned']['cv_macro_f1_mean']:.4f})",
    )
    ax1.set_xlabel("Optuna TPE Trial Number")
    ax1.set_ylabel("5-Fold GroupKFold CV Macro F1")
    ax1.set_title("Bayesian Hyperparameter Optimization History (Optuna TPE)", fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    # Right: Untuned vs Tuned bar comparison across key metrics
    metrics_labels = ["CV Macro F1", "STest Top-1 Acc", "STest Macro F1", "Route Continuity"]
    untuned_vals = [
        comparison["untuned"]["cv_macro_f1_mean"],
        comparison["untuned"]["student_test_top1_acc"],
        comparison["untuned"]["student_test_macro_f1"],
        comparison["untuned"]["student_test_route_continuity"],
    ]
    tuned_vals = [
        comparison["tuned"]["cv_macro_f1_mean"],
        comparison["tuned"]["student_test_top1_acc"],
        comparison["tuned"]["student_test_macro_f1"],
        comparison["tuned"]["student_test_route_continuity"],
    ]
    x = np.arange(len(metrics_labels))
    w = 0.34
    b1 = ax2.bar(x - w / 2, untuned_vals, width=w, label="Untuned HGB", color="#94a3b8")
    b2 = ax2.bar(x + w / 2, tuned_vals, width=w, label="Bayesian-Tuned HGB", color="#2563eb")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels, fontsize=9.5)
    ax2.set_ylabel("Score")
    ax2.set_ylim(min(min(untuned_vals), min(tuned_vals)) - 0.04, 1.02)
    ax2.set_title("Untuned vs. Bayesian-Tuned HistGradientBoosting", fontweight="bold")
    ax2.legend(loc="lower right", frameon=True)
    for grp in (b1, b2):
        for bar in grp:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(fig_dir / "optuna_optimization.png", bbox_inches="tight")
    plt.close(fig)


def main(n_trials: int = 15) -> None:
    t0 = time.perf_counter()
    cfg = get_config()
    cand_cfg = CandidateConfig(
        radius_m=cfg.candidate_radius_m,
        fallback_radius_m=cfg.candidate_fallback_radius_m,
        top_k=cfg.candidate_top_k,
    )

    train_ds = build_or_load_partition_cache(cfg, "train", cfg.train_dir, cand_cfg, force_rebuild=False)
    stest_ds = build_or_load_partition_cache(cfg, "student_test", cfg.student_test_dir, cand_cfg, force_rebuild=False)
    stest_prepared = load_prepared_trips(cfg.student_test_dir, stest_ds.trip_ids, "student_test")

    # Baseline untuned HGB configuration (identical to compare.py)
    untuned_params = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "max_bins": 255,
    }
    print("Evaluating untuned HistGradientBoosting baseline (5-fold GroupKFold)...", flush=True)
    untuned_cv = evaluate_hgb_cv(untuned_params, train_ds, n_splits=5, seed=cfg.random_seed)

    all_tr_clean = _filter_aligned_train_rows(train_ds, np.arange(len(train_ds.y)))
    untuned_clf = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=cfg.random_seed,
        **untuned_params,
    )
    tf0 = time.perf_counter()
    untuned_clf.fit(train_ds.X[all_tr_clean], train_ds.y[all_tr_clean])
    untuned_fit_s = time.perf_counter() - tf0

    ti0 = time.perf_counter()
    untuned_st_probs = untuned_clf.predict_proba(stest_ds.X)[:, 1]
    untuned_inf_s = time.perf_counter() - ti0
    untuned_st = evaluate_predictions_on_partition(stest_ds, untuned_st_probs, prepared_trips=stest_prepared)

    # Optuna TPE Bayesian optimization over 3-fold GroupKFold CV (80 training trips ONLY)
    # Followed by exact 5-fold GroupKFold confirmation of the winning configuration.
    search_space_doc = {
        "learning_rate": {"type": "float_log", "low": 0.03, "high": 0.20},
        "max_iter": {"type": "int", "low": 150, "high": 390, "step": 30},
        "max_leaf_nodes": {"type": "int", "low": 31, "high": 127},
        "max_depth": {"type": "categorical", "choices": [None, 10, 14, 18]},
        "min_samples_leaf": {"type": "int", "low": 20, "high": 140},
        "l2_regularization": {"type": "float_log", "low": 1e-2, "high": 15.0},
        "max_bins": {"type": "categorical", "choices": [128, 255]},
    }

    trial_records = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.20, log=True),
            "max_iter": trial.suggest_int("max_iter", 150, 390, step=30),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 31, 127),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 14, 18]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 140),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-2, 15.0, log=True),
            "max_bins": trial.suggest_categorical("max_bins", [128, 255]),
        }
        ts = time.perf_counter()
        cv_m = evaluate_hgb_cv(params, train_ds, n_splits=3, seed=cfg.random_seed)
        elapsed = time.perf_counter() - ts

        trial.set_user_attr("cv_top1_mean", cv_m["cv_top1_mean"])
        trial.set_user_attr("cv_log_loss_mean", cv_m["cv_log_loss_mean"])
        trial.set_user_attr("cv_macro_f1_std", cv_m["cv_macro_f1_std"])
        trial.set_user_attr("duration_s", elapsed)

        rec = {
            "trial_number": trial.number,
            **params,
            **cv_m,
            "duration_s": elapsed,
        }
        trial_records.append(rec)
        print(
            f"  [Trial {trial.number + 1:02d}/{n_trials}] "
            f"3-Fold CV Macro F1={cv_m['cv_macro_f1_mean']:.4f} ± {cv_m['cv_macro_f1_std']:.4f} | "
            f"CV Top-1={cv_m['cv_top1_mean']:.4f} | CV LogLoss={cv_m['cv_log_loss_mean']:.4f} | "
            f"({elapsed:.1f}s)",
            flush=True,
        )
        return cv_m["cv_macro_f1_mean"]

    print(f"\nRunning Optuna TPE Bayesian optimization ({n_trials} trials, 3-fold GroupKFold)...", flush=True)
    sampler = optuna.samplers.TPESampler(seed=cfg.random_seed, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="pathsnap_hgb_tpe")
    # Enqueue baseline as trial 0 so TPE always evaluates from the baseline anchor
    study.enqueue_trial(
        {
            "learning_rate": 0.08,
            "max_iter": 200,
            "max_leaf_nodes": 63,
            "max_depth": None,
            "min_samples_leaf": 30,
            "l2_regularization": 1.0,
            "max_bins": 255,
        }
    )
    opt_t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials)
    opt_duration_s = time.perf_counter() - opt_t0

    best_params = dict(study.best_params)
    best_trial = study.best_trial
    print(f"\nBest Trial #{best_trial.number}: 3-Fold CV Macro F1 = {best_trial.value:.4f}", flush=True)
    print(f"Best Params: {json.dumps(best_params, indent=2)}", flush=True)

    # Confirm tuned model on exact same 5-fold GroupKFold splits as untuned baseline
    print("Confirming tuned model on exact 5-fold GroupKFold splits...", flush=True)
    tuned_cv5 = evaluate_hgb_cv(best_params, train_ds, n_splits=5, seed=cfg.random_seed)

    # Refit best model on all 80 permitted training trips ONLY (aligned observations)
    tuned_clf = HistGradientBoostingClassifier(
        loss="log_loss",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        random_state=cfg.random_seed,
        **best_params,
    )
    tf1 = time.perf_counter()
    tuned_clf.fit(train_ds.X[all_tr_clean], train_ds.y[all_tr_clean])
    tuned_fit_s = time.perf_counter() - tf1

    ti1 = time.perf_counter()
    tuned_st_probs = tuned_clf.predict_proba(stest_ds.X)[:, 1]
    tuned_inf_s = time.perf_counter() - ti1
    tuned_st = evaluate_predictions_on_partition(stest_ds, tuned_st_probs, prepared_trips=stest_prepared)

    # Save trials CSV
    trials_df = pd.DataFrame(trial_records)
    trials_csv = cfg.experiments_dir / "optimization_trials.csv"
    trials_df.to_csv(trials_csv, index=False)

    # Save final trained model & artifacts
    model_path = cfg.artifacts_dir / "models" / "final_model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tuned_clf, model_path)

    feature_cfg_path = cfg.artifacts_dir / "preprocessors" / "feature_config.json"
    feature_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    feature_cfg_path.write_text(
        json.dumps(
            {
                "feature_columns": FEATURE_COLUMNS,
                "n_features": len(FEATURE_COLUMNS),
                "candidate_config": {
                    "radius_m": cand_cfg.radius_m,
                    "fallback_radius_m": cand_cfg.fallback_radius_m,
                    "top_k": cand_cfg.top_k,
                },
                "model_type": "HistGradientBoostingClassifier",
                "trained_partition": "train (80 trips only)",
                "random_seed": cfg.random_seed,
            },
            indent=2,
        )
    )

    comparison = {
        "sampler": "Optuna TPESampler (Tree-structured Parzen Estimator)",
        "objective": "3-fold GroupKFold CV Macro F1 during TPE search, confirmed via 5-fold GroupKFold CV (grouped by trip_id)",
        "n_trials": n_trials,
        "optimization_time_s": opt_duration_s,
        "search_space": search_space_doc,
        "best_trial_number": int(best_trial.number),
        "best_params": best_params,
        "untuned": {
            "params": untuned_params,
            **untuned_cv,
            "student_test_top1_acc": untuned_st["top1_accuracy"],
            "student_test_top1_given_hit": untuned_st["top1_accuracy_given_hit"],
            "student_test_macro_f1": untuned_st["cand_macro_f1"],
            "student_test_log_loss": untuned_st["log_loss"],
            "student_test_route_continuity": untuned_st["network_reachable_continuity"],
            "train_time_s": untuned_fit_s,
            "student_test_infer_time_s": untuned_inf_s,
        },
        "tuned": {
            "params": best_params,
            **tuned_cv5,
            "search_3fold_cv_macro_f1": float(best_trial.value),
            "student_test_top1_acc": tuned_st["top1_accuracy"],
            "student_test_top1_given_hit": tuned_st["top1_accuracy_given_hit"],
            "student_test_macro_f1": tuned_st["cand_macro_f1"],
            "student_test_log_loss": tuned_st["log_loss"],
            "student_test_route_continuity": tuned_st["network_reachable_continuity"],
            "train_time_s": tuned_fit_s,
            "student_test_infer_time_s": tuned_inf_s,
        },
    }

    best_hp_path = cfg.artifacts_dir / "configurations" / "best_hyperparameters.json"
    best_hp_path.parent.mkdir(parents=True, exist_ok=True)
    best_hp_path.write_text(json.dumps(comparison, indent=2))

    (cfg.experiments_dir / "tuned_vs_untuned.json").write_text(json.dumps(comparison, indent=2))

    _plot_optuna_results(trials_df, comparison, cfg.reports_dir / "figures")

    print("\n--- Untuned vs. Bayesian-Tuned Summary ---")
    print(
        f"Untuned -> CV F1: {comparison['untuned']['cv_macro_f1_mean']:.4f} | "
        f"STest Top-1: {comparison['untuned']['student_test_top1_acc']:.4f} | "
        f"STest F1: {comparison['untuned']['student_test_macro_f1']:.4f} | "
        f"LogLoss: {comparison['untuned']['student_test_log_loss']:.4f}"
    )
    print(
        f"Tuned   -> CV F1: {comparison['tuned']['cv_macro_f1_mean']:.4f} | "
        f"STest Top-1: {comparison['tuned']['student_test_top1_acc']:.4f} | "
        f"STest F1: {comparison['tuned']['student_test_macro_f1']:.4f} | "
        f"LogLoss: {comparison['tuned']['student_test_log_loss']:.4f}"
    )
    print(f"Saved final model -> {model_path} (total {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main(n_trials=15)
