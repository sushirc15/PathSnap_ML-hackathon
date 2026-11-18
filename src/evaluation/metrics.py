"""Candidate-level and observation-level classification evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def select_top1_arcs_per_observation(
    obs_indices: np.ndarray,
    arc_ids: np.ndarray,
    scores: np.ndarray,
    n_obs: int,
) -> np.ndarray:
    """For each GPS observation t in [0, n_obs), pick the candidate arc with highest score."""
    pred_arcs = np.full(n_obs, -1, dtype=np.int32)
    best_scores = np.full(n_obs, -np.inf, dtype=np.float64)

    for obs_t, aid, sc in zip(obs_indices, arc_ids, scores):
        t = int(obs_t)
        s = float(sc)
        if s > best_scores[t]:
            best_scores[t] = s
            pred_arcs[t] = int(aid)
    return pred_arcs


def compute_candidate_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute candidate-level binary classification metrics (Macro F1, precision, recall, log loss, CM)."""
    y_true_i = np.asarray(y_true, dtype=np.int32)
    y_prob_f = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob_f >= threshold).astype(np.int32)

    cm = confusion_matrix(y_true_i, y_pred, labels=[0, 1])
    try:
        ll = float(log_loss(y_true_i, y_prob_f, labels=[0, 1]))
    except ValueError:
        ll = float("nan")

    return {
        "cand_accuracy": float(accuracy_score(y_true_i, y_pred)),
        "cand_macro_f1": float(f1_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "cand_pos_f1": float(f1_score(y_true_i, y_pred, pos_label=1, zero_division=0)),
        "cand_precision": float(precision_score(y_true_i, y_pred, pos_label=1, zero_division=0)),
        "cand_recall": float(recall_score(y_true_i, y_pred, pos_label=1, zero_division=0)),
        "cand_macro_precision": float(precision_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "cand_macro_recall": float(recall_score(y_true_i, y_pred, average="macro", zero_division=0)),
        "log_loss": ll,
        "confusion_matrix": cm.tolist(),
    }


def compute_observation_top1_metrics(
    aligned_ref_arcs: np.ndarray,
    predicted_arcs: np.ndarray,
    candidate_hit: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute observation-level Top-1 accuracy (overall and among candidate-hit observations)."""
    ref = np.asarray(aligned_ref_arcs, dtype=np.int32)
    pred = np.asarray(predicted_arcs, dtype=np.int32)
    n = len(ref)
    if n == 0:
        return {
            "top1_accuracy": 0.0,
            "top1_accuracy_given_hit": 0.0,
            "candidate_miss_rate": 0.0,
        }

    correct = (pred == ref) & (ref >= 0)
    top1_acc = float(np.mean(correct))

    if candidate_hit is not None:
        hit_mask = np.asarray(candidate_hit, dtype=bool)
        miss_rate = float(1.0 - np.mean(hit_mask))
        top1_given_hit = float(np.mean(correct[hit_mask])) if np.any(hit_mask) else 0.0
    else:
        miss_rate = float("nan")
        top1_given_hit = float("nan")

    return {
        "top1_accuracy": top1_acc,
        "top1_accuracy_given_hit": top1_given_hit,
        "candidate_miss_rate": miss_rate,
    }
