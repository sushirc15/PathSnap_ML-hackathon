"""Model definitions for the six-algorithm PathSnap model comparison suite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


class SubsampledKNNClassifier(BaseEstimator, ClassifierMixin):
    """Distance-weighted KNN wrapper with deterministic stratified subsampling inside fit().

    Because full training has ~1.8M candidate rows across 36 features, exact KNN inference
    scales as O(N_train * N_query * d). Stratified subsampling inside each training fold
    keeps class balance exact, prevents data leakage, and enables practical benchmarking.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        weights: str = "distance",
        max_train_samples: int = 15000,
        random_state: int = 42,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.max_train_samples = max_train_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SubsampledKNNClassifier":
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        if n > self.max_train_samples:
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            pos_frac = len(pos_idx) / n
            n_pos = max(1, int(round(self.max_train_samples * pos_frac)))
            n_neg = max(1, self.max_train_samples - n_pos)
            sel_pos = rng.choice(pos_idx, size=min(n_pos, len(pos_idx)), replace=False)
            sel_neg = rng.choice(neg_idx, size=min(n_neg, len(neg_idx)), replace=False)
            idx = np.concatenate([sel_pos, sel_neg])
            rng.shuffle(idx)
            X_fit = X[idx]
            y_fit = y[idx]
        else:
            X_fit = X
            y_fit = y

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_fit)
        # In 36 dimensions, BLAS-accelerated brute-force search is substantially faster than KD/Ball trees
        self.knn_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm="brute",
            n_jobs=-1,
        )
        self.knn_.fit(X_scaled, y_fit)
        self.classes_ = self.knn_.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler_.transform(X)
        return self.knn_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(np.int32)


@dataclass(frozen=True)
class ModelSpec:
    """Metadata and factory for a candidate classifier."""

    key: str
    name: str
    family: str
    config_summary: str
    loss_objective: str
    factory: Callable[[int], BaseEstimator]


def get_model_specs() -> list[ModelSpec]:
    """Return the six diverse model families to benchmark on PathSnap."""
    return [
        ModelSpec(
            key="logistic_regression",
            name="Logistic Regression",
            family="Linear / Probabilistic",
            config_summary="StandardScaler + L2 C=1.0, solver=lbfgs, max_iter=500",
            loss_objective="L2-regularized binary cross-entropy (log loss)",
            factory=lambda seed: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=1.0,
                            solver="lbfgs",
                            max_iter=500,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            key="knn",
            name="Distance-Weighted KNN",
            family="Instance-Based",
            config_summary="StandardScaler + k=15, weights=distance, brute BLAS, stratified 15k fit subsample",
            loss_objective="Non-parametric inverse-distance neighbor voting",
            factory=lambda seed: SubsampledKNNClassifier(
                n_neighbors=15,
                weights="distance",
                max_train_samples=15000,
                random_state=seed,
            ),
        ),
        ModelSpec(
            key="decision_tree",
            name="Decision Tree Classifier",
            family="Tree-Based",
            config_summary="criterion=gini, max_depth=14, min_samples_leaf=50",
            loss_objective="Gini impurity minimization",
            factory=lambda seed: DecisionTreeClassifier(
                criterion="gini",
                max_depth=14,
                min_samples_leaf=50,
                random_state=seed,
            ),
        ),
        ModelSpec(
            key="random_forest",
            name="Random Forest Classifier",
            family="Bagging Ensemble",
            config_summary="n_estimators=120, max_depth=18, min_samples_leaf=20, max_features=sqrt",
            loss_objective="Bagged Gini impurity minimization across 120 trees",
            factory=lambda seed: RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=20,
                max_features="sqrt",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        ModelSpec(
            key="hist_gradient_boosting",
            name="HistGradientBoosting Classifier",
            family="Gradient Boosting",
            config_summary="max_iter=200, lr=0.08, max_leaf_nodes=63, min_samples_leaf=30, l2=1.0",
            loss_objective="Histogram-binned binary cross-entropy gradient boosting",
            factory=lambda seed: HistGradientBoostingClassifier(
                loss="log_loss",
                max_iter=200,
                learning_rate=0.08,
                max_leaf_nodes=63,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=seed,
            ),
        ),
        ModelSpec(
            key="mlp",
            name="MLP Neural Network",
            family="Neural Network",
            config_summary="StandardScaler + hidden=(64, 32), ReLU, Adam lr=1e-3, alpha=1e-3, batch=1024",
            loss_objective="L2-regularized binary cross-entropy via Adam optimization",
            factory=lambda seed: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            solver="adam",
                            alpha=1e-3,
                            batch_size=1024,
                            learning_rate_init=1e-3,
                            max_iter=25,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=6,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]
