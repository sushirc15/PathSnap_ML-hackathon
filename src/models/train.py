"""Reproducible end-to-end training orchestrator for PathSnap.

Runs:
1. Feature cache construction (80 train + 10 student_test trips, strictly monotonic DP labels)
2. 6-model GroupKFold CV & student-test comparison (`src/models/compare.py`)
3. Optuna TPE Bayesian hyperparameter optimization (`src/models/optimize.py`)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.candidate_generation import CandidateConfig
from src.models.compare import main as run_compare
from src.models.dataset_cache import build_or_load_partition_cache
from src.models.optimize import main as run_optimize


def main() -> None:
    parser = argparse.ArgumentParser(description="PathSnap end-to-end model training pipeline")
    parser.add_argument("--rebuild-cache", action="store_true", help="Force rebuild feature caches")
    parser.add_argument("--trials", type=int, default=25, help="Number of Optuna TPE Bayesian trials")
    args = parser.parse_args()

    cfg = get_config()
    cand_cfg = CandidateConfig(
        radius_m=cfg.candidate_radius_m,
        fallback_radius_m=cfg.candidate_fallback_radius_m,
        top_k=cfg.candidate_top_k,
    )

    print("Step 1/3: Building/loading processed feature caches...")
    build_or_load_partition_cache(cfg, "train", cfg.train_dir, cand_cfg, force_rebuild=args.rebuild_cache)
    build_or_load_partition_cache(cfg, "student_test", cfg.student_test_dir, cand_cfg, force_rebuild=args.rebuild_cache)

    print("\nStep 2/3: Comparing six distinct ML model families...")
    run_compare()

    print(f"\nStep 3/3: Running Optuna TPE Bayesian optimization ({args.trials} trials)...")
    run_optimize(n_trials=args.trials)


if __name__ == "__main__":
    main()
