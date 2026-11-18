"""Build and cache processed feature matrices and aligned labels for train and student_test."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_config
from src.features.candidate_generation import CandidateConfig
from src.models.dataset_cache import build_or_load_partition_cache


def main() -> None:
    t0 = time.perf_counter()
    cfg = get_config()
    cand_cfg = CandidateConfig(
        radius_m=cfg.candidate_radius_m,
        fallback_radius_m=cfg.candidate_fallback_radius_m,
        top_k=cfg.candidate_top_k,
    )

    print(f"Building feature caches with CandidateConfig={cand_cfg}")
    train_ds = build_or_load_partition_cache(cfg, "train", cfg.train_dir, cand_cfg, force_rebuild=True)
    stest_ds = build_or_load_partition_cache(cfg, "student_test", cfg.student_test_dir, cand_cfg, force_rebuild=True)

    # Save label alignment summary CSV across both partitions
    rows = []
    for ds in (train_ds, stest_ds):
        for tid in ds.trip_ids:
            m = ds.trip_meta[tid]
            rows.append(
                {
                    "partition": ds.partition,
                    "trip_id": tid,
                    "n_obs": m["n_obs"],
                    "n_cands": m["n_cands"],
                    "avg_cands_per_obs": m["n_cands"] / max(m["n_obs"], 1),
                    "candidate_miss_rate": m["candidate_miss_rate"],
                    "low_confidence_rate": m["low_confidence_rate"],
                    "median_aligned_dist_m": m["median_aligned_dist_m"],
                    "mean_aligned_dist_m": m["mean_aligned_dist_m"],
                }
            )
    align_df = pd.DataFrame(rows)
    out_csv = cfg.experiments_dir / "label_alignment_summary.csv"
    align_df.to_csv(out_csv, index=False)

    for part in ("train", "student_test"):
        sub = align_df[align_df["partition"] == part]
        total_obs = sub["n_obs"].sum()
        weighted_miss = (sub["candidate_miss_rate"] * sub["n_obs"]).sum() / total_obs
        weighted_lowc = (sub["low_confidence_rate"] * sub["n_obs"]).sum() / total_obs
        print(
            f"[{part}] trips={len(sub)}, obs={total_obs:,}, cands={sub['n_cands'].sum():,}, "
            f"weighted_miss_rate={weighted_miss:.4f}, low_conf_rate={weighted_lowc:.4f}, "
            f"median_of_median_dist={sub['median_aligned_dist_m'].median():.2f}m"
        )

    print(f"Saved alignment summary -> {out_csv} (total {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
