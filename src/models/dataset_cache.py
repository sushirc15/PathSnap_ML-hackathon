"""Build and cache candidate features and aligned reference labels for train & student_test."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import PathSnapConfig
from src.data.loader import list_trip_ids, load_trip_inputs, load_trip_route
from src.data.preprocessing import prepare_trip
from src.features.candidate_generation import CandidateConfig, generate_candidates
from src.features.feature_extraction import FEATURE_COLUMNS, extract_candidate_features
from src.features.label_alignment import align_gps_to_reference_route


@dataclass
class PartitionFeatureDataset:
    """Flattened candidate-level dataset for one partition with trip-group boundaries."""

    partition: str
    X: np.ndarray  # (N_cand, F) float32
    y: np.ndarray  # (N_cand,) int8 binary label (1 if candidate == aligned ref arc)
    obs_idx: np.ndarray  # (N_cand,) int32 observation index within trip
    arc_ids: np.ndarray  # (N_cand,) int32 candidate arc ID
    trip_group: np.ndarray  # (N_cand,) int32 integer trip index for GroupKFold
    trip_ids: List[str]
    trip_meta: Dict[str, Dict]  # per-trip offsets, aligned_arc_ids, candidate_hit, miss_rate


def build_or_load_partition_cache(
    cfg: PathSnapConfig,
    partition: str,
    partition_dir: Path,
    cand_cfg: CandidateConfig,
    force_rebuild: bool = False,
) -> PartitionFeatureDataset:
    """Build or load cached feature matrix and monotonic DP labels for a partition."""
    cache_prefix = f"{partition}_r{int(cand_cfg.radius_m)}_k{cand_cfg.top_k}"
    npz_path = cfg.processed_dir / f"{cache_prefix}.npz"
    meta_path = cfg.processed_dir / f"{cache_prefix}_meta.json"

    if not force_rebuild and npz_path.exists() and meta_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        meta = json.loads(meta_path.read_text())
        # Restore per-trip aligned arrays from npz
        trip_meta = meta["trip_meta"]
        for tid in meta["trip_ids"]:
            trip_meta[tid]["aligned_arc_ids"] = data[f"aligned_arc_{tid}"]
            trip_meta[tid]["candidate_hit"] = data[f"cand_hit_{tid}"].astype(bool)
            trip_meta[tid]["aligned_dist_m"] = data[f"aligned_dist_{tid}"]
        return PartitionFeatureDataset(
            partition=partition,
            X=data["X"],
            y=data["y"],
            obs_idx=data["obs_idx"],
            arc_ids=data["arc_ids"],
            trip_group=data["trip_group"],
            trip_ids=meta["trip_ids"],
            trip_meta=trip_meta,
        )

    t0 = time.perf_counter()
    trip_ids = list_trip_ids(partition_dir)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    obs_list: List[np.ndarray] = []
    arc_list: List[np.ndarray] = []
    grp_list: List[np.ndarray] = []
    trip_meta: Dict[str, Dict] = {}
    extra_arrays: Dict[str, np.ndarray] = {}

    row_cursor = 0
    for g_idx, tid in enumerate(trip_ids):
        raw = load_trip_inputs(partition_dir, tid, partition=partition)
        route = load_trip_route(partition_dir, tid)
        if route is None:
            raise ValueError(f"Partition {partition} trip {tid} is missing .route file")

        prep = prepare_trip(raw)
        cands = generate_candidates(prep, cand_cfg)
        align = align_gps_to_reference_route(prep, route, cands)
        X_t, obs_t, arcs_t = extract_candidate_features(prep, cands)
        y_t = (arcs_t == align.aligned_arc_ids[obs_t]).astype(np.int8)

        n_rows = len(X_t)
        X_list.append(X_t)
        y_list.append(y_t)
        obs_list.append(obs_t)
        arc_list.append(arcs_t)
        grp_list.append(np.full(n_rows, g_idx, dtype=np.int32))

        trip_meta[tid] = {
            "group_idx": g_idx,
            "row_start": row_cursor,
            "row_end": row_cursor + n_rows,
            "n_obs": int(len(prep.gps_x)),
            "n_cands": int(n_rows),
            "candidate_miss_rate": float(align.candidate_miss_rate),
            "low_confidence_rate": float(align.low_confidence_rate),
            "median_aligned_dist_m": float(np.median(align.aligned_dist_m)),
            "mean_aligned_dist_m": float(np.mean(align.aligned_dist_m)),
        }
        extra_arrays[f"aligned_arc_{tid}"] = align.aligned_arc_ids.astype(np.int32)
        extra_arrays[f"cand_hit_{tid}"] = align.candidate_hit.astype(np.int8)
        extra_arrays[f"aligned_dist_{tid}"] = align.aligned_dist_m.astype(np.float32)

        row_cursor += n_rows
        if (g_idx + 1) % 10 == 0 or (g_idx + 1) == len(trip_ids):
            print(
                f"  [{partition}] processed {g_idx + 1}/{len(trip_ids)} trips "
                f"({row_cursor:,} candidates, elapsed {time.perf_counter() - t0:.1f}s)"
            )

    X_all = np.vstack(X_list).astype(np.float32)
    y_all = np.concatenate(y_list).astype(np.int8)
    obs_all = np.concatenate(obs_list).astype(np.int32)
    arc_all = np.concatenate(arc_list).astype(np.int32)
    grp_all = np.concatenate(grp_list).astype(np.int32)

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X_all,
        y=y_all,
        obs_idx=obs_all,
        arc_ids=arc_all,
        trip_group=grp_all,
        **extra_arrays,
    )
    meta_out = {
        "partition": partition,
        "feature_columns": FEATURE_COLUMNS,
        "candidate_config": {
            "radius_m": cand_cfg.radius_m,
            "fallback_radius_m": cand_cfg.fallback_radius_m,
            "top_k": cand_cfg.top_k,
        },
        "trip_ids": trip_ids,
        "trip_meta": trip_meta,
    }
    meta_path.write_text(json.dumps(meta_out, indent=2))

    # Attach arrays in memory before returning
    for tid in trip_ids:
        trip_meta[tid]["aligned_arc_ids"] = extra_arrays[f"aligned_arc_{tid}"]
        trip_meta[tid]["candidate_hit"] = extra_arrays[f"cand_hit_{tid}"].astype(bool)
        trip_meta[tid]["aligned_dist_m"] = extra_arrays[f"aligned_dist_{tid}"]

    return PartitionFeatureDataset(
        partition=partition,
        X=X_all,
        y=y_all,
        obs_idx=obs_all,
        arc_ids=arc_all,
        trip_group=grp_all,
        trip_ids=trip_ids,
        trip_meta=trip_meta,
    )
