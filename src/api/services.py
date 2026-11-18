"""Server-side inference and isolated post-prediction evaluation services for PathSnap API.

STRICT GROUND-TRUTH SEPARATION:
- `run_prediction_only(...)` ONLY reads `.track`, `.nodes`, `.arcs`, and saved model/HMM artifacts.
- `evaluate_server_side(...)` runs ONLY AFTER prediction completes, reads `.route` exclusively
  on the server, and returns aggregate scalar metrics (never reference arc IDs or geometry).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from src.config import PathSnapConfig, get_config
from src.data.loader import (
    has_trip_route,
    list_trip_ids,
    load_trip_inputs,
    load_trip_route,
)
from src.data.preprocessing import PreparedTrip, prepare_trip
from src.data.validation import validate_trip
from src.decoder.hmm import HMMConfig, PathSnapHMM
from src.decoder.viterbi import ViterbiDecodeResult
from src.evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_metrics,
    select_top1_arcs_per_observation,
)
from src.evaluation.route_metrics import compute_route_continuity, compute_sequence_route_metrics
from src.features.candidate_generation import CandidateConfig, generate_candidates
from src.features.feature_extraction import FEATURE_COLUMNS, extract_candidate_features
from src.features.label_alignment import align_gps_to_reference_route
from src.geo.spatial_index import CandidateSet
from src.visualization.folium_maps import (
    _extract_local_road_segments,
    _merge_arc_polylines,
    build_trip_folium_map,
)

logger = logging.getLogger("pathsnap.api")


@dataclass
class PredictionBundle:
    """In-memory prediction output (strictly free of any reference-route data)."""

    trip_id: str
    partition: str
    is_development_fallback: bool
    prep: PreparedTrip
    candidates: List[CandidateSet]
    obs_idx: np.ndarray
    arc_ids: np.ndarray
    probs: np.ndarray
    clf_obs_arcs: np.ndarray
    decode_result: ViterbiDecodeResult
    timings_ms: Dict[str, float]
    continuity: Dict[str, Any]
    validation_warnings: List[str]


class PathSnapInferenceService:
    """Loads saved ML/HMM artifacts and serves isolated map-matching predictions."""

    def __init__(self, cfg: Optional[PathSnapConfig] = None) -> None:
        self.cfg = cfg or get_config()
        self.model = None
        self.feature_config: Dict[str, Any] = {}
        self.cand_config: CandidateConfig = CandidateConfig()
        self.hmm_config: HMMConfig = HMMConfig()
        self.hmm: PathSnapHMM = PathSnapHMM(self.hmm_config)
        self.best_hp_meta: Dict[str, Any] = {}
        self._map_html_cache: Dict[str, str] = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.cfg.artifacts_dir / "models" / "final_model.joblib"
        feat_path = self.cfg.artifacts_dir / "preprocessors" / "feature_config.json"
        hmm_path = self.cfg.artifacts_dir / "configurations" / "hmm_config.json"
        hp_path = self.cfg.artifacts_dir / "configurations" / "best_hyperparameters.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Trained model artifact not found at {model_path}")

        self.model = joblib.load(model_path)
        if feat_path.exists():
            self.feature_config = json.loads(feat_path.read_text())
            self.cand_config = CandidateConfig(**self.feature_config.get("candidate_config", {}))
        if hmm_path.exists():
            self.hmm_config = HMMConfig.from_dict(json.loads(hmm_path.read_text()))
        self.hmm = PathSnapHMM(self.hmm_config)
        if hp_path.exists():
            self.best_hp_meta = json.loads(hp_path.read_text())

        logger.info(
            "Loaded PathSnap artifacts: model=%s features=%d cand=%s hmm=%s",
            type(self.model).__name__,
            len(FEATURE_COLUMNS),
            self.cand_config,
            self.hmm_config,
        )

    def resolve_active_eval_source(self) -> Tuple[Path, str, bool]:
        """Return (active_dir, partition_label, is_development_fallback).

        Uses `evaluator_test_dir` automatically whenever trips are supplied there;
        otherwise falls back to `student_test_dir` and explicitly flags it as development-only.
        """
        eval_trips = list_trip_ids(self.cfg.evaluator_test_dir)
        if len(eval_trips) > 0:
            return self.cfg.evaluator_test_dir, "evaluator_test", False
        return self.cfg.student_test_dir, "student_test", True

    def list_available_trips(self) -> Dict[str, Any]:
        """List available trips from the active evaluation directory (never hardcoded)."""
        active_dir, partition, is_dev_fallback = self.resolve_active_eval_source()
        trip_ids = list_trip_ids(active_dir)

        trips_out = []
        for tid in trip_ids:
            raw = load_trip_inputs(active_dir, tid, partition=partition)
            ts = raw.track["timestamp"].to_numpy(dtype=np.float64)
            duration_s = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
            trips_out.append(
                {
                    "trip_id": tid,
                    "partition": partition,
                    "is_development_only": is_dev_fallback,
                    "n_gps_observations": int(len(raw.track)),
                    "n_road_nodes": int(len(raw.nodes)),
                    "n_road_arcs": int(len(raw.arcs)),
                    "duration_s": round(duration_s, 1),
                    "center_lat": round(float(raw.track["lat"].median()), 5),
                    "center_lon": round(float(raw.track["lon"].median()), 5),
                    "has_server_side_reference": has_trip_route(active_dir, tid),
                }
            )

        return {
            "active_partition": partition,
            "is_development_fallback": is_dev_fallback,
            "partition_notice": (
                "Reserved evaluator trips loaded from evaluator_test directory."
                if not is_dev_fallback
                else "Development-only student_test partition active (no reserved evaluator trips present in data/evaluator_test)."
            ),
            "trip_count": len(trips_out),
            "trips": trips_out,
        }

    def get_health_info(self) -> Dict[str, Any]:
        """Return backend health, model metadata, and active dataset status."""
        active_dir, partition, is_dev_fallback = self.resolve_active_eval_source()
        trip_ids = list_trip_ids(active_dir)
        return {
            "status": "ok",
            "service": "PathSnap Distributed Map-Matching API",
            "model_name": self.feature_config.get("model_type", type(self.model).__name__),
            "model_trained_on": self.feature_config.get("trained_partition", "train (80 trips)"),
            "n_features": len(FEATURE_COLUMNS),
            "hmm_enabled": True,
            "hmm_config": self.hmm_config.to_dict(),
            "candidate_config": {
                "radius_m": self.cand_config.radius_m,
                "fallback_radius_m": self.cand_config.fallback_radius_m,
                "top_k": self.cand_config.top_k,
            },
            "active_partition": partition,
            "is_development_fallback": is_dev_fallback,
            "available_trip_count": len(trip_ids),
        }

    def run_prediction_only(self, trip_id: str) -> PredictionBundle:
        """Execute strictly ground-truth-free prediction pipeline for one trip."""
        t_total = time.perf_counter()
        active_dir, partition, is_dev_fallback = self.resolve_active_eval_source()
        available = list_trip_ids(active_dir)
        if trip_id not in available:
            raise KeyError(f"Trip '{trip_id}' not found in active partition '{partition}'")

        # 1. Load ONLY .track, .nodes, .arcs
        t0 = time.perf_counter()
        raw = load_trip_inputs(active_dir, trip_id, partition=partition)
        val_report = validate_trip(raw, route=None, include_graph_stats=False)
        if not val_report.is_valid:
            raise ValueError(f"Trip '{trip_id}' failed validation: {'; '.join(val_report.errors)}")
        prep = prepare_trip(raw)
        t_prep_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Spatial candidate generation + 36-feature extraction
        t1 = time.perf_counter()
        cands = generate_candidates(prep, self.cand_config)
        X, obs_idx, arc_ids = extract_candidate_features(prep, cands)
        t_feat_ms = (time.perf_counter() - t1) * 1000.0

        # 3. Trained ML classifier scoring
        t2 = time.perf_counter()
        if len(X) > 0:
            probs = self.model.predict_proba(X)[:, 1]
        else:
            probs = np.zeros(0, dtype=np.float64)
        clf_obs_arcs = select_top1_arcs_per_observation(obs_idx, arc_ids, probs, len(prep.gps_x))
        t_clf_ms = (time.perf_counter() - t2) * 1000.0

        # 4. HMM / Viterbi route reconstruction
        t3 = time.perf_counter()
        dec = self.hmm.decode_from_flat_probs(prep, cands, obs_idx, probs)
        t_hmm_ms = (time.perf_counter() - t3) * 1000.0

        # 5. Topological continuity (ground-truth free)
        cont = compute_route_continuity(prep.network, dec.obs_arc_ids)
        recon_arcs = dec.reconstructed_route_arcs
        if len(recon_arcs) > 1:
            recon_conn = sum(
                1
                for i in range(len(recon_arcs) - 1)
                if prep.network.is_directly_connected(recon_arcs[i], recon_arcs[i + 1])
            )
            recon_direct_cont = float(recon_conn / (len(recon_arcs) - 1))
        else:
            recon_direct_cont = 1.0
        cont["reconstructed_route_direct_continuity"] = recon_direct_cont

        total_ms = (time.perf_counter() - t_total) * 1000.0
        timings = {
            "preprocess_ms": round(t_prep_ms, 2),
            "candidate_and_features_ms": round(t_feat_ms, 2),
            "classifier_ms": round(t_clf_ms, 2),
            "hmm_viterbi_ms": round(t_hmm_ms, 2),
            "total_ms": round(total_ms, 2),
        }

        # Cache evaluator-safe Folium map HTML (NO reference_arcs passed!)
        fmap = build_trip_folium_map(
            prep,
            predicted_arcs=dec.reconstructed_route_arcs,
            classifier_only_arcs=None,
            reference_arcs=None,
            title=f"PathSnap Prediction — Trip {trip_id} ({partition})",
            show_road_network=True,
        )
        self._map_html_cache[trip_id] = fmap.get_root().render()

        return PredictionBundle(
            trip_id=trip_id,
            partition=partition,
            is_development_fallback=is_dev_fallback,
            prep=prep,
            candidates=cands,
            obs_idx=obs_idx,
            arc_ids=arc_ids,
            probs=probs,
            clf_obs_arcs=clf_obs_arcs,
            decode_result=dec,
            timings_ms=timings,
            continuity=cont,
            validation_warnings=val_report.warnings,
        )

    def evaluate_server_side(self, bundle: PredictionBundle) -> Dict[str, Any]:
        """Compute aggregate server-side metrics ONLY AFTER prediction finishes.

        Never returns reference arc IDs or reference geometry.
        Distinguishes unavailable metrics (`None`) cleanly from numeric `0.0`.
        """
        active_dir, _, _ = self.resolve_active_eval_source()
        route = load_trip_route(active_dir, bundle.trip_id)
        if route is None or len(route) == 0:
            return {
                "ground_truth_available": False,
                "top1_accuracy": None,
                "top1_accuracy_given_hit": None,
                "classifier_only_top1_accuracy": None,
                "candidate_miss_rate": None,
                "macro_f1": None,
                "log_loss": None,
                "seq_arc_f1": None,
                "seq_length_f1": None,
            }

        align = align_gps_to_reference_route(bundle.prep, route, bundle.candidates)
        y_true = (bundle.arc_ids == align.aligned_arc_ids[bundle.obs_idx]).astype(np.int32)

        cand_m = compute_candidate_classification_metrics(y_true, bundle.probs)
        hmm_top1 = compute_observation_top1_metrics(
            align.aligned_arc_ids, bundle.decode_result.obs_arc_ids, align.candidate_hit
        )
        clf_top1 = compute_observation_top1_metrics(
            align.aligned_arc_ids, bundle.clf_obs_arcs, align.candidate_hit
        )
        seq_m = compute_sequence_route_metrics(
            bundle.prep.network, bundle.decode_result.reconstructed_route_arcs, route
        )

        return {
            "ground_truth_available": True,
            "top1_accuracy": round(hmm_top1["top1_accuracy"], 4),
            "top1_accuracy_given_hit": round(hmm_top1["top1_accuracy_given_hit"], 4),
            "classifier_only_top1_accuracy": round(clf_top1["top1_accuracy"], 4),
            "candidate_miss_rate": round(hmm_top1["candidate_miss_rate"], 4),
            "macro_f1": round(cand_m["cand_macro_f1"], 4),
            "log_loss": round(cand_m["log_loss"], 4) if np.isfinite(cand_m["log_loss"]) else None,
            "seq_arc_f1": round(seq_m["seq_arc_f1"], 4),
            "seq_length_f1": round(seq_m["seq_length_f1"], 4),
        }

    def predict_and_evaluate(self, trip_id: str) -> Dict[str, Any]:
        """Run isolated prediction followed by separate server-side aggregate evaluation."""
        bundle = self.run_prediction_only(trip_id)
        eval_metrics = self.evaluate_server_side(bundle)

        prep = bundle.prep
        dec = bundle.decode_result

        # Subsample GPS points for lightweight frontend Leaflet rendering (keep start & end exact)
        n_obs = len(prep.lat)
        step = max(1, n_obs // 900)
        gps_indices = list(range(0, n_obs, step))
        if gps_indices[-1] != n_obs - 1:
            gps_indices.append(n_obs - 1)
        gps_points = [
            {
                "idx": int(i),
                "lat": round(float(prep.lat[i]), 6),
                "lon": round(float(prep.lon[i]), 6),
                "t": round(float(prep.timestamp[i]), 1),
            }
            for i in gps_indices
        ]

        predicted_polylines = _merge_arc_polylines(prep, dec.reconstructed_route_arcs)
        local_roads = _extract_local_road_segments(prep, buffer_m=160.0, max_arcs=3500)

        logger.info(
            "PREDICT trip=%s partition=%s obs=%d pred_arcs=%d hmm_top1=%s cont=%.4f total_ms=%.1f",
            trip_id,
            bundle.partition,
            n_obs,
            len(dec.reconstructed_route_arcs),
            eval_metrics.get("top1_accuracy"),
            bundle.continuity["reconstructed_route_direct_continuity"],
            bundle.timings_ms["total_ms"],
        )

        return {
            "trip_id": bundle.trip_id,
            "partition": bundle.partition,
            "is_development_fallback": bundle.is_development_fallback,
            "model_name": self.feature_config.get("model_type", type(self.model).__name__),
            "hmm_status": {
                "enabled": True,
                "algorithm": "Log-Space Viterbi with Directed Shortest-Path Transitions",
                "bridged_intermediate_arcs": dec.n_bridged_arcs,
                "segment_breaks": len(dec.segment_breaks),
                "gps_gap_breaks": len(dec.gap_breaks),
            },
            "prediction": {
                "observation_count": n_obs,
                "matched_observation_count": int(np.sum(dec.obs_arc_ids >= 0)),
                "predicted_route_arc_count": len(dec.reconstructed_route_arcs),
                "predicted_route_arc_ids": [int(a) for a in dec.reconstructed_route_arcs],
                "utm_crs": prep.projector.crs_name,
            },
            "continuity": {
                "direct_step_continuity": round(bundle.continuity["direct_step_continuity"], 4),
                "network_reachable_continuity": round(bundle.continuity["network_reachable_continuity"], 4),
                "transition_reachable_continuity": round(bundle.continuity["transition_reachable_continuity"], 4),
                "reconstructed_route_direct_continuity": round(
                    bundle.continuity["reconstructed_route_direct_continuity"], 4
                ),
                "disconnected_transitions": bundle.continuity["disconnected_transitions"],
            },
            "evaluation": eval_metrics,
            "timings_ms": bundle.timings_ms,
            "validation_warnings": bundle.validation_warnings,
            "map_url": f"/api/trips/{trip_id}/map",
            "geometry": {
                "bounds": [
                    [float(np.min(prep.lat)), float(np.min(prep.lon))],
                    [float(np.max(prep.lat)), float(np.max(prep.lon))],
                ],
                "start": [float(prep.lat[0]), float(prep.lon[0])],
                "end": [float(prep.lat[-1]), float(prep.lon[-1])],
                "gps_points": gps_points,
                "predicted_polylines": predicted_polylines,
                "local_road_segments": local_roads,
            },
        }

    def get_map_html(self, trip_id: str) -> str:
        """Return cached evaluator-safe Folium HTML map (generating on demand if needed)."""
        if trip_id not in self._map_html_cache:
            self.run_prediction_only(trip_id)
        return self._map_html_cache[trip_id]
