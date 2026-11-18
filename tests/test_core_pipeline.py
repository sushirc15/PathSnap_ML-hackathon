"""Unit and integration tests for PathSnap data loading, projection, road graph, spatial index,
features, label alignment, and HMM/Viterbi decoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import get_config
from src.data.loader import list_trip_ids, load_trip_inputs, load_trip_route
from src.data.preprocessing import PreparedTrip, prepare_trip
from src.data.validation import validate_trip
from src.decoder.hmm import HMMConfig, PathSnapHMM
from src.decoder.viterbi import viterbi_decode
from src.evaluation.metrics import (
    compute_candidate_classification_metrics,
    compute_observation_top1_metrics,
)
from src.evaluation.route_metrics import compute_route_continuity, compute_sequence_route_metrics
from src.features.candidate_generation import CandidateConfig, generate_candidates
from src.features.feature_extraction import FEATURE_COLUMNS, extract_candidate_features
from src.features.label_alignment import align_gps_to_reference_route
from src.geo.geometry import angular_diff_deg, compute_bearings_deg, point_to_segments_projection
from src.geo.projection import TripProjector, utm_epsg_from_lonlat
from src.geo.road_graph import RoadNetwork
from src.geo.spatial_index import CandidateSet, RoadSpatialIndex


def _make_synthetic_trip(disconnected: bool = False, gap_s: float = 1.0) -> PreparedTrip:
    """Create a small deterministic metric road network & GPS track for unit tests."""
    # Simple east-west road along lat=49.85, lon from 8.600 to 8.606 (4 nodes, 3 directed arcs)
    # Plus an optional disconnected north parallel component (~67m north, nodes 4 -> 5)
    nodes = np.array(
        [
            [8.600, 49.850],
            [8.602, 49.850],
            [8.604, 49.850],
            [8.606, 49.850],
            [8.600, 49.8506],
            [8.602, 49.8506],
        ],
        dtype=np.float64,
    )
    if disconnected:
        arcs = np.array([[0, 1], [1, 2], [4, 5]], dtype=np.int32)
    else:
        arcs = np.array([[0, 1], [1, 2], [2, 3], [1, 0], [4, 5]], dtype=np.int32)

    # GPS observations moving east
    if disconnected:
        track = pd.DataFrame(
            {
                "lon": [8.6005, 8.6015, 8.6025, 8.6005, 8.6015],
                "lat": [49.85002, 49.85001, 49.85002, 49.85061, 49.85062],
                "timestamp": [1.0, 2.0, 3.0, 3.0 + gap_s, 4.0 + gap_s],
            }
        )
    else:
        track = pd.DataFrame(
            {
                "lon": [8.6005, 8.6015, 8.6025, 8.6035, 8.6045, 8.6055],
                "lat": [49.85002, 49.85001, 49.85003, 49.85001, 49.85002, 49.85001],
                "timestamp": [1.0, 2.0, 2.0 + gap_s, 3.0 + gap_s, 4.0 + gap_s, 5.0 + gap_s],
            }
        )

    from pathlib import Path
    from src.data.loader import TripInputs

    raw = TripInputs(
        trip_id="synthetic_01",
        partition="test",
        trip_dir=Path("."),
        track=track,
        nodes=nodes,
        arcs=arcs,
    )
    return prepare_trip(raw)


def test_utm_projection_hemispheres_and_metric_distance() -> None:
    assert utm_epsg_from_lonlat(8.6, 49.85) == 32632  # Germany (Northern Hemisphere)
    assert utm_epsg_from_lonlat(-58.38, -34.60) == 32721  # Buenos Aires (Southern Hemisphere)
    proj = TripProjector.from_lonlat(np.array([8.6, 8.601]), np.array([49.85, 49.85]))
    x, y = proj.to_xy(np.array([8.6, 8.6]), np.array([49.85, 49.851]))
    # 0.001 deg latitude is ~111.2 meters
    dist = float(np.hypot(x[1] - x[0], y[1] - y[0]))
    assert 110.0 < dist < 112.5
    lon_back, lat_back = proj.to_lonlat(x, y)
    assert np.allclose(lat_back, [49.85, 49.851], atol=1e-7)


def test_geometry_bearings_and_projections() -> None:
    b = compute_bearings_deg(np.array([0.0, 1.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0, 0.0]))
    assert np.allclose(b, [0.0, 90.0, 180.0, 270.0])
    assert float(angular_diff_deg(355.0, 5.0)) == pytest.approx(10.0)
    assert float(angular_diff_deg(10.0, 190.0)) == pytest.approx(180.0)

    dist, frac, raw_frac, px, py = point_to_segments_projection(
        5.0, 3.0, np.array([0.0]), np.array([0.0]), np.array([10.0]), np.array([0.0])
    )
    assert float(dist[0]) == pytest.approx(3.0)
    assert float(frac[0]) == pytest.approx(0.5)
    assert float(px[0]) == pytest.approx(5.0)
    assert float(py[0]) == pytest.approx(0.0)


def test_directed_road_graph_and_shortest_path() -> None:
    prep = _make_synthetic_trip(disconnected=False)
    net = prep.network
    assert net.num_nodes == 6
    assert net.num_arcs == 5
    # Arc 0 is 0->1 and Arc 3 is 1->0 => both have reverse arcs
    assert net.has_reverse_arc[0] == 1
    assert net.has_reverse_arc[3] == 1
    # Arc 1 (1->2) is one-way
    assert net.has_reverse_arc[1] == 0
    assert net.is_directly_connected(0, 1)
    assert not net.is_directly_connected(0, 2)

    # Shortest arc path from arc 0 (0->1) to arc 2 (2->3) must bridge via arc 1 (1->2)
    bridge = net.shortest_arc_path(0, 2, max_dist_m=1000.0)
    assert bridge == [1]
    # Unreachable component (arc 4 is on nodes 4->5)
    assert net.shortest_arc_path(0, 4, max_dist_m=2000.0) is None


def test_spatial_candidates_features_and_label_alignment() -> None:
    prep = _make_synthetic_trip(disconnected=False)
    cands = generate_candidates(prep, CandidateConfig(radius_m=100.0, top_k=5))
    assert len(cands) == len(prep.gps_x)
    assert all(len(c.arc_ids) >= 1 for c in cands)

    X, obs_idx, arc_ids = extract_candidate_features(prep, cands)
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert np.all(np.isfinite(X))
    assert len(obs_idx) == len(X) == len(arc_ids)

    # Monotonic alignment along route [0, 1, 2]
    route = np.array([0, 1, 2], dtype=np.int32)
    align = align_gps_to_reference_route(prep, route, cands)
    assert np.all(np.diff(align.aligned_route_pos) >= 0)  # strictly monotonic
    assert align.aligned_arc_ids[0] == 0
    assert align.aligned_arc_ids[-1] == 2
    assert align.candidate_miss_rate == pytest.approx(0.0)


def test_viterbi_decoding_happy_path_disconnected_and_gps_gap() -> None:
    # 1. Happy path connected trajectory
    prep = _make_synthetic_trip(disconnected=False)
    cands = generate_candidates(prep, CandidateConfig(radius_m=100.0, top_k=5))
    # Give high emission prob to forward arcs (0, 1, 2) and lower to reverse arc 3
    obs_probs = [
        np.array([0.85 if a in (0, 1, 2) else 0.15 for a in c.arc_ids], dtype=np.float64)
        for c in cands
    ]
    res = viterbi_decode(prep, cands, obs_probs)
    assert res.obs_arc_ids[0] == 0
    assert res.obs_arc_ids[-1] == 2
    assert res.reconstructed_route_arcs == [0, 1, 2]
    assert len(res.segment_breaks) == 0

    # 2. Disconnected graph component jump (~67m north, no connecting edge) => triggers segment_breaks
    prep_disc = _make_synthetic_trip(disconnected=True, gap_s=1.0)
    cands_disc = generate_candidates(prep_disc, CandidateConfig(radius_m=30.0, fallback_radius_m=40.0, top_k=2))
    obs_probs_disc = [np.full(len(c.arc_ids), 0.7, dtype=np.float64) for c in cands_disc]
    res_disc = viterbi_decode(prep_disc, cands_disc, obs_probs_disc)
    assert len(res_disc.segment_breaks) >= 1
    assert res_disc.obs_arc_ids[-1] == 2  # arc index 2 is (4->5) in disconnected graph

    # 3. Large GPS gap (dt=60s) => must detect gap_break cleanly
    prep_gap = _make_synthetic_trip(disconnected=False, gap_s=60.0)
    cands_gap = generate_candidates(prep_gap, CandidateConfig(radius_m=100.0, top_k=4))
    obs_probs_gap = [
        np.array([0.9 if a in (0, 1, 2) else 0.1 for a in c.arc_ids], dtype=np.float64)
        for c in cands_gap
    ]
    res_gap = viterbi_decode(prep_gap, cands_gap, obs_probs_gap)
    assert len(res_gap.gap_breaks) >= 1
    assert res_gap.obs_arc_ids[0] == 0
    assert res_gap.obs_arc_ids[-1] == 2

    # 4. Empty candidate set at middle observation => must handle gracefully without crashing
    cands_empty = list(cands)
    empty_i = np.zeros(0, dtype=np.int32)
    empty_f = np.zeros(0, dtype=np.float64)
    cands_empty[2] = CandidateSet(2, empty_i, empty_f, empty_f, empty_f, empty_f, empty_f)
    obs_probs_empty = list(obs_probs)
    obs_probs_empty[2] = empty_f
    res_empty = viterbi_decode(prep, cands_empty, obs_probs_empty)
    assert res_empty.obs_arc_ids[2] == -1
    assert res_empty.obs_arc_ids[0] == 0
    assert res_empty.obs_arc_ids[-1] == 2


def test_real_dataset_trip_loading_and_validation() -> None:
    cfg = get_config()
    st_ids = list_trip_ids(cfg.student_test_dir)
    assert len(st_ids) == 10
    raw = load_trip_inputs(cfg.student_test_dir, st_ids[0], partition="student_test")
    route = load_trip_route(cfg.student_test_dir, st_ids[0])
    assert route is not None and len(route) > 0
    rep = validate_trip(raw, route=route, include_graph_stats=True)
    assert rep.is_valid
    assert rep.metrics["invalid_arc_refs"] == 0
    assert rep.metrics["invalid_gps_coords"] == 0
