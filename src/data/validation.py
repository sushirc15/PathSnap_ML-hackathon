"""Validation and audit helpers for PathSnap trips."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.loader import TripInputs, get_trip_file_sizes, load_trip_route
from src.geo.projection import TripProjector
from src.geo.road_graph import RoadNetwork


@dataclass
class ValidationReport:
    """Validation findings for a single trip."""

    trip_id: str
    partition: str
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def validate_trip(
    trip: TripInputs,
    route: Optional[np.ndarray] = None,
    include_graph_stats: bool = True,
) -> ValidationReport:
    """Audit coordinates, timestamps, node/arc indices, graph connectivity, and optional route."""
    warnings: List[str] = []
    errors: List[str] = []

    track = trip.track
    nodes = trip.nodes
    arcs = trip.arcs

    n_gps = len(track)
    n_nodes = len(nodes)
    n_arcs = len(arcs)

    # Missing values
    track_na = int(track.isna().sum().sum())
    nodes_na = int(np.isnan(nodes).sum())
    arcs_na = int(np.isnan(arcs).sum()) if np.issubdtype(arcs.dtype, np.floating) else 0

    if track_na > 0:
        errors.append(f"Track has {track_na} NaN values")
    if nodes_na > 0:
        errors.append(f"Nodes has {nodes_na} NaN values")
    if n_gps == 0:
        errors.append("Empty GPS track")
    if n_nodes == 0 or n_arcs == 0:
        errors.append("Empty road network")

    # Coordinate range checks
    lon = track["lon"].to_numpy(dtype=np.float64)
    lat = track["lat"].to_numpy(dtype=np.float64)
    ts = track["timestamp"].to_numpy(dtype=np.float64)

    invalid_gps_coords = int(
        np.sum((lon < -180.0) | (lon > 180.0) | (lat < -90.0) | (lat > 90.0) | np.isnan(lon) | np.isnan(lat))
    )
    invalid_node_coords = int(
        np.sum(
            (nodes[:, 0] < -180.0)
            | (nodes[:, 0] > 180.0)
            | (nodes[:, 1] < -90.0)
            | (nodes[:, 1] > 90.0)
            | np.isnan(nodes[:, 0])
            | np.isnan(nodes[:, 1])
        )
    )
    if invalid_gps_coords > 0:
        errors.append(f"Invalid GPS coordinates: {invalid_gps_coords}")
    if invalid_node_coords > 0:
        errors.append(f"Invalid node coordinates: {invalid_node_coords}")

    # Node ID bounds in arcs
    invalid_arc_refs = int(
        np.sum(
            (arcs[:, 0] < 0)
            | (arcs[:, 0] >= n_nodes)
            | (arcs[:, 1] < 0)
            | (arcs[:, 1] >= n_nodes)
        )
    )
    if invalid_arc_refs > 0:
        errors.append(f"Out-of-range node references in arcs: {invalid_arc_refs}")

    # Self-loops and duplicate directed arcs
    self_loops = int(np.sum(arcs[:, 0] == arcs[:, 1]))
    if self_loops > 0:
        warnings.append(f"Road network contains {self_loops} self-loop arcs")

    # Timestamps
    if n_gps > 1:
        dt = np.diff(ts)
        non_monotonic_ts = int(np.sum(dt < 0))
        duplicate_ts = int(np.sum(dt == 0))
        exact_1s = int(np.sum(np.isclose(dt, 1.0, atol=1e-3)))
        gt_5s_gaps = int(np.sum(dt > 5.0))
        gt_30s_gaps = int(np.sum(dt > 30.0))
        min_dt = float(np.min(dt))
        median_dt = float(np.median(dt))
        max_dt = float(np.max(dt))
        duration_s = float(ts[-1] - ts[0])
    else:
        non_monotonic_ts = duplicate_ts = exact_1s = gt_5s_gaps = gt_30s_gaps = 0
        min_dt = median_dt = max_dt = duration_s = 0.0

    if non_monotonic_ts > 0:
        warnings.append(f"Non-monotonic timestamps detected: {non_monotonic_ts}")
    if duplicate_ts > 0:
        warnings.append(f"Duplicate timestamps detected: {duplicate_ts}")
    if gt_30s_gaps > 0:
        warnings.append(f"Large GPS time gaps (>30s): {gt_30s_gaps} (max={max_dt:.1f}s)")

    # Geographic extent & projected displacement
    projector = TripProjector.from_lonlat(lon, lat)
    gx, gy = projector.to_xy(lon, lat)
    if n_gps > 1:
        step_dist = np.hypot(np.diff(gx), np.diff(gy))
        total_gps_dist_m = float(np.sum(step_dist))
        max_gps_jump_m = float(np.max(step_dist))
        gt_100m_jumps = int(np.sum(step_dist > 100.0))
        median_step_m = float(np.median(step_dist))
    else:
        total_gps_dist_m = max_gps_jump_m = median_step_m = 0.0
        gt_100m_jumps = 0

    metrics: Dict[str, Any] = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_gps": n_gps,
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "track_na": track_na,
        "nodes_na": nodes_na,
        "arcs_na": arcs_na,
        "invalid_gps_coords": invalid_gps_coords,
        "invalid_node_coords": invalid_node_coords,
        "invalid_arc_refs": invalid_arc_refs,
        "self_loops": self_loops,
        "non_monotonic_ts": non_monotonic_ts,
        "duplicate_ts": duplicate_ts,
        "pct_1s_steps": (100.0 * exact_1s / max(n_gps - 1, 1)),
        "min_dt_s": min_dt,
        "median_dt_s": median_dt,
        "max_dt_s": max_dt,
        "duration_s": duration_s,
        "gt_5s_gaps": gt_5s_gaps,
        "gt_30s_gaps": gt_30s_gaps,
        "lon_min": float(np.nanmin(lon)),
        "lon_max": float(np.nanmax(lon)),
        "lat_min": float(np.nanmin(lat)),
        "lat_max": float(np.nanmax(lat)),
        "utm_epsg": projector.epsg_code,
        "total_gps_dist_m": total_gps_dist_m,
        "median_step_m": median_step_m,
        "max_gps_jump_m": max_gps_jump_m,
        "gt_100m_jumps": gt_100m_jumps,
    }

    # File sizes
    metrics.update(get_trip_file_sizes(trip.trip_dir.parent, trip.trip_id))

    # Graph connectivity
    if include_graph_stats and n_nodes > 0 and n_arcs > 0 and invalid_arc_refs == 0:
        net = RoadNetwork(nodes, arcs, projector)
        metrics.update(net.graph_connectivity_stats())
        metrics["mean_arc_len_m"] = float(np.mean(net.arc_lengths))
        metrics["median_arc_len_m"] = float(np.median(net.arc_lengths))
        metrics["max_arc_len_m"] = float(np.max(net.arc_lengths))
        metrics["pct_two_way_arcs"] = float(100.0 * np.mean(net.has_reverse_arc))

    # Reference route audit (only when explicitly supplied)
    if route is not None:
        n_route = len(route)
        invalid_route_arcs = int(np.sum((route < 0) | (route >= n_arcs)))
        if invalid_route_arcs > 0:
            errors.append(f"Route contains {invalid_route_arcs} out-of-range arc IDs")
        # Check direct topological continuity of reference route
        if n_route > 1 and invalid_route_arcs == 0:
            from_next = arcs[route[1:], 0]
            to_prev = arcs[route[:-1], 1]
            same_arc = route[1:] == route[:-1]
            connected_steps = int(np.sum((from_next == to_prev) | same_arc))
            route_continuity = float(connected_steps / (n_route - 1))
            route_breaks = int((n_route - 1) - connected_steps)
            unique_route_arcs = int(len(np.unique(route)))
            has_repeated_arcs = int(unique_route_arcs < n_route)
        else:
            route_continuity = 1.0
            route_breaks = 0
            unique_route_arcs = n_route
            has_repeated_arcs = 0

        metrics.update(
            {
                "has_route": True,
                "n_route_arcs": n_route,
                "unique_route_arcs": unique_route_arcs,
                "has_repeated_route_arcs": has_repeated_arcs,
                "invalid_route_arcs": invalid_route_arcs,
                "route_direct_continuity": route_continuity,
                "route_breaks": route_breaks,
            }
        )
    else:
        metrics.update(
            {
                "has_route": False,
                "n_route_arcs": 0,
                "unique_route_arcs": 0,
                "has_repeated_route_arcs": 0,
                "invalid_route_arcs": 0,
                "route_direct_continuity": np.nan,
                "route_breaks": 0,
            }
        )

    return ValidationReport(
        trip_id=trip.trip_id,
        partition=trip.partition,
        is_valid=(len(errors) == 0),
        warnings=warnings,
        errors=errors,
        metrics=metrics,
    )
