"""Dataset loading utilities for PathSnap trips.

Strict separation rule:
- Prediction-time code calls `load_trip_inputs(...)` which NEVER reads `.route`.
- Training and server-side evaluation explicitly call `load_trip_route(...)` only where permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TripInputs:
    """Prediction-time inputs for a single trip (no ground-truth route)."""

    trip_id: str
    partition: str
    trip_dir: Path
    track: pd.DataFrame  # columns: lon, lat, timestamp
    nodes: np.ndarray  # shape (N, 2) float64: [lon, lat]
    arcs: np.ndarray  # shape (M, 2) int32: [from_node_id, to_node_id]


def list_trip_ids(partition_dir: Path) -> List[str]:
    """Return sorted list of valid trip IDs inside a partition directory."""
    if not partition_dir.exists() or not partition_dir.is_dir():
        return []
    trip_ids: List[str] = []
    for entry in sorted(partition_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        tid = entry.name
        if (
            (entry / f"{tid}.track").exists()
            and (entry / f"{tid}.nodes").exists()
            and (entry / f"{tid}.arcs").exists()
        ):
            trip_ids.append(tid)
    return trip_ids


def _resolve_trip_file(trip_dir: Path, trip_id: str, suffix: str) -> Path:
    """Resolve `<trip_id>.<suffix>` or fallback to single `*.<suffix>` inside trip_dir."""
    exact = trip_dir / f"{trip_id}.{suffix}"
    if exact.exists():
        return exact
    matches = sorted(trip_dir.glob(f"*.{suffix}"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing .{suffix} file for trip {trip_id} in {trip_dir}")


def load_trip_inputs(
    partition_dir: Path, trip_id: str, partition: str = "unknown"
) -> TripInputs:
    """Load `.track`, `.nodes`, and `.arcs` for a trip. Never touches `.route`."""
    trip_dir = partition_dir / trip_id
    if not trip_dir.exists():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track_path = _resolve_trip_file(trip_dir, trip_id, "track")
    nodes_path = _resolve_trip_file(trip_dir, trip_id, "nodes")
    arcs_path = _resolve_trip_file(trip_dir, trip_id, "arcs")

    # Fast C-engine loading of whitespace-delimited files without headers
    track = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64},
        engine="c",
    )

    nodes_df = pd.read_csv(
        nodes_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat"],
        dtype=np.float64,
        engine="c",
    )
    nodes = nodes_df.to_numpy(dtype=np.float64, copy=True)

    arcs_df = pd.read_csv(
        arcs_path,
        sep=r"\s+",
        header=None,
        names=["from_node", "to_node"],
        dtype=np.int32,
        engine="c",
    )
    arcs = arcs_df.to_numpy(dtype=np.int32, copy=True)

    return TripInputs(
        trip_id=trip_id,
        partition=partition,
        trip_dir=trip_dir,
        track=track,
        nodes=nodes,
        arcs=arcs,
    )


def has_trip_route(partition_dir: Path, trip_id: str) -> bool:
    """Check whether a `.route` file exists for a trip."""
    trip_dir = partition_dir / trip_id
    if not trip_dir.exists():
        return False
    return (trip_dir / f"{trip_id}.route").exists() or len(list(trip_dir.glob("*.route"))) == 1


def load_trip_route(partition_dir: Path, trip_id: str) -> Optional[np.ndarray]:
    """Load the chronological `.route` arc ID array if available.

    Must ONLY be called during training label alignment, offline evaluation,
    or server-side post-prediction evaluation.
    """
    trip_dir = partition_dir / trip_id
    try:
        route_path = _resolve_trip_file(trip_dir, trip_id, "route")
    except FileNotFoundError:
        return None
    route_df = pd.read_csv(
        route_path,
        sep=r"\s+",
        header=None,
        names=["arc_id"],
        dtype=np.int32,
        engine="c",
    )
    return route_df["arc_id"].to_numpy(dtype=np.int32, copy=True)


def get_trip_file_sizes(partition_dir: Path, trip_id: str) -> Dict[str, int]:
    """Return file sizes in bytes for each trip file."""
    trip_dir = partition_dir / trip_id
    sizes: Dict[str, int] = {}
    for ext in ("track", "nodes", "arcs", "route"):
        try:
            p = _resolve_trip_file(trip_dir, trip_id, ext)
            sizes[f"{ext}_bytes"] = p.stat().st_size
        except FileNotFoundError:
            sizes[f"{ext}_bytes"] = 0
    return sizes
