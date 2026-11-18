"""Trip preprocessing: coordinate projection, GPS kinematic derivation, and road network setup.

Strictly uses ONLY prediction-time inputs (`.track`, `.nodes`, `.arcs`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.loader import TripInputs
from src.geo.geometry import compute_bearings_deg
from src.geo.projection import TripProjector
from src.geo.road_graph import RoadNetwork
from src.geo.spatial_index import RoadSpatialIndex


@dataclass
class PreparedTrip:
    """Projected GPS kinematics, directed road network, and spatial index for one trip."""

    trip_id: str
    partition: str
    projector: TripProjector
    network: RoadNetwork
    spatial_index: RoadSpatialIndex

    # Cleaned/ordered GPS arrays (length T)
    lon: np.ndarray
    lat: np.ndarray
    timestamp: np.ndarray
    gps_x: np.ndarray
    gps_y: np.ndarray

    # Kinematics derived purely from GPS trajectory
    dt_prev: np.ndarray  # seconds since previous observation
    dt_next: np.ndarray  # seconds to next observation
    disp_prev: np.ndarray  # Euclidean displacement from previous observation (m)
    disp_next: np.ndarray  # Euclidean displacement to next observation (m)
    speed_prev: np.ndarray  # m/s from previous observation
    speed_next: np.ndarray  # m/s to next observation
    speed_smooth: np.ndarray  # local smoothed speed (m/s)
    heading_deg: np.ndarray  # estimated vehicle heading in [0, 360)
    heading_valid: np.ndarray  # 1 if movement > threshold so heading is reliable, else 0
    turn_rate_deg: np.ndarray  # local GPS turn angle magnitude (deg)


def _compute_gps_kinematics(
    gps_x: np.ndarray,
    gps_y: np.ndarray,
    timestamp: np.ndarray,
    min_move_m: float = 2.0,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute prediction-time GPS velocities, headings, and displacements."""
    n = len(gps_x)
    dt_prev = np.zeros(n, dtype=np.float64)
    dt_next = np.zeros(n, dtype=np.float64)
    disp_prev = np.zeros(n, dtype=np.float64)
    disp_next = np.zeros(n, dtype=np.float64)
    speed_prev = np.zeros(n, dtype=np.float64)
    speed_next = np.zeros(n, dtype=np.float64)
    speed_smooth = np.zeros(n, dtype=np.float64)
    heading_deg = np.zeros(n, dtype=np.float64)
    heading_valid = np.zeros(n, dtype=np.float64)
    turn_rate_deg = np.zeros(n, dtype=np.float64)

    if n <= 1:
        return (
            dt_prev,
            dt_next,
            disp_prev,
            disp_next,
            speed_prev,
            speed_next,
            speed_smooth,
            heading_deg,
            heading_valid,
            turn_rate_deg,
        )

    dt = np.maximum(np.diff(timestamp), 1e-3)
    dx = np.diff(gps_x)
    dy = np.diff(gps_y)
    dist = np.hypot(dx, dy)
    spd = np.clip(dist / dt, 0.0, 60.0)  # Cap extreme GPS-glitch speeds at 60 m/s (~216 km/h)

    dt_prev[1:] = dt
    dt_prev[0] = dt[0]
    dt_next[:-1] = dt
    dt_next[-1] = dt[-1]

    disp_prev[1:] = dist
    disp_prev[0] = dist[0]
    disp_next[:-1] = dist
    disp_next[-1] = dist[-1]

    speed_prev[1:] = spd
    speed_prev[0] = spd[0]
    speed_next[:-1] = spd
    speed_next[-1] = spd[-1]

    # Centered 3-point smoothed speed
    speed_smooth = 0.5 * (speed_prev + speed_next)

    # Centered heading vector over a short window to reduce 1 Hz GPS jitter
    # Uses (i+1) - (i-1) when moving; widens to ±2 or forward/backward if near-stationary
    hx = np.zeros(n, dtype=np.float64)
    hy = np.zeros(n, dtype=np.float64)
    hx[0] = gps_x[1] - gps_x[0]
    hy[0] = gps_y[1] - gps_y[0]
    hx[-1] = gps_x[-1] - gps_x[-2]
    hy[-1] = gps_y[-1] - gps_y[-2]
    if n > 2:
        hx[1:-1] = gps_x[2:] - gps_x[:-2]
        hy[1:-1] = gps_y[2:] - gps_y[:-2]

    h_norm = np.hypot(hx, hy)

    # Where centered 2-step displacement is small, try wider ±3 step displacement
    if n > 6:
        small = h_norm < min_move_m
        for idx in np.where(small)[0]:
            lo = max(0, idx - 3)
            hi = min(n - 1, idx + 3)
            wx = gps_x[hi] - gps_x[lo]
            wy = gps_y[hi] - gps_y[lo]
            wn = np.hypot(wx, wy)
            if wn >= min_move_m:
                hx[idx] = wx
                hy[idx] = wy
                h_norm[idx] = wn

    heading_deg = compute_bearings_deg(hx, hy)
    heading_valid = (h_norm >= min_move_m).astype(np.float64)

    # Forward-fill then backward-fill heading across stationary stops (e.g., red lights)
    valid_indices = np.where(heading_valid > 0.5)[0]
    if len(valid_indices) > 0:
        last_valid = heading_deg[valid_indices[0]]
        for i in range(n):
            if heading_valid[i] > 0.5:
                last_valid = heading_deg[i]
            else:
                heading_deg[i] = last_valid

    # Local turn angle magnitude between incoming and outgoing GPS vectors
    if n > 2:
        b_in = compute_bearings_deg(dx[:-1], dy[:-1])
        b_out = compute_bearings_deg(dx[1:], dy[1:])
        diff = np.abs(b_out - b_in) % 360.0
        turn = np.minimum(diff, 360.0 - diff)
        # Only count turn when both steps have meaningful movement
        move_ok = (dist[:-1] >= 1.5) & (dist[1:] >= 1.5)
        turn_rate_deg[1:-1] = np.where(move_ok, turn, 0.0)

    return (
        dt_prev,
        dt_next,
        disp_prev,
        disp_next,
        speed_prev,
        speed_next,
        speed_smooth,
        heading_deg,
        heading_valid,
        turn_rate_deg,
    )


def prepare_trip(trip: TripInputs) -> PreparedTrip:
    """Clean/sort GPS track if needed, project to UTM meters, build road graph and STRtree."""
    df = trip.track.copy()
    # Drop any NaN or out-of-range GPS rows (defensive cleaning)
    valid_mask = (
        df["lon"].notna()
        & df["lat"].notna()
        & df["timestamp"].notna()
        & df["lon"].between(-180.0, 180.0)
        & df["lat"].between(-90.0, 90.0)
    )
    if not valid_mask.all():
        df = df.loc[valid_mask].reset_index(drop=True)

    # Ensure chronological timestamp ordering (stable sort preserves original order on ties)
    if len(df) > 1 and not df["timestamp"].is_monotonic_increasing:
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    lon = df["lon"].to_numpy(dtype=np.float64)
    lat = df["lat"].to_numpy(dtype=np.float64)
    timestamp = df["timestamp"].to_numpy(dtype=np.float64)

    projector = TripProjector.from_lonlat(lon, lat)
    gps_x, gps_y = projector.to_xy(lon, lat)

    network = RoadNetwork(trip.nodes, trip.arcs, projector)
    spatial_index = RoadSpatialIndex(network)

    (
        dt_prev,
        dt_next,
        disp_prev,
        disp_next,
        speed_prev,
        speed_next,
        speed_smooth,
        heading_deg,
        heading_valid,
        turn_rate_deg,
    ) = _compute_gps_kinematics(gps_x, gps_y, timestamp)

    return PreparedTrip(
        trip_id=trip.trip_id,
        partition=trip.partition,
        projector=projector,
        network=network,
        spatial_index=spatial_index,
        lon=lon,
        lat=lat,
        timestamp=timestamp,
        gps_x=gps_x,
        gps_y=gps_y,
        dt_prev=dt_prev,
        dt_next=dt_next,
        disp_prev=disp_prev,
        disp_next=disp_next,
        speed_prev=speed_prev,
        speed_next=speed_next,
        speed_smooth=speed_smooth,
        heading_deg=heading_deg,
        heading_valid=heading_valid,
        turn_rate_deg=turn_rate_deg,
    )
