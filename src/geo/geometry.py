"""Vectorized metric geometry helpers for segments, projections, and bearings."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def compute_bearings_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compute clockwise compass bearing in [0, 360) from projected metric (dx, dy).

    North (+y) is 0 deg, East (+x) is 90 deg, South (-y) is 180 deg, West (-x) is 270 deg.
    """
    angles = np.degrees(np.arctan2(dx, dy))
    return np.mod(angles, 360.0)


def angular_diff_deg(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Smallest undirected angular difference in [0, 180] degrees between two bearings."""
    diff = np.abs(np.asarray(b1, dtype=np.float64) - np.asarray(b2, dtype=np.float64)) % 360.0
    return np.minimum(diff, 360.0 - diff)


def signed_angular_diff_deg(b_from: np.ndarray, b_to: np.ndarray) -> np.ndarray:
    """Signed turn angle in (-180, 180] degrees from b_from to b_to."""
    diff = (np.asarray(b_to, dtype=np.float64) - np.asarray(b_from, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return diff


def point_to_segments_projection(
    px: float,
    py: float,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized projection of a single point (px, py) onto K directed line segments.

    Returns:
        dist_m: perpendicular/clamped Euclidean distance in meters, shape (K,)
        frac: clamped fraction along segment in [0, 1], shape (K,)
        raw_frac: unclamped projection fraction along infinite line, shape (K,)
        proj_x: projected x coordinate on segment, shape (K,)
        proj_y: projected y coordinate on segment, shape (K,)
    """
    vx = x2 - x1
    vy = y2 - y1
    seg_len_sq = vx * vx + vy * vy

    wx = px - x1
    wy = py - y1

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_frac = np.where(seg_len_sq > 1e-12, (wx * vx + wy * vy) / seg_len_sq, 0.0)
    frac = np.clip(raw_frac, 0.0, 1.0)

    proj_x = x1 + frac * vx
    proj_y = y1 + frac * vy

    dx = px - proj_x
    dy = py - proj_y
    dist_m = np.hypot(dx, dy)
    return dist_m, frac, raw_frac, proj_x, proj_y
