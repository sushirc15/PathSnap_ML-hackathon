"""Candidate road-segment generation for PathSnap trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.data.preprocessing import PreparedTrip
from src.geo.spatial_index import CandidateSet


@dataclass(frozen=True)
class CandidateConfig:
    """Configurable spatial candidate search parameters."""

    radius_m: float = 100.0
    fallback_radius_m: float = 300.0
    top_k: int = 16


def generate_candidates(
    trip: PreparedTrip,
    config: CandidateConfig | None = None,
) -> List[CandidateSet]:
    """Generate top-K nearby directed road segments for every GPS observation in a trip."""
    cfg = config or CandidateConfig()
    return trip.spatial_index.query_trajectory(
        gps_x=trip.gps_x,
        gps_y=trip.gps_y,
        radius_m=cfg.radius_m,
        fallback_radius_m=cfg.fallback_radius_m,
        top_k=cfg.top_k,
    )
