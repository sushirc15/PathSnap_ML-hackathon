"""High-level PathSnap HMM Map-Matcher combining classifier probabilities with Viterbi decoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import numpy as np

from src.data.preprocessing import PreparedTrip
from src.decoder.transitions import TransitionConfig
from src.decoder.viterbi import ViterbiDecodeResult, viterbi_decode
from src.geo.spatial_index import CandidateSet


@dataclass
class HMMConfig:
    """Complete serializable configuration for the PathSnap HMM/Viterbi decoder."""

    emission_weight: float = 1.5
    transition_weight: float = 1.0
    prob_floor: float = 1e-6
    beta_m: float = 12.0
    max_speed_mps: float = 45.0
    base_search_slack_m: float = 120.0
    max_search_dist_m: float = 1500.0
    uturn_penalty: float = 3.5
    same_arc_bonus: float = 0.25
    gap_dt_threshold_s: float = 15.0
    gap_dist_threshold_m: float = 350.0
    unreachable_penalty: float = -25.0

    def to_transition_config(self) -> TransitionConfig:
        return TransitionConfig(
            beta_m=self.beta_m,
            max_speed_mps=self.max_speed_mps,
            base_search_slack_m=self.base_search_slack_m,
            max_search_dist_m=self.max_search_dist_m,
            uturn_penalty=self.uturn_penalty,
            same_arc_bonus=self.same_arc_bonus,
            gap_dt_threshold_s=self.gap_dt_threshold_s,
            gap_dist_threshold_m=self.gap_dist_threshold_m,
            unreachable_penalty=self.unreachable_penalty,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HMMConfig":
        valid_fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


class PathSnapHMM:
    """Hidden Markov Model route reconstructor powered by classifier emission probabilities."""

    def __init__(self, config: HMMConfig | None = None) -> None:
        self.config = config or HMMConfig()

    def decode_from_flat_probs(
        self,
        trip: PreparedTrip,
        candidates: List[CandidateSet],
        obs_indices: np.ndarray,
        flat_probs: np.ndarray,
    ) -> ViterbiDecodeResult:
        """Group flat candidate probabilities by observation and run Viterbi decoding."""
        n_obs = len(candidates)
        obs_probs: List[np.ndarray] = [np.zeros(0, dtype=np.float64) for _ in range(n_obs)]

        cursor = 0
        for t, cset in enumerate(candidates):
            k = len(cset.arc_ids)
            if k > 0:
                obs_probs[t] = np.asarray(flat_probs[cursor : cursor + k], dtype=np.float64)
                cursor += k

        return viterbi_decode(
            trip=trip,
            candidates=candidates,
            obs_emission_probs=obs_probs,
            emission_weight=self.config.emission_weight,
            transition_weight=self.config.transition_weight,
            prob_floor=self.config.prob_floor,
            transition_cfg=self.config.to_transition_config(),
        )
