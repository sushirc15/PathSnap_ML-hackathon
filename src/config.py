"""PathSnap configuration and path management."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class PathSnapConfig:
    """Centralized configurable paths and default pipeline parameters."""

    project_root: Path = PROJECT_ROOT
    train_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_TRAIN_DIR", str(PROJECT_ROOT / "data/raw/train"))
        )
    )
    student_test_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_STUDENT_TEST_DIR", str(PROJECT_ROOT / "data/raw/student_test"))
        )
    )
    evaluator_test_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_EVALUATOR_TEST_DIR", str(PROJECT_ROOT / "data/evaluator_test"))
        )
    )
    processed_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_PROCESSED_DIR", str(PROJECT_ROOT / "data/processed"))
        )
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_CACHE_DIR", str(PROJECT_ROOT / "data/cache"))
        )
    )
    artifacts_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_ARTIFACTS_DIR", str(PROJECT_ROOT / "artifacts"))
        )
    )
    experiments_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_EXPERIMENTS_DIR", str(PROJECT_ROOT / "experiments"))
        )
    )
    reports_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("PATHSNAP_REPORTS_DIR", str(PROJECT_ROOT / "reports"))
        )
    )

    # Candidate generation defaults
    candidate_radius_m: float = 100.0
    candidate_fallback_radius_m: float = 300.0
    candidate_top_k: int = 16
    random_seed: int = 42


def get_config() -> PathSnapConfig:
    """Return active PathSnap configuration."""
    return PathSnapConfig()
