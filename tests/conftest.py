import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ExperimentConfig  # noqa: E402
from controllers.base import ControllerState  # noqa: E402

# Same clock ladder used in the team plan's examples.
CLOCKS = [180, 667, 1147, 1635, 2122, 2602, 3090]


def make_state(**overrides) -> ControllerState:
    """Calm baseline state (SLO 8.5 s, fresh telemetry, empty queue); override any field."""
    base = dict(slo_target_s=8.5, p95_latency_s=2.0, p99_latency_s=2.5, queue_len=0,
                telemetry_age_ms=0.0, error_budget_burn=0.0, arrival_cv=0.0,
                miss_rate=0.0, phase="steady")
    base.update(overrides)
    return ControllerState(**base)


# Named synthetic scenarios shared by all controller tests.
SCENARIOS = {
    "calm": make_state(),
    "slo_violation": make_state(p95_latency_s=10.0, p99_latency_s=12.0, miss_rate=0.2),
    "budget_burn": make_state(p95_latency_s=6.0, error_budget_burn=3.0, miss_rate=0.03),
    "stale_calm": make_state(telemetry_age_ms=600.0),
    "very_stale": make_state(telemetry_age_ms=1500.0),
    "queue_buildup": make_state(queue_len=12, p95_latency_s=5.0),
    "bursty": make_state(queue_len=8, arrival_cv=1.5, telemetry_age_ms=500.0, phase="burst"),
    "near_target": make_state(p95_latency_s=8.0),
}


@pytest.fixture
def clocks():
    return list(CLOCKS)


@pytest.fixture
def cfg():
    return ExperimentConfig()
