"""Member 2: integration checks for shared state, factory, and dwell contract."""

import time

import pytest

from config import CONTROLLER_NAMES, ExperimentConfig
from controllers import get_controller
from controllers.base import BaseController, ControllerState
from controllers.naive_slo import NaiveSLOController
from controllers.robust_delay_budget import RobustDelayBudgetController
from controllers.static import StaticController

CLOCKS = [180, 667, 1147, 1635, 2122, 2602, 3090]


@pytest.mark.parametrize("name", CONTROLLER_NAMES)
def test_all_configured_names_have_a_factory(name):
    controller = get_controller(name, CLOCKS, ExperimentConfig())
    expected = (
        NaiveSLOController if name == "naive_slo" else
        RobustDelayBudgetController if name.startswith("robust") or name == "oracle" else
        StaticController
    )
    assert isinstance(controller, expected)
    assert controller.name


def test_unknown_controller_and_empty_clocks_are_rejected():
    with pytest.raises(ValueError, match="Unknown controller"):
        get_controller("typo", CLOCKS, ExperimentConfig())
    with pytest.raises(ValueError, match="at least one"):
        get_controller("static_high", [], ExperimentConfig())


@pytest.mark.parametrize("clocks", [[], [667, 180], [180, 180], [0, 180]])
def test_invalid_clock_lists_are_rejected(clocks):
    with pytest.raises(ValueError):
        StaticController(clocks)


def test_static_targets_and_runtime_fallbacks():
    cfg = ExperimentConfig()
    state = ControllerState()
    for name, expected in [("default", 3090), ("static_high", 3090), ("static_low", 180), ("static_efficient", 667)]:
        controller = get_controller(name, CLOCKS, cfg)
        assert controller.clock_mhz(controller.decide(state, 3)) == expected
        assert controller.clock_mhz(controller.decide(state, 0)) == expected
    cfg.static_low_mhz = 1100
    cfg.static_high_mhz = 2602
    assert get_controller("static_low", CLOCKS, cfg).decide(state, 0) == 2
    assert get_controller("static_high", CLOCKS, cfg).decide(state, 0) == 5
    assert get_controller("default", CLOCKS, cfg).decide(state, 0) == 6


def test_clocks_are_copied_and_indices_clamped():
    clocks = CLOCKS.copy()
    controller = StaticController(clocks)
    clocks.clear()
    assert controller.clock_mhz(-10) == 180
    assert controller.clock_mhz(100) == 3090
    assert controller.find_index(1000) == 2
    assert controller.find_index(-1) == 0
    assert controller.find_index(10000) == 6


def test_state_timestamps_and_slack():
    before = time.monotonic()
    state = ControllerState()
    assert before <= state.t <= time.monotonic()
    assert ControllerState(slo_target_s=8.5, p95_latency_s=4.25).latency_slack == 0.5
    assert ControllerState(slo_target_s=8.5, p95_latency_s=17).latency_slack == -1.0


def test_dwell_delays_reductions_but_never_emergency_increases():
    controller = NaiveSLOController(CLOCKS, min_dwell=3)
    safe = ControllerState(slo_target_s=8.5, p95_latency_s=2.0)
    emergency = ControllerState(slo_target_s=8.5, p95_latency_s=10.0)
    assert controller.decide(safe, 3) == 3
    assert "DWELL_HOLD" in controller.last_reason
    assert controller.decide(emergency, 3) == 4
    assert controller.last_reason == "EMERGENCY_SLO"
    assert controller.decide(safe, 4) == 4
    assert controller.decide(safe, 4) == 4
    assert controller.decide(safe, 4) == 3


def test_explicit_dwell_override_and_change_timestamp(monkeypatch):
    controller = NaiveSLOController(CLOCKS, min_dwell=10)
    monkeypatch.setattr("controllers.base.time.monotonic", lambda: 456.0)
    assert controller.decide(ControllerState(), 3, min_dwell=0) == 2
    assert controller._last_change_t == 456.0
    with pytest.raises(ValueError, match="nonnegative"):
        controller.decide(ControllerState(), 3, min_dwell=-1)


def test_base_clamps_policy_output():
    class OutOfRangeController(BaseController):
        def _policy(self, state, current_idx):
            return 100 if state.queue_len else -100

    controller = OutOfRangeController(CLOCKS)
    assert controller.decide(ControllerState(queue_len=1), 3) == 6
    assert controller.decide(ControllerState(), 3) == 0
