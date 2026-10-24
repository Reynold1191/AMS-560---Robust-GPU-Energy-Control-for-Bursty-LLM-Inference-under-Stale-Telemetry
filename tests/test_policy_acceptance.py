"""Member 2: policy acceptance, ablation, and burst/recovery behavior."""

from dataclasses import replace

import pytest

from config import ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState
from controllers.naive_slo import NaiveSLOController
from controllers.robust_delay_budget import RobustDelayBudgetController

CLOCKS = [180, 667, 1147, 1635, 2122, 2602, 3090]
SAFE = ControllerState(slo_target_s=8.5, p95_latency_s=2.0)


@pytest.mark.parametrize(
    "state, expected, reason",
    [
        (replace(SAFE, p95_latency_s=10.0), 4, "EMERGENCY_SLO"),
        (SAFE, 2, "SAFE_DOWNCLOCK"),
        (replace(SAFE, queue_len=1), 3, "HOLD"),
        (replace(SAFE, p95_latency_s=8.5), 3, "HOLD"),
        (replace(SAFE, p95_latency_s=8.5 * 0.8), 3, "HOLD"),
    ],
)
def test_naive_policy_acceptance(state, expected, reason):
    controller = NaiveSLOController(CLOCKS)
    assert controller.decide(state, 3, min_dwell=0) == expected
    assert controller.last_reason == reason


def test_naive_ignores_age_and_budget():
    controller = NaiveSLOController(CLOCKS)
    state = replace(SAFE, telemetry_age_ms=1000, error_budget_burn=50, arrival_cv=2)
    assert controller.decide(state, 3, min_dwell=0) == 2


def test_risk_formula_and_saturation():
    controller = RobustDelayBudgetController(CLOCKS)
    state = replace(SAFE, queue_len=5, telemetry_age_ms=600, error_budget_burn=0.5, arrival_cv=0.25)
    assert controller._risk(state) == pytest.approx(0.4975)
    saturated = replace(state, queue_len=100, telemetry_age_ms=10000, error_budget_burn=50, arrival_cv=4)
    assert controller._risk(saturated) == pytest.approx(1.0)
    negative = replace(state, queue_len=-1, telemetry_age_ms=-1, error_budget_burn=-1, arrival_cv=-1)
    assert controller._risk(negative) == 0.0


def test_stale_observation_holds_but_age_ablation_reduces_clock():
    stale = replace(SAFE, telemetry_age_ms=600)
    robust = RobustDelayBudgetController(CLOCKS)
    no_age = RobustDelayBudgetController(CLOCKS, use_telemetry_age=False, use_error_budget=False)
    assert robust.decide(stale, 3, min_dwell=0) == 3
    assert "STALE_OBSERVATION" in robust.last_reason
    assert "DOWNCLOCK" not in robust.last_reason
    assert no_age.decide(stale, 3, min_dwell=0) == 2
    assert "DOWNCLOCK" in no_age.last_reason


@pytest.mark.parametrize("age_ms, expected", [(0, 2), (250, 2), (300, 2), (301, 3), (600, 3), (1000, 3)])
def test_stale_guard_boundary(age_ms, expected):
    controller = RobustDelayBudgetController(CLOCKS)
    assert controller.decide(replace(SAFE, telemetry_age_ms=age_ms), 3, min_dwell=0) == expected


@pytest.mark.parametrize(
    "state, expected, reason",
    [
        (replace(SAFE, p95_latency_s=10.0, error_budget_burn=3), 5, "EMERGENCY_SLO"),
        (replace(SAFE, error_budget_burn=3), 5, "EMERGENCY_BUDGET"),
        (replace(SAFE, queue_len=10, telemetry_age_ms=1000), 4, "HIGH_RISK"),
        (replace(SAFE, queue_len=10, error_budget_burn=1, arrival_cv=1), 4, "HIGH_RISK"),
        (replace(SAFE, p95_latency_s=8.5 * 0.8), 3, "HOLD"),
        (SAFE, 2, "SAFE_DOWNCLOCK"),
    ],
)
def test_robust_priorities(state, expected, reason):
    controller = RobustDelayBudgetController(CLOCKS)
    assert controller.decide(state, 3, min_dwell=0) == expected
    assert controller.last_reason.startswith(reason)
    assert controller.last_risk == controller._risk(state)


def test_risk_thresholds_are_strict():
    for low, high in [(0.25, 0.5), (0.5, 0.75)]:
        controller = RobustDelayBudgetController(
            CLOCKS, w_q=1, w_age=0, w_burn=0, w_cv=0,
            low_risk_thresh=low, high_risk_thresh=high,
        )
        assert controller.decide(replace(SAFE, queue_len=5), 3, min_dwell=0) == 3
        assert controller.last_reason == "HOLD(r=0.50)"


def test_budget_ablation_removes_risk_term_and_emergency():
    state = replace(SAFE, error_budget_burn=50)
    robust = RobustDelayBudgetController(CLOCKS)
    no_budget = RobustDelayBudgetController(CLOCKS, use_error_budget=False)
    assert robust.decide(state, 3, min_dwell=100) == 5
    assert no_budget._risk(state) == 0
    assert no_budget.decide(state, 3, min_dwell=0) == 2
    # Budget ablation still respects the age guard and SLO emergency.
    assert no_budget.decide(replace(state, telemetry_age_ms=600), 3, min_dwell=0) == 3
    assert no_budget.decide(replace(state, p95_latency_s=10), 3) == 5


def test_factory_respects_config_and_named_ablation_overrides():
    cfg = ExperimentConfig(use_telemetry_age=False, use_error_budget=False, min_dwell_intervals=4)
    robust = get_controller("robust", CLOCKS, cfg)
    assert (robust.use_telemetry_age, robust.use_error_budget, robust.min_dwell) == (False, False, 4)
    no_age = get_controller("robust_no_age", CLOCKS, cfg)
    assert (no_age.use_telemetry_age, no_age.use_error_budget) == (False, True)
    no_budget = get_controller("robust_no_budget", CLOCKS, cfg)
    assert (no_budget.use_telemetry_age, no_budget.use_error_budget) == (True, False)


def test_burst_and_recovery_sequence():
    controller = RobustDelayBudgetController(CLOCKS, min_dwell=2)
    trace = [
        SAFE,
        SAFE,
        replace(SAFE, telemetry_age_ms=600),
        replace(SAFE, telemetry_age_ms=600, queue_len=10, arrival_cv=1),
        replace(SAFE, error_budget_burn=3),
        SAFE,
        SAFE,
    ]
    idx = 3
    decisions = []
    for state in trace:
        idx = controller.decide(state, idx)
        decisions.append(idx)
    assert decisions == [3, 2, 2, 3, 5, 5, 4]


@pytest.mark.parametrize("controller_type", [NaiveSLOController, RobustDelayBudgetController])
def test_decisions_respect_frequency_bounds(controller_type):
    controller = controller_type(CLOCKS)
    assert controller.decide(SAFE, 0, min_dwell=0) == 0
    assert controller.decide(replace(SAFE, p95_latency_s=10), 6) == 6
    single = controller_type([667])
    assert single.decide(SAFE, 0, min_dwell=0) == 0
    assert single.decide(replace(SAFE, p95_latency_s=10), 0) == 0


@pytest.mark.parametrize("name, emergency_step", [("naive_slo", 1), ("robust", 2)])
@pytest.mark.parametrize("target", [2.0, 7.0, 8.5, 12.0])
@pytest.mark.parametrize("state_target", [None, 2.0, 20.0])
def test_decisions_use_configured_slo(name, emergency_step, target, state_target):
    """Default or conflicting state metadata cannot change the experiment SLO."""
    controller = get_controller(name, CLOCKS, ExperimentConfig(slo_target_s=target))
    overrides = {} if state_target is None else {"slo_target_s": state_target}
    calm = ControllerState(p95_latency_s=target / 2, **overrides)
    violation = replace(calm, p95_latency_s=target + 1)
    assert controller.decide(calm, 3, min_dwell=0) == 2
    assert "SAFE_DOWNCLOCK" in controller.last_reason
    assert controller.decide(violation, 3, min_dwell=0) == 3 + emergency_step
    assert controller.last_reason == "EMERGENCY_SLO"


@pytest.mark.parametrize("controller_type", [NaiveSLOController, RobustDelayBudgetController])
def test_direct_controller_slo_default_matches_config(controller_type):
    assert controller_type(CLOCKS).slo_target_s == ExperimentConfig().slo_target_s
