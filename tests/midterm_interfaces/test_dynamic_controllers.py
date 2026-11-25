"""Scenario tests for naive_slo and robust. Skipped until those files are merged."""
import pytest

from conftest import SCENARIOS, make_state
from controllers import get_controller

naive_mod = pytest.importorskip("controllers.naive_slo")
robust_mod = pytest.importorskip("controllers.robust_delay_budget")

MID = 3  # 1635 MHz, room to move either way


def test_naive_emergency_upclocks(clocks, cfg):
    c = get_controller("naive_slo", clocks, cfg)
    assert c._policy(SCENARIOS["slo_violation"], MID) == MID + 1
    assert c.last_reason == "EMERGENCY_SLO"


def test_naive_downclocks_when_calm(clocks, cfg):
    c = get_controller("naive_slo", clocks, cfg)
    assert c._policy(SCENARIOS["calm"], MID) == MID - 1
    assert c.last_reason == "SAFE_DOWNCLOCK"


def test_naive_ignores_staleness(clocks, cfg):
    """The baseline's flaw: stale-but-calm telemetry still triggers a down-clock."""
    c = get_controller("naive_slo", clocks, cfg)
    assert c._policy(SCENARIOS["stale_calm"], MID) == MID - 1


def test_naive_holds_with_queue(clocks, cfg):
    c = get_controller("naive_slo", clocks, cfg)
    assert c._policy(make_state(queue_len=1), MID) == MID
    assert c.last_reason == "HOLD"


def test_robust_emergency_steps_two(clocks, cfg):
    c = get_controller("robust", clocks, cfg)
    assert c._policy(SCENARIOS["slo_violation"], MID) == MID + 2
    assert c.last_reason == "EMERGENCY_SLO"


def test_robust_budget_emergency(clocks, cfg):
    c = get_controller("robust", clocks, cfg)
    assert c._policy(SCENARIOS["budget_burn"], MID) == MID + 2
    assert c.last_reason == "EMERGENCY_BUDGET"


def test_robust_holds_under_stale_telemetry(clocks, cfg):
    c = get_controller("robust", clocks, cfg)
    c._policy(SCENARIOS["stale_calm"], MID)
    # With the plan's weights, age alone adds at most w_age * 1.0 = 0.35 risk, which is
    # below low_risk_thresh (0.40). Passing this needs an explicit staleness guard
    # (e.g. no down-clock when age > 300 ms) or retuned weights/thresholds.
    assert "DOWNCLOCK" not in c.last_reason, (
        f"robust down-clocked on 600 ms stale telemetry (reason={c.last_reason})")


def test_robust_no_age_downclocks_when_stale(clocks, cfg):
    c = get_controller("robust_no_age", clocks, cfg)
    c._policy(SCENARIOS["stale_calm"], MID)
    assert "DOWNCLOCK" in c.last_reason


def test_robust_high_risk_on_burst(clocks, cfg):
    c = get_controller("robust", clocks, cfg)
    assert c._policy(SCENARIOS["bursty"], MID) == MID + 1
    assert c.last_reason.startswith("HIGH_RISK")
    assert "STALE_OBSERVATION" in c.last_reason


def test_robust_downclocks_when_calm(clocks, cfg):
    c = get_controller("robust", clocks, cfg)
    assert c._policy(SCENARIOS["calm"], MID) == MID - 1


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("name", ["naive_slo", "robust", "robust_no_age", "robust_no_budget"])
def test_decide_stays_in_range(clocks, cfg, name, scenario):
    c = get_controller(name, clocks, cfg)
    for start in (0, MID, len(clocks) - 1):
        idx = c.decide(SCENARIOS[scenario], start, cfg.min_dwell_intervals)
        assert 0 <= idx < len(clocks)
        assert isinstance(c.last_reason, str) and c.last_reason != "INIT"
