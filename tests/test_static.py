import pytest

from conftest import SCENARIOS, make_state
from controllers.base import ControllerState
from controllers.static import StaticController


def test_default_target_is_highest(clocks):
    c = StaticController(clocks)
    assert c.target_mhz == 3090
    assert c.name == "Static(3090 MHz)"


def test_target_snaps_to_nearest_clock(clocks):
    assert StaticController(clocks, target_mhz=700).target_mhz == 667
    assert StaticController(clocks, target_mhz=1300).target_mhz == 1147
    assert StaticController(clocks, target_mhz=99999).target_mhz == 3090


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_ignores_state(clocks, scenario):
    c = StaticController(clocks, target_mhz=667)
    for start in range(len(clocks)):
        assert c._policy(SCENARIOS[scenario], start) == 1
    assert c.last_reason == "STATIC(667MHz)"


def test_upclock_to_target_is_immediate(clocks):
    c = StaticController(clocks, target_mhz=2602)
    assert c.decide(make_state(), current_idx=0, min_dwell=5) == 5


def test_downclock_to_target_respects_dwell(clocks):
    c = StaticController(clocks, target_mhz=180)
    state = make_state()
    assert c.decide(state, 6, min_dwell=2) == 6  # 1st interval: hold
    assert c.decide(state, 6, min_dwell=2) == 0  # 2nd interval: move


def test_empty_clocks_rejected():
    with pytest.raises(ValueError):
        StaticController([])


def test_clock_mhz_clamps(clocks):
    c = StaticController(clocks)
    assert c.clock_mhz(-3) == 180
    assert c.clock_mhz(42) == 3090


def test_latency_slack():
    assert ControllerState(slo_target_s=8.5, p95_latency_s=0.0).latency_slack == 1.0
    assert ControllerState(slo_target_s=8.5, p95_latency_s=8.5).latency_slack == 0.0
    assert ControllerState(slo_target_s=8.5, p95_latency_s=17.0).latency_slack == -1.0
