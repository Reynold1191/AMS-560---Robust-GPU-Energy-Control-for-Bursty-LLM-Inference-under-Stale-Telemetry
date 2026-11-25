import pytest

from config import DELAY_VALUES_MS, VERIFIED_CLOCKS_MHZ
from experiments.stale_delay_sweep import (
    CLOCKS, DELAYS_MS, age_threshold_ms, robust_settings, run_sweep, summarize,
)

SEEDS = 10


@pytest.fixture(scope="module")
def summary():
    return summarize(run_sweep(seeds=SEEDS)).set_index(["controller", "delay_ms"])


@pytest.fixture(scope="module")
def no_guard_summary():
    return summarize(run_sweep(seeds=SEEDS, guard=False)).set_index(["controller", "delay_ms"])


def _value(summary, controller, delay, column):
    return float(summary.loc[(controller, delay), column])


def test_sweep_uses_default_robust_with_300ms_guard():
    assert robust_settings(guard=True)["stale_downclock_ms"] == 300.0


def test_sweep_uses_verified_clocks_and_calibrated_delays():
    assert CLOCKS == VERIFIED_CLOCKS_MHZ
    assert set(DELAY_VALUES_MS) <= set(DELAYS_MS)


def test_static_high_saves_nothing(summary):
    for delay in summary.loc["static_high"].index:
        assert _value(summary, "static_high", delay, "energy_saved_pct") == pytest.approx(0.0)


def test_robust_falls_back_to_static_high_above_guard(summary):
    for delay in (500, 1250, 2500):
        assert _value(summary, "robust", delay, "miss_rate") == pytest.approx(
            _value(summary, "static_high", delay, "miss_rate"))
        assert _value(summary, "robust", delay, "energy_saved_pct") == pytest.approx(0.0, abs=0.1)


def test_robust_saves_energy_below_guard(summary):
    for delay in (0, 100, 250):
        assert _value(summary, "robust", delay, "energy_saved_pct") > 1.0


@pytest.mark.parametrize("controller", ["naive_slo", "robust_no_age"])
def test_age_blind_controllers_do_not_react_to_delay(summary, controller):
    misses = summary.loc[controller]["miss_rate"]
    static = summary.loc["static_high"]["miss_rate"]
    assert (summary.loc[controller]["energy_saved_pct"] > 1.0).all()
    assert (misses > static + 0.10).all()
    assert misses.max() - misses.min() < 0.05


def test_no_guard_diagnostic_switches_at_age_threshold(no_guard_summary):
    assert 500 < age_threshold_ms() < 1250
    assert _value(no_guard_summary, "robust_no_guard", 500, "energy_saved_pct") > 1.0
    for delay in (1250, 2500):
        assert _value(no_guard_summary, "robust_no_guard", delay,
                      "energy_saved_pct") == pytest.approx(0.0, abs=0.1)
