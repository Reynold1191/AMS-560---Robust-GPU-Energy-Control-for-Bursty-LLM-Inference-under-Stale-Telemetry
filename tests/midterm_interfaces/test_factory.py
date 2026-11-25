import importlib.util

import pytest

from config import CONTROLLER_NAMES, ExperimentConfig
from controllers import get_controller
from controllers.static import StaticController

HAS_NAIVE = importlib.util.find_spec("controllers.naive_slo") is not None
HAS_ROBUST = importlib.util.find_spec("controllers.robust_delay_budget") is not None
needs_naive = pytest.mark.skipif(not HAS_NAIVE, reason="controllers/naive_slo.py not merged yet")
needs_robust = pytest.mark.skipif(not HAS_ROBUST, reason="controllers/robust_delay_budget.py not merged yet")


@pytest.mark.parametrize("name, mhz", [
    ("default", 3090),
    ("static_high", 3090),
    ("static_low", 180),        # static_low_mhz=0 -> lowest clock
    ("static_efficient", 667),  # static_efficient_mhz=667
])
def test_static_names(clocks, cfg, name, mhz):
    c = get_controller(name, clocks, cfg)
    assert isinstance(c, StaticController)
    assert c.target_mhz == mhz


def test_static_low_uses_config_when_set(clocks):
    cfg = ExperimentConfig(static_low_mhz=1100)
    assert get_controller("static_low", clocks, cfg).target_mhz == 1147


def test_static_efficient_follows_config(clocks):
    cfg = ExperimentConfig(static_efficient_mhz=2100)
    assert get_controller("static_efficient", clocks, cfg).target_mhz == 2122


def test_unsorted_clocks_are_sorted(cfg):
    c = get_controller("static_low", [3090, 180, 1147], cfg)
    assert c.clocks == [180, 1147, 3090]
    assert c.target_mhz == 180


def test_unknown_name_raises(clocks, cfg):
    with pytest.raises(ValueError, match="Unknown controller"):
        get_controller("turbo", clocks, cfg)


def test_empty_clocks_raises(cfg):
    with pytest.raises(ValueError, match="selected_clocks"):
        get_controller("static_efficient", [], cfg)


@needs_naive
def test_naive_built_from_config(clocks):
    cfg = ExperimentConfig(slo_target_s=7.0, min_dwell_intervals=3)
    c = get_controller("naive_slo", clocks, cfg)
    assert c.name == "NaiveSLO"
    assert c.slo_target_s == 7.0


@needs_robust
@pytest.mark.parametrize("name, age, budget", [
    ("robust", True, True),
    ("robust_no_age", False, True),
    ("robust_no_budget", True, False),
])
def test_robust_variants(clocks, cfg, name, age, budget):
    c = get_controller(name, clocks, cfg)
    assert c.use_telemetry_age is age
    assert c.use_error_budget is budget


@needs_robust
def test_robust_respects_config_ablation_flags(clocks):
    cfg = ExperimentConfig(use_telemetry_age=False)
    assert get_controller("robust", clocks, cfg).use_telemetry_age is False


@needs_robust
def test_oracle_without_table_still_builds(clocks, cfg):
    assert cfg.oracle_table_path is None
    get_controller("oracle", clocks, cfg)


@needs_naive
@needs_robust
@pytest.mark.parametrize("name", CONTROLLER_NAMES)
def test_every_config_name_builds(clocks, cfg, name):
    assert get_controller(name, clocks, cfg).name


@needs_robust
@pytest.mark.parametrize("workload", ["steady", "ramp", "sharp_burst"])
@pytest.mark.parametrize("name", ["robust", "robust_no_age", "robust_no_budget", "oracle"])
def test_factory_forwards_workload(clocks, workload, name):
    cfg = ExperimentConfig(workload=workload)
    assert get_controller(name, clocks, cfg).workload == workload


@needs_robust
def test_oracle_receives_table_path(clocks, tmp_path):
    table = tmp_path / "oracle.csv"
    table.write_text("workload_bucket,min_clock_mhz\nsharp_burst,1635\n", encoding="utf-8")
    cfg = ExperimentConfig(workload="sharp_burst", oracle_table_path=str(table))
    controller = get_controller("oracle", clocks, cfg)
    assert controller.workload == cfg.workload
    assert controller.oracle_table is not None
    assert controller.oracle_table.iloc[0]["min_clock_mhz"] == 1635
