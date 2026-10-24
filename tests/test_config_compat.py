"""Checks that config.py and the controller interfaces agree."""
import dataclasses

from config import CONTROLLER_NAMES, ExperimentConfig
from controllers import ROBUST_NAMES, STATIC_NAMES
from controllers.base import ControllerState

FACTORY_NAMES = set(STATIC_NAMES) | {"naive_slo"} | set(ROBUST_NAMES)

# Controllers that run_demo.py loops over (team plan, Member 5).
DEMO_CONTROLLERS = ["static_efficient", "naive_slo", "robust_no_age", "robust"]


def test_factory_covers_config_names():
    assert set(CONTROLLER_NAMES) == FACTORY_NAMES


def test_demo_controllers_are_known():
    assert set(DEMO_CONTROLLERS) <= set(CONTROLLER_NAMES)


def test_default_controller_is_known(cfg):
    assert cfg.controller in CONTROLLER_NAMES


def test_config_has_fields_the_factory_reads(cfg):
    for f in ("slo_target_s", "min_dwell_intervals", "use_telemetry_age",
              "use_error_budget", "oracle_table_path", "static_efficient_mhz",
              "static_low_mhz", "selected_clocks"):
        assert hasattr(cfg, f), f


def test_config_field_types(cfg):
    assert isinstance(cfg.min_dwell_intervals, int) and cfg.min_dwell_intervals >= 0
    assert isinstance(cfg.use_telemetry_age, bool)
    assert isinstance(cfg.use_error_budget, bool)
    assert cfg.slo_target_s > 0


def test_selected_clocks_not_shared():
    a, b = ExperimentConfig(), ExperimentConfig()
    a.selected_clocks.append(667)
    assert b.selected_clocks == []


def test_slo_budget_summary_maps_onto_state():
    """run_experiment.py builds ControllerState from SLOBudget.summary()."""
    from slo_budget import SLOBudget
    s = SLOBudget(8.5, 0.01, 60.0).summary()
    state_fields = {f.name for f in dataclasses.fields(ControllerState)}
    mapping = {"p95_latency_s": "p95_latency_s", "p99_latency_s": "p99_latency_s",
               "burn_rate": "error_budget_burn", "miss_rate": "miss_rate"}
    for src, dst in mapping.items():
        assert src in s and dst in state_fields


def test_state_slo_default_matches_config(cfg):
    """State metadata and the experiment use the same default target."""
    assert ControllerState().slo_target_s == cfg.slo_target_s
