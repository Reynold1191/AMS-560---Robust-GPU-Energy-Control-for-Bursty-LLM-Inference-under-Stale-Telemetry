from typing import List

from .base import BaseController, ControllerState
from .static import StaticController

__all__ = ["BaseController", "ControllerState", "StaticController", "get_controller"]

STATIC_NAMES = ("default", "static_high", "static_low", "static_efficient")
ROBUST_NAMES = ("robust", "robust_no_age", "robust_no_budget", "oracle")


def _static_target(name: str, clocks: List[int], config) -> int:
    if name == "default":
        return clocks[-1]
    if name == "static_high":
        # config.py has no static_high_mhz; fall back to the top clock.
        return getattr(config, "static_high_mhz", 0) or clocks[-1]
    if name == "static_low":
        return getattr(config, "static_low_mhz", 0) or clocks[0]
    return getattr(config, "static_efficient_mhz", 0) or clocks[0]


def _robust_kwargs(name: str, config) -> dict:
    kw = dict(slo_target_s=config.slo_target_s, min_dwell=config.min_dwell_intervals,
              workload=config.workload)
    if name == "robust":
        kw.update(use_telemetry_age=config.use_telemetry_age,
                  use_error_budget=config.use_error_budget, oracle_mode=False)
    elif name == "robust_no_age":
        kw.update(use_telemetry_age=False, use_error_budget=True)
    elif name == "robust_no_budget":
        kw.update(use_telemetry_age=True, use_error_budget=False)
    else:  # oracle
        kw.update(oracle_mode=True, oracle_table_path=config.oracle_table_path)
    return kw


def get_controller(name: str, clocks: List[int], config) -> BaseController:
    if not clocks:
        raise ValueError("clocks must contain at least one frequency; populate config.selected_clocks first")
    clocks = sorted(clocks)

    if name in STATIC_NAMES:
        return StaticController(clocks, target_mhz=_static_target(name, clocks, config))

    # Imported lazily so static controllers work before the dynamic ones exist.
    if name == "naive_slo":
        from .naive_slo import NaiveSLOController
        return NaiveSLOController(clocks, slo_target_s=config.slo_target_s,
                                  min_dwell=config.min_dwell_intervals)

    if name in ROBUST_NAMES:
        from .robust_delay_budget import RobustDelayBudgetController
        return RobustDelayBudgetController(clocks, **_robust_kwargs(name, config))

    raise ValueError(f"Unknown controller {name!r}. Expected one of "
                     f"{STATIC_NAMES + ('naive_slo',) + ROBUST_NAMES}")
