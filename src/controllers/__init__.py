"""
controllers/__init__.py
Controller package.

Import helpers so callers can write:
    from controllers import get_controller
"""

from controllers.base import BaseController, ControllerState
from controllers.static import StaticController
from controllers.naive_slo import NaiveSLOController
from controllers.robust_delay_budget import RobustDelayBudgetController


def _robust_kwargs(config) -> dict:
    return dict(
        stale_downclock_ms=getattr(config, "stale_downclock_ms", None),
        age_scale_ms=getattr(config, "robust_age_scale_ms", 5000.0),
        delay_aware=getattr(config, "robust_delay_aware", True),
        clock_settle_s=getattr(config, "clock_settle_s", 0.7),
    )


DYNAMIC = ("naive_slo", "robust", "robust_no_age", "robust_no_budget", "robust_v1")


def get_controller(name: str, clocks, config) -> BaseController:
    """Build a controller; dynamic ones get the measured capacity floor."""
    ctrl = _make_controller(name, clocks, config)
    cap = getattr(config, "capacity_rps", None)
    if name.lower() in DYNAMIC and cap is not None:
        ctrl.set_capacity(cap, getattr(config, "batch_time_s", None),
                          getattr(config, "capacity_margin", 1.25))
    return ctrl


def _make_controller(name: str, clocks, config) -> BaseController:
    """
    Factory that returns a controller instance by name.

    Parameters
    ----------
    name   : One of 'default', 'static_high', 'static_low', 'static_efficient',
             'naive_slo', 'robust', 'robust_v1', 'robust_no_age', 'robust_no_budget',
             'oracle'.
    clocks : Sorted list of supported GPU clock values (MHz).
    config : ExperimentConfig.
    """
    name = name.lower()

    if name == "default":
        return StaticController(clocks, target_mhz=clocks[-1] if clocks else 0)

    if name == "static_high":
        mhz = getattr(config, "static_high_mhz", 0) or (clocks[-1] if clocks else 0)
        return StaticController(clocks, target_mhz=mhz)

    if name == "static_low":
        mhz = config.static_low_mhz or (clocks[0] if clocks else 0)
        return StaticController(clocks, target_mhz=mhz)

    if name == "static_efficient":
        # Best J/token clock from sweep; default 1147 MHz confirmed on RTX 5060
        mhz = getattr(config, "static_efficient_mhz", 1147)
        # Find closest supported clock
        best = min(clocks, key=lambda c: abs(c - mhz)) if clocks else mhz
        return StaticController(clocks, target_mhz=best)

    if name == "naive_slo":
        return NaiveSLOController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
        )

    if name == "robust":
        return RobustDelayBudgetController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
            use_telemetry_age=config.use_telemetry_age,
            use_error_budget=config.use_error_budget,
            oracle_mode=False,
            oracle_table_path=config.oracle_table_path,
            **_robust_kwargs(config),
        )

    if name == "robust_v1":
        # Ablation: the attempt_8-11 controller, without the age-compensated
        # load forecast and the lag-aware treatment of P95 / burn.
        return RobustDelayBudgetController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
            use_telemetry_age=config.use_telemetry_age,
            use_error_budget=config.use_error_budget,
            oracle_mode=False,
            forecast=False,
            lag_aware=False,
            **_robust_kwargs(config),
        )

    if name == "robust_no_age":
        return RobustDelayBudgetController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
            use_telemetry_age=False,   # ablation: remove age term
            use_error_budget=True,
            oracle_mode=False,
            **_robust_kwargs(config),
        )

    if name == "robust_no_budget":
        return RobustDelayBudgetController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
            use_telemetry_age=True,
            use_error_budget=False,    # ablation: remove budget term
            oracle_mode=False,
            **_robust_kwargs(config),
        )

    if name in ("oracle",):
        return RobustDelayBudgetController(
            clocks,
            slo_target_s=config.slo_target_s,
            min_dwell=config.min_dwell_intervals,
            use_telemetry_age=config.use_telemetry_age,
            use_error_budget=config.use_error_budget,
            oracle_mode=True,
            oracle_table_path=config.oracle_table_path,
        )

    raise ValueError(
        f"Unknown controller '{name}'. "
        "Valid: default, static_high, static_low, static_efficient, "
        "naive_slo, robust, robust_v1, robust_no_age, robust_no_budget, oracle"
    )
