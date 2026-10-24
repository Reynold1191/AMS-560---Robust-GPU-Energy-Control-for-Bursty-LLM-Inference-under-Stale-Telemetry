"""Member 2: CSV compatibility and robust fallback for the oracle baseline."""

import pytest

from config import ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState
from controllers.robust_delay_budget import RobustDelayBudgetController

CLOCKS = [180, 667, 1147, 1635, 2122, 2602, 3090]
SAFE = ControllerState(slo_target_s=8.5, p95_latency_s=2.0, phase="burst")


def test_phase_table_selects_matching_row_and_bypasses_robust_policy(tmp_path):
    table = tmp_path / "oracle.csv"
    table.write_text("phase,min_clock_mhz\nsteady,667\nburst,2122\n", encoding="utf-8")
    controller = RobustDelayBudgetController(CLOCKS, oracle_mode=True, oracle_table_path=table)
    assert controller.decide(SAFE, 2) == 4
    assert controller.last_reason == "ORACLE(2122MHz)"
    assert controller.name == "Oracle"


def test_member3_workload_bucket_schema_uses_config_workload(tmp_path):
    table = tmp_path / "oracle.csv"
    table.write_text(
        "workload_bucket,min_clock_mhz,j_per_token,miss_rate\nsteady,667,0.1,0\nsharp_burst,1635,0.2,0\n",
        encoding="utf-8",
    )
    cfg = ExperimentConfig(oracle_table_path=str(table), workload="sharp_burst")
    controller = get_controller("oracle", CLOCKS, cfg)
    assert controller.decide(SAFE, 2) == 3
    assert controller.last_reason == "ORACLE(1635MHz)"


def test_oracle_minimum_rounds_up_to_supported_clock(tmp_path):
    table = tmp_path / "oracle.csv"
    table.write_text("phase,min_clock_mhz\nburst,1200\n", encoding="utf-8")
    controller = RobustDelayBudgetController(CLOCKS, oracle_mode=True, oracle_table_path=table)
    assert controller.decide(SAFE, 2) == 3


@pytest.mark.parametrize(
    "contents",
    [
        "phase,min_clock_mhz\nsteady,667\n",
        "phase,wrong_column\nburst,667\n",
        "wrong_column,min_clock_mhz\nburst,667\n",
        "phase,min_clock_mhz\nburst,invalid\n",
        "phase,min_clock_mhz\nburst,nan\n",
        "phase,min_clock_mhz\nburst,inf\n",
        "phase,min_clock_mhz\nburst,-1\n",
        "phase,min_clock_mhz\nburst,4000\n",
        "phase,min_clock_mhz\n",
        "",
    ],
)
def test_unusable_oracle_tables_fall_back_without_recursion(tmp_path, contents):
    table = tmp_path / "oracle.csv"
    table.write_text(contents, encoding="utf-8")
    controller = RobustDelayBudgetController(CLOCKS, oracle_mode=True, oracle_table_path=table)
    assert controller.decide(SAFE, 3, min_dwell=0) == 2
    assert controller.last_reason.startswith("SAFE_DOWNCLOCK")


def test_missing_oracle_path_logs_and_falls_back(tmp_path, caplog):
    controller = RobustDelayBudgetController(
        CLOCKS, oracle_mode=True, oracle_table_path=tmp_path / "missing.csv"
    )
    assert controller.decide(SAFE, 3, min_dwell=0) == 2
    assert "using robust policy" in caplog.text


def test_oracle_without_table_remains_usable():
    controller = get_controller("oracle", CLOCKS, ExperimentConfig())
    assert controller.decide(SAFE, 3, min_dwell=0) == 2
