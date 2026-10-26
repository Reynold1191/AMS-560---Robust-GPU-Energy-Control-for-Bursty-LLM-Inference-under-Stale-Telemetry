"""Member 4 tests (load_generator.py, scale_simulator.py) -- from TEAM_PLAN.md.

Run from repo root:  python -m pytest tests/test_member4.py -v
"""
import os
import sys
import time

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ExperimentConfig
from request_queue import RequestQueue
from load_generator import LoadGenerator, workload_rate


# ---------------------------------------------------------------- load_generator

def test_steady_rate():
    cfg = ExperimentConfig(base_arrival_rate=2.0, duration_s=3.0, warmup_s=0)
    q = RequestQueue()
    gen = LoadGenerator(q, cfg, workload="steady", seed=42)
    gen.start()
    time.sleep(3.5)
    gen.stop()
    n = q.total_enqueued
    assert 3 <= n <= 10, f"Expected 3-10, got {n}"


def test_burst_workload():
    cfg = ExperimentConfig(base_arrival_rate=0.5, duration_s=5.0, warmup_s=0,
                           burst_start_s=2.0, burst_duration_s=2.0, burst_multiplier=5.0)
    q = RequestQueue()
    gen = LoadGenerator(q, cfg, workload="sharp_burst", seed=42)
    gen.start()
    time.sleep(5.5)
    gen.stop()
    assert q.total_enqueued >= 5


def test_phase_tags():
    cfg = ExperimentConfig(base_arrival_rate=5.0, duration_s=8.0, warmup_s=2.0,
                           burst_start_s=4.0, burst_duration_s=2.0, burst_multiplier=3.0)
    q = RequestQueue()
    gen = LoadGenerator(q, cfg, workload="sharp_burst", seed=42)
    gen.start()
    time.sleep(8.5)
    gen.stop()
    phases = set()
    while not q.empty():
        r = q.get(timeout=0.1)
        if r:
            phases.add(r.phase)
    assert {"warmup", "steady", "burst"} <= phases, phases


def test_rate_profiles():
    kw = dict(base_rate=0.15, burst_multiplier=5.0, burst_start_s=40, burst_duration_s=20,
              ramp_end_rate=2.0, duration_s=120)
    assert workload_rate("steady", 50, **kw) == 0.15
    assert workload_rate("sharp_burst", 50, **kw) == pytest.approx(0.75)
    assert workload_rate("sharp_burst", 70, **kw) == 0.15
    assert workload_rate("ramp", 0, **kw) == pytest.approx(0.15)
    assert workload_rate("ramp", 120, **kw) == pytest.approx(2.0)
    assert workload_rate("ramp", 500, **kw) == pytest.approx(2.0)  # clamped
    with pytest.raises(ValueError):
        workload_rate("bogus", 0, **kw)


# ---------------------------------------------------------------- scale_simulator

SWEEP = pd.DataFrame({"clock_mhz": [667, 1147, 2602], "mean_lat": [8.0, 5.0, 3.0],
                      "workload": ["steady"] * 3})


def test_fit_service_times():
    from scale_simulator import fit_service_times
    times = fit_service_times(SWEEP)
    assert times[667] == 8.0
    assert times[1147] == 5.0


def test_simulation_runs():
    from scale_simulator import simulate
    from controllers import get_controller
    cfg = ExperimentConfig()
    ctrl = get_controller("naive_slo", [667, 1147, 2602], cfg)
    result = simulate(n_workers=4, n_requests=50, controller=ctrl, delay_ms=0,
                      workload="steady", sweep_df=SWEEP, slo_target_s=8.5,
                      base_rate=0.5, burst_mult=1.0, seed=42)
    for k in ("controller", "delay_ms", "workload", "miss_rate", "p95_latency_s",
              "total_j", "j_per_token", "n_requests"):
        assert k in result
    assert 0.0 <= result["miss_rate"] <= 1.0
    assert result["n_requests"] == 50


def test_simulation_deterministic():
    from scale_simulator import simulate
    from controllers import get_controller
    cfg = ExperimentConfig()
    args = dict(n_workers=8, n_requests=500, delay_ms=250, workload="sharp_burst",
                sweep_df=SWEEP, slo_target_s=8.5, base_rate=1.2, burst_mult=5.0, seed=7)
    r1 = simulate(controller=get_controller("robust", [667, 1147, 2602], cfg), **args)
    r2 = simulate(controller=get_controller("robust", [667, 1147, 2602], cfg), **args)
    assert r1 == r2
