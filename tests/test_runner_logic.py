"""GPU-free tests for the experiment runner's timing, delay and calibration logic."""

import csv
import math
import dataclasses
import os
import time

import pytest

from config import DELAY_VALUES_MS, VERIFIED_CLOCKS_MHZ, ExperimentConfig
from controllers import get_controller
from delay_buffer import DelayBuffer
from observation import (
    ControlObservation,
    arrival_rate_at,
    clock_tracking_pct,
    delayed_state,
    phase_at,
    starting_index,
    validate_timeline,
)
from telemetry import GpuSample


def _cfg(**kw):
    return dataclasses.replace(ExperimentConfig(), **kw)


# ── Timeline ──────────────────────────────────────────────────────────────

def test_default_timeline_has_burst_and_recovery():
    c = _cfg(workload="sharp_burst")
    validate_timeline(c)
    assert phase_at(-1, c) == "warmup"
    assert phase_at(c.burst_start_s - 0.1, c) == "steady"
    assert phase_at(c.burst_start_s, c) == "burst"
    assert phase_at(c.burst_start_s + c.burst_duration_s, c) == "recovery"
    assert arrival_rate_at(c.burst_start_s + 1, c) == pytest.approx(
        c.base_arrival_rate * c.burst_multiplier)
    assert arrival_rate_at(None, c) == c.base_arrival_rate


def test_old_demo_timeline_is_rejected():
    # The old demo (60 s total, 5 s warm-up, burst at 40 s for 20 s) cut the
    # burst short and left no recovery period.
    with pytest.raises(ValueError):
        validate_timeline(_cfg(workload="sharp_burst", duration_s=60, warmup_s=5))


def test_ramp_reaches_end_rate_at_end_of_measurement():
    c = _cfg(workload="ramp")
    m = c.duration_s - c.warmup_s
    assert arrival_rate_at(m, c) == pytest.approx(c.base_arrival_rate * c.ramp_end_multiplier)
    assert arrival_rate_at(0, c) == pytest.approx(c.base_arrival_rate)
    # The ramp goes UP from whatever base rate the benchmark chose.
    assert arrival_rate_at(m / 2, c) > c.base_arrival_rate
    c2 = _cfg(workload="ramp", ramp_end_rate=2.0)
    assert arrival_rate_at(m, c2) == pytest.approx(2.0)


# ── Starting clock ────────────────────────────────────────────────────────

def test_starting_index_static_vs_dynamic():
    clocks = list(VERIFIED_CLOCKS_MHZ)
    c = _cfg(static_efficient_mhz=667)
    assert clocks[starting_index(get_controller("static_efficient", clocks, c), clocks)] == 667
    assert starting_index(get_controller("static_high", clocks, c), clocks) == len(clocks) - 1
    assert starting_index(get_controller("robust", clocks, c), clocks) == len(clocks) // 2
    # Sweep: single-clock ladder, static_high locks exactly that clock.
    assert starting_index(get_controller("static_high", [1147], c), [1147]) == 0


# ── Delay applies to every controller input ──────────────────────────────

def _gpu(power):
    return GpuSample(t=time.monotonic(), power_w=power, graphics_mhz=1147, sm_mhz=1147,
                     util_pct=90, temp_c=60, phase="steady")


def _obs(q, p95, burn):
    return ControlObservation(t=time.monotonic(), queue_len=q, arrival_rate=0.3,
                              arrival_cv=1.0, p95_latency_s=p95, p99_latency_s=p95,
                              error_budget_burn=burn, miss_rate=0.0)


def test_queue_p95_and_burn_are_delayed_not_just_power():
    gpu, obs = DelayBuffer(), DelayBuffer()
    gpu.push(_gpu(20.0))
    obs.push(_obs(q=0, p95=1.0, burn=0.0))      # calm, before the burst
    time.sleep(0.35)
    gpu.push(_gpu(70.0))
    obs.push(_obs(q=9, p95=12.0, burn=4.0))     # burst has arrived

    fresh, _, age0 = delayed_state(gpu, obs, 0.0, 8.0)
    assert (fresh.queue_len, fresh.p95_latency_s, fresh.error_budget_burn) == (9, 12.0, 4.0)
    assert age0 < 0.1

    stale, _, age = delayed_state(gpu, obs, 0.3, 8.0)
    assert (stale.queue_len, stale.p95_latency_s, stale.error_budget_burn) == (0, 1.0, 0.0)
    assert stale.power_w == 20.0
    assert stale.telemetry_age_ms == pytest.approx(age * 1000) and age >= 0.3


def test_naive_controller_reacts_to_delay():
    """With delayed inputs, naive_slo makes a different call on stale data."""
    clocks = list(VERIFIED_CLOCKS_MHZ)
    gpu, obs = DelayBuffer(), DelayBuffer()
    gpu.push(_gpu(20.0)); obs.push(_obs(q=0, p95=1.0, burn=0.0))
    time.sleep(0.35)
    gpu.push(_gpu(70.0)); obs.push(_obs(q=9, p95=12.0, burn=4.0))
    c = _cfg(slo_target_s=8.0)
    fresh, _, _ = delayed_state(gpu, obs, 0.0, 8.0)
    stale, _, _ = delayed_state(gpu, obs, 0.3, 8.0)
    mid = len(clocks) // 2
    assert get_controller("naive_slo", clocks, c).decide(fresh, mid, min_dwell=0) > mid
    assert get_controller("naive_slo", clocks, c).decide(stale, mid, min_dwell=0) < mid


def test_delay_levels_are_the_measured_ones():
    assert DELAY_VALUES_MS == [0, 500, 1250, 2500]


# ── Clock tracking ────────────────────────────────────────────────────────

def _s(t, mhz, util=90):
    return GpuSample(t=t, power_w=50, graphics_mhz=mhz, sm_mhz=mhz, util_pct=util,
                     temp_c=60, phase="steady")


def test_clock_tracking_detects_unlocked_clock():
    cmds = [(0.0, 1635)]
    good = [_s(t, 1627) for t in (2, 3, 4)]
    bad = [_s(t, 1380) for t in (2, 3, 4)]   # what the old "1635 MHz" sweep showed
    assert clock_tracking_pct(good, cmds) == 100.0
    assert clock_tracking_pct(bad, cmds) == 0.0
    # samples during settling or idle are ignored
    assert clock_tracking_pct([_s(0.5, 180), _s(3, 180, util=0)], cmds) != \
        clock_tracking_pct([_s(0.5, 180), _s(3, 180, util=0)], cmds)  # NaN


# ── Load generator follows the shared origin ─────────────────────────────

def test_load_generator_stops_at_end_of_measurement_and_tags_phases():
    from load_generator import LoadGenerator
    from request_queue import RequestQueue

    c = _cfg(workload="steady", base_arrival_rate=40.0, duration_s=0.6, warmup_s=0.1)
    rq = RequestQueue()
    gen = LoadGenerator(rq, c, [{"prompt": "x", "max_new_tokens": 4}])
    gen.start()
    time.sleep(0.15)
    gen.set_origin(time.monotonic())
    time.sleep(1.0)
    assert not gen._thread.is_alive()
    phases = []
    while True:
        r = rq.get(block=False)
        if r is None:
            break
        phases.append(r.phase)
    assert "steady" in phases
    assert set(phases) <= {"warmup", "steady"}


# ── Sweep aggregation and SLO calibration ────────────────────────────────

def _sweep_row(mhz, seed, p95, svc, jtok, util=80.0, track=99.0):
    return {"run_id": f"sweep_{mhz}mhz_s{seed}", "clock_mhz": mhz, "workload": "sharp_burst",
            "p95": p95, "p50": p95 / 2, "p99": p95, "mean_lat": p95 / 2, "miss_rate": 0.0,
            "j_per_token": jtok, "mean_service_s": svc, "mean_power_w": 40.0,
            "mean_util_pct": util, "clock_tracking_pct": track, "total_j": 1.0,
            "output_tokens": 100, "n_requests": 40, "n_unfinished": 0}


def test_aggregate_sweep_means_over_seeds():
    from run_experiment import aggregate_sweep
    rows = [_sweep_row(180, 41, 20.0, 8.0, 0.5), _sweep_row(180, 42, 22.0, 8.0, 0.7),
            _sweep_row(2122, 41, 4.0, 2.0, 0.9)]
    agg = {r["clock_mhz"]: r for r in aggregate_sweep(rows)}
    assert agg[180]["n_seeds"] == 2 and agg[180]["p95"] == pytest.approx(21.0)
    assert agg[180]["p95_std"] > 0 and agg[2122]["p95_std"] == 0.0


def _write_requests(d, run_id, lats):
    os.makedirs(os.path.join(d, "sweep"), exist_ok=True)
    with open(os.path.join(d, "sweep", f"{run_id}_requests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "latency_s"])
        for i, x in enumerate(lats):
            w.writerow([i, x])


def test_calibration_picks_feasible_slo_and_efficient_clock(tmp_path):
    from calibrate_slo import calibrate
    d = str(tmp_path)
    rows = []
    spec = {180: (30.0, 10.0, 0.40), 1147: (7.0, 3.0, 0.45), 2122: (4.0, 2.0, 0.80)}
    for mhz, (p95, svc, jtok) in spec.items():
        for seed in (41, 42):
            r = _sweep_row(mhz, seed, p95, svc, jtok)
            rows.append(r)
            _write_requests(d, r["run_id"], [p95 * 0.5] * 38 + [p95] * 2)
    res = calibrate(rows, d, budget_ratio=0.05)
    assert res["slo_target_s"] == 6.0           # ceil_0.5(1.5 x 4.0)
    assert res["feasible"] and res["ok"]
    # 1147 MHz misses exactly 5% at 6.0 s (within budget) and is cheaper than 2122
    assert res["static_efficient_mhz"] == 1147
    by = {c["clock_mhz"]: c for c in res["per_clock"]}
    assert by[180]["miss_rate_at_slo"] == 1.0
    assert res["clock_sensitivity"] == pytest.approx(5.0)


def test_calibration_blocks_when_clock_does_not_matter(tmp_path):
    from calibrate_slo import calibrate
    d = str(tmp_path)
    rows = []
    for mhz in (180, 2122):   # service time flat across clocks (the old testbed)
        r = _sweep_row(mhz, 42, 6.0, 3.3, 0.5, util=20.0)
        rows.append(r)
        _write_requests(d, r["run_id"], [3.0] * 40)
    res = calibrate(rows, d, budget_ratio=0.05)
    assert not res["ok"]
    assert any("barely depends on the clock" in b for b in res["blocking"])


# ── Proposal run: 24 primary conditions (+3 static_high) x 3 paired seeds ─

def test_primary_only_matrix_is_24_conditions_per_paired_seed():
    from run_demo import build_matrix
    seeds = [41, 42, 43]
    runs = build_matrix(seeds, quick=False, primary_only=True)
    assert len(runs) == 105                     # (24 + 3 static_high + 8 ablation) x 3
    assert len(build_matrix(seeds, False, True, ablation=False)) == 81
    abl = [r for r in runs if r[1] in ("robust_v1", "robust_no_age")]
    assert len(abl) == 24 and {w for *_, w in abl} == {"sharp_burst"}
    dynamic = [r for r in runs if r[1] in ("naive_slo", "robust")]
    assert {c for _, c, _, _ in dynamic} == {"naive_slo", "robust"}
    assert {w for _, _, _, w in dynamic} == {"steady", "ramp", "sharp_burst"}
    assert {d for _, _, d, _ in dynamic} == set(DELAY_VALUES_MS)
    per_seed = {s: {(c, d, w) for s2, c, d, w in dynamic if s2 == s} for s in seeds}
    assert all(len(v) == 24 for v in per_seed.values())
    assert per_seed[41] == per_seed[42] == per_seed[43]   # paired
    refs = {(c, w) for _, c, _, w in runs if c == "static_high"}
    assert refs == {("static_high", w) for w in ("steady", "ramp", "sharp_burst")}


def test_primary_only_one_seed_is_24_runs():
    from run_demo import build_matrix, parse_args
    a = parse_args(["--primary-only", "--seeds", "42"])
    assert a.seeds == [42] and a.sweep_seeds == [42]
    assert len(build_matrix(a.seeds, a.quick, a.primary_only)) == 35
    assert a.slo_target == 5.0 and not a.calibrated_slo
    with pytest.raises(SystemExit):
        parse_args(["--primary-only", "--quick"])
    a = parse_args(["--primary-only"])
    assert a.seeds == [41, 42, 43] and a.sweep_seeds == [41, 42, 43]


def test_paired_summary_flags_missing_seeds(tmp_path):
    import pandas as pd
    from run_demo import build_matrix, summarize_paired
    rows = [{"run_id": f"{c}_d{d}_{w}_s{s}", "seed": s, "controller": c, "delay_ms": d,
             "workload": w, "j_per_token": 0.5, "miss_rate": 0.02, "p95": 6.0}
            for s, c, d, w in build_matrix([41, 42, 43], False, True)]
    rows = rows[1:]   # drop one run
    pd.DataFrame(rows).to_csv(tmp_path / "run_summary.csv", index=False)
    out = pd.read_csv(summarize_paired(str(tmp_path), [41, 42, 43], False, True))
    assert len(out) == 35
    assert (out["n_seeds"] == 3).sum() == 34


# ── GPU check verdict ─────────────────────────────────────────────────────

def test_power_capped_clocks_do_not_fail_the_gpu_check():
    from check_gpu_timing import summarize_tracking
    measured = {180: (180, 13.6), 667: (660, 22.7), 1147: (1140, 31.9), 1635: (1627, 47.3),
                2122: (2115, 70.0), 2602: (2138, 80.5), 3090: (2298, 79.9)}
    rows = [{"target_mhz": m, "repeat": rep, "lock_rc": 0, "observed_mhz": o,
             "power_w": p, "error_mhz": o - m}
            for rep in (0, 1) for m, (o, p) in measured.items()]
    r = summarize_tracking(rows, list(measured))
    assert r["usable_clocks"] == [180, 667, 1147, 1635, 2122]
    assert all(v.startswith("CAPPED") for v in r["excluded_clocks"].values())
    assert r["clock_control_reliable"]
    # A mid-range clock that misses its target is still a real failure.
    bad = [dict(x, observed_mhz=1380) if x["target_mhz"] == 1635 else x for x in rows]
    assert not summarize_tracking(bad, list(measured))["clock_control_reliable"]


# ── Workload fixes: token counting, benchmark choice, long prompts ───────

def test_count_generated_stops_at_first_stop_token_and_caps():
    from llm_worker import count_generated
    assert count_generated([5, 6, 2, 2, 2], {2}, cap=32) == 3      # through first EOS
    assert count_generated([5, 6, 7, 8], {2}, cap=3) == 3          # own budget
    assert count_generated([5, 6, 7], {2}, cap=32) == 3            # never stopped


def test_benchmark_picks_most_clock_sensitive_usable_setup():
    from bench_clock_sensitivity import pick
    cands = [
        {"model": "m", "batch_size": 4, "sensitivity": 1.2, "service_top_s": 1.0},   # flat
        {"model": "m", "batch_size": 8, "sensitivity": 6.0, "service_top_s": 1.4},
        {"model": "m", "batch_size": 16, "sensitivity": 9.0, "service_top_s": 12.0},  # too slow
    ]
    c = pick(cands, max_service_s=8.0)
    assert c["batch_size"] == 8
    assert c["base_rate"] == pytest.approx(1.70)   # floor_0.05(0.3 * 8 / 1.4)
    assert pick(cands[:1], 8.0) is None


def test_long_context_prompt_file_is_long_prompt_short_answer():
    import json
    recs = [json.loads(l) for l in open("prompts/long_context.jsonl", encoding="utf-8")]
    assert len(recs) == 30
    assert min(r["prompt_tokens"] for r in recs) >= 700
    assert max(r["max_new_tokens"] for r in recs) <= 32
    assert ExperimentConfig().prompts_file == "prompts/long_context.jsonl"


# ── attempt_6 fixes: ramp generation, capacity floor, delay-aware robust ──

def test_ramp_generator_keeps_producing_requests():
    # attempt_6: redrawing the gap on every rate change starved the ramp
    # (~5 requests per run). Expect roughly the Poisson mean here.
    from load_generator import LoadGenerator
    from request_queue import RequestQueue
    c = _cfg(workload="ramp", base_arrival_rate=10.0, ramp_end_multiplier=2.0,
             duration_s=3.0, warmup_s=0.0, seed=1)
    gen = LoadGenerator(RequestQueue(), c, [{"prompt": "x", "max_new_tokens": 4}])
    gen.set_origin(time.monotonic())
    gen.start()
    time.sleep(3.2)
    gen.stop()
    assert gen._request_counter >= 25  # mean 45 over 3 s ramping 10 -> 20 req/s


def test_capacity_model_fits_benchmark_points():
    from capacity_model import build_capacity
    bench = {"chosen": {"batch_size": 16, "per_clock": {
        "180": {"batch_time_s": 18.08}, "1147": {"batch_time_s": 3.25},
        "2122": {"batch_time_s": 2.01}}}}
    cap = build_capacity(bench, VERIFIED_CLOCKS_MHZ)
    t = dict(zip(cap["clocks_mhz"], cap["batch_time_s"]))
    assert t[180] == pytest.approx(18.08, rel=0.03)
    assert t[2122] == pytest.approx(2.01, rel=0.03)
    assert cap["capacity_rps"] == sorted(cap["capacity_rps"])


def _state(**kw):
    from controllers.base import ControllerState
    d = dict(t=100.0, slo_target_s=8.0, queue_len=0, p95_latency_s=1.0,
             error_budget_burn=0.0, telemetry_age_ms=0.0, arrival_cv=0.0,
             arrival_rate=2.35)
    d.update(kw)
    return ControllerState(**d)


CAP = [0.88, 3.05, 4.90, 6.56, 8.01]   # attempt_6 fit, req/s per verified clock
BATCH = [18.08, 5.25, 3.26, 2.44, 2.00]


@pytest.mark.parametrize("name", ["naive_slo", "robust", "robust_no_age"])
def test_capacity_floor_blocks_infeasible_clock(name):
    c = _cfg(capacity_rps=CAP, batch_time_s=BATCH)
    ctrl = get_controller(name, VERIFIED_CLOCKS_MHZ, c)
    # Empty queue and low P95 would step down 667 -> 180, which serves 0.88
    # req/s against 2.35 arriving (the attempt_6 start-up trap).
    assert ctrl.decide(_state(), 1, min_dwell=0) == 1
    assert "CAPACITY_FLOOR" in ctrl.last_reason
    # A burst seen in the arrival rate lifts the clock immediately.
    assert ctrl.decide(_state(t=101.0, arrival_rate=9.4), 1, min_dwell=0) == 4


def test_static_controllers_have_no_floor():
    c = _cfg(capacity_rps=CAP, batch_time_s=BATCH, static_efficient_mhz=180)
    ctrl = get_controller("static_efficient", VERIFIED_CLOCKS_MHZ, c)
    assert ctrl.capacity_rps is None
    assert ctrl.decide(_state(arrival_rate=9.4), 0) == 0


def test_robust_waits_for_effect_of_last_change_scaled_by_age():
    c = _cfg(capacity_rps=CAP, batch_time_s=BATCH)
    ctrl = get_controller("robust", VERIFIED_CLOCKS_MHZ, c)
    ctrl.last_change_t = 100.0
    # 2122 MHz (idx 4): wait = 2.5 s age + 0.7 s settling + 2.0 s batch = 5.2 s
    calm = dict(p95_latency_s=1.0, arrival_rate=1.0, telemetry_age_ms=2500.0)
    assert ctrl._policy(_state(t=104.0, **calm), 4) == 4
    assert "WAIT_EFFECT" in ctrl.last_reason
    assert ctrl._policy(_state(t=105.5, **calm), 4) == 3
    # Fresh telemetry: only settling + batch (2.7 s).
    assert ctrl._policy(_state(t=103.0, p95_latency_s=1.0, arrival_rate=1.0), 4) == 3


def test_robust_is_not_frozen_by_injected_delay():
    # attempt_6: the 300 ms hard guard blocked every down-step at >= 500 ms.
    c = _cfg()
    ctrl = get_controller("robust", VERIFIED_CLOCKS_MHZ, c)
    for age in (500.0, 1250.0, 2500.0):
        assert ctrl._policy(_state(telemetry_age_ms=age), 3) == 2, ctrl.last_reason


def test_robust_needs_more_headroom_when_telemetry_is_old():
    ctrl = get_controller("robust", VERIFIED_CLOCKS_MHZ, _cfg())
    s = dict(p95_latency_s=5.0)  # 62 % of an 8 s SLO
    assert ctrl._policy(_state(telemetry_age_ms=0.0, **s), 3) == 2
    assert ctrl._policy(_state(telemetry_age_ms=2500.0, **s), 3) == 3


# ── attempt_7 fixes: scheduled arrivals, in-stack capacity, load rule ────

def test_scheduled_arrivals_survive_a_busy_cpu():
    # attempt_7: sleeping each gap off under GIL contention delivered ~78 %
    # of the planned burst rate. Arrivals now come from a fixed schedule.
    import threading
    from load_generator import LoadGenerator
    from request_queue import RequestQueue
    c = _cfg(workload="steady", base_arrival_rate=40.0, duration_s=2.0, warmup_s=0.0, seed=3)
    rq = RequestQueue()
    gen = LoadGenerator(rq, c, [{"prompt": "x", "max_new_tokens": 4}])
    stop = threading.Event()

    def hog():
        while not stop.is_set():
            sum(i * i for i in range(20000))

    threads = [threading.Thread(target=hog, daemon=True) for _ in range(3)]
    for t in threads:
        t.start()
    gen.set_origin(time.monotonic())
    gen.start()
    time.sleep(2.3)
    gen.stop()
    stop.set()
    assert gen._request_counter >= 64           # mean 80 over 2 s
    reqs = []
    while (r := rq.get(block=False)) is not None:
        reqs.append(r)
    t = [r.arrival_t for r in reqs]
    assert t == sorted(t)                       # stamped with scheduled times


def test_arrival_schedule_matches_burst_rate():
    from load_generator import LoadGenerator
    from request_queue import RequestQueue
    c = _cfg(workload="sharp_burst", base_arrival_rate=2.0, burst_multiplier=4.0,
             burst_start_s=40.0, burst_duration_s=20.0, seed=5)
    gen = LoadGenerator(RequestQueue(), c, [{"prompt": "x"}])
    gen._origin = 0.0
    t, n_burst, n_base = 0.0, 0, 0
    while t < 110.0:
        t = gen._next_arrival(t)
        if 40.0 <= t < 60.0:
            n_burst += 1
        elif t < 110.0:
            n_base += 1
    assert 160 * 0.8 < n_burst < 160 * 1.2      # 8 req/s x 20 s
    assert 180 * 0.8 < n_base < 180 * 1.2       # 2 req/s x 90 s


def test_capacity_prefers_in_stack_measurements(tmp_path):
    import csv as _csv
    from capacity_model import build_capacity, measured_points
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    for mhz, svc in ((1147, 3.2), (2122, 2.7)):
        with open(sweep / f"sweep_{mhz}mhz_s42_requests.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["request_id", "start_t", "service_time_s", "phase"])
            rid = 0
            for b in range(4):
                for _ in range(16):
                    w.writerow([rid, 100.0 + 10 * b, svc, "steady"])
                    rid += 1
            w.writerow([rid, 999.0, 1.0, "steady"])     # partial batch: ignored
    pts = measured_points(16, str(sweep))
    assert pts == {1147: 3.2, 2122: 2.7}
    bench = {"chosen": {"batch_size": 16, "per_clock": {
        "180": {"batch_time_s": 18.08}, "2122": {"batch_time_s": 2.01}}}}
    cap = build_capacity(bench, VERIFIED_CLOCKS_MHZ, pts)
    by = dict(zip(cap["clocks_mhz"], cap["batch_time_s"]))
    assert by[2122] == pytest.approx(2.7)       # measured, not the 2.0 s benchmark
    assert cap["source"].startswith("in-stack")


def test_burst_peak_is_set_below_measured_top_capacity():
    from run_demo import BURST_PEAK_FRAC, burst_base_rate
    base = burst_base_rate(5.9, 4.0)
    assert base == pytest.approx(1.3)
    assert base * 4.0 <= BURST_PEAK_FRAC * 5.9 + 1e-9


def test_smoke_mode_is_small_and_reuses_latest_sweep(tmp_path):
    from run_demo import SMOKE_MATRIX, build_matrix, latest_sweep_dir
    runs = build_matrix([42], False, False, smoke=True)
    assert len(runs) == len(SMOKE_MATRIX) == 6
    assert {w for *_, w in runs} == {"sharp_burst"}
    for n in (3, 12):
        d = tmp_path / f"attempt_{n}"
        d.mkdir()
        for f in ("sweep_summary.csv", "sweep_runs.csv", "clock_sensitivity.json"):
            (d / f).write_text("x")
    (tmp_path / "attempt_20").mkdir()          # no sweep: skipped
    assert latest_sweep_dir(str(tmp_path)).endswith("attempt_12")


def test_repeat_mode_plans_consecutive_attempts(tmp_path):
    from run_demo import parse_args, repeat_plan
    for n in (1, 2, 3):
        (tmp_path / f"attempt_{n}").mkdir()
    a = parse_args(["--repeat-1seed", "2", "--repeat-3seed", "2", "--skip-gpu-check"])
    plan = [(os.path.basename(d), s) for d, s in repeat_plan(a, str(tmp_path))]
    assert plan == [("attempt_4", [42]), ("attempt_5", [42]),
                    ("attempt_6", [41, 42, 43]), ("attempt_7", [41, 42, 43])]
    with pytest.raises(SystemExit):
        parse_args(["--repeat-1seed", "1", "--sweep-from", "results/attempt_7"])


# ── After attempt_8-11: age-compensated forecast, lag-aware P95, sim ─────

def _robust_v2(**kw):
    from controllers.robust_delay_budget import RobustDelayBudgetController
    c = RobustDelayBudgetController(list(VERIFIED_CLOCKS_MHZ), slo_target_s=5.0, **kw)
    c.set_capacity([0.67, 3.10, 4.95, 6.61, 7.94], [23.7, 5.16, 3.23, 2.42, 2.02])
    return c


def _feed(ctrl, idx, states):
    for st in states:
        idx = ctrl.decide(st, idx, min_dwell=2)
    return idx


def test_forecast_steps_up_on_a_stale_surge_before_the_floor_does():
    from controllers.base import ControllerState
    from controllers.naive_slo import NaiveSLOController
    # attempt_8-11: every up-step came from the shared capacity floor on a
    # 10 s arrival rate that lags the burst. The queue grows first.
    seq = [ControllerState(t=100.0 + i, telemetry_age_ms=2500.0, queue_len=q,
                           arrival_rate=r, p95_latency_s=2.5, slo_target_s=5.0)
           for i, (q, r) in enumerate([(1, 1.8), (2, 1.9), (5, 2.1), (9, 2.4)])]
    robust = _robust_v2()
    naive = NaiveSLOController(list(VERIFIED_CLOCKS_MHZ), slo_target_s=5.0)
    naive.set_capacity(robust.capacity_rps, robust.batch_time_s)
    assert _feed(naive, 1, seq) == 1                 # floor still happy at 667 MHz
    assert _feed(robust, 1, seq) >= 3                # robust already at >= 1635 MHz
    assert "FORECAST_UP" in robust.last_reason


def test_no_age_ablation_has_a_shorter_forecast_horizon():
    from controllers.base import ControllerState
    st = ControllerState(t=10.0, telemetry_age_ms=2500.0)
    assert _robust_v2()._horizon_s(st) == pytest.approx(3.2)
    assert _robust_v2(use_telemetry_age=False)._horizon_s(st) == pytest.approx(0.7)


def test_stale_p95_after_a_burst_does_not_pin_the_top_clock():
    from controllers.base import ControllerState
    # After the burst the 60 s P95 window stays above the SLO for up to a
    # minute; naive and robust_v1 held 2122 MHz to the end of every run.
    ctrl = _robust_v2()
    seq = [ControllerState(t=200.0 + i, telemetry_age_ms=2500.0, queue_len=0,
                           arrival_rate=1.8, p95_latency_s=6.5, error_budget_burn=3.0,
                           slo_target_s=5.0) for i in range(12)]
    assert _feed(ctrl, 4, seq) < 4
    assert any(k in ctrl.last_reason for k in ("CAPACITY_DOWN", "WAIT_EFFECT", "HOLD"))
    # A real backlog still escalates.
    hot = ControllerState(t=300.0, telemetry_age_ms=2500.0, queue_len=20, arrival_rate=6.0,
                          p95_latency_s=6.5, slo_target_s=5.0)
    assert ctrl.decide(hot, 2, min_dwell=2) == 4


def test_robust_v1_is_the_old_controller():
    from controllers import get_controller
    c = _cfg(capacity_rps=[0.67, 3.10, 4.95, 6.61, 7.94],
             batch_time_s=[23.7, 5.16, 3.23, 2.42, 2.02])
    v1 = get_controller("robust_v1", list(VERIFIED_CLOCKS_MHZ), c)
    assert not v1.forecast and not v1.lag_aware and v1.capacity_rps is not None


def test_serving_sim_reproduces_the_delay_effect_and_robust_beats_naive():
    import statistics
    from serving_sim import SimModel, make_cfg, simulate_run
    m = SimModel.from_results(None)
    res = {}
    for c in ("naive_slo", "robust"):
        for d in (0, 2500):
            runs = [simulate_run(make_cfg(c, d, "sharp_burst", s, 5.0), m) for s in range(6)]
            res[c, d] = (statistics.mean(r["miss_rate"] for r in runs),
                         statistics.mean(r["j_per_token"] for r in runs))
    assert res["naive_slo", 2500][0] > res["naive_slo", 0][0]      # stale telemetry hurts
    assert res["robust", 2500][0] < res["naive_slo", 2500][0]      # robust misses less
    assert res["robust", 2500][1] < res["naive_slo", 2500][1]      # and uses less energy
