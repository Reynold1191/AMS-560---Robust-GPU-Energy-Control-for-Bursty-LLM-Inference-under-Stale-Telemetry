"""
scale_simulator.py -- Member 4 (Scale-Out Simulation, Deliverable D10)

Trace-driven, CPU-only simulation of N virtual GPU workers. Service times per
clock come from the fixed-frequency sweep (`sweep_summary.csv`, Member 5), and
the same controllers as the real experiment make a decision every control
interval. Answers: what would latency / energy look like at N-x scale?

Model (kept deliberately simple, see README notes):
  * Arrivals   : Poisson, same profiles as LoadGenerator (steady/ramp/sharp_burst)
  * Workers    : n_workers, one request at a time each, FIFO dispatch to the
                 worker that is free soonest
  * Service    : Normal(mean_s(clock), 0.1*mean_s) + Uniform(0, 50 ms) overhead,
                 mean_s(clock) interpolated from the sweep
  * Clock      : one shared clock for all workers, chosen by the controller
                 every `control_interval_s`; a request's service time uses the
                 clock at the moment it is dispatched
  * Telemetry  : GPU fields (power/util/clock) the controller sees are
                 `delay_ms` old and telemetry_age_ms = delay_ms. Queue length and
                 latency stats are fresh, matching run_experiment.py. Pass
                 delay_all_signals=True to also delay queue/latency signals.
  * Energy     : per-worker power P(clock) = linear fit from the sweep (or a
                 default model); idle workers draw P_idle.

Usage:
  python scale_simulator.py --n-workers 8 --n-requests 10000 \
      --controller robust --delay-ms 250 --workload sharp_burst \
      --sweep-csv results/sweep_summary.csv --results-dir results
"""

import argparse
import bisect
import csv
import heapq
import math
import os
import random
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from load_generator import load_prompts, workload_rate

# Default power model if the sweep has no power column:
# P(clock) = DEFAULT_P_IDLE_W + DEFAULT_W_PER_MHZ * clock  (~100 W at 3 GHz)
DEFAULT_P_IDLE_W = 10.0
DEFAULT_W_PER_MHZ = 0.03
SERVICE_OVERHEAD_MAX_S = 0.05


# ---------------------------------------------------------------------------
# Sweep fitting
# ---------------------------------------------------------------------------

def fit_service_times(sweep_df: pd.DataFrame) -> Dict[int, float]:
    """Map clock_mhz -> mean service time (s). Uses mean_lat, else p95/2."""
    if "clock_mhz" not in sweep_df.columns:
        raise ValueError("sweep_df must contain a 'clock_mhz' column")
    if "mean_lat" in sweep_df.columns:
        col = sweep_df["mean_lat"]
    elif "p95" in sweep_df.columns:
        col = sweep_df["p95"] / 2.0
    else:
        raise ValueError("sweep_df needs 'mean_lat' or 'p95' to estimate service time")
    grouped = col.groupby(sweep_df["clock_mhz"]).mean()
    return {int(clk): float(v) for clk, v in grouped.items()}


def fit_power_model(sweep_df: pd.DataFrame):
    """Return (p_idle_w, slope_w_per_mhz, intercept_w) for P(clock)."""
    for col in ("mean_power_w", "avg_power_w", "power_w"):
        if col in sweep_df.columns:
            g = sweep_df.groupby("clock_mhz")[col].mean()
            if len(g) >= 2:
                slope, intercept = np.polyfit(g.index.values.astype(float), g.values, 1)
                p_idle = max(float(g.min()) * 0.3, 1.0)
                return p_idle, float(slope), float(intercept)
    return DEFAULT_P_IDLE_W, DEFAULT_W_PER_MHZ, DEFAULT_P_IDLE_W


def _interp_service(service_map: Dict[int, float], clock: int) -> float:
    xs = sorted(service_map)
    ys = [service_map[x] for x in xs]
    return float(np.interp(clock, xs, ys))


def _percentile(sorted_vals: List[float], p: float) -> float:
    """Same index rule as SLOBudget.percentile_latency()."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    return sorted_vals[min(int(p * n), n - 1)]


# ---------------------------------------------------------------------------
# Arrivals
# ---------------------------------------------------------------------------

def generate_arrivals(n_requests: int, workload: str, base_rate: float, burst_mult: float,
                      rng: random.Random, burst_start_s: float = 40.0,
                      burst_duration_s: float = 20.0, ramp_end_rate: float = 2.0,
                      duration_s: float = 120.0) -> List[float]:
    """N arrival times with the same rate profile as LoadGenerator."""
    t, out = 0.0, []
    while len(out) < n_requests:
        rate = workload_rate(workload, t, base_rate, burst_mult, burst_start_s,
                             burst_duration_s, ramp_end_rate, duration_s)
        t += rng.expovariate(rate) if rate > 0 else 1.0
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate(n_workers: int, n_requests: int, controller, delay_ms: float, workload: str,
             sweep_df: pd.DataFrame, slo_target_s: float, base_rate: float,
             burst_mult: float, seed: int = 42, *,
             control_interval_s: float = 1.0, min_dwell: int = 2,
             slo_budget_ratio: float = 0.01, budget_window_s: float = 60.0,
             burst_start_s: float = 40.0, burst_duration_s: float = 20.0,
             ramp_end_rate: float = 2.0, duration_s: float = 120.0,
             tokens_per_request: Optional[List[int]] = None,
             delay_all_signals: bool = False,
             trace: Optional[list] = None) -> dict:
    """Run one simulated condition and return a summary dict."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    clocks = list(controller.clocks)
    service_map = fit_service_times(sweep_df)
    mean_service = {c: _interp_service(service_map, c) for c in clocks}
    p_idle, slope, intercept = fit_power_model(sweep_df)
    power_active = {c: max(slope * c + intercept, p_idle) for c in clocks}

    if not tokens_per_request:
        tokens_per_request = [128]

    arrivals = generate_arrivals(n_requests, workload, base_rate, burst_mult, rng,
                                 burst_start_s, burst_duration_s, ramp_end_rate, duration_s)

    # Lazy import so fit_service_times() works without controllers/ present
    from controllers.base import ControllerState

    worker_free = [0.0] * n_workers          # min-heap of times each worker becomes free
    heapq.heapify(worker_free)
    pending = deque()                        # request indices waiting for a worker
    finish_heap = []                         # (finish_t, latency) not yet reported
    lat_window = deque()                     # (finish_t, latency) inside budget window
    miss_window = deque()
    arr_window = deque()                     # recent arrival times for rate/CV

    latencies, active_j, out_tokens = [], 0.0, 0
    clock_time = {c: 0.0 for c in clocks}
    obs_history = []                         # per-tick snapshots for delayed reads
    current_idx = len(clocks) - 1           # start at max clock (driver default)
    n_changes, next_arr, t = 0, 0, 0.0
    max_ticks = int(arrivals[-1] / control_interval_s) + 100_000

    for _ in range(max_ticks):
        t_end = t + control_interval_s
        clock = clocks[current_idx]

        # 1. arrivals in this tick
        while next_arr < n_requests and arrivals[next_arr] < t_end:
            pending.append(next_arr)
            arr_window.append(arrivals[next_arr])
            next_arr += 1

        # 2. dispatch FIFO to the soonest-free worker
        while pending:
            free_at = worker_free[0]
            i = pending[0]
            start = max(arrivals[i], free_at)
            if start >= t_end:
                break
            pending.popleft()
            heapq.heappop(worker_free)
            mu = mean_service[clock]
            svc = max(np_rng.normal(mu, 0.1 * mu), 0.05 * mu) + rng.uniform(0, SERVICE_OVERHEAD_MAX_S)
            finish = start + svc
            heapq.heappush(worker_free, finish)
            heapq.heappush(finish_heap, (finish, finish - arrivals[i]))
            active_j += svc * (power_active[clock] - p_idle)
            clock_time[clock] += svc
            out_tokens += tokens_per_request[i % len(tokens_per_request)]

        # 3. completions visible at end of tick
        while finish_heap and finish_heap[0][0] <= t_end:
            f, lat = heapq.heappop(finish_heap)
            latencies.append(lat)
            lat_window.append((f, lat))
            if lat > slo_target_s:
                miss_window.append((f, lat))
        for w in (lat_window, miss_window):
            while w and w[0][0] < t_end - budget_window_s:
                w.popleft()
        while arr_window and arr_window[0] < t_end - 10.0:
            arr_window.popleft()

        # 4. observation at end of tick
        busy = sum(1 for f in worker_free if f > t_end)
        win_lats = sorted(l for _, l in lat_window)
        n_win = len(win_lats)
        miss_rate_w = len(miss_window) / n_win if n_win else 0.0
        gaps = np.diff(list(arr_window)) if len(arr_window) >= 3 else np.array([])
        cv = float(gaps.std() / gaps.mean()) if gaps.size and gaps.mean() > 0 else 0.0
        obs = dict(
            t=t_end,
            power_w=p_idle + busy / n_workers * (power_active[clock] - p_idle),
            graphics_mhz=clock, sm_mhz=clock,
            util_pct=int(100 * busy / n_workers),
            queue_len=len(pending),
            arrival_rate=len(arr_window) / 10.0,
            arrival_cv=cv,
            p95_latency_s=_percentile(win_lats, 0.95),
            p99_latency_s=_percentile(win_lats, 0.99),
            miss_rate=miss_rate_w,
            error_budget_burn=miss_rate_w / slo_budget_ratio,
        )
        obs_history.append(obs)

        # 5. controller decision on (partly) stale observation
        delayed = _delayed_obs(obs_history, t_end - delay_ms / 1000.0)
        gpu_src = delayed
        sig_src = delayed if delay_all_signals else obs
        state = ControllerState(
            t=t_end,
            power_w=gpu_src["power_w"], graphics_mhz=gpu_src["graphics_mhz"],
            sm_mhz=gpu_src["sm_mhz"], util_pct=gpu_src["util_pct"], temp_c=60,
            telemetry_age_ms=float(delay_ms),
            queue_len=sig_src["queue_len"], arrival_rate=sig_src["arrival_rate"],
            arrival_cv=sig_src["arrival_cv"], p95_latency_s=sig_src["p95_latency_s"],
            p99_latency_s=sig_src["p99_latency_s"], slo_target_s=slo_target_s,
            error_budget_burn=sig_src["error_budget_burn"], miss_rate=sig_src["miss_rate"],
            phase="burst" if (workload == "sharp_burst" and
                              burst_start_s <= t_end <= burst_start_s + burst_duration_s) else "steady",
        )
        new_idx = controller.decide(state, current_idx, min_dwell)
        if trace is not None:
            trace.append({"t": t_end, "clock_mhz": clock, "queue_len": obs["queue_len"],
                          "p95_latency_s": obs["p95_latency_s"], "busy_workers": busy,
                          "reason": getattr(controller, "last_reason", "")})
        if new_idx != current_idx:
            n_changes += 1
            current_idx = new_idx

        t = t_end
        if next_arr >= n_requests and not pending and not finish_heap:
            break

    sim_duration = t
    idle_j = n_workers * p_idle * sim_duration
    total_j = active_j + idle_j
    lat_sorted = sorted(latencies)
    n = len(lat_sorted)
    misses = sum(1 for l in lat_sorted if l > slo_target_s)
    busy_total = sum(clock_time.values())

    return {
        "controller": getattr(controller, "name", type(controller).__name__),
        "delay_ms": delay_ms,
        "workload": workload,
        "n_workers": n_workers,
        "miss_rate": misses / n if n else 0.0,
        "p50_latency_s": _percentile(lat_sorted, 0.50),
        "p95_latency_s": _percentile(lat_sorted, 0.95),
        "p99_latency_s": _percentile(lat_sorted, 0.99),
        "mean_latency_s": float(np.mean(lat_sorted)) if n else 0.0,
        "total_j": total_j,
        "j_per_token": total_j / out_tokens if out_tokens else float("nan"),
        "n_requests": n,
        "sim_duration_s": sim_duration,
        "mean_clock_mhz": (sum(c * s for c, s in clock_time.items()) / busy_total) if busy_total else float("nan"),
        "n_clock_changes": n_changes,
        "delay_all_signals": delay_all_signals,
    }


def _delayed_obs(history: List[dict], target_t: float) -> dict:
    """Newest snapshot with t <= target_t; oldest if none (like DelayBuffer)."""
    ts = [h["t"] for h in history]
    i = bisect.bisect_right(ts, target_t) - 1
    return history[max(i, 0)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _append_csv(path: str, row: dict):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main(argv=None):
    from config import ExperimentConfig
    from controllers import get_controller

    ap = argparse.ArgumentParser(description="Scale-out DVFS simulation (CPU only)")
    ap.add_argument("--n-workers", type=int, default=8)
    ap.add_argument("--n-requests", type=int, default=10000)
    ap.add_argument("--controller", default="robust")
    ap.add_argument("--delay-ms", type=float, default=0.0)
    ap.add_argument("--workload", default="sharp_burst", choices=["steady", "ramp", "sharp_burst"])
    ap.add_argument("--sweep-csv", default="results/sweep_summary.csv")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--slo-target", type=float, default=None)
    ap.add_argument("--base-rate", type=float, default=None,
                    help="System arrival rate (req/s). Default: config base rate x n_workers")
    ap.add_argument("--burst-mult", type=float, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--delay-all-signals", action="store_true",
                    help="Also delay queue/latency signals, not only GPU telemetry")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig(controller=args.controller, workload=args.workload,
                           delay_ms=args.delay_ms, seed=args.seed,
                           results_dir=args.results_dir)
    if args.slo_target is not None:
        cfg.slo_target_s = args.slo_target

    sweep_df = pd.read_csv(args.sweep_csv)
    clocks = sorted(int(c) for c in sweep_df["clock_mhz"].unique())
    ctrl = get_controller(args.controller, clocks, cfg)

    base_rate = args.base_rate if args.base_rate is not None else cfg.base_arrival_rate * args.n_workers
    burst_mult = args.burst_mult if args.burst_mult is not None else cfg.burst_multiplier
    tokens = [int(p["max_new_tokens"]) for p in load_prompts(cfg.prompts_file)]

    result = simulate(
        n_workers=args.n_workers, n_requests=args.n_requests, controller=ctrl,
        delay_ms=args.delay_ms, workload=args.workload, sweep_df=sweep_df,
        slo_target_s=cfg.slo_target_s, base_rate=base_rate, burst_mult=burst_mult,
        seed=args.seed, control_interval_s=cfg.control_interval_s,
        min_dwell=cfg.min_dwell_intervals, slo_budget_ratio=cfg.slo_budget_ratio,
        budget_window_s=cfg.budget_window_s, burst_start_s=cfg.burst_start_s,
        burst_duration_s=cfg.burst_duration_s, ramp_end_rate=cfg.ramp_end_rate * args.n_workers,
        duration_s=cfg.duration_s, tokens_per_request=tokens,
        delay_all_signals=args.delay_all_signals,
    )

    os.makedirs(args.results_dir, exist_ok=True)
    out = os.path.join(args.results_dir, "scale_sim_summary.csv")
    _append_csv(out, result)
    print(f"[scale_sim] {result['controller']:>16} delay={args.delay_ms:>6.0f}ms "
          f"{args.workload:<11} miss={result['miss_rate']:.3f} "
          f"p95={result['p95_latency_s']:.2f}s J/tok={result['j_per_token']:.3f} "
          f"n={result['n_requests']}  -> {out}")
    return result


if __name__ == "__main__":
    main()
