"""Stale-telemetry delay sweep using the local controllers.

One worker, the sharp-burst workload (5x arrivals for 20 s), and every signal
the controller reads delayed. Each point averages many seeds. Energy saved is
measured against the static high clock (top verified clock) on the same seed.

"Robust" is the default controller from get_controller("robust"), including
its 300 ms stale-telemetry guard. --no-guard swaps in a guard-free robust
controller; that run is a diagnostic and is labeled as such.

Usage (from the project root):
    python experiments/stale_delay_sweep.py
    python experiments/stale_delay_sweep.py --no-guard
    python experiments/stale_delay_sweep.py --service-model compute
    python experiments/stale_delay_sweep.py --min-clock 1147
"""

import argparse
import bisect
import heapq
import os
import random
import sys
from collections import deque

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import DELAY_VALUES_MS, VERIFIED_CLOCKS_MHZ, ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState

CLOCKS = list(VERIFIED_CLOCKS_MHZ)
# Calibrated levels plus 100 and 250 ms, which sit below the 300 ms guard.
DELAYS_MS = sorted(set(DELAY_VALUES_MS) | {100, 250})
DYNAMIC = [
    ("naive_slo", "Naive SLO", "#3B75C4", "o"),
    ("robust_no_age", "Robust, no age", "#E8692E", "s"),
    ("robust", "Robust (default, 300 ms guard)", "#1F9E77", "^"),
]
NO_GUARD_LABEL = "Robust, no guard (diagnostic)"
STATIC_HIGH = "default"
P_IDLE_W = 10.0
W_PER_MHZ = 0.03
TOKENS_PER_REQUEST = 128
REF_MHZ = 3090


def service_time_s(clock_mhz: int, model: str) -> float:
    if model == "memory":
        # 1.5 s at 3090 MHz, 1.8 s at 2122 MHz, 11.2 s at 180 MHz.
        return 0.9 + 0.6 * REF_MHZ / clock_mhz
    if model == "compute":
        return 1.5 * REF_MHZ / clock_mhz
    raise ValueError(f"Unknown service model {model!r}")


def make_config(guard: bool = True) -> ExperimentConfig:
    # Midterm simulation: the original robust controller (300 ms hard guard,
    # age saturating at 1 s, no delay-aware wait), so the midterm figure is
    # reproducible after the hardware controller changed.
    legacy = dict(robust_age_scale_ms=1000.0, robust_delay_aware=False)
    return ExperimentConfig(stale_downclock_ms=300.0 if guard else None, **legacy)


def burst_rate(t: float, cfg: ExperimentConfig) -> float:
    if cfg.burst_start_s <= t <= cfg.burst_start_s + cfg.burst_duration_s:
        return cfg.base_arrival_rate * cfg.burst_multiplier
    return cfg.base_arrival_rate


def arrivals(n: int, cfg: ExperimentConfig, rng: random.Random):
    t, out = 0.0, []
    while len(out) < n:
        t += rng.expovariate(burst_rate(t, cfg))
        out.append(t)
    return out


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(int(p * len(sorted_vals)), len(sorted_vals) - 1)]


def simulate(controller, clocks, delay_ms: float, seed: int, n_requests: int, model: str,
             cfg: ExperimentConfig) -> dict:
    """One worker, FIFO queue. The controller sees an observation that is delay_ms old."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    arr = arrivals(n_requests, cfg, rng)
    mean_service = {c: service_time_s(c, model) for c in clocks}
    power = {c: max(P_IDLE_W + W_PER_MHZ * c, P_IDLE_W) for c in clocks}

    worker_free = 0.0
    pending = deque()
    finish_heap = []
    lat_window, miss_window, arr_window = deque(), deque(), deque()
    latencies, active_j = [], 0.0
    history, history_t = [], []
    current_idx = len(clocks) - 1
    n_changes, nxt, t = 0, 0, 0.0
    dt = cfg.control_interval_s

    while True:
        t_end = t + dt
        clock = clocks[current_idx]
        while nxt < n_requests and arr[nxt] < t_end:
            pending.append(nxt)
            arr_window.append(arr[nxt])
            nxt += 1
        while pending:
            i = pending[0]
            start = max(arr[i], worker_free)
            if start >= t_end:
                break
            pending.popleft()
            mu = mean_service[clock]
            svc = max(np_rng.normal(mu, 0.1 * mu), 0.05 * mu) + rng.uniform(0, 0.05)
            worker_free = start + svc
            heapq.heappush(finish_heap, (worker_free, worker_free - arr[i]))
            active_j += svc * (power[clock] - P_IDLE_W)

        while finish_heap and finish_heap[0][0] <= t_end:
            finish, lat = heapq.heappop(finish_heap)
            latencies.append(lat)
            lat_window.append((finish, lat))
            if lat > cfg.slo_target_s:
                miss_window.append((finish, lat))
        for window in (lat_window, miss_window):
            while window and window[0][0] < t_end - cfg.budget_window_s:
                window.popleft()
        while arr_window and arr_window[0] < t_end - 10.0:
            arr_window.popleft()

        win = sorted(lat for _, lat in lat_window)
        miss_rate = len(miss_window) / len(win) if win else 0.0
        gaps = np.diff(list(arr_window)) if len(arr_window) >= 3 else np.array([])
        cv = float(gaps.std() / gaps.mean()) if gaps.size and gaps.mean() > 0 else 0.0
        busy = worker_free > t_end
        history.append(dict(
            power_w=P_IDLE_W + (power[clock] - P_IDLE_W) * busy,
            graphics_mhz=clock, util_pct=100 * int(busy), queue_len=len(pending),
            arrival_rate=len(arr_window) / 10.0, arrival_cv=cv,
            p95_latency_s=_percentile(win, 0.95), p99_latency_s=_percentile(win, 0.99),
            miss_rate=miss_rate, error_budget_burn=miss_rate / cfg.slo_budget_ratio,
        ))
        history_t.append(t_end)

        idx = bisect.bisect_right(history_t, t_end - delay_ms / 1000.0) - 1
        obs = history[max(idx, 0)]
        in_burst = cfg.burst_start_s <= t_end <= cfg.burst_start_s + cfg.burst_duration_s
        state = ControllerState(
            t=t_end, power_w=obs["power_w"], graphics_mhz=obs["graphics_mhz"],
            sm_mhz=obs["graphics_mhz"], util_pct=obs["util_pct"], temp_c=60,
            telemetry_age_ms=float(delay_ms), queue_len=obs["queue_len"],
            arrival_rate=obs["arrival_rate"], arrival_cv=obs["arrival_cv"],
            p95_latency_s=obs["p95_latency_s"], p99_latency_s=obs["p99_latency_s"],
            slo_target_s=cfg.slo_target_s, error_budget_burn=obs["error_budget_burn"],
            miss_rate=obs["miss_rate"], phase="burst" if in_burst else "steady",
        )
        new_idx = controller.decide(state, current_idx, cfg.min_dwell_intervals)
        if new_idx != current_idx:
            n_changes += 1
            current_idx = new_idx

        t = t_end
        if nxt >= n_requests and not pending and not finish_heap:
            break

    total_j = active_j + P_IDLE_W * t
    lat_sorted = sorted(latencies)
    return {
        "miss_rate": sum(lat > cfg.slo_target_s for lat in lat_sorted) / len(lat_sorted),
        "p95_latency_s": _percentile(lat_sorted, 0.95),
        "j_per_token": total_j / (TOKENS_PER_REQUEST * len(lat_sorted)),
        "n_clock_changes": n_changes,
    }



def variants(guard: bool = True):
    """(row name, factory name, config, legend label, color, marker) per dynamic controller."""
    out = []
    for name, label, color, marker in DYNAMIC:
        if name == "robust" and not guard:
            out.append(("robust_no_guard", "robust", make_config(False), NO_GUARD_LABEL,
                        color, marker))
        else:
            out.append((name, name, make_config(True), label, color, marker))
    return out


def run_sweep(seeds=50, n_requests=40, model="memory", min_clock=0, delays=DELAYS_MS,
              guard=True):
    """Return one row per (controller, delay, seed), including the static high baseline."""
    cfg = make_config(True)
    clocks = [c for c in CLOCKS if c >= min_clock]
    rows = []
    for seed in range(seeds):
        static = simulate(get_controller(STATIC_HIGH, clocks, cfg), clocks, 0, seed,
                          n_requests, model, cfg)
        for delay in delays:
            rows.append(_row("static_high", delay, seed, static, static))
            for row_name, factory_name, v_cfg, *_ in variants(guard):
                result = simulate(get_controller(factory_name, clocks, v_cfg), clocks, delay,
                                  seed, n_requests, model, v_cfg)
                rows.append(_row(row_name, delay, seed, result, static))
    return pd.DataFrame(rows)


def _row(name, delay, seed, result, static):
    return {
        "controller": name,
        "delay_ms": delay,
        "seed": seed,
        "miss_rate": result["miss_rate"],
        "p95_latency_s": result["p95_latency_s"],
        "j_per_token": result["j_per_token"],
        "energy_saved_pct": 100.0 * (1.0 - result["j_per_token"] / static["j_per_token"]),
        "n_clock_changes": result["n_clock_changes"],
    }


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    summary = (
        per_seed.groupby(["controller", "delay_ms"], as_index=False)
        .agg(miss_rate=("miss_rate", "mean"),
             energy_saved_pct=("energy_saved_pct", "mean"),
             p95_latency_s=("p95_latency_s", "mean"),
             n_clock_changes=("n_clock_changes", "mean"))
    )
    summary["miss_rate_pct"] = 100.0 * summary["miss_rate"]
    return summary


def robust_settings(guard: bool = True) -> dict:
    robust = get_controller("robust", CLOCKS, make_config(guard))
    return {"stale_downclock_ms": robust.stale_downclock_ms, "w_q": robust.w_q,
            "w_age": robust.w_age, "w_burn": robust.w_burn, "w_cv": robust.w_cv,
            "low_risk_thresh": robust.low_risk_thresh,
            "high_risk_thresh": robust.high_risk_thresh}


def age_threshold_ms() -> float:
    """No-guard diagnostic only: delay at which the age term alone reaches the
    down-clock threshold. The default controller blocks earlier, at its guard."""
    s = robust_settings(guard=False)
    return 1000.0 * s["low_risk_thresh"] / s["w_age"]


def boundary(guard: bool) -> tuple[float, str]:
    if guard:
        ms = robust_settings(True)["stale_downclock_ms"]
        return ms, f"stale guard\n({ms:.0f} ms)"
    ms = age_threshold_ms()
    return ms, f"diagnostic: age term\nreaches threshold ({ms:.0f} ms)"


def caption(guard: bool, seeds: int, n_requests: int, model: str, clocks, cfg) -> str:
    s = robust_settings(guard)
    robust_desc = (
        f"the default robust controller (stale-telemetry guard: no down-clock on telemetry "
        f"older than {s['stale_downclock_ms']:.0f} ms)"
        if guard else
        "a NO-GUARD DIAGNOSTIC variant of the robust controller (stale guard disabled; "
        f"down-clocks stop only when the age term reaches the threshold, about "
        f"{age_threshold_ms():.0f} ms). This is not the default controller"
    )
    return (
        f"Simulated stale-telemetry sweep with {robust_desc}. Risk weights "
        f"q/age/burn/cv = {s['w_q']}/{s['w_age']}/{s['w_burn']}/{s['w_cv']}, "
        f"down-clock below risk {s['low_risk_thresh']}, up-clock above {s['high_risk_thresh']}. "
        f"One worker, sharp burst ({cfg.burst_multiplier:g}x for {cfg.burst_duration_s:g} s), "
        f"{n_requests} requests, {seeds} seeds, {model}-bound service-time model, "
        f"verified clock ladder {list(clocks)} MHz; static high = {clocks[-1]} MHz. "
        f"All controller inputs are delayed. Calibrated delay levels {DELAY_VALUES_MS} ms "
        f"come from the GPU timing check; 100 and 250 ms are extra points below the guard. "
        f"SLO {cfg.slo_target_s:g} s with a {100 * cfg.slo_budget_ratio:g}% miss budget. "
        f"Simulation, not a GPU measurement."
    )


def plot(summary: pd.DataFrame, path: str, cfg: ExperimentConfig, guard: bool = True) -> None:
    static_miss = float(summary[summary["controller"] == "static_high"]["miss_rate_pct"].mean())
    edge_ms, edge_label = boundary(guard)
    last = DELAYS_MS[-1]
    fig, (ax_miss, ax_energy) = plt.subplots(1, 2, figsize=(10.5, 3.5), dpi=150)

    for row_name, _, _, label, color, marker in variants(guard):
        part = summary[summary["controller"] == row_name].sort_values("delay_ms")
        ax_miss.plot(part["delay_ms"], part["miss_rate_pct"], marker=marker, color=color,
                     label=label, lw=2)
        ax_energy.plot(part["delay_ms"], part["energy_saved_pct"], marker=marker, color=color,
                       label=label, lw=2)

    budget_pct = 100 * cfg.slo_budget_ratio
    ax_miss.axhline(static_miss, color="black", ls="--", lw=1.2)
    ax_miss.text(last, static_miss + 0.8, f"static high ({CLOCKS[-1]} MHz)", ha="right",
                 fontsize=8)
    ax_miss.axhline(budget_pct, color="#C0392B", lw=1.2)
    ax_miss.set_ylim(-4, max(summary["miss_rate_pct"].max() * 1.15, 10))
    ax_miss.text(last, budget_pct - 3, f"{budget_pct:.0f}% budget", ha="right", fontsize=8,
                 color="#C0392B")
    ax_energy.axhline(0, color="black", ls="--", lw=1.2)
    ax_energy.text(last, 0.25, f"static high ({CLOCKS[-1]} MHz)", ha="right", fontsize=8)

    for ax in (ax_miss, ax_energy):
        ax.axvline(edge_ms, color="gray", ls=":", lw=1)
        ax.set_xlabel("Injected telemetry delay (ms)")
        ax.set_xticks(DELAY_VALUES_MS)
        ax.grid(axis="y", alpha=0.25)
    ax_miss.text(edge_ms - 30, ax_miss.get_ylim()[1] * 0.45, edge_label, fontsize=8,
                 color="gray", ha="right")
    ax_miss.set_title(f"(a) Requests over the {cfg.slo_target_s:g} s SLO", loc="left")
    ax_miss.set_ylabel("SLO miss rate (%)")
    ax_miss.legend(frameon=False, fontsize=8)
    ax_energy.set_title("(b) Energy per token saved", loc="left")
    ax_energy.set_ylabel("Energy saved vs static high (%)")
    if not guard:
        fig.suptitle("NO-GUARD DIAGNOSTIC (not the default robust controller)", fontsize=9,
                     color="black")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--n-requests", type=int, default=40)
    parser.add_argument("--service-model", choices=["memory", "compute"], default="memory")
    parser.add_argument("--min-clock", type=int, default=0,
                        help="Drop clocks below this MHz value from the controller ladder")
    parser.add_argument("--no-guard", action="store_true",
                        help="Diagnostic: disable the robust controller's stale guard")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "results", "stale_delay_sweep"))
    args = parser.parse_args(argv)

    guard = not args.no_guard
    cfg = make_config(True)
    clocks = [c for c in CLOCKS if c >= args.min_clock]
    per_seed = run_sweep(args.seeds, args.n_requests, args.service_model, args.min_clock,
                         guard=guard)
    summary = summarize(per_seed)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = (args.service_model + (f"_min{args.min_clock}" if args.min_clock else "")
           + ("" if guard else "_noguard_diagnostic"))
    per_seed.to_csv(os.path.join(args.out_dir, f"per_seed_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, f"summary_{tag}.csv"), index=False)
    fig_path = os.path.join(args.out_dir, f"fig_stale_delay_sweep_{tag}.png")
    plot(summary, fig_path, cfg, guard)
    text = caption(guard, args.seeds, args.n_requests, args.service_model, clocks, cfg)
    with open(os.path.join(args.out_dir, f"caption_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print("SLO miss rate (%)")
    print(summary.pivot(index="delay_ms", columns="controller", values="miss_rate_pct")
          .round(1).to_string())
    print("\nEnergy saved vs static high (%)")
    print(summary.pivot(index="delay_ms", columns="controller", values="energy_saved_pct")
          .round(1).to_string())
    print(f"\nCaption: {text}")
    print(f"\nWrote {fig_path}")
    return fig_path


if __name__ == "__main__":
    main()
