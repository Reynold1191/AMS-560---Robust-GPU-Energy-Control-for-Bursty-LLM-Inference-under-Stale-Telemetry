"""
run_experiment.py
Main experiment runner.

Usage (from an Administrator PowerShell):
  python src/run_experiment.py --controller robust --delay-ms 500 --workload sharp_burst --seed 42
  python src/run_experiment.py --controller naive_slo --delay-ms 1250
  python src/run_experiment.py --controller default   --delay-ms 0   --workload steady
  python src/run_experiment.py --sweep-clocks --seeds 41 42 43   # fixed-frequency characterisation

All logs go to results/<run_id>_*.csv and results/run_summary.csv.

What the controller sees
------------------------
Every input the controller acts on is delayed by --delay-ms: the NVML
sample (power, clock, utilisation) AND the queue length, rolling P95,
error-budget burn and arrival statistics.  Both are sampled every
telemetry_poll_s into their own DelayBuffer and read with the same delay
draw (see observation.py).  The oracle always reads fresh inputs.

Timeline
--------
  lock start clock -> warm-up (warmup_s) -> measurement (duration_s - warmup_s)
The burst starts burst_start_s after warm-up ends, for both the load
generator and the control loop, so analysis can use the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import os
import random
import statistics
import sys
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path when run from a subdirectory
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config as cfg_module
from config import ExperimentConfig
from controllers import DYNAMIC, get_controller
from capacity_model import load_capacity
from delay_buffer import DelayBuffer
from gpu_clock import reset_graphics_clock, set_graphics_clock
from load_generator import LoadGenerator, load_prompts
from metrics import MetricsCollector
from observation import (
    ObservationSampler,
    ramp_end_rate,
    clock_tracking_pct,
    delayed_state,
    draw_delay_s,
    measurement_length_s,
    phase_at,
    starting_index,
    take_observation,
    validate_timeline,
)
from request_queue import RequestQueue
from slo_budget import SLOBudget
from telemetry import TelemetrySampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_experiment")


class ClockLockError(RuntimeError):
    """nvidia-smi refused to lock the clock (usually: not an Administrator terminal)."""


# ---------------------------------------------------------------------------
# File logging helper
# ---------------------------------------------------------------------------

def _add_file_handler(log_path: str) -> logging.FileHandler:
    """Attach a FileHandler to the root logger and return it for later removal."""
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    return fh


def _remove_file_handler(fh: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(fh)
    fh.close()


# ---------------------------------------------------------------------------
# Burst recovery detection
# ---------------------------------------------------------------------------

def detect_recovery(
    control_events: list,
    burst_end_t: float,
    slo_target_s: float,
    k_stable: int = 3,
) -> Optional[float]:
    """
    Find the time after burst_end_t when both queue and latency have
    stabilised for k_stable consecutive control intervals (true, undelayed
    values).  Returns elapsed seconds from burst_end_t, or None.
    """
    stable_count = 0
    for ev in control_events:
        if ev["t"] < burst_end_t:
            continue
        if ev["queue_len"] == 0 and ev["p95_latency_s"] < slo_target_s:
            stable_count += 1
            if stable_count >= k_stable:
                return ev["t"] - burst_end_t
        else:
            stable_count = 0
    return None


# ---------------------------------------------------------------------------
# Clock helpers
# ---------------------------------------------------------------------------

def _lock_clock(mhz: int) -> None:
    rc = set_graphics_clock(mhz)
    if rc != 0:
        raise ClockLockError(
            f"nvidia-smi -lgc {mhz},{mhz} failed (rc={rc}). "
            "Run from an Administrator terminal."
        )


# ---------------------------------------------------------------------------
# Clock sweep (fixed-frequency characterisation – Deliverable D2)
# ---------------------------------------------------------------------------

_SWEEP_AGG_FIELDS = ["j_per_token", "p50", "p95", "p99", "mean_lat", "miss_rate",
                     "mean_service_s", "mean_power_w", "mean_util_pct",
                     "clock_tracking_pct", "total_j", "output_tokens", "n_requests"]


def aggregate_sweep(rows: List[dict]) -> List[dict]:
    """One row per clock: mean over seeds, plus spread of P95 and J/token."""
    by_clock: dict = {}
    for r in rows:
        by_clock.setdefault(int(r["clock_mhz"]), []).append(r)
    out = []
    for mhz in sorted(by_clock):
        group = by_clock[mhz]
        agg = {"clock_mhz": mhz, "n_seeds": len(group),
               "workload": group[0].get("workload", "")}
        for k in _SWEEP_AGG_FIELDS:
            vals = [float(g[k]) for g in group
                    if g.get(k) not in (None, "") and float(g[k]) == float(g[k])]
            agg[k] = statistics.mean(vals) if vals else float("nan")
        for k in ("p95", "j_per_token"):
            vals = [float(g[k]) for g in group if float(g[k]) == float(g[k])]
            agg[f"{k}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(agg)
    return out


def _write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def run_clock_sweep(cfg: ExperimentConfig, seeds: List[int], cooldown_s: float = 5.0) -> None:
    """
    Run the same workload with the clock locked at each verified clock,
    for every seed.  Per-run files go to <results_dir>/sweep/, the per-seed
    table to sweep_runs.csv and the per-clock mean to sweep_summary.csv.
    Clocks are interleaved within each seed so thermal drift is not
    confounded with clock.
    """
    from llm_worker import load_model

    logger.info("=== Fixed-frequency clock sweep ===")
    clocks = list(cfg.selected_clocks)
    logger.info("Sweep clocks (MHz): %s  seeds: %s  workload: %s",
                clocks, seeds, cfg.workload)

    model, tokenizer = load_model(cfg.model_id)
    prompts = load_prompts(cfg.prompts_file)

    os.makedirs(cfg.results_dir, exist_ok=True)
    sweep_dir = os.path.join(cfg.results_dir, "sweep")
    rows = []

    for seed in seeds:
        order = clocks[::-1] if seed % 2 else clocks  # alternate direction per seed
        for mhz in order:
            run_id = f"sweep_{mhz}mhz_s{seed}"
            logger.info("-- Clock %d MHz, seed %d --", mhz, seed)
            sweep_cfg = dataclasses.replace(
                cfg,
                controller="static_high",
                delay_ms=0.0,
                delay_jitter_ms=0.0,
                seed=seed,
                run_id=run_id,
                results_dir=sweep_dir,
                selected_clocks=[mhz],   # single-clock ladder: static_high == mhz
            )
            result = _run_single(sweep_cfg, model, tokenizer, prompts)
            result["clock_mhz"] = mhz
            rows.append(result)
            _write_csv(os.path.join(cfg.results_dir, "sweep_runs.csv"), rows)
            time.sleep(cooldown_s)

    summary = aggregate_sweep(rows)
    path = os.path.join(cfg.results_dir, "sweep_summary.csv")
    _write_csv(path, summary)
    logger.info("Sweep summary saved → %s", path)
    print("\nSweep results (mean over seeds):")
    for r in summary:
        print(
            f"  {r['clock_mhz']:5d} MHz | J/tok={r['j_per_token']:.4f} | "
            f"P95={r['p95']:.2f}±{r['p95_std']:.2f} s | svc={r['mean_service_s']:.2f} s | "
            f"util={r['mean_util_pct']:.0f}% | clock ok={r['clock_tracking_pct']:.0f}% | "
            f"miss={r['miss_rate']:.3f}"
        )


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def _run_single(
    cfg: ExperimentConfig,
    model,
    tokenizer,
    prompts: list,
) -> dict:
    """
    Run one complete experiment condition.  Returns the run-summary dict.
    Thread lifecycle:
        TelemetrySampler  → gpu DelayBuffer ┐
        ObservationSampler→ obs DelayBuffer ┴→ Controller (main thread)
        LoadGenerator     → RequestQueue → LLMWorker
    """
    from llm_worker import LLMWorker

    validate_timeline(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ── Per-run log file ──────────────────────────────────────────────────
    log_path = os.path.join(cfg.results_dir, f"{cfg.run_id}.log")
    fh = _add_file_handler(log_path)
    logger.info("Run log: %s", log_path)

    # ── Infrastructure ────────────────────────────────────────────────────
    rq = RequestQueue()
    gpu_buf = DelayBuffer()
    obs_buf = DelayBuffer()
    slo = SLOBudget(
        slo_target_s=cfg.slo_target_s,
        budget_ratio=cfg.slo_budget_ratio,
        window_s=cfg.budget_window_s,
    )
    metrics = MetricsCollector(cfg.slo_target_s, cfg.results_dir, cfg.run_id)

    def on_complete(req):
        if req.latency_s is not None:
            slo.record(req.finish_t, req.latency_s)
            metrics.on_request(req)

    worker = LLMWorker(model, tokenizer, rq, on_complete=on_complete,
                       batch_size=cfg.max_batch_size)
    sampler = TelemetrySampler(gpu_buf, poll_interval_s=cfg.telemetry_poll_s)
    gen = LoadGenerator(rq, cfg, prompts)
    obs_sampler = ObservationSampler(lambda: take_observation(rq, gen, slo),
                                     obs_buf, poll_interval_s=cfg.telemetry_poll_s)

    exp_clocks = list(cfg.selected_clocks)
    if cfg.capacity_file:
        cfg.capacity_rps, cfg.batch_time_s = load_capacity(cfg.capacity_file, exp_clocks)
        if cfg.capacity_rps:
            logger.info("Capacity floor (margin %.2f): %s", cfg.capacity_margin,
                        ", ".join(f"{c} MHz={r:.2f} req/s" for c, r in zip(exp_clocks, cfg.capacity_rps)))
    controller = get_controller(cfg.controller, exp_clocks, cfg)
    locks_clock = cfg.controller != "default"
    current_idx = starting_index(controller, exp_clocks)

    # The oracle is the fresh-telemetry reference: it never sees delay.
    is_oracle = cfg.controller == "oracle"
    delay_s = 0.0 if is_oracle else cfg.delay_ms / 1000.0
    jitter_s = 0.0 if is_oracle else cfg.delay_jitter_ms / 1000.0
    delay_rng = random.Random(cfg.seed + 7919)

    clock_commands: list = []   # (t, mhz) for clock-tracking check
    control_events: list = []
    max_queue_len = 0
    lock_failed = False
    exp_start = None

    try:
        logger.info(
            "Starting run_id=%s  controller=%s  delay=%.0f ms  workload=%s  "
            "SLO=%.2f s  budget=%.0f%%  clocks=%s",
            cfg.run_id, cfg.controller, cfg.delay_ms, cfg.workload,
            cfg.slo_target_s, cfg.slo_budget_ratio * 100, exp_clocks,
        )

        # ── Lock the starting clock BEFORE warm-up ────────────────────────
        if locks_clock:
            _lock_clock(exp_clocks[current_idx])
            clock_commands.append((time.monotonic(), exp_clocks[current_idx]))
            logger.info("Start clock locked at %d MHz", exp_clocks[current_idx])

        # ── Warm-up ───────────────────────────────────────────────────────
        sampler.start()
        worker.warmup(n=3)
        gen.start()
        worker.start()
        obs_sampler.start()
        time.sleep(cfg.warmup_s)

        # ── Shared origin: end of warm-up ─────────────────────────────────
        exp_start = time.monotonic()
        gen.set_origin(exp_start)
        metrics.set_warmup_end(exp_start)
        sampler.set_phase(phase_at(0.0, cfg))
        logger.info("Warm-up complete (%.1f s)", cfg.warmup_s)

        measure_s = measurement_length_s(cfg)
        next_tick = exp_start

        # ── Main control loop ─────────────────────────────────────────────
        while True:
            now = time.monotonic()
            elapsed = now - exp_start
            if elapsed >= measure_s:
                break
            sampler.set_phase(phase_at(elapsed, cfg))

            # ── Delayed view of ALL controller inputs ─────────────────────
            d = draw_delay_s(delay_s, jitter_s, delay_rng)
            state, obs, age_s = delayed_state(gpu_buf, obs_buf, d, cfg.slo_target_s)
            if state is None:
                next_tick += cfg.control_interval_s
                time.sleep(max(0.0, next_tick - time.monotonic()))
                continue

            true_q = rq.qsize
            max_queue_len = max(max_queue_len, true_q)

            # ── Controller decision ───────────────────────────────────────
            action_start = time.monotonic()
            new_idx = controller.decide(state, current_idx, min_dwell=cfg.min_dwell_intervals)
            new_idx = max(0, min(new_idx, len(exp_clocks) - 1))

            if new_idx != current_idx and locks_clock:
                if set_graphics_clock(exp_clocks[new_idx]) == 0:
                    clock_commands.append((time.monotonic(), exp_clocks[new_idx]))
                    current_idx = new_idx
                else:
                    lock_failed = True
                    logger.warning("Clock change to %d MHz failed; holding %d MHz",
                                   exp_clocks[new_idx], exp_clocks[current_idx])
            elif not locks_clock:
                current_idx = new_idx
            action_ms = (time.monotonic() - action_start) * 1000.0

            # ── Log control event: true values + what the controller saw ──
            ev = {
                "t":                   now,
                "elapsed_s":           elapsed,
                "controller":          cfg.controller,
                "delay_ms":            cfg.delay_ms,
                "applied_delay_ms":    d * 1000.0,
                "telemetry_t":         state.t - age_s,
                "telemetry_age_ms":    age_s * 1000.0,
                "queue_len":           true_q,
                "p95_latency_s":       slo.p95(),
                "p99_latency_s":       slo.p99(),
                "burn_rate":           slo.burn_rate(),
                "miss_rate":           slo.miss_rate(),
                "obs_queue_len":       state.queue_len,
                "obs_p95_latency_s":   state.p95_latency_s,
                "obs_burn_rate":       state.error_budget_burn,
                "obs_power_w":         state.power_w,
                "obs_graphics_mhz":    state.graphics_mhz,
                "arrival_rate":        gen.recent_arrival_rate(),
                "obs_arrival_rate":    state.arrival_rate,
                "capacity_floor_mhz":  exp_clocks[controller.capacity_floor_idx(state.arrival_rate)],
                "arrival_cv":          state.arrival_cv,
                "selected_mhz":        exp_clocks[current_idx],
                "action_ms":           action_ms,
                "phase":               phase_at(elapsed, cfg),
                "decision_reason":     getattr(controller, "last_reason", ""),
            }
            control_events.append(ev)
            metrics.add_control_event(ev)

            next_tick += cfg.control_interval_s
            time.sleep(max(0.0, next_tick - time.monotonic()))

        # ── Drain remaining requests ───────────────────────────────────────
        logger.info("Draining queue …")
        drain_start = time.monotonic()
        while not rq.empty() and (time.monotonic() - drain_start) < cfg.drain_timeout_s:
            time.sleep(0.5)

    finally:
        gen.stop()
        worker.stop()
        obs_sampler.stop()
        sampler.stop()
        if locks_clock:
            reset_graphics_clock()
        _remove_file_handler(fh)

    # ── Post-run analysis ─────────────────────────────────────────────────
    burst_end_t = exp_start + cfg.burst_start_s + cfg.burst_duration_s
    recovery_s = (detect_recovery(control_events, burst_end_t, cfg.slo_target_s)
                  if cfg.workload == "sharp_burst" else None)

    summary = metrics.run_summary(
        sampler.samples, cfg, recovery_s=recovery_s, max_queue_len=max_queue_len,
    )

    # Requests that never finished are SLO misses, not missing data.
    n_unfinished = max(0, rq.total_enqueued - len(worker.completed))
    n_done = summary["n_requests"]
    misses_done = summary["miss_rate"] * n_done if n_done else 0.0
    summary["miss_rate_completed"] = summary["miss_rate"]
    summary["n_unfinished"] = n_unfinished
    total = n_done + n_unfinished
    summary["miss_rate"] = (misses_done + n_unfinished) / total if total else float("nan")

    meas = [s for s in sampler.samples if s.t >= exp_start]
    done = metrics.measurement_requests()
    svc = [r.service_time_s for r in done if r.service_time_s is not None]
    summary.update({
        "slo_target_s":       cfg.slo_target_s,
        "slo_budget_ratio":   cfg.slo_budget_ratio,
        "base_rate":          cfg.base_arrival_rate,
        "ramp_end_rate":      ramp_end_rate(cfg) if cfg.workload == "ramp" else "",
        "capacity_floor":     bool(cfg.capacity_rps) and cfg.controller in DYNAMIC,
        "capacity_margin":    cfg.capacity_margin if cfg.capacity_rps else "",
        "batch_size":         cfg.max_batch_size,
        "model":              cfg.model_id,
        "start_mhz":          clock_commands[0][1] if clock_commands else "",
        "n_clock_changes":    max(0, len(clock_commands) - 1),
        "clock_lock_failed":  lock_failed,
        "clock_tracking_pct": clock_tracking_pct(meas, clock_commands) if locks_clock else float("nan"),
        "mean_service_s":     statistics.mean(svc) if svc else float("nan"),
        "j_per_request":      summary["total_j"] / n_done if n_done else float("nan"),
        "prompt_tokens":      sum(r.prompt_tokens or 0 for r in done),
        "prompts_file":       cfg.prompts_file,
        "mean_power_w":       statistics.mean(s.power_w for s in meas) if meas else float("nan"),
        "mean_util_pct":      statistics.mean(s.util_pct for s in meas) if meas else float("nan"),
    })
    if locks_clock and summary["clock_tracking_pct"] == summary["clock_tracking_pct"] \
            and summary["clock_tracking_pct"] < 90:
        logger.warning("Only %.0f%% of loaded telemetry samples matched the commanded "
                       "clock; this run's clock labels are unreliable.",
                       summary["clock_tracking_pct"])

    # ── Save CSV logs ─────────────────────────────────────────────────────
    req_path   = metrics.save_requests_csv()
    telem_path = metrics.save_telemetry_csv(sampler.samples)
    ctrl_path  = metrics.save_control_csv()
    summ_path  = metrics.append_run_summary(summary)

    logger.info(
        "Run complete → J/tok=%.4f  P95=%.3f s  miss=%.3f (%d unfinished)  "
        "util=%.0f%%  clock ok=%.0f%%",
        summary["j_per_token"], summary["p95"], summary["miss_rate"], n_unfinished,
        summary["mean_util_pct"], summary["clock_tracking_pct"],
    )
    logger.info("Logs: %s  %s  %s  %s", req_path, telem_path, ctrl_path, summ_path)
    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    d = ExperimentConfig()  # defaults come from config.py, one source of truth
    p = argparse.ArgumentParser(description="AMS 560 GPU energy control experiment")
    p.add_argument("--controller",    default=d.controller, choices=cfg_module.CONTROLLER_NAMES)
    p.add_argument("--delay-ms",      type=float, default=d.delay_ms)
    p.add_argument("--jitter-ms",     type=float, default=d.delay_jitter_ms)
    p.add_argument("--workload",      default=d.workload, choices=cfg_module.WORKLOAD_NAMES)
    p.add_argument("--duration",      type=float, default=d.duration_s,
                   help="Total seconds including warm-up")
    p.add_argument("--warmup",        type=float, default=d.warmup_s)
    p.add_argument("--burst-start",   type=float, default=d.burst_start_s,
                   help="Seconds after warm-up when the burst starts")
    p.add_argument("--burst-duration", type=float, default=d.burst_duration_s)
    p.add_argument("--seed",          type=int,   default=d.seed)
    p.add_argument("--model",         default=d.model_id)
    p.add_argument("--slo-target",    type=float, default=d.slo_target_s)
    p.add_argument("--budget-ratio",  type=float, default=d.slo_budget_ratio)
    p.add_argument("--results-dir",   default=d.results_dir)
    p.add_argument("--run-id",        default=None)
    p.add_argument("--sweep-clocks",  action="store_true",
                   help="Run fixed-frequency sweep (D2 deliverable)")
    p.add_argument("--seeds",         type=int, nargs="+", default=None,
                   help="Seeds for --sweep-clocks (default: --seed only)")
    p.add_argument("--no-age",        action="store_true",
                   help="Ablation: disable telemetry-age term")
    p.add_argument("--no-budget",     action="store_true",
                   help="Ablation: disable error-budget term")
    p.add_argument("--oracle-table",  default=None)
    p.add_argument("--prompts-file",  default=d.prompts_file)
    p.add_argument("--static-efficient-mhz", type=int, default=d.static_efficient_mhz)
    p.add_argument("--batch-size",    type=int, default=d.max_batch_size)
    p.add_argument("--burst-multiplier", type=float, default=d.burst_multiplier)
    p.add_argument("--base-rate",     type=float, default=d.base_arrival_rate)
    p.add_argument("--capacity-file", default=None,
                   help="capacity.json (capacity_model.py): per-clock capacity floor for "
                        "dynamic controllers")
    p.add_argument("--capacity-margin", type=float, default=d.capacity_margin)
    return p


def config_from_args(args) -> ExperimentConfig:
    run_id = args.run_id or (
        f"{args.controller}_d{int(args.delay_ms)}_{args.workload}_s{args.seed}"
    )
    return ExperimentConfig(
        controller=args.controller,
        delay_ms=args.delay_ms,
        delay_jitter_ms=args.jitter_ms,
        workload=args.workload,
        duration_s=args.duration,
        warmup_s=args.warmup,
        burst_start_s=args.burst_start,
        burst_duration_s=args.burst_duration,
        seed=args.seed,
        model_id=args.model,
        slo_target_s=args.slo_target,
        slo_budget_ratio=args.budget_ratio,
        results_dir=args.results_dir,
        run_id=run_id,
        use_telemetry_age=not args.no_age,
        use_error_budget=not args.no_budget,
        oracle_table_path=args.oracle_table,
        prompts_file=args.prompts_file,
        static_efficient_mhz=args.static_efficient_mhz,
        max_batch_size=args.batch_size,
        burst_multiplier=args.burst_multiplier,
        base_arrival_rate=args.base_rate,
        capacity_file=args.capacity_file,
        capacity_margin=args.capacity_margin,
    )


def main():
    args = build_parser().parse_args()
    c = config_from_args(args)
    validate_timeline(c)

    if args.sweep_clocks:
        run_clock_sweep(c, seeds=args.seeds or [args.seed])
        return

    from llm_worker import load_model
    model, tokenizer = load_model(c.model_id)
    prompts = load_prompts(c.prompts_file)
    try:
        _run_single(c, model, tokenizer, prompts)
    except ClockLockError as exc:
        logger.error("%s", exc)
        sys.exit(2)


if __name__ == "__main__":
    main()
