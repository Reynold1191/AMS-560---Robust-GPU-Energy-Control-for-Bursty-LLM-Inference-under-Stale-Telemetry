"""
run_demo.py
End-to-end hardware experiment, from an Administrator terminal.

Phases
  0  environment + admin check
  1  GPU timing check (check_gpu_timing.py -> results/gpu_check/)
  1b clock-sensitivity benchmark: model / batch size
  1c top-clock probe in the real serving stack: measured capacity sets the
     base rate so the burst peak is 0.9 x top-clock capacity
  2  fixed-clock sweep, sharp-burst workload, every verified clock x sweep seeds
  3  SLO calibration from the sweep (calibrate_slo.py) - stops here if the
     testbed cannot support a DVFS comparison (unless --force)
  4  oracle lookup table
  5  controller matrix, paired seeds, run order shuffled within each seed
       primary  : naive_slo, robust          x 4 delays x 3 workloads
                  static_high (2122 MHz)     x 3 workloads (same-session reference)
       ablation : robust_v1, robust_no_age   x 4 delays (sharp_burst; also in
                  --primary-only unless --no-ablation)
       reference: static_efficient           x 3 workloads (delay-independent)
                  oracle                     (sharp_burst, fresh telemetry)
  6  scale-out simulation and figures

Delay levels come from config.DELAY_VALUES_MS, derived from the measured
NVML refresh (~500 ms) and clock/power settling (~1.25 s) on this GPU.

Usage (Administrator PowerShell, project env active):
    python src/run_demo.py                     # 3 seeds, ~5-6 h
    python src/run_demo.py --seeds 42          # 1 seed,  ~2 h
    python src/run_demo.py --quick             # 1 seed, sharp_burst only, ~1 h
    python src/run_demo.py --primary-only      # proposal run: 24 primary conditions + 3 static_high
                                           # + 8 sharp_burst ablations x 3 paired seeds
                                           # (105 runs), ~5 h
    python src/run_demo.py --primary-only --seeds 42   # 35 runs, 1 seed, ~1.9 h

The SLO is fixed (--slo-target, default 5.0 s) so attempts are comparable;
--calibrated-slo uses the per-attempt calibration instead.

Model, batch size and arrival rate are chosen by bench_clock_sensitivity.py
(phase 1b) so the GPU clock actually changes decode speed; pass
--no-auto-config to use --model / --batch-size / --base-rate instead.
    python src/run_demo.py --sweep-from results/attempt_4   # reuse a finished sweep
    python src/run_demo.py --smoke             # ~15 min code check, see SMOKE_MATRIX

Repeat mode: independent primary-only runs back to back, each in its own
results/attempt_N (N, N+1, ...), each with its own benchmark, probe and sweep.
1-seed runs go first (fast feedback), then 3-seed runs:
    python src/run_demo.py --repeat-1seed 2 --repeat-3seed 2 --skip-gpu-check
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repository root: results/, prompts/
sys.path.insert(0, HERE)
from config import DELAY_VALUES_MS, ExperimentConfig  # noqa: E402

PYTHON = sys.executable
D = ExperimentConfig()

PROBE_OVERLOAD = 1.5    # probe arrival rate / benchmark top-clock capacity
BURST_PEAK_FRAC = 0.9   # burst peak (base x multiplier) / measured top-clock capacity


def burst_base_rate(capacity_top: float, burst_multiplier: float) -> float:
    """
    Base rate so the burst peak is BURST_PEAK_FRAC x the top clock's measured
    capacity: the top clock can just keep up, every lower clock cannot, so a
    controller that reacts late builds a backlog. attempt_7 sized the load
    from the isolated benchmark (8.0 req/s at 2122 MHz) while the real stack
    served ~5.9 req/s, so the burst overloaded every controller.
    """
    rate = BURST_PEAK_FRAC * capacity_top / burst_multiplier
    return max(0.05, int(rate * 20) / 20)


def probe_capacity(out: str, args, chosen: dict):
    """
    Phase 1c: overload the top clock in the real runner (sampler, generator
    and control threads all running) and measure full-batch throughput.
    Returns the probe record, or None.
    """
    import statistics
    from capacity_model import full_batch_times
    from config import VERIFIED_CLOCKS_MHZ
    probe_dir = os.path.join(out, "probe")
    rate = round(PROBE_OVERLOAD * chosen["batch_size"] / chosen["service_top_s"], 2)
    run([PYTHON, os.path.join(HERE, "run_experiment.py"), "--controller", "static_high", "--workload", "steady",
         "--base-rate", rate, "--duration", 40, "--warmup", 10, "--seed", 1,
         "--run-id", "probe_top", "--model", args.model, "--prompts-file", args.prompts_file,
         "--batch-size", args.batch_size, "--results-dir", probe_dir])
    times = full_batch_times(os.path.join(probe_dir, "probe_top_requests.csv"),
                             int(args.batch_size))
    if len(times) < 3:
        return None
    t = statistics.median(times)
    rec = {"clock_mhz": VERIFIED_CLOCKS_MHZ[-1], "probe_rate": rate, "n_full_batches": len(times),
           "batch_time_s": t, "capacity_rps": args.batch_size / t,
           "bench_batch_time_s": chosen["service_top_s"],
           "burst_peak_frac": BURST_PEAK_FRAC, "burst_multiplier": args.burst_multiplier}
    rec["base_rate"] = burst_base_rate(rec["capacity_rps"], args.burst_multiplier)
    with open(os.path.join(out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return rec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="AMS 560 end-to-end hardware experiment")
    p.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    p.add_argument("--sweep-seeds", type=int, nargs="+", default=None,
                   help="Seeds for the fixed-clock sweep (default: --seeds)")
    p.add_argument("--quick", action="store_true",
                   help="One seed (42), sharp_burst only")
    p.add_argument("--primary-only", action="store_true",
                   help="Only the 24 primary conditions (naive_slo, robust x 4 delays "
                        "x 3 workloads) plus static_high per workload as a same-session "
                        "reference (27 runs per seed); the proposal run uses 3 seeds")
    p.add_argument("--smoke", action="store_true",
                   help="~15 min code check: reuse the latest sweep + benchmark, run a fresh "
                        "probe, then 5 sharp_burst runs (naive_slo/robust at 0 and 2500 ms, "
                        "static_high). Writes results/smoke_N. Not for the report.")
    p.add_argument("--repeat-1seed", type=int, default=0, metavar="M",
                   help="Repeat mode: M independent --primary-only runs with --repeat-seed")
    p.add_argument("--repeat-3seed", type=int, default=0, metavar="N",
                   help="Repeat mode: N independent --primary-only runs with --repeat-seeds")
    p.add_argument("--repeat-seed", type=int, default=42,
                   help="Seed for the 1-seed repeats (default 42)")
    p.add_argument("--repeat-seeds", type=int, nargs="+", default=[41, 42, 43],
                   help="Seeds for the 3-seed repeats (default 41 42 43)")
    p.add_argument("--sweep-from", default=None,
                   help="Reuse sweep_runs.csv + sweep/ from an earlier attempt folder")
    p.add_argument("--skip-gpu-check", action="store_true")
    p.add_argument("--no-auto-config", dest="auto_config", action="store_false",
                   help="Use --model / --batch-size / --base-rate as given instead of "
                        "choosing them with bench_clock_sensitivity.py")
    p.add_argument("--force", action="store_true",
                   help="Continue even if SLO calibration reports a blocking problem")
    p.add_argument("--model", default=D.model_id)
    p.add_argument("--prompts-file", default=D.prompts_file)
    p.add_argument("--batch-size", type=int, default=D.max_batch_size)
    p.add_argument("--base-rate", type=float, default=D.base_arrival_rate)
    p.add_argument("--burst-multiplier", type=float, default=D.burst_multiplier)
    p.add_argument("--budget-ratio", type=float, default=D.slo_budget_ratio)
    p.add_argument("--duration", type=float, default=D.duration_s)
    p.add_argument("--warmup", type=float, default=D.warmup_s)
    p.add_argument("--results-dir", default=None, help="Default: next results/attempt_N")
    p.add_argument("--slo-target", type=float, default=FIXED_SLO_S,
                   help=f"Fixed latency SLO in seconds (default {FIXED_SLO_S}); the same "
                        "in every attempt so results can be pooled")
    p.add_argument("--calibrated-slo", action="store_true",
                   help="Use the per-attempt calibrated SLO instead of --slo-target")
    p.add_argument("--no-ablation", dest="ablation", action="store_false",
                   help="--primary-only without the robust_v1 / robust_no_age runs")
    args = p.parse_args(argv)
    if args.quick and args.primary_only:
        p.error("--quick and --primary-only cannot be combined")
    args.repeat = args.repeat_1seed > 0 or args.repeat_3seed > 0
    if args.repeat:
        if args.repeat_1seed < 0 or args.repeat_3seed < 0:
            p.error("repeat counts must be >= 0")
        if args.smoke or args.quick or args.sweep_from or args.results_dir:
            p.error("repeat mode cannot be combined with --smoke, --quick, --sweep-from "
                    "or --results-dir (each repeat makes its own attempt folder and sweep)")
        return args
    if args.smoke:
        if args.quick or args.primary_only:
            p.error("--smoke cannot be combined with --quick or --primary-only")
        args.seeds = [42]
        args.skip_gpu_check = True
        if args.sweep_from is None:
            args.sweep_from = latest_sweep_dir(os.path.join(ROOT, "results"))
            if args.sweep_from is None:
                p.error("--smoke needs a finished sweep (results/attempt_N/sweep_summary.csv) "
                        "or --sweep-from")
    if args.quick:
        args.seeds = [42]
    if args.primary_only and len(set(args.seeds)) < 3:
        print(f"NOTE: --primary-only with {len(set(args.seeds))} seed(s). The proposal "
              "run needs 3 paired seeds (default --seeds 41 42 43); this is a pilot.")
    if args.sweep_seeds is None:
        args.sweep_seeds = args.seeds
    return args


def next_attempt_dir(base: str = "results", prefix: str = "attempt") -> str:
    os.makedirs(base, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(base, f"{prefix}_{n}")):
        n += 1
    return os.path.join(base, f"{prefix}_{n}")


def latest_sweep_dir(base: str = "results"):
    """Newest results/attempt_N that has a finished sweep and benchmark, or None."""
    best = None
    if not os.path.isdir(base):
        return None
    for name in os.listdir(base):
        d = os.path.join(base, name)
        if not name.startswith("attempt_") or not name[8:].isdigit():
            continue
        if all(os.path.exists(os.path.join(d, f))
               for f in ("sweep_summary.csv", "sweep_runs.csv", "clock_sensitivity.json")):
            if best is None or int(name[8:]) > best[0]:
                best = (int(name[8:]), d)
    return best[1] if best else None


# Smoke test: the cheapest set that still exercises every new code path
# (scheduled arrivals + burst, capacity floor, delay-aware robust at the
# largest delay, static reference).
SMOKE_MATRIX = [("naive_slo", 0), ("robust", 0), ("naive_slo", 2500), ("robust", 2500),
                ("robust_v1", 2500), ("static_high", 0)]


PRIMARY_CONTROLLERS = ("naive_slo", "robust")
# Ablations on sharp_burst: robust_v1 = robust without the age-compensated
# forecast and lag-aware P95 (the attempt_8-11 controller); robust_no_age =
# robust that does not know the telemetry age.
ABLATION_CONTROLLERS = ("robust_v1", "robust_no_age")
FIXED_SLO_S = 5.0
ALL_WORKLOADS = ("steady", "ramp", "sharp_burst")


def build_matrix(seeds, quick: bool, primary_only: bool = False, smoke: bool = False,
                 ablation: bool = True):
    """
    List of (seed, controller, delay_ms, workload); order shuffled per seed.
    Every seed gets the same conditions, so seeds are paired.
    primary_only: the proposal's 24 primary conditions per seed
    (2 dynamic controllers x 4 delays x 3 workloads) plus static_high per
    workload, a same-session reference measured under the same machine state.
    """
    if smoke:
        return [(seeds[0], c, d, "sharp_burst") for c, d in SMOKE_MATRIX]
    workloads = ["sharp_burst"] if quick else list(ALL_WORKLOADS)
    runs = []
    for seed in seeds:
        block = []
        for w in workloads:
            for d in DELAY_VALUES_MS:
                for c in PRIMARY_CONTROLLERS:
                    block.append((seed, c, d, w))
            if not primary_only:
                block.append((seed, "static_efficient", 0, w))
        if primary_only:
            for w in workloads:
                block.append((seed, "static_high", 0, w))
            if ablation:
                for c in ABLATION_CONTROLLERS:
                    for d in DELAY_VALUES_MS:
                        block.append((seed, c, d, "sharp_burst"))
            random.Random(seed).shuffle(block)
            runs.extend(block)
            continue
        for c in ABLATION_CONTROLLERS:
            for d in DELAY_VALUES_MS:
                block.append((seed, c, d, "sharp_burst"))
        block.append((seed, "oracle", 0, "sharp_burst"))
        random.Random(seed).shuffle(block)
        runs.extend(block)
    return runs


def summarize_paired(results_dir: str, seeds, quick: bool, primary_only: bool,
                     smoke: bool = False, ablation: bool = True) -> str:
    """
    Write paired_summary.csv: mean and std over seeds of the key metrics for
    every planned condition, plus how many seeds actually completed. Returns
    the path. Conditions missing a seed are listed so the gap is visible.
    """
    import pandas as pd

    df = pd.read_csv(os.path.join(results_dir, "run_summary.csv"))
    df = df[~df["run_id"].astype(str).str.startswith("sweep_")]
    planned = {(c, d, w) for (_, c, d, w) in build_matrix(seeds, quick, primary_only, smoke, ablation)}
    keys = ["controller", "delay_ms", "workload"]
    metrics = ["j_per_token", "miss_rate", "p95", "mean_power_w", "n_clock_changes",
               "clock_tracking_pct"]
    metrics = [m for m in metrics if m in df.columns]
    df = df[df["seed"].isin(seeds)]
    g = df.groupby(keys)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    out = out.reset_index()
    for c, d, w in sorted(planned):
        if not ((out.controller == c) & (out.delay_ms == d) & (out.workload == w)).any():
            out.loc[len(out)] = {"controller": c, "delay_ms": d, "workload": w, "n_seeds": 0}
    out = out.sort_values(keys)
    path = os.path.join(results_dir, "paired_summary.csv")
    out.to_csv(path, index=False)

    incomplete = out[out["n_seeds"] < len(seeds)]
    print(f"\nPaired summary ({len(planned)} conditions x {len(seeds)} seeds) -> {path}")
    if incomplete.empty:
        print(f"  All {len(planned)} conditions have all {len(seeds)} seeds.")
    else:
        print(f"  {len(incomplete)} condition(s) missing seeds:")
        for _, r in incomplete.iterrows():
            print(f"    {r.controller} d{int(r.delay_ms)} {r.workload}: {int(r.n_seeds)}/{len(seeds)}")
    return path


def power_warnings() -> list:
    """Laptop power state that would throttle the CPU and distort timings."""
    warnings = []
    try:
        class _SPS(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                        ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                        ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
        sps = _SPS()
        if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps)):
            if sps.ACLineStatus == 0:
                warnings.append("Laptop is on BATTERY. Plug it in: CPU and GPU are throttled.")
            if sps.SystemStatusFlag == 1:
                warnings.append("Battery saver is ON. Turn it off.")
    except Exception:
        pass
    try:
        scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                                text=True).stdout.strip()
        print(f"  Power scheme: {scheme}")
        if "saver" in scheme.lower():
            warnings.append("Power saver plan is active. Use Balanced or High performance.")
    except Exception:
        pass
    return warnings


def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return os.name != "nt" and os.geteuid() == 0


class Tee:
    def __init__(self, path):
        self._log = open(path, "w", encoding="utf-8", buffering=1)
        self._out = sys.stdout

    def write(self, text):
        self._out.write(text)
        self._log.write(text)

    def flush(self):
        self._out.flush()
        self._log.flush()

    def fileno(self):
        return self._out.fileno()


def phase(msg):
    print("\n" + "=" * 70 + f"\n{msg}\n" + "=" * 70, flush=True)


def repeat_plan(args, base: str) -> list:
    """[(attempt_dir, seeds), ...]: 1-seed repeats first, then 3-seed repeats."""
    n = 1
    while os.path.exists(os.path.join(base, f"attempt_{n}")):
        n += 1
    seed_sets = ([[args.repeat_seed]] * args.repeat_1seed
                 + [list(args.repeat_seeds)] * args.repeat_3seed)
    return [(os.path.join(base, f"attempt_{n + i}"), seeds) for i, seeds in enumerate(seed_sets)]


def run_repeats(args) -> int:
    """Run each planned attempt as its own run_demo.py --primary-only process."""
    base = os.path.join(ROOT, "results")
    os.makedirs(base, exist_ok=True)
    plan = repeat_plan(args, base)
    passthrough = []
    for flag in ("skip_gpu_check", "force", "calibrated_slo"):
        if getattr(args, flag):
            passthrough.append("--" + flag.replace("_", "-"))
    if not args.ablation:
        passthrough.append("--no-ablation")
    passthrough += ["--slo-target", args.slo_target]
    if not args.auto_config:
        passthrough += ["--no-auto-config", "--model", args.model,
                        "--batch-size", args.batch_size, "--base-rate", args.base_rate]
    per_seed = 1.6 if args.ablation else 1.3
    hours = sum(0.3 + per_seed * len(sd) for _, sd in plan)
    print(f"Repeat mode: {len(plan)} independent runs, about {hours:.1f} h in total")
    for d, sd in plan:
        print(f"  {os.path.basename(d)}: seeds {sd}")
    t0 = time.monotonic()
    done = []
    for i, (d, sd) in enumerate(plan, 1):
        print(f"\n{'#' * 70}\n# Repeat {i}/{len(plan)}: {os.path.basename(d)}, seeds {sd}, "
              f"elapsed {(time.monotonic() - t0) / 3600:.1f} h\n{'#' * 70}", flush=True)
        cmd = [PYTHON, os.path.join(HERE, "run_demo.py"), "--primary-only", "--seeds", *sd,
               "--results-dir", d, *passthrough]
        rc = run(cmd)
        done.append((d, sd, rc))
        if rc == 2:
            print("Clock lock failed (Administrator rights lost?); stopping the repeats.")
            break
    print(f"\nRepeat mode finished in {(time.monotonic() - t0) / 3600:.1f} h:")
    for d, sd, rc in done:
        print(f"  {os.path.basename(d)}  seeds {sd}  {'OK' if rc == 0 else f'exit {rc}'}")
    for d, sd in plan[len(done):]:
        print(f"  {os.path.basename(d)}  seeds {sd}  NOT RUN")
    return 0 if all(rc == 0 for *_, rc in done) and len(done) == len(plan) else 1


def run(cmd, check=False) -> int:
    print("  $ " + " ".join(str(a) for a in cmd), flush=True)
    rc = subprocess.run([str(a) for a in cmd], cwd=ROOT).returncode
    if rc != 0:
        print(f"  WARNING: exit code {rc}", flush=True)
        if check:
            sys.exit(rc)
    return rc


def main(argv=None):
    args = parse_args(argv)
    if args.repeat:
        sys.exit(run_repeats(args))
    out = args.results_dir or next_attempt_dir(os.path.join(ROOT, "results"),
                                               "smoke" if args.smoke else "attempt")
    os.makedirs(out, exist_ok=True)
    sys.stdout = sys.stderr = Tee(os.path.join(out, "demo_run.log"))
    t0 = time.monotonic()

    print(f"Results folder : {out}")
    print(f"Started        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seeds          : {args.seeds}  (sweep: {args.sweep_seeds})")
    if args.smoke:
        print(f"Matrix         : SMOKE TEST - {len(SMOKE_MATRIX)} sharp_burst runs, sweep + "
              f"benchmark reused from {args.sweep_from}, fresh probe. Code check only.")
    if args.primary_only:
        print("Matrix         : PRIMARY ONLY - naive_slo, robust x "
              f"{DELAY_VALUES_MS} ms x {list(ALL_WORKLOADS)} x {len(args.seeds)} paired seeds"
              + (f" + {list(ABLATION_CONTROLLERS)} on sharp_burst" if args.ablation else ""))
    if args.auto_config:
        print("Load           : chosen by the clock-sensitivity benchmark (phase 1b)")
    else:
        print(f"Load           : {args.base_rate} req/s, burst x{args.burst_multiplier}, "
              f"batch {args.batch_size}, model {args.model}")

    # ── Phase 0 ──────────────────────────────────────────────────────────
    phase("PHASE 0: Environment check")
    if not is_admin():
        print("ERROR: clock locking needs an Administrator terminal. Re-run from an "
              "Administrator PowerShell.")
        sys.exit(1)
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], check=True)
    run([PYTHON, "-c", "import torch, pynvml, transformers; "
         "assert torch.cuda.is_available(), 'CUDA not available'; "
         "print('torch', torch.__version__, torch.cuda.get_device_name(0))"], check=True)
    for w in power_warnings():
        print(f"  WARNING: {w}")

    # ── Phase 1 ──────────────────────────────────────────────────────────
    if not args.skip_gpu_check:
        phase("PHASE 1: GPU timing check (clock control, telemetry, delays)")
        run([PYTHON, os.path.join(HERE, "check_gpu_timing.py")])
        src = os.path.join(ROOT, "results", "gpu_check", "gpu_check.json")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out, "gpu_check.json"))

    # ── Phase 1b ─────────────────────────────────────────────────────────
    if args.auto_config:
        phase("PHASE 1b: Clock-sensitivity benchmark (choose model / batch / rate)")
        bench_out = os.path.join(out, "clock_sensitivity.json")
        reused = args.sweep_from and os.path.exists(
            os.path.join(args.sweep_from, "clock_sensitivity.json"))
        if reused:
            # The reused sweep ran with that benchmark's model / batch / rate;
            # re-benchmarking could pick a different setup than the sweep.
            shutil.copy(os.path.join(args.sweep_from, "clock_sensitivity.json"), bench_out)
            print(f"Reusing benchmark from {args.sweep_from} (matches the reused sweep)")
        else:
            run([PYTHON, os.path.join(HERE, "bench_clock_sensitivity.py"), "--out", bench_out,
                 "--prompts-file", args.prompts_file])
        chosen = None
        if os.path.exists(bench_out):
            with open(bench_out, encoding="utf-8") as f:
                chosen = json.load(f).get("chosen")
        if chosen is None:
            print("\nStopping: no model / batch size makes decode speed depend on the GPU "
                  "clock, so DVFS has nothing to trade. See clock_sensitivity.json.")
            sys.exit(1)
        args.model = chosen["model"]
        args.batch_size = chosen["batch_size"]
        args.base_rate = chosen["base_rate"]
        print(f"Using {args.model}, batch {args.batch_size} "
              f"(clock sensitivity {chosen['sensitivity']:.2f}x)")

        # ── Phase 1c ─────────────────────────────────────────────────────
        phase("PHASE 1c: Top-clock capacity probe (real serving stack) -> base rate")
        old_probe = os.path.join(args.sweep_from, "probe.json") if args.sweep_from else ""
        if args.sweep_from and os.path.exists(old_probe) and not args.smoke:
            shutil.copy(old_probe, os.path.join(out, "probe.json"))
            if os.path.isdir(os.path.join(args.sweep_from, "probe")):
                shutil.copytree(os.path.join(args.sweep_from, "probe"),
                                os.path.join(out, "probe"), dirs_exist_ok=True)
            with open(old_probe, encoding="utf-8") as f:
                probe = json.load(f)
            print(f"Reusing probe from {args.sweep_from} (matches the reused sweep)")
        elif args.sweep_from and not args.smoke:
            probe = None
            print(f"{args.sweep_from} has no probe.json; keeping the benchmark base rate so "
                  "the load matches the reused sweep.")
        else:
            probe = probe_capacity(out, args, chosen)
        if probe:
            args.base_rate = probe["base_rate"]
            print(f"Top clock in the real stack: {probe['batch_time_s']:.2f} s per full batch "
                  f"= {probe['capacity_rps']:.2f} req/s (benchmark: "
                  f"{probe['bench_batch_time_s']:.2f} s). Base rate {args.base_rate} req/s, "
                  f"burst peak {args.base_rate * args.burst_multiplier:.2f} req/s.")
        elif not args.sweep_from:
            print("  WARNING: probe gave < 3 full batches; using the benchmark base rate "
                  f"{args.base_rate} req/s (may overload the burst).")

    common = ["--model", args.model, "--prompts-file", args.prompts_file,
              "--batch-size", args.batch_size,
              "--base-rate", args.base_rate, "--burst-multiplier", args.burst_multiplier,
              "--budget-ratio", args.budget_ratio, "--duration", args.duration,
              "--warmup", args.warmup, "--results-dir", out]

    # ── Phase 2 ──────────────────────────────────────────────────────────
    if args.sweep_from:
        phase(f"PHASE 2: Reusing sweep from {args.sweep_from}")
        shutil.copy(os.path.join(args.sweep_from, "sweep_runs.csv"), out)
        shutil.copy(os.path.join(args.sweep_from, "sweep_summary.csv"), out)
        shutil.copytree(os.path.join(args.sweep_from, "sweep"), os.path.join(out, "sweep"),
                        dirs_exist_ok=True)
    else:
        phase("PHASE 2: Fixed-clock sweep (sharp_burst, every verified clock)")
        run([PYTHON, os.path.join(HERE, "run_experiment.py"), "--sweep-clocks", "--workload", "sharp_burst",
             "--seeds", *args.sweep_seeds, *common], check=True)

    # ── Capacity floor for the dynamic controllers ───────────────────────
    capacity_file = os.path.join(out, "capacity.json")
    bench_path = os.path.join(out, "clock_sensitivity.json")
    if os.path.exists(bench_path):
        run([PYTHON, os.path.join(HERE, "capacity_model.py"), "--bench", bench_path,
             "--sweep-dir", os.path.join(out, "sweep"),
             "--probe-csv", os.path.join(out, "probe", "probe_top_requests.csv"),
             "--out", capacity_file])
    if not os.path.exists(capacity_file):
        capacity_file = None
        print("  WARNING: no clock_sensitivity.json, so no capacity floor; dynamic "
              "controllers may step down to clocks that cannot keep up with the load.")

    # ── Phase 3 ──────────────────────────────────────────────────────────
    phase("PHASE 3: SLO calibration")
    rc = run([PYTHON, os.path.join(HERE, "calibrate_slo.py"), "--results-dir", out,
              "--budget-ratio", args.budget_ratio])
    cal_path = os.path.join(out, "slo_calibration.json")
    if not os.path.exists(cal_path):
        sys.exit(1)
    with open(cal_path, encoding="utf-8") as f:
        cal = json.load(f)
    if rc != 0 and not args.force:
        print("\nStopping: the testbed cannot support the controller comparison yet "
              "(see BLOCKING above). Adjust --batch-size / --base-rate / --model and "
              "re-run, or pass --force to continue anyway.")
        sys.exit(1)
    eff = cal["static_efficient_mhz"]
    if args.calibrated_slo:
        slo = cal["slo_target_s"]
        print(f"Using calibrated SLO {slo:.1f} s")
    else:
        slo = args.slo_target
        print(f"Using fixed SLO {slo:.1f} s (calibration suggested "
              f"{cal['slo_target_s']:.1f} s; pass --calibrated-slo to use it)")
    print(f"Budget {args.budget_ratio:.0%}, static_efficient {eff} MHz")

    # ── Phase 4 ──────────────────────────────────────────────────────────
    phase("PHASE 4: Oracle lookup table")
    oracle_table = os.path.join(out, "oracle_table.csv")
    run([PYTHON, os.path.join(HERE, "oracle_builder.py"), "--sweep-csv", os.path.join(out, "sweep_summary.csv"),
         "--slo-target", slo, "--out", oracle_table])

    # ── Phase 5 ──────────────────────────────────────────────────────────
    matrix = build_matrix(args.seeds, args.quick, args.primary_only, args.smoke, args.ablation)
    phase(f"PHASE 5: Controller matrix ({len(matrix)} runs)")
    t5 = time.monotonic()
    failed = []
    for i, (seed, ctrl, delay, workload) in enumerate(matrix, 1):
        run_id = f"{ctrl}_d{delay}_{workload}_s{seed}"
        el = (time.monotonic() - t5) / 60
        eta = el / (i - 1) * (len(matrix) - i + 1) if i > 1 else float("nan")
        print(f"\n  [{i}/{len(matrix)}] {run_id}   elapsed {el:.0f} min, ETA {eta:.0f} min",
              flush=True)
        cmd = [PYTHON, os.path.join(HERE, "run_experiment.py"), "--controller", ctrl, "--delay-ms", delay,
               "--workload", workload, "--seed", seed, "--slo-target", slo,
               "--static-efficient-mhz", eff, "--run-id", run_id, *common]
        if ctrl == "oracle":
            cmd += ["--oracle-table", oracle_table]
        if capacity_file:
            cmd += ["--capacity-file", capacity_file]
        rc = run(cmd)
        if rc == 2:
            print("Clock lock failed; stopping (Administrator rights lost?).")
            sys.exit(2)
        if rc != 0:
            failed.append(run_id)

    # ── Phase 6 ──────────────────────────────────────────────────────────
    phase("PHASE 6: Scale-out simulation and figures")
    if not args.smoke:
        run([PYTHON, os.path.join(HERE, "scale_simulator.py"), "--results-dir", out, "--slo-target", slo,
             "--base-rate", args.base_rate, "--n-workers", 1, 8, "--seeds", 20])
    run([PYTHON, os.path.join(HERE, "analyze_results.py"), "--results-dir", out, "--slo-target", slo,
         "--workload", "sharp_burst"])
    try:
        summarize_paired(out, args.seeds, args.quick, args.primary_only, args.smoke,
                         args.ablation)
    except Exception as exc:  # never lose a finished run to a summary bug
        print(f"  WARNING: paired summary failed: {exc}")

    print(f"\nDONE in {(time.monotonic() - t0) / 60:.0f} min -> {out}")
    if failed:
        print(f"Failed runs ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
