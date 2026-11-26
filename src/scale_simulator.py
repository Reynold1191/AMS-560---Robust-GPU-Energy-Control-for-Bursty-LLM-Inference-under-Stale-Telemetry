"""
scale_simulator.py
Scale-out simulation (Deliverable D10), built on serving_sim.py.

For every controller x delay x workload and n_workers in --n-workers, runs
--seeds paired simulated runs of the real serving stack (batched workers,
measured batch times and power, every controller input delayed) and writes

  sim_runs.csv     one row per simulated run
  sim_summary.csv  mean over seeds per condition, plus the paired
                   robust - naive differences and how many seeds robust won

The model reads capacity.json from --results-dir (the same in-stack
capacity table the GPU runs used), so it is calibrated to that attempt.
n_workers > 1 is a fleet sharing one clock policy: arrival rate and
capacity scale with n_workers, requests join the shortest queue.

The previous version modelled single-request exponential service with fresh
queue/arrival signals and no capacity floor; every controller and delay gave
identical numbers (attempt_8-11 sim_summary.csv).

Usage:
  python src/scale_simulator.py --results-dir results/attempt_11 --slo-target 5.0 \\
      --base-rate 1.75 --n-workers 1 8 --seeds 20
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import DELAY_VALUES_MS  # noqa: E402
from serving_sim import SimModel, make_cfg, simulate_run  # noqa: E402

CONTROLLERS = ("naive_slo", "robust", "robust_v1", "robust_no_age")
WORKLOADS = ("steady", "ramp", "sharp_burst")
METRICS = ("j_per_token", "miss_rate", "burst_miss_rate", "p95", "n_clock_changes")


def run_grid(model, slo, base_rate, n_workers_list, seeds, controllers=CONTROLLERS,
             workloads=WORKLOADS, delays=DELAY_VALUES_MS):
    import pandas as pd
    rows = []
    for nw in n_workers_list:
        for w in workloads:
            for c in controllers:
                for d in delays:
                    for s in seeds:
                        r = simulate_run(make_cfg(c, d, w, s, slo, base_rate), model,
                                         n_workers=nw, keep_trace=True)
                        r.pop("trace", None)
                        rows.append(r)
    return pd.DataFrame(rows)


def summarize(runs):
    import pandas as pd
    keys = ["n_workers", "workload", "delay_ms"]
    mean = runs.groupby(keys + ["controller"])[list(METRICS)].mean().reset_index()
    mean["n_seeds"] = runs.groupby(keys + ["controller"]).size().values
    idx = keys + ["seed"]
    naive = runs[runs.controller == "naive_slo"].set_index(idx)
    rob = runs[runs.controller == "robust"].set_index(idx)
    if naive.empty or rob.empty:
        return mean
    j = rob.join(naive, lsuffix="_r", rsuffix="_n", how="inner")
    p = pd.DataFrame({
        "d_j_per_token_pct": (j.j_per_token_r / j.j_per_token_n - 1) * 100,
        "d_miss_pp": (j.miss_rate_r - j.miss_rate_n) * 100,
        "robust_fewer_misses": (j.miss_rate_r < j.miss_rate_n).astype(int),
        "robust_more_misses": (j.miss_rate_r > j.miss_rate_n).astype(int),
    }).reset_index().groupby(keys).agg(
        d_j_per_token_pct=("d_j_per_token_pct", "mean"),
        d_miss_pp=("d_miss_pp", "mean"),
        robust_fewer_misses=("robust_fewer_misses", "sum"),
        robust_more_misses=("robust_more_misses", "sum")).reset_index()
    p["controller"] = "robust_vs_naive"
    return pd.concat([mean, p], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Scale-out simulation of the serving stack")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--slo-target", type=float, default=5.0)
    ap.add_argument("--base-rate", type=float, default=1.75)
    ap.add_argument("--n-workers", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--seeds", type=int, default=20, help="number of paired seeds")
    ap.add_argument("--controllers", nargs="+", default=list(CONTROLLERS))
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    args = ap.parse_args(argv)
    logging.disable(logging.CRITICAL)

    model = SimModel.from_results(args.results_dir)
    t0 = time.monotonic()
    runs = run_grid(model, args.slo_target, args.base_rate, args.n_workers,
                    range(1000, 1000 + args.seeds), args.controllers, args.workloads)
    os.makedirs(args.results_dir, exist_ok=True)
    runs.to_csv(os.path.join(args.results_dir, "sim_runs.csv"), index=False)
    summ = summarize(runs)
    path = os.path.join(args.results_dir, "sim_summary.csv")
    summ.to_csv(path, index=False)
    print(f"{len(runs)} simulated runs in {time.monotonic() - t0:.0f} s -> {path}")
    cmp = summ[summ.controller == "robust_vs_naive"]
    if not cmp.empty:
        print("\nrobust vs naive_slo (paired over seeds):")
        print(cmp[["n_workers", "workload", "delay_ms", "d_j_per_token_pct", "d_miss_pp",
                   "robust_fewer_misses", "robust_more_misses"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
