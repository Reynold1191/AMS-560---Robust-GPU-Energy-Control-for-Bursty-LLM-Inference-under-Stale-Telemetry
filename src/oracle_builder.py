"""
oracle_builder.py
Build the oracle lookup table from the fixed-frequency sweep results.

The oracle controller (controllers/robust_delay_budget.py oracle_mode=True)
looks up: for each workload bucket, what is the minimum clock that keeps
P95 latency under the SLO target?

This script reads results/sweep_summary.csv and writes
results/oracle_table.csv with columns:
  bucket, clock_mhz, p95_s, mean_power_w, feasible

Usage:
  python src/oracle_builder.py --slo-target 2.0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def build_oracle_table(
    sweep_path: str,
    slo_target_s: float = 2.0,
    out_path: str = "results/oracle_table.csv",
) -> None:
    if not os.path.exists(sweep_path):
        print(f"ERROR: sweep CSV not found: {sweep_path}")
        print("Run  python src/run_experiment.py --sweep-clocks  first.")
        sys.exit(1)

    rows = []
    with open(sweep_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mhz     = int(float(row.get("clock_mhz", 0)))
            p95     = float(row.get("p95", row.get("p95_latency_s", 99.0)))
            mean_pw = float(row.get("mean_power_w", float("nan")))
            feasible = p95 < slo_target_s

            # Single sweep = steady-light bucket only; copy to other buckets
            for bucket in ["steady_light", "steady_heavy", "burst_light", "burst_heavy"]:
                # For burst buckets assume 20% latency penalty
                penalty = 1.2 if "burst" in bucket else 1.0
                rows.append({
                    "bucket":       bucket,
                    "clock_mhz":    mhz,
                    "p95_s":        p95 * penalty,
                    "mean_power_w": mean_pw,
                    "feasible":     int(p95 * penalty < slo_target_s),
                })

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bucket", "clock_mhz", "p95_s",
                                           "mean_power_w", "feasible"])
        w.writeheader()
        w.writerows(rows)

    # Report minimum feasible clock per bucket
    from collections import defaultdict
    feasible_clocks = defaultdict(list)
    for r in rows:
        if r["feasible"]:
            feasible_clocks[r["bucket"]].append(r["clock_mhz"])

    print(f"\nOracle table saved → {out_path}")
    print(f"SLO target: {slo_target_s} s")
    for bucket, clks in sorted(feasible_clocks.items()):
        min_clk = min(clks) if clks else "NONE"
        print(f"  {bucket:<20s}: minimum feasible clock = {min_clk} MHz")


def main():
    p = argparse.ArgumentParser(description="Build oracle lookup table")
    p.add_argument("--sweep-csv",   default="results/sweep_summary.csv")
    p.add_argument("--slo-target",  type=float, default=2.0)
    p.add_argument("--out",         default="results/oracle_table.csv")
    args = p.parse_args()
    build_oracle_table(args.sweep_csv, args.slo_target, args.out)


if __name__ == "__main__":
    main()
