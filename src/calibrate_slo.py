"""
calibrate_slo.py
Choose a feasible SLO and the static-efficient clock from the fixed-clock sweep.

Rule (written down before looking at controller results):
  * SLO target = 1.5 x the worst per-seed P95 at the top verified clock,
    rounded up to 0.5 s.  The fastest clock must comfortably meet it; slower
    clocks may not, which is what gives a controller something to decide.
  * The SLO is feasible if the top clock's miss rate (unfinished requests
    count as misses) is within the error budget.
  * static_efficient = the lowest-J/token clock whose miss rate is within
    the budget at that SLO.

It also checks the testbed itself:
  * clock tracking  - did the GPU actually run at the locked clock?
  * clock sensitivity - does service time fall as the clock rises?  If the
    lowest clock is barely slower than the top one, DVFS has nothing to
    trade and the controller comparison is meaningless.
  * utilisation      - is the GPU busy enough that energy is not just idle power?

Usage:
  python src/calibrate_slo.py --results-dir results/attempt_4 --budget-ratio 0.05
Writes <results-dir>/slo_calibration.json.  Exit code 1 if a blocking check fails.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List

# Thresholds for the testbed checks
MIN_CLOCK_TRACKING_PCT = 90.0
MIN_CLOCK_SENSITIVITY = 1.5     # service time ratio, lowest clock / top clock
MIN_TOP_UTIL_PCT = 50.0
SLO_FACTOR = 1.5


def _read_csv(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def miss_rate_at(results_dir: str, run_id: str, slo_s: float, n_unfinished: int) -> float:
    """Recompute a sweep run's miss rate at a new SLO from its request log."""
    path = os.path.join(results_dir, "sweep", f"{run_id}_requests.csv")
    lats = [_f(r["latency_s"]) for r in _read_csv(path)] if os.path.exists(path) else []
    lats = [x for x in lats if x == x]
    total = len(lats) + n_unfinished
    if total == 0:
        return float("nan")
    return (sum(1 for x in lats if x > slo_s) + n_unfinished) / total


def calibrate(runs: List[dict], results_dir: str, budget_ratio: float) -> dict:
    by_clock: Dict[int, List[dict]] = {}
    for r in runs:
        by_clock.setdefault(int(_f(r["clock_mhz"])), []).append(r)
    clocks = sorted(by_clock)
    if len(clocks) < 2:
        raise ValueError("Need at least two sweep clocks to calibrate.")

    def mean(k, mhz):
        vals = [_f(r.get(k)) for r in by_clock[mhz]]
        vals = [v for v in vals if v == v]
        return sum(vals) / len(vals) if vals else float("nan")

    # ── Testbed checks ────────────────────────────────────────────────────
    tracking = {m: mean("clock_tracking_pct", m) for m in clocks}
    bad_tracking = [m for m in clocks
                    if tracking[m] == tracking[m] and tracking[m] < MIN_CLOCK_TRACKING_PCT]
    usable = [m for m in clocks if m not in bad_tracking]
    if len(usable) < 2:
        raise ValueError(f"Fewer than two clocks tracked their lock: {tracking}")
    top, low = usable[-1], usable[0]

    svc = {m: mean("mean_service_s", m) for m in usable}
    sensitivity = svc[low] / svc[top] if svc[top] > 0 else float("nan")
    top_util = mean("mean_util_pct", top)

    # ── SLO from the top clock ────────────────────────────────────────────
    worst_top_p95 = max(_f(r["p95"]) for r in by_clock[top])
    slo = math.ceil(SLO_FACTOR * worst_top_p95 * 2) / 2.0

    per_clock = []
    for m in usable:
        misses = [miss_rate_at(results_dir, r["run_id"], slo, int(_f(r.get("n_unfinished"), 0)))
                  for r in by_clock[m]]
        misses = [x for x in misses if x == x]
        per_clock.append({
            "clock_mhz": m,
            "n_seeds": len(by_clock[m]),
            "p95_mean_s": mean("p95", m),
            "mean_service_s": svc[m],
            "j_per_token": mean("j_per_token", m),
            "mean_power_w": mean("mean_power_w", m),
            "mean_util_pct": mean("mean_util_pct", m),
            "clock_tracking_pct": tracking[m],
            "miss_rate_at_slo": sum(misses) / len(misses) if misses else float("nan"),
        })

    within = [c for c in per_clock if c["miss_rate_at_slo"] <= budget_ratio]
    top_row = next(c for c in per_clock if c["clock_mhz"] == top)
    feasible = top_row["miss_rate_at_slo"] <= budget_ratio
    efficient = min(within, key=lambda c: c["j_per_token"])["clock_mhz"] if within else top

    p95s = [c["p95_mean_s"] for c in per_clock]
    monotonic = all(b <= a * 1.10 for a, b in zip(p95s, p95s[1:]))

    blocking, warnings = [], []
    if bad_tracking:
        warnings.append(f"Clocks {bad_tracking} did not track their lock and were dropped.")
    if not (sensitivity >= MIN_CLOCK_SENSITIVITY):
        blocking.append(
            f"Service time at {low} MHz is only {sensitivity:.2f}x the time at {top} MHz "
            f"(need >= {MIN_CLOCK_SENSITIVITY}). Latency barely depends on the clock, so "
            "DVFS has nothing to trade. Increase --batch-size or use a larger model.")
    if not (top_util >= MIN_TOP_UTIL_PCT):
        warnings.append(
            f"Mean GPU utilisation at {top} MHz is {top_util:.0f}% (< {MIN_TOP_UTIL_PCT:.0f}%). "
            "Energy will be dominated by idle power; consider a higher --base-rate.")
    if not monotonic:
        warnings.append("P95 does not fall monotonically with clock (>10% inversions); "
                        "check per-seed spread before trusting the curve.")
    if not feasible:
        blocking.append(f"Even {top} MHz misses {top_row['miss_rate_at_slo']:.1%} of requests "
                        f"at SLO {slo:.1f} s (budget {budget_ratio:.0%}).")

    return {
        "rule": f"SLO = ceil_0.5({SLO_FACTOR} x worst per-seed P95 at top clock)",
        "slo_target_s": slo,
        "budget_ratio": budget_ratio,
        "top_clock_mhz": top,
        "static_efficient_mhz": efficient,
        "feasible": feasible,
        "clock_sensitivity": sensitivity,
        "top_clock_util_pct": top_util,
        "p95_monotonic": monotonic,
        "usable_clocks": usable,
        "per_clock": per_clock,
        "blocking": blocking,
        "warnings": warnings,
        "ok": not blocking,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--results-dir", required=True)
    p.add_argument("--budget-ratio", type=float, default=0.05)
    args = p.parse_args(argv)

    runs_path = os.path.join(args.results_dir, "sweep_runs.csv")
    if not os.path.exists(runs_path):
        print(f"ERROR: {runs_path} not found. Run: python src/run_experiment.py --sweep-clocks "
              f"--results-dir {args.results_dir}")
        return 1
    result = calibrate(_read_csv(runs_path), args.results_dir, args.budget_ratio)

    out = os.path.join(args.results_dir, "slo_calibration.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nSLO calibration ({result['rule']})")
    print(f"  {'MHz':>5} {'P95':>7} {'svc':>6} {'J/tok':>7} {'util':>5} {'clk ok':>6} {'miss@SLO':>8}")
    for c in result["per_clock"]:
        print(f"  {c['clock_mhz']:>5} {c['p95_mean_s']:>6.2f}s {c['mean_service_s']:>5.2f}s "
              f"{c['j_per_token']:>7.3f} {c['mean_util_pct']:>4.0f}% {c['clock_tracking_pct']:>5.0f}% "
              f"{c['miss_rate_at_slo']:>8.1%}")
    print(f"\n  SLO target        : {result['slo_target_s']:.1f} s  (budget {result['budget_ratio']:.0%})")
    print(f"  static_efficient  : {result['static_efficient_mhz']} MHz")
    print(f"  clock sensitivity : {result['clock_sensitivity']:.2f}x")
    for w in result["warnings"]:
        print(f"  WARNING: {w}")
    for b in result["blocking"]:
        print(f"  BLOCKING: {b}")
    print(f"  -> {out}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
