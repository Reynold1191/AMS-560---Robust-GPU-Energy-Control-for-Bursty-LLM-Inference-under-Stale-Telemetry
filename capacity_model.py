"""
capacity_model.py
Per-clock serving capacity, used as a feasibility floor by the dynamic
controllers.

Why: in attempt_6 every dynamic controller saw an empty queue and a low
(lagging) P95 in the first seconds and stepped down to 180 MHz, where one
batch of 16 takes ~18 s and the GPU serves ~0.9 req/s against 2.35 req/s
arriving. P95 only reacts when requests finish, so the controllers noticed
25-30 s later and most SLO misses came from that, not from the burst.

Model: full-batch service time  t(f) = A + B / f  (fixed overhead plus work
that scales with the clock). Capacity(f) = batch / t(f).

Fit points, in order of preference:
  1. Full batches measured inside the real serving stack: the fixed-clock
     sweep (one CSV per clock) and the top-clock probe run. attempt_7 showed
     the isolated benchmark is optimistic: 2.0 s per batch at 2122 MHz in the
     benchmark, 2.6-2.9 s in the experiment runs (decode is CPU-bound, and
     the runner's sampler / generator / control threads share the CPU).
  2. The isolated clock-sensitivity benchmark (fallback).

Usage:
    python capacity_model.py --bench results/attempt_8/clock_sensitivity.json \
        --sweep-dir results/attempt_8/sweep --probe-csv results/attempt_8/probe/probe_top_requests.csv \
        --out results/attempt_8/capacity.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fit_service_model(points: Dict[int, float]) -> tuple:
    """Least-squares fit of t = A + B/f to {clock_mhz: batch_time_s}."""
    if len(points) < 2:
        raise ValueError("need batch times at >= 2 clocks")
    xs = [1.0 / f for f in points]
    ys = list(points.values())
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    return max(a, 0.0), b


def full_batch_times(requests_csv: str, batch: int) -> List[float]:
    """Service times of the FULL batches (exactly ``batch`` requests) in a run."""
    import csv
    batches: Dict[str, list] = {}
    with open(requests_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("phase") == "warmup" or not row.get("start_t"):
                continue
            batches.setdefault(row["start_t"], []).append(float(row["service_time_s"]))
    return [v[0] for v in batches.values() if len(v) == batch]


def measured_points(batch: int, sweep_dir: Optional[str] = None,
                    probe_csv: Optional[str] = None, probe_mhz: Optional[int] = None,
                    min_batches: int = 3) -> Dict[int, float]:
    """{clock_mhz: median full-batch time} from in-stack runs."""
    import glob
    import re
    import statistics
    times: Dict[int, list] = {}
    if sweep_dir and os.path.isdir(sweep_dir):
        for path in glob.glob(os.path.join(sweep_dir, "sweep_*mhz_s*_requests.csv")):
            m = re.search(r"sweep_(\d+)mhz_", os.path.basename(path))
            if m:
                times.setdefault(int(m.group(1)), []).extend(full_batch_times(path, batch))
    if probe_csv and probe_mhz and os.path.exists(probe_csv):
        times.setdefault(int(probe_mhz), []).extend(full_batch_times(probe_csv, batch))
    return {c: statistics.median(v) for c, v in times.items() if len(v) >= min_batches}


def build_capacity(bench: dict, clocks: List[int],
                   measured: Optional[Dict[int, float]] = None) -> dict:
    """
    Capacity table for ``clocks``. Uses in-stack ``measured`` points when
    they cover >= 2 clocks, else the benchmark in clock_sensitivity.json.
    """
    chosen = bench.get("chosen")
    if not chosen:
        raise ValueError("benchmark has no chosen setup")
    batch = int(chosen["batch_size"])
    if measured and len(measured) >= 2:
        points, source = dict(measured), "in-stack full batches (sweep + probe)"
    else:
        points = {int(float(k)): v["batch_time_s"] for k, v in chosen["per_clock"].items()}
        source = "isolated benchmark"
    a, b = fit_service_model(points)
    # Measured clocks use their own median: near the top the curve flattens
    # (CPU-bound decode), which A + B/f misses (attempt_7: model 2.28 s vs
    # measured 2.68 s at 2122 MHz). The fit only fills clocks not measured.
    times = [points[f] if f in points else a + b / f for f in clocks]
    return {
        "source": source,
        "model": "batch_time_s = A + B / clock_mhz (full batch)",
        "A_s": a, "B_s_mhz": b,
        "batch_size": batch,
        "fit_points": points,
        "clocks_mhz": list(clocks),
        "batch_time_s": times,
        "capacity_rps": [batch / t for t in times],
    }


def load_capacity(path: Optional[str], clocks: List[int]):
    """(capacity_rps, batch_time_s) aligned with ``clocks``, or (None, None)."""
    if not path or not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        cap = json.load(f)
    by_clock = dict(zip(cap["clocks_mhz"], zip(cap["capacity_rps"], cap["batch_time_s"])))
    if not all(c in by_clock for c in clocks):
        a, b, batch = cap["A_s"], cap["B_s_mhz"], cap["batch_size"]
        return [batch / (a + b / c) for c in clocks], [a + b / c for c in clocks]
    return [by_clock[c][0] for c in clocks], [by_clock[c][1] for c in clocks]


def main(argv=None) -> int:
    from config import VERIFIED_CLOCKS_MHZ
    p = argparse.ArgumentParser(description="Per-clock capacity from the clock benchmark")
    p.add_argument("--bench", required=True, help="clock_sensitivity.json")
    p.add_argument("--sweep-dir", default=None, help="Folder with sweep_*mhz_s*_requests.csv")
    p.add_argument("--probe-csv", default=None, help="Top-clock probe requests CSV")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    with open(args.bench, encoding="utf-8") as f:
        bench = json.load(f)
    measured = measured_points(int(bench["chosen"]["batch_size"]), args.sweep_dir,
                               args.probe_csv, VERIFIED_CLOCKS_MHZ[-1])
    cap = build_capacity(bench, VERIFIED_CLOCKS_MHZ, measured)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(cap, f, indent=2)
    print(f"Service model ({cap['source']}): t = {cap['A_s']:.3f} + {cap['B_s_mhz']:.0f}/f s "
          f"(batch {cap['batch_size']}); fit points "
          + ", ".join(f"{c} MHz={t:.2f}s" for c, t in sorted(cap['fit_points'].items())))
    for c, t, r in zip(cap["clocks_mhz"], cap["batch_time_s"], cap["capacity_rps"]):
        print(f"  {c:>5} MHz: batch {t:6.2f} s  capacity {r:5.2f} req/s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
