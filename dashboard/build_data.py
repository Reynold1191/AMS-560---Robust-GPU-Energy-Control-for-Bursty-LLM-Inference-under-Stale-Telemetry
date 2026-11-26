"""
dashboard/build_data.py
Pack one results/attempt_N folder into a single JSON file for the dashboard.

  python dashboard/build_data.py                       # results/attempt_12
  python dashboard/build_data.py --attempt results/attempt_13
  python dashboard/build_data.py --attempt results/attempt_13 --seed 41   # one seed only

Writes dashboard/data/<attempt>.js (window.DASHBOARD_DATA["<attempt>"] = {...}),
or <attempt>_s<seed>.js with --seed. With --seed, figures are read from
dashboard/figures/<attempt>_s<seed>/ when that folder exists (analyze_results.py
run on that seed alone), otherwise from results/<attempt>/.
It is a script rather than a .json file so the dashboard also works when
index.html is opened straight from disk (browsers block fetch() on file://).
The data holds
  meta     probe, capacity table, SLO, matrix size
  runs     one row per GPU run (energy, misses, burst-phase misses, clock use)
  traces   per run: controller decisions, GPU power/clock, every request
  sweep    fixed-clock sweep (J/token and P95 per clock)
  sim      scale_simulator.py summary (1 and 8 workers)
"""

from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOP_MHZ = 2122


def _clean(x):
    """JSON-safe floats (NaN -> None), rounded to keep the file small."""
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 4)
    return x


def _records(df):
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _cols(df, cols, nd=3):
    out = {}
    for c in cols:
        s = df[c]
        if s.dtype.kind in "fi":
            out[c] = [None if pd.isna(v) else round(float(v), nd) for v in s]
        else:
            out[c] = [str(v) for v in s]
    return out


def load_run(adir, run_id):
    ctrl = pd.read_csv(os.path.join(adir, run_id + "_control.csv"))
    tel = pd.read_csv(os.path.join(adir, run_id + "_telemetry.csv"))
    req = pd.read_csv(os.path.join(adir, run_id + "_requests.csv"))
    t0 = float(ctrl.t.iloc[0])

    # Phase boundaries (seconds after warm-up), from the controller log.
    phases = []
    prev = None
    for t, ph in zip(ctrl.t, ctrl.phase):
        if ph != prev:
            phases.append({"phase": ph, "start": round(float(t) - t0, 2)})
            prev = ph

    ctrl = ctrl.assign(t_rel=ctrl.t - t0,
                       reason=ctrl.decision_reason.astype(str).str.replace(r"\(.*", "", regex=True))
    tel = tel[tel.t >= t0 - 1].assign(t_rel=lambda d: d.t - t0)
    # 100 ms samples -> 0.5 s means, enough for the plot.
    tel = tel.assign(b=(tel.t_rel / 0.5).round()).groupby("b").agg(
        t_rel=("t_rel", "mean"), power_w=("power_w", "mean"),
        graphics_mhz=("graphics_mhz", "mean"), util_pct=("util_pct", "mean")).reset_index(drop=True)
    req = req[req.phase != "warmup"].assign(arr_rel=lambda d: d.arrival_t - t0,
                                            fin_rel=lambda d: d.finish_t - t0)

    burst = req[req.phase == "burst"]
    reasons = ctrl.reason.value_counts().to_dict()
    stats = {
        "burst_miss_rate": float(burst.slo_miss.mean()) if len(burst) else None,
        "burst_n": int(len(burst)),
        "burst_misses": int(burst.slo_miss.sum()) if len(burst) else 0,
        "top_clock_frac": float((ctrl.selected_mhz == TOP_MHZ).mean()),
        "end_mhz": float(ctrl.selected_mhz.tail(20).mean()),
        "reasons": {k: int(v) for k, v in reasons.items()},
    }
    trace = {
        "phases": phases,
        "control": _cols(ctrl, ["t_rel", "selected_mhz", "queue_len", "obs_queue_len",
                                "arrival_rate", "obs_arrival_rate", "p95_latency_s",
                                "obs_p95_latency_s", "telemetry_age_ms", "capacity_floor_mhz",
                                "reason"]),
        "gpu": _cols(tel, ["t_rel", "power_w", "graphics_mhz", "util_pct"]),
        "requests": _cols(req, ["arr_rel", "fin_rel", "latency_s", "phase"]),
        "miss": [bool(v) for v in req.slo_miss],
    }
    return stats, trace


def build(adir, seed=None):
    src = os.path.basename(os.path.normpath(adir))
    name = src if seed is None else f"{src}_s{seed}"
    runs = pd.read_csv(os.path.join(adir, "run_summary.csv"))
    if seed is not None:
        runs = runs[runs.seed == seed].reset_index(drop=True)
        if runs.empty:
            raise SystemExit(f"No runs with seed {seed} in {adir}")
    rows, traces = [], {}
    for _, r in runs.iterrows():
        stats, trace = load_run(adir, r.run_id)
        traces[r.run_id] = trace
        row = {k: _clean(r[k]) for k in (
            "run_id", "seed", "controller", "delay_ms", "workload", "total_j", "output_tokens",
            "j_per_token", "p50", "p95", "p99", "mean_lat", "miss_rate", "n_requests",
            "max_queue", "n_clock_changes", "mean_power_w", "mean_util_pct", "slo_target_s",
            "base_rate")}
        row["delay_ms"] = int(r.delay_ms)
        row.update({k: _clean(v) if not isinstance(v, dict) else v for k, v in stats.items()})
        rows.append(row)

    def _json(fn):
        p = os.path.join(adir, fn)
        return json.load(open(p)) if os.path.exists(p) else None

    sim = pd.read_csv(os.path.join(adir, "sim_summary.csv")) if os.path.exists(
        os.path.join(adir, "sim_summary.csv")) else pd.DataFrame()
    sweep = pd.read_csv(os.path.join(adir, "sweep_summary.csv")) if os.path.exists(
        os.path.join(adir, "sweep_summary.csv")) else pd.DataFrame()
    fdir = os.path.join(HERE, "figures", name)
    if seed is not None and os.path.isdir(fdir):
        rel, fsrc = f"figures/{name}", fdir
    else:
        rel, fsrc = f"../results/{src}", adir
    figs = sorted(f for f in os.listdir(fsrc) if f.startswith("fig") and f.endswith(".png"))
    cal = _json("slo_calibration.json") or {}
    return {
        "attempt": name,
        "meta": {
            "source_attempt": src,
            "seed_filter": seed,
            "probe": _json("probe.json"),
            "capacity": _json("capacity.json"),
            "slo_target_s": float(runs.slo_target_s.iloc[0]),
            "calibrated_slo_s": cal.get("slo_target_s"),
            "budget_ratio": cal.get("budget_ratio", 0.05),
            "n_runs": int(len(runs)),
            "seeds": sorted(int(s) for s in runs.seed.unique()),
            "figures": [f"{rel}/{f}" for f in figs],
        },
        "runs": rows,
        "traces": traces,
        "sweep": _records(sweep),
        "sim": _records(sim),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the dashboard data file")
    ap.add_argument("--attempt", default=os.path.join(ROOT, "results", "attempt_12"))
    ap.add_argument("--seed", type=int, default=None, help="Keep only this seed's runs")
    args = ap.parse_args(argv)
    data = build(args.attempt, args.seed)
    out_dir = os.path.join(HERE, "data")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, data["attempt"] + ".js")
    with open(out, "w") as f:
        f.write("window.DASHBOARD_DATA = window.DASHBOARD_DATA || {};\n")
        f.write(f"window.DASHBOARD_DATA[{json.dumps(data['attempt'])}] = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"{data['meta']['n_runs']} runs -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
