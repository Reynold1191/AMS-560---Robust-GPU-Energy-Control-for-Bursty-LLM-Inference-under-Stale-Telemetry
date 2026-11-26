"""
dashboard/build_replication.py
Pool several 3-seed repeat attempts into one small file for the dashboard's
Replication page.

  python dashboard/build_replication.py                                   # attempt_13 attempt_14
  python dashboard/build_replication.py --attempts attempt_13 attempt_14 attempt_15

Writes dashboard/data/replication.js (window.REPLICATION = {...}). Each
(attempt, seed) is one paired replicate: naive and robust saw the same
arrival schedule. Statistics are computed here (scipy), the page only draws.
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOP_MHZ = 2122
DELAYS = [0, 500, 1250, 2500]


def _f(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def load_attempt(adir):
    name = os.path.basename(os.path.normpath(adir))
    runs = pd.read_csv(os.path.join(adir, "run_summary.csv"))
    rows = []
    for _, r in runs.iterrows():
        q = pd.read_csv(os.path.join(adir, r.run_id + "_requests.csv"))
        q = q[q.phase != "warmup"]
        b = q[q.phase == "burst"]
        c = pd.read_csv(os.path.join(adir, r.run_id + "_control.csv"))
        rows.append({
            "att": name, "seed": int(r.seed), "controller": r.controller, "workload": r.workload,
            "delay_ms": int(r.delay_ms), "j_per_token": float(r.j_per_token),
            "miss": float(r.miss_rate) * 100, "p95": float(r.p95),
            "bmiss": float(b.slo_miss.mean()) * 100 if len(b) else np.nan,
            "bk": int(b.slo_miss.sum()), "bn": int(len(b)),
            "chg": int(r.n_clock_changes), "top": float((c.selected_mhz == TOP_MHZ).mean()) * 100,
        })
    probe = json.load(open(os.path.join(adir, "probe.json")))
    info = {"name": name, "n_runs": int(len(runs)), "seeds": sorted(int(s) for s in runs.seed.unique()),
            "base_rate": probe.get("base_rate"), "capacity_rps": _f(probe.get("capacity_rps"), 2)}
    return pd.DataFrame(rows), info


def _p(x):
    """p-values keep 3 significant digits (tiny values must not round to 0)."""
    return None if x is None or math.isnan(x) else float(f"{x:.3g}")


def fisher(kn, nn, kr, nr):
    return stats.fisher_exact([[kn, nn - kn], [kr, nr - kr]])[1]


def wilcoxon(x):
    x = np.asarray(x, float)
    return stats.wilcoxon(x).pvalue if len(x) > 1 and (x != 0).any() else None


def build(adirs):
    parts = [load_attempt(a) for a in adirs]
    t = pd.concat([p[0] for p in parts], ignore_index=True)
    key = ["att", "seed", "workload", "delay_ms"]
    n = t[t.controller == "naive_slo"].set_index(key)
    r = t[t.controller == "robust"].set_index(key)
    pr = pd.DataFrame({"jn": n.j_per_token, "jr": r.j_per_token, "bmn": n.bmiss, "bmr": r.bmiss,
                       "bkn": n.bk, "bkr": r.bk, "bnn": n.bn, "bnr": r.bn, "mn": n.miss, "mr": r.miss,
                       "topn": n.top, "topr": r.top}).reset_index()
    pr["dj"] = (pr.jr / pr.jn - 1) * 100

    paired = []
    for (w, d), g in pr.groupby(["workload", "delay_ms"]):
        burst = w == "sharp_burst"
        paired.append({
            "workload": w, "delay_ms": int(d), "k": int(len(g)),
            "dj_mean": _f(g.dj.mean(), 2), "dj_sd": _f(g.dj.std(), 2),
            "dj_lower": int((g.dj < 0).sum()), "p_dj": _p(stats.ttest_1samp(g.dj, 0).pvalue),
            "miss_n": _f(g.mn.mean(), 2), "miss_r": _f(g.mr.mean(), 2),
            "top_n": _f(g.topn.mean(), 1), "top_r": _f(g.topr.mean(), 1),
            "bk_n": int(g.bkn.sum()) if burst else None, "bk_r": int(g.bkr.sum()) if burst else None,
            "bn": int(g.bnn.sum()) if burst else None,
            "bm_lower": int((g.bmr < g.bmn).sum()) if burst else None,
            "bm_higher": int((g.bmr > g.bmn).sum()) if burst else None,
            "p_fisher": _p(fisher(g.bkn.sum(), g.bnn.sum(), g.bkr.sum(), g.bnr.sum())) if burst else None,
            "p_wil": _p(wilcoxon(g.bmr - g.bmn)) if burst else None,
        })

    sb = pr[pr.workload == "sharp_burst"]
    overall = {
        "pairs": int(len(sb)),
        "energy_lower": int((sb.dj < 0).sum()),
        "p_sign_energy": _p(stats.binomtest(int((sb.dj < 0).sum()), len(sb)).pvalue),
        "dj_mean": _f(sb.dj.mean(), 2), "dj_sd": _f(sb.dj.std(), 2),
        "bm_lower": int((sb.bmr < sb.bmn).sum()), "bm_tie": int((sb.bmr == sb.bmn).sum()),
        "bm_higher": int((sb.bmr > sb.bmn).sum()),
        "dbm_mean": _f((sb.bmr - sb.bmn).mean(), 2), "p_wil_bm": _p(wilcoxon(sb.bmr - sb.bmn)),
    }
    ramp = pr[(pr.workload == "ramp") & (pr.delay_ms == max(DELAYS))]
    overall["ramp_dmax_dj"] = _f(ramp.dj.mean(), 2)
    overall["ramp_dmax_higher"] = int((ramp.dj > 0).sum())

    # Ablation on sharp_burst, pooled.
    ab = []
    a = t[t.workload == "sharp_burst"]
    for (d, c), g in a.groupby(["delay_ms", "controller"]):
        nv = a[(a.delay_ms == d) & (a.controller == "naive_slo")]
        ab.append({"delay_ms": int(d), "controller": c, "bk": int(g.bk.sum()), "bn": int(g.bn.sum()),
                   "j_per_token": _f(g.j_per_token.mean()), "chg": _f(g.chg.mean(), 1),
                   "top": _f(g.top.mean(), 1), "miss": _f(g.miss.mean(), 2),
                   "p_vs_naive": _p(fisher(nv.bk.sum(), nv.bn.sum(), g.bk.sum(), g.bn.sum()))
                   if c != "naive_slo" else None})

    # Savings vs static_high, same attempt and seed.
    st = t[t.controller == "static_high"].set_index(["att", "seed", "workload"]).j_per_token
    sv = []
    for c in ("naive_slo", "robust"):
        x = t[t.controller == c]
        s = pd.Series([(j / st[(aa, ss, w)] - 1) * 100 for aa, ss, w, j in
                       zip(x.att, x.seed, x.workload, x.j_per_token)], index=x.index)
        for (w, d), g in s.groupby([x.workload, x.delay_ms]):
            sv.append({"controller": c, "workload": w, "delay_ms": int(d),
                       "mean": _f(g.mean(), 1), "min": _f(g.min(), 1), "max": _f(g.max(), 1)})

    runs = t.replace({np.nan: None}).to_dict("records")
    for row in runs:
        for k, v in row.items():
            if isinstance(v, float):
                row[k] = _f(v)
    return {"attempts": [p[1] for p in parts], "runs": runs, "paired": paired, "overall": overall,
            "ablation": ab, "savings": sv}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pool 3-seed repeat attempts for the dashboard")
    ap.add_argument("--attempts", nargs="+", default=["attempt_13", "attempt_14"])
    args = ap.parse_args(argv)
    adirs = [a if os.path.isdir(a) else os.path.join(ROOT, "results", a) for a in args.attempts]
    data = build(adirs)
    out = os.path.join(HERE, "data", "replication.js")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("window.REPLICATION = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    o = data["overall"]
    print(f"{len(data['runs'])} runs from {', '.join(a['name'] for a in data['attempts'])} -> {out} "
          f"({os.path.getsize(out) / 1e3:.0f} kB); sharp_burst energy lower in "
          f"{o['energy_lower']}/{o['pairs']} pairs")


if __name__ == "__main__":
    main()
