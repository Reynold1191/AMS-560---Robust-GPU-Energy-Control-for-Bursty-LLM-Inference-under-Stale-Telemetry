"""
dashboard/build_explainers.py
Packs the data behind the two explainer animations (kitchen and machine) into
dashboard/data/explainers.js. Both animations replay real GPU runs: the
showcase replicate (attempt_13, seed 41) on sharp_burst at 2500 ms delay, for
naive_slo and robust. The prompt text comes from prompts/long_context.jsonl.

  python dashboard/build_explainers.py
  python dashboard/build_explainers.py --attempt attempt_13 --seed 41 --delay 2500
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
def say(reason: str, delta: int) -> str:
    """decision_reason (plus the actual clock step) -> words the kitchen manager would say."""
    toks = [t.split("(")[0] for t in reason.split()]
    main = toks[0] if toks else ""
    if main.startswith("EMERGENCY"):
        return "Recent dishes were slow! Full flame." if delta >= 0 else "Turn it down."
    if main == "FORECAST_UP":
        return "The rail is filling fast. A rush is coming: turn up early."
    if main == "CAPACITY_DOWN":
        return "The rail has been empty longer than the board lag. The slow average is old news: turn down."
    if delta > 0:
        return "Orders are arriving faster: raise the flame to keep up."
    if main == "SAFE_DOWNCLOCK":
        return "The board looks quiet. Turn the stove down." if delta < 0 else "Quiet, but keep enough flame for the orders coming in."
    if "WAIT_EFFECT" in toks:
        return "Wait: my last change has not reached the board yet."
    return "Hold the stove where it is."


def prompts():
    rows = [json.loads(x) for x in open(os.path.join(ROOT, "prompts", "long_context.jsonl"), encoding="utf-8")]
    out = []
    for r in rows:
        lines = r["prompt"].strip().split("\n")
        q = next((l for l in lines if l.startswith("Question:")), "")
        out.append(dict(kind=r["kind"], tokens=int(r["prompt_tokens"]), max_new=int(r["max_new_tokens"]),
                        head=lines[0], body=lines[1:4], n_lines=len(lines), question=q))
    return out


def run(adir, ctrl, delay, seed, P):
    rid = f"{ctrl}_d{delay}_sharp_burst_s{seed}"
    c = pd.read_csv(os.path.join(adir, rid + "_control.csv"))
    q = pd.read_csv(os.path.join(adir, rid + "_requests.csv"))
    tel = pd.read_csv(os.path.join(adir, rid + "_telemetry.csv"))
    t0 = float(c.t.iloc[0])
    toks = np.array([p["tokens"] for p in P])
    reqs = []
    for _, r in q.sort_values("arrival_t").iterrows():
        pi = int(np.argmin(np.abs(toks - (r.prompt_tokens - 6))))  # chat template adds ~6 tokens
        reqs.append([round(r.arrival_t - t0, 3), round(r.start_t - t0, 3), round(r.finish_t - t0, 3),
                     int(bool(r.slo_miss)), int(r.prompt_tokens), int(r.output_tokens), pi,
                     {"warmup": 0, "steady": 1, "burst": 2, "recovery": 3}.get(r.phase, 1)])
    ctl, prev = [], int(c.selected_mhz.iloc[0])
    for _, r in c.iterrows():
        ctl.append(dict(t=round(r.elapsed_s, 2), mhz=int(r.selected_mhz), q=int(r.queue_len), oq=int(r.obs_queue_len),
                        p95=round(float(r.p95_latency_s), 2), op95=round(float(r.obs_p95_latency_s), 2),
                        rate=round(float(r.arrival_rate), 2), orate=round(float(r.obs_arrival_rate), 2),
                        reason=str(r.decision_reason), say=say(str(r.decision_reason), int(r.selected_mhz) - prev),
                        phase=str(r.phase)))
        prev = int(r.selected_mhz)
    tel = tel[tel.t >= t0 - 0.5].copy()
    tel["e"] = (tel.power_w * tel.t.diff().fillna(0).clip(0, 1)).cumsum()
    grid = np.arange(0, ctl[-1]["t"] + 1.01, 0.5)
    pw = np.interp(grid, tel.t - t0, tel.power_w)
    en = np.interp(grid, tel.t - t0, tel.e - tel.e.iloc[0])
    summ = pd.read_csv(os.path.join(adir, "run_summary.csv"))
    s = summ[summ.run_id == rid].iloc[0]
    return dict(ctrl=ctrl, reqs=reqs, ctl=ctl, power=[round(x, 1) for x in pw], energy=[round(x, 1) for x in en],
                j_per_token=round(float(s.j_per_token), 4), burst_miss=round(float(s.burst_miss_rate), 4)
                if "burst_miss_rate" in s else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt", default="attempt_13")
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--delay", type=int, default=2500)
    a = ap.parse_args()
    adir = os.path.join(ROOT, "results", a.attempt)
    P = prompts()
    D = dict(source=f"{a.attempt}, seed {a.seed}, sharp_burst, {a.delay} ms delay", delay_ms=a.delay, slo_s=5.0,
             batch=16, clocks=[180, 667, 1147, 1635, 2122], phases=[[0, 40, "steady"], [40, 60, "burst"], [60, 110, "recovery"]],
             prompts=P, runs={c: run(adir, c, a.delay, a.seed, P) for c in ("naive_slo", "robust")})
    out = os.path.join(HERE, "data", "explainers.js")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("window.EXPLAINERS = " + json.dumps(D, separators=(",", ":")) + ";\n")
    for c, r in D["runs"].items():
        late = sum(1 for x in r["reqs"] if x[7] == 2 and x[3])
        burst = sum(1 for x in r["reqs"] if x[7] == 2)
        print(f"{c}: {len(r['reqs'])} requests, burst late {late}/{burst}, energy {r['energy'][-1]:.0f} J")
    print(f"-> {out} ({os.path.getsize(out) // 1024} kB)")


if __name__ == "__main__":
    main()
