"""
docs/make_readme_figures.py
Figures and the burst animation used in README.md (results section).

  python docs/make_readme_figures.py
  python docs/make_readme_figures.py --attempts attempt_13 attempt_14 attempt_15

Pooled statistics come from every (attempt, seed) replicate. The animation
replays one representative replicate (attempt_13, seed 41, 2500 ms delay).
Writes docs/img/*.png and docs/img/burst_replay.gif. No GPU needed.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "img")
NAIVE, ROBUST = "#e07b24", "#0e9f80"
COL = {"naive_slo": NAIVE, "robust": ROBUST, "robust_v1": "#7c6fd6", "robust_no_age": "#d1477a"}
DELAYS = [0, 500, 1250, 2500]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.25,
})


def load(attempts):
    rows = []
    for att in attempts:
        adir = os.path.join(ROOT, "results", att)
        runs = pd.read_csv(os.path.join(adir, "run_summary.csv"))
        for _, r in runs.iterrows():
            q = pd.read_csv(os.path.join(adir, r.run_id + "_requests.csv"))
            b = q[q.phase == "burst"]
            rows.append(dict(att=att, seed=int(r.seed), ctrl=r.controller, w=r.workload, d=int(r.delay_ms),
                             j=float(r.j_per_token), bk=int(b.slo_miss.sum()), bn=len(b),
                             bm=float(b.slo_miss.mean()) * 100 if len(b) else np.nan))
    return pd.DataFrame(rows)


def fig_burst_miss(t, path):
    sb = t[t.w == "sharp_burst"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for c, dx in (("naive_slo", -45), ("robust", 45)):
        g = sb[sb.ctrl == c]
        pooled = g.groupby("d").apply(lambda x: x.bk.sum() / x.bn.sum() * 100)
        ax.scatter(g.d + dx, g.bm, s=22, color=COL[c], alpha=0.35, zorder=2)
        ax.plot(pooled.index, pooled.values, "-o", color=COL[c], lw=3, ms=8, zorder=3,
                label=("naive (trusts the stale view)" if c == "naive_slo" else "robust (age-aware, ours)"))
        for d, v in pooled.items():
            ax.annotate(f"{v:.1f}%", (d, v), textcoords="offset points", xytext=(0, 9 if c == "naive_slo" else -16),
                        ha="center", fontsize=10, fontweight="bold", color=COL[c])
    ax.axhline(5, color="#c0392b", ls="--", lw=1)
    ax.text(30, 6.2, "5 % error budget", color="#c0392b", fontsize=9)
    ax.set_xticks(DELAYS)
    ax.set_xlabel("telemetry delay (ms)")
    ax.set_ylabel("requests late during the burst (%)")
    ax.set_title("Late requests during a 4x traffic burst")
    ax.legend(loc="upper left", frameon=False)
    ax.set_ylim(-10, 100)
    fig.text(0.01, -0.04, f"Line: pooled over {t.groupby(['att', 'seed']).ngroups} paired replicates ({', '.join(sorted(t.att.unique()))}; seeds 41-43). Dots: individual replicates.",
             fontsize=8.5, color="#555")
    fig.savefig(path)
    plt.close(fig)


def fig_energy(t, path):
    sb = t[t.w == "sharp_burst"]
    n = sb[sb.ctrl == "naive_slo"].set_index(["att", "seed", "d"]).j
    r = sb[sb.ctrl == "robust"].set_index(["att", "seed", "d"]).j
    dj = ((r / n - 1) * 100).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rng = np.random.default_rng(3)
    for i, d in enumerate(DELAYS):
        v = dj[dj.d == d].j.values
        ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, s=55, color=ROBUST,
                   edgecolor="white", lw=0.8, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color="#222", lw=2.5, zorder=4)
        ax.text(i + 0.28, v.mean(), f"{v.mean():+.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#333", lw=1)
    ax.axhspan(0, 3, color="#e07b24", alpha=0.07)
    ax.text(-0.45, 1.2, "robust costs more", color=NAIVE, fontsize=9)
    ax.text(-0.45, -11.3, "robust saves energy", color=ROBUST, fontsize=9)
    ax.set_xticks(range(4), [f"{d} ms" for d in DELAYS])
    ax.set_ylim(-12, 3)
    ax.set_ylabel("energy per token vs naive (%)")
    lower = int((dj.j < 0).sum())
    ax.set_title(f"Robust used less energy in {lower} of {len(dj)} paired burst runs")
    fig.savefig(path)
    plt.close(fig)


def fig_ablation(t, path, d=2500):
    sb = t[(t.w == "sharp_burst") & (t.d == d)]
    order = ["naive_slo", "robust_v1", "robust_no_age", "robust"]
    lab = ["naive", "robust v1\n(first design)", "robust v2\nwithout age", "robust v2\n(full)"]
    v = [sb[sb.ctrl == c].bk.sum() / sb[sb.ctrl == c].bn.sum() * 100 for c in order]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(lab, v, color=[COL[c] for c in order], width=0.6)
    for b_, x in zip(bars, v):
        ax.text(b_.get_x() + b_.get_width() / 2, x + 1.2, f"{x:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("late during burst (%)")
    ax.set_title(f"Each mechanism matters (burst, {d / 1000:.1f} s-old telemetry)")
    ax.set_ylim(0, max(v) * 1.18)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def fig_savings(t, path):
    rows = []
    for c in ("naive_slo", "robust"):
        for w in ("sharp_burst", "ramp", "steady"):
            x = t[(t.ctrl == c) & (t.w == w)]
            s = [(-(row.j / t[(t.att == row.att) & (t.seed == row.seed) & (t.w == w) & (t.ctrl == "static_high")].j.iloc[0] - 1)) * 100
                 for row in x.itertuples()]
            rows.append((c, w, np.mean(s)))
    df = pd.DataFrame(rows, columns=["c", "w", "s"])
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ws = ["sharp_burst", "ramp", "steady"]
    for k, c in enumerate(("naive_slo", "robust")):
        v = [df[(df.c == c) & (df.w == w)].s.iloc[0] for w in ws]
        bars = ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, 0.34, color=COL[c], label="naive" if k == 0 else "robust")
        for b_, x in zip(bars, v):
            ax.text(b_.get_x() + b_.get_width() / 2, x + 0.6, f"{x:.0f}%", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(3), ["bursty", "ramp", "steady"])
    ax.set_ylabel("energy saved vs max clock (%)")
    ax.set_title("Energy saved vs running the GPU at max clock all the time")
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 33)
    ax.grid(axis="x", visible=False)
    fig.savefig(path)
    plt.close(fig)


def burst_gif(path, att="attempt_13", seed=41, d=2500):
    adir = os.path.join(ROOT, "results", att)
    S = {}
    for c in ("naive_slo", "robust"):
        rid = f"{c}_d{d}_sharp_burst_s{seed}"
        ctrl = pd.read_csv(os.path.join(adir, rid + "_control.csv"))
        req = pd.read_csv(os.path.join(adir, rid + "_requests.csv"))
        t0 = ctrl.t.iloc[0]
        req = req[req.phase != "warmup"].sort_values("finish_t")
        S[c] = dict(t=(ctrl.t - t0).values, q=ctrl.queue_len.values, oq=ctrl.obs_queue_len.values,
                    mhz=ctrl.selected_mhz.values, phase=ctrl.phase.values,
                    ft=(req.finish_t - t0).values, miss=req.slo_miss.astype(int).cumsum().values)
    tb = S["robust"]["t"]
    ph = S["robust"]["phase"]
    b0 = tb[np.argmax(ph == "burst")]
    b1 = tb[len(ph) - 1 - np.argmax(ph[::-1] == "burst")]
    T = min(S["naive_slo"]["t"][-1], tb[-1])

    fig, axs = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=True, gridspec_kw=dict(hspace=0.25))
    fig.suptitle(f"Replay: one 4x burst with {d / 1000:.1f} s-old telemetry", fontweight="bold", fontsize=13, y=0.985)
    for ax in axs:
        ax.axvspan(b0, b1, color="#f4c542", alpha=0.18, lw=0)
        ax.set_xlim(0, T)
    axs[0].text((b0 + b1) / 2, 1.0, "BURST", transform=axs[0].get_xaxis_transform(), ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#a07c00")
    axs[0].set_ylabel("requests\nwaiting")
    axs[1].set_ylabel("GPU clock\n(MHz)")
    axs[2].set_ylabel("late requests\n(cumulative)")
    axs[2].set_xlabel("time (s)")
    axs[0].set_ylim(0, max(S["naive_slo"]["q"].max(), S["robust"]["q"].max()) * 1.15)
    axs[1].set_ylim(0, 2400)
    axs[1].set_yticks([667, 1147, 1635, 2122])
    axs[2].set_ylim(-1, S["naive_slo"]["miss"][-1] * 1.2 + 2)

    L = {}
    for c, lab in (("naive_slo", "naive"), ("robust", "robust (ours)")):
        L[c] = [axs[0].plot([], [], color=COL[c], lw=2, label=lab)[0],
                axs[1].step([], [], where="post", color=COL[c], lw=2.2)[0],
                axs[2].plot([], [], color=COL[c], lw=2.5)[0],
                axs[2].text(0, 0, "", color=COL[c], fontweight="bold", fontsize=11)]
    axs[0].legend(loc="upper left", frameon=False, ncol=2)
    clock = axs[0].text(0.99, 0.82, "", transform=axs[0].transAxes, ha="right", fontsize=10, color="#444")

    frames = np.arange(0, T + 0.01, 0.75)

    def upd(now):
        for c in S:
            s = S[c]
            k = s["t"] <= now
            L[c][0].set_data(s["t"][k], s["q"][k])
            L[c][1].set_data(s["t"][k], s["mhz"][k])
            m = s["ft"] <= now
            x = np.concatenate([[0], s["ft"][m], [now]])
            y = np.concatenate([[0], s["miss"][m], [s["miss"][m][-1] if m.any() else 0]])
            L[c][2].set_data(x, y)
            L[c][3].set_position((now + 1, y[-1] + 0.5))
            L[c][3].set_text(str(int(y[-1])))
        clock.set_text(f"t = {now:5.1f} s")
        return []

    frames = np.concatenate([frames, np.full(12, frames[-1])])  # hold the last frame
    anim = FuncAnimation(fig, upd, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="README figures")
    ap.add_argument("--attempts", nargs="+", default=["attempt_13", "attempt_14", "attempt_15"])
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    t = load(args.attempts)
    fig_burst_miss(t, os.path.join(OUT, "burst_misses.png"))
    fig_energy(t, os.path.join(OUT, "energy_pairs.png"))
    fig_ablation(t, os.path.join(OUT, "ablation.png"))
    fig_savings(t, os.path.join(OUT, "savings_vs_max_clock.png"))
    burst_gif(os.path.join(OUT, "burst_replay.gif"))
    for f in sorted(os.listdir(OUT)):
        print(f"  docs/img/{f}  {os.path.getsize(os.path.join(OUT, f)) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
