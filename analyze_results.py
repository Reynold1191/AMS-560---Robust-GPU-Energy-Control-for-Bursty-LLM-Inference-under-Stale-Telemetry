"""
analyze_results.py
Post-experiment analysis and plotting.

Produces the figures required for the final report:
  Fig 1.  Telemetry delay vs J/token  (per controller)
  Fig 2.  Telemetry delay vs SLO miss rate  (with budget line)
  Fig 3.  Time-series burst plot (arrival rate, queue, clock, age, P95)
  Fig 4.  Cumulative energy over time
  Fig 5.  Error-budget burn rate over time
  Fig 6.  Energy gap from fresh-telemetry oracle
  Fig 7.  Failure boundary heatmap (delay × burst multiplier)
  Fig 8.  Ablation comparison (full / no-age / no-budget)

Usage:
  python analyze_results.py --results-dir results --slo-target 2.0

All figures saved as PNG in the results directory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # headless backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("analyze_results")

CONTROLLER_COLORS = {
    "default":    "#888888",
    "static_high": "#e41a1c",
    "static_low":  "#377eb8",
    "naive_slo":   "#ff7f00",
    "robust":      "#4daf4a",
    "oracle":      "#984ea3",
    "robust_v1":      "#1f78b4",
    "robust_no_age":  "#a6cee3",
    "robust_no_budget": "#fb9a99",
    "Robust_NoAge":   "#a6cee3",
    "Robust_NoBudget": "#fb9a99",
}

DELAY_ORDER = [0, 100, 250, 500, 1000, 1250, 2500]
"""Calibrated levels (config.DELAY_VALUES_MS) plus the earlier uncalibrated ones,
so both new and old results are plotted."""


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_summary(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "run_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"run_summary.csv not found in {results_dir}")
    df = pd.read_csv(path)
    return df


def load_control_csv(run_id: str, results_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{run_id}_control.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_telemetry_csv(run_id: str, results_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{run_id}_telemetry.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _color(name: str) -> str:
    return CONTROLLER_COLORS.get(name, "#333333")


# ---------------------------------------------------------------------------
# Fig 1: Delay vs J/token
# ---------------------------------------------------------------------------

def fig_delay_vs_jtoken(df: pd.DataFrame, out_dir: str, slo: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ctrl, grp in df.groupby("controller"):
        x = grp.groupby("delay_ms")["j_per_token"].mean()
        err = grp.groupby("delay_ms")["j_per_token"].std()
        ax.errorbar(x.index, x.values, yerr=err.values,
                    marker="o", label=ctrl, color=_color(ctrl), capsize=3)
    ax.set_xlabel("Telemetry delay (ms)")
    ax.set_ylabel("Joules per output token")
    ax.set_title("Energy Efficiency vs Telemetry Delay")
    ax.legend()
    ax.set_xticks(DELAY_ORDER)
    _save(fig, os.path.join(out_dir, "fig1_delay_vs_jtoken.png"))


# ---------------------------------------------------------------------------
# Fig 2: Delay vs SLO miss rate
# ---------------------------------------------------------------------------

def fig_delay_vs_miss(df: pd.DataFrame, out_dir: str, slo: float,
                      budget: float = 0.01) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for ctrl, grp in df.groupby("controller"):
        x = grp.groupby("delay_ms")["miss_rate"].mean()
        ax.plot(x.index, x.values, marker="s", label=ctrl, color=_color(ctrl))
    ax.axhline(y=budget, color="red", linestyle="--", linewidth=1.5,
               label=f"Error budget ({budget*100:.0f}%)")
    ax.set_xlabel("Telemetry delay (ms)")
    ax.set_ylabel("SLO miss rate")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.set_title("SLO Miss Rate vs Telemetry Delay")
    ax.legend()
    ax.set_xticks(DELAY_ORDER)
    _save(fig, os.path.join(out_dir, "fig2_delay_vs_miss.png"))


# ---------------------------------------------------------------------------
# Fig 3: Time-series burst plot
# ---------------------------------------------------------------------------

def fig_burst_timeseries(ctrl_df: pd.DataFrame, out_dir: str,
                          run_id: str, burst_start: float,
                          burst_end: float, slo_target: float) -> None:
    """
    4-panel time-series: queue, clock, telemetry age, P95 latency.
    """
    t0 = ctrl_df["t"].min()
    ctrl_df = ctrl_df.copy()
    ctrl_df["elapsed"] = ctrl_df["t"] - t0

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Queue
    axes[0].fill_between(ctrl_df["elapsed"], ctrl_df["queue_len"],
                          alpha=0.5, color="#e41a1c")
    axes[0].set_ylabel("Queue length")

    # Selected clock
    axes[1].step(ctrl_df["elapsed"], ctrl_df["selected_mhz"],
                 where="post", color="#377eb8")
    axes[1].set_ylabel("GPU clock (MHz)")

    # Telemetry age
    if "telemetry_age_ms" in ctrl_df.columns:
        axes[2].plot(ctrl_df["elapsed"], ctrl_df["telemetry_age_ms"],
                     color="#ff7f00", linewidth=0.8)
        axes[2].set_ylabel("Telem age (ms)")

    # P95 latency
    axes[3].plot(ctrl_df["elapsed"], ctrl_df["p95_latency_s"],
                 color="#4daf4a")
    axes[3].axhline(y=slo_target, color="red", linestyle="--",
                    linewidth=1.2, label=f"SLO={slo_target}s")
    axes[3].set_ylabel("P95 latency (s)")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_xlabel("Elapsed time (s)")

    # Burst shading
    for ax in axes:
        ax.axvspan(burst_start, burst_end, alpha=0.12, color="yellow",
                   label="_burst")

    fig.suptitle(f"Burst time-series: {run_id}", fontsize=11)
    _save(fig, os.path.join(out_dir, f"fig3_timeseries_{run_id}.png"))


# ---------------------------------------------------------------------------
# Fig 4: Cumulative energy
# ---------------------------------------------------------------------------

def fig_cumulative_energy(telem_dfs: Dict[str, pd.DataFrame],
                           out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, tdf in telem_dfs.items():
        tdf = tdf.sort_values("t")
        dt = tdf["t"].diff().fillna(0)
        power = tdf["power_w"]
        cum_e = (0.5 * (power + power.shift(1)).fillna(power) * dt).cumsum()
        ax.plot(tdf["t"] - tdf["t"].iloc[0], cum_e, label=label,
                color=_color(label))
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Cumulative energy (J)")
    ax.set_title("Cumulative GPU Energy over Time")
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig4_cumulative_energy.png"))


# ---------------------------------------------------------------------------
# Fig 5: Error-budget burn rate
# ---------------------------------------------------------------------------

def fig_burn_rate(ctrl_df: pd.DataFrame, out_dir: str,
                   run_id: str) -> None:
    if "burn_rate" not in ctrl_df.columns:
        return
    t0 = ctrl_df["t"].min()
    ctrl_df = ctrl_df.copy()
    ctrl_df["elapsed"] = ctrl_df["t"] - t0

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ctrl_df["elapsed"], ctrl_df["burn_rate"], color="#984ea3")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.2,
               label="Budget = 1.0")
    ax.axhline(2.0, color="darkred", linestyle=":", linewidth=1.2,
               label="Emergency = 2.0")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Budget burn rate")
    ax.set_title(f"SLO Error-Budget Burn Rate: {run_id}")
    ax.legend()
    _save(fig, os.path.join(out_dir, f"fig5_burn_{run_id}.png"))


# ---------------------------------------------------------------------------
# Fig 6: Oracle gap
# ---------------------------------------------------------------------------

def fig_oracle_gap(df: pd.DataFrame, out_dir: str) -> None:
    # Use oracle controller if available, else fall back to static_efficient as reference
    oracle_rows = df[df["controller"] == "oracle"]
    ref_ctrl    = "oracle"
    if oracle_rows.empty:
        oracle_rows = df[df["controller"] == "static_efficient"]
        ref_ctrl    = "static_efficient"
    if oracle_rows.empty:
        logger.warning("No oracle or static_efficient rows; skipping Fig 6")
        return
    oracle_mean = oracle_rows.groupby("delay_ms")["j_per_token"].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    exclude = {"oracle", "static_efficient", "static_high", "static_low"}
    for ctrl, grp in df[~df["controller"].isin(exclude)].groupby("controller"):
        ctrl_mean = grp.groupby("delay_ms")["j_per_token"].mean()
        gap = ctrl_mean - oracle_mean
        gap = gap.reindex(DELAY_ORDER).dropna()
        ax.plot(gap.index, gap.values, marker="o", label=ctrl, color=_color(ctrl))
    ax.axhline(0, color="purple", linestyle="--", linewidth=1.0,
               label=f"{ref_ctrl} (reference)")
    ax.set_xlabel("Telemetry delay (ms)")
    ax.set_ylabel("J/token gap above reference")
    ax.set_title(f"Energy Gap from {ref_ctrl} Reference")
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig6_oracle_gap.png"))


# ---------------------------------------------------------------------------
# Fig 7: Failure boundary heatmap
# ---------------------------------------------------------------------------

def fig_failure_heatmap(df: pd.DataFrame, out_dir: str,
                         controller: str = "naive_slo",
                         budget: float = 0.01) -> None:
    sub = df[df["controller"] == controller].copy()
    if sub.empty:
        logger.warning("No %s rows; skipping heatmap", controller)
        return

    if "burst_multiplier" not in sub.columns or sub["burst_multiplier"].nunique() < 2:
        # Only one burst level - fall back to a bar chart of miss_rate vs delay
        logger.info("Single burst_multiplier; rendering miss_rate bar chart for Fig 7")
        pivot = sub.groupby("delay_ms")["miss_rate"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["green" if v <= budget else "red" for v in pivot.values]
        ax.bar([str(int(d)) for d in pivot.index], pivot.values, color=colors)
        ax.axhline(budget, color="black", linestyle="--", linewidth=1.2,
                   label=f"Budget threshold ({budget*100:.0f}%)")
        ax.set_xlabel("Telemetry delay (ms)")
        ax.set_ylabel("SLO miss rate")
        ax.set_title(f"SLO Miss Rate vs Delay: {controller}")
        ax.legend()
        _save(fig, os.path.join(out_dir, f"fig7_failure_map_{controller}.png"))
        return

    pivot = sub.pivot_table(
        index="burst_multiplier", columns="delay_ms",
        values="miss_rate", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(
        pivot.columns, pivot.index, pivot.values,
        cmap="RdYlGn_r", vmin=0, vmax=0.1
    )
    plt.colorbar(c, ax=ax, label="SLO miss rate")
    ax.axhline(budget * 100, color="white", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Telemetry delay (ms)")
    ax.set_ylabel("Burst multiplier")
    ax.set_title(f"Failure Boundary: {controller}")
    _save(fig, os.path.join(out_dir, f"fig7_failure_map_{controller}.png"))


# ---------------------------------------------------------------------------
# Fig 8: Ablation
# ---------------------------------------------------------------------------

def fig_ablation(df: pd.DataFrame, out_dir: str, slo: float) -> None:
    ablation_ctlrs = [c for c in df["controller"].unique()
                      if "Robust" in c or c.startswith("robust") or c == "naive_slo"]
    sub = df[df["controller"].isin(ablation_ctlrs)].copy()
    if sub.empty:
        logger.warning("No ablation rows; skipping Fig 8")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ctrl, grp in sub.groupby("controller"):
        x = grp.groupby("delay_ms")["j_per_token"].mean()
        axes[0].plot(x.index, x.values, marker="o", label=ctrl, color=_color(ctrl))
        x2 = grp.groupby("delay_ms")["miss_rate"].mean()
        axes[1].plot(x2.index, x2.values, marker="s", label=ctrl, color=_color(ctrl))

    axes[0].set_xlabel("Delay (ms)")
    axes[0].set_ylabel("J/token")
    axes[0].set_title("Ablation: Energy")
    axes[0].legend(fontsize=7)

    axes[1].set_xlabel("Delay (ms)")
    axes[1].set_ylabel("Miss rate")
    axes[1].axhline(0.01, color="red", linestyle="--", linewidth=1.0)
    axes[1].set_title("Ablation: SLO Reliability")
    axes[1].legend(fontsize=7)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    _save(fig, os.path.join(out_dir, "fig8_ablation.png"))


# ---------------------------------------------------------------------------
# Non-redundancy: paired P95 / different budget state
# ---------------------------------------------------------------------------

def fig_budget_nonredundancy(ctrl_dfs: Dict[str, pd.DataFrame],
                              out_dir: str, slo_target: float) -> None:
    """
    Show that error-budget state changes decisions independently of P95.
    Compare naive (no budget) vs robust (with budget) for same P95 level.
    """
    if "naive_slo" not in ctrl_dfs or "robust" not in ctrl_dfs:
        logger.info("Need both naive_slo and robust control logs for this plot")
        return
    naive = ctrl_dfs["naive_slo"]
    robust = ctrl_dfs["robust"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: P95 latency vs selected clock, coloured by burn rate
    if "burn_rate" in robust.columns:
        sc = axes[0].scatter(
            robust["p95_latency_s"],
            robust["selected_mhz"],
            c=robust["burn_rate"].clip(0, 3),
            cmap="RdYlGn_r", alpha=0.5, s=15,
        )
        plt.colorbar(sc, ax=axes[0], label="Burn rate")
        axes[0].set_xlabel("P95 latency (s)")
        axes[0].set_ylabel("Selected clock (MHz)")
        axes[0].set_title("Robust: clock decisions coloured by burn rate")

    # Compare: proportion of down-clock decisions at similar P95 levels
    p95_bins = np.arange(0, slo_target + 0.2, 0.2)
    naive_down  = []
    robust_down = []
    for lo, hi in zip(p95_bins, p95_bins[1:]):
        n = naive[(naive["p95_latency_s"] >= lo) & (naive["p95_latency_s"] < hi)]
        r = robust[(robust["p95_latency_s"] >= lo) & (robust["p95_latency_s"] < hi)]
        if len(n) > 0:
            prev_n = n["selected_mhz"].shift(1)
            naive_down.append((n["selected_mhz"] < prev_n).mean())
        else:
            naive_down.append(float("nan"))
        if len(r) > 0:
            prev_r = r["selected_mhz"].shift(1)
            robust_down.append((r["selected_mhz"] < prev_r).mean())
        else:
            robust_down.append(float("nan"))

    x = (p95_bins[:-1] + p95_bins[1:]) / 2
    axes[1].plot(x, naive_down,  marker="o", label="naive_slo",  color=_color("naive_slo"))
    axes[1].plot(x, robust_down, marker="s", label="robust",     color=_color("robust"))
    axes[1].set_xlabel("P95 latency bin (s)")
    axes[1].set_ylabel("Fraction of down-clock decisions")
    axes[1].set_title("Budget non-redundancy: down-clocking behaviour vs P95")
    axes[1].legend()

    _save(fig, os.path.join(out_dir, "fig_budget_nonredundancy.png"))


# ---------------------------------------------------------------------------
# Statistical summary table
# ---------------------------------------------------------------------------

def print_summary_table(df: pd.DataFrame, slo_target: float) -> None:
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY TABLE")
    print("=" * 80)
    cols = ["controller", "delay_ms", "j_per_token", "p95", "miss_rate",
            "recovery_s", "max_queue"]
    available = [c for c in cols if c in df.columns]
    grp = df[available].groupby(["controller", "delay_ms"]).agg(
        **{c: (c, "mean") for c in available if c not in ["controller", "delay_ms"]}
    ).reset_index()
    print(grp.to_string(index=False))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="AMS 560 results analysis")
    p.add_argument("--results-dir", default="results")
    from config import ExperimentConfig
    _d = ExperimentConfig()
    p.add_argument("--slo-target",  type=float, default=None,
                   help="Default: slo_calibration.json in --results-dir, else config.py")
    p.add_argument("--burst-start", type=float, default=_d.burst_start_s,
                   help="Seconds after warm-up (same origin as the runner)")
    p.add_argument("--burst-end",   type=float,
                   default=_d.burst_start_s + _d.burst_duration_s)
    p.add_argument("--workload",    default="sharp_burst",
                   help="Workload for the comparison figures ('all' mixes workloads)")
    p.add_argument("--run-id",      default=None,
                   help="Specific run_id for time-series / burn plots")
    args = p.parse_args()

    if args.slo_target is None:
        cal = os.path.join(args.results_dir, "slo_calibration.json")
        if os.path.exists(cal):
            import json
            with open(cal, encoding="utf-8") as f:
                args.slo_target = float(json.load(f)["slo_target_s"])
        else:
            args.slo_target = _d.slo_target_s
    print(f"SLO target: {args.slo_target:.2f} s   workload: {args.workload}")

    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Load summary ──────────────────────────────────────────────────────
    try:
        df = load_summary(args.results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if df.empty:
        print("run_summary.csv is empty – run some experiments first.")
        sys.exit(1)

    if args.workload != "all" and "workload" in df.columns:
        df = df[df["workload"] == args.workload]
        if df.empty:
            print(f"No runs with workload={args.workload}.")
            sys.exit(1)

    print_summary_table(df, args.slo_target)

    # ── Main comparison figures ───────────────────────────────────────────
    fig_delay_vs_jtoken(df, out_dir, args.slo_target)
    fig_delay_vs_miss(df, out_dir, args.slo_target)
    fig_oracle_gap(df, out_dir)
    fig_failure_heatmap(df, out_dir, controller="naive_slo")
    fig_ablation(df, out_dir, args.slo_target)

    # ── Time-series and burn-rate for one run ─────────────────────────────
    run_id = args.run_id
    if run_id is None:
        # Pick the most recent robust run
        robust_rows = df[df["controller"] == "robust"]
        if not robust_rows.empty:
            run_id = robust_rows.iloc[-1]["run_id"]

    if run_id:
        cdf = load_control_csv(run_id, args.results_dir)
        tdf = load_telemetry_csv(run_id, args.results_dir)
        if cdf is not None and not cdf.empty:
            fig_burst_timeseries(cdf, out_dir, run_id,
                                  args.burst_start, args.burst_end,
                                  args.slo_target)
            fig_burn_rate(cdf, out_dir, run_id)

        if tdf is not None:
            fig_cumulative_energy({"robust": tdf}, out_dir)

    # ── Budget non-redundancy ─────────────────────────────────────────────
    ctrl_dfs = {}
    for ctrl in ["naive_slo", "robust"]:
        rows = df[df["controller"] == ctrl]
        if not rows.empty:
            rid = rows.iloc[-1]["run_id"]
            cdf = load_control_csv(rid, args.results_dir)
            if cdf is not None:
                ctrl_dfs[ctrl] = cdf
    if ctrl_dfs:
        fig_budget_nonredundancy(ctrl_dfs, out_dir, args.slo_target)

    print(f"\nAll figures saved to: {out_dir}/")


if __name__ == "__main__":
    main()
