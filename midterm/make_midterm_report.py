"""Generate evidence figures and the 2-page Group 10 midterm PDF.

Uses only the code on the shared repository main branch.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..",
                 "AMS-560---Robust-GPU-Energy-Control-for-Bursty-LLM-Inference-under-Stale-Telemetry")
)
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from config import ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState
from delay_buffer import DelayBuffer
from load_generator import workload_rate
from metrics import joules_from_samples, summary_stats
from scale_simulator import simulate

CLOCKS = [180, 667, 1147, 1635, 2122, 2602, 3090]
CTRLS = ["static_efficient", "naive_slo", "robust_no_age", "robust_no_budget", "robust"]
COLORS = {
    "static_efficient": "#2E79B5",
    "naive_slo": "#C85898",
    "robust_no_age": "#F0A040",
    "robust_no_budget": "#7B4B94",
    "robust": "#1F8A65",
}


class FakeSample:
    def __init__(self, t, power_w):
        self.t = t
        self.power_w = power_w


def save_workload(path):
    xs = [i * 0.5 for i in range(241)]
    kw = dict(base_rate=0.15, burst_multiplier=5.0, burst_start_s=40.0,
              burst_duration_s=20.0, ramp_end_rate=2.0, duration_s=120.0)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    for name, color in [("steady", "#4C78A8"), ("ramp", "#F58518"), ("sharp_burst", "#E45756")]:
        ys = [workload_rate(name, x, **kw) for x in xs]
        ax.plot(xs, ys, label=name, color=color, lw=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arrival rate (req/s)")
    ax.set_title("Workload profiles on main (proposal D3, README D4)")
    ax.legend(frameon=False, ncol=3)
    ax.set_xlim(0, 120)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_delay(path):
    import delay_buffer as db

    clock = {"t": 0.0}
    db.time.monotonic = lambda: clock["t"]
    buf = DelayBuffer()
    buf.push(FakeSample(0.0, 30.0))
    clock["t"] = 0.40
    buf.push(FakeSample(0.40, 80.0))
    delays = [0, 100, 250, 500, 1000]
    powers, ages = [], []
    for ms in delays:
        sample, age = buf.get(delay_s=ms / 1000.0)
        powers.append(sample.power_w)
        ages.append(age * 1000.0)
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    ax.plot(delays, powers, "o-", color="#1F8A65", lw=2)
    ax.set_xlabel("Requested telemetry delay (ms)")
    ax.set_ylabel("Returned power (W)")
    ax.set_title("Delay buffer on main (proposal D3, README D5)")
    ax.set_xticks(delays)
    for x, p, a in zip(delays, powers, ages):
        ax.annotate(f"{a:.0f} ms old", (x, p), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return list(zip(delays, powers, ages))


def save_controllers(path):
    cfg = ExperimentConfig()
    scenarios = [
        ("Fresh calm", dict(p95_latency_s=2.0, telemetry_age_ms=0, queue_len=0, error_budget_burn=0)),
        ("Stale 600 ms", dict(p95_latency_s=2.0, telemetry_age_ms=600, queue_len=0, error_budget_burn=0)),
        ("SLO miss", dict(p95_latency_s=10.0, telemetry_age_ms=0, queue_len=0, error_budget_burn=0)),
        ("Budget burn", dict(p95_latency_s=2.0, telemetry_age_ms=0, queue_len=0, error_budget_burn=3)),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=150)
    import numpy as np
    x = np.arange(len(scenarios))
    width = 0.15
    for i, name in enumerate(CTRLS):
        ctrl = get_controller(name, CLOCKS, cfg)
        mhz = []
        for _, kw in scenarios:
            state = ControllerState(slo_target_s=8.5, arrival_cv=0.0, **kw)
            idx = ctrl.decide(state, 3, min_dwell=0)
            chosen = ctrl.clock_mhz(idx)
            mhz.append(chosen)
            rows.append({"scenario": scenarios[i and 0 or 0][0] if False else "", "controller": name,
                         "clock_mhz": chosen, "reason": ctrl.last_reason, "scenario_name": _})
            rows[-1]["scenario"] = _
        ax.bar(x + (i - 2) * width, mhz, width, label=name, color=COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in scenarios])
    ax.set_ylabel("Chosen clock (MHz)")
    ax.set_title("Controller decision from 1635 MHz (proposal D4–D6, README D6, D7, D9)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.set_ylim(0, 3600)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return rows


def save_scale(path):
    sweep = pd.DataFrame({
        "clock_mhz": [667, 1147, 2602],
        "mean_lat": [8.0, 5.0, 3.0],
        "workload": ["steady", "steady", "steady"],
    })
    delays = [0, 250, 500, 1000]
    names = ["naive_slo", "robust_no_age", "robust"]
    cfg = ExperimentConfig()
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    for name in names:
        misses = []
        for delay in delays:
            ctrl = get_controller(name, [667, 1147, 2602], cfg)
            result = simulate(
                n_workers=4, n_requests=120, controller=ctrl, delay_ms=delay,
                workload="sharp_burst", sweep_df=sweep, slo_target_s=8.5,
                base_rate=1.0, burst_mult=5.0, seed=1,
            )
            misses.append(result["miss_rate"])
            rows.append({"controller": name, "delay_ms": delay,
                         "miss_rate": result["miss_rate"],
                         "p95_latency_s": result["p95_latency_s"],
                         "j_per_token": result["j_per_token"],
                         "n_clock_changes": result["n_clock_changes"]})
        ax.plot(delays, misses, "o-", label=name, color=COLORS[name], lw=2)
    ax.set_xlabel("Telemetry delay (ms)")
    ax.set_ylabel("SLO miss rate")
    ax.set_title("CPU scale-out, 4 workers, 120 requests (README D10)")
    ax.set_xticks(delays)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return rows


def energy_check():
    samples = [FakeSample(0.0, 100.0), FakeSample(1.0, 100.0), FakeSample(2.0, 100.0)]
    joules = joules_from_samples(samples)
    stats = summary_stats([float(i) for i in range(1, 11)])
    return joules, stats


class Report(FPDF):
    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", size=8)
        self.set_text_color(90)
        self.cell(0, 6, f"Group 10  |  AMS 560 midterm  |  page {self.page_no()} of 2", align="C")


def build_pdf(fig_work, fig_delay, fig_ctrl, fig_scale, out_pdf):
    pdf = Report(format="Letter")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 7, "AMS 560 Midterm Report  -  Group 10", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 5, "Robust GPU Energy Control for Bursty LLM Inference under Stale Telemetry",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.cell(0, 4, "Due October 27, 2026. Completion counts only the shared GitHub main branch as of October 26.",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Objective", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 4,
        "The project asks whether a GPU clock controller that treats stale telemetry as a risk signal "
        "can lower LLM inference energy without breaking a latency SLO. The planned system serves "
        "Qwen2.5-0.5B-Instruct on one RTX 5060. A controller chooses a graphics clock once per second. "
        "Higher clocks finish tokens sooner and use more energy. The plan uses an 8.5 s SLO and a 1% "
        "error budget. Workloads are steady, ramp, and sharp burst. Baselines are a fixed efficient clock "
        "and a latency-only controller. The proposed controller also uses telemetry age and error-budget burn.",
        new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Timeline", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 4,
        "Sep 20: repository created. Oct 23: data layer and controllers. Oct 25: metrics logger. "
        "Oct 26: Members 1-4 merged to main, and the load generator was aligned with the experiment runner. "
        "After the midterm: GPU testbed, timing measurements, frequency characterization, and the "
        "failure-boundary evaluation.",
        new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Proposal deliverables (Section 6)  -  overall 53%", new_x="LMARGIN", new_y="NEXT")
    proposal = [
        ["D1", "Inference and telemetry testbed", "25%", "Config, queue, prompts, metrics logger. No GPU worker or sampler."],
        ["D2", "Frequency-energy-latency characterization", "0%", "No sweep and no comparison results."],
        ["D3", "Stale-telemetry framework", "67%", "Delay injector and workload generator are tested. No timing measurement."],
        ["D4", "Baseline implementations", "100%", "Default, static, naive, and fresh-telemetry controllers. Fig. 3."],
        ["D5", "Telemetry-age-aware controller", "100%", "Age-aware policy and 204 passing tests. Fig. 3."],
        ["D6", "Failure boundary and ablation", "25%", "Ablation controllers are unit-tested. No GPU map yet. Fig. 3-4."],
    ]
    _table(pdf, [12, 68, 16, 94], proposal)

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "README deliverables  -  overall 55%", new_x="LMARGIN", new_y="NEXT")
    readme = [
        ["D1", "Experiment harness and clock reset", "0%", "run_experiment.py is not on main."],
        ["D2", "Fixed-frequency characterisation", "0%", "No sweep script or sweep_summary.csv."],
        ["D3", "Real timing characterisation", "0%", "measure_timing.py is not on main."],
        ["D4", "Workload suite", "100%", "steady / ramp / sharp burst. Fig. 1."],
        ["D5", "Delay injector", "100%", "0-1000 ms delay buffer. Fig. 2."],
        ["D6", "Baselines and oracle", "100%", "Factory builds every controller name. Fig. 3."],
        ["D7", "Robust controller and tests", "100%", "Acceptance tests pass. Fig. 3."],
        ["D8", "Failure-boundary analysis", "0%", "analyze_results.py Figure 7 is not on main."],
        ["D9", "Ablation and non-redundancy", "50%", "no-age and no-budget policies are tested. Fig. 3."],
        ["D10", "Scale-out simulation", "100%", "CPU simulation is tested. No result file was committed. Fig. 4."],
    ]
    _table(pdf, [12, 62, 16, 100], readme)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Evidence for deliverables above 0%", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=8)
    pdf.multi_cell(0, 3.5,
        "These outputs were produced on October 26 from the code on main. They are not GPU experiment results. "
        "A 100 W sample held for 2 s integrates to 200 J in the metrics logger (proposal D1).",
        new_x="LMARGIN", new_y="NEXT")
    y = pdf.get_y() + 1
    pdf.image(fig_work, x=10, y=y, w=96)
    pdf.image(fig_delay, x=110, y=y, w=96)
    pdf.image(fig_ctrl, x=10, y=y + 48, w=120)
    pdf.image(fig_scale, x=132, y=y + 48, w=74)

    pdf.set_y(y + 100)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Plan for unfinished deliverables", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 4,
        "1. Proposal D1 / README D1: add the LLM worker, GPU telemetry sampler, and experiment runner.\n"
        "2. Proposal D2 / README D2: run the fixed-frequency sweep and commit power, latency, and J/token.\n"
        "3. Proposal D3 / README D3: measure NVML refresh and clock-lock latency. The delay injector and "
        "workload generator are already on main.\n"
        "4. Proposal D6 / README D8 and D9: run the delay x workload matrix with three seeds, including the "
        "age-removed ablation, and commit the failure-boundary and ablation figures. README D10 can supply "
        "a CPU preview before the GPU runs.",
        new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 5, "Difficulties", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.multi_cell(0, 4,
        "Main cannot run a GPU experiment yet: the runner, timing script, and analysis script are not in "
        "the repository. Clock locking works only on the RTX Windows machine in an Administrator terminal, "
        "so the Mac submissions could add tested modules but not the characterization. The published age "
        "weight alone is 0.21 at 600 ms, below the 0.40 down-clock threshold, so the robust controller holds "
        "when telemetry is older than 300 ms. Figure 3 shows that hold. The naive controller and the "
        "age-removed ablation still down-clock. Figure 4 shows the same split in the scale-out simulation. "
        "This behavior is unit-tested and still has to be checked on the GPU.",
        new_x="LMARGIN", new_y="NEXT")

    pdf.output(out_pdf)


def _table(pdf, widths, rows):
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(230, 236, 240)
    headers = ["ID", "Deliverable", "Done", "What is on main"]
    for w, h in zip(widths, headers):
        pdf.cell(w, 5, h, border=1, fill=True)
    pdf.ln()
    pdf.set_font("Helvetica", size=7.5)
    for row in rows:
        pdf.cell(widths[0], 5, row[0], border=1)
        pdf.cell(widths[1], 5, row[1], border=1)
        pdf.cell(widths[2], 5, row[2], border=1, align="C")
        pdf.cell(widths[3], 5, row[3], border=1)
        pdf.ln()


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_work = os.path.join(OUT, "fig1_workload_rates.png")
    fig_delay = os.path.join(OUT, "fig2_delay_buffer.png")
    fig_ctrl = os.path.join(OUT, "fig3_controller_decisions.png")
    fig_scale = os.path.join(OUT, "fig4_scale_miss_rate.png")
    save_workload(fig_work)
    delay_rows = save_delay(fig_delay)
    ctrl_rows = save_controllers(fig_ctrl)
    scale_rows = save_scale(fig_scale)
    joules, stats = energy_check()
    pd.DataFrame(delay_rows, columns=["delay_ms", "returned_power_w", "age_ms"]).to_csv(
        os.path.join(OUT, "delay_buffer_output.csv"), index=False)
    pd.DataFrame(ctrl_rows).drop(columns=["scenario"]).to_csv(
        os.path.join(OUT, "controller_decisions.csv"), index=False)
    pd.DataFrame(scale_rows).to_csv(os.path.join(OUT, "scale_sim_output.csv"), index=False)
    with open(os.path.join(OUT, "metrics_check.txt"), "w", encoding="utf-8") as f:
        f.write(f"joules_from_samples 100W for 2s = {joules:.1f} J\n")
        f.write(f"summary_stats p95 of 1..10 = {stats['p95']}, mean = {stats['mean']}\n")
    pdf_path = os.path.join(OUT, "AMS560_Group10_Midterm_Report.pdf")
    build_pdf(fig_work, fig_delay, fig_ctrl, fig_scale, pdf_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
