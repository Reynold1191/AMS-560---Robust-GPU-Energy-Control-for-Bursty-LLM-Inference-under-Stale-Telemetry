AMS 560 Midterm Report
Group 10
Robust GPU Energy Control for Bursty LLM Inference under Stale Telemetry
Due October 27, 2026

Objective and description

The project asks whether a GPU frequency controller that treats stale telemetry as a risk signal can lower the energy of LLM inference without breaking a latency service-level objective (SLO). The planned system serves Qwen2.5-0.5B-Instruct on one RTX 5060 laptop GPU. A background sampler reads power and clock, a delay buffer can hold those readings back by 0–1000 ms, and a controller runs once per second and chooses a graphics clock. Higher clocks finish requests sooner and use more energy. The SLO target in the plan is 8.5 seconds, with a 1% error budget over a 60-second window. Workloads are steady, ramp, and sharp burst. Compared controllers are a fixed efficient clock, a latency-only baseline (naive SLO), and the proposed robust controller, including ablations that remove the telemetry-age term or the error-budget term.

This report counts only what is on the shared repository main branch (github.com/Reynold1191/AMS-560---Robust-GPU-Energy-Control-for-Bursty-LLM-Inference-under-Stale-Telemetry) as of October 26, 2026.

Timeline and expected deliverables

September 20: repository created with the README and team plan.
October 23: Member 1 data layer and the static-controller branch; Member 2 controller logic.
October 25: Member 3 metrics collector.
October 26: Member 1–3 branches merged to main. Later the same day, Member 4’s load generator and scale simulator were merged, and the load generator was aligned with the experiment runner.
October 27: this report.
After the midterm: finish the GPU testbed, timing measurements, frequency characterization, and failure-boundary evaluation, then commit those results to main.

Percentages are the share of each deliverable that is on `main`. Work that is not in the repository counts as 0%. The two lists use the same D-numbers for different items.

Proposal deliverables (proposal Section 6)

| ID | Deliverable | Done | Note |
|----|-------------|------|------|
| D1 | RTX 5060 inference and synchronized telemetry testbed | 25% | Configuration, the request queue, prompts, and the metrics logger are on main. The LLM worker, GPU telemetry sampler, and experiment runner are not. |
| D2 | Frequency–energy–latency characterization | 0% | No sweep and no power, latency, throughput, or J/token comparison is on main. |
| D3 | Observability and stale-telemetry framework | 67% | The delay injector and the steady/ramp/burst workload generator are on main and tested. Native telemetry-refresh and clock-actuation measurements are not, and no evaluation results are in the repository. |
| D4 | Baseline implementations | 100% | NVIDIA-default, static-clock, naive DVFS, and fresh-telemetry (oracle) controllers are on main and the factory tests pass. No GPU comparison results are in the repository. |
| D5 | Telemetry-age-aware DVFS controller | 100% | The age-aware controller is on main, and the acceptance tests pass (204 tests in the suite). No GPU runs are in the repository. |
| D6 | Failure-boundary evaluation and ablation | 25% | The age-removed ablation controller is on main and unit-tested. The delay × burst failure map, the experimental comparison, and the plots are not. |
| | **Overall** | **53%** | **2 of 6** are fully on main (D4 and D5). (25+0+67+100+100+25) / 6 = 52.8%, reported as 53%. |

README deliverables (repository `README.md` checklist)

| ID | Deliverable | Done | Note |
|----|-------------|------|------|
| D1 | Experiment harness and guaranteed clock reset | 0% | `run_experiment.py` is not on main. |
| D2 | Fixed-frequency characterisation | 0% | The `--sweep-clocks` path and `sweep_summary.csv` are not on main. |
| D3 | Real timing characterisation | 0% | `measure_timing.py` is not on main. |
| D4 | Workload suite (steady / ramp / burst) | 100% | `load_generator.py` is on main and tested. It also accepts the prompt list and phase/arrival calls used by the experiment runner. |
| D5 | Delay injector (0–1000 ms + jitter) | 100% | `delay_buffer.py` is on main and covered by tests. |
| D6 | Baselines and oracle | 100% | `controllers/` is on main and the factory tests pass. No experiment outputs are in the repository. `oracle_builder.py` is not on main. |
| D7 | Robust controller with acceptance tests | 100% | `controllers/robust_delay_budget.py` and the acceptance tests are on main. No GPU result files are in the repository. |
| D8 | Failure-boundary analysis | 0% | `analyze_results.py` Figure 7 is not on main. |
| D9 | Ablation and non-redundancy | 50% | `robust_no_age` and `robust_no_budget` are implemented and tested. The `--no-age` / `--no-budget` run flags and Figure 8 are not on main. |
| D10 | Scale-out simulation and reproducibility | 100% | `scale_simulator.py` is on main and tested. No simulation result file is committed. |
| | **Overall** | **55%** | **5 of 10** are fully on main (D4, D5, D6, D7, and D10). D9 is half. (0+0+0+100+100+100+100+0+50+0) / 10 = 55%. |

Plan for unfinished work

1. Proposal D1 / README D1: add the LLM worker, synchronized GPU telemetry sampler, and experiment runner to main.
2. Proposal D2 / README D2: run the fixed-frequency characterization and commit the power, latency, throughput, and J/token comparison.
3. Proposal D3 / README D3: add the telemetry-refresh and clock-actuation measurements. The delay injector (README D5) and the workload generator (README D4) are already on main.
4. Proposal D6 / README D8 and D9: run the delay × workload × controller matrix, including the age-removed ablation, and commit the failure-boundary and ablation plots. Proposal D4 and D5 (README D6 and D7) and the scale simulator (README D10) can be used as they are.
5. The proposal also calls for at least three paired seeds on the 24 primary conditions (3 workloads × 4 delays × 2 dynamic controllers). That run is not on main.

Difficulties

The shared main branch cannot run a GPU experiment yet. Proposal D1 and D2, and README D1, D2, D3, and D8, still need the runner, the timing script, and the analysis script. README D4 and D10 are now on main. GPU clock locking can be done only on the RTX Windows machine in an Administrator terminal, so the Mac submissions could add tested controller modules but not the characterization or the failure-boundary evaluation. The robust-controller tests on main also show that the published age weight alone does not block a down-clock at 600 ms of staleness, because that term contributes 0.21 and the original low-risk threshold is 0.40. The controller therefore holds when telemetry is older than 300 ms. That passes the unit test, and it still has to be checked on a real GPU run once the experiment runner is on main.
