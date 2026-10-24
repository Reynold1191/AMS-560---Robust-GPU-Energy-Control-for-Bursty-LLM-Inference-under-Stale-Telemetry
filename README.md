# AMS 560 -- Robust GPU Energy Control for Bursty LLM Inference under Stale Telemetry

**A Delay- and SLO-Error-Budget-Aware DVFS Approach**
AMS 560 * Big Data Systems, Algorithms and Networks * Stony Brook University

---

## Implementation Status and Mac Development

The data layer from `origin/member1-data-layer` and static/factory/test harness
from `origin/controllers-static-factory` are integrated with Member 2's base and
dynamic policies on `member2-controller-logic`. The GPU runner, load generation,
metrics, and analysis scripts described below are planned team deliverables.
See [TEAM_PLAN.md](TEAM_PLAN.md) for the current ownership and integration order.

Member 1 owns the data layer, `controllers/static.py`, `controllers/__init__.py`,
shared synthetic scenarios, and controller/config compatibility tests. Member 2
owns `controllers/base.py`, `controllers/naive_slo.py`,
`controllers/robust_delay_budget.py`, and policy/oracle acceptance tests.

Run the implemented tests on Mac without a GPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

To make a controller decision:

```python
from config import ExperimentConfig
from controllers import ControllerState, get_controller

cfg = ExperimentConfig()
clocks = [180, 667, 1147, 1635, 2122, 2602, 3090]
controller = get_controller("robust", clocks, cfg)
state = ControllerState(
    slo_target_s=cfg.slo_target_s,
    p95_latency_s=2.0,
    telemetry_age_ms=600.0,
)
idx = controller.decide(state, current_idx=3)
print(controller.clock_mhz(idx), controller.last_reason)  # Holds at 1635 MHz
```

Supply increasing, positive clock values and the experiment's SLO in each state.
The configured controller target governs policy decisions; state SLO metadata
supports logging and `latency_slack`. Both default to 8.5 seconds. Pass custom
targets from `cfg` when building state so slack and logging use the same target.
`decide()` uses the controller's configured dwell unless explicitly overridden;
all increases bypass dwell. Controllers select indices; Member 5's runner will
apply the corresponding MHz to the GPU. For `default`, the runner must skip
clock locking so the NVIDIA driver controls frequency.

---

## Research Question

> Can a delay-aware GPU frequency controller reduce LLM inference energy
> while maintaining a rolling latency SLO error budget under bursty workloads
> and stale telemetry?

Prior work on GPU DVFS for LLM inference assumes timely observations.
This project studies a different failure mode: what happens when telemetry
is delayed or stale at the exact moment a workload burst arrives?

---

## LLM Model

**Primary model:** `Qwen/Qwen2.5-0.5B-Instruct`
- 0.5 billion parameters, instruction-tuned, by Alibaba Qwen team
- Runs in float16 on a single laptop GPU (RTX 5060)
- Small enough for rapid experimentation while generating real CUDA load

**Optional second model** (for external validity): `Qwen/Qwen2.5-1.5B-Instruct`
- Switch via: `python run_experiment.py --model Qwen/Qwen2.5-1.5B-Instruct`
- Keep model fixed within any experiment block so frequency is the controlled variable

---

## Project Structure

```
ams560_gpu_control/
|-- config.py                  # Central experiment configuration
|-- gpu_clock.py               # Clock enumeration & nvidia-smi control
|-- telemetry.py               # NVML background sampler
|-- delay_buffer.py            # Stale-telemetry injection buffer
|-- slo_budget.py              # Rolling SLO error-budget tracker
|-- request_queue.py           # Request dataclass + thread-safe queue
|-- load_generator.py          # Workload generation (steady/ramp/burst)
|-- llm_worker.py              # HuggingFace LLM inference worker thread
|-- metrics.py                 # Energy (trapz), latency stats, CSV logging
|-- controllers/
|   |-- __init__.py            # Factory: get_controller(name, clocks, cfg)
|   |-- base.py                # BaseController + ControllerState dataclass
|   |-- static.py              # Static frequency (high / low baseline)
|   |-- naive_slo.py           # Naive SLO-aware (no age, no budget) baseline
|   \-- robust_delay_budget.py # Proposed robust controller + oracle mode
|-- run_experiment.py          # Main CLI experiment runner
|-- measure_timing.py          # Characterise real telemetry/clock timing (D3)
|-- oracle_builder.py          # Build oracle lookup table from sweep (D2 -> oracle)
|-- scale_simulator.py         # Trace-driven multi-worker scale-out simulator (D10)
|-- analyze_results.py         # Post-run analysis and 8 required figures
|-- run_all.ps1                # Experiment matrix orchestration (full + demo modes)
|-- setup.ps1                  # One-time environment setup
|-- prompts/
|   \-- prompts.jsonl          # 30 diverse LLM prompts (short/medium/long)
\-- results/                   # All CSV logs and PNG figures (auto-created)
```

---

## Hardware Requirements

| Requirement | Status |
|-------------|--------|
| NVIDIA GPU with DVFS support | [OK] RTX 5060 Laptop GPU confirmed |
| nvidia-smi clock lock (-lgc) | [OK] Tested and passing |
| CUDA-enabled PyTorch | Required |
| Administrator terminal (Windows) | Required for clock control |

---

## Quick Start

### 1. Environment Setup

Run from an **Administrator PowerShell**:

```powershell
.\setup.ps1
```

Checks the GPU, installs Python dependencies, verifies clock locking,
and runs controller unit tests.

### 2. Demo Run (~30 minutes) -- RECOMMENDED FIRST RUN

```powershell
.\run_all.ps1 -Demo
```

Runs the **24 primary conditions**:
- 3 workloads: steady, ramp, sharp_burst
- 4 delays: 0, 100, 250, 500 ms
- 2 dynamic controllers: naive_slo, robust
- 1 seed, 60 s duration, 5 s warmup

Produces all key comparison plots in `results/`.

### 3. Quick Smoke Test (~5 minutes)

```powershell
.\run_all.ps1 -QuickTest
```

2 delays x 2 controllers x 1 workload = 4 conditions.

### 4. Full Experiment Matrix (~4-8 hours)

```powershell
.\run_all.ps1
```

Covers 5 delays x 6 controllers x 3 workloads x 3 seeds = 270 conditions.

### 5. Individual Runs

```powershell
# Robust controller, 500 ms delay, sharp burst
python run_experiment.py --controller robust --delay-ms 500 --workload sharp_burst

# Naive SLO controller, 0 delay, steady workload
python run_experiment.py --controller naive_slo --delay-ms 0 --workload steady

# Fixed-frequency sweep (Deliverable D2)
python run_experiment.py --sweep-clocks

# Ablation: remove telemetry-age term
python run_experiment.py --controller robust --delay-ms 500 --no-age

# Ablation: remove error-budget term
python run_experiment.py --controller robust --delay-ms 500 --no-budget
```

### 6. Timing Characterisation (Deliverable D3)

```powershell
python measure_timing.py
```

### 7. Scale-Out Simulation (Deliverable D10)

```powershell
python scale_simulator.py --n-workers 8 --n-requests 10000 --controller robust --delay-ms 250
python scale_simulator.py --n-workers 8 --n-requests 10000 --controller naive_slo --delay-ms 250
```

### 8. Analysis and Figures

```powershell
python analyze_results.py
```

---

## Controllers

| Name | Description |
|------|-------------|
| default | No clock locking -- NVIDIA driver manages frequency |
| static_high | Locked to highest supported clock |
| static_low | Locked to lowest feasible clock |
| static_efficient | Locked to the supported clock nearest the sweep's efficient clock (default 667 MHz) |
| naive_slo | Adapts to P95 latency + queue from delayed telemetry; ignores age/budget |
| robust | Uses telemetry age AND error-budget burn rate as explicit risk signals |
| robust_no_age | Robust policy with the age risk term and stale-observation guard disabled |
| robust_no_budget | Robust policy with budget risk and budget emergency disabled |
| oracle | Fresh-telemetry upper bound using pre-measured frequency lookup table |

### Robust Controller Risk Score

```
risk = 0.30 * q_norm + 0.35 * age_norm + 0.20 * burn_norm + 0.15 * cv_norm

q_norm    = min(queue_len / 10, 1)
age_norm  = min(telemetry_age_ms / 1000, 1)   <- KEY: staleness signal (largest weight)
burn_norm = min(error_budget_burn, 1)          <- KEY: long-horizon reliability
cv_norm   = min(arrival_cv, 1)
```

Priority rules:
1. Emergency  (P95 > SLO OR burn > 2.0) -> up-clock by +2 (bypasses dwell)
2. High risk  (risk > 0.55)             -> up-clock by +1
3. Safe       (risk < 0.40 AND P95 < 0.80 * SLO) -> down-clock by -1,
              provided age <= 300 ms or use_telemetry_age is disabled
4. Hold       otherwise

Risk inputs are clamped to [0, 1]. Budget emergencies apply only when the budget
feature is enabled. The explicit stale-observation guard preserves the specified
risk weights while preventing reductions in the 600 ms acceptance case; see the
formula/test resolution in the team plan. `last_risk` and `last_reason` expose
the decision signals for logging.

The oracle loads CSVs with `phase,min_clock_mhz` or Member 3's
`workload_bucket,min_clock_mhz` schema. The factory passes `config.workload` for
workload tables. It rounds minimum frequencies up to supported clocks and falls
back to the robust policy if a table cannot be used. Fresh telemetry must be
supplied by the experiment runner for the intended oracle comparison.

---

## Demo Experiment Matrix (24 Conditions -- ~30 min)

| Axis | Values |
|------|--------|
| Controllers | naive_slo, robust |
| Delays (ms) | 0, 100, 250, 500 |
| Workloads | steady, ramp, sharp_burst |
| Seeds | 1 (fixed) |
| Duration | 60 s + 5 s warmup |

Each run ~60-75 s including model inference.
Total wall time ~25-30 min including model loading and analysis.

---

## Deliverables Checklist

| ID | Deliverable | Script |
|----|------------|--------|
| D1 | Experiment harness + guaranteed clock reset | run_experiment.py |
| D2 | Fixed-frequency characterisation | --sweep-clocks flag |
| D3 | Real timing characterisation | measure_timing.py |
| D4 | Workload suite (steady/ramp/burst) | load_generator.py |
| D5 | Delay injector (0-1000 ms + jitter) | delay_buffer.py |
| D6 | Baselines + oracle | controllers/ |
| D7 | Robust controller with acceptance tests | controllers/robust_delay_budget.py |
| D8 | Failure-boundary analysis | analyze_results.py Fig 7 |
| D9 | Ablation + non-redundancy | --no-age / --no-budget flags |
| D10 | Scale-out simulation + reproducibility | scale_simulator.py |

---

## Output Logs (in results/)

| File | Contents |
|------|----------|
| {run_id}_telemetry.csv | t, power_w, graphics_mhz, sm_mhz, util_pct, temp_c, phase |
| {run_id}_requests.csv | request_id, arrival_t, finish_t, latency_s, slo_miss, ... |
| {run_id}_control.csv | t, selected_mhz, telemetry_age_ms, queue_len, burn_rate, ... |
| run_summary.csv | One row per run: j_per_token, p95, miss_rate, recovery_s, ... |
| sweep_summary.csv | Fixed-clock sweep energy/latency characterisation |
| oracle_table.csv | Per-bucket minimum feasible clocks |
| timing_characterisation.json | NVML update rates, command latencies |

---

## Figures Produced

1. Telemetry delay vs J/token (per controller)
2. Telemetry delay vs SLO miss rate (with budget line)
3. Burst time-series (queue, clock, age, P95)
4. Cumulative energy over time
5. Error-budget burn rate over time
6. Energy gap from oracle
7. Failure boundary heatmap (delay x burst intensity)
8. Ablation comparison

---

## References

1. throttLL'eM (2024): https://arxiv.org/abs/2408.05235
2. GreenLLM (2025): https://arxiv.org/abs/2508.16449
3. EcoInfer (2026): https://www.mdpi.com/2079-9292/15/10/2139
4. BEAM MLSys (2026): https://proceedings.mlsys.org/paper_files/paper/2026/hash/eb3c42ddfa16d8421fdba13528107cc1-Abstract-Conference.html
5. KAIROS (2026): https://arxiv.org/abs/2604.16682
6. Google SRE Book: https://sre.google/sre-book/service-level-objectives/
7. NVIDIA nvidia-smi: https://docs.nvidia.com/deploy/nvidia-smi/index.html
