# Objective and Description

We ask whether a GPU frequency controller that accounts for *stale
telemetry* can reduce LLM inference energy while keeping a rolling
latency SLO error budget under bursty load. Prior DVFS work for LLM
serving (e.g., throttLL'eM, GreenLLM) assumes the controller sees
current power and utilization. In practice readings arrive late: if a
burst starts while the controller still sees pre-burst samples, it keeps
down-clocking and misses the SLO.

Our robust controller scores risk from four signals: queue length,
telemetry age, error-budget burn rate and arrival burstiness, with
telemetry age weighted highest (0.35). It refuses to down-clock on
telemetry older than 300 ms. We compare it with a naive SLO controller
that ignores age, static-clock baselines, an oracle using a pre-measured
clock table, and two ablations (no age term, no budget term). The
testbed runs Qwen2.5-0.5B-Instruct in FP16 on an RTX 5060 Laptop GPU,
with an 8.5 s SLO and a 1% miss budget over a 60 s window. Staleness is
injected through a delay buffer that serves the newest NVML sample at
least $d$ ms old, together with its true age.

# Timeline and Expected Deliverables

  **Dates**       **Milestone**
  --------------- ---------------------------------------------------------------------------------------------
  Sep 20          Proposal, team plan and module specifications *(done)*
  Oct 19--26      GPU-free modules built, reviewed and merged to `main` *(done)*
  Oct 27--Nov 2   Hardware verification and timing (D3); delay levels from measurements; runner merged (D1)
  Nov 3--9        Fixed-frequency sweep (D2) and oracle table; two-condition smoke test
  Nov 10--16      24-condition demo matrix; first analysis figures
  Nov 17--30      Full matrix with 3 seeds; ablations (D9); failure boundary (D8); scale-out simulation (D10)
  Dec 1--final    Final report and presentation

# Deliverable Status

All GPU-free components are merged to `main` with 204 unit and
integration tests passing on macOS without CUDA. Percentages are our
estimates of total work, including GPU validation.

  **ID**   **Deliverable**                                    **Done** **Status**
  -------- ------------------------------------------------ ---------- ------------------------------------------------------------------------------------------------------------------------------------
  D1       Experiment harness with guaranteed clock reset          40% Config, queue, SLO budget, metrics and load generator are merged and aligned with the runner interface; GPU runner not yet merged.
  D2       Fixed-frequency characterisation                        10% Clock locking via `nvidia-smi -lgc` tested once; sweep not yet run.
  D3       Real timing characterisation                             5% Clock tracking, NVML refresh and settling times measured on the RTX 5060 (Section 5); script not yet on `main`.
  D4       Workload suite (steady, ramp, burst)                    90% Poisson load generator with phase tags and arrival statistics merged and tested.
  D5       Delay injector (with jitter)                            90% Thread-safe delay buffer merged; calibrated levels 0, 500, 1250, 2500 ms set from D3.
  D6       Baselines and oracle                                    75% Static, naive SLO and oracle controllers merged; oracle table needs the D2 sweep.
  D7       Robust controller with acceptance tests                 85% Risk policy, dwell, emergency rules and stale-telemetry guard merged; GPU tuning pending.
  D8       Failure-boundary analysis                               20% Metrics collector and CSV export merged; analysis and figures not started.
  D9       Ablation study                                          50% Both ablation variants and config flags built and tested; runs pending.
  D10      Scale-out simulation                                    70% Trace-driven 8-worker simulator merged; needs D2 service times.

# Testing and Validation to Date

Every module was built against a written interface and checked with
synthetic inputs before integration. The suite on `main` covers the data
layer (queue ordering, delay-buffer age, rolling-window expiry, prompt
bands), static baselines and the controller factory for all nine
controller names, naive and robust policy decisions on eight synthetic
scenarios (calm, SLO violation, budget burn, stale, very stale, queue
build-up, burst, near target), oracle table lookup and fallback, metrics
energy integration and CSV export, and the load generator's arrival
rates and phase tags. Cross-branch review caught three interface bugs
before any GPU time was spent: the factory did not forward the workload
needed for oracle lookups, the dwell constraint could be bypassed by a
default argument, and the controller state and config disagreed on the
SLO target (2 s against 8.5 s).

# Plan for Unfinished Deliverables

- **D3 first, then D1.** Run timing characterisation before any
  controller experiment, then merge `gpu_clock.py`, `telemetry.py`,
  `llm_worker.py` and `run_experiment.py`. Pass a single-condition run
  with clock reset in `finally`.

- **D2, D6.** Sweep the seven selected clocks under steady load to get
  J/token and latency per clock, then build `oracle_table.csv` with
  `oracle_builder.py`.

- **D7, D9.** Run the 24-condition matrix (4 controllers $\times$ 3
  delays $\times$ 2 workloads), then add the no-age and no-budget
  ablations across all delays. Retune risk thresholds only if robust
  exceeds the 1% budget at zero delay.

- **D8.** Write `analyze_results.py` for the eight figures, including a
  heatmap of miss rate over delay and burst multiplier.

- **D10.** Feed D2 service times into the simulator; compare naive and
  robust at 8 workers, 10,000 requests.

# Incorporating Instructor Feedback

**Verify the RTX 5060 Laptop GPU first.** `check_gpu_timing.py` locked
each of the seven candidate clocks twice under a sustained CUDA load and
read the clock back through NVML. 180, 667, 1147, 1635 and 2122 MHz held
within 8 MHz with no spread between repeats, and power rose with clock
(13.6 W to 72.1 W). 2602 and 3090 MHz did not hold: under load they
settled at about 2150 and 2280 MHz, so they are excluded and the
experiment ladder is 180--2122 MHz. Power responded to load (3.2 W idle,
80 W loaded), and `nvidia-smi -rgc` in a `finally` block restored the
default clock behavior.

**Measure delays, then choose injected levels.** The earlier levels (0,
100, 250, 500, 1000 ms) were not grounded in measurement. The script
reads NVML every 11 ms, but the power and clock values change only every
500 ms; that 500 ms is the real sensor refresh, not the polling
interval. After a lock command (about 70 ms), the reported clock holds
its new value after about 0.7 s, and power settles after about 1.25 s.
The injected levels are therefore 0 ms (fresh), 500 ms (one NVML
refresh), 1250 ms (full power settling after a clock change) and
2500 ms (two settling periods). Because power takes longer to settle
than the 1 s control interval, the two-interval dwell (2 s) is what
keeps a decision from being judged on readings taken before the previous
clock change has settled.

# Team Contributions

  **Member**            **Role**                         **Merged so far**
  --------------------- -------------------------------- -----------------------------------------------------------------------------------------------------------------------------
  Nirzaree Devasthale   Data layer, baselines, factory   Config, request queue, delay buffer, SLO budget, prompts; static controller, factory, synthetic test harness, config checks
  Priyanshi Jain        Dynamic policies                 Controller base and dwell, naive SLO, robust, ablation and oracle controllers; acceptance tests
  Quynh Tran            Metrics and analysis             Metrics collector, energy integration, CSV export
  G. Dakate             Load and scale simulation        Load generator, 8-worker scale simulator and tests
  Thai Toan Tran        GPU, LLM, orchestration          Integration of branches into `main`; runner interface alignment for metrics and load generator

# Difficulties

- **Single GPU bottleneck.** All hardware runs depend on one Windows
  laptop with administrator rights. The full matrix (270 runs) may not
  fit, so we prioritize the 24-condition matrix and use the simulator
  for scale.

- **Hardware uncertainty.** If clock locks are not honored reliably, we
  fall back to a smaller verified clock set or to trace-driven
  simulation from the sweep.

- **Specification gap.** With the 0.40 down-clock threshold on `main`, telemetry age alone
  could never block a down-clock (maximum 0.35); with the tuned 0.30
  threshold it does so only beyond about 857 ms. Synthetic tests caught
  this, and the default controller has an explicit 300 ms
  stale-telemetry guard. Since 300 ms is shorter than one NVML refresh
  (500 ms), the robust controller holds its clock at every nonzero
  calibrated delay; whether the guard should sit at the refresh period
  instead is a tuning question for the GPU runs.
