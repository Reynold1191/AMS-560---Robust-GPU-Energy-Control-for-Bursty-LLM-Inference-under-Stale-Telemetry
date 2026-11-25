"""
config.py
Central configuration for the AMS 560 Robust GPU Energy Control project.

All experiment parameters are defined here so that scripts and modules
can import a single ExperimentConfig object and stay consistent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

# Clocks that tracked their lock target under load in check_gpu_timing.py.
# 2602 and 3090 MHz are excluded: under load they settle at ~2150-2280 MHz.
VERIFIED_CLOCKS_MHZ = [180, 667, 1147, 1635, 2122]


@dataclass
class ExperimentConfig:
    # ------------------------------------------------------------------ #
    # SLO / reliability                                                     #
    # ------------------------------------------------------------------ #
    slo_target_s: float = 8.5
    """End-to-end latency target in seconds (request arrival → completion).
       Placeholder: run calibrate_slo.py on the fixed-clock sweep and pass the
       calibrated value with --slo-target (run_demo.py does this)."""

    slo_budget_ratio: float = 0.05
    """Allowed fraction of requests that may miss the SLO (5 % → 95 % SLO).
       With ~30-60 requests per 60 s window, a 1 % budget is exceeded by a
       single miss (burn 30x), so the burn signal could only be 0 or saturated."""

    budget_window_s: float = 60.0
    """Rolling window length for error-budget accounting (seconds)."""

    # ------------------------------------------------------------------ #
    # Telemetry sampling                                                    #
    # ------------------------------------------------------------------ #
    telemetry_poll_s: float = 0.1
    """How often the background thread polls NVML (seconds)."""

    # ------------------------------------------------------------------ #
    # Delay injection                                                       #
    # ------------------------------------------------------------------ #
    delay_ms: float = 0.0
    """Injected telemetry delay in milliseconds (0 = no injection)."""

    delay_jitter_ms: float = 0.0
    """±Uniform jitter added to delay_ms (milliseconds)."""

    # ------------------------------------------------------------------ #
    # Controller                                                            #
    # ------------------------------------------------------------------ #
    control_interval_s: float = 1.0
    """How often the frequency-control logic runs (seconds)."""

    min_dwell_intervals: int = 2
    """Minimum control intervals the clock must stay fixed before changing."""

    controller: str = "robust"
    """One of: default, static_high, static_low, naive_slo, robust, oracle.
       'default' = do not lock the GPU clock (NVIDIA driver manages it).
    """

    # Ablation flags (override controller logic when set)
    use_telemetry_age: bool = True
    """If False, remove the telemetry-age term from the robust policy."""

    use_error_budget: bool = True
    """If False, remove the error-budget term from the robust policy."""

    stale_downclock_ms: Optional[float] = None
    """Legacy hard guard: robust never down-clocks on telemetry older than
       this. attempt_6 showed that with every injected delay >= 500 ms the
       guard froze robust at its start clock, so it is off by default and
       replaced by the delay-aware wait below (kept for the midterm sim)."""

    robust_age_scale_ms: float = 5000.0
    """Telemetry age at which robust's age term saturates (2 x the largest
       injected delay), so the term grows across the whole tested range
       instead of saturating at 1 s."""

    robust_delay_aware: bool = True
    """Robust waits telemetry age + clock settling + one batch time after a
       clock change before stepping down again, and needs more P95 headroom
       the older its telemetry is."""

    clock_settle_s: float = 0.7
    """Measured graphics-clock settling time (check_gpu_timing.py)."""

    capacity_margin: float = 1.25
    """Capacity floor for dynamic controllers: never run a clock whose
       measured capacity is below margin x the observed arrival rate
       (predicted utilisation <= 80 %)."""

    capacity_file: Optional[str] = None
    """capacity.json written by capacity_model.py (run_demo does this)."""

    capacity_rps: Optional[List[float]] = None
    """Requests/s each clock in ``clocks`` can serve at full batch
       (capacity.json from the clock-sensitivity benchmark). None = no floor."""

    batch_time_s: Optional[List[float]] = None
    """Full-batch service time per clock (same source as capacity_rps)."""

    # ------------------------------------------------------------------ #
    # Workload                                                              #
    # ------------------------------------------------------------------ #
    workload: str = "sharp_burst"
    """One of: steady, ramp, sharp_burst, trace."""

    base_arrival_rate: float = 0.3
    """Base arrival rate (req/s). Raised from 0.15, where the GPU averaged
       11-26 % utilisation and energy was mostly idle power."""

    burst_multiplier: float = 4.0
    """Multiplier applied to arrival rate during a burst."""

    burst_start_s: float = 40.0
    """Seconds after the END OF WARM-UP when the burst begins (shared by the
       load generator, the control loop and analyze_results.py)."""

    burst_duration_s: float = 20.0
    """How long the burst lasts (seconds)."""

    ramp_end_rate: Optional[float] = None
    """Arrival rate at the end of a ramp. None = base_arrival_rate x
       ramp_end_multiplier, so the ramp scales with the auto-chosen base rate
       (attempt_6 used a fixed 1.2 req/s below the 2.35 req/s base, so the
       "ramp" went down)."""

    ramp_end_multiplier: float = 3.0
    """Ramp end rate as a multiple of the base rate (when ramp_end_rate is None)."""

    duration_s: float = 120.0
    """Total experiment duration (seconds, including warm-up)."""

    warmup_s: float = 10.0
    """Warm-up phase excluded from statistics (seconds).
       Measurement = duration_s - warmup_s = 110 s: 40 s steady, 20 s burst,
       50 s recovery."""

    drain_timeout_s: float = 60.0
    """Max seconds to wait for queued requests after the arrivals stop.
       Requests still unfinished are counted as SLO misses."""

    seed: int = 42
    """RNG seed for reproducible request generation."""

    # ------------------------------------------------------------------ #
    # Model                                                                 #
    # ------------------------------------------------------------------ #
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    """HuggingFace model identifier."""

    max_batch_size: int = 8
    """Default batch size; run_demo.py picks it with bench_clock_sensitivity.py."""

    prompts_file: str = "prompts/long_context.jsonl"
    """Long-context requests (~0.75-1.3k prompt tokens, 16-32 answer tokens).
       Decode on this laptop is CPU-launch-bound (~57 ms/step at any clock),
       prefill is GPU-bound, so long prompts make service time clock-sensitive.
       The original short prompts are in prompts/prompts.jsonl."""
    """Path to the JSONL file containing prompt records."""

    # ------------------------------------------------------------------ #
    # Clock management                                                      #
    # ------------------------------------------------------------------ #
    selected_clocks: List[int] = field(default_factory=lambda: list(VERIFIED_CLOCKS_MHZ))
    """Experiment clock ladder. Defaults to the clocks verified to track under load."""

    static_efficient_mhz: int = 2122
    """Lowest-energy clock that meets the SLO in the fixed-clock sweep.
       Placeholder until calibrate_slo.py runs (the earlier 667 MHz choice
       came from a sweep where the clock was not actually locked)."""

    static_low_mhz: int = 0
    """Low static clock selected during the fixed-frequency sweep."""

    # ------------------------------------------------------------------ #
    # Scale-out simulation                                                  #
    # ------------------------------------------------------------------ #
    sim_num_workers: int = 8
    """Number of virtual workers in the scale-out trace simulator."""

    sim_trace_requests: int = 10_000
    """Number of requests to replay in the scale-out simulation."""

    # ------------------------------------------------------------------ #
    # Experiment identification                                             #
    # ------------------------------------------------------------------ #
    run_id: str = field(default_factory=lambda: f"run_{int(time.time())}")
    """Unique identifier for this experiment run (used in filenames)."""

    results_dir: str = "results"
    """Directory where CSV logs and plots are saved."""

    # ------------------------------------------------------------------ #
    # Oracle                                                                #
    # ------------------------------------------------------------------ #
    oracle_table_path: Optional[str] = None
    """Path to the pre-measured frequency→latency CSV used by the oracle."""

    def results_path(self, filename: str) -> str:
        """Return an absolute path inside the results directory."""
        return os.path.join(self.results_dir, filename)

    def to_dict(self) -> dict:
        """Serialize to a plain dict for logging."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def make_config(**overrides) -> ExperimentConfig:
    """Create an ExperimentConfig with optional field overrides."""
    return ExperimentConfig(**overrides)


# Predefined configurations used by run_all.ps1 / run_experiment.py
# Calibrated by check_gpu_timing.py on the RTX 5060 Laptop GPU:
#   0    fresh telemetry
#   500  one NVML power/clock refresh (measured 500 ms)
#   1250 full power settling after a clock change (measured ~1224 ms, clock ~700 ms)
#   2500 two full settling periods (stress)
DELAY_VALUES_MS = [0, 500, 1250, 2500]
CONTROLLER_NAMES = ["default", "static_high", "static_low", "static_efficient",
                    "naive_slo", "robust", "robust_v1", "robust_no_age", "robust_no_budget",
                    "oracle"]
WORKLOAD_NAMES = ["steady", "ramp", "sharp_burst"]
ABLATION_CONFIGS = [
    {"controller": "robust", "use_telemetry_age": True,  "use_error_budget": True},
    {"controller": "robust", "use_telemetry_age": False, "use_error_budget": True},
    {"controller": "robust", "use_telemetry_age": True,  "use_error_budget": False},
]
