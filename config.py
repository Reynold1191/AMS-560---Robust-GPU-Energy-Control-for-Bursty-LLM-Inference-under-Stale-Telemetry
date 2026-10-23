import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

DELAY_VALUES_MS = [0, 100, 250, 500, 1000]
CONTROLLER_NAMES = ["default", "static_high", "static_low", "static_efficient",
                    "naive_slo", "robust", "robust_no_age", "robust_no_budget", "oracle"]
WORKLOAD_NAMES = ["steady", "ramp", "sharp_burst"]


@dataclass
class ExperimentConfig:
    # SLO
    slo_target_s: float = 8.5
    slo_budget_ratio: float = 0.01
    budget_window_s: float = 60.0

    # Telemetry
    telemetry_poll_s: float = 0.1
    delay_ms: float = 0.0
    delay_jitter_ms: float = 0.0

    # Control
    control_interval_s: float = 1.0
    min_dwell_intervals: int = 2
    controller: str = "robust"
    use_telemetry_age: bool = True
    use_error_budget: bool = True

    # Workload
    workload: str = "sharp_burst"
    base_arrival_rate: float = 0.15
    burst_multiplier: float = 5.0
    burst_start_s: float = 40.0
    burst_duration_s: float = 20.0
    ramp_end_rate: float = 2.0
    duration_s: float = 120.0
    warmup_s: float = 10.0
    seed: int = 42

    # Model
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_batch_size: int = 4
    prompts_file: str = "prompts/prompts.jsonl"

    # Clocks
    selected_clocks: List[int] = field(default_factory=list)
    static_efficient_mhz: int = 667
    static_low_mhz: int = 0

    # Scale simulation
    sim_num_workers: int = 8
    sim_trace_requests: int = 10000

    # Output
    run_id: str = field(default_factory=lambda: f"run_{int(time.time())}")
    results_dir: str = "results"
    oracle_table_path: Optional[str] = None

    def results_path(self, filename: str) -> str:
        return os.path.join(self.results_dir, filename)

    def to_dict(self) -> dict:
        return asdict(self)
