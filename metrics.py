"""Energy, latency summaries, and per-run CSV logs.

The collector accepts both call styles used in this project:

- Team plan: ``MetricsCollector(run_id, results_dir, slo_target_s)``
- Experiment runner: ``MetricsCollector(slo_target_s, results_dir, run_id)``
"""

import csv
import os
import threading


def joules_from_samples(samples) -> float:
    """Trapezoidal energy in joules. Returns 0 when fewer than two samples exist."""
    if samples is None or len(samples) < 2:
        return 0.0
    ordered = sorted(samples, key=lambda sample: sample.t)
    total = 0.0
    for previous, current in zip(ordered, ordered[1:]):
        dt = current.t - previous.t
        if dt <= 0:
            continue
        total += 0.5 * (previous.power_w + current.power_w) * dt
    return total


def summary_stats(latencies) -> dict:
    """Percentiles use the same index rule as SLOBudget.percentile_latency."""
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    values = sorted(latencies)
    n = len(values)

    def percentile(p: float) -> float:
        return values[min(int(p * n), n - 1)]

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "mean": sum(values) / n,
    }


class MetricsCollector:
    def __init__(self, run_id, results_dir, slo_target_s, warmup_s=0):
        if isinstance(run_id, (int, float)) and not isinstance(run_id, bool):
            slo_target_s, results_dir, run_id = run_id, results_dir, slo_target_s
        self.run_id = str(run_id)
        self.results_dir = results_dir
        self.output_dir = results_dir
        self.slo_target_s = float(slo_target_s)
        self.warmup_s = warmup_s
        self._lock = threading.Lock()
        self._requests = []
        self._control_events = []
        self._telemetry = []
        self._warmup_end_t = None
        self.requests = self._requests
        self.control_events = self._control_events
        self.telemetry = self._telemetry

    @property
    def warmup_end_t(self):
        return self._warmup_end_t

    @warmup_end_t.setter
    def warmup_end_t(self, value):
        with self._lock:
            self._warmup_end_t = value

    def set_warmup_end(self, t) -> None:
        self.warmup_end_t = t

    def record_request(self, request) -> None:
        with self._lock:
            self._requests.append(request)

    on_request = record_request

    def record_control(self, event) -> None:
        with self._lock:
            self._control_events.append(dict(event))

    add_control_event = record_control
    record_control_event = record_control

    def record_telemetry(self, sample) -> None:
        with self._lock:
            self._telemetry.append(sample)

    def measurement_requests(self):
        with self._lock:
            requests = list(self._requests)
            warmup_end = self._warmup_end_t
        if warmup_end is None:
            return requests
        return [
            request for request in requests
            if request.finish_t is not None and request.finish_t >= warmup_end
        ]

    def latencies(self):
        return [
            request.latency_s
            for request in self.measurement_requests()
            if request.latency_s is not None
        ]

    def total_output_tokens(self) -> int:
        return sum(
            getattr(request, "generated_tokens", 0) or 0
            for request in self.measurement_requests()
        )

    def miss_rate(self) -> float:
        latencies = self.latencies()
        if not latencies:
            return 0.0
        misses = sum(latency > self.slo_target_s for latency in latencies)
        return misses / len(latencies)

    slo_miss_rate = miss_rate

    def _measurement_samples(self, samples):
        if samples is None:
            with self._lock:
                samples = list(self._telemetry)
        warmup_end = self._warmup_end_t
        if warmup_end is None:
            return list(samples)
        return [sample for sample in samples if getattr(sample, "t", 0) >= warmup_end]

    def run_summary(self, samples, config, recovery_s=None, max_queue_len=0) -> dict:
        latencies = self.latencies()
        tokens = self.total_output_tokens()
        total_j = joules_from_samples(self._measurement_samples(samples))
        stats = summary_stats(latencies)
        return {
            "run_id": self.run_id,
            "seed": getattr(config, "seed", 0),
            "controller": getattr(config, "controller", ""),
            "delay_ms": getattr(config, "delay_ms", 0.0),
            "workload": getattr(config, "workload", ""),
            "total_j": total_j,
            "output_tokens": tokens,
            "j_per_token": (total_j / tokens) if tokens else float("nan"),
            "p50": stats["p50"],
            "p95": stats["p95"],
            "p99": stats["p99"],
            "mean_lat": stats["mean"],
            "miss_rate": self.miss_rate(),
            "n_requests": len(latencies),
            "max_queue": max_queue_len,
            "recovery_s": recovery_s,
            "use_age": getattr(config, "use_telemetry_age", True),
            "use_budget": getattr(config, "use_error_budget", True),
            "burst_multiplier": getattr(config, "burst_multiplier", 1.0),
        }

    def energy_per_token(self) -> float:
        tokens = self.total_output_tokens()
        if not tokens:
            return 0.0
        return joules_from_samples(self._measurement_samples(None)) / tokens

    def save_requests_csv(self) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"{self.run_id}_requests.csv")
        fields = [
            "request_id", "arrival_t", "start_t", "finish_t", "latency_s",
            "queue_wait_s", "service_time_s", "prompt_tokens", "output_tokens",
            "phase", "slo_miss",
        ]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for request in self.measurement_requests():
                row = request.to_dict() if hasattr(request, "to_dict") else {}
                latency = row.get("latency_s", getattr(request, "latency_s", None))
                row["slo_miss"] = (
                    latency > self.slo_target_s if latency is not None else None
                )
                writer.writerow({key: row.get(key, "") for key in fields})
        return path

    export_requests_csv = save_requests_csv

    def save_telemetry_csv(self, samples=None) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"{self.run_id}_telemetry.csv")
        if samples is None:
            with self._lock:
                samples = list(self._telemetry)
        fields = ["t", "power_w", "graphics_mhz", "sm_mhz", "util_pct", "temp_c", "phase"]
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                writer.writerow({field: getattr(sample, field, None) for field in fields})
        return path

    export_telemetry_csv = save_telemetry_csv

    def save_control_csv(self) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"{self.run_id}_control.csv")
        with self._lock:
            events = list(self._control_events)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            if not events:
                return path
            fields = list(events[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(events)
        return path

    export_control_csv = save_control_csv

    def append_run_summary(self, summary_dict) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "run_summary.csv")
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_dict.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(summary_dict)
        return path
