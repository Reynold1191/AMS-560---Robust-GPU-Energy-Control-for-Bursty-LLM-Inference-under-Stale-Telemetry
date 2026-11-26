"""
metrics.py
Energy calculation (trapezoidal integration), percentile statistics,
and a thread-safe MetricsCollector used by the experiment runner.
"""

from __future__ import annotations

import csv
import math
import os
import statistics
import threading
import time
from typing import Dict, List, Optional, Tuple

from request_queue import Request
from telemetry import GpuSample


# ---------------------------------------------------------------------------
# Energy integration
# ---------------------------------------------------------------------------

def joules_from_samples(samples: List[GpuSample]) -> float:
    """
    Compute total GPU energy in joules from timestamped power samples using
    the trapezoidal rule:

        E ≈ Σ  0.5 × (P_i + P_{i+1}) × (t_{i+1} − t_i)

    Handles unsorted samples and empty lists gracefully.
    """
    if len(samples) < 2:
        return 0.0
    ordered = sorted(samples, key=lambda s: s.t)
    energy = 0.0
    for a, b in zip(ordered, ordered[1:]):
        dt = b.t - a.t
        energy += 0.5 * (a.power_w + b.power_w) * dt
    return energy


def joules_per_token(samples: List[GpuSample], total_tokens: int) -> float:
    """Normalised energy efficiency metric."""
    j = joules_from_samples(samples)
    return j / total_tokens if total_tokens > 0 else float("nan")


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------

def percentile(values: List[float], p: float) -> float:
    """
    p-th percentile of a list of floats (0 ≤ p ≤ 1).
    Returns NaN for empty lists.
    """
    if not values:
        return float("nan")
    s = sorted(values)
    idx = min(int(math.ceil(p * len(s))) - 1, len(s) - 1)
    return s[max(idx, 0)]


def summary_stats(values: List[float]) -> Dict[str, float]:
    """Return a dict with mean, std, p50, p95, p99, min, max."""
    if not values:
        nan = float("nan")
        return dict(mean=nan, std=nan, p50=nan, p95=nan, p99=nan,
                    min=nan, max=nan, n=0)
    return {
        "mean": statistics.mean(values),
        "std":  statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50":  percentile(values, 0.50),
        "p95":  percentile(values, 0.95),
        "p99":  percentile(values, 0.99),
        "min":  min(values),
        "max":  max(values),
        "n":    len(values),
    }


# ---------------------------------------------------------------------------
# Arrival-rate coefficient of variation
# ---------------------------------------------------------------------------

def arrival_cv(inter_arrivals: List[float]) -> float:
    """CV of inter-arrival times (high = bursty)."""
    if len(inter_arrivals) < 2:
        return 0.0
    mu = statistics.mean(inter_arrivals)
    if mu <= 0:
        return 0.0
    return statistics.stdev(inter_arrivals) / mu


# ---------------------------------------------------------------------------
# Thread-safe metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """
    Accumulates all request and run-level metrics during an experiment.

    Thread safety: on_request() and add_control_event() may be called
    from multiple threads; internal state is protected by a Lock.
    """

    def __init__(self, slo_target_s: float, results_dir: str, run_id: str):
        self.slo_target_s = slo_target_s
        self.results_dir = results_dir
        self.run_id = run_id
        os.makedirs(results_dir, exist_ok=True)

        self._lock = threading.Lock()
        self._requests:      List[Request]  = []
        self._control_events: List[dict]    = []
        self._warmup_end_t:   Optional[float] = None

    # ------------------------------------------------------------------ #
    # Callbacks                                                             #
    # ------------------------------------------------------------------ #

    def set_warmup_end(self, t: float) -> None:
        with self._lock:
            self._warmup_end_t = t

    def on_request(self, req: Request) -> None:
        """Called by LLMWorker for every completed request."""
        with self._lock:
            self._requests.append(req)

    def add_control_event(self, event: dict) -> None:
        """Called by the control loop on every decision."""
        with self._lock:
            self._control_events.append(event)

    # ------------------------------------------------------------------ #
    # Post-run analysis                                                     #
    # ------------------------------------------------------------------ #

    def measurement_requests(self) -> List[Request]:
        """Only requests that completed after the warm-up phase."""
        with self._lock:
            if self._warmup_end_t is None:
                return list(self._requests)
            return [r for r in self._requests
                    if r.finish_t is not None
                    and r.finish_t >= self._warmup_end_t]

    def latencies(self) -> List[float]:
        return [r.latency_s for r in self.measurement_requests()
                if r.latency_s is not None]

    def miss_rate(self) -> float:
        lats = self.latencies()
        if not lats:
            return float("nan")
        return sum(1 for l in lats if l > self.slo_target_s) / len(lats)

    def total_output_tokens(self) -> int:
        return sum(r.generated_tokens for r in self.measurement_requests())

    def run_summary(
        self,
        samples: List[GpuSample],
        config,
        recovery_s: Optional[float] = None,
        max_queue_len: int = 0,
    ) -> dict:
        """Compute the full run summary dict for run_summary.csv."""
        lats = self.latencies()
        tok = self.total_output_tokens()
        meas_samples = [s for s in samples
                        if self._warmup_end_t is None
                        or s.t >= self._warmup_end_t]

        j_total = joules_from_samples(meas_samples)
        lat_stats = summary_stats(lats)

        return {
            "run_id":        self.run_id,
            "seed":          getattr(config, "seed", 0),
            "controller":    config.controller,
            "delay_ms":      config.delay_ms,
            "workload":      config.workload,
            "total_j":       j_total,
            "output_tokens": tok,
            "j_per_token":   j_total / tok if tok > 0 else float("nan"),
            "p50":           lat_stats["p50"],
            "p95":           lat_stats["p95"],
            "p99":           lat_stats["p99"],
            "mean_lat":      lat_stats["mean"],
            "miss_rate":     self.miss_rate(),
            "n_requests":    len(lats),
            "max_queue":     max_queue_len,
            "recovery_s":    recovery_s or float("nan"),
            "use_age":       getattr(config, "use_telemetry_age", True),
            "use_budget":    getattr(config, "use_error_budget", True),
            "burst_multiplier": getattr(config, "burst_multiplier", 1.0),
        }

    # ------------------------------------------------------------------ #
    # CSV serialisation                                                     #
    # ------------------------------------------------------------------ #

    def save_requests_csv(self) -> str:
        """Write requests.csv and return the path."""
        path = os.path.join(self.results_dir, f"{self.run_id}_requests.csv")
        fields = ["request_id", "arrival_t", "start_t", "finish_t",
                  "latency_s", "queue_wait_s", "service_time_s",
                  "prompt_tokens", "output_tokens", "phase",
                  "slo_miss"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in self.measurement_requests():
                row = r.to_dict()
                row["slo_miss"] = (
                    r.latency_s > self.slo_target_s
                    if r.latency_s is not None else None
                )
                w.writerow({k: row.get(k, "") for k in fields})
        return path

    def save_telemetry_csv(self, samples: List[GpuSample]) -> str:
        """Write telemetry.csv and return the path."""
        path = os.path.join(self.results_dir, f"{self.run_id}_telemetry.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t", "power_w", "graphics_mhz", "sm_mhz",
                         "util_pct", "temp_c", "phase"])
            for s in samples:
                w.writerow([
                    s.t, s.power_w, s.graphics_mhz, s.sm_mhz,
                    s.util_pct, s.temp_c, s.phase
                ])
        return path

    def save_control_csv(self) -> str:
        """Write control.csv and return the path."""
        path = os.path.join(self.results_dir, f"{self.run_id}_control.csv")
        with self._lock:
            events = list(self._control_events)
        if not events:
            return path
        fields = list(events[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(events)
        return path

    def append_run_summary(self, summary: dict) -> str:
        """Append one row to the shared run_summary.csv."""
        path = os.path.join(self.results_dir, "run_summary.csv")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()),
                               extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(summary)
        return path
