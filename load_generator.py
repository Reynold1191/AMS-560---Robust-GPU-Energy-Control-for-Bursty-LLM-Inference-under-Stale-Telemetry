"""
load_generator.py -- Member 4 (Load Generation)

Daemon thread that creates LLM inference requests with Poisson arrivals and
pushes them into the shared RequestQueue.

Workload profiles (selected by `workload`):
    steady       constant rate = base_arrival_rate
    sharp_burst  base rate, x burst_multiplier inside
                 [burst_start_s, burst_start_s + burst_duration_s]
    ramp         linear increase from base_arrival_rate to ramp_end_rate
                 over duration_s (clamped at ramp_end_rate)

Depends on: config.py, request_queue.py (Member 1).
"""

import json
import logging
import os
import random
import statistics
import threading
import time
from typing import List, Optional

from request_queue import Request

log = logging.getLogger(__name__)

_FALLBACK_PROMPTS = [
    {"prompt": "Explain what GPU dynamic voltage and frequency scaling (DVFS) is.", "max_new_tokens": 64},
    {"prompt": "Why does LLM inference latency depend on GPU clock frequency?", "max_new_tokens": 96},
    {"prompt": "List three ways cloud providers reduce energy use in data centers.", "max_new_tokens": 128},
    {"prompt": "What is a latency SLO and why do services track an error budget?", "max_new_tokens": 160},
    {"prompt": "Summarize the trade-off between energy efficiency and tail latency.", "max_new_tokens": 200},
]


# ---------------------------------------------------------------------------
# Pure helpers (shared with scale_simulator.py so both use identical profiles)
# ---------------------------------------------------------------------------

def workload_rate(workload: str, elapsed_s: float, base_rate: float,
                  burst_multiplier: float, burst_start_s: float,
                  burst_duration_s: float, ramp_end_rate: float,
                  duration_s: float) -> float:
    """Arrival rate (requests/s) at `elapsed_s` seconds into the run."""
    if workload == "steady":
        return base_rate
    if workload == "sharp_burst":
        if burst_start_s <= elapsed_s <= burst_start_s + burst_duration_s:
            return base_rate * burst_multiplier
        return base_rate
    if workload == "ramp":
        if duration_s <= 0:
            return ramp_end_rate
        rate = base_rate + (elapsed_s / duration_s) * (ramp_end_rate - base_rate)
        # clamp so the rate never overshoots ramp_end_rate (works for up- or down-ramps)
        lo, hi = sorted((base_rate, ramp_end_rate))
        return min(max(rate, lo), hi)
    raise ValueError(f"Unknown workload '{workload}' (expected steady, ramp, sharp_burst)")


def workload_phase(workload: str, elapsed_s: float, warmup_s: float,
                   burst_start_s: float, burst_duration_s: float) -> str:
    """Phase tag for a request created at `elapsed_s`."""
    if elapsed_s < warmup_s:
        return "warmup"
    if workload == "sharp_burst" and burst_start_s <= elapsed_s <= burst_start_s + burst_duration_s:
        return "burst"
    return "steady"


def load_prompts(path: Optional[str]) -> List[dict]:
    """Read a JSONL prompt file; fall back to a built-in list if missing/empty."""
    if not path or not os.path.exists(path):
        log.warning("Prompt file %s not found; using %d fallback prompts", path, len(_FALLBACK_PROMPTS))
        return list(_FALLBACK_PROMPTS)
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    if not prompts:
        log.warning("Prompt file %s is empty; using fallback prompts", path)
        return list(_FALLBACK_PROMPTS)
    return prompts


# ---------------------------------------------------------------------------
# LoadGenerator thread
# ---------------------------------------------------------------------------

class LoadGenerator(threading.Thread):
    """Generates Poisson arrivals according to a workload profile."""

    def __init__(self, queue, config, workload: str = "sharp_burst", seed: int = 42, prompts=None):
        super().__init__(name="LoadGenerator", daemon=True)
        # run_experiment.py calls LoadGenerator(queue, config, prompts).
        if isinstance(workload, (list, tuple)):
            prompts = list(workload)
            workload = config.workload
            seed = config.seed
        if workload not in ("steady", "ramp", "sharp_burst"):
            raise ValueError(f"Unknown workload '{workload}'")
        self.queue = queue
        self.config = config
        self.workload = workload
        self.seed = seed
        self.rng = random.Random(seed)
        self.prompts = list(prompts) if prompts else self._load_prompts(config.prompts_file)
        self._stop_event = threading.Event()
        self._phase_lock = threading.Lock()
        self._phase_override: Optional[str] = None
        self._request_id = 0
        self._start_t: Optional[float] = None
        self.inter_arrivals: List[float] = []
        self._prev_arrival: Optional[float] = None

    # -- helpers ------------------------------------------------------------
    def _load_prompts(self, path) -> List[dict]:
        return load_prompts(path)

    def _current_rate(self, elapsed_s: float) -> float:
        c = self.config
        return workload_rate(self.workload, elapsed_s, c.base_arrival_rate,
                             c.burst_multiplier, c.burst_start_s, c.burst_duration_s,
                             c.ramp_end_rate, c.duration_s)

    def _phase_tag(self, elapsed_s: float) -> str:
        with self._phase_lock:
            if self._phase_override is not None:
                return self._phase_override
        c = self.config
        return workload_phase(self.workload, elapsed_s, c.warmup_s,
                              c.burst_start_s, c.burst_duration_s)

    def set_phase(self, phase: str) -> None:
        """Runner override. Without this call, phase comes from elapsed time."""
        with self._phase_lock:
            self._phase_override = phase

    def arrival_cv(self, last_n: int = 30) -> float:
        """Coefficient of variation of recent inter-arrival times."""
        iat = self.inter_arrivals[-last_n:] if self.inter_arrivals else []
        if len(iat) < 2:
            return 0.0
        mu = statistics.mean(iat)
        if mu <= 0:
            return 0.0
        return statistics.stdev(iat) / mu

    def recent_arrival_rate(self, window_s: float = 10.0) -> float:
        """Requests per second over the most recent inter-arrival window."""
        iat = self.inter_arrivals
        if not iat:
            return 0.0
        acc = 0.0
        count = 0
        for gap in reversed(iat):
            acc += gap
            count += 1
            if acc >= window_s:
                break
        return count / acc if acc > 0 else 0.0

    # -- thread body ----------------------------------------------------------
    def run(self):
        self._start_t = time.monotonic()
        while not self._stop_event.is_set():
            elapsed = time.monotonic() - self._start_t
            if elapsed >= self.config.duration_s:
                break

            rate = self._current_rate(elapsed)
            sleep_s = self.rng.expovariate(rate) if rate > 0 else 1.0

            if self._stop_event.wait(timeout=sleep_s):
                break  # stop() was called while waiting

            elapsed = time.monotonic() - self._start_t
            if elapsed >= self.config.duration_s:
                break

            idx = self.rng.randrange(len(self.prompts))
            p = self.prompts[idx]
            now = time.monotonic()
            if self._prev_arrival is not None:
                self.inter_arrivals.append(now - self._prev_arrival)
            self._prev_arrival = now
            req = Request(
                request_id=self._request_id,
                prompt=p["prompt"],
                max_new_tokens=int(p.get("max_new_tokens", 64)),
                arrival_t=now,
                phase=self._phase_tag(elapsed),
                seed_idx=idx,
            )
            self.queue.put(req)
            self._request_id += 1

        log.info("LoadGenerator(%s) finished: %d requests", self.workload, self._request_id)

    def stop(self):
        self._stop_event.set()
        if self.is_alive() and threading.current_thread() is not self:
            self.join(timeout=5.0)

    @property
    def n_generated(self) -> int:
        return self._request_id
