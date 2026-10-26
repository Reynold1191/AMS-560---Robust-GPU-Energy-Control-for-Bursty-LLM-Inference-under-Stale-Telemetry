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

    def __init__(self, queue, config, workload: str = "sharp_burst", seed: int = 42):
        super().__init__(name="LoadGenerator", daemon=True)
        if workload not in ("steady", "ramp", "sharp_burst"):
            raise ValueError(f"Unknown workload '{workload}'")
        self.queue = queue
        self.config = config
        self.workload = workload
        self.seed = seed
        self.rng = random.Random(seed)
        self.prompts = self._load_prompts(config.prompts_file)
        self._stop_event = threading.Event()
        self._request_id = 0
        self._start_t: Optional[float] = None

    # -- helpers ------------------------------------------------------------
    def _load_prompts(self, path) -> List[dict]:
        return load_prompts(path)

    def _current_rate(self, elapsed_s: float) -> float:
        c = self.config
        return workload_rate(self.workload, elapsed_s, c.base_arrival_rate,
                             c.burst_multiplier, c.burst_start_s, c.burst_duration_s,
                             c.ramp_end_rate, c.duration_s)

    def _phase_tag(self, elapsed_s: float) -> str:
        c = self.config
        return workload_phase(self.workload, elapsed_s, c.warmup_s,
                              c.burst_start_s, c.burst_duration_s)

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
            req = Request(
                request_id=self._request_id,
                prompt=p["prompt"],
                max_new_tokens=int(p["max_new_tokens"]),
                arrival_t=time.monotonic(),
                phase=self._phase_tag(elapsed),
                seed_idx=idx,
            )
            self.queue.put(req)
            self._request_id += 1

        log.info("LoadGenerator(%s) finished: %d requests", self.workload, self._request_id)

    def stop(self):
        self._stop_event.set()

    @property
    def n_generated(self) -> int:
        return self._request_id
