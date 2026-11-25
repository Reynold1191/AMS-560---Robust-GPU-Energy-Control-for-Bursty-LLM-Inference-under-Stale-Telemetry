"""
load_generator.py
Background thread that creates Request objects and pushes them to the
RequestQueue at a configurable arrival rate.

Workload profiles
-----------------
  steady     – Poisson arrivals at a constant rate.
  ramp       – Linearly increasing arrival rate from base → base x ramp_end_multiplier.
  sharp_burst– Sudden spike (burst_multiplier × base rate) for a fixed window,
               followed by a return to the base rate.
  trace      – Replay a pre-recorded inter-arrival sequence.
"""

from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from pathlib import Path
from typing import List, Optional

from request_queue import Request, RequestQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt loader
# ---------------------------------------------------------------------------

def load_prompts(path: str) -> List[dict]:
    """
    Load prompts from a JSONL file.
    Each line: {"prompt": "...", "max_new_tokens": N}
    """
    records = []
    p = Path(path)
    if not p.exists():
        logger.warning("Prompts file not found: %s – using fallback prompts", path)
        return _FALLBACK_PROMPTS

    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return _FALLBACK_PROMPTS
    logger.info("Loaded %d prompts from %s", len(records), path)
    return records


_FALLBACK_PROMPTS = [
    {"prompt": "Explain the concept of energy-efficient computing in two sentences.", "max_new_tokens": 64},
    {"prompt": "What is GPU frequency scaling and why does it matter for AI workloads?", "max_new_tokens": 96},
    {"prompt": "Summarise the main trade-off between GPU power consumption and inference latency.", "max_new_tokens": 80},
    {"prompt": "What is a service-level objective in the context of cloud serving systems?", "max_new_tokens": 72},
    {"prompt": "Describe the impact of bursty traffic on a GPU inference server.", "max_new_tokens": 96},
]


# ---------------------------------------------------------------------------
# Arrival-rate schedule helpers
# ---------------------------------------------------------------------------

def _poisson_sleep(rate: float, rng: random.Random) -> float:
    """Return an exponentially distributed inter-arrival time."""
    if rate <= 0:
        return 1.0
    return rng.expovariate(rate)


def _current_rate(
    elapsed: float,
    workload: str,
    base_rate: float,
    burst_multiplier: float,
    burst_start: float,
    burst_end: float,
    ramp_end_rate: float,
    duration: float,
) -> float:
    """Return the instantaneous target arrival rate at time ``elapsed``."""
    if workload == "steady":
        return base_rate

    if workload == "ramp":
        frac = min(elapsed / duration, 1.0)
        return base_rate + frac * (ramp_end_rate - base_rate)

    if workload == "sharp_burst":
        if burst_start <= elapsed < burst_end:
            return base_rate * burst_multiplier
        return base_rate

    # Default
    return base_rate


# ---------------------------------------------------------------------------
# Load generator thread
# ---------------------------------------------------------------------------

class LoadGenerator:
    """
    Daemon thread that generates requests according to a workload profile.

    Parameters
    ----------
    request_queue  : Shared RequestQueue consumed by LLMWorker.
    config         : ExperimentConfig instance.
    prompts        : List of prompt dicts (loaded from prompts.jsonl).
    """

    def __init__(self, request_queue: RequestQueue, config, prompts: List[dict]):
        self.request_queue = request_queue
        self.config = config
        self.prompts = prompts

        self._stop_event = threading.Event()
        self._rng = random.Random(config.seed)
        self._request_counter = 0
        self._phase = "warmup"
        self._origin: Optional[float] = None
        self._lock = threading.Lock()

        # Inter-arrival times log (for computing arrival CV)
        self.inter_arrivals: List[float] = []

        self._thread = threading.Thread(
            target=self._run, name="load-generator", daemon=True
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._thread.start()
        logger.info("LoadGenerator started (workload=%s, base_rate=%.2f rps)",
                    self.config.workload, self.config.base_arrival_rate)

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info("LoadGenerator stopped (%d requests created)",
                    self._request_counter)

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    # ------------------------------------------------------------------ #
    # Thread body                                                           #
    # ------------------------------------------------------------------ #

    def set_origin(self, t: float) -> None:
        """
        Set the shared experiment origin (end of warm-up, monotonic time).
        From then on the burst schedule and phase tags follow
        observation.phase_at(), the same clock the control loop uses.
        """
        with self._lock:
            self._origin = t

    def _elapsed(self, now: float) -> Optional[float]:
        with self._lock:
            origin = self._origin
        return None if origin is None else now - origin

    def _max_rate(self) -> float:
        from observation import ramp_end_rate
        cfg = self.config
        rates = [cfg.base_arrival_rate, cfg.base_arrival_rate * cfg.burst_multiplier]
        if cfg.workload == "ramp":
            rates.append(ramp_end_rate(cfg))
        return max(rates)

    def _next_arrival(self, prev: float) -> float:
        """
        Next scheduled arrival (monotonic time) after ``prev``: a
        non-homogeneous Poisson process by thinning. Candidates come at the
        peak rate and are kept with probability rate(t) / peak, so step
        changes (burst) and ramps are exact and depend only on the seed.
        """
        from observation import arrival_rate_at
        lam_max = self._max_rate()
        t = prev
        while True:
            t += self._rng.expovariate(lam_max)
            rate = arrival_rate_at(self._elapsed(t), self.config)
            if self._rng.random() * lam_max <= rate:
                return t

    def _run(self) -> None:
        """
        Open-loop arrivals from a pre-drawn schedule.

        attempt_7 drew each gap after the previous request was queued and
        slept it off; while the worker held the GIL (HF decode is CPU-bound)
        those sleeps overran, so a 9.4 req/s burst delivered ~7.4 req/s and
        steady runs ~9 % fewer requests than planned. Now arrival times are
        fixed in advance; a late wake-up enqueues every overdue request at
        once, each stamped with its SCHEDULED arrival time, so queueing
        latency is still counted from when the request should have arrived.
        """
        from observation import measurement_length_s, phase_at

        cfg = self.config
        prev_sched = time.monotonic()
        next_t = self._next_arrival(prev_sched)

        while not self._stop_event.is_set():
            now = time.monotonic()
            if next_t > now:
                self._stop_event.wait(min(next_t - now, 0.05))
                continue

            elapsed = self._elapsed(next_t)
            if elapsed is not None and elapsed >= measurement_length_s(cfg):
                break
            phase = phase_at(elapsed, cfg) if elapsed is not None else "warmup"

            self.inter_arrivals.append(next_t - prev_sched)
            prev_sched = next_t

            p = self.prompts[self._request_counter % len(self.prompts)]
            req = Request(
                request_id=self._request_counter,
                prompt=p["prompt"],
                max_new_tokens=int(p.get("max_new_tokens", 64)),
                arrival_t=next_t,
                phase=phase,
            )
            self._request_counter += 1
            self.request_queue.put(req)
            next_t = self._next_arrival(next_t)

    # ------------------------------------------------------------------ #
    # Diagnostics                                                           #
    # ------------------------------------------------------------------ #

    def arrival_cv(self, last_n: int = 30) -> float:
        """
        Coefficient of variation of recent inter-arrival times.
        High CV indicates bursty traffic.
        """
        import statistics
        iat = self.inter_arrivals[-last_n:] if self.inter_arrivals else []
        if len(iat) < 2:
            return 0.0
        mu = statistics.mean(iat)
        if mu <= 0:
            return 0.0
        return statistics.stdev(iat) / mu

    def recent_arrival_rate(self, window_s: float = 10.0) -> float:
        """
        Estimate requests-per-second in the most recent window.
        Uses the last N inter-arrival times whose sum ≤ window_s.
        """
        iat = self.inter_arrivals
        if not iat:
            return 0.0
        acc = 0.0
        count = 0
        for v in reversed(iat):
            acc += v
            count += 1
            if acc >= window_s:
                break
        return count / acc if acc > 0 else 0.0
