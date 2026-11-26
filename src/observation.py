"""
observation.py
What the controller is allowed to see, and when.

The stale-telemetry experiment only means something if *every* signal the
controller acts on is delayed, not just the NVML power/clock sample.  The
controllers decide from queue length, rolling P95, error-budget burn and
arrival statistics, so those are snapshotted here and pushed through their
own DelayBuffer alongside the GPU samples.

This module has no torch / NVML dependency so it can be unit-tested anywhere.

Timeline
--------
All phase timing (warm-up, burst, recovery) is measured from one origin:
the end of warm-up.  Both the load generator and the control loop call
``phase_at(elapsed, cfg)`` so the burst the generator produces and the
burst the analysis assumes are the same 20 seconds.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from controllers.base import ControllerState


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def measurement_length_s(cfg) -> float:
    """Seconds of measurement after warm-up."""
    return cfg.duration_s - cfg.warmup_s


def phase_at(elapsed: Optional[float], cfg) -> str:
    """
    Experiment phase at ``elapsed`` seconds after the end of warm-up.
    ``None`` or a negative value means warm-up is still running.
    """
    if elapsed is None or elapsed < 0:
        return "warmup"
    if cfg.workload == "sharp_burst":
        if cfg.burst_start_s <= elapsed < cfg.burst_start_s + cfg.burst_duration_s:
            return "burst"
        if elapsed >= cfg.burst_start_s + cfg.burst_duration_s:
            return "recovery"
        return "steady"
    if cfg.workload == "ramp":
        return "ramp"
    return "steady"


def ramp_end_rate(cfg) -> float:
    """Arrival rate at the end of a ramp (explicit, or base x multiplier)."""
    if getattr(cfg, "ramp_end_rate", None) is not None:
        return cfg.ramp_end_rate
    return cfg.base_arrival_rate * getattr(cfg, "ramp_end_multiplier", 3.0)


def arrival_rate_at(elapsed: Optional[float], cfg) -> float:
    """Target arrival rate (req/s) at ``elapsed`` seconds after warm-up."""
    base = cfg.base_arrival_rate
    if elapsed is None or elapsed < 0:
        return base
    if cfg.workload == "sharp_burst":
        return base * cfg.burst_multiplier if phase_at(elapsed, cfg) == "burst" else base
    if cfg.workload == "ramp":
        frac = min(elapsed / max(measurement_length_s(cfg), 1e-9), 1.0)
        return base + frac * (ramp_end_rate(cfg) - base)
    return base


def validate_timeline(cfg) -> None:
    """Raise ValueError if the burst does not fit, with recovery time, in the run."""
    m = measurement_length_s(cfg)
    if m <= 0:
        raise ValueError(f"duration_s ({cfg.duration_s}) must exceed warmup_s ({cfg.warmup_s})")
    if cfg.workload != "sharp_burst":
        return
    burst_end = cfg.burst_start_s + cfg.burst_duration_s
    if burst_end > m:
        raise ValueError(
            f"Burst {cfg.burst_start_s:.0f}-{burst_end:.0f} s does not fit in the "
            f"{m:.0f} s measurement window (duration {cfg.duration_s:.0f} s - "
            f"warm-up {cfg.warmup_s:.0f} s)."
        )
    if m - burst_end < 10.0:
        raise ValueError(
            f"Only {m - burst_end:.0f} s after the burst; need at least 10 s to "
            "observe recovery. Increase --duration."
        )


# ---------------------------------------------------------------------------
# Starting clock
# ---------------------------------------------------------------------------

def starting_index(controller, clocks) -> int:
    """
    Clock index the GPU is locked to before warm-up.
    Static controllers start at their target; dynamic ones in the middle.
    """
    target = getattr(controller, "_target_idx", None)
    if target is not None:
        return int(target)
    return len(clocks) // 2


def clock_tracking_pct(samples, commands, settle_s: float = 1.25,
                       tol_mhz: float = 50.0, min_util_pct: float = 30.0) -> float:
    """
    Percent of loaded telemetry samples whose graphics clock is within
    ``tol_mhz`` of the most recently commanded clock.

    ``commands`` is a time-ordered list of (t, mhz).  Samples within
    ``settle_s`` of a command (measured settling: ~0.7 s clock, ~1.2 s
    power) and idle samples (util < ``min_util_pct``, where the GPU may
    park below the lock) are skipped.  Returns NaN if nothing qualifies.
    """
    if not commands:
        return float("nan")
    times = [c[0] for c in commands]
    ok = n = 0
    j = -1
    for s in samples:
        while j + 1 < len(times) and times[j + 1] <= s.t:
            j += 1
        if j < 0 or s.t - times[j] < settle_s or s.util_pct < min_util_pct:
            continue
        n += 1
        if abs(s.graphics_mhz - commands[j][1]) <= tol_mhz:
            ok += 1
    return 100.0 * ok / n if n else float("nan")


# ---------------------------------------------------------------------------
# Control-signal snapshot
# ---------------------------------------------------------------------------

@dataclass
class ControlObservation:
    """Non-GPU signals the controller reads, captured at time ``t``."""
    t: float
    queue_len: int
    arrival_rate: float
    arrival_cv: float
    p95_latency_s: float
    p99_latency_s: float
    error_budget_burn: float
    miss_rate: float


def take_observation(rq, gen, slo) -> ControlObservation:
    """Snapshot the live queue / workload / SLO state."""
    return ControlObservation(
        t=time.monotonic(),
        queue_len=rq.qsize,
        arrival_rate=gen.recent_arrival_rate(),
        arrival_cv=gen.arrival_cv(),
        p95_latency_s=slo.p95(),
        p99_latency_s=slo.p99(),
        error_budget_burn=slo.burn_rate(),
        miss_rate=slo.miss_rate(),
    )


class ObservationSampler:
    """
    Daemon thread that pushes a ControlObservation into a DelayBuffer every
    ``poll_interval_s`` seconds, the same cadence as the GPU sampler.
    """

    def __init__(self, snapshot_fn: Callable[[], ControlObservation],
                 delay_buffer, poll_interval_s: float = 0.1):
        self._snapshot_fn = snapshot_fn
        self.delay_buffer = delay_buffer
        self.poll_interval_s = poll_interval_s
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="obs-sampler", daemon=True)

    def start(self) -> None:
        self.delay_buffer.push(self._snapshot_fn())
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.poll_interval_s):
            self.delay_buffer.push(self._snapshot_fn())


# ---------------------------------------------------------------------------
# Delayed controller state
# ---------------------------------------------------------------------------

def draw_delay_s(delay_s: float, jitter_s: float, rng: random.Random) -> float:
    """One delay draw shared by both buffers in a control interval."""
    if jitter_s > 0:
        return max(0.0, delay_s + rng.uniform(-jitter_s, jitter_s))
    return delay_s


def delayed_state(gpu_buf, obs_buf, delay_s: float, slo_target_s: float
                  ) -> Tuple[Optional[ControllerState], Optional[ControlObservation], float]:
    """
    Build the ControllerState from samples at least ``delay_s`` old.

    Returns (state, observation, age_s).  ``age_s`` is the age of the older
    of the two samples, which is what the controller is told.  ``state`` is
    None until both buffers have data.
    """
    telem, gpu_age = gpu_buf.get(delay_s=delay_s)
    obs, obs_age = obs_buf.get(delay_s=delay_s)
    if telem is None or obs is None:
        return None, None, 0.0
    age_s = max(gpu_age, obs_age)
    state = ControllerState(
        t=time.monotonic(),
        power_w=telem.power_w,
        graphics_mhz=telem.graphics_mhz,
        sm_mhz=telem.sm_mhz,
        util_pct=telem.util_pct,
        temp_c=telem.temp_c,
        telemetry_age_ms=age_s * 1000.0,
        queue_len=obs.queue_len,
        arrival_rate=obs.arrival_rate,
        arrival_cv=obs.arrival_cv,
        p95_latency_s=obs.p95_latency_s,
        p99_latency_s=obs.p99_latency_s,
        slo_target_s=slo_target_s,
        error_budget_burn=obs.error_budget_burn,
        miss_rate=obs.miss_rate,
        phase=telem.phase,
    )
    return state, obs, age_s
