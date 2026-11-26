"""
telemetry.py
Background GPU telemetry sampler using NVML (nvidia-ml-py).

Design:
  • TelemetrySampler runs a daemon thread that polls NVML every
    poll_interval_s seconds and pushes GpuSample objects into a
    DelayBuffer (and also into a flat list for post-run analysis).
  • Call sampler.start() before the experiment and sampler.stop() (or
    rely on the daemon thread to exit) when done.
  • Use sampler.delay_buffer to let the controller retrieve stale views.
  • Use sampler.samples for trapezoidal energy integration.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class GpuSample:
    """A single point-in-time GPU observation."""

    t: float
    """Monotonic timestamp (time.monotonic()) when the sample was taken."""

    power_w: float
    """Instantaneous GPU power draw in Watts."""

    graphics_mhz: int
    """Reported graphics clock in MHz."""

    sm_mhz: int
    """Reported SM (shader) clock in MHz."""

    util_pct: int
    """GPU utilisation in percent (0–100)."""

    temp_c: int
    """GPU temperature in degrees Celsius."""

    phase: str = "steady"
    """Experiment phase tag ('warmup', 'steady', 'ramp', 'burst')."""


# ---------------------------------------------------------------------------
# NVML helpers
# ---------------------------------------------------------------------------

def _init_nvml():
    if not _NVML_AVAILABLE:
        raise RuntimeError(
            "pynvml is not installed.  Run: pip install nvidia-ml-py"
        )
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def _sample_once(handle, phase: str = "steady") -> GpuSample:
    """Take one sample from NVML."""
    return GpuSample(
        t=time.monotonic(),
        power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
        graphics_mhz=pynvml.nvmlDeviceGetClockInfo(
            handle, pynvml.NVML_CLOCK_GRAPHICS
        ),
        sm_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        util_pct=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
        temp_c=pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        ),
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Sampler thread
# ---------------------------------------------------------------------------

class TelemetrySampler:
    """
    Background daemon thread that continuously samples the GPU and pushes
    results into a DelayBuffer and a flat list.

    Parameters
    ----------
    delay_buffer : DelayBuffer
        Shared buffer used by the controller for stale-telemetry injection.
    poll_interval_s : float
        Interval between NVML queries (default 0.10 s = 100 ms).
    """

    def __init__(self, delay_buffer, poll_interval_s: float = 0.1):
        from delay_buffer import DelayBuffer  # local import to avoid cycles
        self.delay_buffer = delay_buffer
        self.poll_interval_s = poll_interval_s

        self.samples: List[GpuSample] = []
        """All samples collected during the run (for energy integration)."""

        self._stop_event = threading.Event()
        self._phase = "warmup"
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._run, name="telemetry-sampler", daemon=True
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the background sampling thread."""
        self._thread.start()
        logger.info("TelemetrySampler started (poll=%.0f ms)",
                    self.poll_interval_s * 1000)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        logger.info("TelemetrySampler stopped (%d samples collected)",
                    len(self.samples))

    def set_phase(self, phase: str) -> None:
        """Update the experiment phase tag attached to future samples."""
        with self._lock:
            self._phase = phase

    # ------------------------------------------------------------------ #
    # Thread body                                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        try:
            handle = _init_nvml()
        except RuntimeError as exc:
            logger.error("NVML init failed: %s", exc)
            return

        while not self._stop_event.is_set():
            t_loop = time.monotonic()
            with self._lock:
                phase = self._phase
            try:
                s = _sample_once(handle, phase)
            except Exception as exc:
                logger.warning("NVML sample error: %s", exc)
                time.sleep(self.poll_interval_s)
                continue

            self.samples.append(s)
            self.delay_buffer.push(s)

            # Sleep for the remainder of the poll interval
            elapsed = time.monotonic() - t_loop
            sleep_time = max(0.0, self.poll_interval_s - elapsed)
            time.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Convenience: measure effective NVML update rate
# ---------------------------------------------------------------------------

def measure_power_update_rate(n_samples: int = 200,
                              poll_s: float = 0.05) -> dict:
    """
    Measure how often NVML's power reading actually changes.
    Returns a dict with 'update_rate_hz' and 'effective_period_ms'.
    """
    import statistics
    handle = _init_nvml()
    readings: List[float] = []
    timestamps: List[float] = []

    for _ in range(n_samples):
        readings.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        timestamps.append(time.monotonic())
        time.sleep(poll_s)

    pynvml.nvmlShutdown()

    # Count transitions
    transitions = sum(
        1 for a, b in zip(readings, readings[1:]) if a != b
    )
    total_time = timestamps[-1] - timestamps[0]
    update_rate = transitions / total_time if total_time > 0 else 0.0

    inter_sample = [
        (timestamps[i + 1] - timestamps[i]) * 1000.0
        for i in range(len(timestamps) - 1)
    ]

    return {
        "n_samples": n_samples,
        "total_time_s": total_time,
        "value_transitions": transitions,
        "update_rate_hz": update_rate,
        "effective_period_ms": (1000.0 / update_rate) if update_rate > 0 else float("inf"),
        "mean_inter_sample_ms": statistics.mean(inter_sample),
        "std_inter_sample_ms": statistics.stdev(inter_sample) if len(inter_sample) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    sys.path.insert(0, ".")
    from delay_buffer import DelayBuffer

    buf = DelayBuffer()
    sampler = TelemetrySampler(buf, poll_interval_s=0.1)
    sampler.start()
    print("Sampling for 3 seconds …")
    time.sleep(3)
    sampler.stop()
    print(f"Collected {len(sampler.samples)} samples")
    if sampler.samples:
        s = sampler.samples[-1]
        print(
            f"Latest: power={s.power_w:.1f} W  "
            f"clock={s.graphics_mhz} MHz  "
            f"util={s.util_pct}%  "
            f"temp={s.temp_c}°C"
        )
    stats = measure_power_update_rate(n_samples=50, poll_s=0.02)
    print(
        f"NVML power update rate: {stats['update_rate_hz']:.1f} Hz "
        f"(effective period {stats['effective_period_ms']:.1f} ms)"
    )
