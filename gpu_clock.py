"""
gpu_clock.py
GPU clock enumeration and control via nvidia-smi.

All clock-locking operations require an elevated (Administrator) terminal.
Always call reset_graphics_clock() in a finally block to avoid leaving
the GPU in a locked state after the experiment ends.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Clock enumeration
# ---------------------------------------------------------------------------

def supported_graphics_clocks() -> List[int]:
    """
    Query nvidia-smi for all supported graphics clock values.

    Returns a sorted list of MHz values (lowest → highest).
    Raises RuntimeError if nvidia-smi is unavailable.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-q", "-d", "SUPPORTED_CLOCKS"],
            text=True,
            errors="replace",
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "nvidia-smi is unavailable. "
            "Ensure NVIDIA drivers are installed and PATH is correct."
        ) from exc

    vals: List[int] = []
    for line in out.splitlines():
        m = re.search(r"Graphics\s*:\s*(\d+)\s*MHz", line)
        if m:
            vals.append(int(m.group(1)))

    if not vals:
        raise RuntimeError(
            "No Graphics clock values found in nvidia-smi output. "
            "Check that the GPU supports DVFS."
        )
    return sorted(set(vals))


def select_experiment_clocks(all_clocks: List[int], n: int = 7) -> List[int]:
    """
    Choose at most n well-spaced clock values from the full supported list.
    Always includes the minimum and maximum.  Returns sorted ascending.
    """
    if len(all_clocks) <= n:
        return list(all_clocks)
    step = (len(all_clocks) - 1) / (n - 1)
    indices = sorted({round(i * step) for i in range(n)})
    return [all_clocks[i] for i in indices]


# ---------------------------------------------------------------------------
# Clock control
# ---------------------------------------------------------------------------

def set_graphics_clock(mhz: int) -> int:
    """
    Lock the GPU graphics clock to exactly ``mhz`` MHz.

    Must be run from an Administrator terminal.
    Returns the nvidia-smi exit code (0 = success).
    """
    logger.info("Locking graphics clock → %d MHz", mhz)
    t0 = time.monotonic()
    result = subprocess.run(
        ["nvidia-smi", "-lgc", f"{mhz},{mhz}"],
        capture_output=True,
        text=True,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    if result.returncode != 0:
        logger.warning(
            "nvidia-smi -lgc returned %d (%.1f ms): %s",
            result.returncode, elapsed_ms, result.stderr.strip()
        )
    else:
        logger.debug("Clock command accepted (%.1f ms)", elapsed_ms)
    return result.returncode


def reset_graphics_clock() -> None:
    """Reset GPU graphics clock to driver default (nvidia-smi -rgc)."""
    logger.info("Resetting graphics clock to driver default")
    subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Timing characterisation
# ---------------------------------------------------------------------------

def measure_clock_command_latency_ms(n_trials: int = 20) -> dict:
    """
    Measure the wall-clock time for a reset command (nvidia-smi -rgc).
    Used by measure_timing.py to characterise actuation overhead.

    Returns a dict with keys: mean_ms, std_ms, min_ms, max_ms.
    """
    import statistics
    times: List[float] = []
    for _ in range(n_trials):
        t0 = time.monotonic()
        subprocess.run(["nvidia-smi", "-rgc"], capture_output=True, text=True)
        times.append((time.monotonic() - t0) * 1000.0)
    return {
        "mean_ms": statistics.mean(times),
        "std_ms":  statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms":  min(times),
        "max_ms":  max(times),
        "n":       n_trials,
    }


def measure_lock_command_latency_ms(mhz: int, n_trials: int = 20) -> dict:
    """
    Measure the wall-clock time for a lock command (nvidia-smi -lgc).
    """
    import statistics
    times: List[float] = []
    for _ in range(n_trials):
        t0 = time.monotonic()
        subprocess.run(
            ["nvidia-smi", "-lgc", f"{mhz},{mhz}"],
            capture_output=True, text=True
        )
        times.append((time.monotonic() - t0) * 1000.0)
    # reset after measurement
    reset_graphics_clock()
    return {
        "mean_ms": statistics.mean(times),
        "std_ms":  statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms":  min(times),
        "max_ms":  max(times),
        "n":       n_trials,
    }


# ---------------------------------------------------------------------------
# Quick self-test (run directly: python gpu_clock.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clocks = supported_graphics_clocks()
    print(f"Supported graphics clocks ({len(clocks)} values): {clocks[:5]} … {clocks[-1]} MHz")
    chosen = select_experiment_clocks(clocks, n=7)
    print(f"Experiment subset ({len(chosen)} clocks): {chosen}")
    timing = measure_clock_command_latency_ms(n_trials=5)
    print(f"Reset command latency: {timing['mean_ms']:.1f} ± {timing['std_ms']:.1f} ms")
