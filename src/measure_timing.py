"""
measure_timing.py
Characterise the real software-visible timing on this GPU:
  (a) NVML power / clock update rate (how often the sensor actually changes)
  (b) nvidia-smi clock-lock command latency
  (c) Observed actuation delay (time from clock command to clock reading under load)

Run from an Administrator PowerShell:
  python src/measure_timing.py

Outputs:
  results/timing_characterisation.json  (structured measurements)
  results/timing_power_trace.csv        (raw NVML readings for update-rate analysis)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from gpu_clock import (
    measure_clock_command_latency_ms,
    measure_lock_command_latency_ms,
    reset_graphics_clock,
    set_graphics_clock,
    supported_graphics_clocks,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("measure_timing")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# (a) NVML update-rate measurement
# ---------------------------------------------------------------------------

def measure_nvml_update_rate(n_samples: int = 300,
                              poll_s: float = 0.033) -> dict:
    """
    Sample NVML power and clock at poll_s intervals.
    Count how often the reading actually changes to estimate the
    effective sensor update frequency.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception as e:
        logger.error("NVML unavailable: %s", e)
        return {}

    powers: List[float] = []
    clocks: List[int]   = []
    timestamps: List[float] = []

    logger.info("Sampling NVML at ~%.0f ms intervals (%d samples) …",
                poll_s * 1000, n_samples)

    for _ in range(n_samples):
        t0 = time.monotonic()
        p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        c = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
        timestamps.append(t0)
        powers.append(p)
        clocks.append(c)
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, poll_s - elapsed))

    pynvml.nvmlShutdown()

    # Save raw trace
    trace_path = os.path.join(RESULTS_DIR, "timing_power_trace.csv")
    with open(trace_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "power_w", "graphics_mhz"])
        for t, p, c in zip(timestamps, powers, clocks):
            w.writerow([t, p, c])

    # Count value changes
    power_changes = sum(1 for a, b in zip(powers, powers[1:]) if a != b)
    clock_changes = sum(1 for a, b in zip(clocks, clocks[1:]) if a != b)
    total_t = timestamps[-1] - timestamps[0]

    power_rate = power_changes / total_t if total_t > 0 else 0
    clock_rate = clock_changes / total_t if total_t > 0 else 0

    import statistics
    inter = [(b - a) * 1000 for a, b in zip(timestamps, timestamps[1:])]

    result = {
        "n_samples":              n_samples,
        "total_duration_s":       total_t,
        "poll_interval_ms":       poll_s * 1000,
        "mean_inter_sample_ms":   statistics.mean(inter) if inter else 0,
        "std_inter_sample_ms":    statistics.stdev(inter) if len(inter) > 1 else 0,
        "power_value_changes":    power_changes,
        "power_update_rate_hz":   power_rate,
        "power_effective_period_ms": 1000 / power_rate if power_rate > 0 else float("inf"),
        "clock_value_changes":    clock_changes,
        "clock_update_rate_hz":   clock_rate,
        "raw_trace_path":         trace_path,
    }
    logger.info(
        "NVML update rate: power=%.1f Hz (%.0f ms period)  clock=%.1f Hz",
        power_rate, result["power_effective_period_ms"], clock_rate,
    )
    return result


# ---------------------------------------------------------------------------
# (b) Clock-command latency
# ---------------------------------------------------------------------------

def measure_command_latency(supported: List[int]) -> dict:
    """Measure nvidia-smi reset and lock command wall-clock latency."""
    logger.info("Measuring clock-command latency (20 trials each) …")
    reset_lat = measure_clock_command_latency_ms(n_trials=20)
    # Use a mid-range clock for the lock measurement
    mid_mhz = supported[len(supported) // 2] if supported else 1800
    lock_lat = measure_lock_command_latency_ms(mid_mhz, n_trials=20)
    return {
        "reset_command_latency_ms": reset_lat,
        "lock_command_latency_ms":  lock_lat,
        "test_clock_mhz":           mid_mhz,
    }


# ---------------------------------------------------------------------------
# (c) Actuation delay under load
# ---------------------------------------------------------------------------

def measure_actuation_delay(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
                             test_clocks: List[int] = None) -> dict:
    """
    Start a CUDA workload, issue a clock-lock command, and measure how
    long until the observed NVML clock reflects the new setting.

    Returns the mean actuation delay in milliseconds.
    """
    try:
        import torch
        import pynvml
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        logger.warning("Actuation-delay test skipped (missing lib): %s", e)
        return {}

    if test_clocks is None or len(test_clocks) < 2:
        logger.warning("Need ≥2 test clocks; skipping actuation delay test.")
        return {}

    low_mhz  = test_clocks[0]
    high_mhz = test_clocks[-1]

    logger.info("Loading model for actuation delay test …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True
    )
    model.eval()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    delays_ms: List[float] = []

    # Start background CUDA workload
    stop_event = threading.Event()
    def _cuda_load():
        a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        while not stop_event.is_set():
            _ = torch.matmul(a, b)
    t = threading.Thread(target=_cuda_load, daemon=True)
    t.start()
    time.sleep(2.0)  # let clock settle at high_mhz

    for _ in range(5):
        # Step 1: lock to low_mhz
        set_graphics_clock(low_mhz)
        cmd_t = time.monotonic()
        # Poll until observed clock drops
        timeout = 5.0
        while time.monotonic() - cmd_t < timeout:
            obs = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            if obs <= low_mhz + 50:  # allow 50 MHz tolerance
                break
            time.sleep(0.02)
        act_delay = (time.monotonic() - cmd_t) * 1000.0
        delays_ms.append(act_delay)

        # Step 2: unlock and let clock recover
        reset_graphics_clock()
        time.sleep(2.0)

    stop_event.set()
    t.join(timeout=5.0)
    pynvml.nvmlShutdown()
    del model
    torch.cuda.empty_cache()

    import statistics
    result = {
        "test_clocks": [low_mhz, high_mhz],
        "n_trials": len(delays_ms),
        "mean_actuation_delay_ms": statistics.mean(delays_ms) if delays_ms else float("nan"),
        "std_actuation_delay_ms":  statistics.stdev(delays_ms) if len(delays_ms) > 1 else 0.0,
        "min_actuation_delay_ms":  min(delays_ms) if delays_ms else float("nan"),
        "max_actuation_delay_ms":  max(delays_ms) if delays_ms else float("nan"),
    }
    logger.info(
        "Actuation delay: %.1f ± %.1f ms (min=%.1f max=%.1f)",
        result["mean_actuation_delay_ms"], result["std_actuation_delay_ms"],
        result["min_actuation_delay_ms"], result["max_actuation_delay_ms"],
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="GPU timing characterisation")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--skip-actuation", action="store_true",
                   help="Skip the actuation-delay test (faster)")
    args = p.parse_args()

    results = {}

    logger.info("=== Step A: NVML sensor update rate ===")
    results["nvml_update_rate"] = measure_nvml_update_rate()

    supported = supported_graphics_clocks()
    logger.info("Supported clocks: %s … %d MHz", supported[:3], supported[-1])

    logger.info("=== Step B: Clock command latency ===")
    results["command_latency"] = measure_command_latency(supported)

    if not args.skip_actuation:
        logger.info("=== Step C: Actuation delay under load ===")
        test_clocks = [supported[0], supported[-1]]
        results["actuation_delay"] = measure_actuation_delay(
            model_id=args.model, test_clocks=test_clocks
        )

    out_path = os.path.join(RESULTS_DIR, "timing_characterisation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Timing characterisation saved → %s", out_path)

    # Pretty-print summary
    print("\n" + "=" * 60)
    print("TIMING CHARACTERISATION SUMMARY")
    print("=" * 60)
    ur = results.get("nvml_update_rate", {})
    if ur:
        print(f"NVML power update period : {ur.get('power_effective_period_ms', '?'):.0f} ms")
        print(f"NVML poll interval       : {ur.get('mean_inter_sample_ms', '?'):.1f} ms")
    cl = results.get("command_latency", {})
    if cl:
        rl = cl.get("reset_command_latency_ms", {})
        ll = cl.get("lock_command_latency_ms", {})
        if rl:
            print(f"nvidia-smi reset latency : {rl.get('mean_ms', '?'):.1f} ± {rl.get('std_ms', '?'):.1f} ms")
        if ll:
            print(f"nvidia-smi lock latency  : {ll.get('mean_ms', '?'):.1f} ± {ll.get('std_ms', '?'):.1f} ms")
    ad = results.get("actuation_delay", {})
    if ad:
        print(f"Clock actuation delay    : {ad.get('mean_actuation_delay_ms', '?'):.1f} ± {ad.get('std_actuation_delay_ms', '?'):.1f} ms")


if __name__ == "__main__":
    main()
