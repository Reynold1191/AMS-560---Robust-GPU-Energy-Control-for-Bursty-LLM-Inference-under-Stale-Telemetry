"""
check_gpu_timing.py
Week 1 feedback checks for the RTX 5060 Laptop GPU.

  1. Does the GPU support reliable clock control and power telemetry?
  2. Measure the real telemetry and clock-change delays, derive injected
     delay levels from them, and compare with the current DELAY_VALUES_MS
     and with results/timing_characterisation.json.
     Four levels from the measured timescales: 0, one NVML refresh period,
     full settling after a clock change (clock and power), and twice that.
     The proposal levels [0, 500, 1250, 2500] ms are kept unless the
     measurements move a level by more than 20% (or 100 ms).

Clocks that are capped (power/thermal limit) or fail are reported but left out
of the usable experiment clocks.

Run from an Administrator prompt (clock locking needs admin):
    python check_gpu_timing.py

Without admin rights only the read-only telemetry checks run.
The GPU clock is reset to the driver default (nvidia-smi -rgc) on exit,
including on Ctrl+C or an error.

Outputs (results/gpu_check/):
    gpu_check.json      all measurements, verdicts and derived delay levels
    step_traces.csv     raw clock/power traces around each clock change
    fig_gpu_check.png   clock tracking, power vs clock, one step response
"""

from __future__ import annotations

import atexit
import csv
import ctypes
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pynvml

from config import DELAY_VALUES_MS, VERIFIED_CLOCKS_MHZ
from gpu_clock import select_experiment_clocks, supported_graphics_clocks

OUT_DIR = os.path.join(HERE, "results", "gpu_check")
SAVED_TIMING = os.path.join(HERE, "results", "timing_characterisation.json")
POLL_S = 0.01
SETTLE_HOLD_MS = 80.0
PROPOSAL_LEVELS_MS = [0, 500, 1250, 2500]
MATERIAL_CHANGE = 0.20
SAME_REL_TOL = 0.25
SAME_ABS_TOL_MS = 50.0


# ---------------------------------------------------------------------------
# GPU access and restore
# ---------------------------------------------------------------------------

def is_admin() -> bool:
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except AttributeError:
        return os.geteuid() == 0


def nvsmi(*args: str) -> tuple[int, float]:
    t0 = time.perf_counter()
    rc = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True).returncode
    return rc, (time.perf_counter() - t0) * 1000.0


_touched_clocks = False


def restore_gpu() -> None:
    global _touched_clocks
    if _touched_clocks:
        nvsmi("-rgc")
        _touched_clocks = False
        print("GPU clock reset to driver default (nvidia-smi -rgc).")


def lock_clock(mhz: int) -> tuple[int, float]:
    global _touched_clocks
    _touched_clocks = True
    return nvsmi("-lgc", f"{mhz},{mhz}")


class Gpu:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)

    def name(self) -> str:
        n = pynvml.nvmlDeviceGetName(self.h)
        return n.decode() if isinstance(n, bytes) else n

    def read(self) -> dict:
        return {
            "t": time.perf_counter(),
            "power_w": pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
            "clock_mhz": pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_GRAPHICS),
            "util_pct": pynvml.nvmlDeviceGetUtilizationRates(self.h).gpu,
            "temp_c": pynvml.nvmlDeviceGetTemperature(self.h, pynvml.NVML_TEMPERATURE_GPU),
        }

    def trace(self, seconds: float, t_ref: float | None = None) -> list[dict]:
        out, end = [], time.perf_counter() + seconds
        while time.perf_counter() < end:
            s = self.read()
            if t_ref is not None:
                s["t_rel_ms"] = (s["t"] - t_ref) * 1000.0
            out.append(s)
            time.sleep(POLL_S)
        return out


class CudaLoad:
    """Keeps the GPU busy with fp16 matmuls so clocks and power respond."""

    def __init__(self, size: int = 4096):
        import torch
        self.torch = torch
        self.a = torch.randn((size, size), device="cuda", dtype=torch.float16)
        self.b = torch.randn((size, size), device="cuda", dtype=torch.float16)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        self.torch.cuda.set_device(self.a.device)
        while not self._stop.is_set():
            for _ in range(10):
                self.a @ self.b
            self.torch.cuda.synchronize()

    def __enter__(self):
        self._thread.start()
        time.sleep(1.5)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def med(values, key):
    return statistics.median(v[key] for v in values)


def update_period_ms(trace: list[dict], key: str) -> float:
    """Effective NVML update period: median time between changes of a reading.
    This is a property of the GPU/driver, unlike the software polling interval."""
    change_t = [b["t"] for a, b in zip(trace, trace[1:]) if a[key] != b[key]]
    gaps = [(b - a) * 1000.0 for a, b in zip(change_t, change_t[1:])]
    return statistics.median(gaps) if gaps else float("inf")


def settle_time_ms(trace: list[dict], in_band, hold_ms: float = SETTLE_HOLD_MS) -> float:
    """Start of the first stretch in which the reading stays in band for hold_ms."""
    start = None
    for s in trace:
        if not in_band(s):
            start = None
            continue
        if start is None:
            start = s["t_rel_ms"]
        if s["t_rel_ms"] - start >= hold_ms:
            return start
    return float("nan")


def telemetry_checks(gpu: Gpu) -> dict:
    idle = gpu.trace(3.0)
    with CudaLoad():
        load = gpu.trace(8.0)
    idle_w, load_w = med(idle, "power_w"), med(load, "power_w")
    return {
        "idle_power_w": idle_w,
        "load_power_w": load_w,
        "unlocked_load_clock_mhz": med(load, "clock_mhz"),
        "load_util_pct": med(load, "util_pct"),
        "software_poll_interval_ms": 1000.0 * statistics.mean(
            b["t"] - a["t"] for a, b in zip(load, load[1:])),
        "nvml_power_update_period_ms": update_period_ms(load, "power_w"),
        "nvml_clock_update_period_ms": update_period_ms(load, "clock_mhz"),
        "power_responds_to_load": load_w > idle_w + 5.0,
        # Under load every sample must be valid. At idle a laptop GPU can park
        # in a deep power state and report 0 MHz / 0 W; that is expected, so
        # idle zeros are counted and reported, not treated as a failure.
        "readings_valid": all(s["power_w"] > 0 and s["clock_mhz"] > 0 for s in load),
        "load_zero_readings": sum(1 for s in load if s["power_w"] <= 0 or s["clock_mhz"] <= 0),
        "idle_zero_readings": sum(1 for s in idle if s["power_w"] <= 0 or s["clock_mhz"] <= 0),
        "n_idle_samples": len(idle),
        "n_load_samples": len(load),
    }


def clock_tracking(gpu: Gpu, clocks: list[int], repeats: int = 2) -> list[dict]:
    rows = []
    with CudaLoad():
        for rep in range(repeats):
            for mhz in clocks:
                rc, _ = lock_clock(mhz)
                time.sleep(1.0)
                tr = gpu.trace(1.5)
                observed = med(tr, "clock_mhz")
                rows.append({
                    "target_mhz": mhz, "repeat": rep, "lock_rc": rc,
                    "observed_mhz": observed, "power_w": med(tr, "power_w"),
                    "error_mhz": observed - mhz,
                })
    return rows


def summarize_tracking(rows: list[dict], clocks: list[int]) -> dict:
    """
    PASS     observed clock within tolerance of the target
    CAPPED   target is above what the board sustains: the clock tops out
             above the highest passing clock while power sits at the board
             limit (2602/3090 MHz on the RTX 5060 Laptop at ~80 W). These
             are excluded but do not make clock control "unreliable".
    FAIL     lock rejected, wrong clock below the cap, or not repeatable.
    """
    per_clock, ok_all = [], True
    power_cap_w = max(x["power_w"] for x in rows) if rows else 0.0
    best_pass = None
    for mhz in clocks:
        r = [x for x in rows if x["target_mhz"] == mhz]
        obs = [x["observed_mhz"] for x in r]
        mean_obs = statistics.mean(obs)
        tol = max(60.0, 0.05 * mhz)
        if any(x["lock_rc"] != 0 for x in r):
            status = "FAIL (lock rejected)"
        elif abs(mean_obs - mhz) <= tol:
            status = "PASS"
        elif (best_pass is not None and mean_obs < mhz
              and mean_obs >= best_pass - tol
              and statistics.mean(x["power_w"] for x in r) >= 0.95 * power_cap_w):
            status = "CAPPED (power/thermal limit)"
        else:
            status = "FAIL"
        spread = max(obs) - min(obs)
        if status == "PASS" and spread > tol:
            status = "FAIL (not repeatable)"
        if status == "PASS":
            best_pass = mhz
        ok_all &= not status.startswith("FAIL")
        per_clock.append({"target_mhz": mhz, "mean_observed_mhz": mean_obs,
                          "repeat_spread_mhz": spread,
                          "mean_power_w": statistics.mean(x["power_w"] for x in r),
                          "status": status})
    usable = [c for c in per_clock if c["status"] == "PASS"]
    powers = [c["mean_power_w"] for c in usable]
    monotonic = len(powers) >= 2 and all(b >= a - 1.0 for a, b in zip(powers, powers[1:]))
    return {"per_clock": per_clock,
            "usable_clocks": [c["target_mhz"] for c in usable],
            "excluded_clocks": {str(c["target_mhz"]): c["status"]
                                for c in per_clock if c["status"] != "PASS"},
            "clock_control_reliable": ok_all and len(usable) >= 2,
            "power_increases_with_clock": monotonic,
            "power_range_w": [min(powers), max(powers)] if powers else None}


def step_response(gpu: Gpu, low: int, high: int, trials: int = 5) -> tuple[dict, list[dict]]:
    """Lock low<->high under load. Time until the clock reading, then the power
    reading, settles at the new setting and stays there for SETTLE_HOLD_MS."""
    results = {"down": [], "up": []}
    raw = []
    with CudaLoad():
        lock_clock(high)
        time.sleep(2.0)
        for trial in range(trials):
            for direction, target in (("down", low), ("up", high)):
                before = gpu.trace(0.5)
                p_before = med(before, "power_w")
                t_cmd = time.perf_counter()
                rc, cmd_ms = lock_clock(target)
                tr = gpu.trace(3.0, t_ref=t_cmd)
                p_after = med([s for s in tr if s["t_rel_ms"] >= 2000.0], "power_w")
                clock_tol = max(60.0, 0.05 * target)
                power_tol = max(1.5, 0.10 * abs(p_after - p_before))
                clock_hit = settle_time_ms(
                    tr, lambda s: abs(s["clock_mhz"] - target) <= clock_tol)
                power_hit = settle_time_ms(
                    tr, lambda s: abs(s["power_w"] - p_after) <= power_tol)
                results[direction].append({
                    "trial": trial, "lock_rc": rc, "command_ms": cmd_ms,
                    "clock_settle_ms": clock_hit, "power_settle_ms": power_hit,
                    "power_before_w": p_before, "power_after_w": p_after,
                })
                for s in tr:
                    raw.append({"direction": direction, "trial": trial,
                                "t_ms": s["t_rel_ms"], "clock_mhz": s["clock_mhz"],
                                "power_w": s["power_w"]})
    return results, raw


def command_latency(gpu: Gpu, mhz: int, trials: int = 10) -> dict:
    """Each trial is a real transition: default -> locked at mhz -> default, under load."""
    rows = []
    with CudaLoad():
        nvsmi("-rgc")
        time.sleep(1.0)
        for trial in range(trials):
            default_mhz = gpu.read()["clock_mhz"]
            lock_rc, lock_ms = lock_clock(mhz)
            time.sleep(0.6)
            locked_mhz = gpu.read()["clock_mhz"]
            reset_rc, reset_ms = nvsmi("-rgc")
            time.sleep(0.6)
            rows.append({"trial": trial, "lock_rc": lock_rc, "lock_ms": lock_ms,
                         "reset_rc": reset_rc, "reset_ms": reset_ms,
                         "default_mhz": default_mhz, "locked_mhz": locked_mhz})
    real = [r for r in rows if abs(r["default_mhz"] - r["locked_mhz"]) > 100]
    return {"lock_target_mhz": mhz, "trials": rows,
            "real_transitions": len(real),
            "lock_ms": statistics.mean(r["lock_ms"] for r in rows),
            "reset_ms": statistics.mean(r["reset_ms"] for r in rows)}


def verify_restored(gpu: Gpu, unlocked_mhz: float) -> dict:
    with CudaLoad():
        tr = gpu.trace(2.0)
    now = med(tr, "clock_mhz")
    return {"load_clock_after_reset_mhz": now, "unlocked_load_clock_before_mhz": unlocked_mhz,
            "restored": now >= unlocked_mhz - 150}


# ---------------------------------------------------------------------------
# Delay levels and comparisons
# ---------------------------------------------------------------------------

def nanmedian(values):
    v = [x for x in values if x == x]
    return statistics.median(v) if v else float("nan")


def nanmax(*values):
    v = [x for x in values if x == x]
    return max(v) if v else float("nan")


def derive_delay_levels(nvml_refresh_ms: float, clock_settle_ms: float,
                        power_settle_ms: float) -> dict:
    """Four levels tied to measured timescales:
        0       fresh telemetry
        R       one NVML refresh period (rounded to the nearest 50 ms)
        S       full settling after a clock change: max(clock, power settling),
                rounded up to 50 ms
        2S      two full settling periods (stress)
    If every level is within MATERIAL_CHANGE of PROPOSAL_LEVELS_MS, the proposal
    levels are kept."""
    measured = {"nvml_refresh_ms": nvml_refresh_ms, "clock_settle_ms": clock_settle_ms,
                "power_settle_ms": power_settle_ms}
    settle = [v for v in (clock_settle_ms, power_settle_ms) if math.isfinite(v)]
    if not math.isfinite(nvml_refresh_ms) or not settle:
        return {**measured, "derived_ms": [], "levels_ms": [], "uses_proposal_levels": False}
    r = max(50, 50 * round(nvml_refresh_ms / 50.0))
    s = max(50 * math.ceil(max(settle) / 50.0), r + 50)
    derived = [0, r, s, 2 * s]
    close = all(abs(d - p) <= max(100.0, MATERIAL_CHANGE * p)
                for d, p in zip(derived, PROPOSAL_LEVELS_MS))
    return {**measured, "R_ms": r, "S_ms": s, "derived_ms": derived,
            "levels_ms": list(PROPOSAL_LEVELS_MS) if close else derived,
            "uses_proposal_levels": close}


def same(a: float, b: float) -> bool:
    return abs(a - b) <= max(SAME_ABS_TOL_MS, SAME_REL_TOL * abs(b))


def compare_with_saved(now: dict) -> list[dict]:
    if not os.path.exists(SAVED_TIMING):
        return []
    with open(SAVED_TIMING) as f:
        saved = json.load(f)
    ur = saved.get("nvml_update_rate", {})
    cl = saved.get("command_latency", {})
    ad = saved.get("actuation_delay", {})
    pairs = [
        ("NVML power update period (ms)", ur.get("power_effective_period_ms"),
         now.get("nvml_power_update_period_ms")),
        ("lock command latency (ms)", cl.get("lock_command_latency_ms", {}).get("mean_ms"),
         now.get("lock_ms")),
        ("reset command latency (ms)", cl.get("reset_command_latency_ms", {}).get("mean_ms"),
         now.get("reset_ms")),
        ("clock actuation delay, down (ms)", ad.get("mean_actuation_delay_ms"),
         now.get("actuation_down_ms")),
    ]
    rows = []
    for name, old, new in pairs:
        if old is None or new is None:
            verdict = "not in saved file" if old is None else "not measured now"
        else:
            verdict = "SAME" if same(new, old) else "DIFFERENT"
        rows.append({"quantity": name, "saved": old, "now": new, "verdict": verdict})
    rows.append({"quantity": "software poll interval (ms)", "saved": ur.get("mean_inter_sample_ms"),
                 "now": now.get("software_poll_interval_ms"),
                 "verdict": "script setting, not a GPU property"})
    return rows


def compare_levels(derived: list[int], current: list[int]) -> dict:
    matched = {c: min(derived, key=lambda d: abs(d - c)) if derived else None for c in current}
    justified = {c: matched[c] is not None and abs(matched[c] - c) <= SAME_ABS_TOL_MS
                 for c in current}
    return {"current_ms": current, "derived_ms": derived,
            "current_value_justified": {str(k): v for k, v in justified.items()},
            "same": sorted(derived) == sorted(current)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot(path, tracking, raw, power_update_period_ms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)
    if tracking:
        pc = tracking["per_clock"]
        t = [c["target_mhz"] for c in pc]
        axes[0].plot(t, t, "k--", lw=1, label="target")
        axes[0].plot(t, [c["mean_observed_mhz"] for c in pc], "o-", color="#1F9E77",
                     label="observed under load")
        bad = [c for c in pc if c["status"] != "PASS"]
        if bad:
            axes[0].plot([c["target_mhz"] for c in bad], [c["mean_observed_mhz"] for c in bad],
                         "x", color="#C0392B", ms=9, mew=2, label="excluded (capped/fail)")
        axes[0].set_xlabel("Locked clock (MHz)")
        axes[0].set_ylabel("Observed clock (MHz)")
        axes[0].set_title("(a) Clock control", loc="left")
        axes[0].legend(frameon=False)
        axes[1].plot(t, [c["mean_power_w"] for c in pc], "s-", color="#E8692E")
        axes[1].set_xlabel("Locked clock (MHz)")
        axes[1].set_ylabel("Power under load (W)")
        axes[1].set_title("(b) Power telemetry vs clock", loc="left")
    if raw:
        down = [r for r in raw if r["direction"] == "down" and r["trial"] == 0]
        ax = axes[2]
        ax.plot([r["t_ms"] for r in down], [r["clock_mhz"] for r in down], color="#3B75C4")
        ax.set_xlabel("Time after lock command (ms)")
        ax.set_ylabel("Clock (MHz)", color="#3B75C4")
        ax2 = ax.twinx()
        ax2.plot([r["t_ms"] for r in down], [r["power_w"] for r in down], color="#C0392B")
        ax2.set_ylabel("Power (W)", color="#C0392B")
        ax.set_title(f"(c) Step down; NVML power updates every ~{power_update_period_ms:.0f} ms",
                     loc="left")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fmt(x, unit=""):
    if x is None:
        return "-"
    if isinstance(x, float):
        return "inf" if x == float("inf") else f"{x:.0f}{unit}"
    return f"{x}{unit}"


def main() -> int:
    admin = is_admin()
    # A read-only run must not overwrite the results of a full Administrator run.
    out_dir = OUT_DIR if admin else os.path.join(OUT_DIR, "telemetry_only")
    os.makedirs(out_dir, exist_ok=True)
    gpu = Gpu()
    report = {"gpu": gpu.name(), "admin": admin,
              "start_state": gpu.read(), "current_delay_values_ms": DELAY_VALUES_MS}
    print(f"GPU: {report['gpu']}   admin: {admin}")

    tracking = None
    raw = []
    atexit.register(restore_gpu)
    try:
        print("\n[1/5] Telemetry: idle vs load, sensor update periods ...")
        tel = telemetry_checks(gpu)
        report["telemetry"] = tel

        if not admin:
            print("\nNot an Administrator prompt: clock control and delay tests skipped.")
        else:
            supported = supported_graphics_clocks()
            clocks = select_experiment_clocks(supported, n=7)
            report["experiment_clocks"] = clocks

            print(f"[2/5] Clock control: lock each of {clocks} twice under load ...")
            rows = clock_tracking(gpu, clocks)
            tracking = summarize_tracking(rows, clocks)
            report["clock_tracking_raw"] = rows
            report["clock_tracking"] = tracking

            usable = tracking["usable_clocks"]
            if len(usable) >= 2:
                print(f"[3/5] Clock-change delay: {usable[0]} <-> {usable[-1]} MHz "
                      f"(usable clocks only), 5 trials ...")
                steps, raw = step_response(gpu, usable[0], usable[-1])
                report["step_response"] = steps

                print(f"[4/5] nvidia-smi command latency: default -> {usable[0]} MHz -> default ...")
                report["command_latency"] = command_latency(gpu, usable[0])
            else:
                print("Fewer than 2 usable clocks: delay and command tests skipped.")
    finally:
        restore_gpu()

    if admin:
        print("[5/5] Checking the GPU is back to its default behaviour ...")
        report["restore_check"] = verify_restored(gpu, tel["unlocked_load_clock_mhz"])
    pynvml.nvmlShutdown()

    # ---- derived numbers ----
    now = {"software_poll_interval_ms": tel["software_poll_interval_ms"],
           "nvml_power_update_period_ms": tel["nvml_power_update_period_ms"]}
    if "step_response" in report:
        steps = report["step_response"]
        cl = report["command_latency"]
        now.update({
            "lock_ms": cl["lock_ms"], "reset_ms": cl["reset_ms"],
            "actuation_down_ms": nanmedian(x["clock_settle_ms"] for x in steps["down"]),
            "actuation_up_ms": nanmedian(x["clock_settle_ms"] for x in steps["up"]),
            "power_settle_down_ms": nanmedian(x["power_settle_ms"] for x in steps["down"]),
            "power_settle_up_ms": nanmedian(x["power_settle_ms"] for x in steps["up"]),
        })
        now["worst_clock_settle_ms"] = nanmax(now["actuation_down_ms"], now["actuation_up_ms"])
        now["worst_power_settle_ms"] = nanmax(now["power_settle_down_ms"],
                                              now["power_settle_up_ms"])
        derived = derive_delay_levels(now["nvml_power_update_period_ms"],
                                      now["worst_clock_settle_ms"], now["worst_power_settle_ms"])
        report["derived_delay_levels"] = derived
        report["delay_level_comparison"] = compare_levels(derived["levels_ms"], DELAY_VALUES_MS)
        report["clock_ladder_comparison"] = {
            "usable_now": tracking["usable_clocks"], "config_verified": VERIFIED_CLOCKS_MHZ,
            "same": tracking["usable_clocks"] == VERIFIED_CLOCKS_MHZ}
    report["measured"] = now
    report["comparison_with_saved_timing"] = compare_with_saved(now)

    with open(os.path.join(out_dir, "gpu_check.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    if raw:
        with open(os.path.join(out_dir, "step_traces.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw[0]))
            w.writeheader()
            w.writerows(raw)
    fig_path = os.path.join(out_dir, "fig_gpu_check.png")
    plot(fig_path, tracking, raw, now["nvml_power_update_period_ms"])

    # ---- printed summary ----
    print("\n" + "=" * 64)
    print("FEEDBACK 1: clock control and power telemetry")
    print("=" * 64)
    print(f"Power idle -> load          : {tel['idle_power_w']:.1f} W -> {tel['load_power_w']:.1f} W"
          f"  ({'responds' if tel['power_responds_to_load'] else 'DOES NOT respond'})")
    print(f"Readings valid              : {tel['readings_valid']}  (under load; "
          f"{tel.get('idle_zero_readings', 0)}/{tel.get('n_idle_samples', 0)} idle samples read 0, "
          "expected when the laptop GPU parks)")
    print(f"Software poll interval      : {tel['software_poll_interval_ms']:.0f} ms"
          "  (how often this script reads NVML)")
    print(f"NVML power update period    : {fmt(tel['nvml_power_update_period_ms'], ' ms')}"
          "  (how often the power value actually changes)")
    print(f"NVML clock update period    : {fmt(tel['nvml_clock_update_period_ms'], ' ms')}")
    if tracking:
        print(f"{'target MHz':>10} {'observed':>9} {'spread':>7} {'power W':>8}  status")
        for c in tracking["per_clock"]:
            print(f"{c['target_mhz']:>10} {c['mean_observed_mhz']:>9.0f} "
                  f"{c['repeat_spread_mhz']:>7.0f} {c['mean_power_w']:>8.1f}  {c['status']}")
        print(f"Usable experiment clocks    : {tracking['usable_clocks']}")
        if tracking["excluded_clocks"]:
            print(f"Excluded clocks             : {tracking['excluded_clocks']}")
        print(f"Power increases with clock  : {tracking['power_increases_with_clock']}"
              "  (usable clocks)")
        ok = (tracking["clock_control_reliable"] and tracking["power_increases_with_clock"]
              and tel["power_responds_to_load"] and tel["readings_valid"])
        print(f"VERDICT                     : {'RELIABLE' if ok else 'NOT RELIABLE'}")
        rc = report["restore_check"]
        print(f"GPU restored to default     : {rc['restored']} (load clock "
              f"{rc['load_clock_after_reset_mhz']:.0f} MHz, before {rc['unlocked_load_clock_before_mhz']:.0f} MHz)")

    print("\n" + "=" * 64)
    print("FEEDBACK 2: measured delays -> injected delay levels")
    print("=" * 64)
    if "derived_delay_levels" in report:
        cl = report["command_latency"]
        print(f"nvidia-smi lock / reset     : {now['lock_ms']:.0f} / {now['reset_ms']:.0f} ms"
              f"  ({cl['real_transitions']}/{len(cl['trials'])} trials were real "
              f"default -> {cl['lock_target_mhz']} MHz -> default transitions)")
        print(f"Clock settles, down / up    : {fmt(now['actuation_down_ms'])} / "
              f"{fmt(now['actuation_up_ms'])} ms  (in band for {SETTLE_HOLD_MS:.0f} ms)")
        print(f"Power settles, down / up    : {fmt(now['power_settle_down_ms'])} / "
              f"{fmt(now['power_settle_up_ms'])} ms")
        d = report["derived_delay_levels"]
        print("Measured timescales         :")
        print(f"  NVML refresh R            : {d['nvml_refresh_ms']:.1f} ms -> {d.get('R_ms')} ms")
        print(f"  clock settling (worst)    : {d['clock_settle_ms']:.1f} ms")
        print(f"  power settling (worst)    : {d['power_settle_ms']:.1f} ms")
        print(f"  full settling S           : -> {d.get('S_ms')} ms (max of clock/power, rounded up)")
        print(f"Derived [0, R, S, 2S]       : {d['derived_ms']}")
        if d["uses_proposal_levels"]:
            print(f"Proposal delay levels       : {d['levels_ms']}  (measurements match within "
                  f"{MATERIAL_CHANGE:.0%}/100 ms)")
        else:
            print(f"Proposal delay levels       : {d['levels_ms']}  (CHANGED: measurements moved "
                  f"away from {PROPOSAL_LEVELS_MS})")
        cmp_ = report["delay_level_comparison"]
        print(f"Current DELAY_VALUES_MS     : {cmp_['current_ms']}")
        for c, ok in cmp_["current_value_justified"].items():
            print(f"  current {c:>5} ms -> {'matches a measured level' if ok else 'not backed by a measurement'}")
        print(f"Same as current?            : {'YES' if cmp_['same'] else 'NO'}")
        lad = report["clock_ladder_comparison"]
        print(f"Usable clocks vs config     : {lad['usable_now']} vs VERIFIED_CLOCKS_MHZ "
              f"{lad['config_verified']} -> {'SAME' if lad['same'] else 'DIFFERENT'}")
    elif admin:
        print("Fewer than 2 usable clocks, so delay levels could not be derived.")
    else:
        print("Run from an Administrator prompt to measure clock-change delays.")

    rows = report["comparison_with_saved_timing"]
    if rows:
        print("\nCompared with results/timing_characterisation.json:")
        for r in rows:
            print(f"  {r['quantity']:<34} saved={fmt(r['saved']):>6}  now={fmt(r['now']):>6}  {r['verdict']}")
    print(f"\nSaved {out_dir}")
    return 0 if admin else 1


if __name__ == "__main__":
    sys.exit(main())
