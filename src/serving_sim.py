"""
serving_sim.py
Trace-driven replay of the real serving stack, for controller design and
the scale-out study (D10).

The old scale_simulator modelled single-request exponential service, gave
the controllers fresh queue and arrival signals, and set no capacity floor,
so every controller and delay produced the same numbers. This model copies
what run_experiment.py does on the GPU:

  * arrivals : the same non-homogeneous Poisson schedule (thinning on
               observation.arrival_rate_at), warm-up at the base rate;
  * workers  : each takes up to `batch` queued requests when idle (as
               llm_worker does); batch time t(n, f) = a + (T_full(f) - a) n/B
               with T_full from capacity.json (measured in the real stack)
               and a = 0.55 s (fit to the attempt_11 sweep), +-8 % noise;
  * clocks   : one clock for the whole fleet; a change takes effect after
               clock_settle_s (0.7 s measured) and applies to batches that
               start afterwards;
  * power    : idle / busy watts per clock from the sweep telemetry;
  * signals  : queue length, 10 s arrival rate, CV, rolling P95 / burn /
               miss, power, clock - all sampled every 0.1 s and handed to
               the controller `delay` seconds late, exactly like
               observation.delayed_state();
  * control  : controllers.get_controller() with the capacity floor, every
               control_interval_s, with min_dwell.

With n_workers > 1 the arrival rate and the capacity table are scaled by
n_workers and requests join the shortest queue.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import ExperimentConfig, VERIFIED_CLOCKS_MHZ
from controllers import get_controller
from controllers.base import ControllerState
from observation import arrival_rate_at, measurement_length_s, phase_at

BATCH_OVERHEAD_S = 0.55
TOKENS_PER_REQUEST = 22.0     # attempt_11 mean output tokens per request

# attempt_11 sweep telemetry: median watts at util < 5 % and util > 95 %
DEFAULT_POWER = {180: (6.8, 11.7), 667: (11.1, 23.7), 1147: (14.0, 29.7),
                 1635: (15.0, 36.8), 2122: (17.6, 44.4)}
DEFAULT_FULL_BATCH_S = {180: 23.72, 667: 5.16, 1147: 3.23, 1635: 2.42, 2122: 2.02}

# Median batch time (s) by batch size, measured in the attempt_10/11 controller
# runs (~19k batches, clock from telemetry at batch start). Columns are batch
# sizes 1, 3, 5, 7, 9, 11, 13, 16. 180 MHz is never used by the controllers;
# it scales the 667 MHz row by the sweep's full-batch ratio.
BATCH_SIZES = [1, 3, 5, 7, 9, 11, 13, 16]
BATCH_TABLE_S = {
    667:  [0.86, 1.48, 2.00, 2.58, 2.81, 2.91, 2.91, 2.95],
    1147: [0.81, 1.06, 1.41, 1.70, 2.01, 2.22, 2.30, 2.56],
    1635: [0.75, 0.97, 1.17, 1.34, 1.64, 1.81, 1.87, 2.05],
    2122: [0.70, 0.91, 1.06, 1.23, 1.42, 1.61, 1.78, 2.02],
}
BATCH_TABLE_S[180] = [v * 23.72 / 5.16 for v in BATCH_TABLE_S[667]]
POWER_GAMMA = 0.5   # busy watts = idle + (busy - idle) * (n / batch) ** gamma


@dataclass
class SimModel:
    clocks: List[int]
    full_batch_s: Dict[int, float]
    power: Dict[int, tuple]            # clock -> (idle_w, busy_w)
    batch: int = 16
    overhead_s: float = BATCH_OVERHEAD_S
    noise: float = 0.08

    @classmethod
    def from_results(cls, results_dir: Optional[str] = None, batch: int = 16) -> "SimModel":
        clocks = list(VERIFIED_CLOCKS_MHZ)
        full = dict(DEFAULT_FULL_BATCH_S)
        if results_dir:
            cap = os.path.join(results_dir, "capacity.json")
            if os.path.exists(cap):
                with open(cap, encoding="utf-8") as f:
                    c = json.load(f)
                full = {int(m): float(t) for m, t in zip(c["clocks_mhz"], c["batch_time_s"])}
                clocks = sorted(full)
                batch = int(c.get("batch_size", batch))
        return cls(clocks=clocks, full_batch_s=full, power=dict(DEFAULT_POWER), batch=batch)

    def batch_time(self, n: int, mhz: int, rng: random.Random) -> float:
        row = BATCH_TABLE_S.get(mhz)
        if row is None:
            full = self.full_batch_s[mhz]
            a = min(self.overhead_s, 0.5 * full)
            t = a + (full - a) * n / self.batch
        else:
            x = n * 16.0 / self.batch
            j = max(1, min(bisect.bisect_left(BATCH_SIZES, x), len(BATCH_SIZES) - 1))
            x0, x1 = BATCH_SIZES[j - 1], BATCH_SIZES[j]
            f = min(max((x - x0) / (x1 - x0), 0.0), 1.0)
            t = row[j - 1] + f * (row[j] - row[j - 1])
        return t * (1.0 + rng.uniform(-self.noise, self.noise))

    def busy_power(self, n: int, mhz: int) -> float:
        idle, busy = self.power[mhz]
        return idle + (busy - idle) * (n / self.batch) ** POWER_GAMMA

    def capacity_rps(self, n_workers: int = 1) -> List[float]:
        return [n_workers * self.batch / self.full_batch_s[m] for m in self.clocks]


class _Window:
    """Rolling window of (t, value) on simulated time."""

    def __init__(self, span: float):
        self.span = span
        self.buf: deque = deque()

    def add(self, t, v):
        self.buf.append((t, v))

    def values(self, now):
        while self.buf and self.buf[0][0] < now - self.span:
            self.buf.popleft()
        return [v for _, v in self.buf]


def _pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(int(p * len(s)), len(s) - 1)]


def _arrivals(cfg, rng, n_workers):
    """Thinning on arrival_rate_at; times relative to the end of warm-up."""
    m = measurement_length_s(cfg)
    peak = n_workers * max(cfg.base_arrival_rate * max(cfg.burst_multiplier, 1.0),
                           arrival_rate_at(m, cfg), cfg.base_arrival_rate)
    t, out = -cfg.warmup_s, []
    while True:
        t += rng.expovariate(peak)
        if t >= m:
            return out
        rate = n_workers * arrival_rate_at(t if t >= 0 else None, cfg)
        if rng.random() * peak <= rate:
            out.append(t)


def simulate_run(cfg: ExperimentConfig, model: SimModel, n_workers: int = 1,
                 dt: float = 0.05, keep_trace: bool = False) -> dict:
    """One run; returns the same headline metrics as run_summary.csv."""
    rng = random.Random(cfg.seed)
    arr = _arrivals(cfg, random.Random(cfg.seed * 7919 + 1), n_workers)
    clocks = model.clocks
    cfg.capacity_rps = model.capacity_rps(n_workers)
    cfg.batch_time_s = [model.full_batch_s[m] for m in clocks]
    ctrl = get_controller(cfg.controller, clocks, cfg)
    tgt = getattr(ctrl, "_target_idx", None)
    cur = int(tgt) if tgt is not None else len(clocks) // 2
    applied, pending = cur, None            # (time, idx) of a settling change
    m_len = measurement_length_s(cfg)

    queues = [deque() for _ in range(n_workers)]
    busy_until = [-1e9] * n_workers
    in_batch: List[list] = [[] for _ in range(n_workers)]
    batch_mhz = [clocks[cur]] * n_workers
    lat_w, miss_w = _Window(cfg.budget_window_s), _Window(cfg.budget_window_s)
    iat: deque = deque(maxlen=400)
    last_arr = None
    hist: deque = deque()                   # delayed observation samples
    done, energy, ctrl_ev = [], 0.0, []
    ai, t = 0, -cfg.warmup_s
    next_obs, next_ctrl = t, 0.0
    delay_s = cfg.delay_ms / 1000.0
    end = m_len + 120.0

    while t < end:
        # arrivals
        while ai < len(arr) and arr[ai] <= t:
            w = min(range(n_workers), key=lambda i: len(queues[i]) + (busy_until[i] > t) * model.batch)
            queues[w].append(arr[ai])
            if last_arr is not None:
                iat.append(arr[ai] - last_arr)
            last_arr = arr[ai]
            ai += 1
        # clock settling
        if pending and t >= pending[0]:
            applied, pending = pending[1], None
        # workers
        for w in range(n_workers):
            if in_batch[w] and t >= busy_until[w]:
                for a in in_batch[w]:
                    lat = busy_until[w] - a
                    lat_w.add(busy_until[w], lat)
                    if lat > cfg.slo_target_s:
                        miss_w.add(busy_until[w], lat)
                    if a >= 0:
                        done.append((a, busy_until[w], lat))
                in_batch[w] = []
            if not in_batch[w] and queues[w] and t >= busy_until[w]:
                n = min(model.batch, len(queues[w]))
                in_batch[w] = [queues[w].popleft() for _ in range(n)]
                batch_mhz[w] = clocks[applied]
                busy_until[w] = t + model.batch_time(n, batch_mhz[w], rng)
            if 0 <= t < m_len:
                energy += (model.busy_power(len(in_batch[w]), batch_mhz[w]) if in_batch[w]
                           else model.power[clocks[applied]][0]) * dt
        # observations (every 0.1 s)
        if t >= next_obs:
            lats = lat_w.values(t)
            n_l = len(lats)
            miss = len(miss_w.values(t)) / n_l if n_l else 0.0
            acc = cnt = 0
            for v in reversed(iat):
                acc += v
                cnt += 1
                if acc >= 10.0:
                    break
            rate = cnt / acc if acc > 0 else 0.0
            recent = list(iat)[-30:]
            if len(recent) >= 2:
                mu = sum(recent) / len(recent)
                sd = math.sqrt(sum((x - mu) ** 2 for x in recent) / (len(recent) - 1))
                cv = sd / mu if mu > 0 else 0.0
            else:
                cv = 0.0
            pw = sum(model.busy_power(len(in_batch[w]), batch_mhz[w]) if in_batch[w]
                     else model.power[clocks[applied]][0] for w in range(n_workers))
            hist.append(dict(t=t, q=sum(len(q) for q in queues), rate=rate, cv=cv,
                             p95=_pct(lats, 0.95), p99=_pct(lats, 0.99),
                             burn=miss / cfg.slo_budget_ratio, miss=miss,
                             power=pw, mhz=clocks[applied]))
            next_obs += cfg.telemetry_poll_s
        # control
        if 0 <= t < m_len and t >= next_ctrl:
            obs = None
            for h in reversed(hist):
                if h["t"] <= t - delay_s:
                    obs = h
                    break
            if obs is not None:
                st = ControllerState(
                    t=t, power_w=obs["power"], graphics_mhz=obs["mhz"], sm_mhz=obs["mhz"],
                    util_pct=0, telemetry_age_ms=(t - obs["t"]) * 1000.0,
                    queue_len=obs["q"], arrival_rate=obs["rate"], arrival_cv=obs["cv"],
                    p95_latency_s=obs["p95"], p99_latency_s=obs["p99"],
                    slo_target_s=cfg.slo_target_s, error_budget_burn=obs["burn"],
                    miss_rate=obs["miss"], phase=phase_at(t, cfg))
                new = max(0, min(ctrl.decide(st, cur, min_dwell=cfg.min_dwell_intervals),
                                 len(clocks) - 1))
                if new != cur:
                    cur = new
                    pending = (t + getattr(cfg, "clock_settle_s", 0.7), new)
                if keep_trace:
                    ctrl_ev.append(dict(t=t, phase=st.phase, q=sum(len(q) for q in queues),
                                        obs_q=obs["q"], obs_rate=obs["rate"],
                                        obs_p95=obs["p95"], mhz=clocks[cur],
                                        reason=getattr(ctrl, "last_reason", "")))
            next_ctrl += cfg.control_interval_s
        while len(hist) > 2 and hist[1]["t"] < t - delay_s - 5.0:
            hist.popleft()
        if t >= m_len and ai >= len(arr) and not any(queues) and not any(in_batch):
            break
        t += dt

    lats = [l for a, f, l in done if a < m_len]
    burst = [l for a, f, l in done
             if cfg.workload == "sharp_burst" and phase_at(a, cfg) == "burst"]
    n = len(lats)
    tokens = n * TOKENS_PER_REQUEST
    res = dict(controller=cfg.controller, delay_ms=cfg.delay_ms, workload=cfg.workload,
               seed=cfg.seed, n_workers=n_workers, n_requests=n,
               total_j=energy, j_per_token=energy / tokens if tokens else float("nan"),
               p50=_pct(lats, 0.5), p95=_pct(lats, 0.95), p99=_pct(lats, 0.99),
               miss_rate=sum(l > cfg.slo_target_s for l in lats) / n if n else float("nan"),
               burst_miss_rate=(sum(l > cfg.slo_target_s for l in burst) / len(burst)
                                if burst else float("nan")),
               n_clock_changes=sum(1 for i in range(1, len(ctrl_ev))
                                   if ctrl_ev[i]["mhz"] != ctrl_ev[i - 1]["mhz"])
               if keep_trace else None)
    if keep_trace:
        res["trace"] = ctrl_ev
    return res


def make_cfg(controller, delay_ms, workload, seed, slo, base_rate=1.75, **kw) -> ExperimentConfig:
    c = ExperimentConfig(controller=controller, delay_ms=float(delay_ms), workload=workload,
                         seed=seed, slo_target_s=slo, base_arrival_rate=base_rate,
                         burst_multiplier=4.0, duration_s=120.0, warmup_s=10.0,
                         run_id=f"sim_{controller}_d{delay_ms}_{workload}_s{seed}")
    for k, v in kw.items():
        setattr(c, k, v)
    return c
