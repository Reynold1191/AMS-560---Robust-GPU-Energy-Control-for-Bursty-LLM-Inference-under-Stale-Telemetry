"""
slo_budget.py
Rolling SLO error-budget tracker.

Concept (from Google SRE)
--------------------------
For a 99 % SLO, at most 1 % of requests may miss the latency target over
a rolling window.  That 1 % allowance is the "error budget."

  burn_rate = current_miss_rate / budget_ratio

  burn_rate = 1.0  →  consuming the budget at exactly the allowed rate
  burn_rate > 1.0  →  over-budget (reliability is degrading)
  burn_rate > 2.0  →  emergency: the robust controller raises the clock

The controller uses burn_rate as a long-horizon reliability signal that
supplements (not replaces) instantaneous P95 latency.
"""

from __future__ import annotations

import collections
import statistics
import threading
import time
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Rolling window helper
# ---------------------------------------------------------------------------

class _RollingWindow:
    """Thread-safe deque with a wall-clock expiry horizon."""

    def __init__(self, window_s: float):
        self._window_s = window_s
        self._buf: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def append(self, timestamp: float, value) -> None:
        with self._lock:
            self._buf.append((timestamp, value))
            self._trim(timestamp)

    def snapshot(self, now: float = None) -> List:
        """Return all values within the rolling window."""
        if now is None:
            now = time.monotonic()
        with self._lock:
            self._trim(now)
            return [v for _, v in self._buf]

    def _trim(self, now: float) -> None:
        """Remove entries older than window_s (called while lock is held)."""
        cutoff = now - self._window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# ---------------------------------------------------------------------------
# Main budget tracker
# ---------------------------------------------------------------------------

class SLOBudget:
    """
    Tracks request latencies and SLO outcomes in a rolling time window.

    Parameters
    ----------
    slo_target_s  : Latency threshold; a request is a "miss" if it exceeds this.
    budget_ratio  : Allowed miss fraction (e.g. 0.01 for a 99 % SLO).
    window_s      : Rolling window length for budget accounting.
    """

    def __init__(
        self,
        slo_target_s: float = 2.0,
        budget_ratio: float = 0.01,
        window_s: float = 60.0,
    ):
        self.slo_target_s = slo_target_s
        self.budget_ratio = budget_ratio
        self.window_s = window_s

        # Store (finish_time, latency_s, is_miss)
        self._latencies = _RollingWindow(window_s)
        self._misses    = _RollingWindow(window_s)

    # ------------------------------------------------------------------ #
    # Recording                                                             #
    # ------------------------------------------------------------------ #

    def record(self, finish_t: float, latency_s: float) -> None:
        """Record a completed request's latency outcome."""
        is_miss = latency_s > self.slo_target_s
        self._latencies.append(finish_t, latency_s)
        if is_miss:
            self._misses.append(finish_t, latency_s)

    # ------------------------------------------------------------------ #
    # Derived signals                                                       #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Number of requests in the rolling window."""
        return len(self._latencies)

    def miss_count(self) -> int:
        """Number of SLO misses in the rolling window."""
        return len(self._misses)

    def miss_rate(self) -> float:
        """Fraction of requests in the window that missed the SLO."""
        n = self.count()
        return len(self._misses) / n if n > 0 else 0.0

    def burn_rate(self) -> float:
        """
        Budget burn rate.
          1.0 → at budget
          > 1 → over-budget
          0.0 → no violations
        """
        if self.budget_ratio <= 0:
            return float("inf")
        return self.miss_rate() / self.budget_ratio

    def remaining_budget(self) -> float:
        """
        Fraction of the error budget remaining (1.0 = full, 0.0 = depleted).
        Can go negative when the budget is exceeded.
        """
        return 1.0 - (self.miss_rate() / self.budget_ratio)

    def percentile_latency(self, p: float) -> float:
        """
        p-th percentile latency (e.g. p=0.95 for P95) from the rolling window.
        Returns 0.0 if there are no samples.
        """
        vals = self._latencies.snapshot()
        if not vals:
            return 0.0
        vals_sorted = sorted(vals)
        idx = int(p * len(vals_sorted))
        idx = min(idx, len(vals_sorted) - 1)
        return vals_sorted[idx]

    def p95(self) -> float:
        """Convenience wrapper for the 95th percentile latency."""
        return self.percentile_latency(0.95)

    def p99(self) -> float:
        """Convenience wrapper for the 99th percentile latency."""
        return self.percentile_latency(0.99)

    def mean_latency(self) -> float:
        """Mean latency in the rolling window."""
        vals = self._latencies.snapshot()
        return statistics.mean(vals) if vals else 0.0

    def summary(self) -> dict:
        """Return a snapshot dict suitable for logging."""
        return {
            "window_count":   self.count(),
            "window_misses":  self.miss_count(),
            "miss_rate":      self.miss_rate(),
            "burn_rate":      self.burn_rate(),
            "remaining_budget": self.remaining_budget(),
            "p95_latency_s":  self.p95(),
            "p99_latency_s":  self.p99(),
            "mean_latency_s": self.mean_latency(),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    budget = SLOBudget(slo_target_s=2.0, budget_ratio=0.01, window_s=60.0)
    rng = random.Random(42)

    # Simulate 200 requests with 2 % miss rate (twice the budget)
    now = time.monotonic()
    for i in range(200):
        lat = rng.expovariate(1.0 / 1.5)  # mean 1.5 s
        budget.record(now + i * 0.3, lat)

    s = budget.summary()
    print(f"Count:       {s['window_count']}")
    print(f"Miss rate:   {s['miss_rate']:.3f}")
    print(f"Burn rate:   {s['burn_rate']:.2f}  (>1 = over-budget)")
    print(f"P95 latency: {s['p95_latency_s']:.3f} s")
    print(f"P99 latency: {s['p99_latency_s']:.3f} s")
