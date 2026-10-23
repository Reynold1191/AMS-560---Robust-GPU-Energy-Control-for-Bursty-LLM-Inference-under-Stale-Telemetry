import threading
import time
from collections import deque
from typing import List, Optional


class _RollingWindow:
    def __init__(self, window_s: float):
        self.window_s = window_s
        self._buf = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.popleft()

    def append(self, timestamp: float, value: float) -> None:
        with self._lock:
            self._buf.append((timestamp, value))
            self._trim(max(timestamp, time.monotonic()))

    def snapshot(self, now: Optional[float] = None) -> List[float]:
        with self._lock:
            self._trim(time.monotonic() if now is None else now)
            return [v for _, v in self._buf]

    def __len__(self) -> int:
        with self._lock:
            self._trim(time.monotonic())
            return len(self._buf)


class SLOBudget:
    """Rolling-window SLO miss tracking and error-budget burn rate."""

    def __init__(self, slo_target_s: float, budget_ratio: float, window_s: float):
        self.slo_target_s = slo_target_s
        self.budget_ratio = budget_ratio
        self.window_s = window_s
        self._latencies = _RollingWindow(window_s)
        self._misses = _RollingWindow(window_s)

    def record(self, finish_t: float, latency_s: float) -> None:
        self._latencies.append(finish_t, latency_s)
        if latency_s > self.slo_target_s:
            self._misses.append(finish_t, latency_s)

    def count(self) -> int:
        return len(self._latencies)

    def miss_count(self) -> int:
        return len(self._misses)

    def miss_rate(self) -> float:
        n = self.count()
        return self.miss_count() / n if n else 0.0

    def burn_rate(self) -> float:
        return self.miss_rate() / self.budget_ratio

    def remaining_budget(self) -> float:
        return 1.0 - self.miss_rate() / self.budget_ratio

    def percentile_latency(self, p: float) -> float:
        vals = sorted(self._latencies.snapshot())
        n = len(vals)
        if n == 0:
            return 0.0
        return vals[min(int(p * n), n - 1)]

    def p95(self) -> float:
        return self.percentile_latency(0.95)

    def p99(self) -> float:
        return self.percentile_latency(0.99)

    def mean_latency(self) -> float:
        vals = self._latencies.snapshot()
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {
            "window_count": self.count(),
            "miss_rate": self.miss_rate(),
            "burn_rate": self.burn_rate(),
            "p95_latency_s": self.p95(),
            "p99_latency_s": self.p99(),
            "mean_latency_s": self.mean_latency(),
        }
