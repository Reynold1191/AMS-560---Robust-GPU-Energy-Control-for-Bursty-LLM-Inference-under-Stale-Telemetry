import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List


@dataclass
class ControllerState:
    t: float = field(default_factory=time.monotonic)
    power_w: float = 0.0
    graphics_mhz: int = 0
    sm_mhz: int = 0
    util_pct: int = 0
    temp_c: int = 0
    telemetry_age_ms: float = 0.0
    queue_len: int = 0
    arrival_rate: float = 0.0
    arrival_cv: float = 0.0
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    slo_target_s: float = 2.0
    error_budget_burn: float = 0.0
    miss_rate: float = 0.0
    phase: str = "steady"

    @property
    def latency_slack(self) -> float:
        if self.slo_target_s <= 0:
            return 0.0
        return 1.0 - self.p95_latency_s / self.slo_target_s


class BaseController(ABC):
    def __init__(self, clocks: List[int]):
        if not clocks:
            raise ValueError("clocks must be a non-empty list of MHz values")
        self.clocks = list(clocks)
        self._intervals_since_change = 0
        self._last_change_t = time.monotonic()
        self.last_reason = "INIT"

    @abstractmethod
    def _policy(self, state: ControllerState, current_idx: int) -> int:
        """Return the desired clock index."""

    def _clamp(self, idx: int) -> int:
        return max(0, min(idx, len(self.clocks) - 1))

    def decide(self, state: ControllerState, current_idx: int, min_dwell: int = 0) -> int:
        desired = self._clamp(self._policy(state, current_idx))

        # Up-clocks are never delayed by dwell.
        if desired > current_idx:
            self._intervals_since_change = 0
            self._last_change_t = time.monotonic()
            return desired

        self._intervals_since_change += 1
        if desired == current_idx:
            return current_idx
        if self._intervals_since_change >= min_dwell:
            self._intervals_since_change = 0
            self._last_change_t = time.monotonic()
            return desired
        return current_idx

    def clock_mhz(self, idx: int) -> int:
        return self.clocks[self._clamp(idx)]

    def find_index(self, mhz: int) -> int:
        return min(range(len(self.clocks)), key=lambda i: abs(self.clocks[i] - mhz))

    @property
    def name(self) -> str:
        return self.__class__.__name__
