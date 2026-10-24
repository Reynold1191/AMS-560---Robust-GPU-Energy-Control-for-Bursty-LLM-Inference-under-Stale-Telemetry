"""Shared controller state and clock-change dwell enforcement (Member 2)."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


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
    # Mirror the experiment target for state logging and latency_slack.
    # Policy decisions use the target supplied to the controller constructor.
    slo_target_s: float = 8.5
    error_budget_burn: float = 0.0
    miss_rate: float = 0.0
    phase: str = "steady"

    @property
    def latency_slack(self) -> float:
        return 1.0 - self.p95_latency_s / self.slo_target_s


class BaseController(ABC):
    """Policies return indices into clocks supplied in increasing MHz order."""

    def __init__(self, clocks: List[int]):
        if not clocks:
            raise ValueError("clocks must contain at least one frequency")
        if any(clock <= 0 for clock in clocks) or any(
            left >= right for left, right in zip(clocks, clocks[1:])
        ):
            raise ValueError("clocks must be positive and strictly increasing")
        self.clocks = list(clocks)
        self.min_dwell = 0
        self._intervals_since_change = 0
        self._last_change_t = time.monotonic()
        self.last_reason = "INIT"

    @abstractmethod
    def _policy(self, state: ControllerState, current_idx: int) -> int:
        """Return the desired clock index and set last_reason."""

    def decide(
        self, state: ControllerState, current_idx: int, min_dwell: Optional[int] = None
    ) -> int:
        """Apply dwell to reductions; increases always take effect immediately.

        An omitted min_dwell uses the controller's configured default. Passing
        zero explicitly disables dwell for this decision.
        """
        dwell = self.min_dwell if min_dwell is None else min_dwell
        if dwell < 0:
            raise ValueError("min_dwell must be nonnegative")
        current_idx = self._clamp_index(current_idx)
        desired_idx = self._clamp_index(self._policy(state, current_idx))
        if desired_idx > current_idx:
            self._record_change()
            return desired_idx

        self._intervals_since_change += 1
        if desired_idx == current_idx:
            return current_idx
        if self._intervals_since_change >= dwell:
            self._record_change()
            return desired_idx
        self.last_reason += " DWELL_HOLD"
        return current_idx

    def _record_change(self) -> None:
        self._intervals_since_change = 0
        self._last_change_t = time.monotonic()

    def _clamp_index(self, idx: int) -> int:
        return max(0, min(idx, len(self.clocks) - 1))

    def clock_mhz(self, idx: int) -> int:
        return self.clocks[self._clamp_index(idx)]

    def find_index(self, mhz: float) -> int:
        return min(range(len(self.clocks)), key=lambda idx: abs(self.clocks[idx] - mhz))

    @property
    def name(self) -> str:
        return self.__class__.__name__
