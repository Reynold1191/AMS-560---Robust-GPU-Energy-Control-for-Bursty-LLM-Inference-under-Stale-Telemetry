"""
controllers/base.py
Abstract base class and shared ControllerState dataclass.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Controller input: the view the controller sees at each decision point
# ---------------------------------------------------------------------------

@dataclass
class ControllerState:
    """
    All information available to the controller at one decision point.

    The telemetry here is potentially *stale* (delayed by delay_ms).
    The queue_len, p95_latency_s, and error_budget_burn fields are
    derived from the same stale view so the controller does not have
    unfair access to fresh data.
    """

    # ── Time ─────────────────────────────────────────────────────────────
    t: float = field(default_factory=time.monotonic)
    """Wall-clock time of this decision."""

    # ── Delayed telemetry ────────────────────────────────────────────────
    power_w: float = 0.0
    graphics_mhz: int = 0
    sm_mhz: int = 0
    util_pct: int = 0
    temp_c: int = 0
    telemetry_age_ms: float = 0.0
    """Age of the telemetry sample in milliseconds."""

    # ── Workload signals ─────────────────────────────────────────────────
    queue_len: int = 0
    arrival_rate: float = 0.0      # requests/s (recent window)
    arrival_cv: float = 0.0        # coefficient of variation (burstiness)

    # ── SLO signals ──────────────────────────────────────────────────────
    p95_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    slo_target_s: float = 2.0
    error_budget_burn: float = 0.0
    """Burn rate: 1.0 = at budget, >1 = over-budget."""
    miss_rate: float = 0.0

    # ── Phase tag ────────────────────────────────────────────────────────
    phase: str = "steady"

    # Convenience
    @property
    def latency_slack(self) -> float:
        """Fraction of SLO headroom remaining (1.0 = no latency yet)."""
        if self.slo_target_s <= 0:
            return 0.0
        return 1.0 - self.p95_latency_s / self.slo_target_s


# ---------------------------------------------------------------------------
# Abstract base controller
# ---------------------------------------------------------------------------

class BaseController(ABC):
    """
    All frequency controllers implement this interface.

    decide(state, clocks, current_idx) → new_idx
    """

    def __init__(self, clocks: List[int]):
        if not clocks:
            raise ValueError("Clock list must not be empty.")
        self.clocks = clocks
        self._intervals_since_change: int = 0
        self._last_change_t: float = time.monotonic()
        self.last_reason: str = "INIT"
        """Human-readable reason for the most recent frequency decision."""
        # Capacity floor (set_capacity); None = no floor.
        self.capacity_rps: Optional[List[float]] = None
        self.batch_time_s: Optional[List[float]] = None
        self.capacity_margin: float = 1.25
        self.last_change_t: Optional[float] = None
        """state.t of the last decision that changed the clock."""

    def set_capacity(self, capacity_rps, batch_time_s=None, margin: float = 1.25) -> None:
        """
        Per-clock capacity (req/s, aligned with self.clocks). decide() then
        never returns a clock whose capacity is below margin x the observed
        arrival rate. Applied equally to every dynamic controller: it encodes
        the measured feasibility of each clock, not a controller policy.
        """
        if capacity_rps is not None and len(capacity_rps) != len(self.clocks):
            raise ValueError("capacity_rps must have one value per clock")
        self.capacity_rps = list(capacity_rps) if capacity_rps is not None else None
        self.batch_time_s = list(batch_time_s) if batch_time_s is not None else None
        self.capacity_margin = margin

    def capacity_floor_idx(self, arrival_rate: float) -> int:
        """Lowest clock index whose capacity covers margin x arrival_rate."""
        if not self.capacity_rps or arrival_rate <= 0:
            return 0
        need = self.capacity_margin * arrival_rate
        for i, cap in enumerate(self.capacity_rps):
            if cap >= need:
                return i
        return len(self.clocks) - 1

    # ------------------------------------------------------------------ #
    # Interface                                                             #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _policy(
        self,
        state: ControllerState,
        current_idx: int,
    ) -> int:
        """
        Pure frequency policy: return the desired clock *index*.
        Dwell enforcement is applied by BaseController.decide().
        """

    def decide(
        self,
        state: ControllerState,
        current_idx: int,
        min_dwell: int = 0,
    ) -> int:
        """
        Return a clock index (into self.clocks).
        Enforces min_dwell by not changing the clock before the dwell
        expires, *unless* an emergency escalation is needed.
        """
        new_idx = self._decide(state, current_idx, min_dwell)
        if new_idx != current_idx:
            self.last_change_t = state.t
        return new_idx

    def _decide(self, state: ControllerState, current_idx: int, min_dwell: int) -> int:
        desired_idx = self._policy(state, current_idx)
        desired_idx = max(0, min(desired_idx, len(self.clocks) - 1))

        floor = self.capacity_floor_idx(state.arrival_rate)
        if desired_idx < floor:
            desired_idx = floor
            self.last_reason = f"{self.last_reason} CAPACITY_FLOOR"

        # Emergency escalation always bypasses dwell
        if desired_idx > current_idx:
            self._intervals_since_change = 0
            return desired_idx

        self._intervals_since_change += 1
        if desired_idx == current_idx:
            return current_idx

        # Down-clock only if dwell has elapsed
        if self._intervals_since_change >= min_dwell:
            self._intervals_since_change = 0
            return desired_idx

        # Dwell not yet satisfied: hold
        return current_idx

    # ------------------------------------------------------------------ #
    # Helpers                                                               #
    # ------------------------------------------------------------------ #

    def clock_mhz(self, idx: int) -> int:
        return self.clocks[max(0, min(idx, len(self.clocks) - 1))]

    def find_index(self, mhz: int) -> int:
        """Return index of the closest clock value."""
        return min(
            range(len(self.clocks)),
            key=lambda i: abs(self.clocks[i] - mhz)
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__
