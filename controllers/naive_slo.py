"""Latency-only dynamic baseline (Member 2)."""

from .base import BaseController, ControllerState


class NaiveSLOController(BaseController):
    def __init__(self, clocks, slo_target_s=8.5, down_slack=0.80, min_dwell=2):
        super().__init__(clocks)
        self.slo_target_s = slo_target_s
        self.down_slack = down_slack
        self.min_dwell = min_dwell

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        # A snapshot's metadata must not override the configured experiment SLO.
        slo_target = self.slo_target_s
        if state.p95_latency_s > slo_target:
            self.last_reason = "EMERGENCY_SLO"
            return min(current_idx + 1, len(self.clocks) - 1)
        if state.p95_latency_s < self.down_slack * slo_target and state.queue_len == 0:
            self.last_reason = "SAFE_DOWNCLOCK"
            return max(current_idx - 1, 0)
        self.last_reason = "HOLD"
        return current_idx

    @property
    def name(self) -> str:
        return "NaiveSLO"
