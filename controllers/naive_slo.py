"""
controllers/naive_slo.py
Naive SLO-aware DVFS controller.

This controller is the *baseline* that the robust controller must
outperform under stale telemetry.  It reacts to queue length and P95
latency drawn from the (potentially delayed) telemetry view, but does
NOT account for how old that view is or how quickly the error budget is
being consumed.

Policy (from implementation guide §17)
----------------------------------------
  Emergency:     P95 > SLO_target       → up-clock (step +1)
  Down-clock:    P95 < 0.80 × target
                 AND queue_len == 0     → down-clock (step -1)
  Hold:          otherwise              → keep current clock

The naive controller's failure mode: it may still be issuing down-clock
commands during the *delay window* after a burst starts, because the
stale telemetry shows low load even though real load has already surged.
"""

from __future__ import annotations

from controllers.base import BaseController, ControllerState


class NaiveSLOController(BaseController):
    """
    SLO-aware controller with no telemetry-age or error-budget awareness.

    Parameters
    ----------
    clocks       : Sorted list of supported GPU clock values (MHz).
    slo_target_s : Latency SLO threshold.
    down_slack   : Fraction of target below which down-clocking is allowed
                   (default 0.80 means P95 < 80 % of target).
    min_dwell    : Minimum control intervals before changing direction.
    """

    def __init__(
        self,
        clocks,
        slo_target_s: float = 2.0,
        down_slack: float = 0.80,
        min_dwell: int = 2,
    ):
        super().__init__(clocks)
        self.slo_target_s = slo_target_s
        self.down_slack = down_slack
        self.min_dwell = min_dwell

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        # Use the SLO target from state if available; fall back to init value.
        target = state.slo_target_s or self.slo_target_s

        # ── Emergency: latency already exceeds target ─────────────────────
        if state.p95_latency_s > target:
            self.last_reason = "EMERGENCY_SLO"
            return min(current_idx + 1, len(self.clocks) - 1)

        # ── Down-clock: large headroom AND empty queue ────────────────────
        if (state.p95_latency_s < self.down_slack * target
                and state.queue_len == 0):
            self.last_reason = "SAFE_DOWNCLOCK"
            return max(current_idx - 1, 0)

        # ── Hold ──────────────────────────────────────────────────────────
        self.last_reason = "HOLD"
        return current_idx

    @property
    def name(self) -> str:
        return "NaiveSLO"


# ---------------------------------------------------------------------------
# Unit tests (run directly: python -m controllers.naive_slo)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from controllers.base import ControllerState

    clocks = [600, 900, 1200, 1600, 2100, 2700, 3090]
    ctrl = NaiveSLOController(clocks, slo_target_s=2.0)
    start_idx = 3  # 1600 MHz

    def _st(**kw) -> ControllerState:
        defaults = dict(
            slo_target_s=2.0, queue_len=0, p95_latency_s=0.0,
            error_budget_burn=0.0, telemetry_age_ms=0.0, arrival_cv=0.0,
        )
        defaults.update(kw)
        return ControllerState(**defaults)

    # 1. Fresh telemetry + healthy budget + large slack → down-clock
    s = _st(p95_latency_s=0.5, queue_len=0)
    assert ctrl._policy(s, start_idx) == start_idx - 1, "Should down-clock"

    # 2. P95 > target → up-clock
    s = _st(p95_latency_s=2.5, queue_len=0)
    assert ctrl._policy(s, start_idx) == start_idx + 1, "Should up-clock"

    # 3. Queue non-zero → hold even with good P95
    s = _st(p95_latency_s=0.5, queue_len=3)
    assert ctrl._policy(s, start_idx) == start_idx, "Should hold (queue)"

    print("NaiveSLOController unit tests PASSED")
