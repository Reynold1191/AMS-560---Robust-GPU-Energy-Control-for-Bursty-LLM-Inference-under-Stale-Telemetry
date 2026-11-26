"""
controllers/robust_delay_budget.py
Robust delay- and error-budget-aware DVFS controller.

Core novelty
------------
This controller treats telemetry freshness (age) and error-budget burn rate
as first-class control signals alongside the conventional queue-length and
P95-latency signals used by the naive baseline.

Risk score
----------
  risk = w_q × q̃ + w_age × ã + w_burn × b̃ + w_cv × cṽ

  q̃   = min(queue_len / 10, 1)
  ã   = min(telemetry_age_ms / age_scale_ms, 1)  ← age normalised to 0–1
        (age_scale_ms = 5000 = 2 x the largest injected delay)
  b̃   = min(error_budget_burn / 3, 1)
  cṽ  = min(arrival_cv / 2, 1)

Default weights: w_q=0.35, w_age=0.25, w_burn=0.25, w_cv=0.15

Policy rules (applied in priority order)
-----------------------------------------
  1. Emergency:   P95 > SLO_target   OR  burn > 2.0
                  → up-clock by +2 steps immediately (bypasses dwell)

  2. High-risk:   risk > 0.65
                  → up-clock by +1 (protect: do not down-clock under
                    high uncertainty or high budget burn)

  3. Safe-to-save: risk < 0.30  AND  P95 < 0.70 × (1 − 0.5 ã) × SLO_target
                  → down-clock by -1, but only once the effect of the last
                    clock change can have shown up in the telemetry:
                    time since change >= telemetry age + clock settling
                    (0.7 s) + one batch time at the current clock.

  4. Hold:        otherwise.

Delay awareness (default)
-------------------------
  A controller reading telemetry that is `age` old cannot see the result of
  a clock change until age + settling + one batch later. Stepping down
  again before then is acting blind; that is how the age-blind baseline
  walks down to an infeasible clock. Older telemetry also gets less trust:
  the P95 headroom needed to step down grows with age.

Age-compensated forecast and lag-aware signals (v2, after attempt_8-11)
-----------------------------------------------------------------------
  On the GPU, v1 and naive_slo made the same clock choices: every up-step
  came from the shared capacity floor, which reads a 10 s arrival rate that
  is itself `age` old. v2 adds, when a capacity table is set:

  0. Forecast:  roll the observed rate and queue forward over the horizon
                h = telemetry age + clock settling, by their observed trends
                (only when the queue is surging: >= 3x its recent median + 2
                and growing). demand = rate + backlog beyond one batch /
                (SLO / 2). Step straight up to the lowest clock covering
                1.25 x demand; on a stale surge of >= half a batch go to the
                top clock, on a smaller one go one step above that.
  1. Lag-aware: P95 and burn are 60 s windows. When the queue shows no
                backlog over the whole blind window (<= half a batch, not
                growing), they are history: no emergency up-step, and a
                capacity-backed down-step is allowed when the next clock
                down covers 1.35 x the recent peak rate.

  Ablations: robust_v1 (forecast and lag-aware off), robust_no_age (h is
  settling only).

Legacy hard guard
-----------------
  stale_downclock_ms (e.g. 300) blocks every down-clock on telemetry older
  than that. In attempt_6 every injected delay was >= 500 ms, so the guard
  froze the controller at its start clock and its "wins" came from never
  moving. It is off by default and kept only for the midterm simulation.

Ablation flags
--------------
  use_telemetry_age=False → treat ã = 0  (remove age contribution)
  use_error_budget=False  → treat b̃ = 0  (remove budget contribution)

Oracle mode
-----------
  When oracle_mode=True, telemetry_age_ms is forced to 0.0 and the
  controller has access to fresh observations.  This provides an upper-
  bound reference (it is NOT deployable, only an evaluation benchmark).
"""

from __future__ import annotations

import csv
import logging
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

from controllers.base import BaseController, ControllerState

logger = logging.getLogger(__name__)


class RobustDelayBudgetController(BaseController):
    """
    Delay- and budget-aware frequency controller.

    Parameters
    ----------
    clocks             : Sorted list of supported GPU clock values (MHz).
    slo_target_s       : Latency SLO threshold (seconds).
    min_dwell          : Minimum intervals before a down-clock is issued.
    use_telemetry_age  : If False, the age term is zeroed (ablation).
    use_error_budget   : If False, the budget term is zeroed (ablation).
    oracle_mode        : If True, assume zero telemetry age (oracle upper bound).
    oracle_table_path  : CSV path with columns [bucket, clock_mhz, p95_s]
                         for lookup-based oracle decisions.
    w_q / w_age / w_burn / w_cv : Weights for the risk score.
    stale_downclock_ms : Never down-clock on telemetry older than this (ms).
                         None disables the guard (diagnostic only).
    """

    def __init__(
        self,
        clocks: List[int],
        slo_target_s: float = 2.0,
        min_dwell: int = 2,
        use_telemetry_age: bool = True,
        use_error_budget: bool = True,
        oracle_mode: bool = False,
        oracle_table_path: Optional[str] = None,
        # Risk weights (tune on calibration workload, then freeze)
        w_q:   float = 0.30,
        w_age: float = 0.35,   # age carries the most weight: stale = high risk
        w_burn: float = 0.20,
        w_cv:  float = 0.15,
        # Thresholds
        high_risk_thresh: float = 0.65,
        low_risk_thresh:  float = 0.30,
        emergency_burn:   float = 2.0,
        down_slack_frac:  float = 0.70,
        stale_downclock_ms: Optional[float] = None,
        age_scale_ms: float = 5000.0,
        delay_aware: bool = True,
        clock_settle_s: float = 0.7,
        forecast: bool = True,
        lag_aware: bool = True,
        trend_span_s: float = 3.0,
        down_margin: float = 1.35,
    ):
        super().__init__(clocks)
        self.forecast = forecast
        self.lag_aware = lag_aware
        self.trend_span_s = trend_span_s
        self.down_margin = down_margin
        self._hist: deque = deque()      # (observation time, arrival_rate, queue_len)
        if stale_downclock_ms is not None and stale_downclock_ms < 0:
            raise ValueError("stale_downclock_ms must be nonnegative or None")
        self.stale_downclock_ms = stale_downclock_ms
        self.age_scale_ms = age_scale_ms
        self.delay_aware = delay_aware
        self.clock_settle_s = clock_settle_s
        self.slo_target_s = slo_target_s
        self.min_dwell = min_dwell
        self.use_telemetry_age = use_telemetry_age
        self.use_error_budget = use_error_budget
        self.oracle_mode = oracle_mode

        self.w_q    = w_q
        self.w_age  = w_age
        self.w_burn = w_burn
        self.w_cv   = w_cv

        self.high_risk_thresh = high_risk_thresh
        self.low_risk_thresh  = low_risk_thresh
        self.emergency_burn   = emergency_burn
        self.down_slack_frac  = down_slack_frac

        # Oracle lookup table: {bucket_key: {clock_mhz: p95_s}}
        self._oracle_table: Optional[Dict] = None
        if oracle_mode and oracle_table_path:
            self._oracle_table = self._load_oracle_table(oracle_table_path)

    # ------------------------------------------------------------------ #
    # Risk calculation                                                      #
    # ------------------------------------------------------------------ #

    def _risk(self, state: ControllerState) -> float:
        """Compute the composite risk score ∈ [0, 1]."""
        q = min(state.queue_len / 10.0, 1.0)

        age = self._age_norm(state)

        burn = (
            min(state.error_budget_burn / 3.0, 1.0)
            if self.use_error_budget else 0.0
        )

        cv = min(state.arrival_cv / 2.0, 1.0)

        return self.w_q * q + self.w_age * age + self.w_burn * burn + self.w_cv * cv

    def _age_norm(self, state: ControllerState) -> float:
        """Telemetry age scaled to 0-1 (0 when the age signal is not used)."""
        if self.oracle_mode or not self.use_telemetry_age:
            return 0.0
        return min(state.telemetry_age_ms / self.age_scale_ms, 1.0)

    def _effect_visible(self, state: ControllerState, current_idx: int) -> bool:
        """
        True once the last clock change can be reflected in what we observe:
        time since the change >= telemetry age + clock settling + one batch.
        The no-age ablation still waits settling + one batch (it just does not
        know the age).
        """
        if not self.delay_aware or self.last_change_t is None:
            return True
        age_s = 0.0
        if self.use_telemetry_age and not self.oracle_mode:
            age_s = state.telemetry_age_ms / 1000.0
        batch_s = self.batch_time_s[current_idx] if self.batch_time_s else 0.0
        return state.t - self.last_change_t >= age_s + self.clock_settle_s + batch_s

    def _is_stale(self, state: ControllerState) -> bool:
        if self.stale_downclock_ms is None or not self.use_telemetry_age or self.oracle_mode:
            return False
        return state.telemetry_age_ms > self.stale_downclock_ms

    # ------------------------------------------------------------------ #
    # Age-compensated load forecast (v2)                                    #
    # ------------------------------------------------------------------ #

    def _record(self, state: ControllerState) -> None:
        """Remember what was observed, stamped with when it was true."""
        obs_t = state.t - state.telemetry_age_ms / 1000.0
        if not self._hist or obs_t > self._hist[-1][0]:
            self._hist.append((obs_t, state.arrival_rate, state.queue_len))
        while self._hist and self._hist[0][0] < obs_t - 15.0:
            self._hist.popleft()

    def _recent(self, span_s: float):
        if not self._hist:
            return []
        t_end = self._hist[-1][0]
        return [h for h in self._hist if h[0] >= t_end - span_s]

    def _slope(self, k: int) -> float:
        """Rise per second of field k over the last trend_span_s of observations."""
        pts = self._recent(self.trend_span_s)
        if len(pts) < 2 or pts[-1][0] - pts[0][0] < 1.0:
            return 0.0
        return (pts[-1][k] - pts[0][k]) / (pts[-1][0] - pts[0][0])

    def _horizon_s(self, state: ControllerState) -> float:
        """
        How far the observation is behind the moment a new clock takes
        effect: telemetry age + clock settling. The no-age ablation only
        knows the settling time.
        """
        age_s = 0.0
        if self.use_telemetry_age and not self.oracle_mode:
            age_s = state.telemetry_age_ms / 1000.0
        return age_s + self.clock_settle_s

    def _batch(self) -> int:
        if self.capacity_rps and self.batch_time_s:
            return max(1, round(self.capacity_rps[-1] * self.batch_time_s[-1]))
        return 16

    def _demand(self, state: ControllerState, target: float) -> float:
        """
        Arrival rate the GPU must serve once a clock chosen now is in effect:
        the observed rate and backlog, rolled forward over the horizon by
        their observed trends, plus the rate needed to clear any backlog
        beyond one batch within half the SLO.
        """
        h = self._horizon_s(state)
        lam, q = state.arrival_rate, float(state.queue_len)
        # Roll forward only when the queue backs the trend up: the 10 s rate
        # estimate wanders by +-20 % on a steady load, and extrapolating that
        # noise over a 3 s horizon would chase it. "Backs it up" = the queue
        # is growing and well above its own recent level (or half a batch).
        if state.queue_len >= self._surge_queue() and self._slope(2) > 0:
            lam += max(0.0, self._slope(1)) * h
            q += max(0.0, self._slope(2)) * h
        excess = max(0.0, q - self._batch())
        return lam + excess / max(0.5 * target, 1.0)

    def _surge_queue(self) -> float:
        """Queue length that counts as a surge: 3x the recent median + 2."""
        qs = sorted(h[2] for h in self._recent(15.0))
        med = qs[len(qs) // 2] if qs else 0
        return min(self._batch() / 2, max(4.0, 3.0 * med + 2.0))

    def _required_idx(self, demand: float) -> int:
        if not self.capacity_rps or demand <= 0:
            return 0
        need = self.capacity_margin * demand
        for i, cap in enumerate(self.capacity_rps):
            if cap >= need:
                return i
        return len(self.clocks) - 1

    def _backlog_free(self, state: ControllerState, current_idx: int) -> bool:
        """
        No sign of a backlog over the whole stretch the last decision could
        not see: queue at most half a batch and not growing.
        """
        span = self._horizon_s(state)
        if self.batch_time_s:
            span += self.batch_time_s[current_idx]
        pts = self._recent(max(span, 2.0))
        if not pts:
            return False
        return max(p[2] for p in pts) <= self._batch() / 2 and self._slope(2) <= 0

    # ------------------------------------------------------------------ #
    # Policy                                                                #
    # ------------------------------------------------------------------ #

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        target = state.slo_target_s or self.slo_target_s

        # ── Oracle mode: use lookup table if available ────────────────────
        if self.oracle_mode and self._oracle_table is not None:
            self.last_reason = "ORACLE_LOOKUP"
            return self._oracle_lookup(state, current_idx)

        risk = self._risk(state)
        v2 = self.forecast and self.capacity_rps is not None
        if v2:
            self._record(state)
            req = self._required_idx(self._demand(state, target))
            # A surge seen through stale telemetry is already older than the
            # observation: go to the top clock and let the capacity-backed
            # down-step bring it back once the queue is clear.
            if (req > current_idx and self._age_norm(state) > 0
                    and state.queue_len >= self._surge_queue() and self._slope(2) > 0):
                req = len(self.clocks) - 1 if state.queue_len >= self._batch() / 2                     else min(req + 1, len(self.clocks) - 1)
            # ── 0. Forecast: the load the new clock will meet is above
            #       what the current clock can serve ────────────────────────
            if req > current_idx:
                self.last_reason = f"FORECAST_UP(r={risk:.2f})"
                return req
            backlog_free = self._backlog_free(state, current_idx)
        else:
            backlog_free = False

        # P95 and burn are 60 s rolling windows: after a burst they stay high
        # for up to a minute although the queue is already empty. A lag-aware
        # controller treats them as history once the queue shows no backlog.
        lagging = v2 and self.lag_aware and backlog_free

        # ── 1. Emergency: latency OR budget critically bad ────────────────
        if not lagging and (state.p95_latency_s > target
                or (self.use_error_budget
                    and state.error_budget_burn > self.emergency_burn)):
            new_idx = min(current_idx + 2, len(self.clocks) - 1)
            self.last_reason = (
                "EMERGENCY_SLO" if state.p95_latency_s > target
                else "EMERGENCY_BUDGET"
            )
            logger.debug(
                "EMERGENCY -> %d MHz  (P95=%.2f budget_burn=%.2f)",
                self.clocks[new_idx], state.p95_latency_s,
                state.error_budget_burn,
            )
            return new_idx

        # ── 1b. Capacity-backed down-step: queue empty over the blind window
        #       and the next clock down covers the recent peak rate with
        #       down_margin headroom ─────────────────────────────────────────
        if lagging and current_idx > 0:
            peak = max(h[1] for h in self._recent(5.0))
            if self.capacity_rps[current_idx - 1] >= self.down_margin * peak:
                if not self._effect_visible(state, current_idx):
                    self.last_reason = f"HOLD(r={risk:.2f}) WAIT_EFFECT"
                    return current_idx
                self.last_reason = f"CAPACITY_DOWN(r={risk:.2f})"
                return current_idx - 1

        # ── 2. High risk: protect, up-clock or hold ───────────────────────
        if risk > self.high_risk_thresh and not lagging:
            new_idx = min(current_idx + 1, len(self.clocks) - 1)
            age_flag = " STALE_OBSERVATION" if self._is_stale(state) else ""
            self.last_reason = f"HIGH_RISK(r={risk:.2f}){age_flag}"
            logger.debug(
                "HIGH RISK=%.2f -> +1 clock (%d MHz)",
                risk, self.clocks[new_idx],
            )
            return new_idx

        # ── 3. Low risk + ample headroom: save energy ─────────────────────
        slack_frac = self.down_slack_frac
        if self.delay_aware:
            slack_frac *= 1.0 - (0.25 if v2 else 0.5) * self._age_norm(state)
        safe_slack = state.p95_latency_s < slack_frac * target
        if risk < self.low_risk_thresh and safe_slack:
            if self._is_stale(state):
                self.last_reason = f"HOLD(r={risk:.2f}) STALE_OBSERVATION"
                return current_idx
            if not self._effect_visible(state, current_idx):
                self.last_reason = f"HOLD(r={risk:.2f}) WAIT_EFFECT"
                return current_idx
            new_idx = max(current_idx - 1, 0)
            self.last_reason = f"SAFE_DOWNCLOCK(r={risk:.2f})"
            logger.debug(
                "LOW RISK=%.2f -> -1 clock (%d MHz)",
                risk, self.clocks[new_idx],
            )
            return new_idx

        # ── 4. Hold ───────────────────────────────────────────────────────
        self.last_reason = f"HOLD(r={risk:.2f})"
        return current_idx

    # ------------------------------------------------------------------ #
    # Oracle lookup                                                         #
    # ------------------------------------------------------------------ #

    def _oracle_lookup(self, state: ControllerState, current_idx: int) -> int:
        """
        Select the lowest clock whose characterised P95 latency
        fits within the SLO target for the current load bucket.
        Falls back to the robust risk-based policy if lookup fails.
        """
        if not self._oracle_table:
            return current_idx

        target = state.slo_target_s or self.slo_target_s
        bucket = self._workload_bucket(state)

        if bucket not in self._oracle_table:
            # Unknown bucket – fall back to risk policy
            return self._policy_no_oracle(state, current_idx)

        clock_latencies = self._oracle_table[bucket]
        best_idx = current_idx
        for idx, mhz in enumerate(self.clocks):
            p95 = clock_latencies.get(mhz)
            if p95 is not None and p95 < target:
                best_idx = idx  # lowest feasible clock seen so far
                break  # clocks sorted ascending; first feasible is minimum energy
        return best_idx

    def _policy_no_oracle(self, state: ControllerState, current_idx: int) -> int:
        """Risk-based policy used as oracle fallback."""
        self.oracle_mode = False
        idx = self._policy(state, current_idx)
        self.oracle_mode = True
        return idx

    @staticmethod
    def _workload_bucket(state: ControllerState) -> str:
        """Coarse bucket key for oracle table lookup."""
        q = "heavy" if state.queue_len > 3 else "light"
        # Oracle table has only steady/burst buckets; warmup/ramp/recovery use steady.
        phase = "burst" if state.phase == "burst" else "steady"
        return f"{phase}_{q}"

    @staticmethod
    def _load_oracle_table(path: str) -> Dict:
        """
        Parse the oracle CSV: columns run_id, clock_mhz, bucket, p95_s.
        Returns {bucket: {clock_mhz: p95_s}}.
        """
        table: Dict = {}
        if not os.path.exists(path):
            logger.warning("Oracle table not found: %s", path)
            return table
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                bucket = row.get("bucket", "steady_light")
                mhz = int(float(row["clock_mhz"]))
                p95 = float(row["p95_s"])
                table.setdefault(bucket, {})[mhz] = p95
        logger.info("Loaded oracle table from %s (%d buckets)", path, len(table))
        return table

    # ------------------------------------------------------------------ #
    # Diagnostics                                                           #
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        parts = ["Robust"]
        if not self.use_telemetry_age:
            parts.append("NoAge")
        if not self.use_error_budget:
            parts.append("NoBudget")
        if self.stale_downclock_ms is None:
            parts.append("NoGuard")
        if self.oracle_mode:
            parts = ["Oracle"]
        return "_".join(parts)


# ---------------------------------------------------------------------------
# Controller acceptance unit tests (run: python -m controllers.robust_delay_budget)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from controllers.base import ControllerState

    clocks = [600, 900, 1200, 1600, 2100, 2700, 3090]
    ctrl = RobustDelayBudgetController(
        clocks, slo_target_s=2.0, min_dwell=2
    )
    mid = 3  # 1600 MHz

    def _st(**kw) -> ControllerState:
        d = dict(
            slo_target_s=2.0, queue_len=0, p95_latency_s=0.0,
            error_budget_burn=0.0, telemetry_age_ms=0.0,
            arrival_cv=0.0, phase="steady",
        )
        d.update(kw)
        return ControllerState(**d)

    # ── Test 1: Fresh + healthy budget + large slack → down-clock ─────────
    s = _st(p95_latency_s=0.5, queue_len=0,
            telemetry_age_ms=0.0, error_budget_burn=0.1, arrival_cv=0.2)
    result = ctrl._policy(s, mid)
    assert result < mid, f"Expected down-clock, got idx={result}"
    print(f"T1 PASS: fresh+healthy → down-clock to {clocks[result]} MHz")

    # ── Test 2: OLD telemetry → no second down-step before the effect shows ─
    ctrl.last_change_t = 100.0
    s = _st(t=101.0, p95_latency_s=0.5, queue_len=0,
            telemetry_age_ms=2500.0, error_budget_burn=0.1, arrival_cv=0.2)
    result = ctrl._policy(s, mid)
    assert result == mid and "WAIT_EFFECT" in ctrl.last_reason, ctrl.last_reason
    s = _st(t=104.0, p95_latency_s=0.5, queue_len=0,
            telemetry_age_ms=2500.0, error_budget_burn=0.1, arrival_cv=0.2)
    assert ctrl._policy(s, mid) < mid, ctrl.last_reason
    ctrl.last_change_t = None
    print("T2 PASS: stale telemetry → waits age + settling before stepping down")

    # ── Test 3: Fast budget burn → conservative even with good P95 ────────
    s = _st(p95_latency_s=1.2, queue_len=0,
            telemetry_age_ms=0.0, error_budget_burn=2.5, arrival_cv=0.2)
    result = ctrl._policy(s, mid)
    assert result > mid, f"Expected up-clock (burn emergency), got idx={result}"
    print(f"T3 PASS: budget emergency → up-clock to {clocks[result]} MHz")

    # ── Test 4: Emergency SLO violation → up-clock ────────────────────────
    s = _st(p95_latency_s=2.5, queue_len=5,
            telemetry_age_ms=0.0, error_budget_burn=1.5, arrival_cv=0.5)
    result = ctrl._policy(s, mid)
    assert result > mid, f"Expected emergency up-clock, got idx={result}"
    print(f"T4 PASS: SLO violation → emergency up-clock to {clocks[result]} MHz")

    # ── Test 5: Ablation – no age term, stale observation → may down-clock ─
    ctrl_noage = RobustDelayBudgetController(
        clocks, slo_target_s=2.0, use_telemetry_age=False
    )
    s = _st(p95_latency_s=0.5, queue_len=0,
            telemetry_age_ms=900.0, error_budget_burn=0.1, arrival_cv=0.1)
    result_full = ctrl._policy(s, mid)
    result_noage = ctrl_noage._policy(s, mid)
    print(
        f"T5 (ablation): full={clocks[result_full]} MHz  "
        f"no-age={clocks[result_noage]} MHz – "
        f"age matters: {result_noage < result_full or result_noage == mid}"
    )

    # ── Test 6: legacy 300 ms guard still available for the midterm sim ───
    s = _st(p95_latency_s=0.5, queue_len=0,
            telemetry_age_ms=500.0, error_budget_burn=0.1, arrival_cv=0.2)
    legacy = RobustDelayBudgetController(clocks, slo_target_s=2.0, stale_downclock_ms=300.0,
                                         age_scale_ms=1000.0, delay_aware=False)
    result = legacy._policy(s, mid)
    assert result == mid and "STALE_OBSERVATION" in legacy.last_reason, legacy.last_reason
    print(f"T6 PASS: legacy guard holds at {clocks[result]} MHz on 500 ms telemetry")

    print("\nAll acceptance tests PASSED")
