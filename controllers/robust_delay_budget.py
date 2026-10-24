"""Telemetry-age and error-budget-aware policy, ablations, and oracle (Member 2)."""

import logging
import math

from .base import BaseController, ControllerState

logger = logging.getLogger(__name__)


class RobustDelayBudgetController(BaseController):
    def __init__(
        self,
        clocks,
        slo_target_s=8.5,
        min_dwell=2,
        use_telemetry_age=True,
        use_error_budget=True,
        oracle_mode=False,
        oracle_table_path=None,
        w_q=0.30,
        w_age=0.35,
        w_burn=0.20,
        w_cv=0.15,
        high_risk_thresh=0.55,
        low_risk_thresh=0.40,
        age_threshold_s=1.0,
        max_queue_norm=10,
        emergency_burn=2.0,
        down_slack_frac=0.80,
        stale_downclock_ms=300.0,
        workload=None,
    ):
        super().__init__(clocks)
        if age_threshold_s <= 0 or max_queue_norm <= 0:
            raise ValueError("risk normalization scales must be positive")
        weights = (w_q, w_age, w_burn, w_cv)
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("risk weights must be finite and nonnegative")
        if not 0 <= low_risk_thresh <= high_risk_thresh <= 1:
            raise ValueError("risk thresholds must satisfy 0 <= low <= high <= 1")
        if stale_downclock_ms < 0:
            raise ValueError("stale_downclock_ms must be nonnegative")
        self.slo_target_s = slo_target_s
        self.min_dwell = min_dwell
        self.use_telemetry_age = use_telemetry_age
        self.use_error_budget = use_error_budget
        self.oracle_mode = oracle_mode
        self.workload = workload
        self.w_q, self.w_age, self.w_burn, self.w_cv = weights
        self.high_risk_thresh = high_risk_thresh
        self.low_risk_thresh = low_risk_thresh
        self.age_threshold_s = age_threshold_s
        self.max_queue_norm = max_queue_norm
        self.emergency_burn = emergency_burn
        self.down_slack_frac = down_slack_frac
        self.stale_downclock_ms = stale_downclock_ms
        self.last_risk = 0.0
        self.oracle_table = None
        if oracle_table_path is not None:
            try:
                import pandas as pd

                self.oracle_table = pd.read_csv(oracle_table_path)
            except (ImportError, OSError, ValueError) as exc:
                logger.warning("Cannot load oracle table %s: %s; using robust policy", oracle_table_path, exc)

    @staticmethod
    def _unit(value: float) -> float:
        return max(0.0, min(value, 1.0))

    def _risk(self, state: ControllerState) -> float:
        norm_queue = self._unit(state.queue_len / self.max_queue_norm)
        norm_age = (
            self._unit(state.telemetry_age_ms / 1000.0 / self.age_threshold_s)
            if self.use_telemetry_age else 0.0
        )
        norm_burn = self._unit(state.error_budget_burn) if self.use_error_budget else 0.0
        norm_cv = self._unit(state.arrival_cv)
        return self._unit(
            self.w_q * norm_queue
            + self.w_age * norm_age
            + self.w_burn * norm_burn
            + self.w_cv * norm_cv
        )

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        self.last_risk = self._risk(state)
        if self.oracle_mode and self.oracle_table is not None:
            return self._oracle_lookup(state, current_idx)
        return self._policy_robust(state, current_idx)

    def _policy_robust(self, state: ControllerState, current_idx: int) -> int:
        risk = self.last_risk = self._risk(state)
        slo_target = self.slo_target_s
        max_idx = len(self.clocks) - 1
        if state.p95_latency_s > slo_target:
            self.last_reason = "EMERGENCY_SLO"
            return min(current_idx + 2, max_idx)
        if self.use_error_budget and state.error_budget_burn > self.emergency_burn:
            self.last_reason = "EMERGENCY_BUDGET"
            return min(current_idx + 2, max_idx)
        if risk > self.high_risk_thresh:
            self.last_reason = f"HIGH_RISK(r={risk:.2f})"
            if self.use_telemetry_age and state.telemetry_age_ms > self.stale_downclock_ms:
                self.last_reason += " STALE_OBSERVATION"
            return min(current_idx + 1, max_idx)
        if risk < self.low_risk_thresh and state.p95_latency_s < self.down_slack_frac * slo_target:
            # At 600 ms, age alone contributes 0.21 < 0.40. Keep the specified
            # risk formula, but veto reductions on stale data to satisfy T4.
            if self.use_telemetry_age and state.telemetry_age_ms > self.stale_downclock_ms:
                self.last_reason = f"HOLD(r={risk:.2f}) STALE_OBSERVATION"
                return current_idx
            self.last_reason = f"SAFE_DOWNCLOCK(r={risk:.2f})"
            return max(current_idx - 1, 0)
        self.last_reason = f"HOLD(r={risk:.2f})"
        return current_idx

    def _oracle_lookup(self, state: ControllerState, current_idx: int) -> int:
        """Accept phase tables and Member 3's workload_bucket CSV schema."""
        try:
            table = self.oracle_table
            if "phase" in table.columns:
                rows = table[table["phase"] == state.phase]
            else:
                bucket_column = "workload_bucket" if "workload_bucket" in table.columns else "workload"
                rows = table[table[bucket_column] == (self.workload or state.phase)]
            mhz = float(rows.iloc[0]["min_clock_mhz"])
            if not math.isfinite(mhz) or mhz <= 0:
                raise ValueError("oracle frequency must be finite and positive")
            # A table frequency is a minimum: round up to a supported clock.
            candidates = [idx for idx, clock in enumerate(self.clocks) if clock >= mhz]
            if not candidates:
                raise ValueError("oracle frequency exceeds the available clocks")
            target_idx = candidates[0]
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return self._policy_robust(state, current_idx)
        self.last_reason = f"ORACLE({self.clocks[target_idx]}MHz)"
        return target_idx

    @property
    def name(self) -> str:
        if self.oracle_mode:
            return "Oracle"
        if not self.use_telemetry_age and not self.use_error_budget:
            return "Robust_NoAge_NoBudget"
        if not self.use_telemetry_age:
            return "Robust_NoAge"
        if not self.use_error_budget:
            return "Robust_NoBudget"
        return "Robust"
