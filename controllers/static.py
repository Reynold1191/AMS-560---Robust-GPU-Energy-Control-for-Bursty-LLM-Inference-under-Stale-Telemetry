"""
controllers/static.py
Static frequency controller – always requests the same clock.

Used for two baselines:
  static_high  – locks the highest supported clock (maximum performance).
  static_low   – locks a low but feasible clock (maximum power savings).
"""

from __future__ import annotations

from controllers.base import BaseController, ControllerState


class StaticController(BaseController):
    """
    Policy: always return the index closest to ``target_mhz``.
    Once locked, the clock never changes.
    """

    def __init__(self, clocks, target_mhz: int):
        super().__init__(clocks)
        self.target_mhz = target_mhz
        self._target_idx = self.find_index(target_mhz)

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        return self._target_idx

    @property
    def name(self) -> str:
        return f"Static({self.target_mhz} MHz)"
