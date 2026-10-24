from typing import List

from .base import BaseController, ControllerState


class StaticController(BaseController):
    """Always holds one clock. target_mhz <= 0 means the highest clock."""

    def __init__(self, clocks: List[int], target_mhz: int = 0):
        super().__init__(clocks)
        self._target_idx = self.find_index(target_mhz) if target_mhz > 0 else len(self.clocks) - 1

    @property
    def target_mhz(self) -> int:
        return self.clocks[self._target_idx]

    def _policy(self, state: ControllerState, current_idx: int) -> int:
        self.last_reason = f"STATIC({self.target_mhz}MHz)"
        return self._target_idx

    @property
    def name(self) -> str:
        return f"Static({self.target_mhz} MHz)"
