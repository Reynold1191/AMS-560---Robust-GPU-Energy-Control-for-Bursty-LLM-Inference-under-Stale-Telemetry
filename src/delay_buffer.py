"""
delay_buffer.py
A time-ordered ring buffer that delivers stale telemetry samples on demand.

Purpose
-------
Simulate the situation where a GPU-frequency controller acts on an
observation that is already d seconds old (the "stale telemetry" failure
mode described in the project proposal).

Usage
-----
  buf = DelayBuffer()                  # create
  buf.push(sample)                     # called by TelemetrySampler
  sample, age_s = buf.get(delay_s=0.5) # controller reads stale view
"""

from __future__ import annotations

import collections
import random
import threading
import time
from typing import Optional, Tuple

from telemetry import GpuSample


class DelayBuffer:
    """
    Thread-safe FIFO buffer of (push_time, GpuSample) pairs.

    get(delay_s, jitter_s) returns the *most recent* sample whose push
    time is at least delay_s seconds in the past, plus the actual age of
    that sample in seconds.

    If no sample is old enough, the oldest available sample is returned
    (graceful fallback so the controller is never blocked).
    """

    def __init__(self, max_size: int = 50_000):
        self._buf: collections.deque = collections.deque(maxlen=max_size)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Write side (TelemetrySampler thread)                                  #
    # ------------------------------------------------------------------ #

    def push(self, sample: GpuSample) -> None:
        """Append the latest GPU sample (thread-safe)."""
        with self._lock:
            self._buf.append((time.monotonic(), sample))

    # ------------------------------------------------------------------ #
    # Read side (controller thread)                                         #
    # ------------------------------------------------------------------ #

    def get(
        self,
        delay_s: float = 0.0,
        jitter_s: float = 0.0,
    ) -> Tuple[Optional[GpuSample], float]:
        """
        Return the controller's view of the GPU state.

        Parameters
        ----------
        delay_s   : Minimum age of the returned sample (injected delay).
        jitter_s  : ±Uniform noise added to delay_s each call.

        Returns
        -------
        (sample, age_s)
          sample : the stale GpuSample (None if buffer is empty).
          age_s  : how old the returned sample is, in seconds.
        """
        if jitter_s > 0.0:
            delay_s = max(0.0, delay_s + random.uniform(-jitter_s, jitter_s))

        with self._lock:
            if not self._buf:
                return None, 0.0

            now = time.monotonic()

            # Scan from newest to oldest; stop at the first sample that
            # is at least delay_s old.
            result_t: Optional[float] = None
            result_s: Optional[GpuSample] = None
            for push_t, s in reversed(self._buf):
                if (now - push_t) >= delay_s:
                    result_t = push_t
                    result_s = s
                    break

            # Fallback: return the oldest sample in the buffer.
            if result_s is None:
                result_t, result_s = self._buf[0]

        age_s = time.monotonic() - result_t  # type: ignore[operator]
        return result_s, age_s

    def latest(self) -> Tuple[Optional[GpuSample], float]:
        """Return the most recently pushed sample (zero injected delay)."""
        return self.get(delay_s=0.0)

    def size(self) -> int:
        """Number of samples currently in the buffer."""
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        """Discard all buffered samples."""
        with self._lock:
            self._buf.clear()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")

    buf = DelayBuffer()

    # Simulate 10 samples arriving 100 ms apart
    class _FakeSample:
        def __init__(self, v):
            self.power_w = v
            self.graphics_mhz = 3000
            self.sm_mhz = 3000
            self.util_pct = 50
            self.temp_c = 60
            self.phase = "test"
            self.t = time.monotonic()

    print("Pushing 10 fake samples over 1 second …")
    for i in range(10):
        buf.push(_FakeSample(float(i)))
        time.sleep(0.1)

    s, age = buf.get(delay_s=0.0)
    print(f"  Zero delay   → power={s.power_w} age={age*1000:.0f} ms")

    s, age = buf.get(delay_s=0.3)
    print(f"  300 ms delay → power={s.power_w} age={age*1000:.0f} ms (expect ~3 samples old)")

    s, age = buf.get(delay_s=0.9)
    print(f"  900 ms delay → power={s.power_w} age={age*1000:.0f} ms (expect ~9 samples old)")
