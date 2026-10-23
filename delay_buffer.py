import random
import threading
import time
from collections import deque


class DelayBuffer:
    """Ring buffer of (push_time, sample) pairs used to serve stale telemetry."""

    def __init__(self, max_size: int = 50_000):
        self._buf = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def push(self, sample) -> None:
        with self._lock:
            self._buf.append((time.monotonic(), sample))

    def get(self, delay_s: float = 0.0, jitter_s: float = 0.0):
        """Return (sample, age_s) for the newest sample at least delay_s old.

        Falls back to the oldest sample if none is old enough.
        """
        if jitter_s > 0:
            delay_s = max(0.0, delay_s + random.uniform(-jitter_s, jitter_s))

        with self._lock:
            if not self._buf:
                return None, 0.0
            now = time.monotonic()
            for push_t, sample in reversed(self._buf):
                if now - push_t >= delay_s:
                    return sample, now - push_t
            push_t, sample = self._buf[0]
            return sample, now - push_t

    def latest(self):
        return self.get(delay_s=0.0)

    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
