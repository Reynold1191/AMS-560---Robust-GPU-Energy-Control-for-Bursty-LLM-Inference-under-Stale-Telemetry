"""
request_queue.py
Request data class and a thread-safe queue connecting the load generator
to the LLM worker.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Request data class
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """
    Represents one LLM inference request with full timing instrumentation.

    Lifecycle
    ---------
      1. LoadGenerator creates Request with arrival_t set.
      2. LLMWorker dequeues it and sets start_t (= end of queue wait).
      3. LLMWorker sets finish_t and generated_tokens when generation ends.
      4. SLOBudget.record() is called with finish_t and latency_s.
    """

    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_t: float = field(default_factory=time.monotonic)

    # Set by LLMWorker
    start_t:         Optional[float] = None
    finish_t:        Optional[float] = None
    generated_tokens: int = 0
    prompt_tokens:   int  = 0

    # Experiment context
    phase:    str = "steady"
    seed_idx: int = 0

    # ------------------------------------------------------------------ #
    # Derived properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def latency_s(self) -> Optional[float]:
        """End-to-end latency: arrival → completion."""
        if self.finish_t is None:
            return None
        return self.finish_t - self.arrival_t

    @property
    def queue_wait_s(self) -> Optional[float]:
        """Time spent waiting in the queue."""
        if self.start_t is None:
            return None
        return self.start_t - self.arrival_t

    @property
    def service_time_s(self) -> Optional[float]:
        """Actual GPU service time (generation only)."""
        if self.start_t is None or self.finish_t is None:
            return None
        return self.finish_t - self.start_t

    @property
    def slo_miss(self) -> Optional[bool]:
        """True if the request missed the SLO (caller must supply target)."""
        return None  # evaluated externally via SLOBudget

    def to_dict(self) -> dict:
        return {
            "request_id":      self.request_id,
            "arrival_t":       self.arrival_t,
            "start_t":         self.start_t,
            "finish_t":        self.finish_t,
            "latency_s":       self.latency_s,
            "queue_wait_s":    self.queue_wait_s,
            "service_time_s":  self.service_time_s,
            "prompt_tokens":   self.prompt_tokens,
            "output_tokens":   self.generated_tokens,
            "phase":           self.phase,
        }


# ---------------------------------------------------------------------------
# Thread-safe request queue
# ---------------------------------------------------------------------------

class RequestQueue:
    """
    Thin wrapper around queue.Queue that exposes queue-length monitoring
    needed by the frequency controller.
    """

    def __init__(self, maxsize: int = 0):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._total_enqueued: int = 0
        self._total_dequeued: int = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Producer side                                                         #
    # ------------------------------------------------------------------ #

    def put(self, request: Request, block: bool = True,
            timeout: Optional[float] = None) -> None:
        """Enqueue a request."""
        self._q.put(request, block=block, timeout=timeout)
        with self._lock:
            self._total_enqueued += 1

    # ------------------------------------------------------------------ #
    # Consumer side                                                         #
    # ------------------------------------------------------------------ #

    def get(self, block: bool = True,
            timeout: Optional[float] = None) -> Optional[Request]:
        """
        Dequeue the next request.
        Returns None if the queue is empty and timeout expires.
        """
        try:
            req = self._q.get(block=block, timeout=timeout)
            with self._lock:
                self._total_dequeued += 1
            return req
        except queue.Empty:
            return None

    def task_done(self) -> None:
        """Signal that a previously dequeued request has been processed."""
        self._q.task_done()

    # ------------------------------------------------------------------ #
    # Monitoring                                                            #
    # ------------------------------------------------------------------ #

    @property
    def qsize(self) -> int:
        """Approximate number of requests currently waiting."""
        return self._q.qsize()

    @property
    def total_enqueued(self) -> int:
        with self._lock:
            return self._total_enqueued

    @property
    def total_dequeued(self) -> int:
        with self._lock:
            return self._total_dequeued

    def empty(self) -> bool:
        return self._q.empty()
