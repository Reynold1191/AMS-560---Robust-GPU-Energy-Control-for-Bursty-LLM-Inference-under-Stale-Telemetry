import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_t: float = field(default_factory=time.monotonic)
    start_t: Optional[float] = None
    finish_t: Optional[float] = None
    generated_tokens: int = 0
    prompt_tokens: int = 0
    phase: str = "steady"
    seed_idx: int = 0

    @property
    def latency_s(self) -> Optional[float]:
        if self.finish_t is None:
            return None
        return self.finish_t - self.arrival_t

    @property
    def queue_wait_s(self) -> Optional[float]:
        if self.start_t is None:
            return None
        return self.start_t - self.arrival_t

    @property
    def service_time_s(self) -> Optional[float]:
        if self.start_t is None or self.finish_t is None:
            return None
        return self.finish_t - self.start_t

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "arrival_t": self.arrival_t,
            "start_t": self.start_t,
            "finish_t": self.finish_t,
            "latency_s": self.latency_s,
            "queue_wait_s": self.queue_wait_s,
            "service_time_s": self.service_time_s,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.generated_tokens,
            "phase": self.phase,
        }


class RequestQueue:
    def __init__(self, maxsize: int = 0):
        self._q = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._total_enqueued = 0
        self._total_dequeued = 0

    def put(self, request: Request) -> None:
        self._q.put(request)
        with self._lock:
            self._total_enqueued += 1

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Request]:
        try:
            req = self._q.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
        with self._lock:
            self._total_dequeued += 1
        return req

    def task_done(self) -> None:
        self._q.task_done()

    @property
    def qsize(self) -> int:
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
