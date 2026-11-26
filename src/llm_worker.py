"""
llm_worker.py
Background worker thread that dequeues requests, runs LLM inference,
and records timing for every completed request.

Design notes
------------
  • Fixed batch size (cfg.max_batch_size) is used throughout the main
    comparison so batching does not confound the controller comparison.
  • Uses HuggingFace Transformers + PyTorch CUDA.
  • After each batch the worker notifies the SLOBudget and the shared
    MetricsCollector.
  • A separate batch-size sensitivity study (run_experiment.py --batch-sweep)
    can vary the batch size independently.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, List, Optional

import torch


import hf_compat  # noqa: F401,E402  (must run before importing transformers)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from request_queue import Request, RequestQueue

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model(model_id: str):
    """
    Load a HuggingFace CausalLM model onto the CUDA device in fp16.
    Returns (model, tokenizer).
    """
    logger.info("Loading model: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Decoder-only models must be LEFT-padded for batched generation; with right
    # padding the shorter prompts in a batch continue from pad tokens.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded on %s", next(model.parameters()).device)
    return model, tokenizer


def count_generated(new_ids, stop_ids, cap: int) -> int:
    """Tokens a request actually produced: through the first stop token, at most cap."""
    n = 0
    for tok in new_ids:
        n += 1
        if tok in stop_ids:
            break
    return min(n, cap)


# ---------------------------------------------------------------------------
# Batch runner (single inference step)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_batch(
    model,
    tokenizer,
    requests: List[Request],
) -> List[Request]:
    """
    Run inference for a batch of requests and fill in timing fields.

    Returns the same list with start_t, finish_t, and generated_tokens set.
    """
    now = time.monotonic()
    for r in requests:
        r.start_t = now

    prompts = [r.prompt for r in requests]

    # Tokenise (pad to the longest prompt in the batch)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,   # long-context prompts are up to ~1.3k tokens
    ).to("cuda")

    input_len = enc["input_ids"].shape[1]

    # Record prompt token count
    for i, r in enumerate(requests):
        # Unpadded length for each row
        mask = enc["attention_mask"][i]
        r.prompt_tokens = int(mask.sum().item())

    max_new = max(r.max_new_tokens for r in requests)

    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    finish = time.monotonic()

    # The whole batch decodes until its longest row finishes, so every row of
    # `out` has the same length. Count each request's real tokens: up to and
    # including its first end/pad token, capped at its own max_new_tokens.
    stop_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id} - {None}
    for i, r in enumerate(requests):
        r.generated_tokens = count_generated(out[i, input_len:].tolist(), stop_ids,
                                             r.max_new_tokens)
        r.finish_t = finish

    return requests


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class LLMWorker:
    """
    Daemon thread that continuously dequeues and processes requests.

    Parameters
    ----------
    model          : Loaded HuggingFace model.
    tokenizer      : Corresponding tokenizer.
    request_queue  : Shared RequestQueue.
    on_complete    : Callback called with each completed Request.
                     Typically updates SLOBudget and MetricsCollector.
    batch_size     : Fixed batch size (1 for the main comparison).
    """

    def __init__(
        self,
        model,
        tokenizer,
        request_queue: RequestQueue,
        on_complete: Optional[Callable[[Request], None]] = None,
        batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.request_queue = request_queue
        self.on_complete = on_complete or (lambda r: None)
        self.batch_size = batch_size

        self._stop_event = threading.Event()
        self._completed: List[Request] = []
        self._lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._run, name="llm-worker", daemon=True
        )

    # ------------------------------------------------------------------ #
    # Public API                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._thread.start()
        logger.info("LLMWorker started (batch_size=%d)", self.batch_size)

    def stop(self) -> None:
        """Signal the worker to drain the queue and stop."""
        self._stop_event.set()
        self._thread.join(timeout=30.0)
        logger.info("LLMWorker stopped (%d requests completed)",
                    len(self._completed))

    @property
    def completed(self) -> List[Request]:
        with self._lock:
            return list(self._completed)

    def total_output_tokens(self) -> int:
        with self._lock:
            return sum(r.generated_tokens for r in self._completed)

    # ------------------------------------------------------------------ #
    # GPU warm-up                                                           #
    # ------------------------------------------------------------------ #

    def warmup(self, n: int = 3) -> None:
        """
        Run a small number of dummy requests to warm up the GPU and
        JIT caches.  Results are discarded.
        """
        logger.info("Warming up GPU (%d dummy requests) …", n)
        dummy = Request(
            request_id=-1,
            prompt="Hello, world!",
            max_new_tokens=16,
            arrival_t=time.monotonic(),
        )
        for _ in range(n):
            run_batch(self.model, self.tokenizer, [dummy])
        torch.cuda.synchronize()
        logger.info("Warm-up complete")

    # ------------------------------------------------------------------ #
    # Thread body                                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        while not self._stop_event.is_set():
            # Collect up to batch_size requests
            batch: List[Request] = []
            req = self.request_queue.get(block=True, timeout=0.1)
            if req is not None:
                batch.append(req)
                # Non-blocking top-up to form a full batch
                while len(batch) < self.batch_size:
                    extra = self.request_queue.get(block=False)
                    if extra is None:
                        break
                    batch.append(extra)

            if not batch:
                continue

            try:
                completed = run_batch(self.model, self.tokenizer, batch)
            except Exception as exc:
                logger.error("run_batch error: %s", exc, exc_info=True)
                continue

            for r in completed:
                self.request_queue.task_done()
                with self._lock:
                    self._completed.append(r)
                try:
                    self.on_complete(r)
                except Exception as exc:
                    logger.warning("on_complete callback error: %s", exc)
