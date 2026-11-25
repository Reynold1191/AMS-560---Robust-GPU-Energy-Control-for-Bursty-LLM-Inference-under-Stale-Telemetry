"""
bench_clock_sensitivity.py
Does the GPU clock actually change LLM decode speed?  Pick a model / batch
size where it does, before spending hours on the experiment matrix.

The first sweep (attempt_5) showed batch service time flat or even rising
with clock while GPU utilisation fell from 65 % to 12 %: with a 0.5B model
at batch 8, decode is bound by CPU-side kernel launches, so the GPU clock
has almost nothing to trade.  This script locks a few clocks and times one full
batch (prefill + decode, answers forced to the batch's token budget) for
each candidate (model, batch) pair on the real workload prompts
(default prompts/long_context.jsonl: long prompts, short answers).

Choice rule (fixed before looking at controller results):
  sensitivity = batch time at the lowest clock / at the top clock
  usable      = sensitivity >= 1.5 and batch time at the top clock
                <= --max-service-s
  chosen      = usable candidate with the highest sensitivity
  base_rate   = 30 % of the top-clock capacity (batch / service time),
                so the 4x burst overloads every clock and lower clocks
                fall behind sooner.

Run from an Administrator terminal (clock locking):
    python bench_clock_sensitivity.py --out results/attempt_6/clock_sensitivity.json
Exit code 1 if no candidate is usable.  The clock is reset on exit.
"""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import VERIFIED_CLOCKS_MHZ  # noqa: E402
from gpu_clock import reset_graphics_clock, set_graphics_clock  # noqa: E402

DEFAULT_MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]
DEFAULT_BATCHES = [4, 8, 16]
MIN_SENSITIVITY = 1.5
LOAD_FRACTION = 0.3


class _PowerSampler:
    """Mean power / utilisation over a timed window (NVML, 50 ms poll)."""

    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._stop = threading.Event()
        self.samples = []

    def __enter__(self):
        def run():
            while not self._stop.wait(0.05):
                nv, h = self._nv, self._h
                self.samples.append((nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
                                     nv.nvmlDeviceGetUtilizationRates(h).gpu,
                                     nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_GRAPHICS)))
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def mean(self, i):
        return statistics.mean(s[i] for s in self.samples) if self.samples else float("nan")


def pick(candidates: list, max_service_s: float) -> dict | None:
    """Apply the choice rule to measured candidates (pure, unit-tested)."""
    usable = [c for c in candidates
              if c["sensitivity"] >= MIN_SENSITIVITY and c["service_top_s"] <= max_service_s]
    if not usable:
        return None
    best = max(usable, key=lambda c: (c["sensitivity"], -c["batch_size"]))
    rate = LOAD_FRACTION * best["batch_size"] / best["service_top_s"]
    return {**best, "base_rate": max(0.05, math.floor(rate * 20) / 20)}


def bench(models, batches, clocks, reps, prompts_file) -> list:
    """
    Time one full batch (prefill + decode) of real workload prompts at each
    clock. Every row is forced to the batch's answer budget so runs are
    comparable across clocks.
    """
    import torch
    from llm_worker import load_model
    from load_generator import load_prompts

    records = load_prompts(prompts_file)
    results = []
    for model_id in models:
        model, tok = load_model(model_id)
        for b in batches:
            batch = [records[i % len(records)] for i in range(b)]
            enc = tok([r["prompt"] for r in batch], return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to("cuda")
            new = max(int(r.get("max_new_tokens", 32)) for r in batch)
            gen = dict(max_new_tokens=new, min_new_tokens=new, do_sample=False,
                       pad_token_id=tok.pad_token_id)
            row = {"model": model_id, "batch_size": b, "prompt_tokens": int(enc["input_ids"].shape[1]),
                   "new_tokens": new, "per_clock": {}}
            try:
                for mhz in clocks:
                    if set_graphics_clock(mhz) != 0:
                        raise RuntimeError("clock lock failed: run as Administrator")
                    time.sleep(1.5)  # measured settling ~0.7 s clock, ~1.4 s power
                    with torch.inference_mode():
                        model.generate(**enc, **gen)  # warm-up at this clock
                        torch.cuda.synchronize()
                        times = []
                        with _PowerSampler() as ps:
                            for _ in range(reps):
                                t0 = time.perf_counter()
                                model.generate(**enc, **gen)
                                torch.cuda.synchronize()
                                times.append(time.perf_counter() - t0)
                    t = statistics.median(times)
                    row["per_clock"][mhz] = {"batch_time_s": t, "power_w": ps.mean(0),
                                             "util_pct": ps.mean(1), "clock_mhz": ps.mean(2),
                                             "energy_j": t * ps.mean(0)}
                    print(f"  {model_id.split('/')[-1]:<22} batch {b:>2} @ {mhz:>4} MHz: "
                          f"{t:6.2f} s/batch  util {ps.mean(1):3.0f}%  {ps.mean(0):5.1f} W  "
                          f"{t * ps.mean(0):6.1f} J  (observed {ps.mean(2):.0f} MHz)", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  {model_id} batch {b}: out of GPU memory, skipped")
                torch.cuda.empty_cache()
                continue
            lo, hi = row["per_clock"][clocks[0]], row["per_clock"][clocks[-1]]
            row["sensitivity"] = lo["batch_time_s"] / hi["batch_time_s"]
            row["service_top_s"] = hi["batch_time_s"]
            results.append(row)
        del model
        torch.cuda.empty_cache()
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Clock sensitivity of LLM decode")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    p.add_argument("--clocks", type=int, nargs="+",
                   default=[VERIFIED_CLOCKS_MHZ[0], VERIFIED_CLOCKS_MHZ[len(VERIFIED_CLOCKS_MHZ) // 2],
                            VERIFIED_CLOCKS_MHZ[-1]])
    p.add_argument("--prompts-file", default=os.path.join(HERE, "prompts", "long_context.jsonl"))
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--max-service-s", type=float, default=8.0,
                   help="Reject setups whose top-clock batch time exceeds this")
    p.add_argument("--out", default=os.path.join("results", "clock_sensitivity.json"))
    args = p.parse_args(argv)

    clocks = sorted(args.clocks)

    atexit.register(reset_graphics_clock)
    try:
        results = bench(args.models, args.batches, clocks, args.reps, args.prompts_file)
    finally:
        reset_graphics_clock()

    chosen = pick(results, args.max_service_s)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rule": f"sensitivity >= {MIN_SENSITIVITY}, top-clock batch time <= "
                           f"{args.max_service_s} s, highest sensitivity wins; base_rate = "
                           f"{LOAD_FRACTION} x top-clock capacity",
                   "clocks": clocks, "prompts_file": args.prompts_file,
                   "candidates": results, "chosen": chosen}, f, indent=2)

    print(f"\n  {'model':<22} {'batch':>5} {'sens':>5} {'top svc':>8}")
    for r in results:
        print(f"  {r['model'].split('/')[-1]:<22} {r['batch_size']:>5} "
              f"{r['sensitivity']:>5.2f} {r['service_top_s']:>7.2f}s")
    if chosen is None:
        print(f"\nNo candidate reaches {MIN_SENSITIVITY}x clock sensitivity. Try a larger "
              "model (--models Qwen/Qwen2.5-3B-Instruct) or larger --batches.")
        return 1
    print(f"\nChosen: {chosen['model']}  batch {chosen['batch_size']}  "
          f"sensitivity {chosen['sensitivity']:.2f}x  base rate {chosen['base_rate']} req/s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
