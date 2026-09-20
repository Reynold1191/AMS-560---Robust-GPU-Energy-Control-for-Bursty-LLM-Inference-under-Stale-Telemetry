# AMS 560 — Team Assignment Plan
**Project:** Robust GPU Energy Control for LLM Inference Under Stale Telemetry
**Team:** 5 members · 4 Mac + 1 RTX Windows
**Goal:** Implement from scratch so the combined result runs the full 24-condition
experiment (attempt_3) and produces 8 analysis figures.

> **Read this before you start:** You are implementing a system where a controller
> adjusts GPU clock frequency in real time to minimize energy while keeping
> inference latency below a target (the SLO). The core research question is:
> *does explicitly accounting for stale telemetry (delayed sensor readings)
> improve energy efficiency without hurting reliability?*

---

## Who Owns What

| Member | Role | Files | Machine |
|--------|------|-------|---------|
| 1 | Data Layer | `config.py`, `request_queue.py`, `delay_buffer.py`, `slo_budget.py`, `prompts/prompts.jsonl` | Mac |
| 2 | Controller Logic | `controllers/` (all 5 files) | Mac |
| 3 | Metrics & Analysis | `metrics.py`, `analyze_results.py`, `oracle_builder.py` | Mac |
| 4 | Load Generation & Scale Sim | `load_generator.py`, `scale_simulator.py` | Mac |
| 5 | GPU + LLM + Orchestration | `gpu_clock.py`, `telemetry.py`, `measure_timing.py`, `llm_worker.py`, `run_experiment.py`, `run_demo.py`, `run_all.ps1`, `setup.ps1` | **RTX + Windows only** |

## Environment Setup

**Mac (Members 1–4):**
```bash
conda create -n ams560 python=3.10 && conda activate ams560
pip install pandas matplotlib numpy scipy pytest
```

**Windows RTX (Member 5):**
```powershell
conda create -n general python=3.10 && conda activate general
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers accelerate nvidia-ml-py pandas matplotlib scipy pytest
nvidia-smi                        # must succeed
python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

## Merge Order

**Step 1:** Member 1 pushes (everyone depends on `config.py`)
**Step 2:** Members 2, 3, 4 work in parallel (all depend on Member 1)
**Step 3:** Member 5 pulls all branches, integrates, runs 24 conditions

---

---

# Member 1 — Data Layer

**Machine:** Mac  
**Complexity:** Low  
**Files:** `config.py`, `request_queue.py`, `delay_buffer.py`, `slo_budget.py`, `prompts/prompts.jsonl`

## What you are building

You define the shared data types and parameters that every other module imports.
No GPU needed. No third-party libraries beyond Python stdlib.
Push `config.py` as soon as it is working — the entire team is blocked on it.

---

### `config.py`

**Purpose:** A single Python dataclass holding every tunable parameter for the experiment.
All other scripts import `ExperimentConfig` instead of using magic numbers.

**Implement one dataclass `ExperimentConfig` with these fields and their defaults:**

| Field | Type | Default | What it controls |
|-------|------|---------|-----------------|
| `slo_target_s` | float | 8.5 | Latency threshold in seconds; a request that takes longer than this is a "miss" |
| `slo_budget_ratio` | float | 0.01 | Maximum allowed fraction of requests that may miss (1 % = 99 % SLO) |
| `budget_window_s` | float | 60.0 | Length of the rolling window for SLO accounting |
| `telemetry_poll_s` | float | 0.1 | How often the background thread reads GPU power/clock (seconds) |
| `delay_ms` | float | 0.0 | Artificial staleness injected into telemetry (milliseconds) |
| `delay_jitter_ms` | float | 0.0 | Random uniform jitter added on top of `delay_ms` |
| `control_interval_s` | float | 1.0 | How often the frequency controller runs (seconds) |
| `min_dwell_intervals` | int | 2 | Minimum control cycles before the clock may change direction |
| `controller` | str | "robust" | Which controller: `naive_slo`, `robust`, `static_efficient`, etc. |
| `use_telemetry_age` | bool | True | Ablation flag: set False to remove the age term from the robust policy |
| `use_error_budget` | bool | True | Ablation flag: set False to remove the budget term |
| `workload` | str | "sharp_burst" | One of `steady`, `ramp`, `sharp_burst` |
| `base_arrival_rate` | float | 0.15 | Requests per second during normal (non-burst) periods |
| `burst_multiplier` | float | 5.0 | Multiply base rate by this during a burst |
| `burst_start_s` | float | 40.0 | Seconds into the experiment when the burst starts |
| `burst_duration_s` | float | 20.0 | How long the burst lasts |
| `ramp_end_rate` | float | 2.0 | Target arrival rate at the end of a ramp workload |
| `duration_s` | float | 120.0 | Total experiment duration in seconds |
| `warmup_s` | float | 10.0 | Warm-up seconds excluded from statistics |
| `seed` | int | 42 | RNG seed for reproducible request generation |
| `model_id` | str | "Qwen/Qwen2.5-0.5B-Instruct" | HuggingFace model name |
| `max_batch_size` | int | 4 | GPU inference batch size |
| `prompts_file` | str | "prompts/prompts.jsonl" | Path to the prompt dataset |
| `selected_clocks` | List[int] | [] | Filled at runtime by Member 5's clock enumeration |
| `static_efficient_mhz` | int | 667 | Best energy-efficiency clock found in the sweep |
| `static_low_mhz` | int | 0 | Lowest static clock (filled at runtime) |
| `sim_num_workers` | int | 8 | Number of virtual workers in the scale-out simulation |
| `sim_trace_requests` | int | 10000 | Number of requests in the scale simulation |
| `run_id` | str | `f"run_{int(time.time())}"` | Unique identifier for a run (auto-generated) |
| `results_dir` | str | "results" | Directory for CSV files and figures |
| `oracle_table_path` | Optional[str] | None | Path to the oracle lookup CSV |

**Also add two helper methods:**
- `results_path(filename)` — returns `os.path.join(self.results_dir, filename)`
- `to_dict()` — uses `dataclasses.asdict(self)` to return a plain dict for logging

**Also add these module-level constants** (used by scripts to build the experiment matrix):
```
DELAY_VALUES_MS = [0, 100, 250, 500, 1000]
CONTROLLER_NAMES = ["default", "static_high", "static_low", "static_efficient",
                    "naive_slo", "robust", "robust_no_age", "robust_no_budget", "oracle"]
WORKLOAD_NAMES = ["steady", "ramp", "sharp_burst"]
```

**Tests:**
```bash
# 1. Defaults load correctly
python -c "from config import ExperimentConfig; c = ExperimentConfig(); assert c.slo_target_s == 8.5; assert c.burst_multiplier == 5.0; print('PASS defaults')"

# 2. Overrides work
python -c "from config import ExperimentConfig; c = ExperimentConfig(delay_ms=250, controller='naive_slo'); assert c.delay_ms == 250; assert c.controller == 'naive_slo'; print('PASS overrides')"

# 3. to_dict returns all fields
python -c "from config import ExperimentConfig; d = ExperimentConfig().to_dict(); assert 'slo_target_s' in d and 'burst_multiplier' in d; print('PASS to_dict')"

# 4. results_path joins correctly
python -c "from config import ExperimentConfig; c = ExperimentConfig(results_dir='results/attempt_3'); assert c.results_path('foo.csv').endswith('foo.csv'); print('PASS results_path')"
```

---

### `request_queue.py`

**Purpose:** Defines what one LLM inference job looks like (`Request`) and
a thread-safe queue for passing jobs between the load generator and the worker.

**Implement dataclass `Request` with these fields:**
- `request_id: int` — unique integer ID
- `prompt: str` — the text prompt
- `max_new_tokens: int` — how many tokens to generate
- `arrival_t: float` — `time.monotonic()` when the request was created (default_factory)
- `start_t: Optional[float]` — set by the worker when it picks up the request (default None)
- `finish_t: Optional[float]` — set by the worker when generation completes (default None)
- `generated_tokens: int` — number of tokens produced (default 0)
- `prompt_tokens: int` — number of prompt tokens (default 0)
- `phase: str` — experiment phase tag, e.g. "warmup", "steady", "burst" (default "steady")
- `seed_idx: int` — which prompt was used (default 0)

**Add three computed properties** (return `None` if the needed timestamps are missing):
- `latency_s` — `finish_t - arrival_t` (end-to-end latency)
- `queue_wait_s` — `start_t - arrival_t` (time spent waiting in the queue)
- `service_time_s` — `finish_t - start_t` (actual GPU inference time)

**Add method `to_dict()`** that returns a plain dict with all fields plus the three
computed properties. Use key names: `request_id`, `arrival_t`, `start_t`, `finish_t`,
`latency_s`, `queue_wait_s`, `service_time_s`, `prompt_tokens`, `output_tokens`, `phase`.

**Implement class `RequestQueue` (wraps `queue.Queue`)** with:
- `__init__(self, maxsize=0)` — create internal `queue.Queue`; counters for total enqueued/dequeued
- `put(self, request)` — enqueue a request; increment `_total_enqueued`
- `get(self, block=True, timeout=None)` — dequeue and return a request; return `None` on timeout; increment `_total_dequeued`
- `task_done(self)` — forward to the underlying queue
- Property `qsize` — current number of waiting requests
- Property `total_enqueued` — how many have been put in since creation
- Method `empty()` — True if nothing is waiting

Use a `threading.Lock()` to protect the counter updates.

**Tests:**
```bash
# 1. Timing properties
python -c "
import time
from request_queue import Request
r = Request(1, 'hello', 20)
r.start_t = r.arrival_t + 0.5
r.finish_t = r.arrival_t + 1.5
assert abs(r.latency_s - 1.5) < 0.01
assert abs(r.queue_wait_s - 0.5) < 0.01
assert abs(r.service_time_s - 1.0) < 0.01
print('PASS timing properties')
"

# 2. None when not yet finished
python -c "
from request_queue import Request
r = Request(1, 'hello', 20)
assert r.latency_s is None
assert r.queue_wait_s is None
print('PASS None when incomplete')
"

# 3. Queue ordering (FIFO)
python -c "
from request_queue import Request, RequestQueue
q = RequestQueue()
q.put(Request(1, 'a', 10))
q.put(Request(2, 'b', 10))
q.put(Request(3, 'c', 10))
ids = [q.get(timeout=1).request_id for _ in range(3)]
assert ids == [1, 2, 3], f'Expected [1,2,3] got {ids}'
assert q.total_enqueued == 3
print('PASS queue FIFO and counter')
"

# 4. get() returns None on empty timeout
python -c "
from request_queue import RequestQueue
q = RequestQueue()
r = q.get(block=True, timeout=0.1)
assert r is None
print('PASS timeout returns None')
"
```

---

### `delay_buffer.py`

**Purpose:** A thread-safe ring buffer of `(push_time, GpuSample)` pairs.
When the controller calls `get(delay_s=0.5)`, it gets back the most recent
sample that is at least 0.5 seconds old — simulating stale telemetry.

**Implement class `DelayBuffer`:**

- `__init__(self, max_size=50_000)` — create a `collections.deque(maxlen=max_size)` and a `threading.Lock()`

- `push(self, sample)` — acquire lock; append `(time.monotonic(), sample)` to the deque

- `get(self, delay_s=0.0, jitter_s=0.0)` — the core method:
  1. If `jitter_s > 0`, add a random uniform offset `±jitter_s` to `delay_s` (use `random.uniform`)
  2. If buffer is empty, return `(None, 0.0)`
  3. Record `now = time.monotonic()`
  4. Iterate from **newest to oldest** (i.e., `reversed(self._buf)`)
  5. Return the first entry whose `push_time` satisfies `(now - push_time) >= delay_s`
  6. If no entry is old enough, return the **oldest** entry as a fallback (graceful degradation)
  7. Return `(sample, now - push_time)` — the second value is the actual age in seconds

- `latest(self)` — convenience method; calls `get(delay_s=0.0)`

- `size(self)` — returns current buffer length (under lock)

- `clear(self)` — empties the buffer (under lock)

> **Note:** `GpuSample` is defined by Member 5 in `telemetry.py`.
> For local testing, you can use any Python object with a `power_w` attribute.

**Tests:**
```bash
python -c "
import time
from delay_buffer import DelayBuffer

class FakeSample:
    def __init__(self, v): self.power_w = v

buf = DelayBuffer()
buf.push(FakeSample(1.0))
time.sleep(0.35)
buf.push(FakeSample(2.0))

# Zero delay: should get newest (power=2.0)
s, age = buf.get(delay_s=0.0)
assert s.power_w == 2.0, f'Expected 2.0 got {s.power_w}'
assert age < 0.1, f'Expected fresh, got age={age:.3f}s'
print('PASS zero delay returns newest')

# 250ms delay: should get the older sample (power=1.0)
s, age = buf.get(delay_s=0.25)
assert s.power_w == 1.0, f'Expected 1.0 got {s.power_w}'
assert age > 0.3, f'Expected stale, got age={age:.3f}s'
print('PASS 250ms delay returns stale sample')

# Large delay: fallback to oldest
s, age = buf.get(delay_s=999.0)
assert s.power_w == 1.0
print('PASS fallback to oldest when no sample old enough')

print('ALL PASSED')
"
```

---

### `slo_budget.py`

**Purpose:** Tracks a rolling window of completed request latencies and exposes
reliability metrics that the controller uses to decide whether to up-clock.

**Implement helper class `_RollingWindow`:**
- Stores `(timestamp, value)` tuples in a `collections.deque`
- `append(timestamp, value)` — add a new entry, then trim entries older than `window_s`
- `snapshot(now=None)` — return just the values for entries still within the window
- `_trim(now)` — pop from the left while `buf[0][0] < now - window_s`
- `__len__` — return current deque length (under lock)

**Implement main class `SLOBudget(slo_target_s, budget_ratio, window_s)`:**

- `record(self, finish_t, latency_s)` — record one completed request:
  - Always append to `_latencies` window
  - If `latency_s > slo_target_s`, also append to `_misses` window

- `count(self)` — number of requests in the rolling window (`len(_latencies)`)

- `miss_count(self)` — number of SLO misses in the window (`len(_misses)`)

- `miss_rate(self)` — `miss_count / count`, or 0.0 if count is 0

- `burn_rate(self)` — `miss_rate / budget_ratio`; a burn rate of 1.0 means consuming the budget
  at exactly the allowed speed; > 1.0 means degrading reliability

- `remaining_budget(self)` — `1.0 - (miss_rate / budget_ratio)`; can go negative

- `percentile_latency(self, p)` — sort the latencies snapshot; return the value at index `int(p * n)`,
  clamped to `n-1`. Return 0.0 if empty.

- `p95(self)` — calls `percentile_latency(0.95)`

- `p99(self)` — calls `percentile_latency(0.99)`

- `mean_latency(self)` — arithmetic mean of the latency window; 0.0 if empty

- `summary(self)` — return a dict with keys: `window_count`, `miss_rate`, `burn_rate`,
  `p95_latency_s`, `p99_latency_s`, `mean_latency_s`

**Tests:**
```bash
python -c "
import time
from slo_budget import SLOBudget

b = SLOBudget(slo_target_s=5.0, budget_ratio=0.01, window_s=60.0)
now = time.monotonic()

# Record 4 requests: latencies [3, 3, 8, 8] -> 2 misses / 4 total = 50% miss
for i, lat in enumerate([3.0, 3.0, 8.0, 8.0]):
    b.record(now + i, lat)

assert b.count() == 4
assert b.miss_count() == 2
assert abs(b.miss_rate() - 0.5) < 0.001, f'Expected 0.5 got {b.miss_rate()}'
assert abs(b.burn_rate() - 50.0) < 0.1, f'Expected 50.0 got {b.burn_rate()}'
print('PASS miss_rate and burn_rate')

assert b.p95() == 8.0
assert b.p99() == 8.0
print('PASS percentiles')

# Window expiry: record an old miss that should be trimmed
b2 = SLOBudget(slo_target_s=5.0, budget_ratio=0.01, window_s=1.0)
b2.record(time.monotonic() - 5.0, 9.0)  # 5 seconds ago, outside 1s window
b2.record(time.monotonic(), 3.0)         # now, within window
assert b2.miss_count() == 0, 'Old miss should be trimmed'
print('PASS window expiry')

print('ALL PASSED')
"
```

---

### `prompts/prompts.jsonl`

**Purpose:** 30 test prompts with varying output lengths for realistic inference workloads.

Create the directory `prompts/` and write `prompts.jsonl` with exactly 30 lines.
Each line must be valid JSON with exactly two keys: `"prompt"` (a string question or instruction)
and `"max_new_tokens"` (an integer between 32 and 256).

Distribute `max_new_tokens` across three rough bands:
- 10 prompts with 32–64 tokens (short answers)
- 10 prompts with 65–128 tokens (medium answers)
- 10 prompts with 129–256 tokens (long answers)

Make the prompts topically relevant (GPU computing, cloud services, AI inference,
energy efficiency) so the generated text is consistent in difficulty.

**Test:**
```bash
python -c "
import json
lines = open('prompts/prompts.jsonl', encoding='utf-8').readlines()
assert len(lines) == 30, f'Expected 30 lines, got {len(lines)}'
for i, line in enumerate(lines):
    obj = json.loads(line)
    assert 'prompt' in obj and 'max_new_tokens' in obj, f'Line {i} missing keys'
    assert isinstance(obj['max_new_tokens'], int), f'Line {i} max_new_tokens must be int'
    assert 32 <= obj['max_new_tokens'] <= 256, f'Line {i} max_new_tokens out of range'
print('PASS: 30 valid prompts')
"
```

---

---

# Member 2 — Controller Logic

**Machine:** Mac  
**Complexity:** High  
**Files:** `controllers/__init__.py`, `controllers/base.py`, `controllers/static.py`,
`controllers/naive_slo.py`, `controllers/robust_delay_budget.py`

## What you are building

A **controller** is code that runs every second and answers one question:
*which GPU clock frequency should we use right now?*

Higher clock = faster but more energy.  
Lower clock = slower but saves energy.  
The controller sees a snapshot of system state (latencies, queue length, how old the GPU readings are)
and returns a clock index from a list like `[180, 667, 1147, 1635, 2122, 2602, 3090]` MHz.

You implement 4 controllers:
- `StaticController` — always the same clock (used as energy/latency baseline)
- `NaiveSLOController` — reacts to latency and queue, ignores telemetry age (weak baseline)
- `RobustDelayBudgetController` — also accounts for telemetry age and error budget (proposed method)
- Factory `get_controller()` — creates the right one by name

## Dependencies

`config.py` from Member 1. No GPU.

---

### `controllers/base.py`

**Purpose:** Defines the shared input type (`ControllerState`) and the abstract
base class all controllers inherit from.

**Implement dataclass `ControllerState`** with these fields:
- `t: float` — wall-clock time (default `time.monotonic()`)
- `power_w: float` — GPU power in Watts (from stale telemetry, default 0.0)
- `graphics_mhz: int` — reported GPU clock (from stale telemetry, default 0)
- `sm_mhz: int` — shader clock (from stale telemetry, default 0)
- `util_pct: int` — GPU utilisation 0–100 (from stale telemetry, default 0)
- `temp_c: int` — GPU temperature °C (from stale telemetry, default 0)
- `telemetry_age_ms: float` — **how old the telemetry is in milliseconds** (default 0.0)
- `queue_len: int` — number of pending requests (default 0)
- `arrival_rate: float` — requests/second (recent window, default 0.0)
- `arrival_cv: float` — coefficient of variation of inter-arrival times (burstiness, default 0.0)
- `p95_latency_s: float` — 95th percentile latency from `SLOBudget` (default 0.0)
- `p99_latency_s: float` — 99th percentile latency (default 0.0)
- `slo_target_s: float` — the SLO threshold in seconds (default 2.0)
- `error_budget_burn: float` — burn rate from `SLOBudget.burn_rate()` (default 0.0)
- `miss_rate: float` — fraction of recent requests that missed SLO (default 0.0)
- `phase: str` — experiment phase, e.g. "steady", "burst" (default "steady")

**Add computed property `latency_slack`:** returns `1.0 - p95_latency_s / slo_target_s`.
A value of 1.0 means no latency at all; 0.0 means exactly at target; negative means over.

**Implement abstract class `BaseController`:**

- `__init__(self, clocks: List[int])`:
  - Raise `ValueError` if `clocks` is empty
  - Store `self.clocks`, initialize `self._intervals_since_change = 0`,
    `self._last_change_t = time.monotonic()`, `self.last_reason = "INIT"`
  - `last_reason` is a string written by `_policy()` explaining why the last decision was made

- `_policy(self, state, current_idx) -> int` — abstract; subclasses implement this.
  Returns the desired clock *index* (not MHz).

- `decide(self, state, current_idx, min_dwell=0) -> int` — concrete; enforces the dwell constraint:
  1. Call `_policy()` to get `desired_idx`; clamp it to `[0, len(clocks)-1]`
  2. If `desired_idx > current_idx` (emergency up-clock): reset `_intervals_since_change = 0`, return immediately
  3. Increment `_intervals_since_change`
  4. If `desired_idx == current_idx`: return current
  5. If `desired_idx < current_idx` AND `_intervals_since_change >= min_dwell`: reset counter, return `desired_idx`
  6. Otherwise: hold, return `current_idx`

- `clock_mhz(self, idx)` — return `self.clocks[clamped_idx]`

- `find_index(self, mhz)` — return the index of the clock value closest to `mhz`

- Property `name` — return `self.__class__.__name__`

---

### `controllers/static.py`

**Purpose:** Always returns one fixed clock index. Used as the "static" baselines
(`static_high` = max clock, `static_efficient` = most energy-efficient clock from sweep).

**Implement class `StaticController(BaseController)`:**

- `__init__(self, clocks, target_mhz=0)`:
  - Call `super().__init__(clocks)`
  - Find `_target_idx = self.find_index(target_mhz)` if `target_mhz > 0`, else `len(clocks) - 1`

- `_policy(self, state, current_idx)`:
  - Set `self.last_reason = f"STATIC({self.clocks[self._target_idx]}MHz)"`
  - Return `self._target_idx`

- Property `name` — return `f"Static({self.clocks[self._target_idx]} MHz)"`

---

### `controllers/naive_slo.py`

**Purpose:** The "naive" baseline. Watches P95 latency and queue length, but does NOT
look at how old the telemetry is. This is its fatal flaw: under stale telemetry, the
controller keeps down-clocking during the delay window after a burst starts, because
the delayed readings still show low load.

**Implement class `NaiveSLOController(BaseController)`:**

- `__init__(self, clocks, slo_target_s=2.0, down_slack=0.80, min_dwell=2)`

- `_policy(self, state, current_idx)` — implement exactly this three-way logic:
  1. **Emergency** — if `state.p95_latency_s > slo_target_s` (use `state.slo_target_s` if available, else `self.slo_target_s`):
     - Set `last_reason = "EMERGENCY_SLO"` and return `min(current_idx + 1, max_idx)`
  2. **Safe down-clock** — if `state.p95_latency_s < down_slack × slo_target` AND `state.queue_len == 0`:
     - Set `last_reason = "SAFE_DOWNCLOCK"` and return `max(current_idx - 1, 0)`
  3. **Hold** — otherwise:
     - Set `last_reason = "HOLD"` and return `current_idx`

- Property `name` — return `"NaiveSLO"`

---

### `controllers/robust_delay_budget.py`

**Purpose:** The proposed robust controller. The key insight is that **stale telemetry
should increase the system's caution level**. Even if the delayed readings show low load,
if those readings are 500 ms old and the queue is growing, the real GPU state may
already be stressed.

**Risk score formula** (you must implement this exactly):
```
risk = w_q × norm_queue
     + w_age × norm_age      ← this term is the contribution of this paper
     + w_burn × norm_burn
     + w_cv × norm_cv

norm_queue = min(queue_len / 10, 1.0)
norm_age   = min((telemetry_age_ms / 1000) / 1.0, 1.0)    # 0 if use_telemetry_age=False
norm_burn  = min(error_budget_burn, 1.0)                   # 0 if use_error_budget=False
norm_cv    = min(arrival_cv, 1.0)

Default weights: w_q=0.30, w_age=0.35, w_burn=0.20, w_cv=0.15
```
Note `w_age = 0.35` is the largest weight — stale telemetry = the strongest risk signal.

**Implement class `RobustDelayBudgetController(BaseController)`:**

- `__init__` parameters:
  - `clocks, slo_target_s=8.5, min_dwell=2`
  - `use_telemetry_age=True, use_error_budget=True` — ablation flags
  - `oracle_mode=False, oracle_table_path=None`
  - Risk weights: `w_q=0.30, w_age=0.35, w_burn=0.20, w_cv=0.15`
  - Thresholds: `high_risk_thresh=0.55, low_risk_thresh=0.40`
  - Other tunable constants: `age_threshold_s=1.0, max_queue_norm=10, emergency_burn=2.0, down_slack_frac=0.80`
  - If `oracle_table_path` is provided, load it as a DataFrame via `pandas.read_csv`

- `_risk(self, state)` — compute and return the risk score [0, 1] using the formula above

- `_policy(self, state, current_idx)` — four-branch logic:
  1. **Oracle mode** — if `oracle_mode=True` and oracle table loaded, call `_oracle_lookup()`
  2. **Emergency** — if `p95 > slo_target` OR (use_error_budget AND burn > emergency_burn):
     - Set `last_reason = "EMERGENCY_SLO"` or `"EMERGENCY_BUDGET"` accordingly
     - Return `min(current_idx + 2, max_idx)` (step up 2 for emergency)
  3. **High risk** — if `_risk(state) > high_risk_thresh`:
     - Set `last_reason = f"HIGH_RISK(r={risk:.2f})"` (append `" STALE_OBSERVATION"` if age > 300 ms)
     - Return `min(current_idx + 1, max_idx)`
  4. **Safe down-clock** — if `_risk(state) < low_risk_thresh` AND `p95 < down_slack_frac × slo_target`:
     - Set `last_reason = f"SAFE_DOWNCLOCK(r={risk:.2f})"`
     - Return `max(current_idx - 1, 0)`
  5. **Hold** — otherwise:
     - Set `last_reason = f"HOLD(r={risk:.2f})"`
     - Return `current_idx`

- `_oracle_lookup(self, state, current_idx)` — try to find a row in the oracle table
  matching `state.phase`, read `min_clock_mhz`, return `self.find_index(that_mhz)`.
  Fall back to `_policy_robust()` on any error.

- Property `name` — return `"Robust"` (or `"Robust_NoAge"` / `"Robust_NoBudget"` based on ablation flags)

---

### `controllers/__init__.py`

**Purpose:** A factory so any script can call `get_controller("robust", clocks, cfg)`
without knowing which class to import.

**Implement `get_controller(name, clocks, config) -> BaseController`:**

- `"default"` → `StaticController` at `clocks[-1]`
- `"static_high"` → `StaticController` at `config.static_high_mhz` (or `clocks[-1]`)
- `"static_low"` → `StaticController` at `config.static_low_mhz` (or `clocks[0]`)
- `"static_efficient"` → `StaticController` at the clock closest to `config.static_efficient_mhz` (667 MHz)
- `"naive_slo"` → `NaiveSLOController(clocks, slo_target_s=config.slo_target_s, min_dwell=config.min_dwell_intervals)`
- `"robust"` → `RobustDelayBudgetController` with `use_telemetry_age=config.use_telemetry_age, use_error_budget=config.use_error_budget, oracle_mode=False`
- `"robust_no_age"` → `RobustDelayBudgetController` with `use_telemetry_age=False, use_error_budget=True`
- `"robust_no_budget"` → `RobustDelayBudgetController` with `use_telemetry_age=True, use_error_budget=False`
- `"oracle"` → `RobustDelayBudgetController` with `oracle_mode=True, oracle_table_path=config.oracle_table_path`
- Any other name → raise `ValueError`

**Tests:**
```bash
python -c "
from config import ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState

clocks = [180, 667, 1147, 1635, 2122, 2602, 3090]
cfg = ExperimentConfig()

# Test 1: factory creates the right types
for name in ['static_efficient', 'naive_slo', 'robust', 'robust_no_age', 'robust_no_budget']:
    c = get_controller(name, clocks, cfg)
    print(f'  {name} -> {c.name}')
print('PASS factory')

# Test 2: naive SLO - should EMERGENCY when p95 > target
naive = get_controller('naive_slo', clocks, cfg)
high_lat = ControllerState(slo_target_s=8.5, p95_latency_s=10.0, queue_len=0,
                            error_budget_burn=0.0, telemetry_age_ms=0.0, arrival_cv=0.0)
idx = naive._policy(high_lat, 3)
assert idx == 4, f'Expected 4 (up-clock), got {idx}'
assert naive.last_reason == 'EMERGENCY_SLO'
print('PASS naive emergency')

# Test 3: naive SLO - should DOWNCLOCK when p95 small and queue empty
low_lat = ControllerState(slo_target_s=8.5, p95_latency_s=2.0, queue_len=0,
                           error_budget_burn=0.0, telemetry_age_ms=0.0, arrival_cv=0.0)
idx = naive._policy(low_lat, 3)
assert idx == 2, f'Expected 2 (down-clock), got {idx}'
assert naive.last_reason == 'SAFE_DOWNCLOCK'
print('PASS naive downclock')

# Test 4: robust - MUST NOT downclock when telemetry is 600ms stale
robust = get_controller('robust', clocks, cfg)
stale = ControllerState(slo_target_s=8.5, p95_latency_s=2.0, queue_len=0,
                         error_budget_burn=0.0, telemetry_age_ms=600.0, arrival_cv=0.0)
idx = robust._policy(stale, 3)
assert 'DOWNCLOCK' not in robust.last_reason, f'Bug: downclocked with stale telemetry! reason={robust.last_reason}'
print('PASS robust holds under stale telemetry')

# Test 5: robust ablation - with use_telemetry_age=False, stale telemetry should NOT block downclock
from controllers.robust_delay_budget import RobustDelayBudgetController
no_age = RobustDelayBudgetController(clocks, slo_target_s=8.5, use_telemetry_age=False, use_error_budget=False)
idx2 = no_age._policy(stale, 3)
assert 'DOWNCLOCK' in no_age.last_reason, f'Expected downclock without age term, got {no_age.last_reason}'
print('PASS ablation no_age downclocks')

print()
print('ALL PASSED')
"
```

---

---

# Member 3 — Metrics & Analysis

**Machine:** Mac (test using CSVs from `results/attempt_3/` if available)  
**Complexity:** Medium  
**Files:** `metrics.py`, `analyze_results.py`, `oracle_builder.py`

## What you are building

During a run, every completed request and every controller decision must be recorded.
After the run, all CSVs are loaded and 8 figures are produced.

## Dependencies

`config.py` and `request_queue.py` from Member 1.
You can develop and test `analyze_results.py` on Mac against pre-existing CSV data.

---

### `metrics.py`

**Implement helper function `joules_from_samples(samples) -> float`:**
- Takes a list of `GpuSample` objects (each has `.t` and `.power_w`)
- Uses trapezoidal integration: for each consecutive pair of samples,
  `energy += (power_i + power_{i+1}) / 2 × (t_{i+1} - t_i)`
- Return total energy in Joules; return 0.0 if fewer than 2 samples

**Implement helper function `summary_stats(latencies: List[float]) -> dict`:**
- Input: a list of latency values
- Return: `{"p50": ..., "p95": ..., "p99": ..., "mean": ...}`
- Use the same index formula as `SLOBudget.percentile_latency()`
- Return all zeros if the list is empty

**Implement class `MetricsCollector`:**

- `__init__(self, run_id, results_dir, slo_target_s, warmup_s=0)` — store parameters;
  initialize empty lists `_requests` and `_control_events`; `_warmup_end_t = None`

- `set_warmup_end(self, t)` — called once, stores the timestamp when warmup ends

- `record_request(self, req)` — append a completed `Request` to `_requests`

- `record_control(self, event_dict)` — append a dict describing one controller decision to `_control_events`

- `measurement_requests(self)` — return only requests whose `finish_t >= _warmup_end_t`
  (or all requests if warmup end was never set)

- `latencies(self)` — return a list of `latency_s` values for measurement-phase requests
  where `latency_s is not None`

- `total_output_tokens(self)` — sum of `generated_tokens` across measurement-phase requests

- `miss_rate(self)` — fraction of measurement-phase latencies exceeding `slo_target_s`

- `run_summary(self, samples, config, recovery_s=None, max_queue_len=0) -> dict`:
  Return a dict with these keys (used as one row in `run_summary.csv`):
  `run_id, seed, controller, delay_ms, workload, total_j, output_tokens,
  j_per_token, p50, p95, p99, mean_lat, miss_rate, n_requests, max_queue,
  recovery_s, use_age, use_budget, burst_multiplier`
  - `total_j` — from `joules_from_samples()` on measurement-phase samples only
  - `j_per_token` — `total_j / output_tokens` (use `float("nan")` if 0 tokens)

- `save_requests_csv(self) -> str`:
  Write one row per measurement-phase request to `<results_dir>/<run_id>_requests.csv`.
  Columns: `request_id, arrival_t, start_t, finish_t, latency_s, queue_wait_s,
  service_time_s, prompt_tokens, output_tokens, phase, slo_miss`
  where `slo_miss = latency_s > slo_target_s`.

- `save_telemetry_csv(self, samples) -> str`:
  Write one row per sample to `<run_id>_telemetry.csv`.
  Columns: `t, power_w, graphics_mhz, sm_mhz, util_pct, temp_c, phase`

- `save_control_csv(self) -> str`:
  Write one row per control event to `<run_id>_control.csv`.
  Use the keys of the first dict in `_control_events` as column headers.

- `append_run_summary(self, summary_dict) -> str`:
  Append one row to `<results_dir>/run_summary.csv`.
  If the file does not exist yet, write the header first.

---

### `analyze_results.py`

**Purpose:** Load `run_summary.csv` from a results folder and produce 8 figures.
Implement as a script runnable with `python analyze_results.py --results-dir <path> --slo-target 8.5`.

**Color palette to use consistently across all figures:**
```
static_efficient → "#2E79B5" (blue)
naive_slo        → "#C85898" (pink)
robust_no_age    → "#F0A040" (orange)
robust           → "#1F8A65" (green)
```

**Figure 1 — `fig1_delay_vs_jtoken.png`:**
Line plot: X = telemetry delay (ms), Y = mean J/output-token, one line per controller.
Order X axis as `[0, 100, 250, 500, 1000]`.
This shows whether the robust controller saves energy compared to naive_slo as delay grows.

**Figure 2 — `fig2_delay_vs_miss.png`:**
Same axes but Y = mean SLO miss rate.
Expect robust to stay low; naive_slo to rise steeply at high delays.

**Figure 3 — `fig3_timeseries_<run_id>.png`:**
For the most representative robust run (highest delay, sharp_burst workload):
Dual Y-axis plot: GPU power (W) over time and queue length over time.
Load `<run_id>_telemetry.csv` and `<run_id>_control.csv` to get the time series.

**Figure 4 — `fig4_cumulative_energy.png`:**
Cumulative energy (J) over time for the four main controllers at the same delay.
Use the `_telemetry.csv` files and trapezoidal integration.

**Figure 5 — `fig5_burn_<run_id>.png`:**
Rolling burn rate over time for the four controllers.
Load `_requests.csv` files; compute burn rate in a 60-second sliding window.

**Figure 6 — `fig6_oracle_gap.png`:**
Bar chart: J/token for each controller relative to `static_efficient` (the reference).
Lower bar = closer to oracle efficiency. Aggregate across all delays.
If no `oracle` controller runs are present, use `static_efficient` as the 0% reference.

**Figure 7 — `fig7_failure_map.png`:**
If only one burst level is present (the demo case): bar chart showing miss rate per
(controller, delay) combination.
If multiple `burst_multiplier` values are present: heatmap (controller vs delay, color = miss rate).

**Figure 8 — `fig8_ablation.png`:**
Grouped bar chart comparing `robust_no_age`, `robust_no_budget`, and `robust`
side by side on J/token and miss rate at each delay.
This shows how much each component contributes.

All figures should include axis labels, a title, and a legend. Use `dpi=150`.

---

### `oracle_builder.py`

**Purpose:** Reads `sweep_summary.csv` (from Member 5's fixed-frequency sweep) and
creates `oracle_table.csv` — the pre-computed lookup table used by the oracle controller.

**Implement function `build_oracle(sweep_csv, slo_target, out_path)`:**
1. Load `sweep_csv` with pandas
2. Add a column `meets_slo = miss_rate < 0.10` (less than 10% miss = acceptable)
3. Filter rows where `meets_slo == True`; if none exist, use all rows (graceful fallback)
4. For each unique `workload` value, find the row with the lowest `j_per_token`
5. Keep columns: `workload` (rename to `workload_bucket`), `clock_mhz` (rename to `min_clock_mhz`), `j_per_token`, `miss_rate`
6. Save to `out_path` and print a summary

Implement as a runnable script with arguments `--sweep-csv`, `--slo-target`, `--out`.

**Tests:**
```bash
python -c "
import pandas as pd, tempfile, os
from metrics import joules_from_samples, summary_stats

# Test joules_from_samples
class S:
    def __init__(self, t, p): self.t=t; self.power_w=p

samples = [S(0.0, 100.0), S(1.0, 100.0), S(2.0, 100.0)]
j = joules_from_samples(samples)
assert abs(j - 200.0) < 0.001, f'Expected 200J got {j}'
print('PASS trapezoidal integration')

# Test summary_stats
stats = summary_stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
assert stats['p95'] == 10.0, f'p95={stats[\"p95\"]}'
assert abs(stats['mean'] - 5.5) < 0.001
print('PASS summary_stats')

# Test MetricsCollector miss_rate
import time
from config import ExperimentConfig
from request_queue import Request
from metrics import MetricsCollector

cfg = ExperimentConfig()
mc = MetricsCollector('test_run', '/tmp', slo_target_s=5.0)
mc.set_warmup_end(time.monotonic() - 100)  # all requests are past warmup

for i, lat in enumerate([3.0, 3.0, 8.0, 8.0]):
    r = Request(i, 'x', 10)
    r.arrival_t = time.monotonic() - lat
    r.finish_t = time.monotonic()
    mc.record_request(r)

assert abs(mc.miss_rate() - 0.5) < 0.001, f'Expected 0.5 got {mc.miss_rate()}'
print('PASS MetricsCollector miss_rate')

print()
print('ALL PASSED')
"
```

---

---

# Member 4 — Load Generation & Scale Simulation

**Machine:** Mac  
**Complexity:** Medium  
**Files:** `load_generator.py`, `scale_simulator.py`

## What you are building

- `load_generator.py` creates requests at a realistic (Poisson) arrival rate and
  pushes them to the queue. Three workload profiles simulate different real-world scenarios.
- `scale_simulator.py` simulates the full system at 8× scale on CPU using service-time
  data from the GPU sweep, so the team can evaluate controller behavior without waiting for GPU runs.

## Dependencies

`config.py` and `request_queue.py` from Member 1.

---

### `load_generator.py`

**What it does:** A daemon thread. Repeatedly computes the current arrival rate,
draws a Poisson inter-arrival time (`exponential(1/rate)`), sleeps that long, then
creates and enqueues one `Request`.

**Implement class `LoadGenerator(threading.Thread)`:**

- `__init__(self, queue, config, workload="sharp_burst", seed=42)`:
  - Store parameters; create `random.Random(seed)`; load prompts from `config.prompts_file`
  - Use a fallback list of 3–5 prompts if the file does not exist
  - Initialise `_stop_event = threading.Event()`, `_request_id = 0`, `_start_t = None`

- `_load_prompts(self, path) -> List[dict]`:
  Open `path` (JSONL); return list of dicts. Return fallback if file missing.

- `_current_rate(self, elapsed_s) -> float`:
  Compute the arrival rate for `elapsed_s` seconds into the experiment:
  - `steady`: return `config.base_arrival_rate` always
  - `sharp_burst`: if `burst_start_s <= elapsed_s <= burst_start_s + burst_duration_s`,
    return `base_arrival_rate × burst_multiplier`; else return `base_arrival_rate`
  - `ramp`: return `base_arrival_rate + (elapsed_s / duration_s) × (ramp_end_rate - base_arrival_rate)`,
    clamped so the rate never exceeds `ramp_end_rate`

- `_phase_tag(self, elapsed_s) -> str`:
  Return `"warmup"` if `elapsed_s < warmup_s`;
  `"burst"` if inside burst window (for `sharp_burst`);
  otherwise `"steady"`.

- `run(self)`:
  1. Record `_start_t = time.monotonic()`
  2. Loop until `_stop_event` is set or elapsed >= `duration_s`:
     a. Compute `elapsed = time.monotonic() - _start_t`
     b. Compute `rate = _current_rate(elapsed)`
     c. Draw inter-arrival time: `sleep_s = rng.expovariate(rate)` if `rate > 0` else `1.0`
     d. Wait: `_stop_event.wait(timeout=sleep_s)` — this respects the stop signal
     e. If stopped, break
     f. Create a `Request` with `arrival_t=time.monotonic()`, a randomly chosen prompt,
        the correct `phase` tag, and the current `_request_id`
     g. `queue.put(request)`; increment `_request_id`

- `stop(self)` — set `_stop_event`

**Tests:**
```bash
# Test 1: steady workload generates the right approximate count
python -c "
import time
from config import ExperimentConfig
from request_queue import RequestQueue
from load_generator import LoadGenerator

cfg = ExperimentConfig(base_arrival_rate=2.0, duration_s=3.0, warmup_s=0)
q = RequestQueue()
gen = LoadGenerator(q, cfg, workload='steady', seed=42)
gen.start()
time.sleep(3.5)
gen.stop()
n = q.total_enqueued
print(f'Enqueued {n} requests (expected ~6, allow 3-10)')
assert 3 <= n <= 10, f'Expected 3-10, got {n}'
print('PASS steady rate')
"

# Test 2: burst workload generates more requests during the burst window
python -c "
import time
from config import ExperimentConfig
from request_queue import RequestQueue
from load_generator import LoadGenerator

# Short experiment: 5s total, burst starts at 2s for 2s, multiplier=5
cfg = ExperimentConfig(base_arrival_rate=0.5, duration_s=5.0, warmup_s=0,
                       burst_start_s=2.0, burst_duration_s=2.0, burst_multiplier=5.0)
q = RequestQueue()
gen = LoadGenerator(q, cfg, workload='sharp_burst', seed=42)
gen.start()
time.sleep(5.5)
gen.stop()
n = q.total_enqueued
print(f'Total requests: {n} (expect >5 due to burst)')
assert n >= 5, f'Expected burst to produce more, got {n}'
print('PASS burst workload')
"

# Test 3: phase tags are correct
python -c "
import time
from config import ExperimentConfig
from request_queue import RequestQueue
from load_generator import LoadGenerator

cfg = ExperimentConfig(base_arrival_rate=5.0, duration_s=8.0, warmup_s=2.0,
                       burst_start_s=4.0, burst_duration_s=2.0, burst_multiplier=3.0)
q = RequestQueue()
gen = LoadGenerator(q, cfg, workload='sharp_burst', seed=42)
gen.start()
time.sleep(8.5)
gen.stop()

phases = set()
while not q.empty():
    r = q.get(timeout=0.1)
    if r: phases.add(r.phase)

print('Observed phases:', phases)
assert 'warmup' in phases, 'Missing warmup phase'
assert 'steady' in phases, 'Missing steady phase'
assert 'burst' in phases, 'Missing burst phase'
print('PASS phase tags')
"
```

---

### `scale_simulator.py`

**Purpose:** Simulate 8 virtual workers processing requests without any real GPU.
Uses service-time distributions fitted from `sweep_summary.csv` to answer:
*what would the system's latency and energy look like at 8× scale?*

**Implement function `fit_service_times(sweep_df) -> dict`:**
- Input: DataFrame loaded from `sweep_summary.csv` (has columns: `clock_mhz`, `mean_lat`, `workload`)
- For each unique `clock_mhz` value, record its mean service time
- Return a dict mapping `clock_mhz -> mean_service_time_s`
- If `mean_lat` is not present, use `p95 / 2` as a fallback estimate

**Implement function `simulate(n_workers, n_requests, controller, delay_ms,
workload, sweep_df, slo_target_s, base_rate, burst_mult, seed) -> dict`:**
- Generate N requests with Poisson arrivals matching `workload` profile (same logic as `LoadGenerator`)
- Maintain a list of `n_workers` virtual workers, each with a `finish_time` attribute (when they become free)
- For each request: assign to the worker that is free soonest; the service time is drawn from
  `Normal(mean=fit_service_times[current_clock], std=mean*0.1)` plus a uniform queueing delay
- Run the same controller decision every second using a `ControllerState` built from the simulated signals
- Track: `total_j` (estimated from `power ≈ f(clock_mhz)` linear fit), `miss_rate`, `p95_latency_s`
- Return a summary dict with keys `controller, delay_ms, workload, miss_rate, p95_latency_s, total_j, j_per_token, n_requests`

**Implement `main()` with argparse arguments:**
- `--n-workers` (default 8), `--n-requests` (default 10000)
- `--controller` (naive_slo or robust), `--delay-ms`
- `--workload` (sharp_burst or steady)
- `--sweep-csv`, `--results-dir`, `--slo-target`

**Tests:**
```bash
# Test fit_service_times
python -c "
import pandas as pd
from scale_simulator import fit_service_times

df = pd.DataFrame({'clock_mhz': [667, 1147, 2602], 'mean_lat': [8.0, 5.0, 3.0], 'workload': ['steady','steady','steady']})
times = fit_service_times(df)
assert times[667] == 8.0
assert times[1147] == 5.0
print('PASS fit_service_times')
"

# Test full simulation runs without error
python -c "
import pandas as pd
from scale_simulator import simulate
from controllers import get_controller
from config import ExperimentConfig

df = pd.DataFrame({'clock_mhz': [667, 1147, 2602], 'mean_lat': [8.0, 5.0, 3.0], 'workload': ['steady','steady','steady']})
cfg = ExperimentConfig()
ctrl = get_controller('naive_slo', [667, 1147, 2602], cfg)
result = simulate(
    n_workers=4, n_requests=50, controller=ctrl, delay_ms=0,
    workload='steady', sweep_df=df, slo_target_s=8.5,
    base_rate=0.5, burst_mult=1.0, seed=42
)
assert 'miss_rate' in result
assert 0.0 <= result['miss_rate'] <= 1.0
print('PASS simulation output:', result)
"
```

---

---

# Member 5 — GPU Hardware, LLM & Orchestration

**Machine:** RTX + Windows only  
**Complexity:** Highest (most integration)  
**Files:** `gpu_clock.py`, `telemetry.py`, `measure_timing.py`, `llm_worker.py`,
`run_experiment.py`, `run_demo.py`, `run_all.ps1`, `setup.ps1`

## What you are building

You build the GPU interface, the LLM inference worker, and the main experiment loop
that wires everything together. Only you can run real GPU experiments.

**Pull all other members' code before starting.**

## Dependencies

All other members (1–4).

---

### `gpu_clock.py`

**Implement `supported_graphics_clocks() -> List[int]`:**
- Run `nvidia-smi -q -d SUPPORTED_CLOCKS` as a subprocess
- Parse every line matching `Graphics : NNN MHz` (use `re.search`)
- Return a sorted list of unique integer MHz values
- Raise `RuntimeError` if nvidia-smi is unavailable or no values found

**Implement `select_experiment_clocks(all_clocks, n=7) -> List[int]`:**
- Given the full sorted list, return `n` evenly-spaced values
- Always include the minimum and maximum clock
- Use index arithmetic: `indices = linspace(0, len-1, n)` rounded to int

**Implement `set_graphics_clock(mhz: int) -> float`:**
- Run `nvidia-smi --lock-gpu-clocks=<mhz>,<mhz>` as a subprocess
- Record start time, call subprocess, return elapsed seconds
- Log a warning if the command fails instead of raising (so the experiment can continue)
- **Requires elevated/Administrator terminal**

**Implement `reset_graphics_clock() -> float`:**
- Run `nvidia-smi --reset-gpu-clocks`
- Always call this in a `finally` block in `run_experiment.py`
- Return elapsed seconds

**Tests:**
```powershell
python -c "
from gpu_clock import supported_graphics_clocks, select_experiment_clocks
clocks = supported_graphics_clocks()
print('Total supported clocks:', len(clocks))
print('Range:', clocks[0], '—', clocks[-1], 'MHz')
assert len(clocks) > 0
subset = select_experiment_clocks(clocks, 7)
assert len(subset) == 7
assert subset[0] == clocks[0] and subset[-1] == clocks[-1]
print('PASS clock enumeration and selection')
"
```

---

### `telemetry.py`

**Implement dataclass `GpuSample`** with fields:
`t, power_w, graphics_mhz, sm_mhz, util_pct, temp_c, phase`

**Implement class `TelemetrySampler(threading.Thread)`:**

- `__init__(self, delay_buffer, poll_interval_s=0.1)`:
  - Call `pynvml.nvmlInit()` and get device handle for index 0
  - Store `self.samples = []` (all samples, for energy calculation)
  - `_stop_event = threading.Event()`, `_phase = "steady"`
  - Set `daemon = True`

- `set_phase(self, phase)` — update the current phase tag (called from `run_experiment.py`)

- `run(self)`:
  Each iteration (every `poll_interval_s` seconds):
  1. Call NVML to get: power (divide by 1000 to convert mW→W), graphics clock, SM clock, utilisation, temperature
  2. Create a `GpuSample` with `t=time.monotonic()` and the current `_phase`
  3. Call `self.delay_buffer.push(sample)`
  4. Append to `self.samples`
  5. On any NVML exception: log a warning and continue (don't crash)
  6. Wait `poll_interval_s` with `_stop_event.wait(timeout=...)`

- `stop(self)`:
  - Set `_stop_event`
  - Call `pynvml.nvmlShutdown()` (wrap in try/except)

**Tests:**
```powershell
python -c "
import time
from delay_buffer import DelayBuffer
from telemetry import TelemetrySampler

buf = DelayBuffer()
s = TelemetrySampler(buf, poll_interval_s=0.1)
s.start()
time.sleep(0.6)
s.stop()
print('Samples collected:', len(s.samples))
assert len(s.samples) >= 4, 'Expected at least 4 samples in 0.6s'
sample = s.samples[0]
assert sample.power_w > 0, 'Power should be positive'
print('PASS telemetry sampler')
"
```

---

### `measure_timing.py`

**Purpose:** One-time characterisation of the GPU environment. Measures:
- NVML update period (how often the power reading actually changes)
- Clock lock/reset latency
- GPU warm-up time after boot

**Implement `main()` with argparse. Key measurements:**

- `--skip-actuation` flag: if set, skip the clock-lock tests (safe to run without root)

1. Poll NVML 50 times at 50ms intervals; compute the fraction of samples where power actually changed.
   If < 50% change, NVML likely has a slow update period → log a warning.
2. If not skipping actuation: measure time for `set_graphics_clock()` and `reset_graphics_clock()`
   by recording wall time before/after.
3. Save results to `timing_characterisation.json` in the results directory.
4. Print a human-readable summary.

---

### `llm_worker.py`

**Implement `load_model(model_id) -> (model, tokenizer)`:**
- Load with `AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")`
- Set `model.eval()`
- Return `(model, tokenizer)`

**Implement class `LLMWorker(threading.Thread)`:**

- `__init__(self, queue, model, tokenizer, on_complete, batch_size=4)`:
  - `on_complete` is a callback `(req: Request) -> None` called after each request finishes
  - `daemon = True`

- `warmup(self)`:
  Run one small dummy inference (e.g., `["warmup"]` with `max_new_tokens=5`) to initialise
  CUDA kernels so the first real request isn't unfairly slow.

- `run(self)`:
  1. Loop until stop event is set
  2. Collect up to `batch_size` requests from the queue (using `get(timeout=0.5)`)
  3. Set `start_t = time.monotonic()` for each request when dequeued
  4. Tokenize all prompts in the batch with `padding=True, truncation=True`
  5. Call `model.generate()` with `do_sample=False` and `pad_token_id=tokenizer.eos_token_id`
  6. After generation, set `finish_t = time.monotonic()` for all requests in the batch
  7. Extract `generated_tokens` for each response (tokens after the prompt portion)
  8. Call `on_complete(req)` for each request
  9. If batch is empty after timeout, loop again

- `stop(self)` — set stop event

---

### `run_experiment.py`

**Purpose:** The main control loop. Wires all components together for one experiment condition.

**Implement `_run_single(cfg: ExperimentConfig) -> dict`:**

Setup phase:
1. Create `os.makedirs(cfg.results_dir, exist_ok=True)`
2. Set up file logging: create a `FileHandler` for `<results_dir>/<run_id>.log`
3. Call `supported_graphics_clocks()`, then `select_experiment_clocks()` to populate `cfg.selected_clocks`
4. Create `RequestQueue`, `DelayBuffer`, `MetricsCollector`, `SLOBudget`
5. Create `TelemetrySampler(delay_buffer)` and start it
6. Load the model with `load_model(cfg.model_id)`
7. Create `LLMWorker`; its `on_complete` callback should call `metrics.record_request(req)` and `slo_budget.record(req.finish_t, req.latency_s)`
8. Warm up the worker
9. Create `get_controller(cfg.controller, cfg.selected_clocks, cfg)` and set initial clock with `set_graphics_clock()`
10. Create `LoadGenerator(queue, cfg, cfg.workload, cfg.seed)` and start everything

Control loop (runs for `cfg.duration_s`):
```
Every cfg.control_interval_s seconds:
  1. elapsed = time.monotonic() - start_t
  2. Update sampler phase tag based on elapsed
  3. Read stale telemetry: sample, age_s = delay_buffer.get(delay_s=cfg.delay_ms/1000, jitter_s=cfg.delay_jitter_ms/1000)
  4. Build ControllerState from sample + slo_budget.summary() + queue.qsize + recent arrival stats
  5. Set state.telemetry_age_ms = age_s * 1000
  6. new_idx = controller.decide(state, current_idx, cfg.min_dwell_intervals)
  7. If new_idx != current_idx: set_graphics_clock(controller.clock_mhz(new_idx)); current_idx = new_idx
  8. Record control event dict: {t, clock_mhz, decision_reason: controller.last_reason, risk, queue_len, …}
  9. metrics.record_control(event)
```

Cleanup (always in `finally`):
1. Stop all threads (generator, worker, sampler)
2. **Call `reset_graphics_clock()`** — must always run even if experiment fails
3. Remove file log handler
4. Compute `recovery_s` — time from burst_end until miss_rate falls back below `slo_budget_ratio × 2`
5. Build and save all CSVs
6. Return `metrics.run_summary(...)`

**CLI interface:**
```
python run_experiment.py --controller naive_slo --delay-ms 250 \
  --workload sharp_burst --seed 42 --results-dir results/attempt_3
```

---

### `run_demo.py`

**Purpose:** Orchestrates all 24 conditions and saves results to an auto-incremented folder.

**Constants at the top of the file:**
```
CONTROLLERS = ["static_efficient", "naive_slo", "robust_no_age", "robust"]
DELAYS_MS   = [0, 250, 500]
WORKLOADS   = ["steady", "sharp_burst"]
SEED        = 42
SLO_TARGET  = 8.5
BASE_RATE   = 0.15
BURST_MULT  = 5.0
BATCH_SIZE  = 4
# 4 × 3 × 2 = 24 conditions
```

**Implement `_get_attempt_dir(base="results") -> str`:**
- Scan `base/` for existing `attempt_N` folders
- Return `base/attempt_{N+1}` (the next unused number, starting at 1)
- Create the directory

**Implement a `_Tee` class** that wraps two output streams (stdout + a file)
so `print()` output goes to both the terminal and `demo_run.log`.

**`main()` phases:**
1. Call `_get_attempt_dir()` to create `results/attempt_N/`
2. Set up tee logging
3. Print a header with date, attempt number, total conditions
4. Run timing characterisation (`measure_timing.py` logic or subprocess call)
5. Run the 24-condition loop: for each `(controller, delay_ms, workload)`:
   - Build `ExperimentConfig` with the right parameters
   - Call `_run_single(cfg)` from `run_experiment.py`
   - Print a one-line progress summary after each run
6. Call `analyze_results.py` to generate the 8 figures
7. Print a completion summary

---

### `setup.ps1` and `run_all.ps1`

**`setup.ps1`** — one-time environment check, run as Administrator:
1. Print `nvidia-smi` output to confirm GPU is detected
2. Install Python dependencies (torch, transformers, nvidia-ml-py, etc.)
3. Test clock lock: `nvidia-smi --lock-gpu-clocks=1147,1147` then `nvidia-smi --reset-gpu-clocks`
4. Run all unit tests: `python -m pytest tests/`
5. Print "Setup complete" or a list of failures

**`run_all.ps1`** — full experiment matrix for report-quality runs (longer duration):
- Same 24 conditions as `run_demo.py` but `duration_s=300`, `warmup_s=30`
- Also runs the 5-point frequency sweep (needed by oracle_builder)
- Also runs scale simulation for each controller
- Calls `analyze_results.py` at the end

**Tests (Member 5 — run from elevated Anaconda PowerShell Prompt):**
```powershell
# Unit test: single short run
python run_experiment.py --controller static_efficient --delay-ms 0 `
  --workload steady --seed 42 --results-dir results/test_single `
  --duration 30 --warmup 5
# Expect: results/test_single/<run_id>_requests.csv + _telemetry.csv + _control.csv

# Integration test: two conditions
python run_demo.py --quick-test  # add a --quick-test flag that only runs 2 conditions
# Expect: results/attempt_N/ with run_summary.csv and at least fig1 + fig2
```

---

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: pynvml` | `pip install nvidia-ml-py` (Windows only) |
| `ModuleNotFoundError: torch` | `pip3 install torch --index-url https://download.pytorch.org/whl/cu126` |
| `nvidia-smi not found` | Open terminal as Administrator; check PATH |
| `miss_rate = 0% everywhere` | SLO is too loose; try `slo_target_s = 7.0` |
| Controller test T2 failed | Increase `w_age` from 0.35 to 0.40 in `RobustDelayBudgetController` |
| Clock lock fails silently | Must run as Administrator on Windows |
| All test cases pass but fig7 is missing | Check `burst_multiplier` is in `run_summary.csv` |
