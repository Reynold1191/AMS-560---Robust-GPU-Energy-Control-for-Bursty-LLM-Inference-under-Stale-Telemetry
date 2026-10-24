"""Verify Member 1's imported data layer against controller inputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import ExperimentConfig
from controllers import get_controller
from controllers.base import ControllerState
from delay_buffer import DelayBuffer
from request_queue import Request, RequestQueue
from slo_budget import SLOBudget


def test_config_and_request_queue_contract():
    cfg = ExperimentConfig(results_dir="results/test", delay_ms=250)
    assert cfg.to_dict()["delay_ms"] == 250
    assert Path(cfg.results_path("control.csv")) == Path("results/test/control.csv")
    assert ExperimentConfig().selected_clocks is not cfg.selected_clocks
    queue = RequestQueue()
    for i in range(3):
        queue.put(Request(i, "Explain GPU DVFS", 32, arrival_t=100))
    for i in range(3):
        req = queue.get(block=False)
        assert req.request_id == i
        assert req.latency_s is None
        req.start_t, req.finish_t = 100.5, 101.5
        assert (req.latency_s, req.queue_wait_s, req.service_time_s) == (1.5, 0.5, 1.0)
        assert req.to_dict()["output_tokens"] == 0
        queue.task_done()
    assert queue.total_enqueued == queue.total_dequeued == 3
    assert queue.empty() and queue.qsize == 0
    assert queue.get(block=False) is None


def test_delay_buffer_age_is_forwarded_in_milliseconds(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("delay_buffer.time.monotonic", lambda: now[0])
    buffer = DelayBuffer()
    assert buffer.get() == (None, 0.0)
    old, fresh = SimpleNamespace(power_w=30), SimpleNamespace(power_w=60)
    buffer.push(old)
    now[0] = 100.6
    buffer.push(fresh)
    assert buffer.latest()[0] is fresh
    sample, age_s = buffer.get(delay_s=0.5)
    assert sample is old
    assert age_s == pytest.approx(0.6)
    state = ControllerState(slo_target_s=8.5, p95_latency_s=2, power_w=sample.power_w, telemetry_age_ms=age_s * 1000)
    controller = get_controller("robust", [180, 667, 1147], ExperimentConfig())
    assert controller.decide(state, 1, min_dwell=0) == 1
    assert "STALE_OBSERVATION" in controller.last_reason
    assert buffer.get(delay_s=999)[0] is old
    buffer.clear()
    assert buffer.size() == 0


def test_budget_completion_and_expiry_drive_emergency_and_recovery(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("slo_budget.time.monotonic", lambda: now[0])
    cfg = ExperimentConfig()
    budget = SLOBudget(cfg.slo_target_s, cfg.slo_budget_ratio, cfg.budget_window_s)
    for latency in [3, 3, 9, 9]:
        budget.record(now[0], latency)
    summary = budget.summary()
    assert summary["window_count"] == 4
    assert summary["miss_rate"] == 0.5
    assert summary["burn_rate"] == 50
    assert budget.p95() == budget.p99() == 9
    controller = get_controller("robust", [180, 667, 1147, 1635], cfg)
    state = ControllerState(slo_target_s=cfg.slo_target_s, error_budget_burn=summary["burn_rate"])
    assert controller.decide(state, 0) == 2
    assert controller.last_reason == "EMERGENCY_BUDGET"
    now[0] += cfg.budget_window_s + 1
    assert budget.count() == budget.miss_count() == 0
    assert budget.burn_rate() == budget.p95() == budget.p99() == 0
    assert budget.remaining_budget() == 1


def test_imported_prompt_dataset_has_all_length_bands():
    path = Path(__file__).resolve().parents[1] / "prompts" / "prompts.jsonl"
    prompts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(prompts) == 30
    for prompt in prompts:
        assert set(prompt) == {"prompt", "max_new_tokens"}
        assert isinstance(prompt["prompt"], str) and prompt["prompt"].strip()
        assert isinstance(prompt["max_new_tokens"], int)
    for lower, upper in [(32, 64), (65, 128), (129, 256)]:
        assert sum(lower <= prompt["max_new_tokens"] <= upper for prompt in prompts) == 10
