"""Metrics collector contract shared by the team plan and run_experiment.py."""

import time

from config import ExperimentConfig
from metrics import MetricsCollector, joules_from_samples, summary_stats
from request_queue import Request


class Sample:
    def __init__(self, t, power_w, phase="steady"):
        self.t = t
        self.power_w = power_w
        self.graphics_mhz = 667
        self.sm_mhz = 667
        self.util_pct = 10
        self.temp_c = 40
        self.phase = phase


def test_joules_and_summary_stats():
    samples = [Sample(0.0, 100.0), Sample(1.0, 100.0), Sample(2.0, 100.0)]
    assert abs(joules_from_samples(samples) - 200.0) < 0.001
    stats = summary_stats([float(i) for i in range(1, 11)])
    assert stats["p95"] == 10.0
    assert abs(stats["mean"] - 5.5) < 0.001
    assert summary_stats([]) == {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}


def test_team_plan_miss_rate_ignores_warmup():
    collector = MetricsCollector("test_run", "results", slo_target_s=5.0)
    collector.set_warmup_end(50.0)
    for i, latency in enumerate([3.0, 3.0, 8.0, 8.0]):
        request = Request(i, "x", 10)
        request.finish_t = 100.0 + i
        request.arrival_t = request.finish_t - latency
        request.generated_tokens = 10
        collector.record_request(request)
    warmup = Request(99, "x", 10)
    warmup.arrival_t = 1.0
    warmup.finish_t = 40.0
    collector.record_request(warmup)
    assert abs(collector.miss_rate() - 0.5) < 0.001


def test_runner_constructor_and_csv_names(tmp_path):
    cfg = ExperimentConfig(results_dir=str(tmp_path), run_id="run_test", slo_target_s=5.0)
    collector = MetricsCollector(cfg.slo_target_s, cfg.results_dir, cfg.run_id)
    collector.set_warmup_end(time.monotonic() - 10)
    request = Request(1, "hello", 20, arrival_t=time.monotonic() - 2)
    request.start_t = request.arrival_t + 0.2
    request.finish_t = time.monotonic()
    request.generated_tokens = 4
    collector.on_request(request)
    collector.add_control_event({"t": request.finish_t, "selected_mhz": 667, "queue_len": 0})
    summary = collector.run_summary(
        [Sample(request.finish_t - 1, 50.0), Sample(request.finish_t, 50.0)],
        cfg,
        recovery_s=None,
        max_queue_len=1,
    )
    assert summary["run_id"] == "run_test"
    assert summary["output_tokens"] == 4
    assert abs(summary["total_j"] - 50.0) < 0.001
    assert collector.save_requests_csv().endswith("run_test_requests.csv")
    assert collector.save_telemetry_csv([Sample(0.0, 10.0)]).endswith("run_test_telemetry.csv")
    assert collector.save_control_csv().endswith("run_test_control.csv")
    assert collector.append_run_summary(summary).endswith("run_summary.csv")
