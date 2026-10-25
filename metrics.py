import csv
import os
from dataclasses import asdict, is_dataclass

def joules_from_samples(samples):
    """
    Calculate total GPU energy in joules using the trapezoidal rule.

    Each sample must have:
        t: timestamp in seconds
        power_w: GPU power in watts
    """
    if len(samples) < 2:
        return 0.0

    samples = sorted(samples, key=lambda sample: sample.t)

    total_energy = 0.0

    for i in range(1, len(samples)):
        previous = samples[i - 1]
        current = samples[i]

        time_difference = current.t - previous.t
        average_power = (previous.power_w + current.power_w) / 2

        total_energy += average_power * time_difference

    return total_energy


def summary_stats(latencies):
    """
    Calculate summary statistics for request latencies.
    """
    import numpy as np

    if len(latencies) == 0:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    values = np.asarray(latencies, dtype=float)

    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50, method="higher")),
        "p95": float(np.percentile(values, 95, method="higher")),
        "p99": float(np.percentile(values, 99, method="higher")),
    }



class MetricsCollector:
    """
    Collect request, GPU telemetry, and controller metrics
    during an experiment.
    """

    def __init__(self, run_id, output_dir, slo_target_s):
        self.run_id = run_id
        self.output_dir = output_dir
        self.slo_target_s = slo_target_s

        self.requests = []
        self.telemetry = []
        self.control_events = []

        self.warmup_end_t = None


    def record_request(self, request):
        """
        Record a completed LLM request.
        """
        if request.latency_s is None:
            return

        self.requests.append(request)


    def slo_miss_rate(self):
        """
        Calculate the fraction of completed requests
        that exceeded the SLO latency target.
        """
        completed = [
            request for request in self.requests
            if request.latency_s is not None
            and (
                self.warmup_end_t is None
                or request.arrival_t >= self.warmup_end_t
            )
        ]

        if not completed:
            return 0.0

        misses = sum(
            request.latency_s > self.slo_target_s
            for request in completed
        )

        return misses / len(completed)

    
    def record_telemetry(self, sample):
        """
        Record one GPU telemetry sample.
        """
        self.telemetry.append(sample)



    def record_control_event(self, event):
        """
        Record one GPU controller decision or event.
        """
        self.control_events.append(event)


    def energy_per_token(self):
        """
        Calculate GPU energy consumed per generated token.
        """
        valid_requests = [
            request for request in self.requests
            if (
                self.warmup_end_t is None
                or request.arrival_t >= self.warmup_end_t
            )
        ]

        total_tokens = sum(
            request.generated_tokens
            for request in valid_requests
        )

        if total_tokens == 0:
            return 0.0

        valid_telemetry = [
            sample for sample in self.telemetry
            if (
                self.warmup_end_t is None
                or sample.t >= self.warmup_end_t
            )
        ]

        total_energy = joules_from_samples(valid_telemetry)

        return total_energy / total_tokens


    def export_telemetry_csv(self):
        """
        Save GPU telemetry samples to a CSV file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"{self.run_id}_telemetry.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = [
            "t",
            "power_w",
            "graphics_mhz",
            "sm_mhz",
            "util_pct",
            "temp_c",
            "phase",
        ]

        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for sample in self.telemetry:
                row = {
                    field: getattr(sample, field, None)
                    for field in fieldnames
                }
                writer.writerow(row)

        return filepath

    def export_requests_csv(self):
        """
        Save completed LLM requests to a CSV file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"{self.run_id}_requests.csv"
        filepath = os.path.join(self.output_dir, filename)

        fieldnames = [
            "request_id",
            "arrival_t",
            "start_t",
            "finish_t",
            "latency_s",
            "queue_wait_s",
            "generated_tokens",
            "prompt_tokens",
            "phase",
            "seed_idx",
        ]

        with open(filepath, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

            for request in self.requests:
                row = {
                    field: getattr(request, field, None)
                    for field in fieldnames
                }
                writer.writerow(row)

        return filepath

    def export_control_csv(self):
        """
        Save GPU controller events to a CSV file.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        filename = f"{self.run_id}_control.csv"
        filepath = os.path.join(self.output_dir, filename)

        import pandas as pd

        df = pd.DataFrame(self.control_events)
        df.to_csv(filepath, index=False)

        return filepath

    
    def append_run_summary(self, summary_dict):
        """
        Append one experiment summary to run_summary.csv.
        Write column headers if the file does not exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        filepath = os.path.join(
            self.output_dir, "run_summary.csv"
        )

        file_exists = os.path.exists(filepath)
        fieldnames = list(summary_dict.keys())

        with open(filepath, "a", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            if not file_exists or os.path.getsize(filepath) == 0:
                writer.writeheader()

            writer.writerow(summary_dict)

        return filepath

    def run_summary(self, samples, config, recovery_s=None, max_queue_len=0):
        """
        Calculate summary metrics for one experiment.
        """
                # Keep only GPU samples collected after warmup
        valid_samples = [
            sample for sample in samples
            if (
                self.warmup_end_t is None
                or sample.t >= self.warmup_end_t
            )
        ]

        # Calculate total GPU energy in joules
        total_j = joules_from_samples(valid_samples)
                # Keep only requests completed after warmup
        valid_requests = [
            request for request in self.requests
            if (
                request.latency_s is not None
                and (
                    self.warmup_end_t is None
                    or request.arrival_t >= self.warmup_end_t
                )
            )
        ]

        # Count generated tokens during the measurement phase
        output_tokens = sum(
            request.generated_tokens
            for request in valid_requests
        )
                # Calculate energy consumed per generated token
        if output_tokens > 0:
            j_per_token = total_j / output_tokens
        else:
            j_per_token = float("nan")
                    # Collect latency values from measurement-phase requests
        latencies = [
            request.latency_s
            for request in valid_requests
        ]

        # Calculate mean, p50, p95, and p99 latency
        latency_stats = summary_stats(latencies)
                # Calculate SLO miss rate
        n_requests = len(valid_requests)

        if n_requests > 0:
            n_misses = sum(
                latency > self.slo_target_s
                for latency in latencies
            )
            miss_rate = n_misses / n_requests
        else:
            miss_rate = 0.0
                    # Combine experiment metrics into one dictionary
        summary = {
            "run_id": config.run_id,
            "seed": config.seed,
            "controller": config.controller,
            "delay_ms": config.delay_ms,
            "workload": config.workload,
            "total_j": total_j,
            "output_tokens": output_tokens,
            "j_per_token": j_per_token,
            "p50": latency_stats["p50"],
            "p95": latency_stats["p95"],
            "p99": latency_stats["p99"],
            "mean_lat": latency_stats["mean"],
            "miss_rate": miss_rate,
            "n_requests": n_requests,
            "max_queue": max_queue_len,
            "recovery_s": recovery_s,
            "use_age": config.use_telemetry_age,
            "use_budget": config.use_error_budget,
            "burst_multiplier": config.burst_multiplier,
        }

        return summary