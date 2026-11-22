"""Build the oracle clock lookup table from a fixed-frequency sweep.

The TEAM_PLAN.md acceptance rule is a miss rate strictly below 10%.
The --slo-target CLI argument is retained for interface compatibility;
it represents the experiment latency target, not the miss-rate cutoff.
"""

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("workload", "clock_mhz", "j_per_token", "miss_rate")
MISS_RATE_LIMIT = 0.10


def build_oracle(sweep_csv, slo_target, out_path):
    """Select the lowest-energy qualifying clock for each workload.

    If *no* rows in the entire sweep meet the miss-rate limit, fall back
    to all rows, as specified in TEAM_PLAN.md.

    Returns the output DataFrame.
    """
    sweep_path = Path(sweep_csv)
    output_path = Path(out_path)
    if not sweep_path.is_file():
        raise FileNotFoundError(f"Sweep CSV not found: {sweep_path}")

    sweep = pd.read_csv(sweep_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in sweep.columns]
    if missing:
        raise ValueError(f"Sweep CSV missing required columns: {', '.join(missing)}")
    if sweep.empty:
        raise ValueError("Sweep CSV is empty")

    for col in ("clock_mhz", "j_per_token", "miss_rate"):
        sweep[col] = pd.to_numeric(sweep[col], errors="coerce")
    invalid = sweep[list(REQUIRED_COLUMNS)].isna().any(axis=1)
    invalid |= (sweep["clock_mhz"] <= 0) | (sweep["j_per_token"] < 0)
    invalid |= (sweep["miss_rate"] < 0) | (sweep["miss_rate"] > 1)
    if invalid.any():
        raise ValueError(f"Sweep CSV has {int(invalid.sum())} row(s) with missing or invalid values")

    sweep["meets_slo"] = sweep["miss_rate"] < MISS_RATE_LIMIT
    eligible = sweep.loc[sweep["meets_slo"]]
    used_fallback = eligible.empty
    if used_fallback:
        eligible = sweep
        print("Warning: no sweep rows meet miss_rate < 0.10; using all rows.")

    # idxmin chooses the first row in a tie, preserving input order.
    best_indices = eligible.groupby("workload", sort=True)["j_per_token"].idxmin()
    result = eligible.loc[best_indices, list(REQUIRED_COLUMNS)].copy()
    result = result.rename(columns={
        "workload": "workload_bucket",
        "clock_mhz": "min_clock_mhz",
    }).sort_values("workload_bucket").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Loaded {len(sweep)} sweep rows from {sweep_path}")
    print(f"Selected {len(result)} workload configuration(s)")
    print(f"SLO latency target: {slo_target} s (sweep miss-rate cutoff: < 10%)")
    print(f"Saved oracle lookup table to {output_path}")
    print(result.to_string(index=False))
    return result


def main():
    parser = argparse.ArgumentParser(description="Build oracle_table.csv from sweep_summary.csv")
    parser.add_argument("--sweep-csv", required=True, help="Path to sweep_summary.csv")
    parser.add_argument("--slo-target", type=float, default=8.5,
                        help="Latency SLO target in seconds (default: 8.5)")
    parser.add_argument("--out", default="oracle_table.csv", help="Output CSV path")
    args = parser.parse_args()
    build_oracle(args.sweep_csv, args.slo_target, args.out)


if __name__ == "__main__":
    main()
