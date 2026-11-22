"""AMS 560 Member 3 analysis. Input: run_summary.csv and per-run CSVs."""
import argparse
from pathlib import Path
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {'static_efficient': '#2E79B5', 'naive_slo': '#C85898',
          'robust_no_age': '#F0A040', 'robust': '#1F8A65',
          'robust_no_budget': '#9467BD'}
MAIN = ['static_efficient', 'naive_slo', 'robust_no_age', 'robust']
DELAYS = [0, 100, 250, 500, 1000]


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze AMS 560 GPU experiment results')
    parser.add_argument('--results-dir', required=True, type=Path)
    parser.add_argument('--slo-target', type=float, default=8.5)
    return parser.parse_args()


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out / name}')


def require_columns(df, cols, context):
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f'{context}: missing columns {sorted(missing)}')


def plot_delay(df, out, metric, filename, ylabel, title, budget=False):
    require_columns(df, ['controller', 'delay_ms', metric], filename)
    fig, ax = plt.subplots(figsize=(8, 5))
    for controller, group in df.groupby('controller'):
        summary = group.groupby('delay_ms')[metric].agg(['mean', 'std']).sort_index()
        factor = 100 if budget else 1
        ax.errorbar(summary.index, summary['mean'] * factor,
                    yerr=summary['std'].fillna(0) * factor,
                    color=COLORS.get(controller, '#808080'), label=controller,
                    marker='o', capsize=3)
    if budget:
        ax.axhline(1, color='red', linestyle='--', label='1% miss budget')
    ax.set_xticks(sorted(df['delay_ms'].dropna().unique()))
    ax.set_xlabel('Injected telemetry delay (ms)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=.3)
    ax.legend()
    save(fig, out, filename)


def read_run(results_dir, run_id, kind):
    path = results_dir / f'{run_id}_{kind}.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def aligned_queue(telemetry, requests):
    require_columns(telemetry, ['t', 'power_w'], 'telemetry')
    require_columns(requests, ['arrival_t', 'start_t', 'finish_t'], 'requests')
    if telemetry.empty or requests.empty:
        raise ValueError('Empty telemetry or requests')
    t = pd.to_numeric(telemetry['t'], errors='raise').to_numpy()
    arrivals = pd.to_numeric(requests['arrival_t'], errors='raise').to_numpy()
    starts = pd.to_numeric(requests['start_t'], errors='raise').to_numpy()
    finishes = pd.to_numeric(requests['finish_t'], errors='raise').to_numpy()
    if not (np.isfinite(t).all() and np.isfinite(arrivals).all()
            and np.isfinite(starts).all() and np.isfinite(finishes).all()):
        raise ValueError('Nonfinite timestamps')
    if max(arrivals.min(), t.min()) > min(finishes.max(), t.max()):
        raise ValueError('Telemetry and request timestamps do not overlap; check clock origin')
    result = telemetry.copy()
    result['queue_len'] = [int(((arrivals <= time) & (starts > time)).sum()) for time in t]
    return result.sort_values('t')


def plot_timeseries(telemetry, requests, control, out, run_id):
    # Control CSV is required by the team plan and checked for availability.
    require_columns(control, ['t'], 'control')
    data = aligned_queue(telemetry, requests)
    time = data['t'] - data['t'].min()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax2 = ax.twinx()
    a, = ax.plot(time, data['power_w'], color='#2E79B5', label='GPU power')
    b, = ax2.step(time, data['queue_len'], where='post', color='#C85898', label='Queue length')
    ax.set_xlabel('Elapsed telemetry time (s)')
    ax.set_ylabel('GPU power (W)')
    ax2.set_ylabel('Queue length (requests)')
    ax.set_title(f'GPU Power and Queue Length — {run_id}')
    ax.legend([a, b], ['GPU power', 'Queue length'], loc='upper left')
    ax.grid(alpha=.3)
    save(fig, out, f'fig3_timeseries_{run_id}.png')


def cumulative_energy(telemetry):
    require_columns(telemetry, ['t', 'power_w'], 'telemetry')
    data = telemetry[['t', 'power_w']].dropna().sort_values('t')
    if len(data) < 2 or data['t'].duplicated().any():
        raise ValueError('Cumulative energy requires at least two distinct telemetry timestamps')
    t = data['t'].to_numpy(dtype=float)
    p = data['power_w'].to_numpy(dtype=float)
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError('Invalid telemetry values')
    energy = np.r_[0, np.cumsum((p[:-1] + p[1:]) / 2 * np.diff(t))]
    return t - t[0], energy


def plot_energy_comparison(series, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for controller, telemetry in series.items():
        t, energy = cumulative_energy(telemetry)
        ax.plot(t, energy, label=controller, color=COLORS.get(controller, '#808080'))
    ax.set_xlabel('Elapsed telemetry time (s)')
    ax.set_ylabel('Cumulative GPU energy (J)')
    ax.set_title('Cumulative GPU Energy — Same Telemetry Delay')
    ax.legend()
    ax.grid(alpha=.3)
    save(fig, out, 'fig4_cumulative_energy.png')


def rolling_burn(requests, slo_target, window_s=60, budget_ratio=.01):
    require_columns(requests, ['finish_t', 'latency_s'], 'requests')
    data = requests[['finish_t', 'latency_s']].dropna().sort_values('finish_t')
    if data.empty:
        raise ValueError('No completed requests')
    times = data['finish_t'].to_numpy(dtype=float)
    misses = (data['latency_s'].to_numpy(dtype=float) > slo_target).astype(int)
    if not np.isfinite(times).all():
        raise ValueError('Invalid request timestamps')
    prefix = np.r_[0, np.cumsum(misses)]
    left = np.searchsorted(times, times - window_s, side='left')
    counts = np.arange(1, len(times) + 1) - left
    burn = (prefix[1:] - prefix[left]) / counts / budget_ratio
    return times - times[0], burn


def plot_burn_comparison(series, out, run_id, slo_target):
    fig, ax = plt.subplots(figsize=(9, 5))
    for controller, requests in series.items():
        t, burn = rolling_burn(requests, slo_target)
        ax.plot(t, burn, label=controller, color=COLORS.get(controller, '#808080'))
    ax.axhline(1, color='red', linestyle='--', label='Allowed budget')
    ax.set_xlabel('Elapsed request time (s)')
    ax.set_ylabel('Error budget burn (× allowed rate)')
    ax.set_title('Rolling SLO Error Budget Burn (60 s window)')
    ax.legend()
    ax.grid(alpha=.3)
    save(fig, out, f'fig5_burn_{run_id}.png')


def plot_oracle_gap(df, out):
    require_columns(df, ['controller', 'j_per_token'], 'Figure 6')
    mean = df.groupby('controller')['j_per_token'].mean()
    reference = 'oracle' if 'oracle' in mean.index else 'static_efficient'
    if reference not in mean.index or not np.isfinite(mean[reference]) or mean[reference] <= 0:
        raise ValueError('Missing or invalid reference energy')
    gap = ((mean / mean[reference]) - 1).mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(gap.index, gap.values, color=[COLORS.get(c, '#808080') for c in gap.index], label='Energy gap')
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xlabel('Controller')
    ax.set_ylabel('Energy gap (%)')
    ax.set_title(f'Energy Gap Relative to {reference}')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    ax.grid(axis='y', alpha=.3)
    save(fig, out, 'fig6_oracle_gap.png')


def plot_failure_map(df, out):
    require_columns(df, ['controller', 'delay_ms', 'miss_rate'], 'Figure 7')
    grouped = df.pivot_table(index='delay_ms', columns='controller', values='miss_rate', aggfunc='mean').sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'burst_multiplier' not in df or df['burst_multiplier'].nunique() <= 1:
        grouped.mul(100).plot(kind='bar', ax=ax, width=.8,
                              color=[COLORS.get(c, '#808080') for c in grouped.columns])
        ax.set_ylabel('SLO miss rate (%)')
        ax.set_xlabel('Telemetry delay (ms)')
        ax.set_title('SLO Failure Map by Controller and Delay')
        ax.legend(title='Controller')
        ax.tick_params(axis='x', rotation=0)
    else:
        heatmap = grouped.T.mul(100)
        im = ax.imshow(heatmap.to_numpy(), aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(heatmap.columns)), labels=heatmap.columns)
        ax.set_yticks(range(len(heatmap.index)), labels=heatmap.index)
        ax.set_xlabel('Telemetry delay (ms)')
        ax.set_ylabel('Controller')
        ax.set_title('SLO Failure Map (Averaged Across Burst Levels)')
        fig.colorbar(im, ax=ax, label='Mean SLO miss rate (%)')
    save(fig, out, 'fig7_failure_map.png')


def plot_ablation(df, out):
    controllers = ['robust', 'robust_no_age', 'robust_no_budget']
    require_columns(df, ['controller', 'delay_ms', 'j_per_token', 'miss_rate'], 'Figure 8')
    subset = df[df['controller'].isin(controllers)]
    missing = set(controllers) - set(subset['controller'])
    if missing:
        warnings.warn(f'Skipping Figure 8: missing controllers {sorted(missing)}')
        return
    energy = subset.pivot_table(index='delay_ms', columns='controller', values='j_per_token', aggfunc='mean')
    miss = subset.pivot_table(index='delay_ms', columns='controller', values='miss_rate', aggfunc='mean').mul(100)
    delays = sorted(subset['delay_ms'].unique())
    x = np.arange(len(delays))
    width = .25
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for i, controller in enumerate(controllers):
        pos = x + (i - 1) * width
        a.bar(pos, energy.reindex(delays)[controller], width, label=controller, color=COLORS[controller])
        b.bar(pos, miss.reindex(delays)[controller], width, label=controller, color=COLORS[controller])
    a.set_ylabel('Energy per token (J/token)')
    a.set_title('Ablation Study: Energy Efficiency')
    a.legend()
    a.grid(axis='y', alpha=.3)
    b.set_xlabel('Telemetry delay (ms)')
    b.set_ylabel('SLO miss rate (%)')
    b.set_title('Ablation Study: SLO Violations')
    b.set_xticks(x, labels=delays)
    b.grid(axis='y', alpha=.3)
    save(fig, out, 'fig8_ablation.png')


def choose_matched_runs(df, results_dir):
    """Choose one complete set at the highest common delay, workload and seed."""
    require_columns(df, ['run_id', 'controller', 'delay_ms', 'workload'], 'run_summary')
    candidates = df[df['controller'].isin(MAIN) & (df['workload'] == 'sharp_burst')].copy()
    if candidates.empty:
        return None, {}
    # Prefer matched seed, if available. This avoids comparing unrelated random traces.
    keys = ['delay_ms', 'workload'] + (['seed'] if 'seed' in candidates else [])
    for _, group in candidates.groupby(keys, dropna=False):
        if not set(MAIN).issubset(set(group['controller'])):
            continue
        selected = {}
        for controller in MAIN:
            for run_id in sorted(group.loc[group['controller'] == controller, 'run_id'].astype(str)):
                if all((results_dir / f'{run_id}_{kind}.csv').is_file()
                       for kind in ('telemetry', 'requests')):
                    selected[controller] = run_id
                    break
        if len(selected) == len(MAIN):
            yield (group['delay_ms'].iloc[0], selected)


def main():
    args = parse_args()
    results_dir = args.results_dir
    path = results_dir / 'run_summary.csv'
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError('run_summary.csv is empty')
    out = results_dir / 'figures'
    print(f'Loaded {len(df)} experiment runs from {path}')

    plot_delay(df, out, 'j_per_token', 'fig1_delay_vs_jtoken.png',
               'Energy per output token (J/token)', 'Telemetry Delay vs. Energy per Token')
    plot_delay(df, out, 'miss_rate', 'fig2_delay_vs_miss.png',
               'SLO miss rate (%)', 'Telemetry Delay vs. SLO Miss Rate', budget=True)

    # Figure 3: highest-delay robust sharp_burst run with valid, aligned CSVs.
    robust = df[(df['controller'] == 'robust') & (df['workload'] == 'sharp_burst')]
    plotted = False
    for run_id in robust.sort_values(['delay_ms', 'run_id'], ascending=[False, True])['run_id'].astype(str):
        try:
            telemetry = read_run(results_dir, run_id, 'telemetry')
            requests = read_run(results_dir, run_id, 'requests')
            control = read_run(results_dir, run_id, 'control')
            plot_timeseries(telemetry, requests, control, out, run_id)
            plotted = True
            break
        except (FileNotFoundError, ValueError) as exc:
            print(f'Figure 3 candidate {run_id} unavailable: {exc}')
    if not plotted:
        print('Figure 3 skipped: no complete, timestamp-aligned robust run')

    # Figures 4 and 5: all four controllers at the same delay, workload and seed.
    matched = sorted(choose_matched_runs(df, results_dir), key=lambda item: item[0], reverse=True)
    for delay, selected in matched:
        try:
            telemetry = {c: read_run(results_dir, rid, 'telemetry') for c, rid in selected.items()}
            requests = {c: read_run(results_dir, rid, 'requests') for c, rid in selected.items()}
            # Validate all inputs before writing either comparison figure.
            for item in telemetry.values():
                cumulative_energy(item)
            for item in requests.values():
                rolling_burn(item, args.slo_target)
            plot_energy_comparison(telemetry, out)
            plot_burn_comparison(requests, out, selected['robust'], args.slo_target)
            print(f'Figures 4–5 compare delay={delay} ms, runs={selected}')
            break
        except (ValueError, FileNotFoundError) as exc:
            print(f'Figures 4–5 candidate at {delay} ms unavailable: {exc}')
    else:
        print('Figures 4–5 skipped: no complete matched four-controller dataset')

    plot_oracle_gap(df, out)
    plot_failure_map(df, out)
    plot_ablation(df, out)


if __name__ == '__main__':
    main()
