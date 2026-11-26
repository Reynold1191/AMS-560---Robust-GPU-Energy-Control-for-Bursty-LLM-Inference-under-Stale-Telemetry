# Conclusions from attempt_13, attempt_14 and attempt_15 (3-seed GPU repeats)

This note supersedes `CONCLUSIONS_attempt_13_14.md`. It pools all three repeat sessions:
315 GPU runs and 9 paired replicates per condition.

## Setup

| | attempt_13 | attempt_14 | attempt_15 |
|---|---|---|---|
| Runs | 105/105 (35 conditions x seeds 41, 42, 43) | 105/105 | 105/105 |
| Unfinished requests | 0 | 0 | 0 |
| Capacity probe at 2122 MHz | 7.94 req/s | 8.00 req/s | 8.00 req/s |
| Base rate (auto-config) | 1.75 req/s | 1.80 req/s | 1.80 req/s |
| Burst | x4 for 20 s | x4 for 20 s | x4 for 20 s |
| SLO / budget | 5.0 s / 5 % | 5.0 s / 5 % | 5.0 s / 5 % |
| Wall time | 271 min | 272 min | 272 min |

Everything else matches attempt_12: batch 16, clocks 180-2122 MHz, delays 0/500/1250/2500 ms,
workloads steady / ramp / sharp_burst, ablations (robust_v1, robust_no_age) on sharp_burst.

"Pooled" below means 9 paired replicates per condition (3 attempts x 3 seeds). The three attempts
reuse seeds 41-43, so a seed's arrival pattern is the same in each apart from the 0.05 req/s
difference in base rate. The replicates are not fully independent workloads. Read "n = 9" as
3 workloads, each measured three times on the GPU.

## Headline: sharp_burst, robust vs naive (pooled, 9 pairs per delay)

| Delay | Burst-phase misses, naive | Burst-phase misses, robust | Reduction | Energy (J/token), robust vs naive | Time at 2122 MHz, naive -> robust |
|---|---|---|---|---|---|
| 0 ms | 177/1331 (13.3 %) | 25/1331 (1.9 %) | -86 % | -5.9 % (9/9 lower) | 49 % -> 24 % |
| 500 ms | 267/1331 (20.1 %) | 9/1331 (0.7 %) | -97 % | -7.6 % (9/9 lower) | 57 % -> 25 % |
| 1250 ms | 368/1331 (27.6 %) | 33/1331 (2.5 %) | -91 % | -5.3 % (9/9 lower) | 56 % -> 30 % |
| 2500 ms | 658/1331 (49.4 %) | 75/1331 (5.6 %) | -89 % | -4.7 % (8/9 lower) | 55 % -> 30 % |

Fisher exact test on burst-phase misses (request level), naive vs robust: p < 1e-30 at every
delay. The run-level tests below are the honest unit of analysis:

* Energy: robust used less energy than naive in **35 of 36** sharp_burst pairs (sign test
  p = 1.1e-9). Mean -5.9 % (SD 2.8). The one exception is a15 s43 at 2500 ms (+2.3 %), the
  hardest cell, where robust ran at the top clock longer to fight the backlog.
* Burst-phase miss rate: robust was lower in 32 of 36 pairs and tied (both 0) in 4. It was
  never higher. Mean change -23.5 percentage points (Wilcoxon p = 8e-7).
* Per delay, the paired Wilcoxon on burst misses gives p = 0.031 / 0.008 / 0.004 / 0.004.
  0.004 is the smallest p possible with 9 pairs. The energy t-test gives
  p < 0.001 / < 0.001 / < 0.001 / 0.004.

Per replicate (burst-phase miss %, naive -> robust):

| | d0 | d500 | d1250 | d2500 |
|---|---|---|---|---|
| a13 s41 | 6.2 -> 0 | 10.4 -> 0 | 18.8 -> 0.7 | 36.8 -> 0 |
| a13 s42 | 0 -> 0 | 0.8 -> 0 | 2.3 -> 0 | 19.5 -> 0 |
| a13 s43 | 20.1 -> 0 | 32.1 -> 0 | 44.7 -> 0 | 83.6 -> 0 |
| a14 s41 | 2.7 -> 0 | 8.1 -> 0 | 10.1 -> 0 | 34.9 -> 2.7 |
| a14 s42 | 0 -> 0 | 0.7 -> 0 | 6.6 -> 0 | 13.9 -> 2.2 |
| a14 s43 | 35.4 -> 3.0 | 54.3 -> 2.4 | 64.6 -> 11.0 | 94.5 -> 25.6 |
| a15 s41 | 1.3 -> 0 | 12.1 -> 0 | 12.8 -> 0 | 33.6 -> 6.0 |
| a15 s42 | 0 -> 0 | 0 -> 0 | 6.6 -> 0 | 13.9 -> 0 |
| a15 s43 | 43.9 -> 12.2 | 48.8 -> 3.0 | 66.5 -> 8.5 | 92.7 -> 10.4 |

Seeds 41 and 42 behave the same in all three sessions. Robust is at or near 0 at every delay,
with at most 6 % at 2500 ms, while naive climbs to 14-37 %.

Seed 43 is the hardest workload. Its burst brings 159-164 arrivals, against 128-149 for the
other seeds, and even static_high reaches a burst P95 of 4.2-4.3 s, close to the 5 s SLO. On
that seed naive fails badly at every delay, including 0 ms (20-44 %). At base rate 1.75
(attempt_13) robust holds seed 43 at 0 %. At base rate 1.80 (attempt_14 and attempt_15) it
also misses: 3-12 % at low delay and 8-26 % at 1250-2500 ms. That is still 4-9x better than
naive.

**Showcase replicate.** The single-run pages of the dashboard and the README replay use
attempt_13 seed 41. With all 9 replicates available it is still the clearest typical example:
naive rises steadily (6.2 -> 10.4 -> 18.8 -> 36.8 %), robust stays at about 0 % at every delay,
robust is 6-10 % cheaper at every delay, and the ablation order is clean. It is neither the
best replicate (that is a13 s43) nor the worst (a14 s43).

## Steady and ramp

| Workload | Energy, robust vs naive (pooled mean at d0/500/1250/2500) | Misses |
|---|---|---|
| steady | -0.4, +0.5, +0.4, +0.9 % (no delay significant) | 0 for both |
| ramp | +0.5, +1.3, +1.4, +2.7 % | <= 0.73 % for both |

On steady traffic there is no difference. On ramp traffic robust costs a little more energy,
and at 2500 ms the cost is consistent: higher in 9 of 9 pairs, +2.7 % on average (t-test
p = 0.002, range +0.5 to +6.4 %). Age compensation plans ahead on a gentle ramp that naive
handles anyway, so it buys nothing there. This is the price of the robustness and belongs in
the report.

## Savings vs static_high (pooled mean J/token)

| Workload | naive | robust |
|---|---|---|
| steady | -27.4 to -28.0 % | -27.1 to -27.7 % |
| ramp | -20.0 to -20.4 % | -18.0 to -19.6 % |
| sharp_burst | -9.3 to -11.3 % | **-14.7 to -16.6 %** |

On sharp_burst, robust saves more energy than naive and also misses far fewer requests. Naive's
savings come with 6.2-24.1 % overall misses, over the 5 % budget at every delay. Robust stays at
0.3-2.6 % overall.

## Ablation (sharp_burst, pooled burst-phase misses)

| Delay | naive | robust_v1 | robust_no_age | robust (v2) |
|---|---|---|---|---|
| 0 ms | 177/1331 | 169/1331 | 83/1331 | 25/1331 |
| 500 ms | 267/1331 | 259/1331 | 122/1331 | 9/1331 |
| 1250 ms | 368/1331 | 376/1331 | 165/1331 | 33/1331 |
| 2500 ms | 658/1331 | 652/1331 | 442/1331 | 75/1331 |
| J/token at 2500 ms | 0.435 | 0.434 | 0.420 | 0.414 |
| Clock changes at 2500 ms | 4 | 4 | 15.4 | 7.8 |

* **robust_v1 behaves like naive.** Its misses are within noise (Fisher p > 0.69 at every
  delay) and its energy is within 2 %. The first robust design did not fix the problem.
* **robust_no_age** (v2 without age compensation) removes 33-55 % of naive's misses. The
  lag-aware P95 and queue signal help on their own (Fisher p < 1e-9 vs naive).
* **Age compensation does the rest.** Full v2 has 3-14x fewer burst misses than no_age, half
  the clock changes at high delay (7.8 vs 15.4) and lower energy. Robust beat no_age on burst
  misses in 31 of 36 pairs and was never worse (5 ties).

## What we can now conclude (n = 9 replicates, 3 attempts)

1. **Stale telemetry breaks the naive controller on bursts, and the damage grows with delay.**
   Naive burst-phase misses rise 13.3 -> 20.1 -> 27.6 -> 49.4 % from 0 to 2500 ms.
2. **The robust controller fixes it and saves energy at the same time** on sharp_burst. Burst
   misses fall by 86-97 % at every delay, and energy falls by 5-8 %, lower in 35 of 36 pairs.
   The third session reproduces the first two, so this is a stable, run-level result.
3. **The gain comes from age compensation and the lag-aware signals together.** v1 alone does
   nothing, and removing age compensation loses most of the benefit.
4. **There is no free lunch on smooth load.** Steady is a wash, and ramp costs up to +2.7 %
   energy at 2500 ms.
5. Compared with always running at the maximum clock, robust saves 15-17 % on bursts and
   18-28 % on smooth load.
6. **Near capacity robust degrades as well.** At base rate 1.80 the hardest seed pushes robust
   to 8-26 % burst misses at 1250-2500 ms. It is still far better than naive there (65-95 %).

## What attempt_15 changed

attempt_15 used the same configuration as attempt_14. Its pairs agree with the first two
sessions in direction everywhere on sharp_burst. The pooled numbers moved only slightly:

* The burst-miss reduction is 86-97 %, compared with 89-98 % over 6 replicates. Robust at 0 ms
  is now 1.9 % instead of 0.6 %, because a15 s43 missed 12 % at 0 ms.
* The energy saving is -5.9 %, compared with -6.1 %. The only pair where robust cost more is
  a15 s43 at 2500 ms (+2.3 %).
* The ramp cost at 2500 ms is +2.7 %, now 9 of 9 pairs (p = 0.002).
* Every run-level p-value got smaller because there are 9 pairs instead of 6.

## Changes vs attempt_12

attempt_12 used seed 42 only and showed naive at d2500 missing 11.7 % of burst requests. The
same seed gives 19.5 %, 13.9 % and 13.9 % in the three repeats, so attempt_12 was a fair single
sample. Seeds 41 and 43 are harder and make the naive failure much larger. All of attempt_12's
"trend, n = 1" findings are confirmed.

## Simulator cross-check (20 seeds per attempt)

* **One worker.** Robust vs naive on sharp_burst: energy -4 to -8 %, misses -0.05 to -7 pp, and
  robust has fewer misses in most seeds at 1250-2500 ms. On ramp and steady, energy is +0 to
  +4 %, the same sign as on the GPU.
* **Eight workers.** Energy changes by -8 to -14 % on sharp_burst, -7 to -9 % on steady and
  -2 % on ramp, with miss changes within +/-1.3 pp.

The GPU and the simulator agree in direction. attempt_14 and attempt_15 share a base rate and
give the same simulator output.

## Still open

* The failure map still uses one burst size (x4), so D6 remains partial without a burst-size
  axis.
* Seed 43 at base rate 1.80 is close to capacity, and robust starts to miss there at high delay.
