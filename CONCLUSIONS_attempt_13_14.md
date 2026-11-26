# Conclusions from attempt_13 and attempt_14 (3-seed GPU repeats)

attempt_15 (the third repeat) is still running; this note covers the first two.

## Setup

| | attempt_13 | attempt_14 |
|---|---|---|
| Runs | 105/105 (35 conditions x seeds 41, 42, 43) | 105/105 |
| Unfinished requests | 0 | 0 |
| Capacity probe at 2122 MHz | 7.94 req/s | 8.00 req/s |
| Base rate (auto-config) | 1.75 req/s | 1.80 req/s |
| Burst | x4 for 20 s | x4 for 20 s |
| SLO / budget | 5.0 s / 5 % | 5.0 s / 5 % |
| Wall time | 271 min | 272 min |

Everything else matches attempt_12: batch 16, clocks 180-2122 MHz, delays 0/500/1250/2500 ms,
workloads steady / ramp / sharp_burst, ablations (robust_v1, robust_no_age) on sharp_burst.

Pooled below means 6 paired replicates per condition (2 attempts x 3 seeds). The two attempts
reuse seeds 41-43, so a seed's arrival pattern is the same in both apart from the 0.05 req/s
difference in base rate. The replicates are not fully independent workloads, so treat
"n = 6" as 3 workloads measured twice.

## Headline: sharp_burst, robust vs naive (pooled, 6 pairs per delay)

| Delay | Burst-phase misses, naive | Burst-phase misses, robust | Energy (J/token), robust vs naive | Time at 2122 MHz, naive -> robust |
|---|---|---|---|---|
| 0 ms | 103/881 (11.7 %) | 5/881 (0.6 %) | -5.8 % (6/6 lower) | 45 % -> 22 % |
| 500 ms | 169/881 (19.2 %) | 4/881 (0.5 %) | -7.9 % (6/6 lower) | 57 % -> 24 % |
| 1250 ms | 231/881 (26.2 %) | 19/881 (2.2 %) | -5.5 % (6/6 lower) | 56 % -> 29 % |
| 2500 ms | 437/881 (49.6 %) | 49/881 (5.6 %) | -5.3 % (6/6 lower) | 55 % -> 28 % |

Fisher exact test on burst-phase misses (request level), naive vs robust: p < 1e-25 at every
delay. Run-level tests, which are the honest unit:

* Energy: robust used less energy than naive in **24 of 24** sharp_burst pairs (sign test
  p = 1.2e-7); mean -6.1 % (SD 2.8).
* Burst-phase miss rate: robust lower in 22 of 24 pairs, tied (both 0) in 2, never higher;
  mean -23.1 percentage points (Wilcoxon p = 4e-5).
* Per delay, the paired Wilcoxon on burst misses reaches p = 0.031 at 500, 1250 and 2500 ms
  (the smallest p possible with 6 pairs); energy t-test p = 0.014 / <0.001 / 0.003 / 0.005.

Per replicate (burst-phase miss %, naive -> robust):

| | d0 | d500 | d1250 | d2500 |
|---|---|---|---|---|
| a13 s41 | 6.2 -> 0 | 10.4 -> 0 | 18.8 -> 0.7 | 36.8 -> 0 |
| a13 s42 | 0 -> 0 | 0.8 -> 0 | 2.3 -> 0 | 19.5 -> 0 |
| a13 s43 | 20.1 -> 0 | 32.1 -> 0 | 44.7 -> 0 | 83.6 -> 0 |
| a14 s41 | 2.7 -> 0 | 8.1 -> 0 | 10.1 -> 0 | 34.9 -> 2.7 |
| a14 s42 | 0 -> 0 | 0.7 -> 0 | 6.6 -> 0 | 13.9 -> 2.2 |
| a14 s43 | 35.4 -> 3.0 | 54.3 -> 2.4 | 64.6 -> 11.0 | 94.5 -> 25.6 |

Seed 43 is the hardest workload: its burst brings 159-164 arrivals (vs 128-149 for the others),
and even static_high reaches a burst P95 of 4.2-4.3 s, close to the 5 s SLO. On that seed naive
fails badly at every delay, including 0 ms. Robust holds it at 0 % in attempt_13 and degrades
in attempt_14 (base rate 1.80) only at 1250-2500 ms, where it is still 4-6x better than naive.

## Steady and ramp

| Workload | Energy, robust vs naive (pooled mean) | Misses |
|---|---|---|
| steady | -0.5 to +0.6 % (no delay significant) | 0 for both |
| ramp | +0.6, +1.9, +1.0, +3.0 % at d0/500/1250/2500 | <= 0.25 % for both |

Steady: no difference. Ramp: robust costs a little more energy, and at 2500 ms the cost is
consistent (6/6 pairs higher, +3.0 %, t-test p = 0.015, range +0.5 to +6.4 %). The age
compensation plans ahead on a gentle ramp that naive handles anyway, so it buys nothing there.
This is the price of the robustness and belongs in the report.

## Savings vs static_high (pooled mean J/token)

| Workload | naive | robust |
|---|---|---|
| steady | -26.5 to -27.3 % | -26.5 to -26.9 % |
| ramp | -20.0 to -20.6 % | -17.8 to -19.6 % |
| sharp_burst | -8.8 to -11.9 % | **-14.7 to -17.1 %** |

On sharp_burst, robust both saves more energy than naive and misses far fewer requests. Naive's
savings come with 5.4-24 % overall misses (over the 5 % budget at every delay); robust stays at
0.2-2.6 % overall.

## Ablation (sharp_burst, pooled burst-phase misses)

| Delay | naive | robust_v1 | robust_no_age | robust (v2) |
|---|---|---|---|---|
| 0 ms | 103/881 | 98/881 | 33/881 | 5/881 |
| 500 ms | 169/881 | 170/881 | 97/881 | 4/881 |
| 1250 ms | 231/881 | 233/881 | 118/881 | 19/881 |
| 2500 ms | 437/881 | 431/881 | 278/881 | 49/881 |
| J/token at 2500 ms | 0.437 | 0.436 | 0.420 | 0.414 |
| Clock changes at 2500 ms | 4 | 4 | 15 | 7.7 |

* **robust_v1 behaves like naive** (misses within noise, Fisher p > 0.7; energy +0.4 %). The
  first robust design did not fix the problem.
* **robust_no_age** (v2 without age compensation) removes 36-68 % of naive's
  misses: the lag-aware P95 and queue signal help on their own (Fisher p < 1e-5 vs naive).
* **Age compensation does the rest:** full v2 has 6-24x fewer burst misses than no_age, fewer clock
  changes at high delay (7.7 vs 15), and lower energy. Robust beat no_age on burst misses in
  20/24 pairs, never worse (4 ties).

## What we can now conclude (n = 6 replicates, 2 attempts)

1. **Stale telemetry breaks the naive controller on bursts, and the damage grows with delay.**
   Naive burst-phase misses rise 11.7 -> 19.2 -> 26.2 -> 49.6 % from 0 to 2500 ms.
2. **The robust controller fixes it and saves energy at the same time** on sharp_burst:
   burst misses cut by 90-97 % at every delay, energy -5 to -8 %, lower in 24/24 pairs.
   This is no longer a single-seed trend; it is significant at the run level.
3. **The gain comes from age compensation plus the lag-aware signals together**; v1 alone does
   nothing, and removing age compensation loses most of the benefit.
4. **No free lunch on smooth load:** steady is a wash and ramp costs up to +3 % energy at 2500 ms.
5. Compared with always-max clock, robust saves 15-17 % on bursts and 18-27 % on smooth load.

## Changes vs attempt_12

attempt_12 (seed 42 only) showed naive at d2500 missing 11.7 % of burst requests. The same seed
here gives 19.5 % and 13.9 %, so attempt_12 was a fair single sample; seeds 41 and 43 are harder
and make the naive failure much larger. attempt_12's "trend, n = 1" findings are confirmed. The
one attempt_12 cell that looked odd (energy +1.8 % at d500) did not repeat: d500 is now the
largest saving (-7.9 %).

## Simulator cross-check (20 seeds per attempt)

One worker: robust vs naive on sharp_burst, energy -4 to -8 %, misses -0.05 to -7 pp, robust
fewer misses in most seeds at 1250-2500 ms. Ramp/steady energy +0 to +4 % (same sign as GPU).
Eight workers: -8 to -14 % energy on sharp_burst, -7 to -9 % on steady, -2 % on ramp, with
miss changes within +/-1.3 pp. The GPU and simulator agree in direction.

## Still open

* attempt_15 will add a third replicate (9 pairs per condition).
* The failure map still uses one burst size (x4); D6 remains partial without a burst-size axis.
* Seed 43 at base rate 1.80 is close to capacity; robust starts to miss there at high delay.
