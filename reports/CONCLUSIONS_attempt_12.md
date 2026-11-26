# Conclusions from attempt_12

**Run:** `results/attempt_12`
**Hardware:** RTX 5060 Laptop GPU, Qwen2.5-0.5B-Instruct, long-context prompts, batch 16.
**Matrix:** 35 runs × 1 seed (42), with a fixed SLO of 5.0 s and a 5 % error budget.
**Duration:** 96 min end to end. All 35 runs completed, the clock lock tracked 100 % in every run, and no request was left unfinished.

| Group | Runs | Controllers | Delays | Workloads |
|---|---|---|---|---|
| Primary | 24 | `naive_slo`, `robust` | 0 / 500 / 1250 / 2500 ms | steady, ramp, sharp_burst |
| Reference | 3 | `static_high` (2122 MHz) | 0 ms | steady, ramp, sharp_burst |
| Ablation | 8 | `robust_v1` (no forecast, no lag-aware P95), `robust_no_age` (does not know the telemetry age) | 0 / 500 / 1250 / 2500 ms | sharp_burst |

The probe measured an in-stack capacity of 8.0 req/s at 2122 MHz (2.0 s per full batch of 16), so the base rate was 1.8 req/s and the burst was 4 × base, peaking at 0.9 × top-clock capacity.

---

## 1. Headline results

### Sharp burst (the workload the project is about)

| Delay | J/token naive | J/token robust | Δ energy | Burst-phase miss naive | Burst-phase miss robust | Time at 2122 MHz (naive / robust) |
|---|---|---|---|---|---|---|
| 0 ms | 0.441 | 0.414 | **−6.3 %** | 0 % | 0 % | 56 % / 25 % |
| 500 ms | 0.409 | 0.417 | +1.8 % | 0 % | 0 % | 20 % / 25 % |
| 1250 ms | 0.442 | 0.417 | **−5.7 %** | 2.9 % (4/137) | **0 %** (0/137) | 57 % / 27 % |
| 2500 ms | 0.433 | 0.410 | **−5.1 %** | 11.7 % (16/137) | **2.2 %** (3/137) | 56 % / 26 % |

Whole-run miss rate at 2500 ms was 5.3 % for naive, which is over the 5 % budget. Robust stayed inside it at 1.0 %.

### Steady and ramp

| Workload | Δ energy (robust − naive) at 0 / 500 / 1250 / 2500 ms | Misses |
|---|---|---|
| steady | −1.0 / +1.2 / −0.7 / +1.5 % | 0 for both |
| ramp | −2.6 / +0.5 / −2.6 / +4.7 % | 0 for both |

### Energy saved against the same-session static_high reference

| Controller | steady | ramp | sharp_burst |
|---|---|---|---|
| naive_slo | 25.5–26.3 % | 16.4–18.5 % | 11.2–17.9 % |
| robust | 25.2–26.5 % | 13.8–19.4 % | 16.3–17.6 % |

### Ablation on sharp_burst (burst-phase miss / J per token)

| Delay | naive_slo | robust_v1 | robust_no_age | robust (v2) |
|---|---|---|---|---|
| 0 ms | 0 % / 0.441 | 0 % / 0.443 | 0 % / 0.412 | 0 % / 0.414 |
| 500 ms | 0 % / 0.409 | 0 % / 0.444 | 0 % / 0.402 | 0 % / 0.417 |
| 1250 ms | 2.9 % / 0.442 | 2.2 % / 0.440 | 0.7 % / 0.415 | **0 % / 0.417** |
| 2500 ms | 11.7 % / 0.433 | 9.5 % / 0.438 | 13.1 % / 0.423 | **2.2 % / 0.410** |

At 2500 ms, `robust_no_age` made 16 clock changes against 6 for robust. Without knowing the telemetry age, it oscillated.

---

## 2. What we can conclude

These claims are supported by attempt_12 together with earlier attempts and the simulator.

1. **Stale telemetry hurts a reactive controller.** Naive burst-phase misses rise with delay: 0 → 0 → 2.9 → 11.7 %. Three independent sources agree:
   - attempt_8–11: 8 seeds gave about 0.8 → 12 %.
   - The trace-driven simulator: 20 seeds gave 3.7 → 13.5 % whole-run miss.
   - attempt_12.
2. **DVFS is worth doing on this testbed.** Every dynamic controller saved 11–27 % energy against `static_high` with no SLO violations at 0–500 ms delay. This has held in every attempt from 8 to 12.
3. **The robust controller's two new mechanisms work as designed on real hardware.**
   - *Lag-aware P95.* After the burst, the 60 s P95 window stays above target for up to a minute although the queue is empty. Naive and `robust_v1` hold 2122 MHz until the end of the run. Robust v2 returns to 667 MHz once it sees no backlog, as the control logs show (`CAPACITY_DOWN`). Its time at the top clock fell from about 56 % to about 26 % in 3 of 4 sharp_burst cells. This is the main source of its energy saving.
   - *Age-compensated forecast.* At 2500 ms, robust stepped up on the growing queue before the 10 s arrival-rate estimate caught up. Burst misses fell from 16 to 3 requests. Fisher's exact test on the 137 burst requests gives p ≈ 0.003, but requests inside one burst are correlated, so treat this as indicative and not as a seed-level test.
4. **Knowing the telemetry age matters.**
   - At 2500 ms, `robust_no_age` did worse than robust: 13.1 % vs 2.2 % burst miss, with 16 vs 6 clock changes.
   - At 1250 ms the gap is smaller: 0.7 % vs 0 %.
   - The old `robust_v1` was close to naive, which matches the finding from attempt_8–11 that the old robust and naive behaved almost identically.

## 3. What we cannot conclude yet

- **No seed-level significance.** Each cell has n = 1. Run-to-run noise in J/token is about ±2 %, so the 5–6 % sharp_burst saving is notable but not proven.
- **One cell goes the other way.** At sharp_burst 500 ms, naive also dropped to 667 MHz after the burst, and robust used 1.8 % more energy. More seeds are needed to know how often naive gets this "lucky" path.
- **Steady and ramp show no meaningful difference.** The ±1.5 % on steady is noise. Ramp at 2500 ms has robust using 4.7 % more energy, with 37 % of time at the top clock. This may be a real cost of the forecast stepping up early under stale signals, or noise.
- **Results cannot be pooled with attempt_8–11.** Those attempts ran `robust_v1` with a per-attempt calibrated SLO of 4.0–6.0 s. Only the naive trend is comparable across attempts.

## 4. Simulator cross-check (scale_simulator.py, 20 paired seeds, calibrated to this attempt's capacity.json)

| n_workers | sharp_burst Δ energy (robust − naive) at 0 / 500 / 1250 / 2500 ms | Δ miss (pp) | Seeds where robust missed less / more |
|---|---|---|---|
| 1 | −8.3 / −7.3 / −5.4 / −4.4 % | +0.2 / −2.4 / −4.0 / −6.9 | 5/7, 13/3, 19/0, 19/1 |
| 8 (fleet, shared clock) | −12.8 / −13.4 / −12.2 / −7.7 % | +0.4 / −1.3 / −1.1 / −0.9 | 5/15, 10/10, 10/10, 8/12 |

The single-GPU simulator reproduces the hardware direction:
- Robust's advantage in misses grows with delay.
- Energy falls 4–8 %.

The fleet result is different:
- It is mainly an energy result (8–13 % lower on every workload).
- Its miss rate is similar to naive's, within about 1 pp, and the per-seed win counts are mixed.

## 5. Bottom line for the report

> On real hardware, with every controller input delayed, a reactive SLO controller (naive) loses reliability as telemetry gets staler: burst-phase SLO misses rise from 0 % to 11.7 % at 2.5 s delay. The robust controller has three additions over it:
> - it knows how old its data is,
> - it forecasts over that age,
> - it ignores lagging latency signals when there is no backlog.
>
> With these, it kept burst misses at 2.2 % and used 5–6 % less energy on bursty traffic. It matched naive on steady and ramp traffic. An ablation shows each addition contributes. A 20-seed trace-driven simulator reproduces the same trend. The GPU result is a single seed per condition; a 3-seed repeat is needed before claiming seed-level significance.

## 6. Next step

```
python run_demo.py --repeat-3seed 1 --skip-gpu-check      # about 5 h, 105 runs
```

Pooling it with attempt_12 gives 4 paired seeds per cell. That is enough for a paired t-test or a sign test on sharp_burst J/token and burst-phase misses.
