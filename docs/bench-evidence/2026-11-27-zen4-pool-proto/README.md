# Zen 4: the pool prototype at 6 and 12 threads, and wake latency against pinning

2026-11-27, Asus TUF A16 (Ryzen 7 7445HS, 6 cores / 12 threads, Windows 11),
clang-cl 22.1.3, Release, `proto/v3.0.0-pool-wake` at `974704a`. Power plan
Turbo, screen locked. Every measured process was launched directly and opted
out of Windows power throttling; a gate (total CPU under 8% in two 5-second
samples) and a clock probe precede each one. `run.log` has both for every
launch: gates 2.1 to 6.2%, probes 690 to 803 ms. `quiet.ps1.txt` is the
script; the same sitting also ran the quantile bench recorded on the dev
branch.

## Thread count (`r*_gap_t6.csv`, `r*_gap_t12.csv`)

`batch_gap_bench --dist Poisson,GammaCDF --gaps 0,2000,20000` at six sizes,
`LIBSTATS_THREAD_COUNT` 6 and 12, two rounds each. PARALLEL as a fraction of
VECTORIZED time (mean of two rounds):

| Row | Gap | 10,000 | 25,000 | 50,000 | 100,000 | 200,000 |
|---|---|---|---|---|---|---|
| Poisson PDF, 6 threads | none | 0.43 | 0.35 | 0.28 | 0.27 | 0.26 |
| Poisson PDF, 12 threads | none | 0.60 | 0.32 | 0.23 | 0.22 | 0.20 |
| Poisson PDF, 6 threads | 2 ms | 0.71 | 0.46 | 0.40 | 0.39 | 0.37 |
| Poisson PDF, 12 threads | 2 ms | 0.66 | 0.30 | 0.24 | 0.22 | 0.20 |
| Poisson PDF, 6 threads | 20 ms | 0.80 | 0.53 | 0.44 | 0.40 | 0.38 |
| Poisson PDF, 12 threads | 20 ms | 0.86 | 0.42 | 0.33 | 0.31 | 0.21 |
| Gamma CDF, 6 threads | none | 0.32 | 0.27 | 0.27 | 0.26 | 0.25 |
| Gamma CDF, 12 threads | none | 0.37 | 0.19 | 0.19 | 0.21 | 0.19 |
| Gamma CDF, 6 threads | 2 ms | 0.38 | 0.36 | 0.31 | 0.28 | 0.25 |
| Gamma CDF, 12 threads | 2 ms | 0.38 | 0.20 | 0.22 | 0.21 | 0.19 |
| Gamma CDF, 6 threads | 20 ms | 0.37 | 0.34 | 0.32 | 0.29 | 0.26 |
| Gamma CDF, 12 threads | 20 ms | 0.39 | 0.24 | 0.23 | 0.21 | 0.19 |

- From 25,000 elements up, 12 threads are faster than 6 at every gap, by
  1.1x to 1.9x.
- At 10,000, 6 threads are faster back to back (1.15x on Gamma CDF, 1.39x
  on Poisson); with a gap the two are level.
- PARALLEL is never slower than VECTORIZED from 10,000 up. Its worst case
  is 0.86 (Poisson, 10,000, 20 ms, 12 threads).
- At 4,096 PARALLEL equals VECTORIZED: the fixed gate does not fork (#307).

## Wake latency against worker pinning (`r*_wake.txt`)

`wake_probe.cpp.txt`: one task submitted to a `WorkStealingPool` after a busy
gap; latency from `submit()` to task entry. Four passes of 1,500 trials at a
2 ms gap and four of 300 at 20 ms, each way.

| Pinning | Median | Worst | Over 1 ms | Over 10 ms |
|---|---|---|---|---|
| on (the constructor's default) | 30 to 36 us | 25.4 ms | 74 of 7,200 | 10 |
| off | 29 to 36 us | 0.14 ms | 0 of 7,200 | 0 |

Pinning does not change the median. It adds a tail: a pinned worker can run
only on its one core and waits when that core is busy.

## Not covered

One machine. The `--init` and `--auto` pool states were not run. Sizes
between 10,000 and 25,000, where the better thread count changes over.
