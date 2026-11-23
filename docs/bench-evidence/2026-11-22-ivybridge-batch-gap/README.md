# Ivy Bridge — batch-gap check (L7), 2026-11-22

Does PARALLEL still beat VECTORIZED when batches do not arrive back to back?
`tools/bench/batch_gap_bench.cpp`, three runs, each its own process. Why this
is measured: `PLAN.md`, "Pool wake-up and batch spacing".

## Conditions

- MacBook Pro 9,1, i7-3820QM (4 cores / 8 threads, AVX, no AVX2, no FMA),
  Linux Mint 22.3, governor `schedutil`.
- `dev/v3.0.0-corvus-adoption` at `7089876`, GCC 13.3, `-O3 -DNDEBUG
  -std=c++20 -DLIBSTATS_HAS_AVX=1 -DLIBSTATS_HAS_SSE2=1`, linked against the
  Release `libstats.a` of that commit.
- Quiet machine, run from a terminal with nothing else open; the script waited
  for a 1-minute load below 0.30 (0.28 at the start). `conditions.txt` has the
  load after each run; the later figures are the bench's own threads.
- Gap: serial work on the calling thread (a clock-reading loop). Only the time
  inside the batch call is counted; each cell runs 0.4 s unmeasured, then at
  least 1 s and 20 calls; median call.

## Files

| File | Pool state |
|---|---|
| `gap_plain.csv` | only the thread pool, created by the first PARALLEL batch |
| `gap_init.csv` | `initialize_performance_systems()` first, which also starts the work-stealing pool; its idle workers poll |
| `gap_init_auto.csv` | as `gap_init.csv`, with an AUTO column (AUTO never selects work stealing on Linux) |

## Result

Smallest measured size from which PARALLEL wins at every larger size
(sizes 4096, 10000, 25000, 50000, 100000, 200000):

| Gap | Poisson PDF, plain | Poisson PDF, `--init` | Gaussian PDF, plain | Gaussian PDF, `--init` |
|---|---|---|---|---|
| 0 | 25000 | 4096 | 25000 | 25000 |
| 50 µs | 25000 | 4096 | 100000 | 25000 |
| 500 µs | 50000 | 4096 | 200000 | 25000 |
| 2 ms | 50000 | 4096 | 200000 | 25000 |
| 20 ms | 50000 | 4096 | 200000 | 25000 |

ns/element at 10000 elements, Poisson PDF:

| Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` |
|---|---|---|---|---|
| 0 | 34.0 | 38.3 | 50.6 | 18.4 |
| 50 µs | 34.0 | 38.3 | 57.3 | 18.1 |
| 500 µs | 34.0 | 38.3 | 57.1 | 22.3 |
| 2 ms | 34.0 | 38.3 | 47.2 | 18.3 |
| 20 ms | 34.1 | 38.3 | 51.9 | 19.0 |

What it shows on this machine:

1. **The crossover moves with the gap when only the thread pool exists.**
   Poisson: 25000 back to back, 50000 from 500 µs. Gaussian: 25000 back to
   back, 200000 from 500 µs, and there PARALLEL is ahead by only 2–6%. Where
   the crossover is 50000, the margin at 50000 and 100000 is small with long
   gaps (PARALLEL / VECTORIZED 0.94–0.97 and 0.86–0.89 at 2 ms and 20 ms).
2. **With the work-stealing pool started, the gap does not matter**, up to the
   20 ms measured: both crossovers and the PARALLEL times are the same at every
   gap. Forced PARALLEL at the same gap is up to 3 times faster than in the
   plain run (Poisson, 10000: 47–57 against 18–22 ns/element).
3. **Plain and back to back, 10000 is slow and 4096 is not** (Poisson
   PARALLEL / VECTORIZED 0.57 at 4096, 1.49 at 10000, 0.43 at 25000): the
   low-clock state at sizes whose slices divide unevenly over four workers.
4. **The polling costs the serial path about 13%**: VECTORIZED reads a median
   1.13 times slower in the `--init` run than in the plain run, at every size
   and gap.
5. **AUTO follows forced PARALLEL** above its threshold in the `--init` run.
   Its worst cell against the better forced strategy is Gaussian at 25000
   (1.65 times, 50 µs gap), where the kAvx table still selects VECTORIZED.

Not measured here: frequencies (the trace that found the 1.2 GHz state was
taken with the machine in use), gaps by sleeping instead of working, and any
distribution but these two.
