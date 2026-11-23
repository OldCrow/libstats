# Kaby Lake — batch-gap check (L7), 2026-11-23

Does PARALLEL still beat VECTORIZED when batches do not arrive back to back?
`tools/bench/batch_gap_bench.cpp`, three runs, each its own process. Why this
is measured: `PLAN.md`, "Pool wake-up and batch spacing".

## Conditions

- MacBook Pro 14,1, i7-7820HQ (4 cores / 8 threads, AVX2+FMA), macOS 13.7.8.
- `dev/v3.0.0-corvus-adoption` at `c6ae2b9` (code `8c55bfc`), AppleClang 15,
  `-O3 -DNDEBUG -std=c++20 -stdlib=libc++` and the three `LIBSTATS_HAS_*`
  definitions, linked against that commit's Release `libstats.a`. The bench
  compiled without a diagnostic under `-Wall -Wextra`.
- Screen locked, AC power, detached script holding `caffeinate`
  (`quiet_run.sh.txt`). Each run started once the 1-minute load was below 2.0:
  1.83, 1.97 and 1.90 (`run.txt`); the loads at each end are the bench's own
  threads. `environment.txt` has the binary hashes.
- Default busy gap. On this machine AUTO selects WORK_STEALING above a row's
  threshold, so the AUTO column of `gap_init_auto.csv` is work stealing there.
- The same sitting re-ran the dev-against-`main` benches with the sustained
  timer: `../2026-11-22-kabylake-v300-cost/sustained/`.

## Result

Smallest measured size from which PARALLEL wins at every larger size:

| Gap | Poisson PDF, plain | Poisson PDF, `--init` | Gaussian PDF, plain | Gaussian PDF, `--init` |
|---|---|---|---|---|
| 0 | 4096 | 4096 | 50000 | 50000 |
| 50 µs | 4096 | 4096 | 50000 | 50000 |
| 500 µs | 10000 | 4096 | 50000 | 50000 |
| 2 ms | 10000 | 4096 | 50000 | 50000 |
| 20 ms | 25000 | 4096 | 100000 | 50000 |

ns/element, Poisson PDF at 4096 elements:

| Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` | AUTO (`--init --auto`) |
|---|---|---|---|---|---|
| 0 | 21.6 | 24.9 | 12.4 | 17.0 | 14.2 |
| 50 µs | 23.4 | 25.1 | 12.4 | 16.7 | 30.7 |
| 500 µs | 23.4 | 25.2 | 24.2 | 16.9 | 29.6 |
| 2 ms | 21.7 | 25.1 | 29.0 | 17.5 | 27.0 |
| 20 ms | 21.7 | 24.3 | 33.5 | 17.1 | 29.6 |

ns/element, Gaussian PDF at 50000 elements:

| Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` | AUTO (`--init --auto`) |
|---|---|---|---|---|---|
| 0 | 2.58 | 2.88 | 1.47 | 1.95 | 1.02 |
| 50 µs | 2.58 | 2.88 | 1.48 | 1.88 | 3.27 |
| 500 µs | 2.69 | 2.88 | 2.28 | 1.94 | 2.48 |
| 2 ms | 2.77 | 2.95 | 2.72 | 2.05 | 2.59 |
| 20 ms | 2.69 | 3.01 | 2.97 | 2.12 | 2.56 |

What it shows on this machine:

1. **The crossover moves with the gap when only the thread pool exists**, less
   than on Ivy Bridge. Poisson: 4096 back to back, 10000 from 500 µs, 25000 at
   20 ms. Gaussian: 50000 up to 2 ms (PARALLEL / VECTORIZED 0.57 back to back,
   0.98 at 2 ms), 100000 at 20 ms. Forced PARALLEL slows from 500 µs on: at
   4096 elements Poisson goes from 12.4 to 33.5 ns/element at 20 ms.
2. **With the work-stealing pool started, the gap does not matter** up to
   20 ms: Poisson 4096 and Gaussian 50000 at every gap.
3. **Back to back, forced PARALLEL is slower after `--init`**, not faster:
   1.3–1.5 times at 4096 to 50000 (Poisson 4096: 17.0 against 12.4). Ivy
   Bridge read the opposite (up to 3 times faster). With gaps of 2 ms and
   20 ms `--init` is faster, up to 2 times (Poisson 4096 at 20 ms: 17.1
   against 33.5). Over all cells PARALLEL `--init` / plain is 0.47–1.47,
   median 1.05.
4. **The polling costs the serial path about 9%**: VECTORIZED `--init` / plain
   is a median 1.09 (0.91–1.19).
5. **Work stealing (AUTO) beats forced PARALLEL back to back and loses to it
   once batches are spaced, below about 25000 (Poisson) and 200000
   (Gaussian).** `--init --auto` run, AUTO / PARALLEL: Poisson at gap 0 0.82
   (4096), 0.76 (10000); with any gap 1.6–1.9 at 4096 and 1.1–1.2 at 10000;
   from 25000 up 0.71–0.92 at every gap. Gaussian at 50000: 0.52 at gap 0,
   1.2–1.7 with a gap (at 50 µs AUTO is also slower than VECTORIZED, 3.27
   against 2.88); at 100000 0.62 at gap 0 and up to 1.34 with a gap; at 200000
   0.74–1.06. Both columns have the polling pool alive, so the back-to-back
   win is not the polling.
6. **Not described for Ivy Bridge: VECTORIZED itself slows with the gap on
   small Gaussian batches, in both pool states.** Gaussian at 4096: 2.3
   ns/element up to 500 µs, 4.6 at 2 ms and 20 ms (plain), 2.5 to 4.1–4.2
   (`--init`); at 10000, 2.3 to 3.3. Poisson VECTORIZED does not move. The
   cause is not measured here (the caller's own clock or cache state after the
   gap loop are candidates).

Not measured here: frequencies, gaps by sleeping instead of working, and any
distribution but these two.
