# Zen 4 — batch-gap check (L7), 2026-11-24

Does PARALLEL still beat VECTORIZED when batches do not arrive back to back?
`tools/bench/batch_gap_bench.cpp`, three runs, each its own process. Why this
is measured: `PLAN.md`, "Pool wake-up and batch spacing".

## Conditions

- Asus TUF A16, Ryzen 7 7445HS (6 cores / 12 threads, AVX-512), Windows 11 Pro
  10.0.26300, power scheme Performance, on AC.
- `dev/v3.0.0-corvus-adoption` at `164cbce` (last code change `e4026b3`),
  clang-cl 22.1.3, the flags and definitions of the library's own TUs
  (`/EHsc /O2 /Ob2 /DNDEBUG -std:c++20 -MD /arch:AVX512`, the four
  `LIBSTATS_HAS_*`, `NOMINMAX`, `_USE_MATH_DEFINES`), linked against that
  commit's Release `stats_static.lib`, corvus v1.2.0 and Highway 1.4.0. The
  bench compiled under clang-cl without a diagnostic. It was not compiled
  with `cl.exe`.
- One detached script (`quiet.ps1.txt`, phase `l7`), the first phase of the
  L4 sitting. Before each run: total CPU below 8% (5.5%, 3.0%, 3.6%), a 10-s
  clock probe (127.0–127.4 ms per pass each time; boost is 127–130, base clock
  about 190), a 25-s discarded pass of the same binary. Every process was
  launched directly and opted out of Windows power throttling (`run.txt`:
  `control=1 state=0`).
- `noise.txt`: the three busiest other processes once a minute, in cores. The
  largest sum in any sample is 0.05 of a core.
- Default busy gap. AUTO selects work stealing only on hyper-threaded macOS
  (the bench's header), so the AUTO column of `gap_init_auto.csv` is not
  work stealing here.

## Result

Smallest measured size from which PARALLEL wins at every larger size (the `#`
lines of each CSV). Sizes measured: 4096, 10000, 25000, 50000, 100000, 200000.

| Gap | Poisson PDF, plain | Poisson PDF, `--init` | Gaussian PDF, plain | Gaussian PDF, `--init` |
|---|---|---|---|---|
| 0 | 10000 | 10000 | 50000 | 50000 |
| 50 µs | 10000 | 10000 | 50000 | 50000 |
| 500 µs | 10000 | 10000 | 50000 | 50000 |
| 2 ms | 4096 (see below) | 10000 | 100000 | 100000 |
| 20 ms | 10000 | 10000 | 100000 | 100000 |

The 4096 for Poisson at 2 ms, plain, is not a crossover. PARALLEL does not
fork at 4096 on this machine (the fork value is 8192), so both columns run the
same code there and read within 0.3% of each other (10.60 against 10.57
ns/element); the `#` line counts that as a win. Poisson's crossover is 10000,
the first measured size that forks, at every gap.

ns/element, Poisson PDF at 10000 elements:

| Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` | AUTO (`--init --auto`) |
|---|---|---|---|---|---|
| 0 | 10.49 | 10.40 | 5.81 | 5.81 | 5.85 |
| 50 µs | 10.50 | 10.41 | 5.93 | 5.96 | 5.95 |
| 500 µs | 10.52 | 10.42 | 6.18 | 6.19 | 6.24 |
| 2 ms | 10.52 | 11.23 | 7.81 | 7.24 | 8.09 |
| 20 ms | 11.84 | 12.96 | 11.25 | 10.65 | 11.71 |

ns/element, Gaussian PDF at 50000 elements:

| Gap | VECTORIZED plain | VECTORIZED `--init` | PARALLEL plain | PARALLEL `--init` | AUTO (`--init --auto`) |
|---|---|---|---|---|---|
| 0 | 0.92 | 0.92 | 0.60 | 0.60 | 0.60 |
| 50 µs | 0.92 | 0.92 | 0.59 | 0.60 | 0.59 |
| 500 µs | 0.92 | 0.92 | 0.65 | 0.65 | 0.65 |
| 2 ms | 0.93 | 0.92 | 1.13 | 1.15 | 1.03 |
| 20 ms | 1.12 | 1.09 | 1.58 | 1.52 | 1.31 |

## Reading

- **The crossover moves with the gap for Gaussian, by one measured size.** At
  50000 elements PARALLEL is 0.65× VECTORIZED back to back and at 500 µs, and
  1.2–1.4× from 2 ms: it costs 1.9× more per element at 2 ms and 2.6× more at
  20 ms than back to back. Poisson's crossover stays at 10000, but its margin
  goes the same way: PARALLEL is 0.55× VECTORIZED at gap 0 and 0.95× at 20 ms.
- **`--init` changes nothing here.** Forced PARALLEL is within 8% in the two
  runs at every gap in the two tables above; over all sizes and gaps the
  largest difference is 19% (Gaussian at 10000, 20 ms). Starting the work-stealing pool does not
  keep the thread pool's workers fast on this machine, unlike K, where
  `--init` removed the delay.
- **VECTORIZED also slows at a 20 ms gap** (Poisson 10.5 → 11.8–13.0,
  Gaussian 0.92 → 1.09–1.12 ns/element): the calling thread's own core is
  slower after 20 ms of the gap loop. No other machine's README reports this.
- Nothing here is unlike what `PLAN.md` describes for the other machines in
  kind: a cost that appears between 500 µs and 2 ms of gap and grows to 20 ms.
  The bench does not say whether that is a late start or a slow run on Z.
  `scripts/fit_gap_model.py` and `pool_wake_bench` were not run.
