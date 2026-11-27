# Zen 4: scalar quantiles against corvus v1.2.0 and v1.2.1

2026-11-27, Asus TUF A16 (Ryzen 7 7445HS, Windows 11), clang-cl 22.1.3,
Release, AVX-512. Power plan Turbo, screen locked. `tools/bench/
distributions_bench.cpp` built against two builds of this branch at `44d10d9`
(library sources identical):

- `c120`: corvus v1.2.0, the pinned release.
- `c121`: corvus `dev/v1.2.1` at `64629ed` (OldCrow/corvus#117, not yet
  tagged), through `FETCHCONTENT_SOURCE_DIR_CORVUS`. No file in this
  repository was changed for it.

Three alternating rounds after a discarded warm-up, each process launched
directly and opted out of Windows power throttling, with
`LIBSTATS_BENCH_WARMUP_SECONDS=25`. `run.log` has the gate and clock probe of
every launch (gates 2.1 to 6.2%, probes 690 to 803 ms); `quiet.ps1.txt` is the
script. The same sitting ran the pool measurements recorded on
`proto/v3.0.0-pool-wake`.

## Scalar quantile, ns per call

| | v2.5.1 (2026-11-24 quiet run) | corvus v1.2.0 | corvus v1.2.1 |
|---|---:|---:|---:|
| Gaussian | 72 | 105.7 to 106.0 | 75.9 to 76.5 |
| LogNormal | 76 | 116.4 to 116.6 | 85.4 to 85.6 |
| von Mises | | 407.1 to 408.5 | 386.1 to 387.7 |
| Student's t | | 2,323 to 2,328 | 2,258 to 2,260 |

Of the 70 cells the bench prints (ten distributions, seven columns), only
the Gaussian, LogNormal and von Mises quantiles differ between the two builds
by more than 3% (median of three rounds).

## What it shows

corvus v1.2.1 removes 30 ns from each `erfcinv` call libstats makes one value
at a time. The Gaussian quantile is then 6% above v2.5.1 and the LogNormal
12%, inside the 1.25x guard of #156. corvus's own bench shows a larger gain
on the same call (166 to 67 ns); the difference is not explained here. One
candidate: corvus v1.2.1 also fixes a cost that depends on where the linker
places a constant, and this program's link may not have had it.
