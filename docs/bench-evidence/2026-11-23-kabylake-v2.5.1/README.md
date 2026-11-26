# Kaby Lake — v2.5.1 fleet legs P1–P4 at 3dd8358 (2026-11-23)

The K machine's legs for the v2.5.1 patch release (`PLAN.md`, "v2.5.1 patch
release — fleet legs"). `main` at `3dd8358`, whose code is identical to the
validated commit `7069adc` (`git diff --stat 7069adc 3dd8358` over `src`,
`include`, `tests`, `tools`, `cmake`, `CMakeLists.txt` and `CMakePresets.json`
is empty).

MacBook Pro 14,1 (2017), i7-7820HQ (4 cores / 8 threads; AVX2+FMA), macOS
13.7.8 (22H730), CMake 4.5.1, Ninja, AC power. Work done in a separate
worktree of `main` with a new build directory. The runs span 04:33 to 05:14 UTC on
2026-11-24; the directory and the sweep banner carry the local date.

## Build

`appleclang15/` — Apple clang 15.0.0 (clang-1500.1.0.2.5): fresh
`cmake --preset release -G Ninja`, then `cmake --build build-release
--parallel 6`. 339 steps, 0 warnings.

## Results

| Leg | AppleClang 15.0.0 | File |
|---|---|---|
| P1 tier | AVX2+FMA | `system_inspector.txt` |
| P1 cores read | 8 logical, 4 physical | `system_inspector.txt` |
| P1 correctness (`-LE "timing\|benchmark" -j6`) | 132/132 | `ctest.txt` |
| P2 `simd_verification` | 140 pass, 0 fail | `simd_verification.txt` |
| P3 timing (`-j1 -L timing`), quiet | 14/14 | `ctest_timing.txt` |
| P4 accuracy sweep | 26,882 rows, 190 self-checks, 0 contract violations, 0 fallbacks | `oracle.txt` |

- Correctness ran with `-j6`, not the plan's `-j8`: this machine's job cap.
- `ContractGates.Issue268_ExponentialSetParametersIsDefined` and
  `ContractGates.Issue316_CauchyQuantileFiniteAtSubnormalP` ran and passed.
  `ThreadPoolIntegration.OptimalThreadCountMakesNoSystemCallPerCall` is
  Linux-only and absent here.
- The core counts are unchanged from v2.5.0 on this machine (#289).

## P3 — timing

Screen locked, launched detached (`timing_run.sh.txt`); the run started at a
1-minute load of 1.38 against the limit of 2.0 (`timing_run.txt`). 10.4 s,
9.1 s of it `test_parallel_batch_gates`. `timing_environment.txt` shows
`dirty=0`: the regenerated accuracy block (below) had been restored first.

## P4 — accuracy sweep and the `isa=AVX2` block

Scored with `tools/accuracy_vs_mpmath.py` under mpmath 1.4.1, 21 min 41 s.
The sweep was started from inside the validated checkout; its banner reads
`commit=3dd8358  isa=AVX2  date=2026-11-23`.

`accuracy_block_AVX2.diff` is the whole change to
`docs/ACCURACY_CHARACTERIZATION.md`: the banner line and seven rows of the
`cauchy` section. **The regenerated block is not committed.** Applying the
diff to the document reproduces it, as long as the `isa=AVX2` block has not
changed in between.

| Cauchy row | At `d1adc95` (v2.5.0) | At `3dd8358` |
|---|---|---|
| quantile, max abs | 5.403e+289 | 1.505e+289 |
| quantile, max rel | 2.774e-16 | 2.774e-16 |
| cdf, max rel (scalar and batch) | 2.847e-16 | 2.904e-16 |
| pdf, max rel (scalar and batch) | 4.213e-16 | 3.465e-16 |
| pdf, p99 rel (scalar and batch) | 3.465e-16 | 3.209e-16 |
| logpdf, max abs (scalar and batch) | 2.336e-13 | 2.333e-13 |

The moved figures are Ivy Bridge's and Zen 4's. The quantile's max rel is
2.774e-16 here against 2.543e-16 on Ivy Bridge, before and after alike. The
cdf and pdf worst points also move (to x = −3.1831e+265 and 3.1831e+15), for
the reason I's README gives: their x grid is built from the quantile.

## Not done

- No dispatch-table capture, cost bench or separate concurrency leg: out of
  the patch's scope (`PLAN.md`).
