# M1 — v2.5.1 fleet legs P1–P4 at 53cef69 (2026-11-24)

The M machine's legs for the v2.5.1 patch release (`PLAN.md`, "v2.5.1 patch
release — fleet legs"). `main` at `53cef69`, whose code is identical to the
validated commit `7069adc` (`git diff --stat 7069adc 53cef69` over `src`,
`include`, `tests`, `tools`, `cmake`, `CMakeLists.txt` and `CMakePresets.json`
is empty).

Mac Mini M1 (Macmini9,1; 4 performance + 4 efficiency cores; NEON), macOS
27.2.1 (26B131), CMake 4.5.1, Ninja 1.13.2, AC power. Work done in a separate
worktree of `main` with a new build directory. The runs span 21:30 UTC on
2026-11-24 to 00:09 UTC on 2026-11-25; the directory and the sweep banner
carry the local date.

## Build

`appleclang21/` — Apple clang 21.0.0 (clang-2100.4.1.1): fresh
`cmake --preset release -G Ninja` (the build directory removed first), then
`cmake --build build-release --parallel 8`. 337 steps, 0 warnings.

## Results

| Leg | AppleClang 21.0.0 | File |
|---|---|---|
| P1 tier | NEON | `system_inspector.txt` |
| P1 cores read | 8 logical, 8 physical | `system_inspector.txt` |
| P1 correctness (`-LE "timing\|benchmark" -j8`) | 132/132 | `ctest.txt` |
| P2 `simd_verification` | 140 pass, 0 fail | `simd_verification.txt` |
| P3 timing (`-j1 -L timing`), quiet | 14/14 | `ctest_timing.txt` |
| P4 accuracy sweep | 26,882 rows, 190 self-checks, 0 contract violations, 0 fallbacks | `oracle.txt` |

- `ContractGates.Issue268_ExponentialSetParametersIsDefined` and
  `ContractGates.Issue316_CauchyQuantileFiniteAtSubnormalP` ran and passed.
  `ThreadPoolIntegration.OptimalThreadCountMakesNoSystemCallPerCall` is
  Linux-only and absent here.

## P3 — timing

Screen locked, Backblaze paused, launched detached (`timing_run.sh.txt`). The
run waited for ambient CPU (100 − `top` idle) below 8% over two 10 s windows
and started at 4.83% (`timing_run.txt`). 5.3 s, 4.7 s of it
`test_parallel_batch_gates`. `timing_environment.txt` shows `dirty=0`: the
regenerated accuracy block (below) had been restored first.

**Fourteen of the user's own background processes were suspended (SIGSTOP)
before the gate and resumed after the run** [user's decision, 2026-11-24]:
Photos media analysis, Spotlight indexing and sync agents, named in
`timing_environment.txt`. On this machine `mediaanalysisd` and
`corespotlightd` hold one to three cores almost without pause, and no gate
passes otherwise. Four 5-minute attempts found no window; the fifth did.

## P4 — accuracy sweep and the `isa=NEON` block

Scored with `tools/accuracy_vs_mpmath.py` under mpmath 1.4.1, 9 min 49 s.
The sweep was started from inside the validated checkout; its banner reads
`commit=53cef69  isa=NEON  date=2026-11-24`.

`accuracy_block_NEON.diff` is the whole change to
`docs/ACCURACY_CHARACTERIZATION.md`: the banner line and seven rows of the
`cauchy` section. **The regenerated block is not committed.** Applying the
diff to the document reproduces it, as long as the `isa=NEON` block has not
changed in between.

| Cauchy row | At `df78931` (v2.5.0) | At `53cef69` |
|---|---|---|
| quantile, max abs | 5.403e+289 | 1.505e+289 |
| quantile, max rel | 2.774e-16 | 2.774e-16 |
| cdf, max rel (scalar and batch) | 2.847e-16 | 2.904e-16 |
| pdf, max rel (scalar and batch) | 4.713e-16 | 4.713e-16 |
| pdf, p99 rel (scalar and batch) | 4.213e-16 | 3.465e-16 |
| logpdf, max abs (scalar and batch) | 2.336e-13 | 2.333e-13 |

The quantile, cdf and logpdf figures are Ivy Bridge's, Zen 4's and Kaby
Lake's. The pdf row differs from theirs in which column moves: this
machine's max rel is 4.713e-16 at x = −2.18309e+08, outside the region #326
changed, and stays; the 4.213e-16 to 3.465e-16 step that is the maximum on
the other three is the 99th percentile here. The cdf worst point moves to
x = −3.1831e+265 as on the others, for the reason I's README gives: the x
grid is built from the quantile.

## Not done

- No dispatch-table capture, cost bench or separate concurrency leg: out of
  the patch's scope (`PLAN.md`).
