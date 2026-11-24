# Zen 4 — v2.5.1 fleet legs P1–P4 at b5f0311 (2026-11-23)

The Z machine's legs for the v2.5.1 patch release (`PLAN.md`, "v2.5.1 patch
release — fleet legs"). `main` at `b5f0311`, whose code is identical to the
validated commit `7069adc` (`git diff --stat 7069adc b5f0311` over `src`,
`include`, `tests`, `tools`, `cmake`, `CMakeLists.txt` and `CMakePresets.json`
is empty). Run in a detached worktree of `origin/main` with new build
directories.

Asus TUF A16, Ryzen 7 7445HS (6 cores / 12 threads; AVX-512), Windows 11 Pro
10.0.26300, Visual Studio 18 Community, CMake 4.4.4, Ninja 1.13.2. Power
scheme Performance, on AC.

## Builds

Fresh Release configures of the same commit, one per compiler
(`build.ps1.txt`; each `configure.txt` names its compiler):

- `msvc/` — MSVC 19.52.37029 (toolset 14.52.37025), Visual Studio generator:
  `cmake -S . -B <dir>`, `cmake --build <dir> --config Release --parallel`.
  0 warnings. v2.5.0 was validated with toolset 14.51.
- `clangcl/` — clang-cl 22.1.3 (the Visual Studio copy), Ninja, after
  `vcvars64.bat`: `cmake -S . -B <dir> -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl`. 0 warnings.

Both take `/arch:AVX512` from the detected tier. The MSVC dynamic tests import
`VCRUNTIME140.dll`, not the Debug runtime (`dumpbin /imports` on
`test_gaussian_basic_dynamic` and `test_exponential_basic_dynamic`).

## Results

| Leg | MSVC 19.52 | clang-cl 22.1.3 | File |
|---|---|---|---|
| P1 tier | AVX-512 | AVX-512 | `system_inspector.txt` |
| P1 cores read | 12 logical, 6 physical | 12 logical, 6 physical | `system_inspector.txt` |
| P1 correctness (`-LE "timing\|benchmark" -j8`) | 132/132 | 132/132 | `ctest.txt` |
| P2 `simd_verification` | 140 pass, 0 fail | 140 pass, 0 fail | `simd_verification.txt` |
| P3 timing, quiet, opted out of power throttling | not run | 14/14 | `clangcl/ctest_timing.txt` |
| P4 accuracy sweep | 26,882 rows, 190 self-checks, 0 contract violations, 0 fallbacks | the same | `oracle.txt` |

The gates new since v2.5.0 that exist on Windows ran and passed under both
compilers: `ContractGates.Issue268_ExponentialSetParametersIsDefined` and
`ContractGates.Issue316_CauchyQuantileFiniteAtSubnormalP`.
`ThreadPoolIntegration.OptimalThreadCountMakesNoSystemCallPerCall` is Linux
only and is absent here.

## P3 — timing

clang-cl build only, as at v2.5.0: it is the performance build on this
machine. `timing_run.ps1.txt` is the script. ctest lists the 14 tests with
their commands and `GTEST_FILTER`; the script launches each binary itself and
opts it out of Windows power throttling, because the opt-out is per process
and a child of ctest would not inherit it (`docs/VALIDATION_HISTORY.md`,
"Zen 4 timing: power throttling"). `clangcl/ctest_timing.txt` is the script's
log, one line per test with its exit code and opt-out state;
`clangcl/timing/` holds each test's own output.

- Total CPU 6.1% at the start (gate 8%) and 3.4% at the end.
- `clock_probe` for 10 s before and after: 127.2 and 127.1 ms per pass both
  times. v2.5.0 read 127–130 ms at boost clock and about 190 ms at base
  clock, so this run was at boost.
- 14/14 exit 0, 19 test cases. `test_parallel_batch_gates` took 5.5 s; the
  other thirteen took 0.1–0.2 s each.

## P4 — accuracy sweep

Both sweeps were started from inside the worktree, clean, and carry the
banner `commit=b5f0311  isa=AVX-512  date=2026-11-23`. Scored with Python
3.14.10 and mpmath 1.4.1, about seven minutes each.

**clang-cl** is the sweep of record: it produced this machine's block at
v2.5.0. `accuracy_block_AVX-512.diff` is the whole change to
`docs/ACCURACY_CHARACTERIZATION.md`: the banner and seven rows, all in the
`cauchy` section. The regenerated block is **not** applied in this PR.

| Row | v2.5.0 block | This sweep |
|---|---|---|
| quantile scalar, max abs | 5.403e+289 | 1.505e+289 |
| quantile scalar, max rel | 2.543e-16 | 2.543e-16 |
| cdf scalar and batch, max rel | 2.847e-16 | 2.904e-16 |
| pdf scalar and batch, p99 rel | 4.213e-16 | 3.465e-16 |
| logpdf scalar and batch, max abs | 2.336e-13 | 2.333e-13 |

These are the figures Ivy Bridge reported
(`../2026-11-23-ivybridge-v2.5.1/`). No row outside `cauchy` moved.

**MSVC** is the optional second sweep: scored against a copy of the document,
0 contract violations. Its rows differ from clang-cl's, as they did at
v2.5.0, so its block is not comparable with the committed one and no diff is
kept.

## #290, raised by Ivy Bridge

Ivy Bridge noted that `tools/cpp20_features_inspector` links although #290
is open. The link failure was fixed on `main` by `7d56812` (PR #304,
2026-11-17), which links TBB into the standalone tools when it is detected;
that is one day after the 2026-11-16 baseline at `321bb98`. The issue was
re-scoped on 2026-11-19 to the install-contract remainder, which is fixed on
`dev/v3.0.0-corvus-adoption` (`aadc46b`); it stays open until that branch
reaches `main`. Nothing is owed for v2.5.1.
