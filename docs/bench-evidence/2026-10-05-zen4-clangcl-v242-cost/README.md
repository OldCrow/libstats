# Zen 4 R3 and R8 on clang-cl (2026-10-05/06)

Z's R3 and R8 re-run on clang-cl builds [user, 2026-10-05], replacing the MSVC overnight of
2026-10-04/05, which ran under power throttling (`2026-10-05-zen4-msvc-vs-clangcl/`).
Ran 2026-10-05 20:20–22:36 at `69c7be4` (code `606607f`).

## Builds

- clang-cl 22.1.3 (VS 2026 bundled), Ninja, Release, `/arch:AVX512` (uncapped) or the capped
  tier's `/arch:` flag, at the freeze head.
- v2.4.1 for R3's comparison: the `v2.4.1` tag with `v241_clangcl_compat.patch` applied,
  uncommitted, in a detached worktree. v2.4.1 does not compile under clang-cl as tagged. The
  patch is build-only:
  - 6245e4e's three build hunks: `safe_cpuid` tests `_MSC_VER` before `__clang__`; `/W4` for
    clang-cl's `-Wall`; no `/arch:SSE2` on x64.
  - The freeze head's `/arch:` flags for clang-cl, so both sides build the same ISA.
- Both cost benches: `tools/bench/v242_cost_bench.cpp` built with clang-cl `/arch:AVX512`
  against each tree's `stats_static.lib`.

## Run

`overnight_z2.ps1`, detached, screen locked. Every measured process is opted out of power
throttling at launch (`nothrottle.ps1`); children do not inherit that, so each timing test is
launched on its own rather than through ctest. A 10-s `clock_probe` (opted out) opens every
phase: ~128 ms per pass at boost, ~190 at base clock. Each run waits for CPU < 8%; each R8
run follows a 25-s discarded `strategy_profile` pass.

## Results

Clock: every phase opened at 127–130 ms per probe pass (`clock/`): full boost throughout,
screen locked. The R8 bundles are `data/profiles/dispatcher/2026-10-06T0*_windows-*_sha-69c7be4`.

R3 (`v241.csv`, `v242.csv`, `compare.txt`, `run_r3.txt`):

- Over the 10× guide, per call: von Mises quantile 17–23×, Student-t quantile at ν ≥ 1e3
  11.6–13.3×. In absolute terms this machine is the fastest of the three (von Mises(1)
  quantile 418 ns; M1 580, MSVC 435); the ratio is high because clang-cl's v2.4.1 grid
  lookup cost 18 ns (M1 67).
- Beta at large shapes forced VECTORIZED 3–4× (MSVC before `998d985`: 6–11×); Gamma α = 20
  logpdf 3.6× at n = 1e4, 7.7× at 1e5.
- 18 NEW and 15 BOTH AUTO-vs-best gaps, 28 of 33 favouring PARALLEL, 19 of 33 at costly
  parameters (Beta 25 and 1000 shapes, Gamma α ≥ 19.9, Student-t ν ≥ 1e3, large counts).
- Timing 21/22 (`timing.txt`). `test_gaussian_enhanced` failed once on its single-shot
  parallel batch-fit speedup (0.69 against 0.8; `timing_test_gaussian_enhanced.txt`); 40
  reruns (20 clang-cl, 20 MSVC) read 1.00 and never fail. Six small datasets timed once,
  ~30 µs each way.
