# Zen 4 R3 and R8 on clang-cl (2026-10-05/06)

Z's R3 and R8 re-run on clang-cl builds [user, 2026-10-05], replacing the MSVC overnight of
2026-10-04/05, which ran under power throttling (`2026-10-05-zen4-msvc-vs-clangcl/`).
Results are added here after the run.

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
