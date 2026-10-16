# Zen 4 quiet runs on clang-cl at d1adc95 (2026-10-13/14)

Z's checklist at the freeze head `d1adc95` (PLAN.md, "At `d1adc95`, per machine"): the speed
gate alone, R3 (v2.4.1 vs head costs, the timing tests) and R8 (four tiers, three runs each),
all on clang-cl builds. Ran 2026-10-13 22:17 to 2026-10-14 00:20, screen locked, at the branch
tip `df78931` (code `d1adc95`; the tip adds K's AVX2 block, K's R8 bundles and PLAN.md).

## Builds

- VS 2026 18.11 (updated that evening): MSVC 14.51.36231 and the bundled clang-cl 22.1.3.
  Fresh configures of every tree. clang-cl, Ninja, Release: `build-clangcl/` uncapped
  (`/arch:AVX512`, tests on) and `build-clangcl-{avx2,avx,sse2}/` capped through
  `LIBSTATS_MAX_SIMD_TIER` (tests off). All 0 warnings; tier confirmed per tree by the
  configure line and `system_inspector --quick`.
- v2.4.1 for R3's comparison: the `v2.4.1` tag with `v241_clangcl_compat.patch`
  (`2026-10-05-zen4-clangcl-v242-cost/`) applied, uncommitted, in `../libstats-v2.4.1`,
  its `build-clangcl-v242cmp/stats_static.lib` from 2026-10-05 (same compiler, same code).
- Both cost benches: `tools/bench/v242_cost_bench.cpp` built with clang-cl `/arch:AVX512`
  against each tree's `stats_static.lib`.

## Run

`overnight_z4.ps1`, created through WMI (`Win32_Process.Create`) so it outlives the session
that launched it. Every measured process is opted out of power throttling at launch
(`nothrottle.ps1`); children do not inherit that, so each timing test is launched on its
own rather than through ctest. A 10-s clock probe (opted out) opens every phase: ~128 ms per
pass at boost, ~190 at base clock. Each phase waits for CPU < 8%; each R8 run follows a 25-s
discarded `strategy_profile` pass. The timing-test list is `ctest --show-only=json-v1
-L timing` of `build-clangcl/`, taken at start (14 tests after the `4eb0816` split).

## Results

Clock: every phase opened at 127.1–128.2 ms per probe pass (`clock_*.txt`): full boost
throughout. Phases started at 3.0–6.8% CPU (`gates/run_gates.txt`, `r3/run_r3.txt`, the
bundles' `logs/run.txt`).

Speed gate (`gates/`): `test_parallel_batch_gates` alone, PARALLEL/VECTORIZED 0.237–0.265 at
1M for all eight operations (limit 0.75).

R3 (`r3/v241.csv`, `r3/v242.csv`, `r3/compare.txt`, `r3/run_r3.txt`):

- Over the 10× guide, per call: von Mises quantile 15.8–22.8× (18 ns → 293–416 ns; the
  ratio is v2.4.1's grid lookup), Student-t quantile at ν ≥ 1e3 10.6–11.5× (52–65 ns →
  595–683 ns). Both as on 2026-10-05 (17–23×, 11.6–13.3×).
- Forced VECTORIZED: Gamma α = 20 logpdf 4.6× at n = 1e4, 8.2× at 1e5 (its scalar
  fall-back); Beta at large shapes forced SCALAR 5.0–5.1×, VECTORIZED 2.7×.
- Section 3: 45 NEW and 16 BOTH AUTO-vs-best gaps. Nearly every NEW gap is AUTO staying on
  VECTORIZED where PARALLEL is 2–4× faster at n = 1e4–1e5: the Binomial, NegativeBinomial
  and Student-t pdf/logpdf/cdf rows, Gamma α ≥ 10, Beta (1000, 1000). R8's input.
- Timing 14/14 (`r3/timing.txt`, `r3/timing/`).

R8: bundles `data/profiles/dispatcher/2026-10-14T0{2-41-05,3-05-23,3-29-48,3-54-51}Z_windows-*_sha-df78931`
(AVX-512, AVX2, AVX, SSE2); `cross_tier_assessment.{py,txt}` in the AVX-512 bundle compares
all four against the tables, Z's 2026-10-07 captures and K's 2026-10-13 captures.

## Files

- `environment.txt`: sha, power scheme, AC, CPU, OS, compiler, start and finish.
- `gates/`: the speed gate's log and output.
- `r3/`: the two bench CSVs, `compare.txt`, `run_r3.txt`, `timing.txt`, `timing/<test>.txt`.
- `clock_*.txt`: the clock probe before each phase and each R8 run.
- `overnight_z4.ps1`, `nothrottle.ps1`: the runner and the throttling opt-out.
