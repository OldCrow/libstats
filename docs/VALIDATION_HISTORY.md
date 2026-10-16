# libstats Validation History

Historical per-version validation matrices and SIMD speedup benchmarks, extracted from
`AGENTS.md` to keep that file lean for session start. `AGENTS.md` keeps only the
current-release validation matrix and current deferred items; this file holds everything
prior. For detailed per-version change narratives, see `CHANGELOG.md` (auto-generated via
git-cliff). For SIMD methodology, see `docs/SIMD_OPTIMIZATION_REFERENCE.md` and
`docs/SIMD_BENCHMARK_RESULTS.md`.

## Validation matrices by release

### v2.5.0 — validation matrix

Correctness column = `ctest -LE "timing|benchmark"` (132 registered at v2.5.0;
the 14 timing tests are the `*_timing` binaries split out by `4eb0816`). Every
machine validated at the code head `d1adc95` (fix-9 round `8d3c9c7`), then again
at its own dispatch-table header (R8, below). Evidence directories are under
`docs/bench-evidence/`; the R8 bundles under `data/profiles/dispatcher/`.

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Asus TUF A16 (Windows 11) | AVX-512 | 132/132 ✅ MSVC 14.51 and clang-cl 22.1.3 | 14/14 ✅ (clang-cl, quiet, boost) | Native, 2026-10-13/15. Fresh configures, 0 warnings. R9 stress 109 passed + 27 skipped (the work-stealing probe, unreachable on Windows) under both compilers. `isa=AVX-512` block regenerated from the clang-cl sweep (`85e1350`): 26,882 rows, 190 oracle self-checks, 0 contract violations; the MSVC sweep through the same oracle, 0. R8 four tiers (`2026-10-14T0*_windows-*`), table A/B five rounds (`2026-10-15-zen4-threshold-rows/`), R1 again at the kAvx512 header `6c149f1`: 132/132 both compilers, gates 0.235–0.254. Every timing process opted out of power throttling (below). |
| Kaby Lake (2017 MBP, macOS 13.7.8) | AVX2+FMA | 132/132 ✅ | 14/14 ✅ | Native, 2026-10-13/14, AppleClang 15 Release (`34baa1f`). R9 under ThreadSanitizer: stress 136/136, the whole correctness suite, 0 reports (`2026-10-14-kabylake-r9-tsan/`). `isa=AVX2` block regenerated (`fab7345`): 26,882 rows, 0 contract violations. R8 AVX2, AVX, SSE2 (`2026-10-13T*_darwin-*`); R3 costs (`2026-10-14-kabylake-v242-cost/`). Table A/B for kAvx2/kAvx three rounds (`2026-10-15-kabylake-threshold-rows/`), R1 at its header `4c3a133`: 0 warnings, 132/132, gates 0.270–0.296, timing 14/14. |
| Mac Mini M1 (macOS 27.0.1) | NEON | 132/132 ✅ | 14/14 ✅ | Native, 2026-10-13/14, AppleClang 21 Release, outside the sandbox (`10a7ff7`). Quiet batch gates 0.21–0.26. R9 under ThreadSanitizer: stress 109 + 27 skipped, 0 reports (`2026-10-14-m1-concurrency/`). `isa=NEON` block regenerated (`ea3e4f3`): 26,882 rows, 0 contract violations. R8 NEON bundle (2026-10-14); table A/B four rounds (`2026-10-14-m1-threshold-rows/`), R1 at the kNeon header `bb9ac0b`: 0 warnings, 132/132, gates 0.21–0.22. |

Costs of the accuracy work (R3, `tools/bench/v242_cost_bench.cpp` against
v2.4.1, quiet, per call and forced VECTORIZED). The cost moves in steps at
the algorithm switch points in the parameters, so it is stated per band,
never as "distribution X is slower" [user, 2026-10-16]. Kaby Lake's ratios
(v2.5.0 / v2.4.1, median over sizes; above 1 is slower):

| Family | Low band | High band | Inflection |
|---|---|---|---|
| Gamma density | shape 2.5–19.9: 1.0 | shape 20: 3.3 (logpdf); 1000: 2.6 | step at shape 20, the Stirling prefactor (`STIRLING_PREFACTOR_SHAPE`; #266, v3) |
| Gamma CDF | shape 0.01: 2.1; 2.5: 1.5 | shape 10–20: 1.1–1.2; 1000: 1.5 | small-shape `gamma_q` and the rate-product correction |
| Gamma quantile | shape 0.01: 0.15 | 0.6–0.7 elsewhere | faster everywhere (`solve_concave`) |
| Beta density | (2,2): 1.0 | (25,25): 3.4; (1000,1000): 3.0 | large shapes take the Stirling/`lbeta` form; CDF 1.1–1.3 throughout |
| Student-t | ν 7–30: CDF 1.05–1.1, quantile 1.5 | ν ≥ 1e3: CDF 4.4–5.2, quantile 6.5–6.8, pdf 1.9 | large-ν tail paths (BGRAT; log(ν/t²) beyond √ν) |
| von Mises CDF | κ = 1: 2.1 | κ = 30: 0.81; κ = 300: 0.19 | inverted: a fixed-cost quadrature replaced a Bessel series whose cost grew with κ |
| von Mises quantile | 5.3–6.9, flat in κ | — | fixed cost (Newton on the exact CDF, ~300–400 ns) against v2.4.1's grid lookup |
| Poisson CDF | λ 12–1000: 1.3–1.6 | λ = 1e5: 0.21 | large λ 5× faster |
| Discrete densities (Binomial, NegBinomial, Poisson) | 1.3–1.8 | 1.3–1.5 | roughly uniform: the Loader-style log-pmf (#172); slightly dearer at small n or r |
| Discrete quantiles | 0.04–0.74 | — | faster everywhere |

The M1 tracks Kaby Lake within the noise (von Mises quantile 6.4–9.5×,
Student-t quantile at ν ≥ 1e3 7–9×). Zen 4's ratios run larger against
v2.4.1's very cheap lookups there: von Mises quantile 15.8–22.8× (an 18 ns
grid lookup became 0.3–0.6 µs, exact), Student-t quantile at ν ≥ 1e3
10.6–11.5×, Gamma shape ≥ 20 logpdf forced VECTORIZED 4.6–8.2×. The bench
sweeps parameter bands at a fixed input range, so it does not resolve
x-regions (deep tails, near-boundary bands, log-space branches); timing the
accuracy grid per x-region is a v3 input. Each machine's compare.txt lists
the AUTO-vs-best gaps (Kaby Lake 49, M1 46, Zen 4 45 NEW), the input to R8.

Dispatch tables (R8): three quiet `strategy_profile --large` runs per tier on
every machine at `d1adc95`, re-derived into kAvx512, kAvx2, kAvx and kNeon by
`scripts/derive_threshold_rows.py` (`580b7f7`: 162 values on 85 rows), then
verified by a loaded A/B (`tools/bench/table_ab_bench.cpp`, AUTO against the
forced strategies under the old and new headers, ABAB): kNeon (AUTO rows more
than 1.25× behind their best forced strategy 202 → 24 with `580b7f7`, then
13 → 9 with the eight rows of `bb9ac0b`), kAvx512 `6c149f1` (21 more rows;
208 → 5), kAvx2 and kAvx `4c3a133` (15 values on 13 rows; AUTO gaps on the
AVX2, capped-AVX and capped-SSE2 trees 158 → 2, 146 → 2, 173 → 17). The
discrete pmfs, the Student-t and von Mises CDFs and the Beta densities
leave NEVER; most continuous CDFs cross to PARALLEL at 8k–25k instead of
25k–2M; the only NEVER cell left is kNeon Uniform PDF. Kaby Lake's AUTO
goes multi-threaded through WORK_STEALING, which beats PARALLEL below the
profiled crossover; the rows stay PARALLEL-optimal because kAvx2 and kAvx
serve Linux and Windows too (a v3 input).

### v2.4.0 and v2.4.1

No matrix was recorded for these two releases; their validation is described
in their CHANGELOG entries (v2.4.0: three fleet machines, correctness 58 → 74,
every table measurement-backed; v2.4.1: a two-issue correctness patch with
fail-first gates).

### v2.3.1 — validation matrix

Correctness column = `ctest -LE "timing|benchmark"` (58 registered at v2.3.1).

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Asus TUF A16 (Windows) | AVX-512 | 58/58 ✅ | — | Native, 2026-08-25, MSVC Release; timing suite not re-run (nothing in v2.3.1 touches kernels' hot paths — the NaN branches are never taken for finite data) |
| Mac Mini M1 | NEON | 58/58 ✅ | 22/22 ✅ | Native, 2026-08-27/28, AppleClang Dev build; timing serial (`-j1`) after post-boot load settled to ~2.0; `isa=NEON` characterization block regenerated on the 6063-row grid (61 violations — 63 minus the two geometric #125 rows AArch64 saturation renders finite) |
| Kaby Lake (2017 MBP) | AVX2+FMA | 58/58 ✅ | 22/22 ✅ | Native, 2026-08-26, AppleClang Dev build; timing serial (`-j1`) after machine settled below load 2.0; `isa=AVX2` characterization block regenerated on the 6063-row grid (63 violations, matching Zen 4) |

The correctness count grew 53 → 58 with v2.3.1's five new gate binaries:
`test_log_special_gates`, `test_batch_nan_gates`, `test_discrete_quantile_bounds`,
`test_simd_dispatch_gates`, `test_parallel_exception_propagation`.

### v2.3.0 — validation matrix

Carried here verbatim from `AGENTS.md` when v2.3.1 shipped.

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Asus TUF A16 (Windows) | AVX-512 | 53/53 ✅ | 21/22 ⚠️ | Native, 2026-08-20, MSVC Release |
| Mac Mini M1 | NEON | 53/53 ✅ | 22/22 ⚠️ | Native, 2026-08-23, AppleClang 21 Release, v2.3.0 tag + main, Bessel Tier 2; timing ran on a loaded machine (indicative only) |
| Kaby Lake (2017 MBP) | AVX2+FMA | 53/53 ✅ | 22/22 ✅ | Native, 2026-08-22, AppleClang Release (`release` preset); Bessel Tier 2 (libc++ has no `cyl_bessel_i`) |

The one timing failure was the `UniformEnhancedTest` speedup assertion
(flaky on this machine, carried from v2.2.0; 1.44x vs a 1.8x adaptive
threshold at 5000 elements). Correctness = `ctest -LE "timing|benchmark"`,
53 registered.

### v2.2.0 — validation matrix

Carried here verbatim from `AGENTS.md` when v2.3.0 shipped.

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Asus TUF A16 (Windows) | AVX-512 | 49/49 ✅ | 21/22 ⚠️ | Native, 2026-08-16, MSVC 14.51 Release |
| Mac Mini M1 | NEON | — | — | Never natively re-validated for v2.2.0 (CI leg only) |
| Kaby Lake (2017 MBP) | AVX2+FMA | — | — | Never natively re-validated for v2.2.0 (CI leg only) |

Counts are not comparable with v2.1.0's: #94 fixed `run_tests`' `ctest`
filters, which had been written with escaped pipes and matched nothing. The
one timing failure was the `UniformEnhancedTest` speedup assertion (flaky,
carried into the v2.3.0 matrix unchanged).

### v2.1.0 — validation target (three machines)

Carried here verbatim from `AGENTS.md` when v2.2.0 shipped.

| Machine | SIMD | Correctness | Timing | Notes |
|---|---|---|---|---|
| Mac Mini M1 | NEON | 46/46 ✅ | 22/22 ✅ | Validated 2026-07-05 |
| Kaby Lake (2017 MBP) | AVX2+FMA | 46/46 ✅ | — | CI validated |
| Asus TUF A16 (Windows) | AVX-512 | 46/46 ✅ | — | CI validated |

Note the suite counts in this row are not comparable with v2.2.0's. Until
#94 (fixed in v2.2.0) the `run_tests` target's `ctest -LE/-E` filters were
written with escaped pipes and matched nothing, so the selection those
numbers came from was not the selection intended.

### v2.0.0 — validation target (three machines)

Ivy Bridge / macOS Catalina dropped from the ecosystem in v2.0.0 (Catalina EOL;
minimum macOS raised to 13 Ventura).

| Machine | SIMD | Target | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 46/46 ✅ | Audit remediation complete (2026-07-01) |
| Mac Mini M1 | NEON | 46/46 ✅ | Audit remediation complete (2026-07-01) |
| Asus TUF A16 (Windows) | AVX-512 | 46/46 ✅ | Re-validation required after audit remediation |

### v1.5.2 — final v1.x release (four machines)

| Machine | SIMD | Correctness | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 39/39 ✅ | |
| Ivy Bridge (2012 MBP) | AVX | 38/38 ✅ | (last version with Catalina) |
| Mac Mini M1 | NEON | 39/39 ✅ | |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | |

### v1.5.1 — validated on all four machines

`simd_verification` reports **geometric mean speedups** per operation type (PDF/LogPDF/CDF)
and per primitive vector op, not a single composite. See `tools/simd_verification.cpp` for rationale.

| Machine | SIMD | Correctness | Total suite | simd_verification | PDF geomean | LogPDF geomean | CDF geomean |
|---|---|---|---|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 39/39 ✅ | 61 | 61/61 ✅ | 8.0x | 9.6x | 3.3x |
| Mac Mini M1 | NEON | 39/39 ✅ | 61 | 61/61 ✅ | 5.9x | 7.3x | 3.1x |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | 61 | 61/61 ✅ | 4.8x | 5.1x | 2.2x |

Kaby Lake primitive vector op speedups (v1.5.0 Phase 1+2): VectorExp 3.4x, VectorLog 1.7x, VectorErf 2.5x, VectorCos 4.9x.
Mac Mini M1 primitive vector op speedups (v1.5.0 Phase 3): VectorExp 2.1x, VectorLog 1.8x, VectorErf 8.0x, VectorCos 3.0x.
Asus TUF A16 primitive vector op speedups (v1.5.0 Phase 4): VectorExp 5.0x, VectorLog 3.9x, VectorErf 1.3x, VectorCos 8.5x.

### v1.4.0 baseline — all four machines

| Machine | SIMD | Correctness | Total suite | simd_verification | Overall |
|---|---|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2 | 39/39 ✅ | 59 | 54/54 ✅ | 3.35x |
| Mac Mini M1 | NEON | 39/39 ✅ | 59 | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | 59 | 54/54 ✅ | 1.64x |

**Total suite counts differ by machine (v1.5.0):**
- Kaby Lake (61): v1.5.0 adds VonMises distribution rows + 4 primitive vector op rows to `simd_verification`.
- Mac Mini M1 (61): Phase 3 validated ✅.
- Asus TUF A16 (61): Phase 4 validated ✅.

> **v2.0.0:** macOS minimum raised to 13 Ventura. Ivy Bridge / Catalina support dropped.
> `CROSS_PLATFORM` build type and `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` removed.
> Alternate LLVM compiler infrastructure removed; use system AppleClang.

## SIMD batch operation speedups (historical)

### Ivy Bridge AVX — v1.5.0, historical
v1.5.0 results on Ivy Bridge AVX (61/61 simd_verification ✅): PDF geomean 5.6x, LogPDF 6.0x, CDF 2.6x.
Primitive ops: VectorExp 2.2x, VectorLog 1.3x, VectorErf 1.7x, VectorCos 11.0x.

Selected per-distribution speedups:

| Distribution | Op | Speedup |
|---|---|---|
| Uniform | PDF | 122.4x |
| Uniform | LogPDF | 118.4x |
| Uniform | CDF | 27.0x |
| Gaussian | LogPDF | 44.1x |
| Exponential | LogPDF | 35.4x |
| VonMises | LogPDF | 18.2x |
| Exponential | PDF | 14.5x |
| Exponential | CDF | 7.5x |
| Gamma | PDF | 8.2x |

### v1.0.0 validation (historical)

Ivy Bridge AVX (historical — Catalina support dropped in v2.0.0):
- correctness suite: 34/34 PASS
- `simd_verification`: 54/54 PASS, overall 4.10x
- new-distribution speedups: Chi-squared PDF 9.5x/LogPDF 7.0x, Student's t PDF 7.3x/LogPDF 7.6x,
  Beta PDF 4.6x/LogPDF 4.4x

Asus TUF A16 (Windows, AVX-512 — first AVX-512 validation):
- correctness suite: 33/33 PASS (GTest available via vcpkg gtest:x64-windows 1.17.0)
- `simd_verification`: 54/54 PASS, overall 1.64x
- AVX-512 arithmetic/log-space paths: Gaussian LogPDF 21.9x, Exponential LogPDF 11.8x,
  Uniform LogPDF 7.5x — strong where transcendentals are not involved
- Overall speedup limited by transcendental delegation to AVX (see Deferred Items in AGENTS.md)

Kaby Lake AVX2 (2017 MBP):
- correctness suite: 33/33 PASS
- `simd_verification`: 54/54 PASS, overall 3.49x
- new-distribution speedups: Chi-squared PDF 13.8x/LogPDF 10.5x, Student's t PDF 6.3x/LogPDF 18.4x,
  Beta PDF 5.3x/LogPDF 4.1x

All four machines validated at v1.0.0 (Ivy Bridge/Catalina dropped in v2.0.0).

## Zen 4 timing: power throttling

Moved here from AGENTS.md on 2026-09-07: it is needed when interpreting a
timing run, not in every session. Corrected at v2.5.0: the v2.0.3 note blamed
Precision Boost 2 frequency scaling.

**Power throttling, not a threshold anomaly.** On the Zen 4 box (Asus TUF
A16, Ryzen 7 7445), a locked screen puts running processes under Windows power
throttling (EcoQoS): base clock (~3.2 GHz instead of ~4.7 GHz), 1.5× slower
scalar and vectorized, 1.3× slower parallel. That looks like a
dispatch-threshold anomaly: crossovers drift toward PARALLEL. The throttle
starts within seconds of the lock and stays with the process after unlock.
Opt each measured process out (`SetProcessInformation`,
`ProcessPowerThrottling`, `EXECUTION_SPEED`; helper
`docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/nothrottle.ps1`) and
launch binaries directly, not through ctest: the opt-out is per process and a
child does not inherit it. With the screen on, a single-core probe shows no
drop over 90 s. Every v2.5.0 timing run on this machine (R3, R8, the table
A/B) used the clang-cl performance build with each process opted out and a
10-s clock probe before each phase (127–130 ms per pass at boost, ~190 at
base clock), so the captures are comparable with each other. Whether Precision Boost 2 lowers the all-core sustained clock
under heavy parallel load has not been measured here. Evidence:
`docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/README.md`.
