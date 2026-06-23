# AGENTS.md

This file provides project-scoped guidance to AI agents and contributors working in this repository.

# libstats - Modern C++20 Statistical Distributions Library

## Project Overview

libstats is a **design and teaching library**: a demonstration of how to build statistical software correctly in modern C++20, with genuine SIMD and parallel performance. Zero external dependencies.

**Current Status**: v2.0.0 on `feat/v2-architecture` — development complete, pending three-machine
validation before merge to `main`. v1.5.3 is the final v1.x release.

16 distributions implemented across 6 families. 3 additional distributions — Geometric, Laplace,
Cauchy — are registered in the enum and metadata table (pending implementation).
v2.0.0 breaking changes (relative to v1.5.3):
- Platform baseline raised to macOS 13 Ventura; AppleClang 15+, GCC 13+, Clang 17+, MSVC 19.38+.
- Alternate Homebrew LLVM compiler path removed; system AppleClang only on macOS.
- All statistical analysis methods extracted from distribution classes to `stats::analysis` namespace
  (see `MIGRATION_GUIDE.md` for the complete old→new call mapping).
- `likelihoodRatioTest` requires explicit `df` parameter (position 4, before `alpha`).
- `kolmogorovSmirnovTest`/`andersonDarlingTest` constrained to `ContinuousDistribution` concept.
- `VoidResult = Result<std::monostate>`; success path is `VoidResult::ok({})` not `ok(true)`.
- `validateBetaParameters`, `validateChiSquaredParameters`, `validateStudentTParameters` are now
  free functions in `error_handling.h`; they are no longer private static members of their
  respective distribution classes.
- `FittableDistribution` concept enforces `std::default_initializable<D>` + `fit()` on bootstrap
  and cross-validation templates.
- `AnyDistribution` concept now requires `getSkewness()` and `getKurtosis()`.
- `SIMDPolicy::Level` is a type alias for `SIMDLevel` (one enum instead of two).
- `DistributionType` extracted to `include/core/distribution_type.h`.
- `WorkStealingPool::getOptimalThreadCount()` capped at 32 workers.
- Strategy-suffix batch methods, vector-returning batch helpers, `LibDistributionType`,
  `CROSS_PLATFORM` build type, and `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` removed.
- `noexcept` move constructors across all 16 distributions.
- `WorkStealingPool::parallelFor` per-call fence.
- Legacy `validation.cpp` / `validation.h` ecosystem deleted; use `stats::analysis` instead.
- `BinomialDistribution::getEntropy()` now uses exact PMF summation for n ≤ 1000 (nats).
- `PoissonDistribution::sample()` large-lambda path uses `std::poisson_distribution<int>` (exact).
- Include shim uses directory symlink on macOS/Linux; header edits are live without cmake re-run.
- `include/core/distribution_meta.h` (new): canonical `kDistributionMeta[]` table is the single
  registration point for all distribution metadata; `include/core/distribution_type.h` holds the enum.

Three-machine validation ecosystem: Kaby Lake AVX2+FMA, Mac Mini M1 NEON, Asus TUF A16 AVX-512.

## Session Start Baseline Workflow (Required)

At the start of every session, perform these steps in order:

1. Verify machine architecture before making SIMD assumptions.
2. Select the matching build path (macOS vs Windows/MSVC, Intel vs Apple Silicon).
3. Reconfigure/rebuild when the machine or architecture differs from the previous session context.

Quick architecture checks:

```bash
# macOS/Linux shells
uname -m
uname -s
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
```

```powershell
# PowerShell (Windows)
[System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
[System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture
$env:PROCESSOR_IDENTIFIER
```

### Why SIMD Detection Matters

The active SIMD tier changes fundamentally between machines. SIMD code paths, performance thresholds, and test results are architecture-dependent. If the machine has changed since the last session:
- Note the change explicitly.
- Verify the build directory is current for this architecture (`cmake ..` may be needed).
- Dispatch thresholds in `include/core/dispatch_thresholds.h` are architecture-specific.
- Benchmark results are not comparable across architectures.

| SIMD Tier | Example CPUs | Active simd_*.cpp files |
|---|---|---|
| SSE2 + AVX + AVX2 + FMA | Intel Haswell / Kaby Lake and newer | + `simd_avx2.cpp` |
| NEON only | Apple Silicon (M1 and newer) | `simd_neon.cpp` |
| SSE2 + AVX + AVX2 + **AVX-512** | AMD Zen 4 (e.g. Ryzen 7000-series) | + `simd_avx512.cpp` |
| SSE2 + AVX + AVX2 | Linux x86 CI | `simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp` |

The machines in the Development Ecosystem table are examples; any CPU with the same SIMD capabilities follows the same code paths.

Platform routing rules (OS/toolchain selection — SIMD tier is determined automatically at compile time by CPU feature detection):
- **macOS (Ventura 13+ required):** Use the standard CMake flow in the `Essential Build Commands` section.
- **Windows/MSVC:** Follow `Windows Session Setup` below and use Visual Studio 2022 x64 Release commands (defaults shown for Asus TUF A16; paths may differ on other machines).
- **All platforms:** After architecture verification, run `./build/tools/system_inspector --quick` (Unix shells) or `.\build\tools\system_inspector.exe --quick` (Windows PowerShell) to confirm active SIMD capabilities before interpreting performance/test results.

### Current Validation Matrix

**v2.0.0 — validation target (three machines)**

Ivy Bridge / macOS Catalina dropped from the ecosystem in v2.0.0 (Catalina EOL;
minimum macOS raised to 13 Ventura).

| Machine | SIMD | Target | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 43/43 ✅ | v2.0.0 validated; kAvx2 thresholds recalibrated (3-run) |
| Mac Mini M1 | NEON | 43/43 ✅ | v2.0.0 validated; kNeon thresholds recalibrated (3-run) |
| Asus TUF A16 (Windows) | AVX-512 | 43/43 ✅ | v2.0.0 validated; kAvx512 thresholds recalibrated (3-run); 61/61 simd_verification |

**v1.5.2 — final v1.x release (four machines)**

| Machine | SIMD | Correctness | Notes |
|---|---|---|---|
| Kaby Lake (2017 MBP) | AVX2+FMA | 39/39 ✅ | |
| Ivy Bridge (2012 MBP) | AVX | 38/38 ✅ | (last version with Catalina) |
| Mac Mini M1 | NEON | 39/39 ✅ | |
| Asus TUF A16 (Windows) | AVX-512 | 39/39 ✅ | |

**v1.5.1 — validated on all four machines**

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

**v1.4.0 baseline — all four machines**

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

### SIMD Batch Operation Speedups (Ivy Bridge AVX — v1.5.0, historical)
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

### Changes in v2.0.0
- **Distribution metadata table** (`include/core/distribution_meta.h`): canonical
  `kDistributionMeta[]` constexpr array with enum name, display name, `is_discrete`, and
  `is_delegation_wrapper` fields for all 19 registered types. `consteval validateMetaOrdering()`
  enforces index == enum value at compile time. Accessors: `distributionMeta()`,
  `distributionMetaSafe()`, `distributionEnumName()`, `distributionDisplayName()`.
- **`DistributionType` enum extended**: GEOMETRIC(16), LAPLACE(17), CAUCHY(18) appended
  (implementations pending; NEVER dispatch thresholds set in all four kXxx tables).
- **`dispatch_thresholds.h` refactored**: `ArchTable` changed from a named-field struct to
  `using ArchTable = std::array<ThresholdRow, kDistributionTypeCount>`. `parallelThresholdFromTable`
  replaces a 15-case switch with a 3-line array index lookup. Adding a distribution now requires
  only an enum append, a metadata row, and a ThresholdRow per table — no switch edits.
- **Registration drift fixed**: `performance_history.cpp::distributionTypeToString` (was 6/16 cases,
  silent key collision) and `tool_utils.h::distributionTypeToString` (was 9/16 cases) replaced with
  `distributionEnumName()` / `distributionDisplayName()` lookups. `system_inspector.cpp` hardcoded
  5-type list replaced with `kDistributionMeta` iteration.
- **Dispatch profiling infrastructure**: `summarize_dispatcher_profile.py` V→P crossover corrected
  to `min(PARALLEL, WORK_STEALING) < VECTORIZED`; `strategy_profile.cpp` batch grid updated;
  `capture_dispatcher_profile.sh` rewritten; `scripts/PROFILING_METHOD.md` added as canonical
  profiling procedure. kNeon (6 entries), kAvx2 (9 entries), and kAvx (inferences) recalibrated;
  16 pre-v2.0.0 profile bundles removed.
- **`strategy_profile.cpp` `STRATEGIES` array** documented with a registration comment pointing to
  the compiler-enforced `executeStrategy` switch as the completeness counterpart.

43/43 correctness tests pass on Kaby Lake AVX2+FMA and Mac Mini M1 NEON after all v2.0.0
infrastructure work. Asus TUF A16 (AVX-512): re-run `strategy_profile` with corrected
`summarize_dispatcher_profile.py` to regenerate canonical kAvx512 thresholds before PR merge.

### Deferred Items
- `vector_floor` + `vector_blend` primitives across all SIMD backends to enable
  branchless Discrete CDF and Uniform PDF/LogPDF; low priority given existing batch-path speedups
  (Discrete 8–15x, Uniform 39–54x) already achieved through amortization
- `vector_lgamma` — too complex, low immediate distribution impact; indefinitely deferred
- SVE (AArch64 beyond NEON) — no hardware in the ecosystem
- SSE4.1 tier — SSE2 magic-number workaround adequate; not worth a dedicated tier

### Changes in v1.5.2
- **Critical bug fixes (June 2026 audit)**: `gammaQ` infinite recursion fixed with Legendre CF;
  `bayesianCredibleInterval` now reads `credibility_level` and uses exact Gamma posterior
  quantiles; `safe_log(+inf)` returns `+inf`; `safe_exp` underflow returns `0.0`.
- **Thread-safety fixes**: `recordPerformance` min/max updated with CAS loops;
  `WorkStealingPool` destructor drains tasks before shutdown; `GaussianDistribution`
  copy constructor acquires source lock before base-class copy; `GammaDistribution`
  copy constructor uses `shared_lock` on source.
- **Numerical quality**: `betaI_continued_fraction` near-zero guard fires before division;
  `ConvergenceDetector` uses `std::deque` for O(1) `pop_front()`; Newton-Raphson
  near-zero derivative triggers a hard `return x` instead of `break`.
- **Structural**: `result_of_t` consolidated to `include/platform/internal/type_traits.h`;
  `RecoveryStrategy` converted to `enum class`; dead `using std::shared_lock` declarations
  removed from `distribution_common.h`; `PerformanceDispatcher::SIMDArchitecture` and
  unused `Thresholds` distribution-specific fields deprecated; `LibDistributionType` removed.
- **Documentation**: `CachedProperty<T>` and `ThreadSafeCacheManager` dual-flag pattern
  documented; `LogSpaceOps::initialize()` guarded with `call_once`.

Full three-machine validation (v1.5.2): identical correctness to v1.5.1.
- 39/39 correctness on Kaby Lake, M1, Asus TUF A16

### Changes in v1.5.1
- **Dispatch table expanded to all 16 distributions**: `kNeon`, `kAvx`, `kAvx2`, `kAvx512` now
  include calibrated entries for LogNormal, Pareto, Weibull, Rayleigh, VonMises, Binomial, and
  NegativeBinomial. Two Release-mode profiling bundles per architecture.
- **Correctness fixes**: `#include <pair>` → `<utility>`; `shouldUseSIMDBatch` delegates to
  `SIMDPolicy::shouldUseSIMD()`; dead `withCachedParameters` removed; `arch::simd` aliases
  formalized; `/utf-8` MSVC flag added.
- **`simd_verification`**: relative error reported for VectorExp/VectorLog (`max_rel=`);
  absolute diff is meaningless at exp(500)~5e+217 magnitudes.
- **Test fixes**: dispatcher threshold-aware assertion; `minimal_latency()` thread-count
  corrected; VonMises tolerance relaxed to 1e-10 for AVX-512 `vector_cos` error floor.

Full four-machine validation (v1.5.1): identical SIMD performance to v1.5.0.
- 39/39 correctness, 61/61 `simd_verification` on Kaby Lake, M1, Asus TUF A16
- 38/38 correctness, 61/61 `simd_verification` on Ivy Bridge (last Catalina validation)

### Changes in v1.5.0
- **AVX2+FMA native transcendentals**: `vector_exp_avx2` and `vector_log_avx2` replaced
  AVX-delegation stubs with FMA Horner polynomial (SLEEF-inspired, < 1 ULP). `vector_cos_avx2`
  replaced AVX delegation with native FMA Horner. Measured: VectorExp 3.6x → 3.4x average
  (was 1.7x delegating); VectorLog 1.7x (was 1.4x).
- **High-accuracy `vector_erf`** (all x86 backends): replaced A&S 7.1.26 (~1.5×10⁻⁷) with
  musl libc four-region rational polynomial (< 1 ULP; measured max error 2.22×10⁻¹⁶).
  `vector_erf_avx` uses mul+add (−mavx only); `vector_erf_avx2` uses FMA; `vector_erf_sse2`
  uses `__m128d` with SSE2 and/andnot/or blending. Gaussian CDF SIMD error: 6.97×10⁻⁸ → ~0.
- **`simd_verification` coverage and reporting**: added VonMises distribution rows and
  primitive vector op rows (VectorExp/Log/Erf/Cos); 54 → 61 tests. Replaced the single
  wall-clock composite speedup with per-op-type geometric means (PDF/LogPDF/CDF) and
  per-primitive individual rows.
- **NEON native transcendentals** (Phase 3, M1): `vector_exp_neon` (SLEEF FMA Horner,
  < 1 ULP), `vector_log_neon` (SLEEF atanh series, < 1 ULP), `vector_erf_neon`
  (ARM glibc table+Taylor, ~2.29 ULP) — validated ✅. 39/39 correctness, 61/61
  simd_verification. Distribution geomeans: PDF 5.9x, LogPDF 7.3x, CDF 3.1x.
  Primitive ops: VectorExp 2.1x, VectorLog 1.8x, VectorErf 8.0x, VectorCos 3.0x.
  `vector_erf_neon` uses a 769-entry precomputed table (`src/neon_erf_data.inc`,
  12,304 bytes) rather than the musl rational polynomial used by all x86 backends;
  the table approach eliminates the recursive exp call and achieves 8.0x vs 0.9x
  for the pure-polynomial version. See Issue #33 for a proposed cross-architecture
  experiment to evaluate the table approach on exp and log as well.
- **AVX-512 native transcendentals** (Phase 4, Asus TUF A16): `vector_exp_avx512`,
  `vector_log_avx512`, `vector_erf_avx512` — validated ✅. 39/39 correctness, 61/61
  simd_verification. Distribution geomeans: PDF 4.8x, LogPDF 5.1x, CDF 2.2x.
  Primitive ops: VectorExp 5.0x, VectorLog 3.9x, VectorErf 1.3x, VectorCos 8.5x.

Phase 5 (dispatch threshold recalibration):
- All four `ArchTable` entries in `dispatch_thresholds.h` updated from v1.5.0 bundles.
- NEON Gaussian CDF: NEVER (table erf unbeatable up to 500k).
- AVX2 Gaussian PDF: 100000; StudentT PDF: 250000 (FMA exp improvements).
- AVX-512 Exponential PDF / StudentT PDF+LogPDF: NEVER (8-wide native exp).

Full four-machine validation (v1.5.0):
- 39/39 correctness, 61/61 `simd_verification` on Kaby Lake, M1, Asus TUF A16
- 38/38 correctness, 61/61 `simd_verification` on Ivy Bridge/Catalina

### Changes in v1.4.0
- **`vector_cos`** added to `VectorOps` across all five SIMD backends (AVX/AVX2/SSE2/NEON/AVX-512).
  Two-step range reduction + 7-term Horner polynomial; max error ≈ 1×10⁻¹⁰.
  AVX-512 uses native 8-wide path (`_mm512_roundscale_pd`); SSE2 uses magic-number rounding.
- **VonMises LogPDF/PDF batch** now SIMD-accelerated via the 4-step pipeline:
  `scalar_add(−μ)` → `vector_cos` → `scalar_multiply(κ)` → `scalar_add(−ln Z)`.
- **SIMD dispatch table** (Issue #22): `VectorOps::DispatchTable` replaces 11 repeated
  5-tier dispatch chains in `simd_dispatch.cpp`; adding a new SIMD tier now requires
  editing one function (`makeDispatchTable`).
- Code-review fixes (Findings 1–7): domain constant decoupling, `erf_inv` `static constexpr`,
  `FeaturesSingleton` Rule of Five, named magic literals, namespace style, clang-tidy config.

Validation (v1.4.0, Kaby Lake AVX2 primary):
- correctness suite: 39/39 PASS
- `simd_verification`: 54/54 PASS, overall 3.35x

### Distributions Added in v1.3.0
New distributions added in v1.3.0:
- **Binomial** — B(n, p); PMF via lgamma log-space; CDF via I_{1−p}(n−k, k+1);
  MLE closed-form p̂ = k̄/n̂. VECTORIZED = cached scalar loop.
- **Negative Binomial** — NB(r, p); real-valued r; PMF via lgamma; CDF via I_p(r, k+1);
  MLE: MoM seed + Newton–Raphson profile score using digamma/trigamma.
  Sampling: Gamma(r,(1−p)/p)-Poisson mixture.

Shared utility additions:
- `detail::trigamma(x)` added to `math_utils`: A&S §6.4.12, accuracy < 2×10⁻¹⁴.

Validation (v1.3.0, Kaby Lake AVX2 primary):
- correctness suite: 39/39 PASS
- `simd_verification`: 54/54 PASS, overall 4.10x (unchanged — discrete distributions use scalar loops)

### Distributions Added in v1.0.0
New distributions added in v1.0.0:
- **Student's t** — standalone implementation with SIMD log-space PDF/LogPDF and CDF via incomplete beta
- **Chi-squared** — delegation wrapper over Gamma(α=ν/2, β=1/2)
- **Beta** — standalone bounded-support distribution with two-log SIMD PDF/LogPDF and CDF via regularized incomplete beta

Shared utility additions:
- `detail::digamma(x)` promoted into `math_utils`
- `detail::inverse_beta_i(p, a, b)` added for Beta quantiles

Validation (v1.0.0):

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
- Overall speedup limited by transcendental delegation to AVX (see Deferred Items)

Kaby Lake AVX2 (2017 MBP):
- correctness suite: 33/33 PASS
- `simd_verification`: 54/54 PASS, overall 3.49x
- new-distribution speedups: Chi-squared PDF 13.8x/LogPDF 10.5x, Student's t PDF 6.3x/LogPDF 18.4x,
  Beta PDF 5.3x/LogPDF 4.1x

All four machines validated at v1.0.0 (Ivy Bridge/Catalina dropped in v2.0.0).

### Development Ecosystem

| Machine | OS | CPU | SIMD | Notes |
|---|---|---|---|---|
| MacBook Pro 14,1 (2017) | macOS Ventura | Intel Kaby Lake | SSE2 + AVX + AVX2 + FMA | AVX2/FMA validation |
| Mac Mini M1 | macOS Tahoe | Apple Silicon M1 | NEON only | ARM/NEON path validation |
| Asus TUF A16 (2025) | Windows 11 Pro | AMD Ryzen 7 7445 (Zen 4) | SSE2 + AVX + AVX2 + **AVX-512** | Windows/MSVC + first AVX-512 machine |

**Note:** The Asus TUF A16 (Ryzen 7 7445, Zen 4) is the first machine in this ecosystem with AVX-512 support.
`simd_avx512.cpp` was first exercised there at v1.0.0; 54/54 SIMD tests pass.
The `test_simd_policy` AVX-512 string (`"AVX-512"`) was confirmed correct.

**Note:** If setting up a fresh Windows machine, the build environment must be configured from scratch; see the one-time setup notes below.

### Windows Session Setup

> **Windows tool paths vary** by installation method (direct installer, `winget`, `chocolatey`, Microsoft Store, etc.). The paths below are common defaults — adjust for your installation. VS Build Tools and full VS editions use different default directories; see the one-time setup notes below for alternatives and auto-detection.

Before building or running tests in a new PowerShell session on Windows:

```powershell
# 1. Activate MSVC toolchain (required each session — not persistent in PowerShell)
# Default path for VS 2022 Build Tools. For full VS (Community/Professional/Enterprise),
# replace "BuildTools" with your edition under "C:\Program Files\Microsoft Visual Studio\2022\".
# See One-time setup notes below for auto-detection via vswhere.exe.
$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$envVars = cmd /c "`"$vcvars`" > nul && set"
foreach ($line in $envVars) {
    if ($line -match "^([^=]+)=(.*)$") {
        [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
}

# 2. Set UTF-8 output (required for Unicode glyphs in tool output)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 3. Ensure stats.dll is accessible for dynamic linking tests
Copy-Item "build\Release\stats.dll" -Destination "build\tests\" -Force

# 4. Run correctness tests
ctest --test-dir build -C Release -LE "timing|benchmark" --output-on-failure
```

**Important: After any clean rebuild on Windows, verify the dynamic test EXEs are Release builds:**
```powershell
dumpbin /imports build\tests\test_gaussian_basic_dynamic.exe | Select-String vcruntime
# Must show VCRUNTIME140.dll (Release), NOT VCRUNTIME140D.dll (Debug)
# If Debug CRT is shown, the EXE is a stale Debug binary. Fix:
#   Remove-Item build\tests\test_gaussian_basic_dynamic.exe, test_exponential_basic_dynamic.exe -Force
#   cmake --build build --config Release --target test_gaussian_basic_dynamic test_exponential_basic_dynamic
```
The VS generator puts Debug and Release test EXEs in the same `build\tests\` directory.
A stale Debug EXE + Release DLL = CRT mismatch = heap corruption crash. The `cmake --build --clean-first`
flag cleans Release artifacts but leaves existing Debug EXEs untouched if their timestamps appear current.

**One-time setup notes:**
- Visual Studio 2022 Build Tools (not full IDE) is sufficient. Install from https://aka.ms/vs/17/release/vs_buildtools.exe, or `winget install Microsoft.VisualStudio.2022.BuildTools`, or `choco install visualstudio2022buildtools`.
  - Build Tools default path: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\`
  - Full VS (Community/Professional/Enterprise) default path: `C:\Program Files\Microsoft Visual Studio\2022\{edition}\`
  - Auto-detect installation path (any edition): `& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath`
- **Smart App Control must be Off** (Windows Security → App & Browser Control → SAC settings).
  SAC blocks locally compiled executables. Cannot be re-enabled without a Windows reset.
- CMake ≥ 3.20 required. Install from https://cmake.org/download/, `winget install Kitware.CMake`, or `choco install cmake`.
- vcpkg for GTest: `git clone https://github.com/microsoft/vcpkg C:\vcpkg && C:\vcpkg\bootstrap-vcpkg.bat`. The path `C:\vcpkg` is a convention; if installed via `winget install Microsoft.vcpkg` or `choco install vcpkg` the location will differ — run `where vcpkg` to find it.
- Configure: `cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake`
  (adjust the toolchain path if vcpkg is not at `C:\vcpkg`)
- Build: `cmake --build . --config Release --parallel`
- GTest installed via vcpkg (`gtest:x64-windows 1.17.0`) — all 33 correctness tests pass

## Essential Build Commands

### Quick Build
```bash
# macOS/Linux — standard development build (default 'Dev' build type)
cmake -B build
cmake --build build --parallel   # equivalent to make -j$(nproc)
ctest --test-dir build --output-on-failure
```

Windows: use the commands in the `Windows Session Setup` section above.

### Common Build Configurations
```bash
# Development (default) - light optimization with debug info
cmake ..

# Production release - maximum optimization
cmake -DCMAKE_BUILD_TYPE=Release ..

# Full debugging support
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Strict compiler warnings as errors (for compatibility testing)
cmake -DCMAKE_BUILD_TYPE=Strict ..   # v2.0.0: unified Strict mode replaces legacy compiler-specific strict aliases
```

### Build System Features
- **Automatic parallel detection**: Detects CPU cores and configures optimal builds
- **Compiler detection**: System AppleClang on macOS (Ventura 13+); GCC 13+ / Clang 17+ on Linux
- **SIMD optimization**: Runtime CPU feature detection with fallbacks
- **Cross-platform**: Native Windows, macOS, Linux support

### Important Build Directories
- **Executables**: `build/tools/` (never `bin/` - this doesn't exist)
- **Tests**: `build/tests/`
- **Examples**: `build/` (built by examples/CMakeLists.txt)

### Development Tools
```bash
# System analysis and diagnostics
./build/tools/system_inspector --full
./build/tools/cpp20_features_inspector

# Performance analysis
./build/tools/strategy_profile
./build/tools/simd_verification

# Dispatcher profiling bundle capture
./scripts/capture_dispatcher_profile.sh

# Cross-compiler compatibility testing
./scripts/test-cross-compiler.sh --clean
```

## Ad Hoc Compilation Outside CMake

For quick diagnostics and testing, compile directly without CMake. Use the system compiler on
macOS (Ventura 13+); alternate LLVM compiler setup is not required and not supported in v2.0.0.

### macOS / Linux
```bash
# macOS — system AppleClang (recommended)
clang++ -std=c++20 -stdlib=libc++ \
  -I./include \
  -L./build \
  your_test.cpp -o test_output ./build/libstats.a

# Linux — GCC 13+ or Clang 17+
g++ -std=c++20 -Wall -Wextra -O2 \
  -I./include \
  -L./build \
  your_test.cpp -o test_output -lstats
```

#### Quick Test Template
```cpp
#include "libstats.h"
#include <iostream>

int main() {
    auto result = stats::GaussianDistribution::create(0.0, 1.0);
    if (result.isOk()) {
        auto& g = result.value;
        std::cout << "PDF at 0: " << g.getProbability(0.0) << "\n";
        std::cout << "CDF at 1: " << g.getCumulativeProbability(1.0) << "\n";
    }
}
```

### Troubleshooting
- **Library not found**: Use static linking (`./build/libstats.a`) instead of `-lstats`.
- **Header not found**: Verify `-I./include` path is correct relative to the project root.
- **C++20 features not available**: Ensure compiler version meets minimum (AppleClang 15, GCC 13, Clang 17).

## Project Architecture

### High-Level Structure

libstats follows a strict **layered dependency architecture** with 6 levels:

```
Level 0: Foundation (constants, basic platform detection)
Level 1: Core Utilities (math, safety, validation) + Platform (SIMD, threading)
Level 2: Advanced Infrastructure (caching, performance framework)
Level 3: Distribution Framework (base classes, interfaces)
Level 4: Concrete Distributions (Gaussian, Exponential, etc.)
Level 5: Complete Library Interface (libstats.h)
```

### Key Architectural Concepts

#### Dual API Design
- **Auto-dispatch API**: Intelligent automatic strategy selection (recommended for most users)
- **Explicit strategy API**: Direct control over SIMD/parallel execution for power users

#### Performance Systems
- **SIMD Optimization**: Cross-platform runtime detection (SSE2/AVX/AVX2/NEON)
- **Parallel Execution**: Auto-dispatching between scalar, SIMD, and parallel strategies
- **Adaptive Cache**: Performance-aware caching with memory optimization
- **Performance History**: Machine learning for strategy selection improvement

#### Thread Safety
- **Lock-free fast paths**: Atomic parameter access for high-frequency operations
- **Reader-writer locks**: Shared_mutex for cache updates
- **Concurrent batch operations**: Thread-safe SIMD and parallel processing

### Core Components

#### Statistical Distributions (16 implemented + 3 registered, across 6 families)
1. **Gaussian** (Normal) - N(μ, σ²)
2. **Exponential** - Exp(λ)
3. **Uniform** - U(a, b)
4. **Poisson** - P(λ)
5. **Discrete** - Custom discrete distributions
6. **Gamma** - Γ(α, β)
7. **Chi-squared** - χ²(ν) — delegation wrapper over Gamma(α=ν/2, β=1/2)
8. **Student's t** - t(ν) — SIMD log-space PDF/LogPDF and CDF via incomplete beta
9. **Beta** - Beta(α, β) — two-log SIMD PDF/LogPDF and CDF via regularized incomplete beta
10. **Log-Normal** - LogN(μ, σ) — log+exp pipeline
11. **Pareto** - Pareto(xₘ, α) — log-only pipeline, power-law tail
12. **Weibull** - W(k, λ) — log+exp pipeline, reliability engineering
13. **Rayleigh** - R(σ) — x² pipeline, signal processing
14. **Von Mises** - VM(μ, κ) — circular distribution, SIMD via vector_cos
15. **Binomial** - B(n, p) — discrete, PMF via lgamma
16. **Negative Binomial** - NB(r, p) — discrete, real-valued r, Newton–Raphson MLE
17. **Geometric** - Geo(p) — *(registered, pending implementation)* delegate over NegBinomial(r=1)
18. **Laplace** - Laplace(μ, b) — *(registered, pending implementation)* standalone, median/MAD fit
19. **Cauchy** - Cauchy(x₀, γ) — *(registered, pending implementation)* delegate over StudentT(ν=1)

Each implemented distribution provides: PDF/CDF/Quantiles, Statistical Moments, Parameter Estimation (MLE), Random Sampling, Statistical Validation, SIMD batch operations.

#### Platform Optimization
- **CPU Feature Detection**: Runtime SIMD capability detection
- **Threading Systems**: Comprehensive detection (TBB, OpenMP, pthreads, GCD, Windows Thread Pool)
- **Memory Management**: SIMD-aligned allocations and cache-aware algorithms

## Code Organization

### Header Architecture
```
include/
├── libstats.h              # Complete library (single include)
├── core/                   # Core mathematical and statistical components
│   ├── constants/          # Mathematical, precision, statistical constants
│   ├── distribution_type.h     # DistributionType enum (append-only)
│   ├── distribution_meta.h     # kDistributionMeta[] — canonical registration table
│   ├── dispatch_thresholds.h   # Per-architecture parallel thresholds (indexed by DistributionType)
│   ├── distribution_*.h    # Distribution framework components
│   └── *_common.h         # Consolidated headers for faster compilation
├── distributions/          # Concrete distributions (gaussian.h, etc.)
├── stats/
│   └── analysis/           # Statistical tests and estimators (stats::analysis::)
│       ├── analysis.h      # Umbrella include
│       ├── goodness_of_fit.h, bootstrap.h, cross_validation.h, information_criteria.h
│       └── gaussian_analysis.h, poisson_analysis.h, exponential_analysis.h, …
└── platform/              # SIMD, threading, parallel execution
```

### Source Organization
```
src/
├── [Level 0-1] Foundation and utilities (cpu_detection.cpp, safety.cpp)
├── [Level 2] Platform capabilities (thread_pool.cpp, work_stealing_pool.cpp)
├── [Level 3] Infrastructure (benchmark.cpp, performance_dispatcher.cpp)
├── [Level 4] Framework (distribution_base.cpp)
└── [Level 5] Distributions (gaussian.cpp, exponential.cpp, etc.)
```

### Object Library Architecture
The CMake system uses dependency-aware object libraries for parallel compilation:
- `libstats_foundation_obj` → `libstats_core_utilities_obj` → `libstats_infrastructure_obj` → `libstats_framework_obj` → `libstats_distributions_obj`
- Enables optimal incremental builds and clear architectural boundaries

## Common Development Tasks

### Working with Distributions

#### Creating New Distributions
The registration checklist is authoritative in `include/core/distribution_meta.h`. Summary:
1. **Append** the new `DistributionType` enum value to `include/core/distribution_type.h`
   (append-only; never reorder — values are used as array indices).
2. **Append** a `DistributionMeta` row to `kDistributionMeta[]` in `include/core/distribution_meta.h`
   (enum name, display name, `is_discrete`, `is_delegation_wrapper`).
3. **Append** one `ThresholdRow` to each of the four `kXxx` tables in
   `include/core/dispatch_thresholds.h` (use `{NEVER, NEVER, NEVER}` until profiled).
4. **Implement** the distribution header (`include/distributions/`), source (`src/`), and tests
   (`tests/*_basic.cpp`, `tests/*_enhanced.cpp`) — see `exponential.h` as the reference template.
5. **Register** in `CMakeLists.txt` and `include/libstats.h`.

The `consteval validateMetaOrdering()` in `distribution_meta.h` enforces step 1↔2 alignment at
compile time. After any enum or table change, a clean build verifies consistency.

#### Testing Strategy
- **All levels**: GTest-based tests registered with CTest
- Correctness tests: run `ctest -LE "timing|benchmark"` (parallel-safe)
- Timing tests: run `ctest -j1 -L timing` on a quiet machine
- **Coverage**: 23 correctness tests + timing/benchmark suite

### Performance Optimization

#### SIMD Development
- Use `libstats::simd::*` namespace for vectorized operations
- Runtime dispatch automatically selects best available instruction set
- Test with `./build/tools/simd_verification`

#### Parallel Processing
- Auto-dispatch API: `getProbability(std::span<const double>, std::span<double>, hint)`
- Explicit control: span-based batch APIs with `detail::PerformanceHint`
- Dispatch thresholds are per-(architecture, distribution, operation) in `dispatch_thresholds.h`
- Thresholds derived from four-architecture profiling data in `data/profiles/dispatcher/`

### Build System Customization

#### CMake Options
```bash
# Enable verbose build messages for debugging
cmake -DLIBSTATS_VERBOSE_BUILD=ON ..

# Force TBB usage over platform-native threading
cmake -DLIBSTATS_FORCE_TBB=ON ..


# Disable tools or tests
cmake -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_TESTS=OFF ..
```

#### Compiler-Specific Builds
The build system supports cross-compiler compatibility testing with specialized build types that enable consistent warning levels across GCC, Clang, and MSVC.

## Important Guidelines

### Code Standards
- **C++20 Required**: Modern features (concepts, spans, execution policies)
- **Header Guards**: Use `#pragma once` (codebase convention)
- **Naming**: CamelCase classes, snake_case functions/variables
- **Memory Management**: Smart pointers, RAII, no raw pointers
- **Error Handling**: Dual API (Result<T> for factories, exceptions for setters)

### Performance Considerations
- Always rebuild after source changes before running tests
- Use `initialize_performance_systems()` for optimal batch performance
- SIMD operations require 16-byte aligned data (handled automatically)
- Large batch operations (>1000 elements) benefit significantly from parallel execution

### Platform-Specific Notes
- **macOS**: System AppleClang is the default and only supported v2.x compiler path (Ventura 13+).
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows

## Testing and Validation

### Running Tests
```bash
# Run all tests (timing assertions may be flaky under parallel load)
ctest --test-dir build --output-on-failure

# Correctness only — safe to run in parallel, excludes timing-sensitive assertions
ctest --test-dir build --output-on-failure -LE "timing|benchmark"

# Timing validation — run serially on a quiet machine for reliable results
ctest --test-dir build --output-on-failure -j1 -L timing

# Or via make targets (macOS/Linux with Makefile generator only)
make run_tests          # Correctness suite (parallel-safe)
make run_tests_timing   # Timing suite (serial, quiet machine required)
make run_all_tests      # Everything
# Windows equivalent: cmake --build build --target run_tests --config Release

# Run a specific test
ctest --test-dir build -R test_gaussian_basic
ctest --test-dir build -R test_gaussian_enhanced  # Contains timing assertions

# Run cross-compiler compatibility tests
./scripts/test-cross-compiler.sh
```

### Test Labels
- **no label** — correctness tests; safe to run in parallel
- **timing** — contains speedup/overhead assertions; run with `-j1` for reliable results
- **benchmark** — performance benchmarks; not part of the standard test suite

Timing tests fail under CPU contention because parallel strategies show less speedup
when the machine is loaded. This is a measurement problem, not a correctness problem.

### Performance Validation
```bash
# Verify SIMD operations and performance
./build/tools/simd_verification

# Profile forced strategies for threshold tuning
./build/tools/strategy_profile

# System capability analysis
./build/tools/system_inspector --performance
```

The testing infrastructure ensures correctness across all optimization levels and provides regression detection for performance-critical paths.

## Warp Terminal Saved Workflows (warp.dev only)

> **Note for non-Warp users:** These workflows are available only in the Warp terminal. Users of other tools (Claude Code, Cursor, bare shells, etc.) should run the equivalent shell commands listed elsewhere in this file.

Saved workflows in `.warp/workflows/` are available directly in the Warp terminal for common tasks:

- **libstats: Clean Rebuild** — remove `build/` and rebuild from scratch; accepts `build_type` arg (default: `Dev`)
- **libstats: Validate Machine** — architecture detection, SIMD capabilities, correctness suite, and `simd_verification`; requires a current build
- **libstats: Switch Branch + Rebuild** — stash uncommitted changes, fetch, checkout target branch, pull, and clean rebuild in one step
- **libstats: Warning Audit** — build with a strict warning mode and display deduplicated warning counts; accepts `build_type` arg (default: `ClangWarn`)
