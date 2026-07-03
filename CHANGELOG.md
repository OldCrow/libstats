## [2.0.2] - 2026-07-02

### Tests

- **`getSIMDValidationThreshold` AMD/Intel split for AVX-512**: split the
  `complex_multiplier` between AMD (0.62) and Intel (0.70) for AVX-512 builds.
  AMD Zen4+ double-pumps AVX-512 through 256-bit execution units, yielding lower
  measured speedup for `lgamma`-heavy distributions (Poisson, Gamma) than true-512
  Intel hardware. Calibrated from observed 1.33×/1.45× Poisson speedup at batch
  5k/50k on Ryzen Zen4 vs the previous unified 1.40×/1.47× threshold. Resolves
  `test_poisson_enhanced` failures on Windows/Zen4. 69/69 tests pass.

## [2.0.1] - 2026-07-02

### Fixed

- **`PoissonDistribution::getEntropy()` returns 0 for λ ∈ (20, 100]**: The exact-summation loop
  exits early when `P(k=0) = exp(−λ) < 1e-15`, which triggers at λ ≳ 34.5. The Stirling
  branch threshold lowered from `λ > 100` to `λ > 20` (approximation error < 0.001 nats
  at λ = 20); the exact loop now only runs for λ ≤ 20 where `exp(−λ) ≫ 1e-15`.
- **`DiscreteDistribution` rejects degenerate distributions (a == b)**: `validateDiscreteParameters`
  (in `error_handling.h`) and `validateParameters` (in `discrete.cpp`) used `a >= b`, incorrectly
  refusing the valid degenerate case (H = log(1) = 0). Relaxed to `a > b`. `getKurtosis()` gains
  a NaN guard for n = 1 to avoid division by zero in the formula −6(n²+1)/(5(n²−1)).

## [2.0.0] - 2026-07-01

### Breaking changes

**Platform baseline raised**
- Minimum macOS raised to 13 Ventura (Catalina / Ivy Bridge support dropped).
- Minimum compilers: AppleClang 15, GCC 13, Clang 17, MSVC 19.38 (VS 2022 17.8).
- Alternate Homebrew LLVM compiler path removed; system AppleClang is the only supported macOS compiler.

**Namespace**
- `stats::` is now the primary namespace. `libstats::` remains as an alias (`namespace libstats = stats;`) for source compatibility, but all new code should use `stats::`.

**Analysis utilities extracted**
- Statistical analysis methods previously on distribution classes have moved to the `stats::analysis` namespace:
  - `GaussianDistribution::shapiroWilkTest` → `stats::analysis::gaussian::shapiroWilkTest`
  - `GaussianDistribution::kolmogorovSmirnovTest` → `stats::analysis::kolmogorovSmirnovTest`
  - `GaussianDistribution::andersonDarlingTest` → `stats::analysis::andersonDarlingTest`
  - `GaussianDistribution::kFoldCrossValidation` → `stats::analysis::kFoldCrossValidation<D>`
  - `GaussianDistribution::bootstrapParameterConfidenceIntervals` → `stats::analysis::bootstrapMeanCI<D>` / `stats::analysis::bootstrapMeanVarianceCI<D>`
  - All 16 distributions: `informationCriteria` → `stats::analysis::informationCriteria`
  - See MIGRATION_GUIDE.md for the complete mapping.

**`likelihoodRatioTest` signature change**
- `df` is now an explicit required parameter (position 4, before `alpha`).
- Old: `likelihoodRatioTest(data, restricted, unrestricted, alpha=0.05)`
- New: `likelihoodRatioTest(data, restricted, unrestricted, df, alpha=0.05)`
- Automatic df inference from parameter counts was removed (it was unreliable for same-type model comparison).

**KS/AD tests constrained to continuous distributions**
- `stats::analysis::kolmogorovSmirnovTest` and `andersonDarlingTest` now require `ContinuousDistribution` (C++20 concept). Passing a discrete distribution is a compile-time error. Use `chiSquaredGoodnessOfFitTest` for discrete distributions.

**Removed APIs**
- Strategy-suffix batch methods (e.g. `getProbabilityBatch`) removed; use the span-based `getProbability(span, span)` overload.
- `CROSS_PLATFORM` build type removed (replaced by unified `Strict` build type).
- `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` CMake flag removed.
- `LibDistributionType` enum removed (`detail::DistributionType` is the canonical enum).

### Fixed

- **Thread safety (Batch 1)**: Gamma copy/move assignment locking; Poisson/Exponential atomic parameter invalidation; Binomial setter TOCTOU race; `cacheValidAtomic_` shadowing removed from derived classes.
- **Mathematical correctness (Batch 2)**: Gamma CDF duplication using live `detail::gamma_p`; Binomial batch NaN guards for p=0/p=1; Poisson quantile overflow (64-bit); Gaussian standardized-value stale cache rebuild.
- **Build system (Batch 3)**: `safe_xgetbv` AVX OS-support check; compiler minimum enforcement in CMake; `Threads::Threads` link for Linux; pkg-config private deps; CI workflow GCC 13; run_all_tests target.
- **AD p-value continuity (MC-6)**: Single monotone formula replaces two-segment piecewise with a jump discontinuity.
- **Gaussian CI variance estimator (MC-7)**: Bessel-corrected (n-1) variance used throughout when population variance is unknown.
- **Binomial MLE bias (MC-10)**: `fit()` uses method-of-moments n̂ = x̄²/(x̄−s²) instead of max(obs) as the n estimate.

### Changed

- `stats::analysis::andersonDarlingTest` p-value uses a single continuous exponential approximation, calibrated to the 5% critical value of the distribution-agnostic AD asymptotic distribution (Stephens 1974).
- `BinomialDistribution::fit()` uses method-of-moments estimation for n, falling back to max(obs) when data is overdispersed relative to Binomial.
- Version constants in `libstats.h` updated to 2.0.0.
- Include shim uses directory symlink on macOS/Linux (live updates, no cmake re-run required); flat copy with build-time refresh on Windows.
- `NOT noexcept` Doxygen warnings removed from all distribution headers; move constructors and assignment operators are `noexcept` throughout.
- **Distribution metadata table** (`include/core/distribution_meta.h`): new canonical `kDistributionMeta[]`
  constexpr array replaces all scattered per-site distribution registration. `consteval validateMetaOrdering()`
  enforces enum-index alignment at compile time. `distributionEnumName()` / `distributionDisplayName()`
  accessors replace incomplete switches in `performance_history.cpp` (was 6/16 cases — silent key collision
  bug) and `tool_utils.h` (was 9/16 cases). `system_inspector.cpp` hardcoded 5-type list replaced with
  metadata iteration.
- **`DistributionType` enum**: GEOMETRIC(16), LAPLACE(17), CAUCHY(18) appended; implementations pending.
- **`dispatch_thresholds.h`**: `ArchTable` replaced with `using ArchTable = std::array<ThresholdRow,
  kDistributionTypeCount>`; `parallelThresholdFromTable` replaces a 15-case switch with an array index
  lookup; all four `kXxx` tables extended with NEVER rows for the three new types.
- **Dispatch profiling infrastructure**: `summarize_dispatcher_profile.py` V→P crossover definition
  corrected to `min(PARALLEL, WORK_STEALING) < VECTORIZED` (was PARALLEL-only); `strategy_profile.cpp`
  batch grid updated; `capture_dispatcher_profile.sh` rewritten; `scripts/PROFILING_METHOD.md` added as
  canonical profiling procedure; kNeon (6 entries), kAvx2 (9 entries), kAvx (inferences) recalibrated;
  16 stale pre-v2.0.0 profile bundles removed.

- **API rationalization**: `CpuTier` enum (`include/platform/internal/cpu_tier.h`) collapses
  24 vendor-string cascades in platform_constants_impl.cpp. Intel CPU classifier functions
  (`is_sandy_ivy_bridge()` etc.) and redundant cache getters removed from `cpu_detection.h`.
  `empirical_cdf`, `calculate_quantiles`, `sample_moments`, `validate_fitting_data` promoted
  from `stats::detail::` to `stats::analysis::` (`statistical_utilities.h`). `getEntropy()`
  moved from `DistributionBase` to `DistributionInterface`; added to `AnyDistribution` concept.
  `numericalIntegration`, `newtonRaphsonQuantile`, `adaptiveSimpsonIntegration`,
  `betaI_continued_fraction` removed from `DistributionBase` (no callers). `isApproximatelyEqual()`
  parameter changed from `const DistributionBase&` to `const DistributionInterface&`.
  `DistributionValidator`, `ExtendedValidationError`, and 9 dead `detail::` functions removed
  from `distribution_validation.h`. `ComputationComplexity` enum and `complexityToString()`
  removed. Three orphaned performance constants removed.
- **Full dispatch threshold recalibration**: kAvx2 (Kaby Lake) and kNeon (M1) re-profiled with
  3-run standard + 3-run --large; Beta thresholds corrected from NEVER to real values on all
  three architectures; kAvx inferences rebased; kNone three-tier complexity split (2048/8192/16384).
- **Correctness suite**: 44 tests (up from 43); new `test_statistical_utilities` and
  `test_discrete_analysis` added; 9 enhanced test files updated to span+PerformanceHint API;
  stale NaN/kurtosis/Bootstrap assertions corrected.

### Fixed (continued)

**Quantile inversion bugs** — found by the new `quantile_accuracy` tool; all pre-existing:
- **`erf_inv` Moro central-region variable substitution** (`math_utils.cpp`): the polynomial was
  evaluated at `z = a²` instead of the correct `z = (a/2)²`, producing an initial estimate of ~2.8
  instead of ~0.48 for `a ≈ 0.5`. This caused Halley's method to diverge over 48 consecutive grid
  points (p ∈ [0.204, 0.252]), propagating incorrect quantiles to Gaussian, LogNormal, and StudentT.
  Fix: use `y = a/2` and multiply by `1/√2` to convert Φ⁻¹ to erf_inv.
- **`GammaDistribution::computeQuantile` Wilson-Hilferty failure** (`gamma.cpp`): for α > 1 at
  very small p the Wilson-Hilferty cube is negative. Clamping to NEWTON_RAPHSON_TOLERANCE (1e-10)
  caused the first Newton step to produce x ≈ 5×10⁶, after which the PDF underflowed and the
  solver exited with the wrong answer (34 failures for Gamma(α=2), propagating to ChiSquared(k=4)).
  Fix: detect negative WH result and substitute the small-p asymptotic
  `x ≈ (p · Γ(α+1))^(1/α) / β`; added bisection fallback when PDF underflows mid-iteration.
- **`inverse_beta_i` Newton oscillation for small/large p** (`math_utils.cpp`): the normal
  approximation initial estimate goes negative (or very small positive) for p near the tails of
  Beta(a,b) when the mean is far from 0. After clamping to 1e-8, Newton-Raphson oscillated
  between the two boundary sentinels (1e-10 and 1−1e-10) and never converged (60 failures for
  Beta(2,3)). Fix: when normal approximation ≤ 0 use the tail asymptotic
  `x ≈ (p·a·B(a,b))^(1/a)`; for p < 0.1 take `max(normal, asymptotic)` to prevent starting below
  the reliable estimate even when the normal approximation is small-but-positive.

**Tools audit fixes:**
- `cpp20_features_inspector`: execution-policy detection tested `std::execution::seq` (a
  compile-time constant that cannot throw) inside a `try/catch`, then set all three of `has_par`,
  `has_par_unseq`, `has_unseq` to `true` simultaneously — actively misreporting `par` and
  `par_unseq` as available on platforms without TBB. Replaced with per-policy `#ifdef` checks
  using `__cpp_lib_parallel_algorithm` and `__cpp_lib_execution`.
- `parallel_batch_fitting_benchmark`: `--threads N` flag was parsed and stored in `config.num_threads`
  but never forwarded to `parallelBatchFit()` (which takes no thread-count argument). Flag removed.
- `parallel_correctness_verification`: all six `Dist::create()` no-argument calls replaced with
  explicit documented parameters via a `createTestDistribution<Dist>()` helper.
- `simd_verification`: `[[maybe_unused]] LARGE_TEST_SIZE` stub and `prim_scalar`/`prim_simd`
  accumulators (accumulated but never read) removed. String-find tolerance dispatch
  (`find("Gaussian")` etc.) replaced with an exact-match `kToleranceTable[]`.
- `strategy_profile`: `--large` help text said "1M and 2M" but adds four sizes (750K, 1M, 1.5M, 2M).
- `system_inspector`: 3 back-to-back duplicate `using namespace stats::detail` declarations removed.

**Test suite audit fixes:**
- Six basic test files (Gaussian, Exponential, Gamma, Discrete, Poisson, Uniform) had empty
  `start/end` timing pairs for the "traditional" baseline in Test 6, leaving `*_traditional`
  vectors as zero-initialized. The correctness check comparing batch output to zeros always
  printed `❌ Large batch auto-dispatch results differ from traditional methods`.
  All six fixed with proper scalar loops.
- `test_lognormal_enhanced.cpp` `VectorizedSpeedup` was a complete no-op: `out` and `scalar_out`
  were never populated, so the `ASSERT_NEAR` compared 0.0 vs 0.0 for all 50,000 elements.
  Fixed with actual FORCE_VECTORIZED and FORCE_SCALAR batch calls.
- `TestInfrastructure` class in `tests.h` declared 7 static methods with no `.cpp` implementation,
  forming a link-error trap if `QUICK_BENCHMARK_COMPARE` macro was ever expanded. Class and macro
  removed. Dead `test_common.h` (no file included it) deleted.
- Three conflicting `approxEqual` implementations with different semantics removed from
  `fixtures.h`; canonical `nearly_equal(a, b, tol=1e-10)` with relative-error formulation added
  to `constants.h`.
- 10 newer-style basic test files had a VECTORIZED-vs-SCALAR correctness check that always passed
  trivially (large_vec / scl_out were never populated). Fixed by the `runBatchTests` template.

**Tools added to all 16 distributions** (were previously limited to 6):
- `parallel_batch_fitting_benchmark`: LogNormal, Pareto, Weibull, Rayleigh, VonMises, Binomial,
  NegBinomial, Beta, ChiSquared, StudentT added with sample generators using `sample()`.
- `parallel_correctness_verification`: same 10 distributions added with explicit `create()` params
  and domain-appropriate input ranges for each distribution type.

### Added

**New distributions (Geometric, Laplace, Cauchy — 2026-06-28):**
- **GeometricDistribution** — Geo(p), discrete, support {0,1,2,...}; delegates to
  NegativeBinomial(r=1, p). PMF(k)=p(1-p)^k; closed-form quantile; MLE: p̂=1/(1+x̄).
- **LaplaceDistribution** — Laplace(μ, b), continuous, standalone (no delegation).
  PDF=(1/2b)exp(-|x-μ|/b); four-step SIMD pipeline (scalar_add→fabs→scalar_multiply→vector_exp);
  MLE: μ̂=median (O(n log n)), b̂=MAD; closed-form quantile and all moments.
- **CauchyDistribution** — Cauchy(x₀, γ), continuous; delegates to StudentT(ν=1) via
  z=(x-x₀)/γ transform. getMean/getVariance/getSkewness/getKurtosis return NaN (moments undefined);
  getMedian()=getMode()=x₀; getEntropy()=log(4πγ); closed-form quantile=x₀+γ·tan(π(p-0.5));
  MLE: median/IQR seed + 20 Fisher-scoring iterations on the Cauchy score equations.
- All three are fully integrated: 5 tools, integration workflow NaN test, 47/47 correctness tests pass.
- Commits: 64eeedc (Geometric), 04e187e (Laplace), 662e4a3 (Cauchy).

**Test infrastructure:**
- `tests/include/basic_test_runner.h`: `BasicDistConfig` struct and `runBatchTests<Dist>()` /
  `runErrorTests()` template functions. Tests 6 and 8 in all 16 basic test files now use the
  shared template, eliminating ~1500 lines of copy-paste and ensuring the large-batch correctness
  check is always a real scalar-vs-batch comparison.
- `tests/include/enhanced_test_suite.h`: `DistTraitsDefaults`, `DistTraits<T>` specialisation
  template, and `DistributionEnhancedTest<T>` typed fixture with four `TYPED_TEST_P` patterns:
  `LogPDFConsistency`, `BatchMatchesScalar`, `QuantileRoundTrip` (continuous only, skipped for
  discrete via `GTEST_SKIP()`), and `InvalidParameters`. All 16 enhanced test files have
  `DistTraits<T>` specialisations and `INSTANTIATE_TYPED_TEST_SUITE_P` calls, adding 64 additional
  typed test cases that run within the existing 44 CTest targets.

**New tools** (`tools/`):
- `quantile_accuracy`: verifies `CDF(getQuantile(p)) ≈ p` across a 1000-point grid from 0.001 to
  0.999 plus near-boundary p ∈ {1e-3, 1e-4, 1e-5, 1e-6} for all 16 distributions. Discrete
  distributions use the floor-property check `CDF(Q(p)) ≥ p`. Analogous to `simd_verification`
  for PDF/LogPDF/CDF. First run immediately identified the three quantile bugs above.
- `parameter_recovery_benchmark`: runs `fit()` at n ∈ {25, 50, 100, 250, 500, 1000, 2500} with
  M=100 replicates for 10 representative distributions, reporting `getMean()` / `getVariance()`
  bias and RMSE vs. true values. Answers: "what sample size gives stable MLE estimates?"
  Supports `--quick` for a fast 3-size preview.
- `threshold_validator`: reads a `strategy_profile` CSV and compares measured S→V crossovers
  against compiled `parallelThresholdFromTable()` values. Reports MATCH / UPDATE↑ / UPDATE↓ /
  ADD THRESH / SET NEVER? per (distribution, operation) row. Closes the profiler→header
  recalibration feedback loop.

**New and updated examples** (`examples/`):
- `logpdf_and_likelihood_demo.cpp` (new): first example demonstrating `getLogProbability()` as an
  actual API call. Covers scalar and batch LogPDF, log-likelihood computation, underflow avoidance
  for n > ~150, model comparison, and fit-and-score anomaly scoring.
- `distribution_families_demo.cpp`: expanded from 9 to all 16 distributions. LogNormal, Weibull,
  Pareto, Rayleigh added to Family 2 (positive-support); Binomial and NegativeBinomial added to
  Family 4 (discrete); VonMises added as Family 5 (circular). Footer corrected from the wrong
  "14 distributions across 6 families" to "16 distributions across 5 families".
- `basic_usage.cpp`: added `getEntropy()` call to the properties section; added Section 9
  "End-to-End: Fit → Validate → Score" demonstrating the most common real-world workflow.
- `performance_learning_demo.cpp` renamed to `performance_dispatch_demo.cpp`: removed inaccurate
  "adaptive learning from execution history" header claim. Added `demonstrate_forced_strategies()`
  showing FORCE_SCALAR / FORCE_VECTORIZED / FORCE_PARALLEL with correctness verification.
  Expanded dispatcher demo to show per-distribution threshold differences (Gaussian PDF vs.
  Exponential PDF vs. Binomial CDF) across the same batch sizes.
- `parallel_execution_demo.cpp`: added `demonstrate_parallel_batch_fit()` with sequential vs.
  parallel comparison on 50 Gaussian and 50 Exponential datasets. Previously the file never called
  any distribution batch operation despite its name.

### Changed (continued)

- **v2 architecture audit cleanup** (`DistributionBase::validate()`): KS/AD tests now skip for
  discrete distributions (returning NaN stats + recommendation to use `chiSquaredGoodnessOfFitTest`).
  `DistributionInterface` gains `getMedian()` with NaN default; 14 concrete `getMedian()` overrides
  added. Orphaned Section 14 Doxygen blocks removed from all 16 distribution headers. Catalina
  concept-syntax fallback (`LIBSTATS_NEEDS_CATALINA_CONCEPT_SYNTAX_FALLBACK`) removed from
  `utility_common.h` and `math_utils.h`. `constants_bridge.h` deleted (zero includes). Prominent
  `NOT YET WIRED` warning added to `PerformanceHint::thread_count` documentation.
- **AGENTS.md**: `Creating New Distributions` checklist expanded from 5 vague steps to 6 precise
  steps documenting every required element: mandatory header members (`kDistributionType`,
  `kIsDiscrete`, noexcept moves, `parallelBatchFit()`), all pure-virtual overrides, the new test
  infrastructure (`basic_test_runner.h`, `enhanced_test_suite.h`, `DistTraits<T>`), all four
  CMakeLists.txt registration locations, the `using Dist = DistDistribution` type alias in
  `libstats.h`, and post-implementation dispatch profiling via `strategy_profile` + `threshold_validator`.
  Test count corrected from stale "23" to "44 CTest targets".

### Validation

- 44/44 correctness tests pass on Kaby Lake AVX2+FMA and Mac Mini M1 NEON.
  Asus TUF A16 (AVX-512): pending re-run of correctness suite (2 new test files added after
  Asus validation; kAvx512 threshold values unchanged so no regressions expected).
- `quantile_accuracy` tool: 26/26 test cases PASS across all 16 distributions (0 FAIL);
  three pre-existing quantile inversion bugs fixed as a result of first tool run.

### Fixed (audit remediation)

**Thread safety — TOCTOU (time-of-check to time-of-use) in cache locking:**
- All 19 distributions' scalar probability methods (`getProbability`, `getLogProbability`,
  `getCumulativeProbability`, `getMean`, `getVariance`, `sample`, etc.) used a double-checked
  locking pattern with a gap between `ulock.unlock()` and `lock.lock()` where a concurrent
  setter could change parameters and invalidate the cache. All scalar methods now snapshot
  cached fields under the still-held unique_lock (early-return pattern) eliminating the gap.
- All 9 Gaussian batch lambdas (PDF/LogPDF/CDF × SIMD/Parallel/WorkStealing) and the
  Binomial parallel batch lambda had the same gap; converted to scope-block snapshot pattern.

**M-6 (const_cast fully resolved):**
- `updateCacheUnsafe()` is `const noexcept`; `const_cast` in batch lambdas was always
  unnecessary. ~111 instances across `gaussian.cpp` (original P1-8 fix caught 3; 108 more
  in 14 other distribution batch lambdas were introduced by TOCTOU agents) all removed.

**5 no-op tests fixed** (audit found 4; a 6th was found and fixed):
- `BinomialEnhancedTest::VectorizedEqualsScalar` and `::ParallelBatchCorrectness`,
  `NegativeBinomialEnhancedTest::VectorizedEqualsScalar` and `::ParallelBatchCorrectness`,
  `VonMisesEnhancedTest::ParallelBatchCorrectness` — output vectors were never populated;
  tests compared zero-vs-zero trivially. All now call real FORCE_VECTORIZED/FORCE_PARALLEL
  dispatch paths with correctness assertions.

**Numerical correctness:**
- `LogSpaceOps::precomputeLogMatrix`: SIMD path called `vector_log` (returns NaN for negative
  inputs); scalar path called `safeLog` (returns −∞). Post-process NaN→−∞ added so both paths
  agree for zero-probability transitions in HMM-style log-space arithmetic.
- `inverse_chi_squared_cdf` bisection upper bound `df + 10√df` too small for extreme p (> 0.9999).
  Iterative doubling loop added to both bisection sites.
- `VonMisesDistribution::getCumulativeProbability`: fixed 512-step trapezoidal rule losing accuracy
  at high κ. For κ > 50 uses normal approximation CDF(v) ≈ Φ((v−μ)√κ); moderate κ scales step
  count with max(512, 64√κ).
- `openmp_reduce` hardcoded `reduction(+:result)` so non-additive `BinaryOp` arguments (multiply,
  min, max) silently produced wrong answers. Replaced with manual per-thread partials.
- `ParetoDistribution::getQuantile`: pre-lock early exit for `p == 0.0` read `scale_` without
  holding any lock. Shared lock now acquired before member access.

### Added (audit remediation)

**`Result<T>` redesigned as discriminated union:**
- `Result<T>` changed from aggregate struct to `class` backed by `std::variant<T, ErrorInfo>`.
  `makeError()` no longer constructs `T` at all — only `ErrorInfo` (code + message) is stored
  on the error path. Equivalent to C++23 `std::expected<T, E>` in C++20.
- Public fields removed; method API: `*result` / `result.unwrap()` (value), `result.errorCode()`,
  `result.message()`. `unwrap()` has lvalue/const-lvalue/rvalue overloads. See MIGRATION_GUIDE.md.

**New entropy implementations:**
- `DiscreteDistribution::getEntropy()`: exact H = log(n) nats where n = b − a + 1.
- `PoissonDistribution::getEntropy()`: exact PMF summation for λ ≤ 100; Stirling asymptotic
  H ≈ ½ log(2πeλ) − 1/(12λ) for λ > 100.

**New tools:**
- `tools/toctou_validator`: concurrent writer/reader race stress test for all 19 distributions.
  Detects HARD violations (NaN/negative/CDF-out-of-bounds) and MIXED violations (PDF value
  inconsistent with all known-valid parameter sets). Exit code 0 = clean, 1 = violations.
- `tools/copy_move_stress`: concurrent copy/move correctness and throughput tool replacing the
  retired `test_copy_move_stress` (which had a hardcoded 5-second timer, no GTest assertions,
  and only covered 16 of 19 distributions). Configurable `--duration-ms`/`--threads`.
  All 19 distributions. Result column is primary; copy rate is diagnostic.

**Other improvements:**
- `LIBSTATS_PREFER_PLATFORM_THREADING` CMake option: when ON with GCD (macOS) or WTP (Windows)
  active, suppresses OpenMP to prevent two independent thread pools over-subscribing the CPU.
- All 19 `getDistributionName()` return values normalised to PascalCase display names from
  `kDistributionMeta` (14 returned `XxxDistribution`, 1 returned `DiscreteUniform`).
- `LogSpaceOps::initialize()` no-op removed along with `LogSpaceInitializer` RAII class and
  `globalLogSpaceInit` static (lookup table it initialised was removed years earlier).
- `LIBSTATS_LIKELY`/`LIBSTATS_UNLIKELY` macros replaced with direct C++20 `[[likely]]`/
  `[[unlikely]]` attributes in `math_utils.h`; macro definitions removed from `utility_common.h`.
- Vestigial `include/common/distribution_platform_common.h` deleted; all 19 distribution headers
  updated to remove the include.
- Doxyfile.in: MathJax 3 CDN enabled, `WARN_NO_PARAMDOC=YES`, `DOT_NUM_THREADS=4`.
- `forward_declarations.h`: Geometric/Laplace/Cauchy forward declarations and type aliases added;
  stale `LibDistributionType` tombstone comment removed.

### Validation

- 46/46 correctness tests pass on Kaby Lake AVX2+FMA and Mac Mini M1 NEON.
  (Test count reduced from 47: `test_copy_move_stress` retired; `test_copy_move_fix` extended
  with compile-time `static_assert` noexcept guards for all 19 distributions.)
- `toctou_validator`: 1,416,531 concurrent reads, 0 violations across all 19 distributions.
- Asus TUF A16 (AVX-512): re-validation required after audit remediation.

---

## [1.5.3_1] - 2026-06-30

### Fixed
- **Linux/TBB transitive linkage hotfix**: `libstats_static` and `libstats_shared`
  now link TBB publicly when `LIBSTATS_HAS_TBB` is enabled. GCC/libstdc++ parallel
  STL backends can route `std::execution` algorithms through TBB, so consumers that
  call parallel algorithms need the TBB dependency propagated transitively. Modern
  CMake `TBB::tbb` is preferred; pkg-config libraries remain a fallback.
- Removed the broken test-only `${TBB_LIBRARIES}` link path. Tests now inherit TBB
  through `libstats_static`, matching downstream consumers.

### Validation
- v1.5.3 smoke build: `cmake -S . -B build-v153-smoke -DLIBSTATS_BUILD_TESTS=OFF
  -DLIBSTATS_BUILD_TOOLS=OFF -DLIBSTATS_BUILD_EXAMPLES=OFF` and
  `cmake --build build-v153-smoke --target libstats_static --parallel`.

---

## [1.5.3] - 2026-06-20

### Fixed

- **BF-1**: Remove incorrect `noexcept` specifier from `UniformDistribution` move constructor; the `@warning` comment directly contradicted the specifier, risking `std::terminate` on lock acquisition failure.
- **BF-2**: Version constants frozen at v1.2.0 corrected to v1.5.3; `CMakeLists.txt` project `VERSION` bumped to 1.5.3.
- **BF-3**: Type aliases (`Gaussian`, `Normal`, `Exponential`, …) moved inside the `LIBSTATS_FULL_INTERFACE` guard; using them on incomplete types caused confusing linker errors on first use.
- **BF-4**: `libstats-config.cmake.in` target names corrected (`libstats::static` → `libstats::libstats_static`) to match CMake-exported names at install time.

### Changed

- **Deprecation sweep**: Added `[[deprecated(...)]]` to all 48 `*WithStrategy` batch methods (16 distributions), `getBatchProbabilities`/`LogProbabilities`/`CumulativeProbabilities`/`Quantiles`, `getKLDivergence`, `MemoryPool`, `SmallVector`, `StackAllocator`, `simd_vector`, and `refineWithCapabilities`. These APIs will be removed in v2.0.0.
- **VonMises**: Added `getMedian()` (= `getMu()`; symmetric distribution). Replaced stale `vector_cos` absence comments with accurate description of the 4-step SIMD pipeline (added in v1.4.0). Renamed `*BatchImpl` → `*BatchUnsafeImpl` for naming convention alignment.
- **CI**: Fixed swallowed AVX-512 compilation failures; updated `clang-format`/`clang-tidy` from version 15 to 17.

### Validation

- 39/39 correctness tests pass on Kaby Lake AVX2+FMA.

---

## [1.5.2] - 2026-06-19

### Fixed
- **C-1 (`gammaQ` infinite recursion — critical)**: `gammaQ` was implemented as
  `1 - gammaP(a, x)`, causing infinite mutual recursion for `x >= a + 1` and
  terminating via `std::terminate` (both functions are `noexcept`). Replaced with a
  standalone Legendre continued-fraction expansion (Lentz's algorithm). Every
  distribution whose CDF uses the regularised incomplete gamma function — Gamma,
  Chi-Squared, and Poisson — was affected for moderate-to-large observations.
  Added `gammaQuantile(a, p)` (bisection on `gammaP`) as a protected static helper.
- **C-2 (`bayesianCredibleInterval` — critical)**: the `credibility_level` parameter
  was ignored; the function always returned bounds computed with a hardcoded `z = 1.5`.
  Fixed to compute an equal-tailed interval from the exact Gamma posterior via
  `gammaQuantile / post_rate`.
- **N-1 (`safe_log(+inf)`)**: returned `DBL_MAX` instead of `+inf`, misleading
  downstream `std::isinf()` checks. Now returns `+inf`.
- **N-2 (`safe_exp` underflow)**: returned `MIN_PROBABILITY` (1e-300) instead of
  `0.0` for inputs that underflow. IEEE 754 underflow to zero is correct, not an
  error. Both the early guard and the post-`exp` check now return `0.0`.
- **N-3 (`betaI_continued_fraction`)**: the FPMIN guard on `c` fired after
  `aa / c` was already computed, allowing a near-zero `c` from a prior iteration
  to produce `aa / 1e-30 ≈ 1e+30`. Guard now fires before the division.
- **T-1 (`recordPerformance` race)**: `min_time_ns` / `max_time_ns` were updated
  with separate load and store operations on `std::atomic<uint64_t>`, allowing two
  racing threads to silently overwrite each other's result. Replaced with
  compare-exchange loops.
- **T-2 (`WorkStealingPool` destructor)**: destructor set `shutdown_ = true` before
  draining queued tasks. In-flight tasks were abandoned and any `std::future` for
  a dropped task blocked forever. Destructor now calls `waitForAll()` first.
- **T-4 (`GaussianDistribution` copy constructor)**: `DistributionBase(other)` ran
  in the initializer list without holding `other.cache_mutex_`, leaving a window
  where a concurrent `fit()` or `invalidateCache()` could race with the base-class
  copy. Now default-constructs the base class and copies all fields under
  `shared_lock(other.cache_mutex_)`.
- **T-5 (`GammaDistribution` copy constructor)**: acquired `unique_lock` on the
  source for a read-only operation. Changed to `shared_lock`.
- **Q-2 (Newton-Raphson derivative guard)**: replaced `break` with `return x` when
  the derivative is near zero, making the hard-stop intent explicit and avoiding
  the useless remaining iterations.

### Changed
- **Q-1 (`ConvergenceDetector`)**: `history_` changed from `std::vector<double>` to
  `std::deque<double>`; `erase(begin())` replaced with `O(1)` `pop_front()`.
- **A-4 (`RecoveryStrategy`)**: removed the `#undef STRICT/GRACEFUL/ROBUST/ADAPTIVE`
  block (enum values are already renamed to `StrictMode` etc.; no macro names
  conflict). Converted from unscoped to `enum class`.
- **A-2 (`result_of_t`)**: consolidated three identical definitions into a single
  canonical `include/platform/internal/type_traits.h`.
- **A-3 (`parallelTransform`)**: replaced `check_finite(static_cast<double>(size))`
  with a `size == 0` early-return. `size_t` is always finite; the cast was
  semantically wrong and lost precision above 2⁵³.
- **Q-3 (WorkStealingPool startup timeout)**: replaced reuse of
  `MAX_DATA_POINTS_FOR_SW_TEST` (a Shapiro-Wilk sample-size bound) with a named
  `THREAD_STARTUP_TIMEOUT_MS = 5000` local constexpr.
- **A-1 (`LogSpaceOps::initialize`)**: guarded with `std::call_once` to make
  repeated calls from multiple translation units idempotent.
- **F2 (namespace pollution)**: removed dead `using std::shared_lock/shared_mutex/
  unique_lock` from `distribution_common.h`. All distribution `.cpp` files already
  use `std::` qualification.
- **F3 (`Thresholds` struct)**: distribution-specific threshold fields marked
  `[[deprecated]]`; doc comment added pointing to `dispatch_thresholds.h` as the
  authoritative source. Fields are not read by `selectStrategy()`.
- **F4 (`LibDistributionType`)**: removed unused enum from `forward_declarations.h`.
  It was a duplicate of `detail::DistributionType` with no callers.
- **F5 (`PerformanceDispatcher::SIMDArchitecture`)**: enum and related methods
  `detectSIMDArchitecture` / `createForArchitecture` marked `[[deprecated]]`. Both
  already delegate to `SIMDPolicy::Level`; the local enum is legacy scaffolding.

### Documentation
- `CachedProperty<T>`: explicit `@warning Not thread-safe` Doxygen annotation.
  Use only within `ThreadSafeCacheManager` locks.
- `ThreadSafeCacheManager`: documents the intentional dual-flag pattern
  (`cache_valid_` + `cacheValidAtomic_`) and the v2.0.0 consolidation plan.

### Validation
- 39/39 correctness tests pass on Kaby Lake AVX2+FMA (primary dev machine).
  Multi-machine validation pending PR merge.

---

## [1.5.1] - 2026-06-16

### Fixed
- `#include <pair>` → `#include <utility>` in `statistical_utilities.h` (no standard header `<pair>`)
- `shouldUseSIMDBatch()` now delegates to `SIMDPolicy::shouldUseSIMD()` instead of hardcoded threshold 32
- Removed dead `withCachedParameters` template from `dispatch_utils.h` (no callers; contained an unsafe lock-free window)
- Removed stale commented-out `DistributionCacheAdapter` block from `distribution_base.h`
- Formalized `arch::simd` compatibility aliases in `simd.h` — replaced `using namespace` with explicit declarations; removed "temporary" labels from de-facto permanent API surface
- Added `/utf-8` MSVC compile flag to silence C4566 for Unicode literals in tool sources
- `simd_verification`: `verifyVectorOp` now reports **relative** error for VectorExp and VectorLog
  (`max_rel=`) rather than absolute diff, which was meaningless at output magnitudes like
  `exp(500)~5e+217` (a sub-ULP result appeared as `max_diff=3.0e+197`). Pass/fail logic unchanged.

### Performance
- Dispatch table expanded to all 16 distributions: `kNeon`, `kAvx`, `kAvx2`, and `kAvx512` now
  include calibrated entries for LogNormal, Pareto, Weibull, Rayleigh, VonMises, Binomial, and
  NegativeBinomial. Previously these distributions fell through to `NEVER`. Thresholds derived from
  two Release-mode profiling bundles per architecture captured on `fix/audit-remediation`.

### Tests
- `test_performance_dispatcher`: large-batch strategy assertion is now threshold-aware, so it stays
  valid after table recalibrations; `minimal_latency()` thread-count expectation corrected
  (`std::nullopt`, not `1`)
- `test_von_mises_enhanced`: tolerances relaxed to `1e-10` to match `vector_cos` AVX-512 absolute
  error floor (~6e-11)

### Validation
- 39/39 correctness, 61/61 `simd_verification` on Kaby Lake AVX2+FMA, Mac Mini M1 NEON, Asus TUF A16 AVX-512
- 38/38 correctness, 61/61 `simd_verification` on Ivy Bridge AVX (macOS Catalina; `test_work_stealing_pool` skipped)

### Deprecation
- **v1.5.x are the last releases validated on macOS 10.15 Catalina and Ivy Bridge hardware.**
  v2.0.0 will set the minimum macOS to 13 Ventura.

---

## [1.5.0] - 2026-06-15

### Performance
- **AVX2+FMA native exp/log** (Phase 1, Kaby Lake): `vector_exp_avx2` and `vector_log_avx2`
  replace AVX-delegation stubs with SLEEF-inspired FMA Horner polynomial (< 1 ULP).
  `vector_cos_avx2` replaces AVX delegation with FMA Horner. Kaby Lake primitive speedups:
  VectorExp 3.4x, VectorLog 1.7x, VectorCos 4.9x.
- **High-accuracy `vector_erf`** (Phase 2, all x86 backends): replaced Abramowitz & Stegun
  7.1.26 (~1.5×10⁻⁷ max error) with musl libc four-region rational polynomial (< 1 ULP;
  measured max error ~2.2×10⁻¹⁶). `vector_erf_avx` uses mul+add (−mavx only);
  `vector_erf_avx2` uses `_mm256_fmadd_pd`; `vector_erf_sse2` uses `__m128d`.
  Gaussian CDF SIMD error: 6.97×10⁻⁸ → ~0. Kaby Lake VectorErf: 2.5x.
- **AVX-512 native transcendentals** (Phase 4, Asus TUF A16 Zen 4): `vector_exp_avx512`,
  `vector_log_avx512`, and `vector_erf_avx512` replace AVX 4-wide delegation with native
  `__m512d` 8-wide polynomial implementations. No SVML or MKL dependency.
  - `vector_exp_avx512`: SLEEF range-reduction + 10-term FMA Horner; 2^n scaling via
    `_mm512_cvtpd_epi32` → `_mm512_cvtepi32_epi64` → shift (AVX-512F only, no DQ required).
    Speedup: 1.9x → 5.0x.
  - `vector_log_avx512`: SLEEF xlog_u1 2·atanh series + FMA Horner; exponent extraction
    via `_mm512_srli_epi64` (AVX-512F); `_mm512_cvtepi64_pd` for int64→double (AVX-512DQ,
    present on Zen 4). Speedup: 1.8x → 3.9x.
  - `vector_erf_avx512`: musl four-region rational polynomial + FMA Horner; `__mmask8`
    blends; sign via `_mm512_and/or_pd` (AVX-512DQ); calls `vector_exp_avx512` for erfc.
    Speedup: 0.7x → 1.3x.
  - Asus TUF A16 distribution geomeans: PDF 4.8x, LogPDF 5.1x, CDF 2.2x.
- **NEON native transcendentals** (Phase 3, Mac Mini M1): `vector_exp_neon`
  (SLEEF FMA Horner, < 1 ULP), `vector_log_neon` (SLEEF atanh series, < 1 ULP),
  `vector_erf_neon` (ARM glibc `erf_advsimd` algorithm: 769-entry table of
  `{erf(k/128), 2/sqrt(π)·exp(-(k/128)²)}` + 5-term Taylor correction, ~2.29 ULP).
  `vector_erf_neon` deliberately diverges from the musl rational polynomial used by
  all x86 backends: the table approach eliminates the recursive exp call, yielding
  8.0x vs 0.9x for the pure-polynomial version. Gaussian CDF: 2.2x → 13.9x.
  Key implementation note: derive `shift_u` via `vreinterpretq_u64_f64(shift)` —
  hardcoding `0x4168000000000000` (which is 12582912.0, not bits(2⁴⁵)) causes segfaults.
  All three use `vfmaq_f64` FMA Horner; exp/log also use `vcvtq_s64_f64` for
  direct int64↔double conversion (no store/reload needed on aarch64).
  M1 distribution geomeans: PDF 5.9x, LogPDF 7.3x, CDF 3.1x.
  Primitive ops: VectorExp 2.1x, VectorLog 1.8x, VectorErf 8.0x, VectorCos 3.0x.
  Table generator: `scripts/gen_neon_erf_table.py`. See Issue #33 for a proposed
  cross-architecture experiment (table+Taylor vs SLEEF polynomial on AVX2/AVX-512).

### Tests
- `simd_verification` coverage: added `testVonMisesDistribution()` and
  `testPrimitiveVectorOps()` (VectorExp/Log/Erf/Cos); 54 → 61 tests.
- `simd_verification` reporting: replaced single composite speedup with per-op-type
  geometric means (PDF/LogPDF/CDF) and per-primitive individual rows.
- Erf accuracy regression: 10,000 random inputs in [−4, 4] assert
  `|vector_erf(x) − std::erf(x)| ≤ 1×10⁻¹²`. `TOLERANCE_ERF_APPROX` tightened from
  `~1.5×10⁻⁷` to `1×10⁻¹³`.

### Data
- Asus TUF A16 dispatcher profiling bundle (post-Phase 4):
  `data/profiles/dispatcher/2026-06-14T20-36-11Z_windows-x86_64_v1.5-avx512-transcendentals_sha-14bf1ba/`
- Mac Mini M1 dispatcher profiling bundle (post-Phase 3):
  `data/profiles/dispatcher/2026-06-14T23-29-58Z_darwin-arm64_v1.5-neon-transcendentals_sha-5455778/`
  Notable finding: Gaussian CDF VECTORIZED wins at all batch sizes up to 500k (no
  crossover to PARALLEL), a direct result of the table-based erf achieving 13.9x
  over scalar. Phase 5 NEON threshold recalibration should reflect this.
- `include/core/dispatch_thresholds.h`: `kAvx512` annotated as stale (calibrated pre-Phase 4);
  Phase 5 will rederive from the Phase 4 bundle. Measurement resolution caveat added:
  profiler resolution floors at ~0.1–0.2 µs; crossovers derived from batch sizes < 64 are
  unreliable and should be clamped to ≥ 64 when recalibrating.

### Changed
- **Dispatch threshold recalibration** (Phase 5): all four `ArchTable` entries in
  `include/core/dispatch_thresholds.h` updated from v1.5.0 profiling bundles.
  Notable architectural shifts:
  - NEON: Gaussian CDF threshold 10000 → NEVER (table erf 8.0x makes VECTORIZED
    unbeatable at all tested batch sizes; M1 GCD overhead proportionally high).
  - AVX2+FMA: Gaussian PDF 50000 → 100000; StudentT PDF 100000 → 250000
    (FMA exp widened the SIMD competitive window).
  - AVX-512: Exponential PDF 50000 → NEVER; StudentT PDF/LogPDF → NEVER
    (8-wide native exp eliminated the throughput bottleneck); Gaussian PDF 100000 → 500000.
  - AVX: Gaussian CDF 20000 → 50000 (heavier musl erf per element; SIMD competitive longer).
  < 64 floor rule applied throughout; sub-64 crossovers from GCD-overhead noise clamped.

### Validation
- 39/39 correctness, 61/61 `simd_verification` — Kaby Lake AVX2+FMA:
  PDF 8.0x, LogPDF 9.6x, CDF 3.3x; VectorExp 3.4x, VectorLog 1.7x, VectorErf 2.5x, VectorCos 4.9x.
- 39/39 correctness, 61/61 `simd_verification` — Asus TUF A16 AVX-512:
  PDF 4.8x, LogPDF 5.1x, CDF 2.2x; VectorExp 5.0x, VectorLog 3.9x, VectorErf 1.3x, VectorCos 8.5x.
- 39/39 correctness, 61/61 `simd_verification` — Mac Mini M1 NEON:
  PDF 5.9x, LogPDF 7.3x, CDF 3.1x; VectorExp 2.1x, VectorLog 1.8x, VectorErf 8.0x, VectorCos 3.0x.
- 38/38 correctness (1 skipped), 61/61 `simd_verification` — Ivy Bridge AVX (macOS Catalina):
  PDF 5.6x, LogPDF 6.0x, CDF 2.6x; VectorExp 2.2x, VectorLog 1.3x, VectorErf 1.7x, VectorCos 11.0x.

### Deprecation Notices
- **macOS Catalina (10.15) support is deprecated as of v1.5.0** and will be removed in
  a future v2.0.0 release. v1.5.0 is the last version that will be validated on
  macOS 10.15 / Ivy Bridge hardware. Rationale:
  - macOS 10.15 is end-of-life; the upcoming macOS 27 release will cause Homebrew to
    drop Catalina support entirely, eliminating the toolchain maintenance path.
  - The Catalina-specific CMake guards (`CROSS_PLATFORM` build type, Homebrew LLVM
    fallback detection, `LIBSTATS_HAS_REQUIRES_EXPRESSIONS` gating) add complexity
    that is disproportionate to the remaining user base on hardware this old.
  - This aligns with libhmm, which dropped macOS < 13/Ventura support in v4.0.
  - **Minimum macOS from v2.0.0:** macOS 13 Ventura (Apple Clang 14+).

---

## [1.4.0] - 2026-06-14

### Added
- `vector_cos` — vectorised cosine across all five SIMD backends (AVX, AVX2, SSE2,
  NEON, AVX-512). Two-step range reduction (reduce to [−π, π], then to [−π/2, π/2]
  with sign tracking) followed by a 7-term Horner Taylor polynomial; max error ≈
  1×10⁻¹⁰. AVX-512 uses native 8-wide `_mm512_roundscale_pd` + `_mm512_fmadd_pd`
  (no SVML dependency, unlike `vector_exp`/`vector_log`/`vector_erf`). SSE2 uses
  the magic-number rounding trick (`6755399441055744.0 = 2^52+2^51`) to avoid the
  SSE4.1-only `_mm_round_pd`.

### Changed
- **SIMD dispatch table** (closes #22): replaced 11 repeated 5-tier dispatch chains
  in `simd_dispatch.cpp` with a single `VectorOps::DispatchTable` struct populated
  once at startup by `makeDispatchTable()`. Each public method is now a 2-line size
  check + table call. Adding a new SIMD tier or op requires editing one function.
  Net: −385 lines, +157 lines.
- **VonMises LogPDF/PDF batch** now SIMD-accelerated via `vector_cos`: pipeline is
  `scalar_add(−μ)` → `vector_cos` → `scalar_multiply(κ)` → `scalar_add(−ln Z)` →
  scalar fixup for non-finite inputs. PDF reuses LogPDF then calls `vector_exp`.
- `erf_inv` Moro/Acklam coefficients lifted from `static const` to `static constexpr`;
  shared Acklam d/e coefficients hoisted to file-scope `static constexpr`, eliminating
  copy-pasted duplicate blocks (CCN 15 → 14).

### Fixed
- Domain constant pollution (7 call sites across 4 files): `STRONG_CORRELATION`,
  `CONFIDENCE_99`, and `AD_THRESHOLD_1` from `statistical_constants.h` were borrowed
  as numeric coincidences in algorithmic code. Replaced with purpose-specific constants
  in the appropriate headers (`math_constants.h`, `performance_constants.h`,
  `statistical_constants.h`).
- `FeaturesSingleton` Rule of Five: copy/move constructors and assignment operators
  now explicitly `= delete`; default constructor explicitly `= default`.
- Removed redundant `static` from `g_features_manager` (anonymous namespace already
  provides internal linkage).
- Named the two bare magic literals: `ZERO_DENSITY_LOG_PENALTY = -1e10`
  (`validation.cpp`) and `WORK_STEALING_FALLBACK_THRESHOLD = 10000`
  (`performance_history.cpp`).
- `.clang-tidy`: `FunctionCase: camelCase` → `FunctionCase: camelBack` (`camelCase`
  is not a valid clang-tidy identifier-naming token).

### Validation
- 39/39 correctness tests, 54/54 SIMD verification (Kaby Lake AVX2, 3.35x overall).

---

## [1.3.0] - 2026-06-14

### Added
- `LogNormalDistribution` — log-space Gaussian transform; 6-step SIMD LogPDF/PDF pipeline
  (`vector_log` + element-wise `vector_multiply` + `vector_subtract`), 5-step CDF pipeline
  (`vector_log` → `scalar_add` → `scalar_multiply` → `vector_erf` → two scalars).
  Closed-form MLE: μ̂ = mean(log xᵢ), σ̂ = population std(log xᵢ).
- `ParetoDistribution` — power-law/heavy-tail distribution; the simplest SIMD pipeline in
  the library (3-step LogPDF: `vector_log` + 2 scalars; 6-step CDF with no temp buffer).
  Closed-form MLE: x̂_m = min(xᵢ), α̂ = n / Σlog(xᵢ/x̂_m).
- Both distributions: full 24-section template (PDF/LogPDF/CDF/quantile/sampling/MLE/
  `parallelBatchFit`/SIMD batch/auto-dispatch/explicit-strategy/GTest suite/timing labels).
- `LOG_NORMAL` and `PARETO` added to `DistributionType` enum, `DistributionTraits`,
  `forward_declarations.h`, and `libstats.h` type aliases.
- `WeibullDistribution` — reliability/survival analysis; 8-step SIMD LogPDF/PDF (one
  temp buffer) and 8-step CDF (no temp buffer). MLE: method-of-moments seed →
  Newton–Raphson profile score solver (g'(k) = Var_k[log x] + 1/k² > 0, always
  converges). Sampling via `std::weibull_distribution<double>`.
- `RayleighDistribution` — signal-processing magnitudes; standalone implementation
  (not a Weibull delegation) because the quadratic x² structure gives a simpler
  5-step LogPDF pipeline and a 5-step no-temp CDF, vs. Weibull’s 8-step
  log(x/λ) path. Closed-form MLE: σ̂ = √(Σxᵢ²/(2n)).
- `VonMisesDistribution` — circular/directional data; includes `include/core/bessel.h`
  (modified Bessel functions I₀, I₁, log I₀; two-tier: `std::cyl_bessel_i` when
  available, A&S §9.8 polynomial fallback for AppleClang/macOS). LogPDF =
  κ·cos(x−μ) − logNormaliser_; `logNormaliser_` = log(2π) + log I₀(κ) cached.
  VECTORIZED uses a scalar loop (no vector_cos); PARALLEL is recommended for
  large batches. MLE: atan2(S,C) for μ̂; Mardia–Jupp + Newton–Raphson for κ̂.
  Sampling: Best (1979) rejection sampler.
- `WEIBULL`, `RAYLEIGH`, `VON_MISES` added to `DistributionType` enum,
  `DistributionTraits`, `forward_declarations.h`, and `libstats.h` type aliases.
- CMake: `LIBSTATS_HAS_CXX17_BESSEL` capability check (for `std::cyl_bessel_i`).
- `BinomialDistribution` — discrete distribution for success counts in n independent
  Bernoulli trials. PMF via lgamma log-space; CDF = I_{1−p}(n−k, k+1) via
  `detail::beta_i` (O(1)). MLE: n̂ = max(round(xᵢ)), p̂ = k̄/n̂. Sampling via
  `std::binomial_distribution<int>`. VECTORIZED = cached scalar loop (logNFact_,
  logP_, log1mP_ loop-invariants); PARALLEL for large batches.
- `NegativeBinomialDistribution` — discrete over-dispersion model; supports
  real-valued r (dispersion parameter). PMF via lgamma; CDF = I_p(r, k+1) via
  `detail::beta_i` (O(1)). MLE: method-of-moments seed r̂ = k̄²/(s²−k̄) refined
  by Newton–Raphson on the profile score equation using `detail::digamma` and
  `detail::trigamma` (200-iteration, 1e-11·r convergence). Sampling: Gamma(r,
  (1−p)/p)-Poisson mixture — correctly handles real r, unlike
  `std::negative_binomial_distribution<int>`.
- `detail::trigamma` added to `math_utils.h`/`math_utils.cpp`: A&S §6.4.12
  asymptotic series with recurrence shift x ≥ 6; accuracy < 2×10⁻¹⁴.
- `BINOMIAL`, `NEGATIVE_BINOMIAL` added to `DistributionType` enum,
  `DistributionTraits`, `forward_declarations.h`, and `libstats.h` type aliases.
- 2 new GTest suites: `test_binomial_enhanced`, `test_negative_binomial_enhanced`
  (labelled `timing`); 2 correctness suites: `test_binomial_basic`,
  `test_negative_binomial_basic`. Total correctness tests: 39.

### Validation
- 39/39 correctness tests pass on all four architectures (AVX, AVX2, NEON, AVX-512)
- 54/54 SIMD tests pass on all four architectures (unchanged — Binomial/NegBinom use scalar loops)

---

## [1.2.0] - 2026-06-08

### Added
- `detail::batchFitParallel` template helper in `include/core/parallel_batch_fit.h`;
  replaces six hand-rolled `parallelBatchFit` copies across Gaussian, Exponential,
  Uniform, Poisson, Gamma, and Discrete
- `parallelBatchFit` added to the three previously-missing distributions:
  `ChiSquaredDistribution`, `StudentTDistribution`, and `BetaDistribution`
- 11 legacy assert-based tests migrated to GTest `TEST()` suites, integrated into
  CMake via `create_libstats_gtest()` and registered with CTest

### Changed
- `categorizeBatchSize` replaced: if/else ladder → sorted constexpr array +
  `std::lower_bound`, reducing complexity from O(N) to O(log N)
- Four near-identical `getDispatchThreshold` overloads collapsed into one function
  with per-architecture data tables (issues #20, #21)

### Fixed
- `LogSpaceOps::logSumExp` no longer uses the 1024-entry lookup table; replaced
  with `std::log1p(std::exp(diff))` for full double-precision accuracy (~6e-5
  error eliminated)
- `DiscreteDistribution` parameter validation now rejects equal-bounds intervals
  (`a >= b` fails; previously only `a > b` was rejected)

### Documentation
- Removed obsolete files: `VERSIONING.md` (pre-1.0 development), `docs/test_organization.md`
- `PROJECT_CONCEPT.md` rewritten to reflect shipped v1.2.0 library; all Phase/roadmap
  language removed
- `AGENTS.md`, `README.md`, and all `docs/` files updated: Phase references removed,
  WARP.md → AGENTS.md, v0.x version strings removed, parallelBatchFit availability
  on all 9 distributions documented

### Validation
- 54/54 SIMD tests pass on all four architectures (AVX, AVX2, AVX-512, NEON)
- 23/23 correctness tests + 11/11 GTest-migrated tests pass

## [1.1.9](https://github.com/OldCrow/libstats/compare/v1.1.8...v1.1.9) (2026-06-07)

### 🐛 Bug Fixes

* remove imprecise log-space lookup table; reject discrete equal bounds ([c269c79](https://github.com/OldCrow/libstats/commit/c269c79a12a433db11b735ac6401a8b182d70ed8)), closes [#19](https://github.com/OldCrow/libstats/issues/19) [#19](https://github.com/OldCrow/libstats/issues/19)

### 📚 Documentation

* fix switch_branch workflow default branch ([6823d45](https://github.com/OldCrow/libstats/commit/6823d4546e7bf3c6795b71a18475327c423af81e))
* update AGENTS.md and workflows for v1.1.8 state ([34dbe55](https://github.com/OldCrow/libstats/commit/34dbe555bd5865314b0e0dffc81f7cd9115f2dc1))

### ♻️ Refactoring

* collapse 4x CCN-35 dispatch threshold functions into shared impl ([#21](https://github.com/OldCrow/libstats/issues/21)) ([cea6d48](https://github.com/OldCrow/libstats/commit/cea6d48c7016848568ed29e1f98802bba8ac4c16))
* replace categorizeBatchSize() if/else ladder with table + lower_bound ([#20](https://github.com/OldCrow/libstats/issues/20)) ([596d2fc](https://github.com/OldCrow/libstats/commit/596d2fcbd25baf24f614159a5e75af013f59cc6d))

### ✅ Tests

* **quality:** migrate 11 assert-based tests to GTest ([#19](https://github.com/OldCrow/libstats/issues/19)) ([4dac9df](https://github.com/OldCrow/libstats/commit/4dac9df8266491eeb71769ebbbc9b9f0f96dd12a))

## [1.1.8](https://github.com/OldCrow/libstats/compare/v1.1.7...v1.1.8) (2026-05-07)

### 🐛 Bug Fixes

* remove dead GPU dispatch slot from autoDispatch and all distributions ([cc3c1c3](https://github.com/OldCrow/libstats/commit/cc3c1c3bd89f8c1320e93b45fb1344f0673739ae)), closes [#23](https://github.com/OldCrow/libstats/issues/23) [#23](https://github.com/OldCrow/libstats/issues/23)

### ♻️ Refactoring

* **phase2-3:** correctness and static-analysis cleanup ([7a234df](https://github.com/OldCrow/libstats/commit/7a234df677d02f299e8c86d5d4cc69fd171f5be4))

## [1.1.7](https://github.com/OldCrow/libstats/compare/v1.1.6...v1.1.7) (2026-05-06)

### 🐛 Bug Fixes

* audit and correct eight CMake build system issues ([2135ccf](https://github.com/OldCrow/libstats/commit/2135ccf7138f74957952c94a9210656a05d2f3cb))
* correct AVX-512 detection on MSVC and GCC/Clang ([24c355b](https://github.com/OldCrow/libstats/commit/24c355b2b482719fbc5496090339ca50ed14f42a)), closes [#include](https://github.com/OldCrow/libstats/issues/include)
* correct GTest detection on Windows; add FetchContent fallback ([2ee131c](https://github.com/OldCrow/libstats/commit/2ee131c299059fad5bcdea8f3c99a000175c6da4))
* default to AppleClang on macOS; fix Phase 1 correctness bugs ([d50b0f1](https://github.com/OldCrow/libstats/commit/d50b0f1d7cc70a7a7bac7ce436faab1bd414b806))
* label test_math_comprehensive as timing; fix coverage lcov --ignore-errors ([6791b3d](https://github.com/OldCrow/libstats/commit/6791b3d0c821dc83e1ae0047759a270a0bbc8d3a))
* suppress unused-variable warning in test_simd_policy Release build ([8359ca4](https://github.com/OldCrow/libstats/commit/8359ca404bbf76e4e7708a94e2fd472f3161ddaa))

### 📚 Documentation

* add session-start architecture and build routing to WARP ([bbf78b0](https://github.com/OldCrow/libstats/commit/bbf78b04ae743c830a2821df6c8493e3400b1cdb))

### 👷 CI/CD

* add --ignore-errors mismatch to lcov capture ([c9d9654](https://github.com/OldCrow/libstats/commit/c9d9654d682405c98eea9499c2d1d2d902357f8c))
* exclude timing tests from CI; bump Clang matrix from 14/15 to 16/17 ([f5b798a](https://github.com/OldCrow/libstats/commit/f5b798a401668fdd47002198634343eb9cf9aeac))

# Changelog
## [1.1.6] - 2026-04-26

### Fixed
- Catalina fallback path in `src/exponential.cpp` now uses argument-aware
  wrappers so modern toolchains keep `std::ranges::*` calls while older
  libc++ environments safely use iterator-based `std::all_of`/`std::sort`.
- `run_all_tests` in `CMakeLists.txt` now conditionally depends on
  `test_work_stealing_pool` only when that target is available, matching
  `<concepts>/<ranges>` capability gating.
- `run_tests` in `CMakeLists.txt` now escapes regex pipes correctly so CTest
  filter expressions are not interpreted as shell pipelines.
- `tests/test_performance_dispatcher.cpp` edge-case expectation now follows
  `getParallelThreshold(...)` policy instead of hardcoding parallel execution
  for huge `UNIFORM/PDF` batches.
## [1.1.5] - 2026-04-26

### Changed
- Version and release metadata update only.
- No functional code changes from `1.1.4`; this release aligns downstream
  packaging flow for `pylibstats`.
## [1.1.4] - 2026-04-26

### Fixed
- Removed per-file SIMD target pragmas from SIMD implementation translation
  units (`simd_sse2.cpp`, `simd_avx.cpp`, `simd_avx2.cpp`, `simd_avx512.cpp`,
  `simd_neon.cpp`) and now rely on `SIMDDetection.cmake` per-source compile flags
  as the single source of truth, preventing architecture/compiler pragma drift.
- Fixed GCC SIMD pragma scope symmetry in `cpu_detection.cpp` by guarding
  `#pragma GCC pop_options` with the same x86/x64 constraint used for
  `#pragma GCC push_options`, eliminating unmatched pragma behavior on non-x86
  builds.
- Hardened SIMD dependency normalization in `SIMDDetection.cmake` so
  `AVX-512 -> AVX2 -> AVX` source and definition dependencies are enforced in one
  place, including cross-compilation override flows.
## [1.1.3] - 2026-04-26

### Fixed
- `cpu_detection.cpp` now applies x86-only SIMD target pragmas only on x86/x64,
  preventing ARM/aarch64 compile failures from invalid `no-avx*` target attributes.
- SIMD runtime probe kernels now use unaligned stores (`*_storeu_*`) to avoid
  false negatives or crashes caused by stack alignment assumptions.
- SIMD source/definition consistency now forces AVX when AVX2 is enabled, so
  AVX2 transcendental wrappers always resolve AVX helper symbols.
- Added `LIBSTATS_BUILD_EXAMPLES` with an embedded-build-safe default (`OFF`
  when consumed as a subproject), so FetchContent consumers no longer build
  examples unless explicitly requested.
## [1.1.2] - 2026-04-26

### Fixed
- Windows dynamic tests now copy `stats.dll` next to each dynamic test executable
  during post-build, removing runtime loader failures (`0xc0000135`) when DLL
  search paths are not preconfigured.
- Test executable outputs are now separated by build configuration
  (`tests/Debug`, `tests/Release`, etc.), preventing stale Debug/Release binary
  collisions in Visual Studio multi-config builds.
- CTest test registration now uses `$<TARGET_FILE:...>` so each test resolves to
  the correct per-configuration binary path.
## [1.1.1] - 2026-04-26

### Fixed
- Build-tree include shim generation now copies headers into
  `${build}/include_shim/libstats` instead of creating a symbolic link.
  This fixes Windows environments where symlink traversal can fail with
  untrusted mount point behavior during subdirectory/FetchContent builds.

## [1.1.0] - 2026-04-12

### Added
- `constexpr` dispatch threshold lookup table indexed by `(SIMDLevel, DistributionType,
  OperationType)`, derived from 6912 empirical measurements across four architectures
- Profiling bundles for NEON (M1), AVX (Ivy Bridge), AVX2 (Kaby Lake), AVX-512 (Zen 4)
- Canonical `strategy_profile` tool for forced-strategy profiling across all distributions
- `scripts/capture_dispatcher_profile.sh` and `scripts/summarize_dispatcher_profile.py`
  for reproducible profiling workflows
- `beta_i(x, a, b, log_beta_prefix)` overload to hoist lgamma out of batch loops
- `NU_MAX` upper bound in Student-T MLE to prevent Newton-Raphson divergence

### Changed
- CMake global SIMD flag now follows `SIMDDetection` results: `/arch:AVX512` on MSVC
  when AVX-512 is detected, instead of hardcoded `/arch:AVX2`
- Test validators gain AVX-512 awareness: AMD `__AVX512F__` tier, architecture-aware
  SIMD and parallel thresholds reflecting wide-vector crossover characteristics
- Student-T MLE initial estimate clamped to 100; sample size in MLE test increased
  from 500 to 2000 for cross-stdlib convergence stability

### Removed
- `AdaptiveThresholdCalculator` (`src/parallel_thresholds.cpp`,
  `include/platform/parallel_thresholds.h`)
- `DistributionCharacteristics` (`include/core/distribution_characteristics.h`)
- `selectOptimalStrategy` (replaced by `selectStrategy` with `OperationType`)
- Superseded tools: `learning_analyzer`, `parallel_threshold_benchmark`,
  `performance_dispatcher_tool`, `empirical_characteristics_demo`

### Fixed
- AVX-512/MSVC: `__AVX512F__` now defined in all translation units (was only in
  SIMD kernel source files due to hardcoded `/arch:AVX2`)
- `vector<bool>` data race in `SystemCapabilitiesIntegrationTest.ThreadSafety`
  (bit-packing caused concurrent writes to race on the same byte)
- Threading overhead test bound widened from 100μs to 500μs (Windows scheduler jitter)

### Validation
- 45/45 tests pass on all four architectures (NEON, AVX, AVX2, AVX-512)
- 54/54 SIMD correctness tests pass on all architectures
- pylibstats: 168/168 tests, 85/85 SciPy comparison, 8–44× speedups over SciPy

---

## [1.0.0] - 2026-04-11

### Added
- `ChiSquaredDistribution`, `StudentTDistribution`, and `BetaDistribution`
- `detail::digamma(x)` and `detail::inverse_beta_i(p, a, b)` in `math_utils`
- Direct tests and SIMD verification coverage for all three new distributions
- `CHI_SQUARED` added to `DistributionType` enum, `distribution_characteristics`,
  `dispatch_utils`, and all tool distribution lists

### Improved
- Phase 6A SIMD batch paths for Exponential, Gamma, and Uniform
- Dispatch metadata (`getDistributionSpecificParallelThreshold`) now covers all
  9 distributions with explicit cases; `distributionTypeToString` likewise complete
- Documentation and example surface aligned with the current 9-distribution library
- `distribution_families_demo.cpp` added; per-distribution benchmark files removed
  from the examples build
- Explanatory comments added to Gamma, Beta, and Student's t CDF batch
  implementations documenting why scalar special functions cannot be vectorized

### Validation (all four machines, 54/54 SIMD tests)

| Machine | SIMD | Correctness | simd_verification | Speedup |
|---|---|---|---|
| Ivy Bridge (2012 MBP) | AVX | 34/34 ✅ | 54/54 ✅ | 4.10x |
| Kaby Lake (2017 MBP) | AVX2 | 33/33 ✅ | 54/54 ✅ | 3.49x |
| Mac Mini M1 | NEON | 33/33 ✅ | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512/MSVC | 33/33 ✅ | 54/54 ✅ | 1.64x |

---

## Earlier milestones

### Phase 6A — SIMD batch ops for non-Gaussian distributions
Added vectorized `BatchUnsafeImpl` kernels to Exponential (PDF/LogPDF/CDF), Gamma
(PDF/LogPDF), and Uniform (CDF), using the compute+fixup pattern established in
`src/gaussian.cpp`. Speedups on Ivy Bridge AVX: Exponential LogPDF 20.8x,
Exponential PDF/CDF ~10x, Gamma PDF 9.7x, Uniform CDF 25.2x.
Overall `simd_verification` speedup improved from 3.84x to 4.10x.

### Phase 5 — Header optimization and namespace consolidation
Primary namespace changed from `libstats` to `stats`; backward-compatibility alias
`namespace libstats = stats` retained. Header dependency graph cleaned up with
forward-declaration headers and consolidated includes. Compilation overhead reduced.

### Phase 4 — Complete 6-distribution library, SIMD verification
All six core distributions fully implemented with PDF/CDF/quantile/sampling/MLE:
Gaussian, Exponential, Uniform, Poisson, Discrete, Gamma. `simd_verification` tool
validates both correctness and measured speedups. Cross-machine validation completed
on Ivy Bridge (AVX), Kaby Lake (AVX2), M1 (NEON), and Linux CI (AVX2).

### Phase 3 — Performance dispatch infrastructure
`PerformanceDispatcher`, `PerformanceHistory`, and `SystemCapabilities` added.
Architecture-aware parallel thresholds, work-stealing pool, and adaptive strategy
selection via learned performance history.

### Phase 1–2 — Foundation and core infrastructure
Initial library structure: SIMD detection and dispatch (SSE2/AVX/AVX2/NEON),
thread pool, safety utilities, numerical constants, distribution base class, and
fully working `GaussianDistribution` as the reference implementation.
