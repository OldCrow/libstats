# libstats v1 → v2 Migration Guide

This guide covers every breaking change in v2.0.0 and how to update existing code.

## Namespace

**v1.x:** `libstats::` was the primary namespace.
**v2.0.0:** `stats::` is the primary namespace. `libstats::` is retained as an alias
(`namespace libstats = stats;`) for source compatibility, but all new code should use `stats::`.

```cpp
// v1.x
libstats::GaussianDistribution::create(0.0, 1.0);
libstats::initialize_performance_systems();

// v2.0.0
stats::GaussianDistribution::create(0.0, 1.0);
stats::initialize_performance_systems();
```

The alias means existing code compiles without changes, but the primary namespace in
headers, documentation, and examples is now `stats::`.

---

## Platform baseline

| Requirement | v1.x | v2.0.0 |
|---|---|---|
| macOS minimum | 10.15 Catalina | 13 Ventura |
| AppleClang | 13+ | **15+** (Xcode 15+) |
| GCC | 10+ | **13+** |
| Clang | 14+ | **17+** |
| MSVC | 2019 | **2022 17.8+** (19.38+) |

The alternate Homebrew LLVM compiler path is removed in v2.0.0. Use system AppleClang on macOS.

---

## Analysis utilities extracted to `stats::analysis`

All statistical analysis methods that were static members of distribution classes have
moved to the `stats::analysis` namespace. The v1.x methods are gone; there are no
deprecated wrappers.

### Generic tests (all distributions)

| v1.x call | v2.0.0 call |
|---|---|
| `D::kolmogorovSmirnovTest(data, dist, alpha)` | `stats::analysis::kolmogorovSmirnovTest(data, dist, alpha)` |
| `D::andersonDarlingTest(data, dist, alpha)` | `stats::analysis::andersonDarlingTest(data, dist, alpha)` |
| `D::likelihoodRatioTest(data, r, u, alpha)` | `stats::analysis::likelihoodRatioTest(data, r, u, df, alpha)` — see [LRT df change](#likelihoodratiotest-requires-explicit-df) |
| `D::informationCriteria(data, dist)` | `stats::analysis::informationCriteria(data, dist)` |
| `D::kFoldCrossValidation(data, k, seed)` | `stats::analysis::kFoldCrossValidation<D>(data, k, seed)` |
| `D::leaveOneOutCrossValidation(data)` | `stats::analysis::leaveOneOutCrossValidation<D>(data)` |
| `D::bootstrapParameterConfidenceIntervals(data, level, n, seed)` | `stats::analysis::bootstrapMeanVarianceCI<D>(data, level, n, seed)` |

Include the relevant header:

```cpp
#include "libstats/stats/analysis/goodness_of_fit.h"   // KS, AD, LRT
#include "libstats/stats/analysis/information_criteria.h"
#include "libstats/stats/analysis/cross_validation.h"
#include "libstats/stats/analysis/bootstrap.h"
```

Or the umbrella:

```cpp
#include "libstats/stats/analysis/analysis.h"
```

### Gaussian-specific tests

| v1.x call | v2.0.0 call |
|---|---|
| `GaussianDistribution::shapiroWilkTest(data, alpha)` | `stats::analysis::gaussian::shapiroWilkTest(data, alpha)` |
| `GaussianDistribution::jarqueBeraTest(data, alpha)` | `stats::analysis::gaussian::jarqueBeraTest(data, alpha)` |
| `GaussianDistribution::oneSampleTTest(data, mu0, alpha)` | `stats::analysis::gaussian::oneSampleTTest(data, mu0, alpha)` |
| `GaussianDistribution::twoSampleTTest(d1, d2, eq, alpha)` | `stats::analysis::gaussian::twoSampleTTest(d1, d2, eq, alpha)` |
| `GaussianDistribution::pairedTTest(d1, d2, alpha)` | `stats::analysis::gaussian::pairedTTest(d1, d2, alpha)` |
| `GaussianDistribution::confidenceIntervalMean(data, level, known)` | `stats::analysis::gaussian::confidenceIntervalMean(data, level, known)` |
| `GaussianDistribution::confidenceIntervalVariance(data, level)` | `stats::analysis::gaussian::confidenceIntervalVariance(data, level)` |
| `GaussianDistribution::robustEstimation(data, type, c)` | `stats::analysis::gaussian::robustEstimation(data, type, c)` |
| `GaussianDistribution::bayesianEstimation(data, ...)` | `stats::analysis::gaussian::bayesianEstimation(data, ...)` |

```cpp
#include "libstats/stats/analysis/gaussian_analysis.h"
```

### Poisson, Exponential, Gamma, Binomial

Per-distribution analysis headers follow the same extraction pattern:

```cpp
#include "libstats/stats/analysis/poisson_analysis.h"    // stats::analysis::poisson::*
#include "libstats/stats/analysis/exponential_analysis.h" // stats::analysis::exponential::*
#include "libstats/stats/analysis/gamma_analysis.h"       // stats::analysis::gamma::*
#include "libstats/stats/analysis/binomial_analysis.h"    // stats::analysis::binomial::*
```

---

## `likelihoodRatioTest` requires explicit `df`

The `df` (degrees of freedom) parameter is now required and moves before `alpha`.

```cpp
// v1.x — df was inferred (unreliably) from parameter counts
auto [lr, p, reject] = stats::analysis::likelihoodRatioTest(data, restricted, unrestricted);

// v2.0.0 — df is explicit
// For nested models differing in 1 parameter (e.g. fixed mean vs free mean):
auto [lr, p, reject] = stats::analysis::likelihoodRatioTest(data, restricted, unrestricted, 1);
// For joint test on all k parameters of the same distribution type:
auto [lr, p, reject] = stats::analysis::likelihoodRatioTest(data, restricted, unrestricted, 2);
```

Rule of thumb: `df` = number of parameters constrained under the null hypothesis.

---

## KS and AD tests require continuous distributions

`kolmogorovSmirnovTest` and `andersonDarlingTest` are now constrained to
`stats::concepts::ContinuousDistribution`. Passing a discrete distribution
(Poisson, Binomial, etc.) is a **compile-time error**.

```cpp
// v1.x — compiled, silently gave incorrect p-values for discrete data
stats::analysis::kolmogorovSmirnovTest(poisson_data, poisson_dist);

// v2.0.0 — compile-time error; use the correct test instead
stats::analysis::poisson::chiSquareGoodnessOfFit(poisson_data, poisson_dist, 0.05);
```

---

## Batch API changes

### Span-based batch API (primary)

The span-based batch overload is the primary interface in v2.0.0:

```cpp
// v2.0.0
std::vector<double> xs(1000), ys(1000);
dist.getProbability(std::span<const double>(xs), std::span<double>(ys));
dist.getLogProbability(std::span<const double>(xs), std::span<double>(ys));
dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(ys));
```

### Strategy-suffix methods removed

All `*WithStrategy` methods, `getBatchProbabilities`, `getBatchLogProbabilities`,
`getBatchCumulativeProbabilities`, and `getBatchQuantiles` are **removed**.

```cpp
// v1.x (removed in v2.0.0)
dist.getProbabilityBatch(xs.data(), ys.data(), n);
dist.getProbabilitiesWithStrategy(xs, ys, SIMD_STRATEGY);

// v2.0.0
dist.getProbability(std::span<const double>(xs), std::span<double>(ys));
// or with an explicit hint:
detail::PerformanceHint hint;
hint.strategy = detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
dist.getProbability(std::span<const double>(xs), std::span<double>(ys), hint);
```

---

## Result<T> API change (2026-07-01)

`Result<T>` was redesigned from a plain aggregate struct to a discriminated union
(`std::variant<T, ErrorInfo>`). The public **fields** are gone; use the methods below.

| Old (v1.x – v2.0.0-pre) | New (v2.0.0) |
|---|---|
| `result.value` | `*result` (lvalue ref) or `result.unwrap()` (move) |
| `std::move(result.value)` | `std::move(result).unwrap()` |
| `result.error_code` | `result.errorCode()` |
| `result.message` | `result.message()` |

```cpp
// v2.0.0
auto result = stats::GaussianDistribution::create(0.0, 1.0);
if (result.isOk()) {
    auto& g = *result;           // operator* returns T&
    // or to move out:
    auto g = std::move(result).unwrap();
} else {
    std::cerr << result.message() << "\n";  // note: method call
}
```

`makeError()` no longer constructs the value type on the error path.
`isOk()` and `isError()` are unchanged.

### `VoidResult` success sentinel

v1.x used `Result<bool>` with `ok(true)` as the sentinel for void-returning
operations. v2.0.0 introduces `VoidResult = Result<std::monostate>` to make
the absence of a meaningful value explicit.

```cpp
// v1.x
Result<bool> validateFoo(double x) {
    if (bad) return Result<bool>::makeError(code, msg);
    return Result<bool>::ok(true);
}

// v2.0.0
VoidResult validateFoo(double x) {
    if (bad) return VoidResult::makeError(code, msg);
    return VoidResult::ok({});
}
```

---

## Parameter validators moved to free functions

`validateBetaParameters`, `validateChiSquaredParameters`, `validateStudentTParameters`
(and all other `validate*Parameters` functions) are no longer private static
members of their distribution classes. They are now free functions returning
`VoidResult`, defined in `include/core/error_handling.h`, `namespace stats`.

```cpp
// v1.x
BetaDistribution::validateBetaParameters(alpha, beta);  // private, class-internal use only

// v2.0.0
#include "libstats/core/error_handling.h"
stats::VoidResult r = stats::validateBetaParameters(alpha, beta);
if (r.isError()) { /* ... */ }
```

If you were relying on these as private implementation details (uncommon —
they weren't part of the public API in v1.x), switch to calling the free
function directly.

---

## Distribution concepts replace `DistributionTraits<D>` SFINAE

v1.x used `DistributionTraits<D>` template specializations (SFINAE) to
constrain generic code to valid distribution types. v2.0.0 replaces this with
C++20 concepts in `stats::concepts` (`include/core/distribution_concepts.h`):

| Concept | Requirement |
|---|---|
| `AnyDistribution<D>` | Base contract: scalar PDF/CDF/quantile, span-based batch `getProbability`, `getMean`/`getVariance`/`getSkewness`/`getKurtosis`, `sample()`, identity queries, `getEntropy()`, and the `kDistributionType`/`kIsDiscrete` static members |
| `ContinuousDistribution<D>` | `AnyDistribution<D> && !D::kIsDiscrete` |
| `DiscreteDistribution<D>` | `AnyDistribution<D> && D::kIsDiscrete` |
| `FittableDistribution<D>` | `AnyDistribution<D>` + `std::default_initializable<D>` + `fit(const std::vector<double>&)` — required by the bootstrap and cross-validation templates |

Two practical consequences for custom distributions:
- **`getSkewness()`/`getKurtosis()` are now required** by `AnyDistribution` —
  they weren't part of the earlier trait contract. A custom distribution
  missing either method will now fail to satisfy the concept (clearer
  compile error than a SFINAE substitution failure) rather than silently
  compiling with reduced trait support.
- Passing a custom distribution to `stats::analysis` templates now produces a
  concept-based compile error (naming the unsatisfied requirement) instead of
  an opaque SFINAE failure, if it doesn't satisfy the relevant concept.

---

## `SIMDPolicy::Level` unified with `SIMDLevel`

v1.x had two parallel enums for the same concept: a standalone `SIMDLevel`
and `SIMDPolicy`'s own nested level enum. v2.0.0 makes `SIMDPolicy::Level` a
type alias for the canonical `SIMDLevel` (`include/platform/simd_policy.h`):

```cpp
using Level = SIMDLevel;  // was a separate nested enum in v1.x
```

Code that only used `SIMDPolicy::Level::X` values is unaffected; code that
compared or converted between the two former enum types no longer needs to,
since there is only one type now.

---

## `DistributionType` relocated and renamed

v1.x declared the distribution-kind enum as `LibDistributionType` in
`include/core/performance_dispatcher.h`. v2.0.0 extracts it to its own header
as the canonical `detail::DistributionType`:

```cpp
// v1.x
#include "libstats/core/performance_dispatcher.h"
libstats::LibDistributionType t = ...;

// v2.0.0
#include "libstats/core/distribution_type.h"
stats::detail::DistributionType t = ...;
```

See also "Removed symbols" below — `LibDistributionType` itself is gone, not
just relocated; `detail::DistributionType` is a distinct (richer) enum, not
a renamed alias of the old one.

---

## Legacy `validation.h`/`validation.cpp` deleted

v1.x shipped a free-function validation module (`include/core/validation.h`,
`stats::detail` namespace, snake_case names) alongside the distribution-class
static methods covered above — two parallel APIs for the same goodness-of-fit
tests. v2.0.0 deletes this module entirely; both v1.x paths converge on the
single `stats::analysis` API:

```cpp
// v1.x — free-function path (validation.h)
#include "libstats/core/validation.h"
stats::detail::KSTestResult r = stats::detail::kolmogorov_smirnov_test(data, dist);

// v1.x — distribution-class static-method path (see Analysis utilities above)
GaussianDistribution::kolmogorovSmirnovTest(data, dist, alpha);

// v2.0.0 — single path
#include "libstats/stats/analysis/goodness_of_fit.h"
stats::analysis::kolmogorovSmirnovTest(data, dist, alpha);
```

---

## Factory method Doxygen updated

The `create()` factory rationale has changed. In v1.x the comment cited an ABI
workaround for Homebrew LLVM libc++. In v2.0.0 the rationale is the deliberate API
design choice documented in `error_handling.h`. The factory itself is unchanged:

```cpp
auto result = stats::GaussianDistribution::create(0.0, 1.0);
if (result.isOk()) {
    auto& g = *result;  // operator* — see Result<T> API change above
    // ...
} else {
    std::cerr << result.message() << "\n";
}
```

---

## Build system changes

### CMake target name

```cmake
# v1.5.3 install target name
target_link_libraries(your_target PRIVATE libstats::libstats_static)

# v2.0.0 install target name
target_link_libraries(your_target PRIVATE libstats::static)
```

### Build type `CROSS_PLATFORM` removed

Use `Strict` instead:

```bash
# v1.x
cmake -B build -DCMAKE_BUILD_TYPE=CROSS_PLATFORM

# v2.0.0
cmake -B build -DCMAKE_BUILD_TYPE=Strict
```

### CMake flag removed

`LIBSTATS_HAS_REQUIRES_EXPRESSIONS` is no longer defined. C++20 concepts are
available unconditionally on all v2.x baseline compilers.

### Include paths

Headers are accessed via the `libstats/` prefix in both build and install trees:

```cpp
#include "libstats/libstats.h"                          // full interface
#include "libstats/distributions/gaussian.h"            // single distribution
#include "libstats/stats/analysis/goodness_of_fit.h"   // analysis
```

---

## Removed symbols

| Symbol | Removed in | Replacement |
|---|---|---|
| `LibDistributionType` enum | v2.0.0 | `detail::DistributionType` |
| `MemoryPool`, `SmallVector`, `StackAllocator`, `simd_vector` | v2.0.0 | Standard containers |
| `refineWithCapabilities()` | v2.0.0 | No replacement (had no effect on dispatch) |
| `getKLDivergence()` on `DistributionBase` | v2.0.0 | Compute manually from `getLogProbability` |
| All `*WithStrategy` batch methods | v2.0.0 | `getProbability(span, span, hint)` |
| `getBatchProbabilities` / `getBatchLogProbabilities` / etc. | v2.0.0 | Span-based overloads |
| Alternate LLVM compiler path (macOS) | v2.0.0 | System AppleClang |
