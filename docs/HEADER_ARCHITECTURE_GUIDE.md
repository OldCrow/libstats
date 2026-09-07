# Header Architecture Guide

This guide describes the v2.x public header layout.

## Public include styles

Single-header include:

```cpp
#include "libstats.h"
```

Focused include:

```cpp
#include "libstats/distributions/gaussian.h"
#include "libstats/stats/analysis/gaussian_analysis.h"
```

## Top-level layout

```text
include/
└── libstats/
    ├── libstats.h
    ├── common/
    ├── core/
    ├── distributions/
    ├── platform/
    └── stats/
        └── analysis/
```

## Core headers

Important v2.x core headers:

- `core/distribution_base.h` — base class and shared numerical utilities
- `core/distribution_concepts.h` — C++20 concepts for distribution templates
- `core/dispatch_utils.h` — span-based auto-dispatch and PerformanceHint routing
- `core/dispatch_thresholds.h` — architecture-specific calibrated thresholds
- `core/error_handling.h` — `Result<T>`, `VoidResult`, and validation error types

Removed v1.x headers:

- the old core statistical-utilities stub header
- the old distribution-memory utility header

## Distribution headers

Each distribution header owns:

- parameter accessors
- scalar PDF/LogPDF/CDF/quantile methods
- span-based batch APIs
- fitting and sampling
- distribution-specific scalar utilities

Statistical analysis workflows are not class members in v2.x. Use `stats::analysis` headers.

## Analysis headers

Generic analysis headers:

- `stats/analysis/goodness_of_fit.h`
- `stats/analysis/information_criteria.h`
- `stats/analysis/cross_validation.h`
- `stats/analysis/bootstrap.h`
- `stats/analysis/statistical_utilities.h`
- `stats/analysis/analysis.h` (generic umbrella only)

Distribution-specific analysis headers must be included explicitly:

- `stats/analysis/gaussian_analysis.h`
- `stats/analysis/poisson_analysis.h`
- `stats/analysis/exponential_analysis.h`
- `stats/analysis/gamma_analysis.h`
- `stats/analysis/binomial_analysis.h`
- `stats/analysis/discrete_analysis.h`

Do not add distribution-specific analysis headers to `analysis.h`; that umbrella is intentionally generic.

## Concepts

Use `stats::concepts` for generic distribution constraints:

```cpp
template <stats::concepts::AnyDistribution D>
void analyse(const D& dist);
```

The concepts namespace avoids name collisions with concrete distribution classes such as `DiscreteDistribution`.

## Namespace hygiene

Validation helper functions moved to `stats::detail` in v2.0.0:

- `stats::detail::validateParameter`
- `stats::detail::validatePositiveParameter`
- `stats::detail::validateNonNegativeParameter`

Do not expose these helpers as part of the public API.

## Batch APIs

Use spans and optional `PerformanceHint`:

```cpp
std::vector<double> out(values.size());
dist.getProbability(std::span<const double>(values), std::span<double>(out));
```

Removed v1.x APIs:

- explicit strategy suffix methods
- vector-returning base batch helpers

## Installed include path

Installed headers land under:

```text
include/libstats/
```

The source tree mirrors this layout directly (`<src>/include/libstats/`), so
the build tree and the install tree resolve `#include "libstats/core/foo.h"`
identically with no shim, symlink, or copy step involved.

## Distribution roster (27 implemented, across 7 families)
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
17. **Geometric** - Geo(p) — discrete, delegate over NegBinomial(r=1); MLE: p̂=1/(1+x̄)
18. **Laplace** - Laplace(μ, b) — standalone, fabs+vector_exp SIMD; MLE: median/MAD
19. **Cauchy** - Cauchy(x₀, γ) — PDF/LogPDF delegate to StudentT(ν=1), CDF/Quantile closed-form (#48); moments NaN; Fisher-scoring MLE
20. **Logistic** - Logistic(μ, s) — vector_exp pipeline, log1p/expm1 tail stability (#54)
21. **Gumbel** - Gumbel(μ, β) — max-stable (`gumbel_r`) only; double-exp pipeline (#54)
22. **Bernoulli** - Bern(p) — delegation wrapper over Binomial(n=1); p ∈ [0,1] inclusive (#55)
23. **Erlang** - Erlang(k, λ) — pure delegation over Gamma(k, λ); RATE-parameterized, int k (#55)
24. **FisherF** - F(d₁, d₂) — closed-form PDF + steered `detail::` CDF (NOT a Beta delegation) (#56)
25. **InverseGamma** - InvGamma(α, β) — x → 1/x transform over Gamma, complement-native tails (#56)
26. **HalfNormal** - HN(σ) — erf/erfc pipeline over Gaussian machinery (#57)
27. **TruncatedNormal** - TN(μ, σ, a, b) — regime-split erfc normalization; rejects Z-underflow windows (#57)

Each implemented distribution provides: PDF/CDF/Quantiles, Statistical Moments, Parameter Estimation (MLE), Random Sampling, Statistical Validation, SIMD batch operations.

Parenthetical notes record how each distribution is implemented — a
delegation wrapper, a transform over another distribution, or a standalone
pipeline — and the issue that introduced it. Moved here from AGENTS.md on
2026-09-07: the roster is reference material, not per-turn orientation.
