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
- `stats/analysis/analysis.h` (generic umbrella only)

Distribution-specific analysis headers must be included explicitly:

- `stats/analysis/gaussian_analysis.h`
- `stats/analysis/poisson_analysis.h`
- `stats/analysis/exponential_analysis.h`
- `stats/analysis/gamma_analysis.h`
- `stats/analysis/binomial_analysis.h`

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

Installed headers are expected under:

```text
include/libstats/
```

The build tree mirrors this via:

```text
build/include_shim/libstats/
```
