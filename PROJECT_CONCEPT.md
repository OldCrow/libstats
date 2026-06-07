# libstats Project Concept

## Purpose

libstats is a modern C++20 statistical distributions library built as a design and teaching project. The goal is to show how to build statistical software correctly: complete statistical interfaces, careful numerical behaviour, thread safety, SIMD-aware batch processing, and cross-platform portability without external runtime dependencies.

## Current Status

The library is feature-complete and stable at **v1.2.0**.

Nine distributions are fully implemented and validated across four target architectures:

- Gaussian
- Exponential
- Uniform
- Poisson
- Discrete
- Gamma
- Chi-squared
- Student's t
- Beta

All nine distributions share a uniform API including PDF, log-PDF, CDF, quantiles, moments, MLE parameter fitting, random sampling, SIMD batch operations, parallel batch operations, and `parallelBatchFit` for fitting multiple independent datasets in parallel.

Cross-platform SIMD validation status (54/54 SIMD tests, all four machines):

| Machine | SIMD | Correctness | simd_verification | Speedup |
|---|---|---|---|---|
| Ivy Bridge (2012 MBP) | AVX | 34/34 ✅ | 54/54 ✅ | 4.10x |
| Kaby Lake (2017 MBP) | AVX2 | 33/33 ✅ | 54/54 ✅ | 3.49x |
| Mac Mini M1 | NEON | 33/33 ✅ | 54/54 ✅ | 2.31x |
| Asus TUF A16 (Windows) | AVX-512 | 33/33 ✅ | 54/54 ✅ | 1.64x |

## Design Goals

### 1. Complete statistical interfaces

Each distribution provides more than random sampling. The interface for every distribution includes:

- probability evaluation (PDF, log-PDF)
- cumulative probability evaluation (CDF)
- quantiles (inverse CDF)
- statistical moments (mean, variance, skewness, kurtosis)
- random sampling (scalar and batch)
- parameter estimation (MLE via `fit()`)
- parallel batch fitting (`parallelBatchFit`)
- thread-safe parameter mutation with cache invalidation
- validation support (Result<T> factory, trySet* setters)

### 2. Modern C++20 without external dependencies

The library uses the standard library only. It relies on modern C++20 features where they improve clarity or safety, staying grounded in portability and maintainability.

### 3. Real performance work, not decorative optimization

Performance is part of the project concept, not an afterthought:

- runtime SIMD dispatch (SSE2/AVX/AVX2/AVX-512/NEON)
- vectorized batch operations where they provide genuine speedups
- parallel batch execution with empirically-derived architecture-aware thresholds
- performance tooling to validate dispatch choices and measure speedups

### 4. Teach through structure

The codebase is meant to be readable as well as usable. Architectural choices illustrate good design patterns:

- delegation where mathematical identity makes duplication unnecessary (Chi-squared over Gamma)
- log-space computation where numerical stability matters
- layered infrastructure instead of ad hoc optimization
- explicit validation and measurable performance claims
- template helpers to eliminate structural duplication across distributions

## Distribution Families

The nine distributions span four useful statistical families.

### Symmetric, unbounded continuous
- Gaussian
- Student's t

### Positive-support continuous
- Exponential
- Gamma
- Chi-squared

### Bounded continuous
- Uniform
- Beta

### Discrete
- Poisson
- Discrete

## Architecture

The project uses a six-level layered architecture.

### Level 0: Foundation
Platform detection, constants, and low-level safety utilities.

### Level 1: Core utilities
Mathematical utilities, validation, numerical safety, log-space operations, and shared statistical helpers.

### Level 2: Infrastructure
Performance dispatch, benchmarking, threshold management, system capability detection, caching, and initialization support.

### Level 3: Distribution framework
Base classes and common abstractions used by concrete distributions.

### Level 4: Concrete distributions
Individual distribution implementations, each using the common framework while keeping family-specific math local and readable.

### Level 5: Complete library interface
The umbrella header `libstats.h` and supporting configuration.

## Performance Model

Performance decisions are made at runtime based on problem size and machine capabilities.

The library supports four execution strategies:

- **SCALAR** — single-element operations for tiny workloads
- **VECTORIZED** — SIMD batch operations for medium workloads
- **PARALLEL** — multi-threaded execution for larger workloads
- **WORK_STEALING** — dynamic load balancing for large or irregular workloads

Strategy selection uses empirically-derived per-architecture thresholds (from profiling bundles across four machines) stored as a constexpr lookup table in `include/core/dispatch_thresholds.h`. Power users can override strategy selection explicitly via `getXxxWithStrategy()` variants.

## Repository Structure

### Core directories
- `include/` — public headers and internal header structure
- `src/` — implementation files
- `tests/` — correctness, integration, and enhanced tests (all GTest-based)
- `examples/` — usage-oriented demonstrations
- `tools/` — analysis, validation, and benchmarking utilities
- `docs/` — focused guides and reference material
- `scripts/` — build and maintenance helpers
- `data/profiles/dispatcher/` — profiling bundles for empirical threshold derivation

### User-facing entry points
- `include/libstats.h` — umbrella header
- `README.md` — project overview and onboarding
- `AGENTS.md` — repository-specific development guidance for agents

## Tooling

The repo includes tools for two distinct jobs.

### Development and validation tools
Validate correctness, SIMD behaviour, thresholds, and runtime capabilities:

- `system_inspector` — hardware and capability diagnostics
- `simd_verification` — correctness and speedup validation across all distributions
- `strategy_profile` — forced-strategy profiling for threshold tuning
- `parallel_batch_fitting_benchmark` — parallel MLE performance analysis

### Test infrastructure
All tests use GTest and are registered with CTest. Tests are labelled `timing` when they contain speedup assertions sensitive to CPU load; correctness tests carry no label and are safe to run in parallel (`ctest -j8 -LE timing`).
