# libstats Project Concept

## Purpose

libstats is a modern C++20 statistical distributions library built as a design and teaching project. The goal is to show how to build statistical software correctly: complete statistical interfaces, careful numerical behavior, thread safety, SIMD-aware batch processing, and cross-platform portability without external runtime dependencies.

This document is the current high-level concept for the project. It is not a historical implementation sketch.

## Current Status

The library is feature-complete through Phase 6B.

Implemented distribution set:
- Gaussian
- Exponential
- Uniform
- Poisson
- Discrete
- Gamma
- Chi-squared
- Student's t
- Beta

The remaining work before a v1.0.0 release is release-gate validation rather than feature development:
- final cross-machine validation
- final Windows/MSVC validation
- first full AVX-512 validation on the Asus A16
- merge of `phase-6b-new-distributions` back to `main`

## Design Goals

### 1. Complete statistical interfaces

Each distribution should provide more than random sampling. The intended interface includes:
- probability evaluation
- cumulative probability evaluation
- quantiles
- statistical moments
- random sampling
- parameter estimation
- validation support

### 2. Modern C++20 without external dependencies

The library uses the standard library only. It relies on modern C++20 features where they improve clarity or safety, but the design stays grounded in portability and maintainability.

### 3. Real performance work, not decorative optimization

Performance is part of the project concept, not an afterthought. That includes:
- runtime SIMD dispatch
- vectorized batch operations where they are genuinely useful
- parallel batch execution with architecture-aware thresholds
- performance tooling to validate dispatch choices and speedups

### 4. Teach through structure

The codebase is meant to be readable as well as usable. Architectural choices should illustrate good design patterns:
- delegation where mathematical identity makes duplication unnecessary
- log-space computation where numerical stability matters
- layered infrastructure instead of ad hoc optimization
- explicit validation and measurable performance claims

## Distribution Families

The implemented distributions now cover four useful teaching families.

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

This family view is often more useful than treating each distribution as an isolated feature. It also provides a better structure for examples and documentation.

## Architecture

The project uses a layered architecture.

### Foundation
Platform detection, constants, and low-level safety utilities.

### Core utilities
Mathematical utilities, validation, numerical safety, log-space operations, and shared statistical helpers.

### Infrastructure
Performance dispatch, benchmarking, threshold management, system capability detection, caching, and initialization support.

### Distribution framework
Base classes and common abstractions used by concrete distributions.

### Concrete distributions
Individual distribution implementations, each using the common framework while keeping family-specific math local and readable.

## Performance Model

Performance decisions are made at runtime based on problem size and machine capabilities.

The library supports:
- scalar execution for small workloads
- SIMD/vectorized execution for medium workloads
- parallel execution for larger workloads
- work-stealing for large or irregular workloads

The core performance idea is that batch APIs should make optimization accessible without burdening normal users with low-level strategy decisions. Power users can still override strategy selection when needed.

## Repository Surface

### Core directories
- `include/` — public headers and internal header structure
- `src/` — implementation files
- `tests/` — correctness, integration, and enhanced tests
- `examples/` — usage-oriented demonstrations
- `tools/` — analysis, validation, and benchmarking utilities
- `docs/` — focused guides and reference material
- `scripts/` — build and maintenance helpers

### User-facing entry points
- `include/libstats.h` — umbrella header
- `README.md` — project overview and onboarding
- `WARP.md` — repository-specific development guidance

## Tooling Concept

The repo includes tools for two distinct jobs.

### Development and validation tools
These help validate correctness, SIMD behavior, thresholds, and runtime capabilities.
Examples:
- `system_inspector`
- `simd_verification`
- `parallel_threshold_benchmark`
- `performance_dispatcher_tool`
- `learning_analyzer`

### Historical or specialized analysis tools
These support specific refactors or investigations and should be documented as such when retained.
The project should distinguish clearly between active tools and archival or one-off utilities.

## Testing Concept

Testing is intentionally layered.

- distribution-specific tests validate math and API behavior
- cross-cutting tests validate safety, dispatch, initialization, and infrastructure
- SIMD verification validates both correctness and measured speedups
- timing-sensitive tests are separated from correctness tests so noisy machines do not create false failures

The intended release standard is not just "builds successfully" but "builds, passes correctness tests, and passes architecture-appropriate validation tools."

## Release Gate for v1.0.0

The project is ready for final release validation when these are all true:
- documentation matches the implemented library
- examples reflect the implemented distribution families
- stale or superseded tests/tools are cleaned up or explicitly marked historical
- correctness suite passes
- SIMD verification passes
- Windows/MSVC validation passes
- AVX-512 validation on the Asus A16 is complete

At that point, the codebase is ready to merge to `main` and tag v1.0.0.
