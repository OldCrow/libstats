# libstats Header Dependency Analysis

## Overview

This document provides a comprehensive analysis of the libstats header dependency structure, organizing headers into logical abstraction levels from foundational (only standard library dependencies) to the top-level API. The analysis helps understand build dependencies, architectural relationships, and integration points.

## Architecture Approach

The libstats header structure follows a **hybrid layered architecture**:
- **Strict dependency ordering**: Higher levels depend only on lower levels
- **Conceptual grouping**: Headers are grouped by functionality and abstraction level
- **Platform separation**: Platform-dependent headers are clearly distinguished from platform-independent ones
- **Circular dependency prevention**: No circular dependencies exist between headers

## Abstraction Levels

### Level 0: Foundational Headers (Standard Library Only)

These headers have no internal dependencies and only include standard library headers. They provide the foundational constants, utilities, and basic abstractions.

#### Core Constants (Platform-Independent)
- **`core/precision_constants.h`**
  - **Purpose**: Numerical precision, tolerance values, and convergence criteria
  - **Dependencies**: `<limits>` (standard library only)
  - **Key Content**: Machine epsilon values, numerical method tolerances, iteration limits
  - **Usage**: Used by all numerical computation headers

- **`core/mathematical_constants.h`**
  - **Purpose**: Fundamental mathematical constants (π, e, √2, etc.)
  - **Dependencies**: Standard library math constants only
  - **Key Content**: High-precision mathematical constants, special values
  - **Usage**: Used by statistical functions and distributions

- **`core/statistical_constants.h`**
  - **Purpose**: Statistical critical values, test parameters, and distribution constants
  - **Dependencies**: Standard library only
  - **Key Content**: Chi-square critical values, t-distribution tables, normality test thresholds

- **`core/probability_constants.h`**
  - **Purpose**: Probability bounds, safety limits, and validation thresholds
  - **Dependencies**: Standard library only
  - **Key Content**: Probability bounds [0,1], infinity representations, numerical safety limits

- **`core/threshold_constants.h`**
  - **Purpose**: Algorithmic thresholds and decision boundaries
  - **Dependencies**: Standard library only
  - **Key Content**: Convergence thresholds, optimization stopping criteria

- **`core/benchmark_constants.h`**
  - **Purpose**: Performance testing parameters and timing constants
  - **Dependencies**: `<chrono>` (standard library only)
  - **Key Content**: Benchmark iteration counts, timing thresholds, performance baselines

- **`core/robust_constants.h`**
  - **Purpose**: Robust estimation parameters and outlier detection thresholds
  - **Dependencies**: Standard library only
  - **Key Content**: Huber thresholds, breakdown points, influence function parameters

- **`core/statistical_methods_constants.h`**
  - **Purpose**: Advanced statistical method parameters (Bayesian, bootstrap, cross-validation)
  - **Dependencies**: Standard library only
  - **Key Content**: MCMC parameters, bootstrap iteration counts, cross-validation folds

- **`core/goodness_of_fit_constants.h`**
  - **Purpose**: Critical values for goodness-of-fit statistical tests
  - **Dependencies**: Standard library only
  - **Key Content**: Kolmogorov-Smirnov critical values, Anderson-Darling thresholds

#### Basic Platform Headers (Standard Library + Platform APIs)
- **`platform/simd_policy.h`**
  - **Purpose**: SIMD policy definitions and compile-time feature detection
  - **Dependencies**: Standard library only
  - **Key Content**: SIMD capability constants, platform detection macros
  - **Usage**: Included by simd.h for policy-based SIMD selection

### Level 1: Consolidated Constants and Platform Foundation

These headers aggregate Level 0 headers and provide platform-dependent foundations with runtime capabilities.

#### Aggregated Constants
- **`core/constants.h`**
  - **Purpose**: Umbrella header providing access to all core constants
  - **Dependencies**: All Level 0 core constant headers
  - **Key Content**: Single include point for all mathematical, statistical, and precision constants
  - **Usage**: Convenient single-header include for most components

#### Platform Foundation
- **`platform/cpu_detection.h`**
  - **Purpose**: Runtime CPU feature detection and capability assessment
  - **Dependencies**: Standard library + platform-specific APIs (`<cpuid.h>`, `<sys/sysctl.h>`, etc.)
  - **Key Content**: CPU feature detection, cache hierarchy info, performance monitoring utilities
  - **Usage**: Runtime SIMD capability detection, performance optimization decisions

- **`platform/platform_constants.h`**
  - **Purpose**: Platform-dependent optimization constants with runtime adaptation
  - **Dependencies**: `cpu_detection.h` (forward declarations)
  - **Key Content**: SIMD alignment constants, cache-optimized block sizes, architecture-specific thresholds
  - **Usage**: Performance tuning based on detected CPU capabilities

### Level 2: Core Utilities and Platform Capabilities

These headers provide essential utilities and platform-specific optimizations, building on the foundation established in Levels 0-1.

#### Core Utilities (Platform-Independent)
- **`core/math_utils.h`**
  - **Purpose**: Mathematical utility functions and numerical algorithms
  - **Dependencies**: `constants.h`, standard library math functions
  - **Key Content**: Special functions (gamma, beta, error functions), numerical integration, root finding
  - **Usage**: Mathematical operations for statistical distributions

- **`core/log_space_ops.h`**
  - **Purpose**: Log-space arithmetic for numerical stability
  - **Dependencies**: `constants.h`, `math_utils.h`
  - **Key Content**: Log-sum-exp, log-difference operations, stable probability computations
  - **Usage**: Numerical stability in probability calculations

- **`core/safety.h`**
  - **Purpose**: Safe numerical operations and bounds checking
  - **Dependencies**: `constants.h`, standard library
  - **Key Content**: Overflow detection, safe arithmetic, boundary validation
  - **Usage**: Preventing numerical errors in computations

- **`core/validation.h`**
  - **Purpose**: Parameter validation and constraint checking
  - **Dependencies**: `constants.h`, `safety.h`
  - **Key Content**: Distribution parameter validation, data quality checks
  - **Usage**: Input validation for distribution constructors

- **`core/error_handling.h`**
  - **Purpose**: Exception-free error handling system
  - **Dependencies**: `constants.h`, standard library
  - **Key Content**: `Result<T>` types, error codes, exception-safe wrappers
  - **Usage**: Safe error handling without ABI compatibility issues

- **`core/statistical_utilities.h`**
  - **Purpose**: Common statistical computations and algorithms
  - **Dependencies**: `constants.h`, `math_utils.h`, `validation.h`
  - **Key Content**: Descriptive statistics, correlation functions, statistical tests
  - **Usage**: Shared statistical functionality across distributions

#### Platform Capabilities
- **`platform/simd.h`**
  - **Purpose**: SIMD-optimized vector operations and memory management
  - **Dependencies**: `simd_policy.h`, `cpu_detection.h`, platform-specific intrinsics
  - **Key Content**: Vectorized operations, aligned memory allocation, SIMD dispatching
  - **Usage**: High-performance batch operations in distributions

- **`platform/parallel_thresholds.h`**
  - **Purpose**: Architecture-specific parallel processing thresholds
  - **Dependencies**: `platform_constants.h`, `cpu_detection.h`
  - **Key Content**: Optimal grain sizes, parallel execution thresholds by CPU architecture
  - **Usage**: Deciding when to use parallel algorithms

- **`platform/thread_pool.h`**
  - **Purpose**: Thread pool implementation for parallel computations
  - **Dependencies**: `parallel_thresholds.h`, standard library threading
  - **Key Content**: Work-stealing thread pool, task scheduling, load balancing
  - **Usage**: Parallel execution infrastructure

- **`platform/work_stealing_pool.h`**
  - **Purpose**: Advanced work-stealing thread pool for heavy computations
  - **Dependencies**: `thread_pool.h`, `cpu_detection.h`
  - **Key Content**: Work-stealing algorithms, dynamic load balancing, NUMA awareness
  - **Usage**: High-performance parallel computations

### Level 3: Advanced Infrastructure and Distribution Framework

These headers provide advanced infrastructure components and the foundation for statistical distributions.

#### Performance and Caching Infrastructure
- **`platform/adaptive_cache.h`**
  - **Purpose**: Memory-aware adaptive caching with TTL and eviction policies
  - **Dependencies**: `platform_constants.h`, `cpu_detection.h`, standard library containers and threading
  - **Key Content**: `AdaptiveCache<K,V>` class, eviction policies, memory pressure handling
  - **Usage**: Caching expensive statistical computations

- **`platform/distribution_cache.h`**
  - **Purpose**: Specialized caching adapter for distribution computations
  - **Dependencies**: `adaptive_cache.h`, `thread_pool.h`
  - **Key Content**: Distribution-specific cache optimization, precomputation strategies
  - **Usage**: Optimized caching for quantile functions and special function evaluations

- **`platform/parallel_execution.h`**
  - **Purpose**: C++20 parallel algorithm wrapper with automatic fallback
  - **Dependencies**: `parallel_thresholds.h`, `thread_pool.h`, C++20 execution policies
  - **Key Content**: Safe parallel algorithms, automatic threshold detection, fallback mechanisms
  - **Usage**: Parallel batch operations in distributions

- **`platform/benchmark.h`**
  - **Purpose**: Performance benchmarking and profiling utilities
  - **Dependencies**: `cpu_detection.h`, `benchmark_constants.h`
  - **Key Content**: Timing utilities, performance measurement, statistical significance testing
  - **Usage**: Performance validation and optimization

#### Performance Optimization Framework
- **`core/performance_history.h`**
  - **Purpose**: Performance data collection and analysis for smart dispatch
  - **Dependencies**: `constants.h`, `error_handling.h`, standard library threading and atomics
  - **Key Content**: `PerformanceHistory` class, thread-safe performance tracking
  - **Usage**: Learning system for auto-dispatch optimization

- **`core/performance_dispatcher.h`**
  - **Purpose**: Smart auto-dispatch system for performance optimization
  - **Dependencies**: `performance_history.h`, `cpu_detection.h`, `platform_constants.h`
  - **Key Content**: Intelligent algorithm selection, performance monitoring, adaptive optimization
  - **Usage**: Automatic selection of optimal implementation strategies

#### Distribution Framework
- **`core/distribution_interface.h`**
  - **Purpose**: Pure virtual interface defining distribution contract
  - **Dependencies**: `error_handling.h`, `validation.h`, standard library
  - **Key Content**: Abstract base class with pure virtual methods for PDF, CDF, quantile functions
  - **Usage**: Interface contract for all distribution implementations

- **`core/distribution_memory.h`**
  - **Purpose**: Memory management and SIMD-optimized operations for distributions
  - **Dependencies**: `simd.h`, `adaptive_cache.h`, `constants.h`
  - **Key Content**: Memory-efficient batch operations, SIMD optimization, cache management
  - **Usage**: Memory optimization for distribution computations

- **`core/distribution_validation.h`**
  - **Purpose**: Comprehensive validation and diagnostic testing for distributions
  - **Dependencies**: `statistical_utilities.h`, `validation.h`, `error_handling.h`
  - **Key Content**: Goodness-of-fit tests, parameter estimation validation, diagnostic functions
  - **Usage**: Statistical validation and fitting diagnostics

### Level 4: Complete Distribution Framework

This level provides the complete, integrated distribution framework that combines all lower-level components.

- **`core/distribution_base.h`**
  - **Purpose**: Complete base class implementation for statistical distributions
  - **Dependencies**: `distribution_interface.h`, `distribution_memory.h`, `distribution_validation.h`, `distribution_cache.h`, `platform_constants.h`
  - **Key Content**: Full-featured base class with caching, SIMD operations, validation, and threading support
  - **Usage**: Base class for all concrete distribution implementations
  - **Integration Points**: 
    - Inherits from `DistributionInterface` for contract compliance
    - Uses `ThreadSafeCacheManager` from distribution_memory.h for performance
    - Integrates validation and diagnostic capabilities
    - Provides SIMD-optimized batch operations
    - Includes comprehensive rule-of-five implementation

### Level 5: Concrete Distributions

These headers implement specific statistical distributions using the complete framework provided by Level 4.

#### Distribution Implementations
- **`distributions/gaussian.h`**
  - **Purpose**: Gaussian (Normal) distribution implementation
  - **Dependencies**: `distribution_base.h`, `constants.h`, `simd.h`, `error_handling.h`, `performance_dispatcher.h`, platform execution and caching headers
  - **Key Content**: Complete Gaussian distribution with SIMD optimization, thread safety, smart dispatch
  - **Special Features**: Standard normal optimizations, Box-Muller sampling, error function integration

- **`distributions/exponential.h`**
  - **Purpose**: Exponential distribution implementation
  - **Dependencies**: Same as gaussian.h (full distribution framework)
  - **Key Content**: Exponential distribution with memoryless property optimizations
  - **Special Features**: Inverse transform sampling, tail probability optimizations

- **`distributions/uniform.h`**
  - **Purpose**: Uniform distribution implementation
  - **Dependencies**: Same as gaussian.h (full distribution framework)
  - **Key Content**: Uniform distribution with linear interpolation optimizations
  - **Special Features**: Efficient quantile function, rectangular window operations

- **`distributions/poisson.h`**
  - **Purpose**: Poisson distribution implementation (discrete)
  - **Dependencies**: Same as gaussian.h (full distribution framework)
  - **Key Content**: Poisson distribution with large λ approximations
  - **Special Features**: Stirling's approximation, normal approximation for large parameters

- **`distributions/gamma.h`**
  - **Purpose**: Gamma distribution implementation
  - **Dependencies**: Same as gaussian.h (full distribution framework)
  - **Key Content**: Gamma distribution with shape/scale parameterization
  - **Special Features**: Incomplete gamma function integration, Ahrens-Dieter sampling

- **`distributions/discrete.h`**
  - **Purpose**: Generic discrete distribution implementation
  - **Dependencies**: Same as gaussian.h (full distribution framework)
  - **Key Content**: Flexible discrete distribution with arbitrary support
  - **Special Features**: Efficient probability mass function lookup, cumulative sum optimization

### Level 6: Top-Level API

- **`libstats.h`**
  - **Purpose**: Complete library API and public interface
  - **Dependencies**: `distribution_base.h`, `constants.h`, `simd.h`, `cpu_detection.h`, `parallel_execution.h`, `adaptive_cache.h`, all distribution headers
  - **Key Content**: 
    - Public API with comprehensive documentation
    - Type aliases for common usage patterns
    - Version information and feature descriptions
    - Integration guides for SIMD, parallel execution, and caching
  - **Design Philosophy**: Single header include for complete functionality
  - **Special Features**: Extensive usage documentation, performance optimization guides

## Level Justification

### Why 6 Levels?

The 6-level structure reflects natural abstraction boundaries:

1. **Level 0**: Pure constants and basic utilities (no interdependencies)
2. **Level 1**: Aggregation and platform foundation (runtime capabilities)
3. **Level 2**: Core utilities and platform-specific optimizations
4. **Level 3**: Advanced infrastructure (caching, performance, parallel execution)
5. **Level 4**: Complete framework integration
6. **Level 5**: Concrete implementations using the framework
7. **Level 6**: Public API and complete library interface

### Architecture Benefits

1. **Build Efficiency**: Lower levels compile independently, enabling parallel builds
2. **Maintainability**: Clear separation of concerns with minimal coupling
3. **Testing**: Each level can be tested independently
4. **Platform Portability**: Platform-specific code is isolated to specific levels
5. **Performance**: Critical path (distributions) has optimized dependencies

## Integration Points

### Thread Safety
- **Level 2+**: All headers from Level 2 onwards provide thread-safe operations
- **Synchronization**: Uses `std::shared_mutex` for concurrent reads, exclusive writes
- **Cache Management**: Thread-safe caching with atomic operations for performance tracking

### SIMD Integration
- **Detection**: Compile-time (simd.h) + Runtime (cpu_detection.h) capability detection
- **Dispatch**: Automatic selection of best available SIMD implementation
- **Optimization**: Platform-specific tuning constants based on detected CPU features

### Performance Optimization
- **Smart Dispatch**: Performance history learning system for algorithm selection
- **Adaptive Caching**: Memory-aware caching with configurable eviction policies
- **Parallel Execution**: Automatic threshold-based parallel algorithm selection

## Usage Recommendations

### For Library Users
- **Simple Usage**: Include only `libstats.h` for complete functionality
- **Selective Usage**: Include specific distribution headers for faster compilation
- **Platform Optimization**: Library automatically detects and uses best available optimizations

### For Library Developers
- **Adding Distributions**: Inherit from `DistributionBase` and implement pure virtual methods
- **Platform Support**: Add platform-specific constants to Level 1 headers
- **Performance Tuning**: Modify Level 2-3 headers for optimization parameters

### Build Dependencies
- **Minimal Rebuild**: Changes to Level 0 headers require full rebuild
- **Incremental Builds**: Changes to higher levels only affect dependent headers
- **Platform Isolation**: Platform-specific changes isolated to platform/ headers

## Circular Dependency Prevention

The strict level hierarchy prevents circular dependencies:
- Higher levels depend only on lower levels
- Platform headers are isolated from core headers at each level
- Interface headers are separated from implementation headers
- Aggregation headers (like constants.h) only depend on lower-level focused headers

This architecture ensures clean builds, maintainable code, and efficient compilation times.
