# libstats - Modern C++20 Statistical Distributions Library

[![CI](https://github.com/OldCrow/libstats/actions/workflows/ci.yml/badge.svg)](https://github.com/OldCrow/libstats/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/OldCrow/libstats/graph/badge.svg)](https://codecov.io/gh/OldCrow/libstats)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety](https://img.shields.io/badge/Memory%20Safety-Enterprise%20Grade-green.svg)](#safety-features)
[![Performance](https://img.shields.io/badge/Performance-SIMD%20%26%20Parallel-blue.svg)](#performance-features)

A modern C++20 statistical distributions library demonstrating how to build statistical software correctly — with genuine SIMD vectorization, parallel dispatch, thread safety, and zero external dependencies.

**📖 Complete Documentation:** For detailed information about building, architecture, parallel processing, and platform support, see the [comprehensive guides](#documentation) below.

## Features

### 🎯 **Complete Statistical Interface**
- **PDF/CDF/Quantiles**: Full probability density, cumulative distribution, and quantile functions
- **Statistical Moments**: Mean, variance, skewness, kurtosis with thread-safe access
- **Random Sampling**: Integration with std:: distributions for high-quality random number generation
- **Parameter Estimation**: Maximum Likelihood Estimation (MLE) with comprehensive diagnostics
|- **Statistical Validation**: KS and AD Goodness-of-Fit, model selection

### 📊 **Available Distributions**
- **Gaussian (Normal)**: N(μ, σ²) - The cornerstone of statistics ✅
- **Exponential**: Exp(λ) - Waiting times and reliability analysis ✅
- **Uniform**: U(a, b) - Continuous uniform random variables ✅
- **Poisson**: P(λ) - Count data and rare events ✅
- **Discrete**: Custom discrete distributions with arbitrary support ✅
- **Gamma**: Γ(α, β) - Positive continuous variables ✅

### ⚡ **Modern C++20 Design**
- **Thread-Safe**: Concurrent read access with safe cache management
- **Zero Dependencies**: Only standard library required
- **SIMD Optimized**: Vectorized operations for bulk calculations
- **Memory Safe**: RAII principles and smart pointer usage
- **Exception Safe**: Robust error handling throughout
- **C++20 Concepts**: Type-safe mathematical function interfaces
- **Parallel Processing**: Traditional and work-stealing thread pools

### 🛡️ **Safety & Numerical Stability**
- **Memory Safety**: Comprehensive bounds checking and overflow protection
- **Numerical Stability**: Safe mathematical operations and edge case handling
- **Error Recovery**: Multiple strategies for handling numerical failures
- **Convergence Detection**: Advanced monitoring for iterative algorithms
- **Diagnostics**: Automated numerical health assessment

### 🧪 **Statistical Validation**
- **Goodness-of-Fit Tests**: Kolmogorov-Smirnov, Anderson-Darling (✅ implemented)
- **Model Selection**: AIC/BIC information criteria (✅ implemented)
- **Residual Analysis**: Standardized residuals and diagnostics (✅ implemented)
- **Cross-Validation**: K-fold validation framework (✅ implemented)

### 🚀 **Performance Features**
- **SIMD Operations**: Vectorized statistical computations with cross-platform detection
- **Parallel Processing**: Both traditional and work-stealing thread pools
- **C++20 Parallel Algorithms**: Safe wrappers for `std::execution` policies
- **Cache Optimization**: Thread-safe caching with lock-free fast paths

**📖 Cross-Platform SIMD Support**: Automatic detection and optimization for SSE2/AVX/AVX2/AVX-512/NEON instruction sets with runtime safety verification. Validated on Intel (Ivy Bridge/Kaby Lake), Apple Silicon (M1/NEON), AMD Ryzen Zen 4 (AVX-512), and Linux CI.

## Quick Start

### Quick Build

```bash
git clone https://github.com/OldCrow/libstats.git
cd libstats
mkdir build && cd build
cmake ..                    # Auto-detects optimal configuration
make -j$(nproc)            # Parallel build with auto-detected core count
ctest --output-on-failure  # Run tests
```

**📖 For complete build information**, including cross-platform support, SIMD optimization, and advanced configuration options, see [docs/BUILD_SYSTEM_GUIDE.md](docs/BUILD_SYSTEM_GUIDE.md).

### Basic Usage

```cpp
#include "libstats.h"
#include <iostream>
#include <vector>

int main() {
    // Initialize performance systems (recommended)
    libstats::initialize_performance_systems();

    // Create distributions with safe factory methods
    auto gaussian_result = libstats::GaussianDistribution::create(0.0, 1.0);
    if (gaussian_result.isOk()) {
        auto& gaussian = gaussian_result.value;

        // Single-value operations
        std::cout << "PDF at 1.0: " << gaussian.getProbability(1.0) << std::endl;
        std::cout << "CDF at 1.0: " << gaussian.getCumulativeProbability(1.0) << std::endl;

        // High-performance batch operations (auto-optimized)
        std::vector<double> values(10000);
        std::vector<double> results(10000);
        std::iota(values.begin(), values.end(), -5.0);

        gaussian.getProbability(std::span<const double>(values),
                               std::span<double>(results));

        std::cout << "Processed " << values.size() << " values with auto-optimization" << std::endl;
    }
    return 0;
}
```

**📖 For comprehensive parallel processing and batch operation guides**, see [docs/PARALLEL_BATCH_PROCESSING_GUIDE.md](docs/PARALLEL_BATCH_PROCESSING_GUIDE.md).

## Project Structure

```
libstats/
├── include/           # Modular header architecture
│   ├── libstats.h        # Complete library (single include)
│   ├── core/             # Core mathematical and statistical components
│   ├── distributions/    # Statistical distributions (Gaussian, Exponential, etc.)
│   └── platform/         # SIMD, threading, and platform optimizations
├── src/              # Implementation files
├── tests/            # Comprehensive unit and integration tests
├── examples/         # Usage demonstrations
├── tools/            # Performance analysis and optimization utilities
├── docs/             # Complete documentation guides
└── scripts/          # Build and development scripts
```

**📖 For detailed header organization and dependency management**, see [docs/HEADER_ARCHITECTURE_GUIDE.md](docs/HEADER_ARCHITECTURE_GUIDE.md).

## Key Features Summary

### 🎯 **Statistical Completeness**
- PDF, CDF, quantiles, parameter estimation, and validation
- 6 distributions: Gaussian, Exponential, Uniform, Poisson, Discrete, Gamma
- Beyond `std::` distributions with full statistical interfaces

### ⚡ **High Performance**
- Automatic SIMD optimization (SSE2, AVX, AVX2, NEON)
- Intelligent parallel processing with auto-dispatch
- Thread-safe batch operations with work-stealing pools
- Smart caching and adaptive algorithm selection

### 🛡️ **Enterprise Safety**
- Memory-safe operations with comprehensive bounds checking
- Exception-safe error handling with safe factory methods
- Thread-safe concurrent access with reader-writer locks
- Numerical stability with log-space arithmetic

### 🔧 **Modern C++20 Design**
- Zero external dependencies (standard library only)
- C++20 concepts, `std::span`, and execution policies
- Cross-platform: Windows, macOS, Linux with automatic optimization

## Comparison with std:: Library

| Feature | std:: distributions | libstats |
|---------|-------------------|----------|
| **Random Sampling** | ✅ Excellent | ✅ Uses std:: internally |
| **PDF Evaluation** | ❌ Not available | ✅ Complete implementation |
| **CDF Evaluation** | ❌ Not available | ✅ Complete implementation |
| **Quantile Functions** | ❌ Not available | ✅ Complete implementation |
| **Parameter Fitting** | ❌ Not available | ✅ MLE with diagnostics |
| **Statistical Tests** | ❌ Not available | ✅ Comprehensive validation |
| **Thread Safety** | ⚠️ Limited | ✅ Full concurrent access |

## Examples and Tools

### 📚 **Examples** (`examples/` directory)
- `basic_usage.cpp` - Core functionality demonstration
- `statistical_validation_demo.cpp` - Advanced validation and testing
- `parallel_execution_demo.cpp` - High-performance batch processing
- Performance benchmarks for each distribution type

### 🔧 **Analysis Tools** (`tools/` directory)
- `system_inspector` - CPU capabilities and system information
- `parallel_threshold_benchmark` - Optimal parallel threshold analysis
- `performance_dispatcher_tool` - Algorithm performance comparison
- `simd_verification` - SIMD correctness and performance testing


## Testing

```bash
# Correctness suite — parallel-safe, always reliable
make run_tests                           # or: ctest -LE "timing|benchmark"

# Timing/speedup tests — run serially for accurate results
make run_tests_timing                    # or: ctest -j1 -L timing

# Everything (including dynamic linking tests)
make run_all_tests

# SIMD correctness and speedup measurement
./build/tools/simd_verification

# Run a specific test
ctest -R test_gaussian_basic
```

Tests are labelled: **no label** = correctness (parallel-safe); **timing** = speedup assertions (run serially); **benchmark** = performance tools (not in standard suite). The `*_enhanced` GTest tests require GTest installed; they are silently skipped when GTest is absent.

### System Requirements
- **C++20 compatible compiler**: GCC 10+, Clang 14+, MSVC 2019+
- **CMake**: 3.20 or later
- **Platform**: Windows, macOS, Linux (automatic detection and optimization)

#### Common Build Configurations

| Configuration | Command | Use Case |
|---------------|---------|----------|
| **Development** (default) | `cmake ..` | Daily development with light optimization |
| **Release** | `cmake -DCMAKE_BUILD_TYPE=Release ..` | Production builds with maximum optimization |
| **Debug** | `cmake -DCMAKE_BUILD_TYPE=Debug ..` | Full debugging support |

## Documentation

For complete information about libstats, refer to these comprehensive guides:

### 📖 **[BUILD_SYSTEM_GUIDE.md](docs/BUILD_SYSTEM_GUIDE.md)**
Complete build system documentation covering:
- Cross-platform build instructions (Windows, macOS, Linux)
- SIMD detection and optimization
- Parallel build configuration
- Advanced CMake options
- Troubleshooting and manual builds

### 🏗️ **[HEADER_ARCHITECTURE_GUIDE.md](docs/HEADER_ARCHITECTURE_GUIDE.md)**
Header organization and dependency management:
- Modular header architecture
- Consolidated vs individual includes
- Development patterns for distributions, tools, and tests
- Performance optimization through header design

### ⚡ **[PARALLEL_BATCH_PROCESSING_GUIDE.md](docs/PARALLEL_BATCH_PROCESSING_GUIDE.md)**
High-performance parallel and batch processing:
- Auto-dispatch vs explicit strategy control
- SIMD and parallel processing APIs
- Performance optimization guidelines
- Thread safety and memory management

### 🪟 **Windows Support**
For Windows development environment setup (MSVC activation, DLL CRT handling, Smart App Control), see the Windows session setup section in [WARP.md](WARP.md).

## Roadmap

### ✅ Core library
- 6 distributions (Gaussian, Exponential, Uniform, Poisson, Discrete, Gamma) — PDF/CDF/quantile/MLE/validation
- Thread-safe with reader-writer locks and lock-free fast paths
- SIMD batch operations (SSE2/AVX/AVX2/AVX-512/NEON) with runtime dispatch
- Work-stealing parallel thread pool
- Goodness-of-fit tests (KS, AD), information criteria (AIC/BIC), cross-validation, bootstrap

### ✅ Architecture and quality (Phases 1–4)
- Honest strategy naming: SCALAR/VECTORIZED/PARALLEL/WORK_STEALING
- Constants consolidated from 10 micro-headers to 3 semantic groups
- Corrected `vector_exp_avx` underflow bug; Gaussian CDF heap allocation removed
- Cross-platform validated: Intel AVX, Apple Silicon NEON, AMD AVX-512, MSVC/Linux CI
- All compiler warnings addressed (GCC, Clang, MSVC); zero warnings under ClangStrict
- Test labels for parallel-safe correctness runs vs timing-sensitive runs

### 🔧 In progress (Phase 5)
- Packaging: `find_package` / `FetchContent` support, pkg-config

### Planned (Phases 6–7)
- Vectorized batch ops for all 5 non-Gaussian distributions (currently scalar loops)
- New distributions: Student's t, Chi-squared, Beta


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For third-party code attributions and licenses, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgments

This project builds upon concepts and components from [libhmm](https://github.com/wolfman/libhmm), adapting them for general-purpose statistical computing while maintaining the focus on modern C++ design and performance.

Our SIMD implementations incorporate algorithms inspired by the SLEEF library for high-accuracy mathematical functions.

---

**libstats** - Bringing comprehensive statistical computing to modern C++
