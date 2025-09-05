# libstats - Modern C++20 Statistical Distributions Library

[![CI](https://github.com/OldCrow/libstats/actions/workflows/ci.yml/badge.svg)](https://github.com/OldCrow/libstats/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/OldCrow/libstats/graph/badge.svg)](https://codecov.io/gh/OldCrow/libstats)
[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.20%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety](https://img.shields.io/badge/Memory%20Safety-Enterprise%20Grade-green.svg)](#safety-features)
[![Performance](https://img.shields.io/badge/Performance-SIMD%20%26%20Parallel-blue.svg)](#performance-features)

A modern, high-performance C++20 statistical distributions library providing comprehensive statistical functionality with enterprise-grade safety features and zero external dependencies.

**📖 Complete Documentation:** For detailed information about building, architecture, parallel processing, and platform support, see the [comprehensive guides](#documentation) below.

## Features

### 🎯 **Complete Statistical Interface**
- **PDF/CDF/Quantiles**: Full probability density, cumulative distribution, and quantile functions
- **Statistical Moments**: Mean, variance, skewness, kurtosis with thread-safe access
- **Random Sampling**: Integration with std:: distributions for high-quality random number generation
- **Parameter Estimation**: Maximum Likelihood Estimation (MLE) with comprehensive diagnostics
- **Ststistical Validation**: KS and AD Goodness-of-Fit, model selection

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

**📖 Cross-Platform SIMD Support**: Automatic detection and optimization for SSE2/AVX/AVX2/NEON instruction sets with runtime safety verification.

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
# Run all tests
ctest --output-on-failure

# Run specific test categories
ctest -R "test_gaussian"
ctest -R "test_performance"

# Run examples
./examples/basic_usage
./examples/parallel_execution_demo
```

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

### 🪟 **[WINDOWS_SUPPORT_GUIDE.md](docs/WINDOWS_SUPPORT_GUIDE.md)**
Windows development environment support:
- Visual Studio and MSVC configuration
- Windows-specific SIMD optimization
- Build instructions for Windows platforms

## Roadmap

### Phase 1: Core Infrastructure ✅
- [x] Enhanced base class with thread safety
- [x] Basic distribution set (5 distributions)
- [x] Build system and project structure
- [x] C++20 upgrade with concepts and spans
- [x] Memory safety and numerical stability framework
- [x] Parallel processing capabilities (traditional and work-stealing)
- [x] Enterprise-grade safety features

### Phase 2: Statistical Validation ✅
- [x] Goodness-of-fit tests (KS, AD, Chi-squared)
- [x] Information criteria (AIC/BIC)
- [x] Residual analysis
- [x] Cross-validation framework
- [x] Bootstrap confidence intervals

### Phase 3: Performance Optimization ✅
- [x] SIMD bulk operations with cross-platform detection
- [x] Parallel algorithm implementations
- [x] Performance benchmarking tools
- [x] Grain size optimization tools
- [x] CPU feature detection and adaptive constants

### Phase 4: Tools and Utilities ✅
- [x] Comprehensive performance benchmarks
- [x] CPU information and feature detection tools
- [x] Constants inspector for mathematical verification
- [x] Grain size optimizer for parallel performance tuning
- [x] Parallel threshold benchmarking

### Phase 5: Optimization and Cross-Platform Tuning (In Progress) 🔧
- [x] Core performance analysis tools delivered
- [x] Parallel optimization with grain size tuning
- [x] SIMD acceleration with runtime detection
- [ ] Cross-platform testing (Linux, Windows)
- [ ] Compiler compatibility testing (GCC, MSVC)
- [ ] Memory usage optimization
- [ ] Cache efficiency improvements
- [ ] Build system packaging

### Phase 6: Future Enhancements (Planned)
- [ ] Additional distributions (Beta, Chi-squared, Student's t)
- [ ] Automatic distribution selection
- [ ] Comprehensive API documentation
- [ ] Real-world usage examples
- [ ] Header-only distribution option

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
