# libstats - Modern C++20 Statistical Distributions Library

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-blue.svg)](https://cmake.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Safety](https://img.shields.io/badge/Memory%20Safety-Enterprise%20Grade-green.svg)](#safety-features)
[![Performance](https://img.shields.io/badge/Performance-SIMD%20%26%20Parallel-blue.svg)](#performance-features)

A modern, high-performance C++20 statistical distributions library providing comprehensive statistical functionality with enterprise-grade safety features and zero external dependencies.

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
- **Gamma**: Γ(α, β) - Positive continuous variables (in progress)

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

> **📖 SIMD Support**: For detailed information about cross-platform SIMD detection, architecture-specific behavior (x86_64 vs ARM64), and build configuration options, see [SIMD Build System Documentation](docs/simd_build_system.md).

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/libstats.git
cd libstats
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Basic Usage

```cpp
#include "libstats.h"
#include <iostream>
#include <random>

int main() {
    // Create distributions
    libstats::Gaussian normal(0.0, 1.0);
    libstats::Exponential exponential(2.0);
    
    // Statistical properties
    std::cout << "Normal mean: " << normal.getMean() << std::endl;
    std::cout << "Normal variance: " << normal.getVariance() << std::endl;
    
    // PDF evaluation
    std::cout << "P(X=1.0) for N(0,1): " << normal.getProbability(1.0) << std::endl;
    
    // CDF evaluation
    std::cout << "P(X<=1.0) for N(0,1): " << normal.getCumulativeProbability(1.0) << std::endl;
    
    // Quantiles
    std::cout << "95th percentile: " << normal.getQuantile(0.95) << std::endl;
    
    // Random sampling
    std::mt19937 rng(42);
    auto samples = normal.sample(rng, 1000);
    
    // Parameter fitting
    libstats::Gaussian fitted;
    fitted.fit(samples);
    std::cout << "Fitted mean: " << fitted.getMean() << std::endl;
    
    return 0;
}
```

## Project Structure

```
libstats/
├── include/           # Header files
│   ├── libstats.h    # Main umbrella header
│   ├── core/         # Core platform-independent library components
│   │   ├── distribution_base.h    # Enhanced base class
│   │   ├── constants.h            # Mathematical constants and tolerances
│   │   ├── math_utils.h           # Mathematical utilities with C++20 concepts
│   │   ├── safety.h               # Memory safety and numerical stability
│   │   ├── error_handling.h       # Exception-safe error handling
│   │   ├── log_space_ops.h        # Log-space mathematical operations
│   │   ├── statistical_utilities.h # Statistical helper functions
│   │   └── validation.h           # Statistical tests and validation
│   ├── distributions/ # Statistical distributions
│   │   ├── gaussian.h    # Gaussian (Normal) distribution
│   │   ├── exponential.h # Exponential distribution
│   │   ├── uniform.h     # Uniform distribution
│   │   ├── poisson.h     # Poisson distribution
│   │   ├── discrete.h    # Custom discrete distributions
│   │   └── gamma.h       # Gamma distribution (in progress)
│   ├── platform/      # Platform-specific optimizations
│   │   ├── simd.h                 # SIMD optimizations
│   │   ├── cpu_detection.h        # CPU feature detection
│   │   ├── parallel_execution.h   # C++20 parallel execution policies
│   │   ├── thread_pool.h          # Traditional thread pool
│   │   ├── work_stealing_pool.h   # Work-stealing thread pool
│   │   ├── adaptive_cache.h       # Cache-aware algorithms
│   │   ├── benchmark.h            # Performance benchmarking utilities
│   │   ├── parallel_thresholds.h  # Parallel threshold management
│   │   └── platform_constants.h   # Platform-specific constants
│   └── [compatibility headers]    # Root-level headers for backward compatibility
├── src/              # Implementation files
├── tests/            # Unit tests
├── examples/         # Usage examples
├── tools/            # Performance analysis and optimization tools
│   ├── cpu_info.cpp           # CPU feature detection and system info
│   ├── constants_inspector.cpp # Mathematical constants verification
│   ├── performance_benchmark.cpp # Comprehensive performance testing
│   ├── grain_size_optimizer.cpp # Parallel grain size optimization
│   └── parallel_threshold_benchmark.cpp # Parallel threshold analysis
├── benchmarks/       # Performance benchmarks
└── docs/             # Documentation
```

## Design Philosophy

### Modern C++20 Standards
- Complete Rule of Five implementation
- Thread-safe concurrent access
- Memory-safe resource management
- Exception-safe operations
- C++20 concepts for type safety
- `std::span` for safe array access
- `[[likely]]` and `[[unlikely]]` attributes for optimization

### Performance Focus
- Thread-safe cached statistical properties
- SIMD-optimized bulk operations
- Efficient random number generation
- Minimal memory allocations
- Work-stealing thread pools for better load balancing
- C++20 parallel algorithms with automatic fallback
- Lock-free fast paths for hot operations

### Statistical Completeness
- **Beyond std::distributions**: While C++ std:: provides excellent random sampling, libstats adds the missing statistical functionality (PDF, CDF, quantiles, fitting, validation)
- **Hybrid Approach**: Uses std:: distributions for sampling while implementing comprehensive statistical interfaces
- **Validation Framework**: Built-in goodness-of-fit testing and model diagnostics

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

## Examples

See the `examples/` directory for:

### Core Usage Examples
- **basic_usage.cpp**: Introduction to core functionality with PDF/CDF evaluation, sampling, and parameter fitting
- **statistical_validation_demo.cpp**: Advanced statistical validation including goodness-of-fit tests, cross-validation, bootstrap confidence intervals, and information criteria
- **parallel_execution_demo.cpp**: Platform-aware parallel execution with adaptive grain sizing, thread optimization, and cache-aware algorithms

### Performance Benchmarks
- **gaussian_performance_benchmark.cpp**: Comprehensive performance testing for Gaussian distribution with SIMD optimizations
- **exponential_performance_benchmark.cpp**: Performance benchmarks for exponential distribution demonstrating parallel and SIMD capabilities

### Future Examples (Planned)
- **parameter_fitting.cpp**: Advanced data fitting scenarios across multiple distributions
- **cross_distribution_demo.cpp**: Comparative analysis across different distribution types
- **simd_performance_showcase.cpp**: Cross-platform SIMD optimization demonstration

## Performance Analysis Tools

The `tools/` directory contains specialized utilities for performance analysis and optimization:

### 🔧 **System Analysis Tools**
- **cpu_info**: Comprehensive CPU feature detection and system information reporting
- **constants_inspector**: Mathematical constants verification and precision analysis

### ⚡ **Performance Optimization Tools**
- **performance_benchmark**: Multi-threaded performance testing with SIMD optimization analysis
- **grain_size_optimizer**: Automated parallel grain size optimization with efficiency analysis
- **parallel_threshold_benchmark**: Parallel vs serial threshold determination

```bash
# Run system analysis
./build/tools/cpu_info
./build/tools/constants_inspector

# Run performance optimization
./build/tools/performance_benchmark
./build/tools/grain_size_optimizer
./build/tools/parallel_threshold_benchmark
```

These tools generate CSV reports and provide detailed analysis for optimizing libstats performance on your specific hardware configuration.

## Building and Testing

### Requirements
- **C++20 compatible compiler**: GCC 10+, Clang 10+, MSVC 2019 16.11+, LLVM 20+ (recommended)
- **CMake**: 3.15 or later
- **GTest** (optional): For unit testing
- **Intel TBB** (optional): For enhanced parallel algorithm support

### Build Commands
```bash
# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..
make

# Run tests (if GTest available)
ctest --verbose

# Run examples
./examples/basic_usage
./examples/statistical_validation_demo
./examples/parallel_execution_demo
```

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

## Acknowledgments

This project builds upon concepts and components from [libhmm](https://github.com/wolfman/libhmm), adapting them for general-purpose statistical computing while maintaining the focus on modern C++ design and performance.

---

**libstats** - Bringing comprehensive statistical computing to modern C++
