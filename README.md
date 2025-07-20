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

### 📊 **Available Distributions**
- **Gaussian (Normal)**: N(μ, σ²) - The cornerstone of statistics
- **Exponential**: Exp(λ) - Waiting times and reliability analysis
- **Uniform**: U(a, b) - Continuous uniform random variables
- **Poisson**: P(λ) - Count data and rare events
- **Gamma**: Γ(α, β) - Positive continuous variables

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

### 🧪 **Statistical Validation** (Planned)
- **Goodness-of-Fit Tests**: Kolmogorov-Smirnov, Anderson-Darling
- **Model Selection**: AIC/BIC information criteria
- **Residual Analysis**: Standardized residuals and diagnostics
- **Cross-Validation**: K-fold validation framework

### 🚀 **Performance Features**
- **SIMD Operations**: Vectorized statistical computations
- **Parallel Processing**: Both traditional and work-stealing thread pools
- **C++20 Parallel Algorithms**: Safe wrappers for `std::execution` policies
- **Cache Optimization**: Thread-safe caching with lock-free fast paths

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
│   ├── distribution_base.h  # Enhanced base class
│   ├── gaussian.h    # Gaussian distribution
│   ├── exponential.h # Exponential distribution
│   ├── uniform.h     # Uniform distribution
│   ├── poisson.h     # Poisson distribution
│   ├── gamma.h       # Gamma distribution
│   ├── constants.h   # Mathematical constants and tolerances
│   ├── math_utils.h  # Mathematical utilities with C++20 concepts
│   ├── safety.h      # Memory safety and numerical stability
│   ├── simd.h        # SIMD optimizations
│   ├── parallel_execution.h # C++20 parallel execution policies
│   ├── thread_pool.h # Traditional thread pool
│   ├── work_stealing_pool.h # Work-stealing thread pool
│   └── validation.h  # Statistical tests
├── src/              # Implementation files
├── tests/            # Unit tests
├── examples/         # Usage examples
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
- **basic_usage.cpp**: Introduction to core functionality
- **parameter_fitting.cpp**: Data fitting and validation
- **validation_demo.cpp**: Statistical testing examples
- **performance_demo.cpp**: SIMD optimization showcase

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

### Phase 2: Statistical Validation (In Progress)
- [ ] Goodness-of-fit tests (KS, AD, Chi-squared)
- [ ] Information criteria (AIC/BIC)
- [ ] Residual analysis

### Phase 3: Advanced Features (Planned)
- [ ] Additional distributions (Beta, Chi-squared, Student's t)
- [ ] SIMD bulk operations
- [ ] Automatic distribution selection

### Phase 4: Polish and Documentation (Planned)
- [ ] Comprehensive API documentation
- [ ] Performance benchmarks
- [ ] Real-world usage examples

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project builds upon concepts and components from [libhmm](https://github.com/wolfman/libhmm), adapting them for general-purpose statistical computing while maintaining the focus on modern C++ design and performance.

---

**libstats** - Bringing comprehensive statistical computing to modern C++
