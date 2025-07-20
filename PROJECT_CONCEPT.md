# Statistical Distributions Library Project - libstats

## SIMD Support and CPU Feature Detection

Our library supports both compile-time and runtime CPU feature detection to optimally use SIMD instructions without assuming the capabilities of the target machine.

### How to Include SIMD and CPU Detection

1. **Compile-Time Detection:**
   - Include `simd.h` to use compile-time detection.
   - Use feature macros like `LIBSTATS_HAS_AVX` to conditionally compile SIMD code.

2. **Runtime Detection:**
   - Include `cpu_detection.h` to use runtime detection.
   - Use functions like `supports_avx()` to check for runtime support before executing SIMD routines.

### Integration

- Ensure your code includes both `simd.h` for determining compile-time capabilities and `cpu_detection.h` for runtime detection.
- Call appropriate detection functions to ensure safe execution of SIMD code paths based on actual support.
- Example:
  ```cpp
  #include "libstats/performance/simd.h"
  #include "libstats/cpu/cpu_detection.h"
  
  if (libstats::cpu::supports_avx()) {
      // Execute AVX code path
  } else {
      // Fallback to non-AVX path
  }
  ```

## Project Overview

Creating a focused statistical distributions library from libhmm would be **very feasible** and **highly valuable**. The enhanced base class design provides an excellent foundation, and many libhmm components can be reused.

### Name: libstats
**Core Value Proposition:**
- **Complete statistical interface** (PDF, CDF, quantiles, moments)
- **Random sampling** using std:: distributions
- **Parameter estimation** with MLE fitting
- **Statistical validation** (goodness-of-fit tests)
- **Modern C++20** with zero dependencies
- **Thread-safe** concurrent access
- **SIMD optimization** where applicable

## Project Structure

```
libstats/
├── include/
│   ├── libstats.h                    # Main umbrella header
│   ├── distribution_base.h           # Enhanced base class
│   ├── gaussian.h                    # Normal distribution
│   ├── exponential.h                 # Exponential distribution
│   ├── gamma.h                       # Gamma distribution
│   ├── poisson.h                     # Poisson distribution
│   ├── uniform.h                     # Uniform distribution
│   ├── constants.h                   # Mathematical constants and precision tolerances
│   ├── math_utils.h                  # Mathematical utilities with C++20 concepts
│   ├── safety.h                      # Memory safety and numerical stability
│   ├── simd.h                        # SIMD utilities and optimizations
│   ├── parallel_execution.h          # C++20 parallel execution policies
│   ├── thread_pool.h                 # Traditional thread pool
│   ├── work_stealing_pool.h          # Work-stealing thread pool
│   └── validation.h                  # Statistical tests
├── src/
│   ├── distribution_base.cpp         # Base class implementation
│   ├── gaussian.cpp                  # Gaussian implementation
│   ├── exponential.cpp               # Exponential implementation
│   ├── gamma.cpp                     # Gamma implementation
│   ├── poisson.cpp                   # Poisson implementation
│   ├── uniform.cpp                   # Uniform implementation
│   ├── constants.cpp                 # Mathematical constants (if needed)
│   ├── math_utils.cpp                # Mathematical utility implementations
│   ├── safety.cpp                    # Safety utility implementations
│   ├── simd.cpp                      # SIMD implementations
│   ├── parallel_execution.cpp        # Parallel execution implementations
│   ├── thread_pool.cpp               # Thread pool implementations
│   ├── work_stealing_pool.cpp        # Work-stealing pool implementations
│   └── validation.cpp                # Statistical validation
├── tests/
│   ├── test_gaussian.cpp             # Gaussian tests
│   ├── test_exponential.cpp          # Exponential tests
│   ├── test_gamma.cpp                # Gamma tests
│   ├── test_poisson.cpp              # Poisson tests
│   ├── test_uniform.cpp              # Uniform tests
│   ├── test_validation.cpp           # Validation tests
│   └── test_main.cpp                 # Test runner
├── examples/
│   ├── basic_usage.cpp               # Simple examples
│   ├── parameter_fitting.cpp         # Fitting examples
│   ├── validation_demo.cpp           # Validation examples
│   └── performance_demo.cpp          # SIMD performance
├── benchmarks/
│   ├── distribution_performance.cpp  # Speed benchmarks
│   └── memory_usage.cpp              # Memory benchmarks
├── docs/
│   ├── README.md                     # Getting started
│   ├── API_REFERENCE.md              # Complete API docs
│   └── EXAMPLES.md                   # Usage examples
├── CMakeLists.txt                    # Main CMake file
├── README.md                         # Project overview
└── LICENSE                           # MIT license
```

## Reusable Components from libhmm

### 1. **Common Utilities** (`libhmm/common/`)
```cpp
// From libhmm/common/common.h
namespace libstats {
    // Mathematical constants
    constexpr double PI = 3.14159265358979323846;
    constexpr double E = 2.71828182845904523536;
    constexpr double SQRT_2PI = 2.50662827463100050242;
    constexpr double LOG_2PI = 1.83787706640934548356;
    
    // Numerical precision
    constexpr double EPSILON = 1e-15;
    constexpr double TOLERANCE = 1e-12;
    
    // Special values for probability calculations
    constexpr double ZERO_PROBABILITY = 1e-100;
    constexpr double LOG_ZERO = -230.258509299;  // log(1e-100)
}
```

### 2. **SIMD Support** (`libhmm/performance/`)
```cpp
// From libhmm/performance/simd_platform_detection.h
#include "simd.h"

namespace libstats::simd {
    // CPU feature detection
    bool hasAVX();
    bool hasSSE2();
    bool hasNEON();
    
    // Vectorized operations for bulk PDF/CDF calculations
    void vectorized_gaussian_pdf(const double* x, double* result, 
                                size_t n, double mu, double sigma);
    void vectorized_exponential_cdf(const double* x, double* result,
                                   size_t n, double lambda);
}
```

### 3. **Thread Safety** (`libhmm/distributions/`)
```cpp
// From libhmm's existing thread-safe cache management
#include <shared_mutex>

class DistributionBase {
protected:
    mutable std::shared_mutex cache_mutex_;
    mutable bool cache_valid_{false};
    
    template<typename Func>
    auto getCachedValue(Func&& accessor) const -> decltype(accessor());
};
```

## Initial Distribution Set (5 Distributions)

### Priority Order for Implementation:

#### 1. **Gaussian Distribution** (Most Important)
```cpp
// include/gaussian.h
#pragma once
#include "distribution_base.h"

namespace libstats {
    class GaussianDistribution : public DistributionBase {
    private:
        double mu_{0.0};      // Mean
        double sigma_{1.0};   // Standard deviation
        
    public:
        // Constructors
        GaussianDistribution(double mu = 0.0, double sigma = 1.0);
        
        // Distribution-specific interface
        double getMu() const { return mu_; }
        void setMu(double mu) { mu_ = mu; invalidateCache(); }
        double getSigma() const { return sigma_; }
        void setSigma(double sigma);
        double getSigmaSquared() const { return sigma_ * sigma_; }
        
        // Pure virtual implementations
        double getProbability(double x) const override;
        double getCumulativeProbability(double x) const override;
        double getQuantile(double p) const override;
        double sample(std::mt19937& rng) const override;
        void fit(const std::vector<double>& data) override;
        
        // Statistical properties
        double getMean() const override { return mu_; }
        double getVariance() const override { return sigma_ * sigma_; }
        double getSkewness() const override { return 0.0; }
        double getKurtosis() const override { return 0.0; }
        
        // Metadata
        std::string getDistributionName() const override { return "Gaussian"; }
        int getNumParameters() const override { return 2; }
        bool isDiscrete() const override { return false; }
        double getSupportLowerBound() const override { return -std::numeric_limits<double>::infinity(); }
        double getSupportUpperBound() const override { return std::numeric_limits<double>::infinity(); }
    };
}
```

#### 2. **Exponential Distribution** (Simple, Good Example)
#### 3. **Uniform Distribution** (Simple CDF/Quantile)
#### 4. **Poisson Distribution** (Discrete Example)
#### 5. **Gamma Distribution** (More Complex, Good Test Case)

## CMake Configuration

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(libstats VERSION 1.0.0 LANGUAGES CXX)

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Library target
add_library(libstats
    src/distribution_base.cpp
    src/gaussian.cpp
    src/exponential.cpp
    src/uniform.cpp
    src/poisson.cpp
    src/gamma.cpp
    src/constants.cpp
    src/math_utils.cpp
    src/safety.cpp
    src/simd.cpp
    src/parallel_execution.cpp
    src/thread_pool.cpp
    src/work_stealing_pool.cpp
    src/validation.cpp
)

target_include_directories(libstats PUBLIC include)

# Optional: Header-only version
add_library(libstats_header_only INTERFACE)
target_include_directories(libstats_header_only INTERFACE include)

# Tests
find_package(GTest QUIET)
if(GTest_FOUND)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
add_subdirectory(examples)

# Install
install(TARGETS libstats DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/libstats)
```

## Main Header (libstats.h)

```cpp
// include/libstats.h
#pragma once

/**
 * @file libstats.h
 * @brief Modern C++20 statistical distributions library
 * 
 * Provides comprehensive statistical interface with:
 * - PDF, CDF, and quantile functions
 * - Random sampling using std:: distributions
 * - Parameter estimation with MLE
 * - Statistical validation and diagnostics
 * - Thread-safe concurrent access
 * - SIMD optimization
 * - Zero external dependencies
 */

// Core framework
#include "distribution_base.h"
#include "constants.h"
#include "math_utils.h"
#include "safety.h"
#include "validation.h"

// Performance and parallelization
#include "simd.h"
#include "parallel_execution.h"
#include "thread_pool.h"
#include "work_stealing_pool.h"

// Distributions
#include "gaussian.h"
#include "exponential.h"
#include "uniform.h"
#include "poisson.h"
#include "gamma.h"

// Convenience namespace
namespace libstats {
    // Type aliases for common usage
    using Gaussian = GaussianDistribution;
    using Normal = GaussianDistribution;
    using Exponential = ExponentialDistribution;
    using Uniform = UniformDistribution;
    using Poisson = PoissonDistribution;
    using Gamma = GammaDistribution;
    
    // Version information
    constexpr int VERSION_MAJOR = 1;
    constexpr int VERSION_MINOR = 0;
    constexpr int VERSION_PATCH = 0;
    constexpr const char* VERSION_STRING = "1.0.0";
}
```

## Example Usage

```cpp
// examples/basic_usage.cpp
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
    std::cout << "95th percentile of N(0,1): " << normal.getQuantile(0.95) << std::endl;
    
    // Random sampling
    std::mt19937 rng(42);
    auto samples = normal.sample(rng, 1000);
    
    // Parameter fitting
    normal.fit(samples);
    std::cout << "Fitted mean: " << normal.getMean() << std::endl;
    
    // Validation
    auto validation = normal.validate(samples);
    std::cout << "KS p-value: " << validation.ks_p_value << std::endl;
    
    return 0;
}
```

## Migration Strategy from libhmm

### Phase 1: Core Infrastructure (Week 1-2)
1. **Extract reusable components** from libhmm
2. **Implement enhanced base class** with thread safety
3. **Create build system** and basic project structure
4. **Implement Gaussian distribution** as proof of concept

### Phase 2: Basic Distribution Set (Week 3-4)
1. **Add Exponential, Uniform distributions** (simple cases)
2. **Implement statistical validation** framework
3. **Add comprehensive tests** for all distributions
4. **Create usage examples** and documentation

### Phase 3: Advanced Features (Week 5-6)
1. **Add Poisson and Gamma distributions** (more complex)
2. **Implement SIMD optimizations** for bulk operations
3. **Add parameter fitting diagnostics** (AIC/BIC)
4. **Performance benchmarking** and optimization

### Phase 4: Polish and Release (Week 7-8)
1. **Complete documentation** and API reference
2. **Package for distribution** (header-only option)
3. **Integration testing** with real datasets
4. **Performance comparison** with other libraries

## Benefits of This Approach

### **Technical Benefits**
- **Focused scope**: Statistical distributions only, not HMM-specific
- **Reusable foundation**: Can be used in many statistical applications
- **Modern design**: C++17 best practices, thread safety, SIMD
- **Zero dependencies**: Only standard library required

### **Strategic Benefits**
- **Faster development**: Reuses proven libhmm components
- **Market validation**: Addresses clear gap in C++ ecosystem
- **Stepping stone**: Could lead to more comprehensive statistical library
- **Community building**: Attracts users interested in C++ statistics

### **Development Benefits**
- **Simple structure**: Flat hierarchy, easy to navigate
- **Clear separation**: Each distribution is self-contained
- **Testable**: Each component can be tested independently
- **Extensible**: Easy to add new distributions following the pattern

This focused statistical distributions library would be **much easier to develop** than enhancing all of libhmm, while providing **immediate value** to the C++ statistical computing community. The flat structure keeps complexity manageable while the enhanced base class provides a solid foundation for future growth.

---

*Document created: 2025-01-10*  
*Version: 1.0*  
*Author: AI Assistant*  
*Status: Project Concept Document*
