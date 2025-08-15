# libstats Parallel and Batch Processing Guide

## Overview

This guide provides comprehensive information about the parallel and batch processing capabilities in libstats. The library implements a **dual API approach**: intelligent auto-dispatch for most users and explicit strategy control for power users who need precise performance control.

## Design Philosophy

### Core Principles

1. **Simplicity First**: Clean, intuitive API for 95% of use cases
2. **Power User Access**: Explicit strategy selection for advanced performance tuning
3. **Performance Preservation**: Zero regression in optimized code paths
4. **Intelligent Dispatch**: Automatic strategy selection based on data characteristics and system capabilities
5. **Thread Safety**: All batch operations are fully thread-safe

### Strategy Selection Logic

The library automatically selects the optimal processing strategy based on:

- **Data Size**: Small batches use scalar processing, large batches use parallel processing
- **System Capabilities**: CPU cores, SIMD support, memory bandwidth
- **Computational Complexity**: Simple operations favor SIMD, complex operations favor parallelism
- **Performance History**: Adaptive learning improves strategy selection over time

## Public API Reference

### Auto-Dispatch Methods (Recommended)

These methods automatically select the optimal processing strategy based on your data and system capabilities:

```cpp
// Probability Density Function (PDF)
void getProbability(std::span<const double> values, std::span<double> results,
                   const PerformanceHint& hint = {}) const;

// Log Probability Density Function (Log PDF)
void getLogProbability(std::span<const double> values, std::span<double> results,
                      const PerformanceHint& hint = {}) const;

// Cumulative Distribution Function (CDF)  
void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                             const PerformanceHint& hint = {}) const;
```

#### Performance Hints

You can provide optional hints to guide strategy selection:

```cpp
PerformanceHint hint;
hint.strategy = PerformanceHint::PreferredStrategy::AUTO;          // Default: automatic
hint.strategy = PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY;  // Favor speed
hint.strategy = PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT; // Favor parallelism
hint.strategy = PerformanceHint::PreferredStrategy::FORCE_SCALAR;   // Force scalar processing
hint.strategy = PerformanceHint::PreferredStrategy::FORCE_SIMD;     // Force SIMD processing
hint.strategy = PerformanceHint::PreferredStrategy::FORCE_PARALLEL; // Force parallel processing
```

### Explicit Strategy Methods (Power Users)

These methods give you direct control over the processing strategy:

```cpp
// Probability Density Function with explicit strategy
void getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                               libstats::performance::Strategy strategy) const;

// Log Probability Density Function with explicit strategy
void getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                  libstats::performance::Strategy strategy) const;

// Cumulative Distribution Function with explicit strategy
void getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                         libstats::performance::Strategy strategy) const;
```

#### Available Strategies

```cpp
enum class Strategy {
    SCALAR,         // Simple loop processing (best for small batches < 8 elements)
    SIMD_BATCH,     // Vectorized SIMD operations (best for medium batches 8-1000 elements)
    PARALLEL_SIMD,  // Multi-threaded parallel processing (best for large batches > 1000 elements)
    WORK_STEALING,  // Dynamic load balancing (best for irregular workloads)
    CACHE_AWARE     // Cache-optimized processing (specialized use cases)
};
```

## Usage Examples

### Basic Auto-Dispatch Usage

```cpp
#include "libstats.h"

// Create a distribution
auto result = libstats::GaussianDistribution::create(0.0, 1.0);
if (result.isOk()) {
    auto gaussian = std::move(result.value);
    
    // Prepare data
    std::vector<double> input_values(10000);
    std::vector<double> pdf_results(10000);
    std::vector<double> cdf_results(10000);
    
    // Fill input with test data
    std::iota(input_values.begin(), input_values.end(), -5.0);
    
    // Auto-dispatch will select optimal strategy (likely PARALLEL_SIMD for 10k elements)
    gaussian.getProbability(std::span<const double>(input_values), 
                           std::span<double>(pdf_results));
    
    gaussian.getCumulativeProbability(std::span<const double>(input_values), 
                                     std::span<double>(cdf_results));
    
    std::cout << "Processed " << input_values.size() << " values" << std::endl;
}
```

### Performance-Guided Usage

```cpp
// For latency-sensitive applications
PerformanceHint low_latency;
low_latency.strategy = PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY;

gaussian.getProbability(std::span<const double>(small_batch), 
                       std::span<double>(results), low_latency);

// For throughput-focused applications
PerformanceHint high_throughput;
high_throughput.strategy = PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT;

gaussian.getProbability(std::span<const double>(large_batch), 
                       std::span<double>(results), high_throughput);
```

### Explicit Strategy Control

```cpp
// Power user: explicitly control the processing strategy
using Strategy = libstats::performance::Strategy;

std::vector<double> input(5000);
std::vector<double> results(5000);

// Force scalar processing (useful for debugging or comparison)
gaussian.getProbabilityWithStrategy(std::span<const double>(input), 
                                   std::span<double>(results), Strategy::SCALAR);

// Force parallel processing (useful when you know your data characteristics)
gaussian.getProbabilityWithStrategy(std::span<const double>(input), 
                                   std::span<double>(results), Strategy::PARALLEL_SIMD);

// Use work-stealing for irregular workloads
gaussian.getProbabilityWithStrategy(std::span<const double>(input), 
                                   std::span<double>(results), Strategy::WORK_STEALING);
```

### Performance Benchmarking

```cpp
#include <chrono>

// Benchmark different strategies
auto benchmark_strategy = [&](Strategy strategy, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    
    gaussian.getProbabilityWithStrategy(std::span<const double>(test_data), 
                                       std::span<double>(results), strategy);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << ": " << duration.count() << " μs" << std::endl;
};

benchmark_strategy(Strategy::SCALAR, "Scalar");
benchmark_strategy(Strategy::SIMD_BATCH, "SIMD");
benchmark_strategy(Strategy::PARALLEL_SIMD, "Parallel");
benchmark_strategy(Strategy::WORK_STEALING, "Work-Stealing");
```

## Performance Characteristics

### Strategy Selection Guidelines

| Batch Size | Recommended Strategy | Rationale |
|------------|---------------------|-----------|
| 1-8 elements | `SCALAR` | Loop overhead dominates, simple iteration is fastest |
| 8-100 elements | `SIMD_BATCH` | Vectorization provides significant speedup |
| 100-1000 elements | `SIMD_BATCH` | SIMD still optimal, parallelization overhead too high |
| 1000-10000 elements | `PARALLEL_SIMD` | Multi-threading becomes beneficial |
| 10000+ elements | `PARALLEL_SIMD` or `WORK_STEALING` | Full parallelization provides best throughput |

### Expected Performance Gains

**SIMD Batch Processing:**
- **Uniform/Discrete**: 20-70x speedup (simple range checks vectorize extremely well)
- **Gaussian/Exponential**: 2-8x speedup (mathematical functions benefit moderately)
- **Gamma/Poisson**: 1.5-4x speedup (complex mathematical operations)

**Parallel Processing:**
- **Linear Scaling**: Up to N× speedup where N = number of CPU cores
- **Memory-Bound**: 2-4× speedup (limited by memory bandwidth)
- **CPU-Bound**: 4-16× speedup (scales with core count)

**Work-Stealing:**
- **Irregular Workloads**: 10-30% better than standard parallel
- **Uniform Workloads**: Similar to standard parallel
- **Mixed Data**: Excellent load balancing for heterogeneous inputs

## Advanced Features

### System Capability Detection

The library automatically detects and uses available system capabilities:

```cpp
// Query detected capabilities (for informational purposes)
std::cout << "SIMD level: " << libstats::cpu::best_simd_level() << std::endl;
std::cout << "Vector width: " << libstats::cpu::optimal_double_width() << std::endl;
std::cout << "CPU cores: " << std::thread::hardware_concurrency() << std::endl;
```

### Performance System Initialization

For optimal performance in production applications:

```cpp
int main() {
    // Initialize performance systems to eliminate cold-start delays
    libstats::initialize_performance_systems();
    
    // Your application code here...
    auto gaussian = libstats::GaussianDistribution::create(0.0, 1.0);
    // Batch operations will have optimal performance from the start
}
```

### Thread Safety Guarantees

All batch processing methods are **fully thread-safe**:

```cpp
// Safe to call from multiple threads simultaneously
std::vector<std::thread> threads;
for (int i = 0; i < std::thread::hardware_concurrency(); ++i) {
    threads.emplace_back([&gaussian, i]() {
        std::vector<double> local_input(1000);
        std::vector<double> local_results(1000);
        
        // Fill with thread-specific data
        std::iota(local_input.begin(), local_input.end(), i * 1000.0);
        
        // Thread-safe batch processing
        gaussian.getProbability(std::span<const double>(local_input),
                               std::span<double>(local_results));
    });
}

for (auto& t : threads) {
    t.join();
}
```

## Distribution-Specific Considerations

### Discrete Distributions (Discrete, Poisson)

- **Strengths**: Extremely fast SIMD processing due to simple integer operations
- **Cache Behavior**: Excellent cache locality for repeated discrete values
- **Optimal Strategy**: SIMD_BATCH for most batch sizes

```cpp
PoissonDistribution poisson(3.0);
std::vector<double> discrete_values = {0, 1, 2, 3, 4, 5, 1, 2, 3, 4}; // Repeated values
std::vector<double> probabilities(discrete_values.size());

// SIMD_BATCH will be extremely fast due to cache-friendly discrete lookups
poisson.getProbability(std::span<const double>(discrete_values),
                       std::span<double>(probabilities));
```

### Continuous Distributions (Gaussian, Exponential, Uniform, Gamma)

- **Mathematical Complexity**: Varies by distribution type
- **Memory Patterns**: Sequential access patterns optimize well
- **Optimal Strategy**: Depends on computational complexity

```cpp
// Uniform: Extremely simple (range checks) - SIMD dominates
UniformDistribution uniform(0.0, 1.0);

// Gaussian: Moderate complexity - balanced SIMD/Parallel trade-off  
GaussianDistribution gaussian(0.0, 1.0);

// Gamma: High complexity - Parallel often better than SIMD
GammaDistribution gamma(2.0, 1.5);
```

## Error Handling and Edge Cases

### Input Validation

```cpp
// The library validates inputs automatically
std::vector<double> input(1000);
std::vector<double> results(500);  // Wrong size!

try {
    gaussian.getProbability(std::span<const double>(input),
                           std::span<double>(results));
} catch (const std::invalid_argument& e) {
    std::cout << "Error: " << e.what() << std::endl;
    // "Input and output spans must have the same size"
}
```

### Empty Input Handling

```cpp
// Empty inputs are handled gracefully
std::vector<double> empty_input;
std::vector<double> empty_results;

// No-op, returns immediately
gaussian.getProbability(std::span<const double>(empty_input),
                       std::span<double>(empty_results));
```

### Exception-Free Alternative

```cpp
// Use safe factory methods for exception-free operation
auto safe_gaussian = libstats::GaussianDistribution::create(mean, std_dev);
if (safe_gaussian.isOk()) {
    // Process successfully
    safe_gaussian.value.getProbability(input_span, results_span);
} else {
    std::cout << "Error: " << safe_gaussian.message << std::endl;
}
```

## Performance Optimization Tips

### 1. Choose Appropriate Data Sizes

```cpp
// Avoid very small batches in loops - combine when possible
// ❌ Inefficient: Many small batch calls
for (const auto& small_batch : many_small_batches) {
    distribution.getProbability(small_batch, results);
}

// ✅ Efficient: Single large batch call
std::vector<double> combined_input;
for (const auto& batch : many_small_batches) {
    combined_input.insert(combined_input.end(), batch.begin(), batch.end());
}
distribution.getProbability(std::span<const double>(combined_input), combined_results);
```

### 2. Reuse Memory Allocations

```cpp
// ✅ Reuse vectors to avoid allocation overhead
std::vector<double> reusable_input(max_batch_size);
std::vector<double> reusable_results(max_batch_size);

for (size_t batch_start = 0; batch_start < total_size; batch_start += batch_size) {
    size_t current_batch_size = std::min(batch_size, total_size - batch_start);
    
    // Resize without reallocation (if capacity is sufficient)
    reusable_input.resize(current_batch_size);
    reusable_results.resize(current_batch_size);
    
    // Fill input data...
    
    distribution.getProbability(std::span<const double>(reusable_input),
                               std::span<double>(reusable_results));
}
```

### 3. Use Performance Hints Appropriately

```cpp
// For real-time applications where latency matters
PerformanceHint real_time;
real_time.strategy = PerformanceHint::PreferredStrategy::MINIMIZE_LATENCY;

// For batch processing where throughput matters
PerformanceHint batch_processing;
batch_processing.strategy = PerformanceHint::PreferredStrategy::MAXIMIZE_THROUGHPUT;
```

### 4. Profile Your Specific Use Case

```cpp
// Benchmark your specific data patterns and sizes
auto profile_batch_sizes = [&](const std::vector<size_t>& sizes) {
    for (size_t size : sizes) {
        std::vector<double> test_input(size);
        std::vector<double> test_results(size);
        
        // Fill with representative data...
        
        auto start = std::chrono::high_resolution_clock::now();
        distribution.getProbability(std::span<const double>(test_input),
                                   std::span<double>(test_results));
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double throughput = static_cast<double>(size) / duration.count(); // elements per μs
        
        std::cout << "Size " << size << ": " << throughput << " elements/μs" << std::endl;
    }
};

profile_batch_sizes({100, 1000, 10000, 100000});
```

## Migration from Legacy APIs

### From Deprecated Batch Methods

If you're upgrading from older versions of libstats:

```cpp
// ❌ Old deprecated API (removed)
// distribution.getProbabilityBatch(values, results, count);

// ✅ New auto-dispatch API
distribution.getProbability(std::span<const double>(values, count),
                           std::span<double>(results, count));

// ✅ Or explicit strategy API
distribution.getProbabilityWithStrategy(std::span<const double>(values, count),
                                       std::span<double>(results, count),
                                       libstats::performance::Strategy::SIMD_BATCH);
```

### From Individual Function Calls

```cpp
// ❌ Inefficient: Individual calls in a loop
std::vector<double> results(input.size());
for (size_t i = 0; i < input.size(); ++i) {
    results[i] = distribution.getProbability(input[i]);
}

// ✅ Efficient: Single batch call
distribution.getProbability(std::span<const double>(input),
                           std::span<double>(results));
```

## Debugging and Diagnostics

### Strategy Selection Verification

```cpp
// To understand which strategy was selected, use explicit strategy methods for comparison
std::vector<double> input(5000);
std::vector<double> auto_results(5000);
std::vector<double> explicit_results(5000);

// Auto-dispatch
auto auto_start = std::chrono::high_resolution_clock::now();
distribution.getProbability(std::span<const double>(input), std::span<double>(auto_results));
auto auto_end = std::chrono::high_resolution_clock::now();
auto auto_time = std::chrono::duration_cast<std::chrono::microseconds>(auto_end - auto_start);

// Try different explicit strategies to see which matches auto-dispatch performance
auto test_strategy = [&](libstats::performance::Strategy strategy, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    distribution.getProbabilityWithStrategy(std::span<const double>(input), 
                                           std::span<double>(explicit_results), strategy);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << name << ": " << time.count() << " μs";
    if (std::abs(time.count() - auto_time.count()) < auto_time.count() * 0.1) {
        std::cout << " ← Likely auto-selected strategy";
    }
    std::cout << std::endl;
};

std::cout << "Auto-dispatch: " << auto_time.count() << " μs" << std::endl;
test_strategy(libstats::performance::Strategy::SCALAR, "SCALAR");
test_strategy(libstats::performance::Strategy::SIMD_BATCH, "SIMD_BATCH");
test_strategy(libstats::performance::Strategy::PARALLEL_SIMD, "PARALLEL_SIMD");
```

### Correctness Verification

```cpp
// Verify that all strategies produce identical results
bool verify_strategies(const std::vector<double>& input) {
    std::vector<double> scalar_results(input.size());
    std::vector<double> simd_results(input.size());
    std::vector<double> parallel_results(input.size());
    
    distribution.getProbabilityWithStrategy(std::span<const double>(input), 
                                           std::span<double>(scalar_results), 
                                           libstats::performance::Strategy::SCALAR);
    
    distribution.getProbabilityWithStrategy(std::span<const double>(input), 
                                           std::span<double>(simd_results), 
                                           libstats::performance::Strategy::SIMD_BATCH);
    
    distribution.getProbabilityWithStrategy(std::span<const double>(input), 
                                           std::span<double>(parallel_results), 
                                           libstats::performance::Strategy::PARALLEL_SIMD);
    
    // Check for numerical differences
    for (size_t i = 0; i < input.size(); ++i) {
        if (std::abs(scalar_results[i] - simd_results[i]) > 1e-12 ||
            std::abs(scalar_results[i] - parallel_results[i]) > 1e-12) {
            std::cout << "Mismatch at index " << i << ": scalar=" << scalar_results[i] 
                      << ", simd=" << simd_results[i] << ", parallel=" << parallel_results[i] << std::endl;
            return false;
        }
    }
    
    return true;
}
```

## Conclusion

The libstats parallel and batch processing system provides:

1. **Simple Interface**: Auto-dispatch handles optimization automatically for most users
2. **Power User Control**: Explicit strategy selection for performance-critical applications
3. **Excellent Performance**: Significant speedups through SIMD and parallel processing
4. **Thread Safety**: Full concurrent access support
5. **Cross-Platform**: Works on Windows, macOS, and Linux with automatic capability detection

For most applications, the auto-dispatch methods provide optimal performance with minimal complexity. Power users can leverage explicit strategy control for fine-tuned performance optimization in specialized scenarios.

The system is designed to scale from small embedded applications to high-performance computing workloads, automatically adapting to available system resources while maintaining mathematical accuracy and thread safety.

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-13  
**Covers**: Complete parallel/batch processing API and usage patterns  
**Replaces**: `batch-processing-refactoring.md`, `deprecated_batch_cleanup_process.md`
