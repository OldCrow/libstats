# Derived Class Implementation Guide

This guide provides detailed instructions for implementing distribution classes that inherit from `DistributionBase`, with proper integration of Level 0 and Level 1 utilities.

## Table of Contents

1. [Quick Start Template](#quick-start-template)
2. [Level 0 Utilities Integration](#level-0-utilities-integration)
3. [Level 1 Utilities Integration](#level-1-utilities-integration)
4. [Thread Parallelism Patterns](#thread-parallelism-patterns)
5. [Thread Safety Implementation](#thread-safety-implementation)
6. [Performance Optimization](#performance-optimization)
7. [Enhanced Caching Strategies](#enhanced-caching-strategies)
8. [Memory Optimization Features](#memory-optimization-features)
9. [SIMD Batch Operations](#simd-batch-operations)
10. [Error Handling Patterns](#error-handling-patterns)
11. [Testing Guidelines](#testing-guidelines)
12. [Common Pitfalls](#common-pitfalls)

## Quick Start Template

Here's a complete template for implementing a new distribution:

```cpp
#ifndef LIBSTATS_MY_DISTRIBUTION_H_
#define LIBSTATS_MY_DISTRIBUTION_H_

#include "distribution_base.h"
#include "constants.h"
#include "error_handling.h"
#include "safety.h"
#include "simd.h"
#include "cpu_detection.h"
#include "math_utils.h"
#include "log_space_ops.h"

namespace libstats {

class MyDistribution : public DistributionBase {
public:
    // Safe factory pattern (recommended)
    static Result<MyDistribution> create(double param1, double param2);
    
    // Rule of Five
    MyDistribution(const MyDistribution& other);
    MyDistribution(MyDistribution&& other) noexcept;
    MyDistribution& operator=(const MyDistribution& other);
    MyDistribution& operator=(MyDistribution&& other) noexcept;
    ~MyDistribution() = default;

    // Core probability interface
    double getProbability(double x) const override;
    double getLogProbability(double x) const override;
    double getCumulativeProbability(double x) const override;
    double getQuantile(double p) const override;

    // Statistical moments
    double getMean() const override;
    double getVariance() const override;
    double getSkewness() const override;
    double getKurtosis() const override;

    // Random sampling
    double sample(std::mt19937& rng) const override;
    std::vector<double> sample(std::mt19937& rng, size_t n) const override;

    // Parameter estimation
    void fit(const std::vector<double>& data) override;
    void reset() noexcept override;

    // Metadata
    int getNumParameters() const override { return 2; }
    std::string getDistributionName() const override { return "MyDistribution"; }
    std::string toString() const override;

    // Distribution properties
    bool isDiscrete() const override { return false; }
    double getSupportLowerBound() const override;
    double getSupportUpperBound() const override;

    // Parameter accessors
    double getParam1() const;
    double getParam2() const;
    
    // Lock-free atomic parameter getters (high performance)
    double getParam1Atomic() const noexcept;
    double getParam2Atomic() const noexcept;
    
    // Thread-safe parameter setters with atomic invalidation
    VoidResult setParam1(double value);
    VoidResult setParam2(double value);
    VoidResult setParameters(double param1, double param2);

    // SIMD batch operations
    std::vector<double> getBatchProbabilities(const std::vector<double>& x_values) const;
    std::vector<double> getBatchLogProbabilities(const std::vector<double>& x_values) const;
    std::vector<double> getBatchCDF(const std::vector<double>& x_values) const;

protected:
    void updateCacheUnsafe() const override;

private:
    // Private constructor for factory pattern
    MyDistribution(double param1, double param2);
    
    // CORE DISTRIBUTION PARAMETERS
    double param1_;
    double param2_;
    
    // ATOMIC PARAMETER COPIES (for lock-free access)
    mutable std::atomic<double> atomicParam1_;
    mutable std::atomic<double> atomicParam2_;
    mutable std::atomic<bool> atomicParamsValid_{false};
    
    // CACHED VALUES (protected by cache_mutex_ from base)
    mutable double cached_mean_;
    mutable double cached_variance_;
    mutable double cached_log_normalization_;
    
    // SIMD computation helpers
    void computeBatchScalar(const std::vector<double>& x_values,
                           std::vector<double>& results,
                           double p1, double p2) const;
    void computeBatchSIMD(const std::vector<double>& x_values,
                         std::vector<double>& results,
                         double p1, double p2) const;
};

} // namespace libstats

#endif // LIBSTATS_MY_DISTRIBUTION_H_
```

## Level 0 Utilities Integration

### Constants Reference

The `constants.h` header provides a comprehensive set of typed constants organized into namespaces to eliminate magic numbers and ensure numerical consistency. Always use these predefined constants instead of hardcoded values:

```cpp
#include "constants.h"

namespace constants = libstats::constants;
```

#### Mathematical Constants
Fundamental mathematical values with high precision:
- `constants::mathematical::PI`, `constants::mathematical::E`, `constants::mathematical::SQRT_TWO_PI`
- `constants::mathematical::LOG_PI`, `constants::mathematical::LOG_TWO_PI`
- `constants::mathematical::PHI` (golden ratio), `constants::mathematical::EULER_MASCHERONI`
- Reciprocal constants: `constants::mathematical::INV_PI`, `constants::mathematical::INV_SQRT_TWO_PI`

#### Numerical Precision and Tolerances
Precision values for different computational needs:
- `constants::precision::DEFAULT_TOLERANCE`, `constants::precision::HIGH_PRECISION_TOLERANCE`
- `constants::precision::LOOSE_TOLERANCE`, `constants::precision::ULTRA_HIGH_PRECISION_TOLERANCE`
- `constants::precision::CONVERGENCE_TOLERANCE`, `constants::precision::INTEGRATION_TOLERANCE`

#### Probability Bounds and Safety
Numerical bounds to ensure stable probability calculations:
- `constants::probability::MIN_PROBABILITY`, `constants::probability::MAX_PROBABILITY`
- `constants::probability::MIN_LOG_PROBABILITY`, `constants::probability::MAX_LOG_PROBABILITY`
- `constants::probability::EPSILON_PROBABILITY`, `constants::probability::UNDERFLOW_THRESHOLD`

#### Statistical Critical Values
Precomputed critical values for common statistical tests:
- Normal distribution: `constants::statistical::NORMAL_95_CRITICAL`, `constants::statistical::NORMAL_99_CRITICAL`
- Chi-square: `constants::statistical::CHI_SQUARE_95_1DF`, `constants::statistical::CHI_SQUARE_99_1DF`
- t-distribution: `constants::statistical::T_95_CRITICAL_30DF`, `constants::statistical::T_99_CRITICAL_INF`
- F-distribution: `constants::statistical::F_95_CRITICAL_1_30`, `constants::statistical::F_99_CRITICAL_1_INF`

#### SIMD and Performance Constants
Optimization parameters for vectorized operations:
- Block sizes: `constants::simd::AVX512_BLOCK_SIZE`, `constants::simd::AVX2_BLOCK_SIZE`
- Memory alignment: `constants::simd::MEMORY_ALIGNMENT_AVX512`, `constants::simd::MEMORY_ALIGNMENT_NEON`
- Thresholds: `constants::simd::MIN_SIMD_SIZE`, `constants::simd::SIMD_UNROLL_FACTOR`

#### Bootstrap and Cross-Validation Defaults
Standard parameters for resampling methods:
- `constants::bootstrap::DEFAULT_BOOTSTRAP_SAMPLES`, `constants::bootstrap::MIN_BOOTSTRAP_SAMPLES`
- `constants::cross_validation::DEFAULT_FOLDS`, `constants::cross_validation::MIN_SAMPLE_SIZE_PER_FOLD`

#### Usage Examples

```cpp
// Mathematical constants for probability calculations
double MyDistribution::getProbability(double x) const {
    double result = std::exp(-0.5 * x * x) / constants::mathematical::SQRT_TWO_PI;
    return std::clamp(result, 
                     constants::probability::MIN_PROBABILITY,
                     constants::probability::MAX_PROBABILITY);
}

// Log-space calculations with precision constants
double MyDistribution::getLogProbability(double x) const {
    double log_result = -0.5 * x * x - constants::mathematical::LOG_SQRT_TWO_PI;
    return std::clamp(log_result,
                     constants::probability::MIN_LOG_PROBABILITY,
                     constants::probability::MAX_LOG_PROBABILITY);
}

// Numerical integration with appropriate tolerance
double MyDistribution::getCDF(double x) const {
    auto pdf_func = [this](double t) { return getProbability(t); };
    return math_utils::adaptive_simpson(pdf_func, getSupportLowerBound(), x,
                                       constants::precision::INTEGRATION_TOLERANCE);
}

// Statistical hypothesis testing with critical values
bool MyDistribution::performGoodnessOfFitTest(const std::vector<double>& data) const {
    double test_statistic = computeKSStatistic(data);
    return test_statistic < constants::statistical::KS_CRITICAL_95_N100;
}
```

### Error Handling Integration

Implement the safe factory pattern:

```cpp
#include "error_handling.h"

Result<MyDistribution> MyDistribution::create(double param1, double param2) {
    // Validate parameters without throwing exceptions
    if (!std::isfinite(param1) || param1 <= 0.0) {
        return Result<MyDistribution>::error(ValidationError::InvalidParameter);
    }
    
    if (!std::isfinite(param2) || param2 <= 0.0) {
        return Result<MyDistribution>::error(ValidationError::InvalidParameter);
    }
    
    // Use validation functions when available
    auto validation_result = validation::validatePositiveParameter(param1);
    if (!validation_result.is_success()) {
        return Result<MyDistribution>::error(validation_result.get_error());
    }
    
    return Result<MyDistribution>::success(MyDistribution(param1, param2));
}

// Thread-safe parameter setters with MANDATORY atomic invalidation
VoidResult MyDistribution::setParam1(double value) {
    // Validate first (no locks held)
    if (!std::isfinite(value) || value <= 0.0) {
        return VoidResult::error(ValidationError::InvalidParameter);
    }
    
    // Update parameter under lock
    {
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        param1_ = value;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        
        // CRITICAL: Invalidate atomic parameters when parameters change
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    
    return VoidResult::success();
}

// Safe factory method with exception-free construction
static Result<MyDistribution> MyDistribution::create(double param1, double param2) noexcept {
    auto validation = validateMyDistributionParameters(param1, param2);
    if (validation.isError()) {
        return Result<MyDistribution>::makeError(validation.error_code, validation.message);
    }
    
    // Use private factory to bypass validation
    return Result<MyDistribution>::ok(createUnchecked(param1, param2));
}

// Safe parameter updates without exceptions  
VoidResult MyDistribution::trySetParameters(double param1, double param2) noexcept {
    auto validation = validateMyDistributionParameters(param1, param2);
    if (validation.isError()) {
        return validation;
    }
    
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    param1_ = param1;
    param2_ = param2;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    
    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
    
    return VoidResult::ok(true);
}
```

### CPU Detection and SIMD with VectorOps Wrappers

Use runtime CPU detection and lower-level adapter functions for optimal performance:

```cpp
#include "cpu_detection.h"
#include "simd.h"
#include "vectorized_math.h"  // VectorOps wrappers

std::vector<double> MyDistribution::getBatchProbabilities(
    const std::vector<double>& x_values) const {
    
    std::vector<double> results(x_values.size());
    
    // Get parameters once under shared lock
    double p1, p2;
    {
        std::shared_lock lock(cache_mutex_);
        p1 = param1_;
        p2 = param2_;
    }
    
    // Use VectorOps for automatic SIMD dispatch and alignment handling
    if (x_values.size() >= VectorOps::MIN_SIMD_SIZE) {
        computeBatchVectorized(x_values, results, p1, p2);
    } else {
        computeBatchScalar(x_values, results, p1, p2);
    }
    
    return results;
}

// Use VectorOps wrappers for automatic SIMD optimization
void MyDistribution::computeBatchVectorized(const std::vector<double>& x_values,
                                           std::vector<double>& results,
                                           double p1, double p2) const {
    // VectorOps automatically handles:
    // - CPU feature detection
    // - Memory alignment
    // - SIMD instruction selection
    // - Fallback to scalar operations
    
    // Example: Gaussian PDF computation using VectorOps
    VectorOps::subtract_scalar(x_values.data(), p1, results.data(), x_values.size()); // x - mu
    VectorOps::divide_scalar(results.data(), p2, results.data(), x_values.size());    // (x - mu) / sigma
    VectorOps::square(results.data(), results.data(), x_values.size());               // ((x - mu) / sigma)^2
    VectorOps::multiply_scalar(results.data(), -0.5, results.data(), x_values.size()); // -0.5 * ((x - mu) / sigma)^2
    VectorOps::exp(results.data(), results.data(), x_values.size());                  // exp(-0.5 * ((x - mu) / sigma)^2)
    
    // Apply normalization constant
    double norm_const = 1.0 / (p2 * std::sqrt(2.0 * M_PI));
    VectorOps::multiply_scalar(results.data(), norm_const, results.data(), x_values.size());
}

// Lower-level adapter function for custom SIMD implementations
void MyDistribution::computeBatchCustomSIMD(const std::vector<double>& x_values,
                                           std::vector<double>& results,
                                           double p1, double p2) const {
    // Use lower-level SIMD adapter functions for fine-grained control
    const size_t simd_width = SIMDTraits<double>::width();
    const size_t aligned_size = (x_values.size() / simd_width) * simd_width;
    
    // Process aligned chunks with custom SIMD implementation
    if (aligned_size > 0) {
        simd_adapter::gaussian_pdf_batch(x_values.data(), results.data(), aligned_size, p1, p2);
    }
    
    // Handle remainder with scalar operations
    for (size_t i = aligned_size; i < x_values.size(); ++i) {
        results[i] = gaussian_pdf_scalar(x_values[i], p1, p2);
    }
}

// Scalar implementation for reference and remainder processing
double MyDistribution::gaussian_pdf_scalar(double x, double mu, double sigma) const {
    double diff = x - mu;
    double normalized = diff / sigma;
    double exponent = -0.5 * normalized * normalized;
    double norm_const = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    return norm_const * std::exp(exponent);
}
```

### VectorOps Integration Patterns

Use VectorOps wrappers for consistent, high-performance vectorized operations:

```cpp
// Example: Exponential distribution PDF using VectorOps
void ExponentialDistribution::computeBatchVectorized(
    const std::vector<double>& x_values,
    std::vector<double>& results,
    double lambda) const {
    
    // Check for negative values (exponential support is [0, ∞))
    VectorOps::clamp_lower(x_values.data(), 0.0, results.data(), x_values.size());
    
    // Compute -lambda * x
    VectorOps::multiply_scalar(results.data(), -lambda, results.data(), x_values.size());
    
    // Compute exp(-lambda * x)
    VectorOps::exp(results.data(), results.data(), x_values.size());
    
    // Multiply by lambda (normalization)
    VectorOps::multiply_scalar(results.data(), lambda, results.data(), x_values.size());
}

// Example: Discrete uniform PMF using VectorOps
void DiscreteDistribution::computeBatchVectorized(
    const std::vector<double>& x_values,
    std::vector<double>& results,
    int lower, int upper) const {
    
    const double pmf_value = 1.0 / (upper - lower + 1);
    
    // Check if values are in support range
    VectorOps::in_range_discrete(x_values.data(), lower, upper, results.data(), x_values.size());
    
    // Multiply by PMF value (results contains 1.0 for valid values, 0.0 for invalid)
    VectorOps::multiply_scalar(results.data(), pmf_value, results.data(), x_values.size());
}
```

## Thread Parallelism Patterns

Libstats provides three levels of parallelism optimized for different computational patterns in statistical computing:

### SIMD Vectorization (Data-Level Parallelism)

**Best for:** Element-wise operations on arrays, batch probability calculations, mathematical transformations

**Characteristics:**
- Single instruction, multiple data processing
- Operates on 4-8 elements simultaneously (depending on instruction set)
- Automatic CPU feature detection and fallback
- Memory alignment requirements for optimal performance

```cpp
// Use SIMD for batch probability calculations
std::vector<double> MyDistribution::getBatchProbabilities(
    const std::vector<double>& x_values) const {
    
    std::vector<double> results(x_values.size());
    
    // Get parameters once under shared lock
    double p1, p2;
    {
        std::shared_lock lock(cache_mutex_);
        p1 = param1_;
        p2 = param2_;
    }
    
    // Use SIMD threshold from constants
    if (x_values.size() >= constants::simd::MIN_SIMD_SIZE && 
        cpu::supports_avx2()) {
        computeBatchSIMD(x_values, results, p1, p2);
    } else {
        computeBatchScalar(x_values, results, p1, p2);
    }
    
    return results;
}
```

### ThreadPool (Task-Level Parallelism)

**Best for:** Independent computational tasks, embarrassingly parallel problems, parameter estimation

**Characteristics:**
- Fixed number of persistent worker threads
- Task queue with FIFO execution
- Future-based result handling
- Optimized for CPU-intensive statistical computations
- Integrated with CPU detection and constants for optimal thread count

```cpp
#include "thread_pool.h"

// Use ThreadPool for parameter estimation across multiple datasets
void MyDistribution::fitMultipleDatasets(
    const std::vector<std::vector<double>>& datasets) {
    
    ThreadPool pool(ThreadPool::getOptimalThreadCount());
    std::vector<std::future<EstimationResult>> futures;
    
    // Submit fitting tasks for each dataset
    for (const auto& dataset : datasets) {
        auto future = pool.submit([this, dataset]() {
            return estimateParametersMLEInternal(dataset);
        });
        futures.push_back(std::move(future));
    }
    
    // Collect results
    std::vector<EstimationResult> results;
    for (auto& future : futures) {
        results.push_back(future.get());
    }
    
    // Process aggregated results...
    processEstimationResults(results);
}

// Use ThreadPool for Monte Carlo simulations
std::vector<double> MyDistribution::monteCarloEstimation(
    size_t num_samples, std::mt19937& base_rng) const {
    
    ThreadPool pool;
    const size_t num_threads = pool.getThreadCount();
    const size_t samples_per_thread = num_samples / num_threads;
    
    std::vector<std::future<std::vector<double>>> futures;
    
    for (size_t i = 0; i < num_threads; ++i) {
        // Create independent RNG for each thread
        std::mt19937 thread_rng(base_rng());
        
        auto future = pool.submit([this, samples_per_thread, thread_rng]() mutable {
            return this->sample(thread_rng, samples_per_thread);
        });
        futures.push_back(std::move(future));
    }
    
    // Aggregate results from all threads
    std::vector<double> all_samples;
    for (auto& future : futures) {
        auto thread_samples = future.get();
        all_samples.insert(all_samples.end(), 
                          thread_samples.begin(), thread_samples.end());
    }
    
    return all_samples;
}
```

### WorkStealingPool (Dynamic Load Balancing)

**Best for:** Uneven workloads, recursive algorithms, adaptive computations with variable execution times

**Characteristics:**
- Per-thread work queues with work stealing
- Dynamic load balancing for uneven workloads
- Optimized for cache locality and NUMA awareness
- Statistics tracking for performance monitoring
- Built-in parallel-for with automatic work distribution

```cpp
#include "work_stealing_pool.h"

// Use WorkStealingPool for adaptive numerical integration
double MyDistribution::adaptiveIntegration(
    double lower, double upper, double tolerance) const {
    
    WorkStealingPool pool(WorkStealingPool::getOptimalThreadCount());
    std::atomic<double> total_integral{0.0};
    std::atomic<double> total_error{0.0};
    
    // Initial subdivision into coarse intervals
    const size_t initial_intervals = pool.getThreadCount() * 4;
    const double interval_width = (upper - lower) / initial_intervals;
    
    // Submit adaptive integration tasks
    for (size_t i = 0; i < initial_intervals; ++i) {
        double a = lower + i * interval_width;
        double b = lower + (i + 1) * interval_width;
        
        pool.submit([this, a, b, tolerance, &total_integral, &total_error, &pool]() {
            auto result = adaptiveIntegrationRecursive(a, b, tolerance, pool);
            total_integral.fetch_add(result.integral, std::memory_order_relaxed);
            total_error.fetch_add(result.error, std::memory_order_relaxed);
        });
    }
    
    pool.waitForAll();
    return total_integral.load();
}

// Use WorkStealingPool's parallel-for for range-based computations
std::vector<double> MyDistribution::computeQuantiles(
    const std::vector<double>& probabilities) const {
    
    WorkStealingPool pool;
    std::vector<double> quantiles(probabilities.size());
    
    // Parallel computation with automatic work distribution
    pool.parallelFor(0, probabilities.size(), 
        [this, &probabilities, &quantiles](size_t i) {
            quantiles[i] = this->getQuantile(probabilities[i]);
        },
        constants::parallel::DEFAULT_GRAIN_SIZE
    );
    
    pool.waitForAll();
    return quantiles;
}

// Example of recursive work generation in work-stealing context
IntegrationResult MyDistribution::adaptiveIntegrationRecursive(
    double a, double b, double tolerance, WorkStealingPool& pool) const {
    
    auto coarse_result = simpsonRule(a, b);
    auto fine_result = compositeSimpsonRule(a, b, 2);
    
    double error_estimate = std::abs(fine_result - coarse_result) / 15.0;
    
    if (error_estimate <= tolerance) {
        return {fine_result, error_estimate};
    }
    
    // Subdivide and submit more work if error too large
    double midpoint = (a + b) / 2.0;
    
    std::atomic<double> left_integral{0.0}, right_integral{0.0};
    std::atomic<double> left_error{0.0}, right_error{0.0};
    
    // Submit left half
    pool.submit([this, a, midpoint, tolerance, &pool, &left_integral, &left_error]() {
        auto result = adaptiveIntegrationRecursive(a, midpoint, tolerance/2, pool);
        left_integral.store(result.integral);
        left_error.store(result.error);
    });
    
    // Compute right half in current thread
    auto right_result = adaptiveIntegrationRecursive(midpoint, b, tolerance/2, pool);
    
    // Wait for left half completion (work stealing will handle load balancing)
    pool.waitForAll();
    
    return {left_integral.load() + right_result.integral,
            left_error.load() + right_result.error};
}
```

### Choosing the Right Parallelism Pattern

| Use Case | SIMD | ThreadPool | WorkStealingPool |
|----------|------|------------|------------------|
| Batch PDF/CDF calculations | ✓ | | |
| Independent parameter fitting | | ✓ | |
| Monte Carlo simulations | | ✓ | |
| Adaptive numerical methods | | | ✓ |
| Recursive algorithms | | | ✓ |
| Uneven computational loads | | | ✓ |
| Cache-sensitive workloads | ✓ | | ✓ |
| Memory bandwidth bound | ✓ | | |
| CPU compute bound | | ✓ | ✓ |

### Performance Monitoring

```cpp
// Monitor work-stealing efficiency
void MyDistribution::analyzeParallelPerformance() {
    WorkStealingPool pool;
    
    // Perform some parallel computation...
    computeLargeDatasetStatistics(large_dataset, pool);
    
    // Check work-stealing statistics
    auto stats = pool.getStatistics();
    std::cout << "Tasks executed: " << stats.tasksExecuted << std::endl;
    std::cout << "Work steals: " << stats.workSteals << std::endl;
    std::cout << "Steal success rate: " << stats.stealSuccessRate << std::endl;
    
    // High steal success rate (>0.7) indicates good load balancing
    // Low steal success rate may indicate need for smaller grain sizes
    if (stats.stealSuccessRate < 0.5) {
        std::cout << "Consider reducing grain size for better load balancing" << std::endl;
    }
}
```

### Safety Functions

Use safe mathematical operations:

```cpp
#include "safety.h"

double MyDistribution::getProbability(double x) const {
    // Use safe mathematical functions
    double safe_x_squared = safety::safe_multiply(x, x);
    double safe_exp_arg = safety::safe_multiply(-0.5, safe_x_squared);
    double safe_exp_result = safety::safe_exp(safe_exp_arg);
    
    // Use safe division
    double denominator = std::sqrt(2.0 * constants::mathematical::PI);
    return safety::safe_divide(safe_exp_result, denominator);
}

double MyDistribution::getQuantile(double p) const {
    // Use bounds checking
    safety::check_probability_bounds(p, "quantile probability");
    
    // Use convergence detection for iterative methods
    safety::ConvergenceDetector detector(constants::numerical::CONVERGENCE_TOLERANCE, 
                                        constants::numerical::MAX_ITERATIONS);
    
    double current_estimate = 0.0; // initial guess
    
    while (!detector.hasConverged(current_estimate)) {
        double cdf_value = getCumulativeProbability(current_estimate);
        double pdf_value = getProbability(current_estimate);
        
        // Newton-Raphson step with safe division
        double step = safety::safe_divide(cdf_value - p, pdf_value);
        current_estimate = safety::safe_subtract(current_estimate, step);
        
        if (detector.hasReachedMaxIterations()) {
            throw std::runtime_error("Quantile calculation failed to converge");
        }
    }
    
    return current_estimate;
}
```

### Mathematical Utilities

Use advanced mathematical functions:

```cpp
#include "math_utils.h"

double MyDistribution::getCumulativeProbability(double x) const {
    // Use numerical integration for complex CDFs
    auto pdf_func = [this](double t) { return this->getProbability(t); };
    
    double lower_bound = getSupportLowerBound();
    return math_utils::adaptive_simpson(pdf_func, lower_bound, x, 
                                       constants::numerical::INTEGRATION_TOLERANCE);
}

void MyDistribution::fit(const std::vector<double>& data) {
    // Use statistical utilities
    double sample_mean = math_utils::calculate_mean(data);
    double sample_variance = math_utils::calculate_variance(data, sample_mean);
    
    // Method of moments estimation
    double new_param1 = sample_mean;
    double new_param2 = sample_variance;
    
    // Use root finding for MLE if needed
    if (use_mle_estimation) {
        auto log_likelihood = [&data](double param) {
            // Calculate negative log-likelihood
            double nll = 0.0;
            for (double x : data) {
                nll -= this->getLogProbability(x);  // This won't work in lambda
            }
            return nll;
        };
        
        // Use optimization
        new_param1 = math_utils::golden_section_search(log_likelihood, 
                                                      0.1, 10.0,
                                                      constants::numerical::OPTIMIZATION_TOLERANCE);
    }
    
    // Update parameters using thread-safe setters
    auto result1 = setParam1(new_param1);
    auto result2 = setParam2(new_param2);
    
    if (!result1.is_success() || !result2.is_success()) {
        throw std::runtime_error("Parameter fitting failed");
    }
}
```

### Log-Space Operations

Use log-space arithmetic for numerical stability:

```cpp
#include "log_space_ops.h"

double MyDistribution::getLogProbability(double x) const {
    // Compute in log space for numerical stability
    double log_numerator = -0.5 * x * x;
    double log_denominator = 0.5 * constants::mathematical::LOG_TWO_PI;
    
    return log_numerator - log_denominator;
}

std::vector<double> MyDistribution::getBatchLogProbabilities(
    const std::vector<double>& x_values) const {
    
    std::vector<double> log_results(x_values.size());
    
    // Get parameters
    double p1, p2;
    {
        std::shared_lock lock(param_mutex_);
        p1 = param1_;
        p2 = param2_;
    }
    
    // Compute log probabilities
    for (size_t i = 0; i < x_values.size(); ++i) {
        log_results[i] = getLogProbability(x_values[i]);
    }
    
    // Use log-space operations if needed
    if (need_normalization) {
        double log_sum = log_space_ops::logSumExpVector(log_results);
        for (double& log_val : log_results) {
            log_val = log_space_ops::logSubtract(log_val, log_sum);
        }
    }
    
    return log_results;
}
```

## Thread Safety Implementation

### Atomic Parameter Management and Cache Invalidation

All distributions must implement **atomic parameter getters** and proper **cache invalidation patterns** for thread-safe, lock-free parameter access:

```cpp
class MyDistribution : public DistributionBase {
private:
    // Core distribution parameters
    double param1_{1.0};
    double param2_{1.0};
    
    // Atomic copies for lock-free access
    mutable std::atomic<double> atomicParam1_{1.0};
    mutable std::atomic<double> atomicParam2_{1.0};
    mutable std::atomic<bool> atomicParamsValid_{false};
    
public:
    // Regular parameter getters (thread-safe with locking)
    double getParam1() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return param1_;
    }
    
    // Atomic parameter getters (lock-free, high performance)
    double getParam1Atomic() const noexcept {
        if (atomicParamsValid_.load(std::memory_order_acquire)) {
            return atomicParam1_.load(std::memory_order_acquire);
        }
        // Fallback to locked version if atomic copy not valid
        return getParam1();
    }
    
    // Parameter setters with MANDATORY atomic invalidation
    void setParam1(double value) {
        // Validate parameters outside of any lock
        validateParameters(value, getParam2());
        
        // Set parameter under lock
        std::unique_lock<std::shared_mutex> lock(cache_mutex_);
        param1_ = value;
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        
        // CRITICAL: Invalidate atomic parameters when parameters change
        atomicParamsValid_.store(false, std::memory_order_release);
    }
};
```

### Cache Update Pattern with Atomic Parameter Management

```cpp
void MyDistribution::updateCacheUnsafe() const {
    // This method is called under unique lock from base class
    // Compute expensive values once
    cached_mean_ = computeMean(param1_, param2_);
    cached_variance_ = computeVariance(param1_, param2_);
    cached_log_normalization_ = computeLogNormalization(param1_, param2_);
    
    // Mark cache as valid (required)
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
    
    // CRITICAL: Update atomic parameters for lock-free access
    atomicParam1_.store(param1_, std::memory_order_release);
    atomicParam2_.store(param2_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

// Thread-safe property access
double MyDistribution::getMean() const {
    return getCachedValue([this]() { return cached_mean_; });
}

// Copy constructor with proper locking
MyDistribution::MyDistribution(const MyDistribution& other) 
    : DistributionBase() {
    std::shared_lock other_lock(other.param_mutex_);
    param1_ = other.param1_;
    param2_ = other.param2_;
    // Cache will be invalidated/rebuilt as needed
}

// Assignment operator with proper locking and atomic invalidation
MyDistribution& MyDistribution::operator=(const MyDistribution& other) {
    if (this != &other) {
        // Lock both objects in consistent order to prevent deadlock
        std::lock(cache_mutex_, other.cache_mutex_);
        std::unique_lock this_lock(cache_mutex_, std::adopt_lock);
        std::shared_lock other_lock(other.cache_mutex_, std::adopt_lock);
        
        param1_ = other.param1_;
        param2_ = other.param2_;
        
        // Invalidate cache and atomic parameters after parameters change
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
    return *this;
}
```

### Deadlock Prevention

```cpp
// Always acquire locks in consistent order
void MyDistribution::copyParametersFrom(const MyDistribution& other) {
    // Use std::lock for multiple mutex acquisition
    std::lock(param_mutex_, other.param_mutex_);
    std::unique_lock this_lock(param_mutex_, std::adopt_lock);
    std::shared_lock other_lock(other.param_mutex_, std::adopt_lock);
    
    param1_ = other.param1_;
    param2_ = other.param2_;
    
    // Invalidate cache last
    invalidateCache();
}

// Avoid calling base class copy/move operators that acquire locks
MyDistribution& MyDistribution::operator=(MyDistribution&& other) noexcept {
    if (this != &other) {
        std::lock(param_mutex_, other.param_mutex_);
        std::unique_lock this_lock(param_mutex_, std::adopt_lock);
        std::unique_lock other_lock(other.param_mutex_, std::adopt_lock);
        
        param1_ = std::move(other.param1_);
        param2_ = std::move(other.param2_);
        
        // Don't call base class move operator - it would deadlock
        invalidateCache();
    }
    return *this;
}
```

## Performance Optimization

### Cache Management

```cpp
// Cache only expensive computations
void MyDistribution::updateCacheUnsafe() const {
    std::shared_lock param_lock(param_mutex_);
    
    // Cache expensive special function evaluations
    cached_log_gamma_param1_ = math_utils::lgamma(param1_);
    cached_log_gamma_param2_ = math_utils::lgamma(param2_);
    cached_log_beta_normalization_ = cached_log_gamma_param1_ + 
                                    cached_log_gamma_param2_ - 
                                    math_utils::lgamma(param1_ + param2_);
    
    // Simple arithmetic can be computed on-demand
    cached_mean_ = param1_ / (param1_ + param2_);
    cached_variance_ = (param1_ * param2_) / 
                      (std::pow(param1_ + param2_, 2) * (param1_ + param2_ + 1));
    
    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);
}
```

### SIMD Optimization Guidelines

```cpp
// Use SIMD for appropriate operations
void MyDistribution::computeBatchSIMD(const std::vector<double>& x_values,
                                     std::vector<double>& results,
                                     double p1, double p2) const {
    // Check if SIMD is beneficial
    if (!cpu::supports_avx2() || x_values.size() < 8) {
        return computeBatchScalar(x_values, results, p1, p2);
    }
    
    // Ensure proper alignment
    const size_t alignment = simd::optimal_alignment();
    if (reinterpret_cast<uintptr_t>(x_values.data()) % alignment != 0) {
        // Fall back to scalar if not aligned
        return computeBatchScalar(x_values, results, p1, p2);
    }
    
    // Use vectorized operations
    const size_t simd_width = 4; // AVX2 processes 4 doubles
    const size_t simd_end = (x_values.size() / simd_width) * simd_width;
    
    for (size_t i = 0; i < simd_end; i += simd_width) {
        simd::vector_exp(&x_values[i], &results[i], simd_width);
    }
    
    // Handle remaining elements
    for (size_t i = simd_end; i < x_values.size(); ++i) {
        results[i] = std::exp(x_values[i]);
    }
}
```

## Enhanced Caching Strategies

### Configure Cache Settings

Optimize cache performance based on your application's memory constraints:

```cpp
// Configure cache for memory-constrained environments
void MyDistribution::optimizeForMemoryConstraints() {
    CacheConfig config;
    config.max_memory_usage = 512 * 1024;  // 512KB limit
    config.cache_ttl = std::chrono::milliseconds(2000);  // 2 second TTL
    config.memory_pressure_aware = true;
    config.adaptive_sizing = true;
    configureCacheSettings(config);
}

// Configure cache for high-performance scenarios
void MyDistribution::optimizeForHighPerformance() {
    CacheConfig config;
    config.max_memory_usage = 4 * 1024 * 1024;  // 4MB limit
    config.cache_ttl = std::chrono::milliseconds(10000);  // 10 second TTL
    config.memory_pressure_aware = false;  // Don't evict aggressively
    config.adaptive_sizing = true;
    configureCacheSettings(config);
}
```

### Monitor Cache Performance

Regularly monitor cache metrics to optimize performance:

```cpp
void MyDistribution::monitorCachePerformance() {
    auto metrics = getCacheMetrics();
    
    // Log cache performance
    std::cout << "Cache hit rate: " << metrics.hitRate() << std::endl;
    std::cout << "Memory efficiency: " << metrics.memoryEfficiency() << std::endl;
    std::cout << "Memory usage: " << metrics.memory_usage << " bytes" << std::endl;
    std::cout << "Cache evictions: " << metrics.cache_evictions << std::endl;
    
    // Adjust cache settings based on performance
    if (metrics.hitRate() < 0.7) {
        // Low hit rate - increase cache TTL
        auto config = getCacheConfig();
        config.cache_ttl = std::chrono::milliseconds(config.cache_ttl.count() * 2);
        configureCacheSettings(config);
    }
    
    if (metrics.memoryEfficiency() < 0.5) {
        // Poor memory efficiency - reduce cache size
        auto config = getCacheConfig();
        config.max_memory_usage = static_cast<size_t>(config.max_memory_usage * 0.8);
        configureCacheSettings(config);
    }
}
```

### Adaptive Cache Implementation

Implement adaptive caching for complex distributions:

```cpp
class MyDistribution : public DistributionBase {
private:
    // Use adaptive cache for expensive quantile computations
    mutable AdaptiveCache<double, double> quantile_cache_;
    
    // Initialize adaptive cache in constructor
    MyDistribution(double param1, double param2) 
        : param1_(param1), param2_(param2),
          quantile_cache_(getCacheConfig(), &cache_metrics_) {
    }
    
public:
    double getQuantile(double p) const override {
        // Try cache first
        if (auto cached = quantile_cache_.get(p)) {
            return *cached;
        }
        
        // Compute and cache result
        double result = computeQuantile(p);
        quantile_cache_.put(p, result);
        return result;
    }
    
    // Clear adaptive caches when parameters change
    void invalidateAdaptiveCaches() {
        quantile_cache_.clear();
    }
};
```

## Memory Optimization Features

### Thread-Local Memory Pools

Use thread-local memory pools for temporary allocations:

```cpp
std::vector<double> MyDistribution::getBatchProbabilities(
    const std::vector<double>& x_values) const {
    
    // Use thread-local memory pool for temporary allocations
    auto& pool = getThreadPool();
    
    const size_t needed_size = x_values.size() * sizeof(double);
    if (pool.hasSpace(needed_size)) {
        // Use pool for temporary storage
        double* temp_storage = static_cast<double*>(pool.allocate(needed_size));
        
        // Perform computations using pool memory
        computeBatchWithPool(x_values, temp_storage);
        
        // Copy results (pool memory will be reused automatically)
        std::vector<double> results(temp_storage, temp_storage + x_values.size());
        return results;
    }
    
    // Fall back to heap allocation if pool exhausted
    return computeBatchWithHeap(x_values);
}
```

### SIMD-Aligned Memory

Use SIMD-aligned vectors for optimal performance:

```cpp
std::vector<double> MyDistribution::getBatchProbabilities(
    const std::vector<double>& x_values) const {
    
    // Use SIMD-aligned vector for results
    simd_vector<double> aligned_results(x_values.size());
    
    // Get parameters once
    double p1, p2;
    {
        std::shared_lock lock(param_mutex_);
        p1 = param1_;
        p2 = param2_;
    }
    
    // Use SIMD operations if beneficial
    if (shouldUseSIMDBatch(x_values.size())) {
        computeBatchSIMD(x_values, aligned_results, p1, p2);
    } else {
        computeBatchScalar(x_values, aligned_results, p1, p2);
    }
    
    // Convert to standard vector if needed
    return std::vector<double>(aligned_results.begin(), aligned_results.end());
}
```

### Small Vector Optimization

Use small vectors for temporary collections:

```cpp
double MyDistribution::computeComplexStatistic(const std::vector<double>& data) const {
    // Use SmallVector for temporary results (avoids heap allocation for small sizes)
    SmallVector<double, 16> temp_results;
    
    // Process data in chunks
    for (size_t i = 0; i < data.size(); i += 16) {
        size_t chunk_size = std::min(16UL, data.size() - i);
        
        // Compute chunk result
        double chunk_result = 0.0;
        for (size_t j = 0; j < chunk_size; ++j) {
            chunk_result += getProbability(data[i + j]);
        }
        
        temp_results.push_back(chunk_result);
    }
    
    // Aggregate results
    double total = 0.0;
    for (double result : temp_results) {
        total += result;
    }
    
    return total / data.size();
}
```

### Stack-Based Allocation

Use stack allocators for temporary computations:

```cpp
template<typename Func>
double MyDistribution::computeWithStackMemory(Func computation) const {
    StackAllocator<4096> stack_alloc;  // 4KB stack
    
    try {
        // Allocate temporary arrays on stack
        double* temp_array1 = stack_alloc.allocate<double>(512);
        double* temp_array2 = stack_alloc.allocate<double>(512);
        
        // Perform computation using stack memory
        double result = computation(temp_array1, temp_array2);
        
        // Memory is automatically cleaned up
        return result;
        
    } catch (const std::bad_alloc&) {
        // Fall back to heap allocation if stack exhausted
        return computeWithHeapMemory(computation);
    }
}
```

## SIMD Batch Operations

### SIMD Implementation Decision Matrix

Before implementing SIMD operations, use this decision matrix to determine the best approach:

| Scenario | Use VectorOps | Use Custom SIMD | Use Scalar |
|----------|---------------|-----------------|------------|
| Standard mathematical operations (exp, log, sqrt) | ✓ | | |
| Custom mathematical formulas | | ✓ | |
| Simple arithmetic (add, multiply) | ✓ | | |
| Complex branching logic | | | ✓ |
| Memory-bound operations | | | ✓ |
| Small arrays (< 32 elements) | | | ✓ |
| Unaligned memory access | ✓ | | ✓ |
| CPU feature uncertainty | ✓ | | ✓ |

### VectorOps vs. Custom SIMD Guidelines

**Use VectorOps when:**
- Performing standard mathematical operations available in the VectorOps library
- You need automatic CPU feature detection and fallbacks
- Memory alignment is uncertain or variable
- Development time is constrained
- Maintainability is prioritized over maximum performance

**Use Custom SIMD when:**
- Implementing domain-specific mathematical formulas not in VectorOps
- Maximum performance is critical and you can ensure proper alignment
- You have complex multi-step SIMD operations that benefit from register reuse
- Memory access patterns are predictable and optimizable

### Implementing SIMD Batch Methods

Override base class batch methods with SIMD optimizations:

```cpp
class MyDistribution : public DistributionBase {
public:
    // Override base class batch methods for SIMD optimization
    std::vector<double> getBatchProbabilities(
        const std::vector<double>& x_values) const override {
        
        std::vector<double> results(x_values.size());
        
        // Get parameters once under shared lock
        double p1, p2;
        {
            std::shared_lock lock(param_mutex_);
            p1 = param1_;
            p2 = param2_;
        }
        
        // Use comprehensive SIMD decision logic
        if (shouldUseSIMDBatch(x_values, p1, p2)) {
            computeBatchSIMD(x_values, results, p1, p2);
        } else {
            computeBatchScalar(x_values, results, p1, p2);
        }
        
        return results;
    }
    
    std::vector<double> getBatchLogProbabilities(
        const std::vector<double>& x_values) const override {
        
        std::vector<double> results(x_values.size());
        
        double p1, p2;
        {
            std::shared_lock lock(param_mutex_);
            p1 = param1_;
            p2 = param2_;
        }
        
        if (shouldUseSIMDBatch(x_values.size())) {
            computeBatchLogSIMD(x_values, results, p1, p2);
        } else {
            computeBatchLogScalar(x_values, results, p1, p2);
        }
        
        return results;
    }
};
```

### SIMD Implementation Patterns

Implement SIMD operations with proper fallbacks:

```cpp
void MyDistribution::computeBatchSIMD(const std::vector<double>& x_values,
                                     std::vector<double>& results,
                                     double p1, double p2) const {
    // Verify SIMD support at runtime
    if (!cpu::supports_avx2()) {
        return computeBatchScalar(x_values, results, p1, p2);
    }
    
    const size_t simd_width = 4;  // AVX2 processes 4 doubles
    const size_t simd_end = (x_values.size() / simd_width) * simd_width;
    
    // Process SIMD chunks
    for (size_t i = 0; i < simd_end; i += simd_width) {
        // Load 4 doubles into SIMD register
        __m256d x_vec = _mm256_load_pd(&x_values[i]);
        
        // Perform SIMD computation
        __m256d result_vec = computeSIMDProbability(x_vec, p1, p2);
        
        // Store results
        _mm256_store_pd(&results[i], result_vec);
    }
    
    // Handle remaining elements with scalar code
    for (size_t i = simd_end; i < x_values.size(); ++i) {
        results[i] = computeScalarProbability(x_values[i], p1, p2);
    }
}

// Helper function for SIMD computation
__m256d MyDistribution::computeSIMDProbability(__m256d x_vec, double p1, double p2) const {
    // Example: Gaussian PDF computation
    // pdf(x) = (1/sqrt(2*pi*sigma^2)) * exp(-0.5 * ((x-mu)/sigma)^2)
    
    __m256d mu_vec = _mm256_set1_pd(p1);        // broadcast mu
    __m256d sigma_vec = _mm256_set1_pd(p2);     // broadcast sigma
    __m256d half_vec = _mm256_set1_pd(0.5);
    
    // Compute (x - mu) / sigma
    __m256d diff_vec = _mm256_sub_pd(x_vec, mu_vec);
    __m256d normalized_vec = _mm256_div_pd(diff_vec, sigma_vec);
    
    // Compute -(x - mu)^2 / (2 * sigma^2)
    __m256d squared_vec = _mm256_mul_pd(normalized_vec, normalized_vec);
    __m256d exp_arg_vec = _mm256_mul_pd(_mm256_sub_pd(_mm256_setzero_pd(), half_vec), 
                                        squared_vec);
    
    // Compute exp(-(x - mu)^2 / (2 * sigma^2))
    __m256d exp_vec = simd::vector_exp_pd(exp_arg_vec);
    
    // Compute normalization factor
    __m256d norm_factor = _mm256_set1_pd(1.0 / (sigma_vec[0] * sqrt(2.0 * M_PI)));
    
    return _mm256_mul_pd(exp_vec, norm_factor);
}
```

### Testing SIMD Operations

Ensure SIMD operations produce correct results:

```cpp
// Test SIMD vs scalar consistency
TEST_F(MyDistributionTest, SIMDConsistency) {
    const size_t test_size = 1000;
    std::vector<double> x_values(test_size);
    
    // Generate test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-5.0, 5.0);
    for (size_t i = 0; i < test_size; ++i) {
        x_values[i] = uniform(rng);
    }
    
    // Get scalar results
    std::vector<double> scalar_results(test_size);
    dist.computeBatchScalar(x_values, scalar_results);
    
    // Get SIMD results
    std::vector<double> simd_results(test_size);
    dist.computeBatchSIMD(x_values, simd_results);
    
    // Compare results
    for (size_t i = 0; i < test_size; ++i) {
        EXPECT_NEAR(scalar_results[i], simd_results[i], 1e-14)
            << "Mismatch at index " << i << " for x=" << x_values[i];
    }
}

// Test SIMD performance improvement
TEST_F(MyDistributionTest, SIMDPerformance) {
    const size_t large_size = 100000;
    std::vector<double> x_values(large_size);
    
    // Generate large test dataset
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> uniform(-3.0, 3.0);
    for (size_t i = 0; i < large_size; ++i) {
        x_values[i] = uniform(rng);
    }
    
    // Time scalar implementation
    auto start = std::chrono::high_resolution_clock::now();
    auto scalar_results = dist.getBatchProbabilities(x_values);
    auto scalar_time = std::chrono::high_resolution_clock::now() - start;
    
    // Time SIMD implementation (if available)
    if (cpu::supports_avx2()) {
        start = std::chrono::high_resolution_clock::now();
        auto simd_results = dist.getBatchProbabilities(x_values);
        auto simd_time = std::chrono::high_resolution_clock::now() - start;
        
        // SIMD should be significantly faster for large datasets
        double speedup = static_cast<double>(scalar_time.count()) / simd_time.count();
        EXPECT_GT(speedup, 1.5) << "SIMD implementation should be at least 1.5x faster";
        
        std::cout << "SIMD speedup: " << speedup << "x" << std::endl;
    }
}
```

## Testing Guidelines

### Testing Guidelines

```cpp
// Test file: test_my_distribution.cpp
#include "my_distribution.h"
#include <gtest/gtest.h>

class MyDistributionTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto result = MyDistribution::create(1.0, 2.0);
        ASSERT_TRUE(result.isOk());
        dist = std::move(result.value);
    }
    
    MyDistribution dist;
    static constexpr double tolerance = 1e-10;
};

// Test atomic parameter invalidation (MANDATORY)
TEST_F(MyDistributionTest, AtomicParameterInvalidation) {
    // Verify initial atomic values
    EXPECT_DOUBLE_EQ(dist.getParam1Atomic(), 1.0);
    EXPECT_DOUBLE_EQ(dist.getParam2Atomic(), 2.0);
    
    // Test setParam1 invalidation
    dist.setParam1(5.0);
    EXPECT_DOUBLE_EQ(dist.getParam1(), 5.0);
    EXPECT_DOUBLE_EQ(dist.getParam1Atomic(), 5.0);
    EXPECT_DOUBLE_EQ(dist.getParam2Atomic(), 2.0);
    
    // Test setParam2 invalidation
    dist.setParam2(3.0);
    EXPECT_DOUBLE_EQ(dist.getParam2(), 3.0);
    EXPECT_DOUBLE_EQ(dist.getParam1Atomic(), 5.0);
    EXPECT_DOUBLE_EQ(dist.getParam2Atomic(), 3.0);
    
    // Test setParameters invalidation
    auto result = dist.trySetParameters(10.0, 20.0);
    ASSERT_TRUE(result.isOk());
    EXPECT_DOUBLE_EQ(dist.getParam1Atomic(), 10.0);
    EXPECT_DOUBLE_EQ(dist.getParam2Atomic(), 20.0);
}

// Test safe factory pattern (MANDATORY)
TEST_F(MyDistributionTest, SafeFactoryValidation) {
    // Valid parameters
    auto valid_result = MyDistribution::create(1.0, 2.0);
    EXPECT_TRUE(valid_result.isOk());
    
    // Invalid parameters
    auto invalid_result = MyDistribution::create(-1.0, 2.0);
    EXPECT_TRUE(invalid_result.isError());
    EXPECT_EQ(invalid_result.error_code, ValidationError::InvalidParameter);
    
    // Test exception-free parameter updates
    auto update_result = dist.trySetParameters(5.0, 10.0);
    EXPECT_TRUE(update_result.isOk());
    
    auto invalid_update = dist.trySetParameters(-1.0, 10.0);
    EXPECT_TRUE(invalid_update.isError());
}

// Test thread safety with atomic getters (MANDATORY)
TEST_F(MyDistributionTest, ThreadSafetyWithAtomicGetters) {
    std::vector<std::thread> threads;
    std::atomic<bool> all_passed{true};
    
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&]() {
            for (int j = 0; j < 10000; ++j) {
                // Test both regular and atomic getters
                double param1_regular = dist.getParam1();
                double param1_atomic = dist.getParam1Atomic();
                double param2_regular = dist.getParam2();
                double param2_atomic = dist.getParam2Atomic();
                
                // Atomic getters should return valid values
                if (!std::isfinite(param1_regular) || !std::isfinite(param1_atomic) ||
                    !std::isfinite(param2_regular) || !std::isfinite(param2_atomic)) {
                    all_passed = false;
                }
                
                // Values should be consistent (within reasonable bounds)
                if (std::abs(param1_regular - param1_atomic) > 1e-14 ||
                    std::abs(param2_regular - param2_atomic) > 1e-14) {
                    all_passed = false;
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    EXPECT_TRUE(all_passed);
}

// Test atomic getter performance
TEST_F(MyDistributionTest, AtomicGetterPerformance) {
    const int iterations = 100000;
    
    // Time regular getters
    auto start = std::chrono::high_resolution_clock::now();
    volatile double sum_regular = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum_regular += dist.getParam1();
    }
    auto regular_time = std::chrono::high_resolution_clock::now() - start;
    
    // Time atomic getters
    start = std::chrono::high_resolution_clock::now();
    volatile double sum_atomic = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum_atomic += dist.getParam1Atomic();
    }
    auto atomic_time = std::chrono::high_resolution_clock::now() - start;
    
    // Atomic getters should be competitive or faster
    double speedup = static_cast<double>(regular_time.count()) / atomic_time.count();
    EXPECT_GE(speedup, 0.8) << "Atomic getters should be at least 80% as fast as regular getters";
    
    std::cout << "Atomic getter performance: " << speedup << "x relative to regular getters" << std::endl;
}

// Test SIMD operations
TEST_F(MyDistributionTest, SIMDOperations) {
    std::vector<double> x_values = {0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5};
    
    // Get scalar results
    std::vector<double> scalar_results;
    for (double x : x_values) {
        scalar_results.push_back(dist.getProbability(x));
    }
    
    // Get SIMD results
    std::vector<double> simd_results = dist.getBatchProbabilities(x_values);
    
    // Compare results
    ASSERT_EQ(scalar_results.size(), simd_results.size());
    for (size_t i = 0; i < scalar_results.size(); ++i) {
        EXPECT_NEAR(scalar_results[i], simd_results[i], tolerance);
    }
}
```

## Common Pitfalls

### 1. Deadlock in Copy/Move Operations

**Problem:**
```cpp
// DON'T DO THIS - causes deadlock
MyDistribution& MyDistribution::operator=(const MyDistribution& other) {
    if (this != &other) {
        std::unique_lock lock(param_mutex_);
        param1_ = other.param1_; // other.param1_ needs lock too!
        param2_ = other.param2_;
        
        // This calls base class operator= which tries to acquire cache_mutex_
        // but we might already hold it, causing deadlock
        DistributionBase::operator=(other);
    }
    return *this;
}
```

**Solution:**
```cpp
// DO THIS - proper lock ordering
MyDistribution& MyDistribution::operator=(const MyDistribution& other) {
    if (this != &other) {
        std::lock(param_mutex_, other.param_mutex_);
        std::unique_lock this_lock(param_mutex_, std::adopt_lock);
        std::shared_lock other_lock(other.param_mutex_, std::adopt_lock);
        
        param1_ = other.param1_;
        param2_ = other.param2_;
        
        // Don't call base class operator= - just invalidate cache
        invalidateCache();
    }
    return *this;
}
```

### 2. Incorrect Cache Invalidation

**Problem:**
```cpp
// DON'T DO THIS - incomplete cache and atomic invalidation
void MyDistribution::setParam1(double value) {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    param1_ = value;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    // Missing: atomicParamsValid_.store(false, std::memory_order_release);
}
```

**Solution:**
```cpp
// DO THIS - ALWAYS invalidate both cache AND atomic parameters
void MyDistribution::setParam1(double value) {
    // Validate first (no locks held)
    validateParameters(value, getParam2());
    
    // Set parameter under lock
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    param1_ = value;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
    
    // CRITICAL: Must also invalidate atomic parameters
    atomicParamsValid_.store(false, std::memory_order_release);
}
```

### 3. Improper SIMD Usage

**Problem:**
```cpp
// DON'T DO THIS - no runtime checks
void MyDistribution::computeBatch(const std::vector<double>& x_values,
                                 std::vector<double>& results) const {
    // Assumes AVX2 is always available
    simd::vector_exp(x_values.data(), results.data(), x_values.size());
}
```

**Solution:**
```cpp
// DO THIS - runtime checks and fallbacks
void MyDistribution::computeBatch(const std::vector<double>& x_values,
                                 std::vector<double>& results) const {
    if (cpu::supports_avx2() && x_values.size() >= 8) {
        computeBatchSIMD(x_values, results);
    } else {
        computeBatchScalar(x_values, results);
    }
}
```

### 4. Exception Throwing in Constructors

**Problem:**
```cpp
// DON'T DO THIS - throws exceptions
MyDistribution::MyDistribution(double param1, double param2) 
    : param1_(param1), param2_(param2) {
    if (param1 <= 0.0) {
        throw std::invalid_argument("param1 must be positive");
    }
}
```

**Solution:**
```cpp
// DO THIS - use safe factory pattern
static Result<MyDistribution> MyDistribution::create(double param1, double param2) {
    if (param1 <= 0.0) {
        return Result<MyDistribution>::error(ValidationError::InvalidParameter);
    }
    return Result<MyDistribution>::success(MyDistribution(param1, param2));
}

private:
MyDistribution::MyDistribution(double param1, double param2) 
    : param1_(param1), param2_(param2) {
    // No validation here - already done in factory
}
```

This guide provides the foundation for implementing robust, thread-safe, and high-performance distribution classes in the libstats library. Always refer to existing implementations like `GaussianDistribution` and `ExponentialDistribution` for concrete examples.
