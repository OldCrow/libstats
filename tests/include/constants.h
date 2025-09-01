#pragma once

/**
 * @file tests/constants.h
 * @brief Test-specific constants, tolerances, thresholds, and iteration counts
 *
 * This header defines constants that are specifically used by the test infrastructure,
 * separate from the production constants in stats::detail. These include:
 * - Test tolerances and precision thresholds
 * - Benchmark iteration counts and thresholds
 * - Test data generation parameters
 * - Performance verification constants
 *
 * Phase 3E: Test Infrastructure Namespace
 * Part of the stats::tests:: namespace hierarchy reorganization
 */

#include <chrono>
#include <cstddef>

namespace stats {
namespace tests {
namespace constants {

//==============================================================================
// Test Precision and Tolerance Constants
//==============================================================================

/// Default tolerance for approximate equality in tests
inline constexpr double DEFAULT_TOLERANCE = 1e-10;

/// High precision tolerance for critical mathematical tests
inline constexpr double HIGH_PRECISION_TOLERANCE = 1e-12;

/// Ultra-high precision tolerance for numerical stability tests
inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1e-15;

/// Relaxed tolerance for performance-sensitive tests
inline constexpr double RELAXED_TOLERANCE = 1e-8;

/// Strict tolerance for exact mathematical computations
inline constexpr double STRICT_TOLERANCE = 1e-14;

/// Tolerance for SIMD vs scalar comparison tests
inline constexpr double SIMD_COMPARISON_TOLERANCE = 1e-12;

/// Tolerance for parallel vs sequential comparison tests
inline constexpr double PARALLEL_COMPARISON_TOLERANCE = 1e-11;

/// Tolerance for statistical approximations
inline constexpr double STATISTICAL_TOLERANCE = 1e-6;

/// Tolerance for goodness-of-fit test approximations
inline constexpr double GOF_TOLERANCE = 1e-4;

//==============================================================================
// Performance and Benchmark Constants
//==============================================================================

/// Expected minimum SIMD speedup for batch operations
inline constexpr double SIMD_SPEEDUP_MIN_EXPECTED = 1.5;

/// Expected minimum parallel speedup for large batch operations
inline constexpr double PARALLEL_SPEEDUP_MIN_EXPECTED = 2.0;

/// Default number of benchmark iterations for performance tests
inline constexpr std::size_t DEFAULT_BENCHMARK_ITERATIONS = 1000;

/// Small benchmark iteration count for quick tests
inline constexpr std::size_t SMALL_BENCHMARK_ITERATIONS = 100;

/// Large benchmark iteration count for thorough performance analysis
inline constexpr std::size_t LARGE_BENCHMARK_ITERATIONS = 10000;

/// Benchmark warmup iterations to stabilize CPU caches
inline constexpr std::size_t BENCHMARK_WARMUP_ITERATIONS = 50;

/// Maximum acceptable benchmark variance (coefficient of variation)
inline constexpr double MAX_BENCHMARK_VARIANCE = 0.20;  // 20%

/// Minimum number of measurements for statistical validity
inline constexpr std::size_t MIN_BENCHMARK_MEASUREMENTS = 5;

//==============================================================================
// Test Data Generation Constants
//==============================================================================

/// Small dataset size for basic functionality tests
inline constexpr std::size_t SMALL_DATASET_SIZE = 100;

/// Medium dataset size for performance verification
inline constexpr std::size_t MEDIUM_DATASET_SIZE = 5000;

/// Large dataset size for scalability and parallel testing
inline constexpr std::size_t LARGE_DATASET_SIZE = 50000;

/// Extra large dataset size for stress testing
inline constexpr std::size_t EXTRA_LARGE_DATASET_SIZE = 1000000;

/// Small batch size for SIMD threshold testing
inline constexpr std::size_t SMALL_BATCH_SIZE = 5;

/// Medium batch size for SIMD optimization testing
inline constexpr std::size_t MEDIUM_BATCH_SIZE = 500;

/// Large batch size for parallel processing testing
inline constexpr std::size_t LARGE_BATCH_SIZE = 50000;

/// Maximum number of random samples to generate for tests
inline constexpr std::size_t MAX_RANDOM_SAMPLES = 100000;

/// Default random seed for reproducible tests
inline constexpr uint32_t DEFAULT_RANDOM_SEED = 12345;

/// Number of different parameter sets to test per distribution
inline constexpr std::size_t TEST_PARAMETER_SETS = 6;

//==============================================================================
// Statistical Test Constants
//==============================================================================

/// Default significance level for statistical tests
inline constexpr double DEFAULT_ALPHA = 0.05;

/// Strict significance level for critical tests
inline constexpr double STRICT_ALPHA = 0.01;

/// Relaxed significance level for exploratory tests
inline constexpr double RELAXED_ALPHA = 0.10;

/// Number of bootstrap samples for bootstrap tests
inline constexpr std::size_t DEFAULT_BOOTSTRAP_SAMPLES = 1000;

/// Minimum number of bootstrap samples for validity
inline constexpr std::size_t MIN_BOOTSTRAP_SAMPLES = 100;

/// Maximum number of bootstrap samples for thorough testing
inline constexpr std::size_t MAX_BOOTSTRAP_SAMPLES = 10000;

/// Default confidence level for confidence intervals
inline constexpr double DEFAULT_CONFIDENCE_LEVEL = 0.95;

/// Number of cross-validation folds for model validation
inline constexpr std::size_t DEFAULT_CV_FOLDS = 5;

/// Minimum data points required for statistical tests
inline constexpr std::size_t MIN_DATA_POINTS_FOR_TESTS = 10;

/// Maximum data points for exact statistical computations
inline constexpr std::size_t MAX_DATA_POINTS_FOR_EXACT = 1000;

//==============================================================================
// Numerical Validation Constants
//==============================================================================

/// Maximum acceptable relative error in numerical computations
inline constexpr double MAX_RELATIVE_ERROR = 1e-10;

/// Maximum acceptable absolute error in probability computations
inline constexpr double MAX_PROBABILITY_ERROR = 1e-12;

/// Maximum acceptable log-probability error
inline constexpr double MAX_LOG_PROBABILITY_ERROR = 1e-10;

/// Minimum probability value for numerical stability tests
inline constexpr double MIN_TEST_PROBABILITY = 1e-15;

/// Maximum probability value for numerical stability tests
inline constexpr double MAX_TEST_PROBABILITY = 1.0 - 1e-15;

/// Number of test points for comprehensive parameter sweep
inline constexpr std::size_t PARAMETER_SWEEP_POINTS = 50;

/// Number of extreme values to test for edge cases
inline constexpr std::size_t EXTREME_VALUE_TEST_COUNT = 10;

//==============================================================================
// Threading and Concurrency Test Constants
//==============================================================================

/// Number of concurrent threads for thread safety testing
inline constexpr std::size_t THREAD_SAFETY_TEST_THREADS = 8;

/// Number of operations per thread in concurrency tests
inline constexpr std::size_t OPERATIONS_PER_THREAD = 1000;

/// Maximum time to wait for thread completion (milliseconds)
inline constexpr std::chrono::milliseconds THREAD_TIMEOUT{5000};

/// Number of parallel batch operations for race condition testing
inline constexpr std::size_t PARALLEL_BATCH_COUNT = 10;

//==============================================================================
// Cache and Memory Test Constants
//==============================================================================

/// Number of cache invalidation operations for cache testing
inline constexpr std::size_t CACHE_INVALIDATION_COUNT = 100;

/// Memory allocation size for large data structure tests
inline constexpr std::size_t LARGE_ALLOCATION_SIZE = 10000000;  // 10M elements

/// Number of memory pressure iterations for stress testing
inline constexpr std::size_t MEMORY_PRESSURE_ITERATIONS = 50;

/// Maximum memory usage growth acceptable in leak tests (bytes)
inline constexpr std::size_t MAX_MEMORY_GROWTH_BYTES = 1024 * 1024;  // 1MB

//==============================================================================
// Distribution-Specific Test Constants
//==============================================================================

/// Range of parameter values for Gaussian distribution tests
namespace gaussian {
inline constexpr double MIN_MEAN = -100.0;
inline constexpr double MAX_MEAN = 100.0;
inline constexpr double MIN_SIGMA = 0.1;
inline constexpr double MAX_SIGMA = 50.0;
}  // namespace gaussian

/// Range of parameter values for Exponential distribution tests
namespace exponential {
inline constexpr double MIN_LAMBDA = 0.01;
inline constexpr double MAX_LAMBDA = 100.0;
}  // namespace exponential

/// Range of parameter values for Uniform distribution tests
namespace uniform {
inline constexpr double MIN_LOWER = -100.0;
inline constexpr double MAX_UPPER = 100.0;
inline constexpr double MIN_RANGE = 0.1;
}  // namespace uniform

/// Range of parameter values for Poisson distribution tests
namespace poisson {
inline constexpr double MIN_LAMBDA = 0.1;
inline constexpr double MAX_LAMBDA = 1000.0;
inline constexpr int MAX_K_VALUE = 2000;
}  // namespace poisson

/// Range of parameter values for Gamma distribution tests
namespace gamma {
inline constexpr double MIN_ALPHA = 0.1;
inline constexpr double MAX_ALPHA = 100.0;
inline constexpr double MIN_BETA = 0.01;
inline constexpr double MAX_BETA = 100.0;
}  // namespace gamma

/// Range of parameter values for Discrete distribution tests
namespace discrete {
inline constexpr int MIN_LOWER = -50;
inline constexpr int MAX_UPPER = 50;
inline constexpr int MIN_RANGE = 2;
inline constexpr int MAX_RANGE = 100;
}  // namespace discrete

}  // namespace constants
}  // namespace tests
}  // namespace stats
