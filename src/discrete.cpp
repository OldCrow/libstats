#include "libstats/distributions/discrete.h"
using stats::detail::validateParameter;
using stats::detail::validatePositiveParameter;
using stats::detail::validateNonNegativeParameter;

#include "libstats/core/parallel_batch_fit.h"

// Core functionality - lightweight headers
#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/dispatch_utils.h"
#include "libstats/core/log_space_ops.h"
#include "libstats/core/math_utils.h"
#include "libstats/core/statistical_constants.h"

// Platform headers - use forward declarations where available
#include "libstats/common/cpu_detection_fwd.h"  // Lightweight CPU detection
// Note: parallel_execution.h is transitively included via dispatch_utils.h
// Note: thread_pool.h and work_stealing_pool.h are transitively included via dispatch_utils.h

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>

namespace stats {

//==============================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==============================================================================

DiscreteDistribution::DiscreteDistribution(int a, int b) : DistributionBase(), a_(a), b_(b) {
    validateParameters(a, b);
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

DiscreteDistribution::DiscreteDistribution(const DiscreteDistribution& other)
    : DistributionBase(other) {
    std::shared_lock<std::shared_mutex> lock(other.cache_mutex_);
    a_ = other.a_;
    b_ = other.b_;

    // If the other's cache is valid, copy cached values for efficiency
    if (other.cache_valid_) {
        range_ = other.range_;
        probability_ = other.probability_;
        mean_ = other.mean_;
        variance_ = other.variance_;
        logProbability_ = other.logProbability_;
        isBinary_ = other.isBinary_;
        isStandardDie_ = other.isStandardDie_;
        isSymmetric_ = other.isSymmetric_;
        isSmallRange_ = other.isSmallRange_;
        isLargeRange_ = other.isLargeRange_;
        cache_valid_ = true;
        cacheValidAtomic_.store(true, std::memory_order_release);

        // Update atomic parameters
        atomicA_.store(a_, std::memory_order_release);
        atomicB_.store(b_, std::memory_order_release);
        atomicParamsValid_.store(true, std::memory_order_release);
    } else {
        // Cache will be updated on first use
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        atomicParamsValid_.store(false, std::memory_order_release);
    }
}

DiscreteDistribution& DiscreteDistribution::operator=(const DiscreteDistribution& other) {
    if (this != &other) {
        // Acquire locks in a consistent order to prevent deadlock
        std::unique_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
        std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
        std::lock(lock1, lock2);

        // Copy parameters (don't call base class operator= to avoid deadlock)
        a_ = other.a_;
        b_ = other.b_;

        // If the other's cache is valid, copy cached values for efficiency
        if (other.cache_valid_) {
            range_ = other.range_;
            probability_ = other.probability_;
            mean_ = other.mean_;
            variance_ = other.variance_;
            logProbability_ = other.logProbability_;
            isBinary_ = other.isBinary_;
            isStandardDie_ = other.isStandardDie_;
            isSymmetric_ = other.isSymmetric_;
            isSmallRange_ = other.isSmallRange_;
            isLargeRange_ = other.isLargeRange_;
            cache_valid_ = true;
            cacheValidAtomic_.store(true, std::memory_order_release);

            // Update atomic parameters
            atomicA_.store(a_, std::memory_order_release);
            atomicB_.store(b_, std::memory_order_release);
            atomicParamsValid_.store(true, std::memory_order_release);
        } else {
            // Cache will be updated on first use
            cache_valid_ = false;
            cacheValidAtomic_.store(false, std::memory_order_release);
            atomicParamsValid_.store(false, std::memory_order_release);
        }
    }
    return *this;
}

DiscreteDistribution::DiscreteDistribution(DiscreteDistribution&& other) noexcept
    : DistributionBase(std::move(other)) {
    a_ = other.a_;
    b_ = other.b_;
    other.a_ = detail::ZERO_INT;
    other.b_ = detail::ONE_INT;
    other.cache_valid_ = false;
    other.cacheValidAtomic_.store(false, std::memory_order_release);
    // Cache will be updated on first use
}

DiscreteDistribution& DiscreteDistribution::operator=(DiscreteDistribution&& other) noexcept {
    if (this != &other) {

        a_ = other.a_;
        b_ = other.b_;
        other.a_ = detail::ZERO_INT;
        other.b_ = detail::ONE_INT;

        cache_valid_ = false;
        other.cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
        other.cacheValidAtomic_.store(false, std::memory_order_release);
    }
    return *this;
}

//==========================================================================
// 2. SAFE FACTORY METHODS (Exception-free construction)
//==========================================================================

// Note: All methods in this section currently implemented inline in the header
// This section maintained for template compliance

//==============================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==============================================================================

double DiscreteDistribution::getMean() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return mean_;
}

double DiscreteDistribution::getVariance() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return variance_;
}

void DiscreteDistribution::setLowerBound(int a) {
    validateParameters(a, b_);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::setUpperBound(int b) {
    validateParameters(a_, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::setBounds(int a, int b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::setParameters(int a, int b) {
    validateParameters(a, b);
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // CRITICAL: Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);
}

// Implementation moved from header - no longer inline
int DiscreteDistribution::getLowerBound() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return a_;
}

// Implementation moved from header - no longer inline
int DiscreteDistribution::getUpperBound() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return b_;
}

// Simple getters for constant values - no longer inline
double DiscreteDistribution::getSkewness() const noexcept {
    return 0.0;  // Discrete uniform distribution is perfectly symmetric
}

double DiscreteDistribution::getKurtosis() const noexcept {
    // For discrete uniform: excess kurtosis ≈ -1.2 for large ranges
    // For exact calculation: -6/5 * (n²+1)/(n²-1) where n = b-a+1
    // But approximation is sufficient for performance-critical inline method
    return -1.2;
}

int DiscreteDistribution::getNumParameters() const noexcept {
    return 2;  // Lower bound (a) and upper bound (b)
}

std::string DiscreteDistribution::getDistributionName() const {
    return "DiscreteUniform";
}

bool DiscreteDistribution::isDiscrete() const noexcept {
    return true;  // Discrete uniform distribution is always discrete
}

double DiscreteDistribution::getSupportLowerBound() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return static_cast<double>(a_);
}

double DiscreteDistribution::getSupportUpperBound() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return static_cast<double>(b_);
}

double DiscreteDistribution::getMode() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;
}

double DiscreteDistribution::getMedian() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;
}

int DiscreteDistribution::getRange() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return range_;
}

double DiscreteDistribution::getSingleOutcomeProbability() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }
    return probability_;
}

//==============================================================================
// 4. RESULT-BASED SETTERS (C++20 Best Practice: Complex implementations in .cpp)
//==============================================================================

VoidResult DiscreteDistribution::trySetLowerBound(int a) noexcept {
    // Copy current upper bound for validation (thread-safe)
    int currentB;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentB = b_;
    }

    auto validation = validateDiscreteParameters(a, currentB);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult DiscreteDistribution::trySetUpperBound(int b) noexcept {
    // Copy current lower bound for validation (thread-safe)
    int currentA;
    {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        currentA = a_;
    }

    auto validation = validateDiscreteParameters(currentA, b);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult DiscreteDistribution::trySetBounds(int a, int b) noexcept {
    auto validation = validateDiscreteParameters(a, b);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

VoidResult DiscreteDistribution::trySetParameters(int a, int b) noexcept {
    auto validation = validateDiscreteParameters(a, b);
    if (validation.isError()) {
        return validation;
    }

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = a;
    b_ = b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);

    // Invalidate atomic parameters when parameters change
    atomicParamsValid_.store(false, std::memory_order_release);

    return VoidResult::ok({});
}

//==============================================================================
// 5. CORE PROBABILITY METHODS
//==============================================================================

double DiscreteDistribution::getProbability(double x) const {
    // For discrete distribution, check if x is an integer in range
    if (std::floor(x) != x) {
        return detail::ZERO_DOUBLE;  // Not an integer
    }

    const int k = static_cast<int>(x);
    if (k < a_ || k > b_) {
        return detail::ZERO_DOUBLE;  // Outside support
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path optimizations
    if (isBinary_) {
        return detail::HALF;  // detail::AD_THRESHOLD_1 for binary [0,1]
    }

    return probability_;  // 1/(b-a+1)
}

double DiscreteDistribution::getLogProbability(double x) const noexcept {
    // For discrete distribution, check if x is an integer in range
    if (std::floor(x) != x) {
        return detail::NEGATIVE_INFINITY;  // Not an integer
    }

    const int k = static_cast<int>(x);
    if (k < a_ || k > b_) {
        return detail::NEGATIVE_INFINITY;  // Outside support
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path optimizations
    if (isBinary_) {
        return -detail::LN2;  // log(detail::AD_THRESHOLD_1)
    }

    return logProbability_;  // -log(b-a+1)
}

double DiscreteDistribution::getCumulativeProbability(double x) const {
    if (x < static_cast<double>(a_)) {
        return detail::ZERO_DOUBLE;
    }
    if (x >= static_cast<double>(b_)) {
        return detail::ONE;
    }

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // For discrete uniform: F(k) = (floor(k) - a + 1) / (b - a + 1)
    const int k = static_cast<int>(std::floor(x));
    const int numerator = k - a_ + detail::ONE_INT;

    // Fast path optimizations
    if (isBinary_) {
        return (k >= 0) ? detail::ONE : detail::ZERO_DOUBLE;
    }

    return static_cast<double>(numerator) / static_cast<double>(range_);
}

double DiscreteDistribution::getQuantile(double p) const {
    if (p < detail::ZERO_DOUBLE || p > detail::ONE) {
        throw std::invalid_argument("Probability must be between 0 and 1");
    }

    if (p == detail::ZERO_DOUBLE)
        return static_cast<double>(a_);
    if (p == detail::ONE)
        return static_cast<double>(b_);

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // For discrete uniform: quantile(p) = a + ceil(p * (b-a+1)) - 1
    // But we need to handle edge cases carefully
    const double scaled = p * static_cast<double>(range_);
    const int k = static_cast<int>(std::ceil(scaled)) - 1;

    return static_cast<double>(a_ + std::max(0, std::min(k, range_ - 1)));
}

double DiscreteDistribution::sample(std::mt19937& rng) const {
    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        return static_cast<double>(dis(rng));
    }

    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    return static_cast<double>(dis(rng));
}

std::vector<double> DiscreteDistribution::sample(std::mt19937& rng, std::size_t n) const {
    std::vector<double> samples(n);

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        for (size_t i = 0; i < n; ++i) {
            samples[i] = static_cast<double>(dis(rng));
        }
        return samples;
    }

    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    for (size_t i = 0; i < n; ++i) {
        samples[i] = static_cast<double>(dis(rng));
    }

    return samples;
}

//==============================================================================
// 6. DISTRIBUTION MANAGEMENT
//==============================================================================

void DiscreteDistribution::fit(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot fit discrete distribution to empty data");
    }

    // For discrete uniform, we round fractional values to nearest integers
    // then find the min and max of the rounded values
    std::vector<int> rounded_values;
    rounded_values.reserve(values.size());

    for (double val : values) {
        if (!isValidIntegerValue(val)) {
            throw std::invalid_argument("Value outside valid integer range");
        }
        rounded_values.push_back(roundToInt(val));
    }

    int new_a = *std::min_element(rounded_values.begin(), rounded_values.end());
    int new_b = *std::max_element(rounded_values.begin(), rounded_values.end());

    validateParameters(new_a, new_b);

    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = new_a;
    b_ = new_b;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

void DiscreteDistribution::parallelBatchFit(const std::vector<std::vector<double>>& datasets,
                                            std::vector<DiscreteDistribution>& results) {
    detail::batchFitParallel(datasets, results);
}

void DiscreteDistribution::reset() noexcept {
    std::unique_lock<std::shared_mutex> lock(cache_mutex_);
    a_ = detail::ZERO_INT;
    b_ = detail::ONE_INT;
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

std::string DiscreteDistribution::toString() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    std::ostringstream oss;
    oss << "DiscreteUniform(a=" << a_ << ", b=" << b_ << ")";
    return oss.str();
}

//==========================================================================
// 7. ADVANCED STATISTICAL METHODS
//==========================================================================

std::tuple<double, double, bool> DiscreteDistribution::discreteUniformityTest(
    const std::vector<double>& data, double significance_level) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (significance_level <= detail::ZERO_DOUBLE || significance_level >= detail::ONE) {
        throw std::invalid_argument("Significance level must be between 0 and 1");
    }

    // Convert to integers and find range
    std::map<int, int> frequency_map;
    int min_val = std::numeric_limits<int>::max();
    int max_val = std::numeric_limits<int>::min();
    int total_count = 0;

    for (double val : data) {
        if (std::floor(val) == val && isValidIntegerValue(val)) {
            int int_val = static_cast<int>(val);
            frequency_map[int_val]++;
            min_val = std::min(min_val, int_val);
            max_val = std::max(max_val, int_val);
            total_count++;
        }
    }

    if (total_count == 0) {
        throw std::invalid_argument("No valid integer values in data");
    }

    const int range = max_val - min_val + detail::ONE_INT;
    const double expected_frequency = static_cast<double>(total_count) / range;

    // Chi-squared test for uniformity
    double chi_squared = detail::ZERO_DOUBLE;
    for (int k = min_val; k <= max_val; ++k) {
        const int observed = frequency_map[k];  // defaults to 0 if not found
        const double diff = observed - expected_frequency;
        chi_squared += (diff * diff) / expected_frequency;
    }

    // Degrees of freedom = number of categories - 1
    [[maybe_unused]] const int degrees_of_freedom = range - detail::ONE_INT;

    // Simple p-value approximation
    const double critical_value =
        detail::CHI2_95_DF_1;  // Chi-squared critical value for alpha=detail::ALPHA_05, df=1
    double p_value = (chi_squared > critical_value)
                         ? detail::ALPHA_01
                         : detail::AD_THRESHOLD_1;  // Rough approximation

    const bool reject_uniformity = p_value < significance_level;

    return std::make_tuple(chi_squared, p_value, reject_uniformity);
}

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================

std::tuple<double, double, bool> DiscreteDistribution::chiSquaredGoodnessOfFitTest(
    const std::vector<double>& data, const DiscreteDistribution& distribution, double alpha) {
    if (data.empty()) {
        throw std::invalid_argument("Data vector cannot be empty");
    }

    if (alpha <= detail::ZERO_DOUBLE || alpha >= detail::ONE) {
        throw std::invalid_argument("Alpha must be between 0 and 1");
    }

    // Get distribution parameters
    const int a = distribution.getLowerBound();
    const int b = distribution.getUpperBound();
    const int range = b - a + detail::ONE_INT;

    // Count observed frequencies for each possible outcome
    std::map<int, int> observed_counts;
    int total_count = 0;

    for (double value : data) {
        if (std::floor(value) == value && value >= a && value <= b) {
            int k = static_cast<int>(value);
            observed_counts[k]++;
            total_count++;
        }
    }

    // Calculate expected frequency for each outcome
    const double expected_freq = static_cast<double>(total_count) / range;

    // Check minimum expected frequency requirement (typically >= 5)
    if (expected_freq < detail::FIVE) {
        // Chi-squared test may not be reliable with low expected frequencies
        // But we'll proceed with a warning
    }

    // Calculate chi-squared statistic
    double chi_squared = detail::ZERO_DOUBLE;
    for (int k = a; k <= b; ++k) {
        const int observed = observed_counts[k];  // defaults to 0 if not found
        const double diff = observed - expected_freq;
        chi_squared += (diff * diff) / expected_freq;
    }

    // Degrees of freedom = number of categories - 1 - number of estimated parameters
    // For discrete uniform, we estimate 0 parameters (a and b are given)
    const int degrees_of_freedom = range - detail::ONE_INT;

    // Calculate p-value using chi-squared distribution
    // For simplicity, we'll use a basic approximation
    // In a full implementation, you'd use a proper chi-squared CDF
    const double critical_value =
        detail::CHI2_95_DF_1;  // Chi-squared critical value for alpha=detail::ALPHA_05, df=1

    // Simple p-value approximation (this should use proper chi-squared CDF)
    double p_value;
    if (degrees_of_freedom == 1) {
        p_value = (chi_squared > critical_value) ? detail::ALPHA_01
                                                 : detail::AD_THRESHOLD_1;  // Rough approximation
    } else {
        // For higher df, use a rough approximation
        const double mean_chi = degrees_of_freedom;
        const double std_chi = std::sqrt(detail::TWO * degrees_of_freedom);
        const double z_score = (chi_squared - mean_chi) / std_chi;
        p_value = (z_score > detail::Z_95)
                      ? 0.025
                      : detail::AD_THRESHOLD_1;  // Very rough normal approximation
    }

    const bool reject_null = p_value < alpha;

    return std::make_tuple(chi_squared, p_value, reject_null);
}

//==============================================================================
// 9. CROSS-VALIDATION METHODS
//==============================================================================

//==========================================================================
// 10. INFORMATION CRITERIA
//==========================================================================

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================

//==========================================================================
// 12. DISTRIBUTION-SPECIFIC UTILITY METHODS
//==========================================================================

std::vector<int> DiscreteDistribution::sampleIntegers(std::mt19937& rng, std::size_t count) const {
    std::vector<int> samples(count);

    // Ensure cache is valid
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    if (!cache_valid_) {
        lock.unlock();
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        ulock.unlock();
        lock.lock();
    }

    // Fast path for binary distribution
    if (isBinary_) {
        std::uniform_int_distribution<int> dis(0, 1);
        for (size_t i = 0; i < count; ++i) {
            samples[i] = dis(rng);
        }
        return samples;
    }

    // General case: uniform integer distribution
    std::uniform_int_distribution<int> dis(a_, b_);
    for (size_t i = 0; i < count; ++i) {
        samples[i] = dis(rng);
    }

    return samples;
}

bool DiscreteDistribution::isInSupport(double x) const noexcept {
    // Check if x is an integer in the range [a, b]
    if (std::floor(x) != x) {
        return false;  // Not an integer
    }

    if (!isValidIntegerValue(x)) {
        return false;  // Outside integer bounds
    }

    const int k = static_cast<int>(x);

    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return (k >= a_ && k <= b_);
}

std::vector<int> DiscreteDistribution::getAllOutcomes() const {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);

    // Safety check for reasonable range size
    const int range = b_ - a_ + detail::ONE_INT;

    if (range > 1000000) {  // 1M elements max (4MB memory)
        throw std::runtime_error(
            "Range too large for getAllOutcomes() - maximum 1,000,000 elements");
    }

    if (range > 10000) {  // Warning for large ranges
        // Could log a warning here if logging system exists
        // For now, just proceed but this indicates potentially expensive operation
    }

    std::vector<int> outcomes;
    outcomes.reserve(static_cast<std::size_t>(range));

    for (int k = a_; k <= b_; ++k) {
        outcomes.push_back(k);
    }

    return outcomes;
}

VoidResult DiscreteDistribution::validateCurrentParameters() const noexcept {
    std::shared_lock<std::shared_mutex> lock(cache_mutex_);
    return validateDiscreteParameters(a_, b_);
}

int DiscreteDistribution::getLowerBoundAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicA_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getLowerBound();
}

int DiscreteDistribution::getUpperBoundAtomic() const noexcept {
    // Fast path: check if atomic parameters are valid
    if (atomicParamsValid_.load(std::memory_order_acquire)) {
        // Lock-free atomic access with proper memory ordering
        return atomicB_.load(std::memory_order_acquire);
    }

    // Fallback: use traditional locked getter if atomic parameters are stale
    return getUpperBound();
}

//==============================================================================
// 13. SMART AUTO-DISPATCH BATCH METHODS
//==============================================================================

void DiscreteDistribution::getProbability(std::span<const double> values, std::span<double> results,
                                          const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::PDF,
        [](const DiscreteDistribution& dist, double value) { return dist.getProbability(value); },
        [](const DiscreteDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_prob = dist.probability_;
            lock.unlock();

            // Call private implementation directly
            dist.getProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b, cached_prob);
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_prob = dist.probability_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (std::floor(vals[i]) == vals[i] &&
                        DiscreteDistribution::isValidIntegerValue(vals[i])) {
                        const int k = static_cast<int>(vals[i]);
                        res[i] =
                            (k >= cached_a && k <= cached_b) ? cached_prob : detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::ZERO_DOUBLE;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (std::floor(vals[i]) == vals[i] &&
                        DiscreteDistribution::isValidIntegerValue(vals[i])) {
                        const int k = static_cast<int>(vals[i]);
                        res[i] =
                            (k >= cached_a && k <= cached_b) ? cached_prob : detail::ZERO_DOUBLE;
                    } else {
                        res[i] = detail::ZERO_DOUBLE;
                    }
                }
            }
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_prob = dist.probability_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (std::floor(vals[i]) == vals[i] &&
                    DiscreteDistribution::isValidIntegerValue(vals[i])) {
                    const int k = static_cast<int>(vals[i]);
                    res[i] = (k >= cached_a && k <= cached_b) ? cached_prob : detail::ZERO_DOUBLE;
                } else {
                    res[i] = detail::ZERO_DOUBLE;
                }
            });
        });
}

void DiscreteDistribution::getLogProbability(std::span<const double> values,
                                             std::span<double> results,
                                             const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::LOG_PDF,
        [](const DiscreteDistribution& dist, double value) {
            return dist.getLogProbability(value);
        },
        [](const DiscreteDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_log_prob = dist.logProbability_;
            lock.unlock();

            // Call private implementation directly
            dist.getLogProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                  cached_log_prob);
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_log_prob = dist.logProbability_;
            const bool cached_is_binary = dist.isBinary_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (std::floor(vals[i]) == vals[i] &&
                        DiscreteDistribution::isValidIntegerValue(vals[i])) {
                        const int k = static_cast<int>(vals[i]);
                        if (k >= cached_a && k <= cached_b) {
                            res[i] = cached_is_binary ? -detail::LN2 : cached_log_prob;
                        } else {
                            res[i] = detail::NEGATIVE_INFINITY;
                        }
                    } else {
                        res[i] = detail::NEGATIVE_INFINITY;
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (std::floor(vals[i]) == vals[i] &&
                        DiscreteDistribution::isValidIntegerValue(vals[i])) {
                        const int k = static_cast<int>(vals[i]);
                        if (k >= cached_a && k <= cached_b) {
                            res[i] = cached_is_binary ? -detail::LN2 : cached_log_prob;
                        } else {
                            res[i] = detail::NEGATIVE_INFINITY;
                        }
                    } else {
                        res[i] = detail::NEGATIVE_INFINITY;
                    }
                }
            }
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_log_prob = dist.logProbability_;
            const bool cached_is_binary = dist.isBinary_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (std::floor(vals[i]) == vals[i] &&
                    DiscreteDistribution::isValidIntegerValue(vals[i])) {
                    const int k = static_cast<int>(vals[i]);
                    if (k >= cached_a && k <= cached_b) {
                        res[i] = cached_is_binary ? -detail::LN2 : cached_log_prob;
                    } else {
                        res[i] = detail::NEGATIVE_INFINITY;
                    }
                } else {
                    res[i] = detail::NEGATIVE_INFINITY;
                }
            });
        });
}

void DiscreteDistribution::getCumulativeProbability(std::span<const double> values,
                                                    std::span<double> results,
                                                    const detail::PerformanceHint& hint) const {
    detail::DispatchUtils::autoDispatch(
        *this, values, results, hint, detail::OperationType::CDF,
        [](const DiscreteDistribution& dist, double value) {
            return dist.getCumulativeProbability(value);
        },
        [](const DiscreteDistribution& dist, const double* vals, double* res, size_t count) {
            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution&>(dist).updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for batch processing
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const double cached_inv_range = detail::ONE / static_cast<double>(dist.range_);
            lock.unlock();

            // Call private implementation directly
            dist.getCumulativeProbabilityBatchUnsafeImpl(vals, res, count, cached_a, cached_b,
                                                         cached_inv_range);
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe parallel access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const int cached_range = dist.range_;
            const bool cached_is_binary = dist.isBinary_;
            lock.unlock();

            // Use ParallelUtils::parallelFor for Level 0-3 integration
            if (arch::should_use_parallel(count)) {
                ParallelUtils::parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                    if (vals[i] < static_cast<double>(cached_a)) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (vals[i] >= static_cast<double>(cached_b)) {
                        res[i] = detail::ONE;
                    } else {
                        const int k = static_cast<int>(std::floor(vals[i]));
                        if (cached_is_binary) {
                            res[i] = (k >= 0) ? detail::ONE : detail::ZERO_DOUBLE;
                        } else {
                            const int numerator = k - cached_a + detail::ONE_INT;
                            res[i] =
                                static_cast<double>(numerator) / static_cast<double>(cached_range);
                        }
                    }
                });
            } else {
                // Serial processing for small datasets
                for (std::size_t i = 0; i < count; ++i) {
                    if (vals[i] < static_cast<double>(cached_a)) {
                        res[i] = detail::ZERO_DOUBLE;
                    } else if (vals[i] >= static_cast<double>(cached_b)) {
                        res[i] = detail::ONE;
                    } else {
                        const int k = static_cast<int>(std::floor(vals[i]));
                        if (cached_is_binary) {
                            res[i] = (k >= 0) ? detail::ONE : detail::ZERO_DOUBLE;
                        } else {
                            const int numerator = k - cached_a + detail::ONE_INT;
                            res[i] =
                                static_cast<double>(numerator) / static_cast<double>(cached_range);
                        }
                    }
                }
            }
        },
        [](const DiscreteDistribution& dist, std::span<const double> vals, std::span<double> res,
           WorkStealingPool& pool) {
            if (vals.size() != res.size()) {
                throw std::invalid_argument("Input and output spans must have the same size");
            }

            const std::size_t count = vals.size();
            if (count == 0)
                return;

            // Ensure cache is valid
            std::shared_lock<std::shared_mutex> lock(dist.cache_mutex_);
            if (!dist.cache_valid_) {
                lock.unlock();
                std::unique_lock<std::shared_mutex> ulock(dist.cache_mutex_);
                if (!dist.cache_valid_) {
                    const_cast<DiscreteDistribution*>(&dist)->updateCacheUnsafe();
                }
                ulock.unlock();
                lock.lock();
            }

            // Cache parameters for thread-safe work-stealing access
            const int cached_a = dist.a_;
            const int cached_b = dist.b_;
            const int cached_range = dist.range_;
            const bool cached_is_binary = dist.isBinary_;
            lock.unlock();

            // Use work-stealing pool for dynamic load balancing
            pool.parallelFor(std::size_t{0}, count, [&](std::size_t i) {
                if (vals[i] < static_cast<double>(cached_a)) {
                    res[i] = detail::ZERO_DOUBLE;
                } else if (vals[i] >= static_cast<double>(cached_b)) {
                    res[i] = detail::ONE;
                } else {
                    const int k = static_cast<int>(std::floor(vals[i]));
                    if (cached_is_binary) {
                        res[i] = (k >= 0) ? detail::ONE : detail::ZERO_DOUBLE;
                    } else {
                        const int numerator = k - cached_a + detail::ONE_INT;
                        res[i] = static_cast<double>(numerator) / static_cast<double>(cached_range);
                    }
                }
            });
        });
}

//==============================================================================
// 14. EXPLICIT STRATEGY BATCH METHODS (Power User Interface)
//==============================================================================

//==============================================================================
// 15. COMPARISON OPERATORS
//==============================================================================

bool DiscreteDistribution::operator==(const DiscreteDistribution& other) const {
    // Thread-safe comparison with ordered lock acquisition
    if (this == &other)
        return true;

    std::shared_lock<std::shared_mutex> lock1(cache_mutex_, std::defer_lock);
    std::shared_lock<std::shared_mutex> lock2(other.cache_mutex_, std::defer_lock);
    std::lock(lock1, lock2);

    return (a_ == other.a_) && (b_ == other.b_);
}

// Implementation moved from header - no longer inline
bool DiscreteDistribution::operator!=(const DiscreteDistribution& other) const {
    return !(*this == other);
}

//==============================================================================
// 16. STREAM OPERATORS
//==============================================================================

std::ostream& operator<<(std::ostream& os, const DiscreteDistribution& distribution) {
    return os << distribution.toString();
}

std::istream& operator>>(std::istream& is, DiscreteDistribution& distribution) {
    std::string token;
    int a, b;

    // Expected format: "DiscreteUniform(a=<value>, b=<value>)"
    // We'll parse this step by step

    // Skip whitespace and read the first part
    is >> token;
    if (!token.starts_with("DiscreteUniform(")) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract a value
    if (token.find("a=") == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    size_t a_pos = token.find("a=") + 2;
    size_t comma_pos = token.find(",", a_pos);
    if (comma_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string a_str = token.substr(a_pos, comma_pos - a_pos);
        a = std::stoi(a_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Extract b value
    size_t b_pos = token.find("b=", comma_pos);
    if (b_pos == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }
    b_pos += 2;

    size_t close_paren = token.find(")", b_pos);
    if (close_paren == std::string::npos) {
        is.setstate(std::ios::failbit);
        return is;
    }

    try {
        std::string b_str = token.substr(b_pos, close_paren - b_pos);
        b = std::stoi(b_str);
    } catch (...) {
        is.setstate(std::ios::failbit);
        return is;
    }

    // Validate and set parameters using the safe API
    auto result = distribution.trySetParameters(a, b);
    if (result.isError()) {
        is.setstate(std::ios::failbit);
    }

    return is;
}

//==========================================================================
// 17. PRIVATE FACTORY METHODS
//==========================================================================

// Implementation moved from header - NOT inline (needs external linkage)
DiscreteDistribution DiscreteDistribution::createUnchecked(int a, int b) noexcept {
    DiscreteDistribution dist(a, b, true);  // bypass validation
    return dist;
}

// Implementation moved from header - NOT inline (private constructor)
DiscreteDistribution::DiscreteDistribution(int a, int b, bool /*bypassValidation*/) noexcept
    : DistributionBase(), a_(a), b_(b) {
    // Cache will be updated on first use
    cache_valid_ = false;
    cacheValidAtomic_.store(false, std::memory_order_release);
}

//==============================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==============================================================================

void DiscreteDistribution::getProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                         std::size_t count, int a, int b,
                                                         double probability) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: Discrete distributions with integer checking are not well-suited to SIMD
        // but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
                const int k = static_cast<int>(values[i]);
                results[i] = (k >= a && k <= b) ? probability : detail::ZERO_DOUBLE;
            } else {
                results[i] = detail::ZERO_DOUBLE;
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For discrete distributions, vectorization typically doesn't provide significant
    // benefits due to the nature of integer checking and branching logic, but we implement for
    // consistency In practice, this will mostly fall back to scalar due to the nature of the
    // operation

    // Use scalar implementation even when SIMD is available because discrete distribution
    // operations are not amenable to vectorization (primarily integer checking with branches)
    for (std::size_t i = 0; i < count; ++i) {
        if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
            const int k = static_cast<int>(values[i]);
            results[i] = (k >= a && k <= b) ? probability : detail::ZERO_DOUBLE;
        } else {
            results[i] = detail::ZERO_DOUBLE;
        }
    }
}

void DiscreteDistribution::getLogProbabilityBatchUnsafeImpl(const double* values, double* results,
                                                            std::size_t count, int a, int b,
                                                            double log_probability) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: Discrete distributions with integer checking are not well-suited to SIMD
        // but we use centralized policy for consistency
        for (std::size_t i = 0; i < count; ++i) {
            if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
                const int k = static_cast<int>(values[i]);
                results[i] = (k >= a && k <= b) ? log_probability : detail::NEGATIVE_INFINITY;
            } else {
                results[i] = detail::NEGATIVE_INFINITY;
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For discrete distributions, vectorization typically doesn't provide significant
    // benefits due to the nature of integer checking and branching logic, but we implement for
    // consistency In practice, this will mostly fall back to scalar due to the nature of the
    // operation

    // Use scalar implementation even when SIMD is available because discrete distribution
    // operations are not amenable to vectorization (primarily integer checking with branches)
    for (std::size_t i = 0; i < count; ++i) {
        if (std::floor(values[i]) == values[i] && isValidIntegerValue(values[i])) {
            const int k = static_cast<int>(values[i]);
            results[i] = (k >= a && k <= b) ? log_probability : detail::NEGATIVE_INFINITY;
        } else {
            results[i] = detail::NEGATIVE_INFINITY;
        }
    }
}

void DiscreteDistribution::getCumulativeProbabilityBatchUnsafeImpl(
    const double* values, double* results, std::size_t count, int a, int b,
    double inv_range) const noexcept {
    // Check if vectorization is beneficial and CPU supports it (following centralized SIMDPolicy)
    const bool use_simd = arch::simd::SIMDPolicy::shouldUseSIMD(count);

    if (!use_simd) {
        // Use scalar implementation for small arrays or when SIMD overhead isn't beneficial
        // Note: Discrete CDF computation involves comparisons and arithmetic
        // so SIMD rarely provides benefits for discrete distributions
        for (std::size_t i = 0; i < count; ++i) {
            if (values[i] < static_cast<double>(a)) {
                results[i] = detail::ZERO_DOUBLE;
            } else if (values[i] >= static_cast<double>(b)) {
                results[i] = detail::ONE;
            } else {
                const int k = static_cast<int>(std::floor(values[i]));
                const int numerator = k - a + detail::ONE_INT;
                results[i] = static_cast<double>(numerator) * inv_range;
            }
        }
        return;
    }

    // Runtime CPU detection passed - use vectorized implementation if possible
    // Note: For discrete CDF, vectorization typically doesn't provide significant benefits
    // due to the nature of comparisons and floor operations, but we implement for consistency
    // In practice, this will mostly fall back to scalar due to the nature of the operation

    // Use scalar implementation even when SIMD is available because discrete distribution
    // operations are not amenable to vectorization (primarily branching logic)
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] < static_cast<double>(a)) {
            results[i] = detail::ZERO_DOUBLE;
        } else if (values[i] >= static_cast<double>(b)) {
            results[i] = detail::ONE;
        } else {
            const int k = static_cast<int>(std::floor(values[i]));
            const int numerator = k - a + detail::ONE_INT;
            results[i] = static_cast<double>(numerator) * inv_range;
        }
    }
}

//==========================================================================
// 19. PRIVATE COMPUTATIONAL METHODS
//==========================================================================

// Implementation moved from header - NOT inline due to complexity
void DiscreteDistribution::updateCacheUnsafe() const noexcept {
    // Primary calculations - compute once, reuse multiple times
    range_ = b_ - a_ + 1;
    probability_ = 1.0 / static_cast<double>(range_);
    mean_ = (static_cast<double>(a_) + static_cast<double>(b_)) / 2.0;

    // Variance for discrete uniform: ((b-a)(b-a+2))/12
    const double width = static_cast<double>(b_ - a_);
    variance_ = (width * (width + 2.0)) / 12.0;

    logProbability_ = -std::log(static_cast<double>(range_));

    // Optimization flags
    isBinary_ = (a_ == 0 && b_ == 1);
    isStandardDie_ = (a_ == 1 && b_ == 6);
    isSymmetric_ = (a_ == -b_);
    isSmallRange_ = (range_ <= 10);
    isLargeRange_ = (range_ > 1000);

    cache_valid_ = true;
    cacheValidAtomic_.store(true, std::memory_order_release);

    // Update atomic parameters for lock-free access
    atomicA_.store(a_, std::memory_order_release);
    atomicB_.store(b_, std::memory_order_release);
    atomicParamsValid_.store(true, std::memory_order_release);
}

// Static validation method moved from header for better compile times
void DiscreteDistribution::validateParameters(int a, int b) {
    if (a >= b) {
        throw std::invalid_argument(
            "Upper bound (b) must be strictly greater than lower bound (a)");
    }
    // Check for integer overflow in range calculation
    if (b > INT_MAX - 1 || a < INT_MIN + 1) {
        throw std::invalid_argument("Parameter range too large - risk of integer overflow");
    }
    // Additional safety check for very large ranges
    const long long range_check = static_cast<long long>(b) - static_cast<long long>(a) + 1;
    if (range_check > INT_MAX) {
        throw std::invalid_argument("Parameter range exceeds maximum supported size");
    }
}

//==========================================================================
// 20. PRIVATE UTILITY METHODS
//==========================================================================

// Static utility methods moved from header for better compile times
inline int DiscreteDistribution::roundToInt(double x) noexcept {
    return static_cast<int>(std::round(x));
}

inline bool DiscreteDistribution::isValidIntegerValue(double x) noexcept {
    return (x >= static_cast<double>(INT_MIN) && x <= static_cast<double>(INT_MAX));
}

//==============================================================================
// 21. DISTRIBUTION PARAMETERS
//==============================================================================

// Note: Distribution parameters are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 22. PERFORMANCE CACHE
//==============================================================================

// Note: Performance cache variables are declared in the header as mutable private members
// This section exists for standardization and documentation purposes

//==============================================================================
// 23. OPTIMIZATION FLAGS
//==============================================================================

// Note: Optimization flags are declared in the header as private member variables
// This section exists for standardization and documentation purposes

//==============================================================================
// 24. SPECIALIZED CACHES
//==============================================================================

// Note: Specialized caches are declared in the header as private member variables
// This section exists for standardization and documentation purposes

}  // namespace stats
