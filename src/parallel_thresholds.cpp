#include "../include/platform/parallel_thresholds.h"

#include "../include/platform/parallel_execution_constants.h"

#include <algorithm>
#include <cctype>

namespace libstats {
namespace parallel {

ArchitectureProfile AdaptiveThresholdCalculator::detectArchitectureProfile() const {
    ArchitectureProfile profile;

    // Get CPU features
    const auto& features = cpu::get_features();

    // Base architecture detection and configuration
    using namespace constants::parallel_execution::architecture;
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon: Excellent threading performance
    profile.thread_creation_cost_us = APPLE_SILICON_THREAD_COST_US;
    profile.simd_width_elements = APPLE_SILICON_SIMD_WIDTH;  // NEON 128-bit
    profile.thread_efficiency_factor = APPLE_SILICON_EFFICIENCY;
    profile.base_parallel_threshold = APPLE_SILICON_BASE_THRESHOLD;
#elif defined(__x86_64__) && (defined(__AVX2__) || defined(__AVX512F__))
    // High-end x86_64: Good threading, excellent SIMD
    profile.thread_creation_cost_us = HIGH_END_X86_THREAD_COST_US;
    profile.simd_width_elements = AVX2_SIMD_WIDTH;  // AVX2 256-bit / 4 doubles
    profile.thread_efficiency_factor = HIGH_END_X86_EFFICIENCY;
    profile.base_parallel_threshold = HIGH_END_X86_BASE_THRESHOLD;
#elif defined(__x86_64__)
    // Standard x86_64: Moderate threading
    profile.thread_creation_cost_us = STANDARD_X86_THREAD_COST_US;
    profile.simd_width_elements = SSE_SIMD_WIDTH;  // SSE 128-bit / 2 doubles
    profile.thread_efficiency_factor = STANDARD_X86_EFFICIENCY;
    profile.base_parallel_threshold = STANDARD_X86_BASE_THRESHOLD;
#else
    // Conservative defaults for other architectures
    profile.thread_creation_cost_us = DEFAULT_THREAD_COST_US;
    profile.simd_width_elements = NO_SIMD_WIDTH;  // No SIMD assumed
    profile.thread_efficiency_factor = DEFAULT_EFFICIENCY;
    profile.base_parallel_threshold = DEFAULT_BASE_THRESHOLD;
#endif

    // Set L3 cache size
    profile.l3_cache_size_elements = features.l3_cache_size / sizeof(double);
    if (profile.l3_cache_size_elements == 0) {
        // Reasonable default if detection fails
        profile.l3_cache_size_elements = DEFAULT_L3_CACHE_ELEMENTS;  // 2MB worth of doubles
    }

    return profile;
}

std::string toLower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

OperationComplexity AdaptiveThresholdCalculator::getOperationComplexity(
    const std::string& operation) const {
    std::string op = toLower(operation);

    if (op == "pdf" || op == "logpdf") {
        return OperationComplexity::SIMPLE;
    } else if (op == "cdf") {
        return OperationComplexity::MODERATE;
    } else {
        return OperationComplexity::MODERATE;
    }
}

DistributionComplexity AdaptiveThresholdCalculator::getDistributionComplexity(
    const std::string& distribution) const {
    std::string dist = toLower(distribution);

    if (dist == "uniform") {
        return DistributionComplexity::UNIFORM;
    } else if (dist == "discrete") {
        return DistributionComplexity::DISCRETE;
    } else if (dist == "exponential") {
        return DistributionComplexity::EXPONENTIAL;
    } else if (dist == "poisson") {
        return DistributionComplexity::POISSON;
    } else if (dist == "gaussian" || dist == "normal") {
        return DistributionComplexity::GAUSSIAN;
    } else {
        return DistributionComplexity::EXPONENTIAL;  // Default to moderate complexity
    }
}

std::size_t AdaptiveThresholdCalculator::getThreshold(const std::string& distribution,
                                                      const std::string& operation) const {
    std::string key = toLower(distribution) + "_" + toLower(operation);

    // Check cache first
    auto it = cached_thresholds_.find(key);
    if (it != cached_thresholds_.end()) {
        return it->second;
    }

    // Calculate threshold based on benchmark results
    DistributionComplexity dist_complexity = getDistributionComplexity(distribution);
    OperationComplexity op_complexity = getOperationComplexity(operation);

    std::size_t threshold;

    // Use empirical results from our benchmark
    std::string dist_lower = toLower(distribution);
    std::string op_lower = toLower(operation);

    using namespace constants::parallel_execution::thresholds;

    if (dist_lower == "uniform") {
        if (op_lower == "pdf") {
            threshold = uniform::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = uniform::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = uniform::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold = uniform::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = uniform::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "discrete") {
        if (op_lower == "pdf") {
            threshold = discrete::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = discrete::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = discrete::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold = discrete::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = discrete::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "exponential") {
        if (op_lower == "pdf") {
            threshold = exponential::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = exponential::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = exponential::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold =
                exponential::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = exponential::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "gaussian" || dist_lower == "normal") {
        if (op_lower == "pdf") {
            threshold = gaussian::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = gaussian::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = gaussian::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold = gaussian::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = gaussian::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "poisson") {
        if (op_lower == "pdf") {
            threshold = poisson::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = poisson::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = poisson::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold = poisson::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = poisson::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "gamma") {
        if (op_lower == "pdf") {
            threshold = gamma::PDF_THRESHOLD;
        } else if (op_lower == "logpdf") {
            threshold = gamma::LOGPDF_THRESHOLD;
        } else if (op_lower == "cdf") {
            threshold = gamma::CDF_THRESHOLD;
        } else if (op_lower == "batch_fit") {
            threshold = gamma::BATCH_FIT_THRESHOLD;  // Lower threshold for batch_fit operations
        } else {
            threshold = gamma::DEFAULT_THRESHOLD;
        }
    } else if (dist_lower == "generic") {
        // Generic operations use moderate thresholds
        if (op_lower == "fill" || op_lower == "transform" || op_lower == "for_each") {
            threshold = generic::FILL_TRANSFORM_THRESHOLD;
        } else if (op_lower == "sort" || op_lower == "partial_sort") {
            threshold = generic::SORT_THRESHOLD;
        } else if (op_lower == "scan") {
            threshold = generic::SCAN_THRESHOLD;
        } else if (op_lower == "search" || op_lower == "count") {
            threshold = generic::SEARCH_COUNT_THRESHOLD;
        } else {
            threshold = generic::DEFAULT_THRESHOLD;  // Default for generic operations
        }
    } else {
        // Fallback to calculated threshold
        threshold = calculateThreshold(dist_complexity, op_complexity);
    }

    // Cache the result
    cached_thresholds_[key] = threshold;

    return threshold;
}

bool AdaptiveThresholdCalculator::shouldUseParallel(const std::string& distribution,
                                                    const std::string& operation,
                                                    std::size_t data_size) const {
    std::size_t threshold = getThreshold(distribution, operation);
    return data_size >= threshold;
}

void AdaptiveThresholdCalculator::updateFromMeasurement(const std::string& distribution,
                                                        const std::string& operation,
                                                        std::size_t data_size,
                                                        bool parallel_beneficial) {
    // Future enhancement: adapt thresholds based on runtime measurements
    // For now, this is a placeholder
    (void)distribution;
    (void)operation;
    (void)data_size;
    (void)parallel_beneficial;
}

AdaptiveThresholdCalculator& getGlobalThresholdCalculator() {
    static AdaptiveThresholdCalculator instance;
    return instance;
}

}  // namespace parallel
}  // namespace libstats
