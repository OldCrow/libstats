#include "../include/platform/parallel_thresholds.h"

#include <algorithm>
#include <cctype>

namespace libstats {
namespace parallel {

ArchitectureProfile AdaptiveThresholdCalculator::detectArchitectureProfile() const {
    ArchitectureProfile profile;

    // Get CPU features
    const auto& features = cpu::get_features();

// Base architecture detection and configuration
#if defined(__APPLE__) && defined(__aarch64__)
    // Apple Silicon: Excellent threading performance
    profile.thread_creation_cost_us = 2;
    profile.simd_width_elements = 2;  // NEON 128-bit
    profile.thread_efficiency_factor = 0.95;
    profile.base_parallel_threshold = 1024;
#elif defined(__x86_64__) && (defined(__AVX2__) || defined(__AVX512F__))
    // High-end x86_64: Good threading, excellent SIMD
    profile.thread_creation_cost_us = 5;
    profile.simd_width_elements = 4;  // AVX2 256-bit / 4 doubles
    profile.thread_efficiency_factor = 0.85;
    profile.base_parallel_threshold = 2048;
#elif defined(__x86_64__)
    // Standard x86_64: Moderate threading
    profile.thread_creation_cost_us = 8;
    profile.simd_width_elements = 2;  // SSE 128-bit / 2 doubles
    profile.thread_efficiency_factor = 0.75;
    profile.base_parallel_threshold = 4096;
#else
    // Conservative defaults for other architectures
    profile.thread_creation_cost_us = 10;
    profile.simd_width_elements = 1;  // No SIMD assumed
    profile.thread_efficiency_factor = 0.7;
    profile.base_parallel_threshold = 8192;
#endif

    // Set L3 cache size
    profile.l3_cache_size_elements = features.l3_cache_size / sizeof(double);
    if (profile.l3_cache_size_elements == 0) {
        // Reasonable default if detection fails
        profile.l3_cache_size_elements = 2 * 1024 * 1024;  // 2MB worth of doubles
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

    if (dist_lower == "uniform") {
        if (op_lower == "pdf") {
            threshold = 16384;
        } else if (op_lower == "logpdf") {
            threshold = 64;
        } else if (op_lower == "cdf") {
            threshold = 16384;
        } else if (op_lower == "batch_fit") {
            threshold = 64;  // Lower threshold for batch_fit operations
        } else {
            threshold = 8192;
        }
    } else if (dist_lower == "discrete") {
        if (op_lower == "pdf") {
            threshold = 1048576;
        } else if (op_lower == "logpdf") {
            threshold = 32768;
        } else if (op_lower == "cdf") {
            threshold = 65536;
        } else if (op_lower == "batch_fit") {
            threshold = 64;  // Lower threshold for batch_fit operations
        } else {
            threshold = 32768;
        }
    } else if (dist_lower == "exponential") {
        if (op_lower == "pdf") {
            threshold = 64;
        } else if (op_lower == "logpdf") {
            threshold = 128;
        } else if (op_lower == "cdf") {
            threshold = 64;
        } else if (op_lower == "batch_fit") {
            threshold = 32;  // Lower threshold for batch_fit operations
        } else {
            threshold = 64;
        }
    } else if (dist_lower == "gaussian" || dist_lower == "normal") {
        if (op_lower == "pdf") {
            threshold = 64;
        } else if (op_lower == "logpdf") {
            threshold = 256;
        } else if (op_lower == "cdf") {
            threshold = 64;
        } else if (op_lower == "batch_fit") {
            threshold = 32;  // Lower threshold for batch_fit operations
        } else {
            threshold = 256;
        }
    } else if (dist_lower == "poisson") {
        if (op_lower == "pdf") {
            threshold = 4096;
        } else if (op_lower == "logpdf") {
            threshold = 8192;
        } else if (op_lower == "cdf") {
            threshold = 512;
        } else if (op_lower == "batch_fit") {
            threshold = 64;  // Lower threshold for batch_fit operations
        } else {
            threshold = 4096;
        }
    } else if (dist_lower == "gamma") {
        if (op_lower == "pdf") {
            threshold = 256;
        } else if (op_lower == "logpdf") {
            threshold = 512;
        } else if (op_lower == "cdf") {
            threshold = 128;
        } else if (op_lower == "batch_fit") {
            threshold = 64;  // Lower threshold for batch_fit operations
        } else {
            threshold = 256;
        }
    } else if (dist_lower == "generic") {
        // Generic operations use moderate thresholds
        if (op_lower == "fill" || op_lower == "transform" || op_lower == "for_each") {
            threshold = 8192;
        } else if (op_lower == "sort" || op_lower == "partial_sort") {
            threshold = 4096;
        } else if (op_lower == "scan") {
            threshold = 16384;
        } else if (op_lower == "search" || op_lower == "count") {
            threshold = 8192;
        } else {
            threshold = 8192;  // Default for generic operations
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
