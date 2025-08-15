#pragma once

/**
 * @file parallel_thresholds.h
 * @brief Architecture-aware parallel execution thresholds
 * 
 * This header provides a scalable solution for determining when parallel execution 
 * is beneficial for different distributions and operations, without requiring 
 * an explosion of architecture-specific constants.
 */

#include "platform_common.h"
#include <unordered_map>

namespace libstats {
namespace parallel {

/**
 * @brief Operation complexity categories for threshold determination
 */
enum class OperationComplexity {
    TRIVIAL,    // Simple bounds checking, constant operations (uniform PDF/LogPDF)
    SIMPLE,     // Basic arithmetic, single function calls (discrete PMF, exponential PDF)
    MODERATE,   // Multiple function calls, some computation (poisson PMF, gaussian PDF)
    COMPLEX,    // Heavy computation, special functions (gamma CDF, complex CDFs)
    EXPENSIVE   // Very expensive operations (iterative algorithms, integration)
};

/**
 * @brief Distribution complexity categories
 */
enum class DistributionComplexity {
    UNIFORM,        // Trivial operations: bounds checking, linear interpolation
    DISCRETE,       // Simple arithmetic: integer operations, lookups
    EXPONENTIAL,    // Moderate computation: exp() calls, logarithms
    POISSON,        // Moderate-Complex: factorial, gamma functions
    GAUSSIAN        // Complex: erf(), exp(), more expensive functions
};

/**
 * @brief Architecture performance characteristics
 */
struct ArchitectureProfile {
    std::size_t thread_creation_cost_us;     // Microseconds to create/sync threads
    std::size_t simd_width_elements;         // SIMD vector width in doubles
    std::size_t l3_cache_size_elements;      // L3 cache size in doubles
    double thread_efficiency_factor;         // Threading efficiency (0.0-1.0)
    std::size_t base_parallel_threshold;     // Base threshold for parallel ops
};

/**
 * @brief Adaptive threshold calculator
 * 
 * This class calculates optimal thresholds based on:
 * 1. Hardware architecture characteristics
 * 2. Distribution complexity
 * 3. Operation complexity
 * 4. Runtime performance measurements (future enhancement)
 */
class AdaptiveThresholdCalculator {
private:
    ArchitectureProfile arch_profile_;
    mutable std::unordered_map<std::string, std::size_t> cached_thresholds_;
    
    /**
     * @brief Detect current architecture profile
     */
    ArchitectureProfile detectArchitectureProfile() const;
    
    /**
     * @brief Calculate threshold for specific operation
     */
    std::size_t calculateThreshold([[maybe_unused]] DistributionComplexity dist_complexity,
                                  OperationComplexity op_complexity) const {
        std::size_t base_threshold = arch_profile_.base_parallel_threshold;
        
        // Adjust based on complexity
        switch (op_complexity) {
            case OperationComplexity::TRIVIAL:
                return base_threshold * 10;
            case OperationComplexity::SIMPLE:
                return base_threshold * 5;
            case OperationComplexity::MODERATE:
                return base_threshold * 2;
            case OperationComplexity::COMPLEX:
                return base_threshold;
            case OperationComplexity::EXPENSIVE:
                return base_threshold / 2;
            default:
                return base_threshold;
        }
    }
    
    /**
     * @brief Get operation complexity from operation name
     */
    OperationComplexity getOperationComplexity(const std::string& operation) const;
    
    /**
     * @brief Get distribution complexity from distribution name
     */
    DistributionComplexity getDistributionComplexity(const std::string& distribution) const;

public:
AdaptiveThresholdCalculator() {
        arch_profile_ = detectArchitectureProfile();
    }
    
    /**
     * @brief Get optimal threshold for specific distribution and operation
     * @param distribution Distribution name (e.g., "uniform", "poisson")
     * @param operation Operation name (e.g., "pdf", "logpdf", "cdf")
     * @return Optimal threshold in number of elements
     */
    std::size_t getThreshold(const std::string& distribution, 
                           const std::string& operation) const;
    
    /**
     * @brief Check if parallel execution should be used
     * @param distribution Distribution name
     * @param operation Operation name
     * @param data_size Number of elements to process
     * @return true if parallel execution is recommended
     */
    bool shouldUseParallel(const std::string& distribution,
                          const std::string& operation,
                          std::size_t data_size) const;
    
    /**
     * @brief Update threshold based on runtime measurements (future enhancement)
     * @param distribution Distribution name
     * @param operation Operation name
     * @param data_size Size that was tested
     * @param parallel_beneficial Whether parallel was beneficial
     */
    void updateFromMeasurement(const std::string& distribution,
                              const std::string& operation,
                              std::size_t data_size,
                              bool parallel_beneficial);
};

/**
 * @brief Global adaptive threshold calculator instance
 * 
 * This singleton provides easy access to threshold calculations throughout
 * the library without requiring each distribution to manage its own calculator.
 */
AdaptiveThresholdCalculator& getGlobalThresholdCalculator();

/**
 * @brief Convenience function for checking if parallel execution should be used
 * 
 * This function provides a clean interface for distribution implementations
 * to check whether they should use parallel execution.
 * 
 * @param distribution Distribution name (case-insensitive)
 * @param operation Operation name (case-insensitive) 
 * @param data_size Number of elements to process
 * @return true if parallel execution is recommended
 */
inline bool shouldUseDistributionParallel(const std::string& distribution,
                                         const std::string& operation,
                                         std::size_t data_size) {
    return getGlobalThresholdCalculator().shouldUseParallel(distribution, operation, data_size);
}

} // namespace parallel
} // namespace libstats
