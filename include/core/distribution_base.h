#pragma once

// Include the common base and split components
#include "distribution_base_common.h"
#include "distribution_interface.h"
#include "distribution_memory.h"
#include "distribution_validation.h"

// Platform components
#include "../platform/distribution_cache.h"
#include "../platform/platform_constants.h"

namespace libstats {

/**
 * @brief Base class for probability distributions with comprehensive functionality
 * 
 * This class provides a complete implementation foundation for statistical distributions,
 * combining the core distribution interface with advanced features like:
 * - Thread-safe caching with performance monitoring
 * - Memory-optimized operations and SIMD batch processing
 * - Comprehensive validation and diagnostics
 * - Numerical utilities and special mathematical functions
 * 
 * The class inherits from multiple specialized interfaces to provide a clean
 * separation of concerns while maintaining full backwards compatibility.
 * 
 * @par Usage Example:
 * @code
 * class NormalDistribution : public DistributionBase {
 * public:
 *     // Implement pure virtual methods from DistributionInterface
 *     double getProbability(double x) const override;
 *     double getCumulativeProbability(double x) const override;
 *     // ... other required methods
 * 
 * protected:
 *     void updateCacheUnsafe() const override {
 *         // Update cached statistical properties
 *         cached_mean_ = computeMean();
 *         cached_variance_ = computeVariance();
 *         // Mark cache as valid (handled by base class)
 *     }
 * };
 * @endcode
 */
class DistributionBase : public DistributionInterface,
                        public ThreadSafeCacheManager {
public:
    // =============================================================================
    // RULE OF FIVE - Modern C++20 Implementation
    // =============================================================================
    
    /**
     * @brief Default constructor with system initialization
     */
    DistributionBase();
    
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~DistributionBase() = default;
    
    /**
     * @brief Copy constructor with thread-safe cache handling
     */
    DistributionBase(const DistributionBase& other);
    
    /**
     * @brief Move constructor with thread-safe state transfer
     */
    DistributionBase(DistributionBase&& other) noexcept;
    
    /**
     * @brief Copy assignment operator with thread-safe implementation
     */
    DistributionBase& operator=(const DistributionBase& other);
    
    /**
     * @brief Move assignment operator with thread-safe implementation  
     */
    DistributionBase& operator=(DistributionBase&& other) noexcept;

    // =============================================================================
    // CORE PROBABILITY INTERFACE - Must be implemented by derived classes
    // =============================================================================
    // Note: Pure virtual methods from DistributionInterface must be implemented
    
    /**
     * @brief Log probability density/mass function evaluation (default implementation)
     * @param x Value at which to evaluate the log distribution
     * @return Log probability density/mass at x
     * @note Override for numerical stability when possible
     */
    double getLogProbability(double x) const override {
        double prob = getProbability(x);
        return prob > 0.0 ? std::log(prob) : -std::numeric_limits<double>::infinity();
    }

    // =============================================================================
    // ENHANCED FUNCTIONALITY - Built on top of core interface
    // =============================================================================
    
    // Multi-sample method implementation inherited from DistributionInterface
    
    // =============================================================================
    // SIMD-OPTIMIZED BATCH OPERATIONS - Virtual (Override for Performance)
    // =============================================================================
    
    /**
     * @brief Batch probability density/mass function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of probability densities/masses
     * @note Base implementation uses parallel processing for large datasets; override for SIMD optimization
     */
    virtual std::vector<double> getBatchProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch log probability density/mass function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of log probability densities/masses
     * @note Base implementation uses parallel processing for large datasets; override for SIMD optimization
     */
    virtual std::vector<double> getBatchLogProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch cumulative distribution function evaluation
     * @param x_values Vector of values to evaluate
     * @return Vector of cumulative probabilities P(X <= x)
     * @note Base implementation uses parallel processing for large datasets; override for SIMD optimization
     */
    virtual std::vector<double> getBatchCumulativeProbabilities(const std::vector<double>& x_values) const;
    
    /**
     * @brief Batch quantile function evaluation
     * @param p_values Vector of probability values in [0,1]
     * @return Vector of quantile values
     * @note Base implementation validates inputs and uses parallel processing; override for SIMD optimization
     * @throws std::invalid_argument if any p_value not in [0,1]
     */
    virtual std::vector<double> getBatchQuantiles(const std::vector<double>& p_values) const;
    
    /**
     * @brief Check if SIMD batch operations should be used for given size
     * @param vector_size Size of the input vector
     * @return True if SIMD operations would be beneficial
     */
    static bool shouldUseSIMDBatch(size_t vector_size) noexcept {
        return vector_size >= 32; // Threshold for SIMD benefit
    }

    // =============================================================================
    // INFORMATION THEORY METRICS - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Calculate entropy of the distribution
     * @return Differential entropy (continuous) or entropy (discrete)
     * @note Return NaN if not analytically computable
     */
    virtual double getEntropy() const {
        return std::numeric_limits<double>::quiet_NaN();
    }
    
    /**
     * @brief Calculate Kullback-Leibler divergence from another distribution
     * @param other Distribution to compare against
     * @return KL divergence D(this||other)
     * @note Base implementation uses numerical integration; override for efficiency
     */
    virtual double getKLDivergence(const DistributionBase& other) const;

    // =============================================================================
    // DISTRIBUTION COMPARISON - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Check if two DistributionBase objects are approximately equal
     * @param other Distribution to compare
     * @param tolerance Numerical tolerance for comparison
     * @return true if distributions are approximately equal
     */
    virtual bool isApproximatelyEqual(const DistributionBase& other, 
                                    double tolerance = 1e-10) const;

    // =============================================================================
    // VALIDATION AND DIAGNOSTICS - Virtual (Override Optional)
    // =============================================================================
    
    /**
     * @brief Validate distribution fit against data (comprehensive implementation)
     * @param data Vector of observations to validate against
     * @return Validation results with test statistics and p-values
     * @note Performs Kolmogorov-Smirnov and Anderson-Darling tests
     */
    virtual ValidationResult validate(const std::vector<double>& data) const;
    
    /**
     * @brief Fit distribution with comprehensive diagnostics (comprehensive implementation)
     * @param data Vector of observations
     * @return Detailed fitting results and validation
     * @note Performs fitting with AIC/BIC calculation, residual analysis, and validation
     */
    virtual FitResults fitWithDiagnostics(const std::vector<double>& data);
    
    // =============================================================================
    // DISTRIBUTION CACHE ACCESS - Protected Interface
    // =============================================================================
    
    /**
     * @brief Get distribution cache adapter instance
     * @return Reference to platform-optimized cache adapter
     */
    template<typename Key, typename Value>
    DistributionCacheAdapter<Key, Value>& getDistributionCache() const {
        static thread_local DistributionCacheAdapter<Key, Value> cache;
        return cache;
    }

protected:
    // =============================================================================
    // NUMERICAL UTILITIES - Protected Static Methods
    // =============================================================================
    
    /**
     * @brief Numerical integration for CDF calculation
     * @param pdf_func PDF function to integrate
     * @param lower_bound Integration lower bound
     * @param upper_bound Integration upper bound
     * @param tolerance Numerical tolerance
     * @return Integral approximation
     */
    static double numericalIntegration(std::function<double(double)> pdf_func,
                                     double lower_bound, double upper_bound,
                                     double tolerance = 1e-8);
    
    /**
     * @brief Newton-Raphson method for quantile calculation
     * @param cdf_func CDF function
     * @param target_probability Target probability value
     * @param initial_guess Initial guess for root
     * @param tolerance Numerical tolerance
     * @return Quantile approximation
     */
    static double newtonRaphsonQuantile(std::function<double(double)> cdf_func,
                                      double target_probability,
                                      double initial_guess,
                                      double tolerance = 1e-10);
    
    /**
     * @brief Validate data for fitting
     * @param data Data to validate
     * @throws std::invalid_argument if data is invalid
     */
    static void validateFittingData(const std::vector<double>& data);
    
    /**
     * @brief Calculate empirical CDF from data
     * @param data Sorted data vector
     * @return Vector of empirical CDF values
     */
    static std::vector<double> calculateEmpiricalCDF(const std::vector<double>& data);

    // =============================================================================
    // SPECIAL MATHEMATICAL FUNCTIONS - Protected Static Methods
    // =============================================================================
    
    /**
     * @brief Error function erf(x)
     */
    static double erf(double x) noexcept;
    
    /**
     * @brief Complementary error function erfc(x)
     */
    static double erfc(double x) noexcept;
    
    /**
     * @brief Log gamma function ln(Γ(x))
     */
    static double lgamma(double x) noexcept;
    
    /**
     * @brief Regularized incomplete gamma function P(a,x)
     */
    static double gammaP(double a, double x) noexcept;
    
    /**
     * @brief Regularized incomplete gamma function Q(a,x) = 1 - P(a,x)
     */
    static double gammaQ(double a, double x) noexcept;
    
    /**
     * @brief Regularized incomplete beta function I_x(a,b)
     */
    static double betaI(double x, double a, double b) noexcept;

private:
    // =============================================================================
    // INTERNAL IMPLEMENTATION DETAILS
    // =============================================================================
    
    /**
     * @brief Helper function for incomplete beta function continued fraction
     */
    static double betaI_continued_fraction(double x, double a, double b) noexcept;
    
    /**
     * @brief Helper function for adaptive Simpson's rule integration
     */
    static double adaptiveSimpsonIntegration(std::function<double(double)> func,
                                           double a, double b, double tolerance,
                                           int depth, int max_depth);
};

} // namespace libstats
