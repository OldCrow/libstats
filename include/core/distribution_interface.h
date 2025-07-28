#ifndef LIBSTATS_DISTRIBUTION_INTERFACE_H_
#define LIBSTATS_DISTRIBUTION_INTERFACE_H_

#include <vector>
#include <string>
#include <random>

namespace libstats {

/**
 * @brief Pure interface for probability distributions
 * 
 * This interface defines the core functionality that all statistical distributions
 * must implement. It focuses purely on the statistical interface without
 * implementation details like caching, memory management, or thread safety.
 * 
 * @par Core Statistical Interface:
 * - Probability density/mass functions (PDF/PMF)
 * - Cumulative distribution function (CDF)
 * - Quantile function (inverse CDF)
 * - Statistical moments (mean, variance, skewness, kurtosis)
 * - Random number generation
 * - Parameter estimation and fitting
 * - Distribution metadata
 */
class DistributionInterface {
public:
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~DistributionInterface() = default;

    // =============================================================================
    // CORE PROBABILITY FUNCTIONS - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Probability density/mass function evaluation
     * @param x Value at which to evaluate the distribution
     * @return Probability density (continuous) or mass (discrete) at x
     */
    virtual double getProbability(double x) const = 0;
    
    /**
     * @brief Log probability density/mass function evaluation
     * @param x Value at which to evaluate the log distribution
     * @return Log probability density/mass at x
     * @note Override for numerical stability when possible
     */
    virtual double getLogProbability(double x) const = 0;
    
    /**
     * @brief Cumulative distribution function evaluation
     * @param x Value at which to evaluate the CDF
     * @return P(X <= x)
     */
    virtual double getCumulativeProbability(double x) const = 0;
    
    /**
     * @brief Quantile function (inverse CDF)
     * @param p Probability value in [0,1]
     * @return x such that P(X <= x) = p
     * @throws std::invalid_argument if p not in [0,1]
     */
    virtual double getQuantile(double p) const = 0;

    // =============================================================================
    // STATISTICAL MOMENTS - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Distribution mean (first moment)
     * @return Expected value E[X]
     */
    virtual double getMean() const = 0;
    
    /**
     * @brief Distribution variance (second central moment)
     * @return Variance Var(X) = E[(X - μ)²]
     * @note Prefer variance over standard deviation as fundamental property
     */
    virtual double getVariance() const = 0;
    
    /**
     * @brief Distribution skewness (third standardized moment)
     * @return Skewness coefficient
     * @note Return NaN if undefined for distribution
     */
    virtual double getSkewness() const = 0;
    
    /**
     * @brief Distribution kurtosis (fourth standardized moment)
     * @return Excess kurtosis (kurtosis - 3)
     * @note Return NaN if undefined for distribution
     */
    virtual double getKurtosis() const = 0;

    // =============================================================================
    // RANDOM NUMBER GENERATION - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Generate single random sample from distribution
     * @param rng Random number generator
     * @return Single random sample
     */
    virtual double sample(std::mt19937& rng) const = 0;
    
    /**
     * @brief Generate multiple random samples from distribution
     * @param rng Random number generator
     * @param n Number of samples to generate
     * @return Vector of random samples
     * @note Base implementation in DistributionBase calls sample() n times; override for efficiency
     */
    virtual std::vector<double> sample(std::mt19937& rng, size_t n) const {
        std::vector<double> samples;
        samples.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            samples.push_back(sample(rng));
        }
        return samples;
    }

    // =============================================================================
    // PARAMETER ESTIMATION - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Fit distribution parameters to data using Maximum Likelihood Estimation
     * @param data Vector of observations
     * @throws std::invalid_argument if data is empty or contains invalid values
     */
    virtual void fit(const std::vector<double>& data) = 0;
    
    /**
     * @brief Reset distribution to default parameter values
     */
    virtual void reset() noexcept = 0;

    // =============================================================================
    // DISTRIBUTION METADATA - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Get number of parameters for this distribution
     * @return Number of free parameters
     * @note Used for AIC/BIC calculations
     */
    virtual int getNumParameters() const = 0;
    
    /**
     * @brief Get distribution name
     * @return Human-readable distribution name
     */
    virtual std::string getDistributionName() const = 0;
    
    /**
     * @brief Get string representation of distribution with current parameters
     * @return Formatted string description
     */
    virtual std::string toString() const = 0;

    // =============================================================================
    // DISTRIBUTION PROPERTIES - Pure Virtual (Must Override)
    // =============================================================================
    
    /**
     * @brief Check if distribution is discrete
     * @return true if discrete, false if continuous
     */
    virtual bool isDiscrete() const = 0;
    
    /**
     * @brief Get distribution support lower bound
     * @return Minimum possible value (or -infinity)
     */
    virtual double getSupportLowerBound() const = 0;
    
    /**
     * @brief Get distribution support upper bound  
     * @return Maximum possible value (or +infinity)
     */
    virtual double getSupportUpperBound() const = 0;
};

// =============================================================================
// DERIVED STATISTICAL PROPERTIES (Helper Functions)
// =============================================================================

/**
 * @brief Calculate standard deviation from variance
 * @param distribution Distribution to query
 * @return Standard deviation σ = √Var(X)
 */
inline double getStandardDeviation(const DistributionInterface& distribution) {
    double var = distribution.getVariance();
    return var >= 0.0 ? std::sqrt(var) : std::numeric_limits<double>::quiet_NaN();
}

/**
 * @brief Calculate coefficient of variation
 * @param distribution Distribution to query
 * @return CV = σ/μ (if μ ≠ 0)
 */
inline double getCoefficientOfVariation(const DistributionInterface& distribution) {
    double mean = distribution.getMean();
    return (mean != 0.0) ? getStandardDeviation(distribution) / std::abs(mean) : 
                           std::numeric_limits<double>::infinity();
}

} // namespace libstats

#endif // LIBSTATS_DISTRIBUTION_INTERFACE_H_
