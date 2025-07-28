#ifndef LIBSTATS_CORE_STATISTICAL_METHODS_CONSTANTS_H_
#define LIBSTATS_CORE_STATISTICAL_METHODS_CONSTANTS_H_

/**
 * @file core/statistical_methods_constants.h
 * @brief Constants for statistical methods and techniques
 * 
 * This header contains constants used for various statistical methods
 * including Bayesian estimation, bootstrap, and cross-validation.
 */

namespace libstats {
namespace constants {

/// Bayesian estimation default priors
namespace bayesian {
    /// Default prior parameters for normal-inverse-gamma conjugate prior
    namespace priors {
        /// Default prior mean
        inline constexpr double DEFAULT_PRIOR_MEAN = 0.0;
        
        /// Default prior precision (inverse variance)
        inline constexpr double DEFAULT_PRIOR_PRECISION = 0.001;
        
        /// Default prior shape parameter
        inline constexpr double DEFAULT_PRIOR_SHAPE = 1.0;
        
        /// Default prior rate parameter
        inline constexpr double DEFAULT_PRIOR_RATE = 1.0;
    }
}

/// Default bootstrap parameters
namespace bootstrap {
    /// Default number of bootstrap samples
    inline constexpr int DEFAULT_BOOTSTRAP_SAMPLES = 1000;
    
    /// Default random seed for reproducible results
    inline constexpr unsigned int DEFAULT_RANDOM_SEED = 42;
}

/// Cross-validation defaults
namespace cross_validation {
    /// Default number of folds for k-fold cross-validation
    inline constexpr int DEFAULT_K_FOLDS = 5;
}

} // namespace constants
} // namespace libstats

#endif // LIBSTATS_CORE_STATISTICAL_METHODS_CONSTANTS_H_
