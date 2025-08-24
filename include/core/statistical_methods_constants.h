#pragma once

/**
 * @file core/statistical_methods_constants.h
 * @brief Constants for statistical methods and techniques
 *
 * This header contains constants used for various statistical methods
 * including Bayesian estimation, bootstrap, and cross-validation.
 */

namespace stats {
namespace detail {
/// Bayesian estimation default priors
/// Default prior parameters for normal-inverse-gamma conjugate prior
/// Default prior mean
inline constexpr double PRIOR_DEFAULT_MEAN = 0.0;

/// Default prior precision (inverse variance)
inline constexpr double PRIOR_DEFAULT_PRECISION = 0.001;

/// Default prior shape parameter
inline constexpr double PRIOR_DEFAULT_SHAPE = 1.0;

/// Default prior rate parameter
inline constexpr double PRIOR_DEFAULT_RATE = 1.0;

/// Default bootstrap parameters
/// Default number of bootstrap samples
inline constexpr int BOOTSTRAP_DEFAULT_SAMPLES = 1000;

/// Default random seed for reproducible results
inline constexpr unsigned int BOOTSTRAP_DEFAULT_RANDOM_SEED = 42;

/// Cross-validation defaults
/// Default number of folds for k-fold cross-validation
inline constexpr int CV_DEFAULT_K_FOLDS = 5;

}  // namespace detail
}  // namespace stats
