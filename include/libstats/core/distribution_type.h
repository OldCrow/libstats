#pragma once

#include <cstdint>  // uint8_t for enum class underlying type

/**
 * @file core/distribution_type.h
 * @brief Canonical DistributionType enum with no platform dependencies.
 *
 * Extracted from performance_dispatcher.h in v2.0.0 (AQ-2) so that
 * distribution_concepts.h can include only this lightweight header instead
 * of pulling in the full performance-dispatcher machinery (which transitively
 * includes SIMD and threading infrastructure).
 *
 * performance_dispatcher.h includes this header and re-exports the enum
 * via `stats::detail::DistributionType`; no existing code needs updating.
 */

namespace stats {
namespace detail {

/**
 * @brief Distribution types for strategy optimization and dispatch metadata.
 *
 * Each distribution exposes this as a static constexpr member (`kDistributionType`)
 * so the dispatch system and generic analysis templates can identify the distribution
 * without relying on dynamic dispatch or RTTI.
 */
enum class DistributionType : uint8_t {
    UNIFORM,            ///< Uniform distribution
    GAUSSIAN,           ///< Gaussian (Normal) distribution
    EXPONENTIAL,        ///< Exponential distribution
    DISCRETE,           ///< Discrete uniform distribution
    POISSON,            ///< Poisson distribution
    GAMMA,              ///< Gamma distribution
    STUDENT_T,          ///< Student's t distribution
    BETA,               ///< Beta distribution
    CHI_SQUARED,        ///< Chi-squared distribution (delegates to Gamma)
    LOG_NORMAL,         ///< Log-Normal distribution
    PARETO,             ///< Pareto distribution
    WEIBULL,            ///< Weibull distribution
    RAYLEIGH,           ///< Rayleigh distribution
    VON_MISES,          ///< Von Mises distribution
    BINOMIAL,           ///< Binomial distribution
    NEGATIVE_BINOMIAL,  ///< Negative Binomial distribution
    // --- v2.0.0 additions (append only; existing values must never change) ---
    GEOMETRIC,  ///< Geometric distribution (delegates to NegativeBinomial with r=1)
    LAPLACE,    ///< Laplace (double-exponential) distribution
    CAUCHY,     ///< Cauchy distribution (delegates to StudentT with ν=1)
    // --- v2.4.0 additions (append only; existing values must never change) ---
    LOGISTIC,          ///< Logistic distribution
    GUMBEL,            ///< Gumbel (Type I extreme-value, max-stable) distribution
    BERNOULLI,         ///< Bernoulli distribution (delegates to Binomial with n=1)
    ERLANG,            ///< Erlang distribution (delegates to Gamma with integer shape)
    FISHER_F,          ///< F (Fisher–Snedecor) distribution (delegates to Beta)
    INVERSE_GAMMA,     ///< Inverse Gamma distribution (delegates to Gamma via x → 1/x)
    HALF_NORMAL,       ///< Half-Normal distribution
    TRUNCATED_NORMAL,  ///< Truncated Normal distribution (renormalized Gaussian)
};

}  // namespace detail
}  // namespace stats
