#pragma once

/**
 * @file core/distribution_concepts.h
 * @brief C++20 concepts replacing the DistributionTraits<> SFINAE mechanism.
 *
 * Three concept levels:
 *   AnyDistribution<D>        — minimum contract for all distributions
 *   ContinuousDistribution<D> — AnyDistribution + kIsDiscrete == false
 *   DiscreteDistribution<D>   — AnyDistribution + kIsDiscrete == true
 *
 * Every distribution class must expose two static constexpr members:
 *   static constexpr detail::DistributionType kDistributionType;
 *   static constexpr bool kIsDiscrete;
 *
 * These replace DistributionTraits<D> specialisations and allow dispatch_utils.h
 * to derive the DistributionType directly from D, and allow generic
 * stats::analysis templates (step 5) to be constrained with
 * template<stats::concepts::ContinuousDistribution D>.
 */

#include <concepts>
#include <random>
#include <span>
#include <string>
#include <string_view>

#include "distribution_type.h"  // for detail::DistributionType (AQ-2: minimal include)

namespace stats::concepts {

// ---------------------------------------------------------------------------
// AnyDistribution — base contract shared by all distributions
// ---------------------------------------------------------------------------

/**
 * @brief Base concept satisfied by every libstats distribution.
 *
 * Requires the scalar probability interface, span-based batch getProbability,
 * statistical moments, sampling, identity queries, and the two static constexpr
 * metadata members that replace DistributionTraits<D>.
 */
template <typename D>
concept AnyDistribution =
    requires(const D& d, double x, std::span<const double> xs,
             std::span<double> ys, std::mt19937& rng) {
        // Scalar probability interface
        { d.getProbability(x) }           -> std::convertible_to<double>;
        { d.getLogProbability(x) }        -> std::convertible_to<double>;
        { d.getCumulativeProbability(x) } -> std::convertible_to<double>;
        { d.getQuantile(x) }              -> std::convertible_to<double>;

        // Statistical moments (AR-2: getSkewness/getKurtosis added; pure virtual
        // on DistributionInterface, implemented by all 16 distributions)
        { d.getMean() }      -> std::convertible_to<double>;
        { d.getVariance() }  -> std::convertible_to<double>;
        { d.getSkewness() }  -> std::convertible_to<double>;
        { d.getKurtosis() }  -> std::convertible_to<double>;

        // Sampling
        { d.sample(rng) } -> std::convertible_to<double>;

        // Span-based batch operations (void return)
        d.getProbability(xs, ys);

        // Identity
        { d.getDistributionName() } -> std::convertible_to<std::string_view>;
        { d.getNumParameters() }    -> std::convertible_to<int>;

        // Support bounds (pure virtual on DistributionInterface)
        { d.getSupportLowerBound() } -> std::convertible_to<double>;
        { d.getSupportUpperBound() } -> std::convertible_to<double>;

        // Dispatch metadata — replaces DistributionTraits<D>
        { D::kDistributionType } -> std::convertible_to<detail::DistributionType>;
        { D::kIsDiscrete }       -> std::convertible_to<bool>;
    };

// ---------------------------------------------------------------------------
// ContinuousDistribution — AnyDistribution where kIsDiscrete == false
// ---------------------------------------------------------------------------

/**
 * @brief Concept for continuous statistical distributions.
 *
 * Satisfied by Gaussian, Exponential, Uniform, Gamma, Chi-squared, Student's t,
 * Beta, Log-Normal, Pareto, Weibull, Rayleigh, Von Mises.
 *
 * Constraining generic analysis templates with this concept (step 5) prevents
 * applying continuous-only tests (K-S, Anderson-Darling) to discrete
 * distributions at compile time.
 */
template <typename D>
concept ContinuousDistribution = AnyDistribution<D> && !D::kIsDiscrete;

// ---------------------------------------------------------------------------
// DiscreteDistribution — AnyDistribution where kIsDiscrete == true
// ---------------------------------------------------------------------------

/**
 * @brief Concept for discrete statistical distributions.
 *
 * Satisfied by Poisson, Discrete, Binomial, Negative Binomial.
 */
template <typename D>
concept DiscreteDistribution = AnyDistribution<D> && D::kIsDiscrete;

// ---------------------------------------------------------------------------
// FittableDistribution — AnyDistribution that can be default-constructed
// and fitted to data via fit(const std::vector<double>&)
// ---------------------------------------------------------------------------

/**
 * @brief Concept for distributions that can be fitted to data.
 *
 * Extends AnyDistribution with the default-constructibility and fit() method
 * required by bootstrap and cross-validation templates.
 *
 * All 16 standard libstats distributions satisfy this concept.
 */
template <typename D>
concept FittableDistribution =
    AnyDistribution<D> &&
    std::default_initializable<D> &&
    requires(D d, const std::vector<double>& data) {
        d.fit(data);
    };

}  // namespace stats::concepts
