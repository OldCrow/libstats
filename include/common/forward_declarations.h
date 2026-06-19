#pragma once

namespace stats {
// Core classes - forward declarations only
class DistributionBase;
class DistributionInterface;

// Distribution classes - forward declarations only
class GaussianDistribution;
class ExponentialDistribution;
class UniformDistribution;
class PoissonDistribution;
class GammaDistribution;
class DiscreteDistribution;
class ChiSquaredDistribution;
class StudentTDistribution;
class BetaDistribution;
class LogNormalDistribution;
class ParetoDistribution;
class WeibullDistribution;
class RayleighDistribution;
class VonMisesDistribution;
class BinomialDistribution;
class NegativeBinomialDistribution;

// Type aliases for common usage
using Gaussian = GaussianDistribution;
using Normal = GaussianDistribution;
using Exponential = ExponentialDistribution;
using Uniform = UniformDistribution;
using Poisson = PoissonDistribution;
using Gamma = GammaDistribution;
using Discrete = DiscreteDistribution;
using ChiSquared = ChiSquaredDistribution;
using StudentT = StudentTDistribution;
using Beta = BetaDistribution;
using LogNormal = LogNormalDistribution;
using Pareto = ParetoDistribution;
using Weibull = WeibullDistribution;
using Rayleigh = RayleighDistribution;
using VonMises = VonMisesDistribution;
using Binomial = BinomialDistribution;
using NegativeBinomial = NegativeBinomialDistribution;

// Platform classes
class SimdProcessor;
class ParallelExecutor;

// Utility forward declarations
template <typename T>
class StatisticalBuffer;

template <typename T>
class DistributionParameter;

// LibDistributionType was a duplicate of stats::detail::DistributionType
// (defined in include/core/performance_dispatcher.h). It had no usages outside
// this file and has been removed. Use stats::detail::DistributionType instead.

enum class OptimizationLevel { None, Basic, SIMD, Parallel, Full };
}  // namespace stats
