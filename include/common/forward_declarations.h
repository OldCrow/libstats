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

// Platform classes
class SimdProcessor;
class ParallelExecutor;

// Utility forward declarations
template <typename T>
class StatisticalBuffer;

template <typename T>
class DistributionParameter;

// Common enums and constants that don't need heavy includes
enum class LibDistributionType {
    Gaussian,
    Exponential,
    Uniform,
    Poisson,
    Gamma,
    Discrete,
    ChiSquared,
    StudentT,
    Beta,
    LogNormal,
    Pareto
};

enum class OptimizationLevel { None, Basic, SIMD, Parallel, Full };
}  // namespace stats
