// tests/test_batch_nan_gates.cpp
//
// Batch-path NaN propagation gate across ALL 19 distributions (issue #102).
// Scalar paths propagate NaN correctly everywhere; the batch (span/SIMD)
// paths of uniform, gamma, chi-squared, laplace, pareto, weibull, beta, and
// von Mises (plus, via the #105 vector_log laundering, lognormal/student_t/
// cauchy on non-AVX-512 tiers) instead returned finite, plausible values --
// e.g. uniform batch pdf(NaN) = the full in-support density because the NaN
// range comparison silently took the in-range branch; pareto batch
// cdf(NaN) = 0.9992.
//
// The non-victim distributions are asserted too: they are the regression
// guard that keeps a future kernel change from introducing the same class
// of bug where it does not exist today.
//
// NaN leads each input array (the sweep bug that hid #105 was
// specials-last: everything appended after the finite grid fell into the
// scalar libm tail and the SIMD body went untested) and closes it, so the
// special is evaluated by the vector body AND by the scalar tail on every
// tier width (69 = 8*8+5: tail length 1 mod 2, 1 mod 4, 5 mod 8). A benign
// lane is asserted non-NaN so a fix cannot pass by laundering everything.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <vector>

using namespace stats;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kN = 69;  // 8*8+5; >= every SIMD threshold, non-multiple of 2/4/8

template <typename Dist>
void checkBatchNaN(const char* name, Dist& dist, double benign) {
    std::vector<double> xs(kN, benign);
    xs[0] = kNaN;
    xs[kN - 1] = kNaN;
    std::vector<double> out(kN, 0.0);
    const detail::PerformanceHint force_simd{
        detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED, std::nullopt};

    dist.getProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << name << " batch pdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[kN - 1])) << name << " batch pdf(NaN) [tail] = " << out[kN - 1];
    EXPECT_FALSE(std::isnan(out[1])) << name << " batch pdf(" << benign << ") went NaN";

    dist.getLogProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << name << " batch logpdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[kN - 1])) << name << " batch logpdf(NaN) [tail] = " << out[kN - 1];
    EXPECT_FALSE(std::isnan(out[1])) << name << " batch logpdf(" << benign << ") went NaN";

    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), force_simd);
    EXPECT_TRUE(std::isnan(out[0])) << name << " batch cdf(NaN) [body] = " << out[0];
    EXPECT_TRUE(std::isnan(out[kN - 1])) << name << " batch cdf(NaN) [tail] = " << out[kN - 1];
    EXPECT_FALSE(std::isnan(out[1])) << name << " batch cdf(" << benign << ") went NaN";
}

}  // namespace

// Benign fill values sit strictly inside each distribution's support so the
// in-support branch (the one the NaN comparisons silently fell into) is the
// live alternative on every lane.

TEST(BatchNaNGates, Gaussian) {
    auto d = GaussianDistribution::create(0.0, 1.0).unwrap();
    checkBatchNaN("Gaussian", d, 0.5);
}

TEST(BatchNaNGates, Exponential) {
    auto d = ExponentialDistribution::create(1.0).unwrap();
    checkBatchNaN("Exponential", d, 0.5);
}

TEST(BatchNaNGates, Uniform) {
    auto d = UniformDistribution::create(0.0, 1.0).unwrap();
    checkBatchNaN("Uniform", d, 0.5);
}

TEST(BatchNaNGates, Poisson) {
    auto d = PoissonDistribution::create(3.0).unwrap();
    checkBatchNaN("Poisson", d, 3.0);
}

TEST(BatchNaNGates, Discrete) {
    auto d = DiscreteDistribution::create(0, 9).unwrap();
    checkBatchNaN("Discrete", d, 3.0);
}

TEST(BatchNaNGates, Gamma) {
    auto d = GammaDistribution::create(2.0, 1.0).unwrap();
    checkBatchNaN("Gamma", d, 1.5);
}

TEST(BatchNaNGates, ChiSquared) {
    auto d = ChiSquaredDistribution::create(3.0).unwrap();
    checkBatchNaN("ChiSquared", d, 1.5);
}

TEST(BatchNaNGates, StudentT) {
    auto d = StudentTDistribution::create(5.0).unwrap();
    checkBatchNaN("StudentT", d, 0.5);
}

TEST(BatchNaNGates, Beta) {
    auto d = BetaDistribution::create(2.0, 3.0).unwrap();
    checkBatchNaN("Beta", d, 0.5);
}

TEST(BatchNaNGates, LogNormal) {
    auto d = LogNormalDistribution::create(0.0, 1.0).unwrap();
    checkBatchNaN("LogNormal", d, 1.5);
}

TEST(BatchNaNGates, Pareto) {
    auto d = ParetoDistribution::create(1.0, 2.0).unwrap();
    checkBatchNaN("Pareto", d, 1.5);
}

TEST(BatchNaNGates, Weibull) {
    auto d = WeibullDistribution::create(2.0, 1.0).unwrap();
    checkBatchNaN("Weibull", d, 1.5);
}

TEST(BatchNaNGates, Rayleigh) {
    auto d = RayleighDistribution::create(1.0).unwrap();
    checkBatchNaN("Rayleigh", d, 1.5);
}

TEST(BatchNaNGates, VonMises) {
    auto d = VonMisesDistribution::create(0.0, 2.0).unwrap();
    checkBatchNaN("VonMises", d, 0.5);
}

TEST(BatchNaNGates, Binomial) {
    auto d = BinomialDistribution::create(10, 0.5).unwrap();
    checkBatchNaN("Binomial", d, 3.0);
}

TEST(BatchNaNGates, NegativeBinomial) {
    auto d = NegativeBinomialDistribution::create(3.0, 0.5).unwrap();
    checkBatchNaN("NegativeBinomial", d, 3.0);
}

TEST(BatchNaNGates, Geometric) {
    auto d = GeometricDistribution::create(0.5).unwrap();
    checkBatchNaN("Geometric", d, 3.0);
}

TEST(BatchNaNGates, Laplace) {
    auto d = LaplaceDistribution::create(0.0, 1.0).unwrap();
    checkBatchNaN("Laplace", d, 0.5);
}

TEST(BatchNaNGates, Cauchy) {
    auto d = CauchyDistribution::create(0.0, 1.0).unwrap();
    checkBatchNaN("Cauchy", d, 0.5);
}
