// tests/test_copy_move_cache.cpp
//
// Copy/move cache-state gate across all 27 distributions (issue #163).
// GammaDistribution's copy and move constructors copied cache_valid_ from the
// source without copying the cached fields behind it, so the copy computed
// with default-constructed cache values: a copy of Gamma(2, 1) evaluated as
// Exponential(1). GaussianDistribution's copy constructor had the same
// defect. Every distribution is asserted, not just the two victims, so the
// pattern cannot return elsewhere unnoticed.
//
// Each case warms the cache of the source (and, for the assignments, of a
// target with different parameters, so a stale target cache is the wrong
// answer) before the operation, then requires the result to agree with a
// fresh instance of the source parameters on scalar pdf/logpdf/cdf/quantile,
// the forced-SIMD batch pdf/logpdf/cdf, and the mean and variance.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace stats;

namespace {

constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

template <typename Dist>
void warm(const Dist& d, double x) {
    (void)d.getProbability(x);
    (void)d.getMean();
}

// Exact equality, with NaN equal to NaN: Cauchy's mean and variance are undefined.
::testing::AssertionResult same(double got, double want) {
    if ((std::isnan(got) && std::isnan(want)) || got == want)
        return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << "got " << got << ", want " << want;
}

template <typename Dist>
void expectSameAs(const std::string& what, const Dist& got, const Dist& ref,
                  const std::vector<double>& probes) {
    for (double x : probes) {
        EXPECT_TRUE(same(got.getProbability(x), ref.getProbability(x)))
            << what << " pdf(" << x << ")";
        EXPECT_TRUE(same(got.getLogProbability(x), ref.getLogProbability(x)))
            << what << " logpdf(" << x << ")";
        EXPECT_TRUE(same(got.getCumulativeProbability(x), ref.getCumulativeProbability(x)))
            << what << " cdf(" << x << ")";
    }
    for (double p : {0.1, 0.5, 0.9}) {
        EXPECT_TRUE(same(got.getQuantile(p), ref.getQuantile(p)))
            << what << " quantile(" << p << ")";
    }
    EXPECT_TRUE(same(got.getMean(), ref.getMean())) << what << " mean";
    EXPECT_TRUE(same(got.getVariance(), ref.getVariance())) << what << " variance";

    std::vector<double> xs(kN);
    for (std::size_t i = 0; i < kN; ++i)
        xs[i] = probes[i % probes.size()];
    std::vector<double> a(kN), b(kN);
    const detail::PerformanceHint force_simd{
        detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED, std::nullopt};
    const auto batch = [&](const char* op, auto&& call) {
        call(got, std::span<double>(a));
        call(ref, std::span<double>(b));
        for (std::size_t i = 0; i < kN; ++i)
            EXPECT_TRUE(same(a[i], b[i])) << what << " batch " << op << "(" << xs[i] << ")";
    };
    batch("pdf", [&](const Dist& d, std::span<double> out) {
        d.getProbability(std::span<const double>(xs), out, force_simd);
    });
    batch("logpdf", [&](const Dist& d, std::span<double> out) {
        d.getLogProbability(std::span<const double>(xs), out, force_simd);
    });
    batch("cdf", [&](const Dist& d, std::span<double> out) {
        d.getCumulativeProbability(std::span<const double>(xs), out, force_simd);
    });
}

// makeA and makeB return distributions with different parameters; probes lie
// inside the support of both.
template <typename MakeA, typename MakeB>
void checkCopyMove(const char* name, MakeA makeA, MakeB makeB, const std::vector<double>& probes) {
    const auto ref = makeA();
    const double x0 = probes.front();
    const std::string n(name);

    {
        auto src = makeA();
        warm(src, x0);
        auto copy(src);
        expectSameAs(n + " copy-constructed", copy, ref, probes);
    }
    {
        auto src = makeA();
        warm(src, x0);
        auto moved(std::move(src));
        expectSameAs(n + " move-constructed", moved, ref, probes);
    }
    {
        auto src = makeA();
        auto dst = makeB();
        warm(src, x0);
        warm(dst, x0);
        dst = src;
        expectSameAs(n + " copy-assigned", dst, ref, probes);
    }
    {
        auto src = makeA();
        auto dst = makeB();
        warm(src, x0);
        warm(dst, x0);
        dst = std::move(src);
        expectSameAs(n + " move-assigned", dst, ref, probes);
    }
}

}  // namespace

TEST(CopyMoveCache, Gaussian) {
    checkCopyMove(
        "Gaussian", [] { return GaussianDistribution::create(5.0, 2.0).unwrap(); },
        [] { return GaussianDistribution::create(-1.0, 0.5).unwrap(); }, {-1.0, 0.5, 4.0, 7.0});
}

TEST(CopyMoveCache, Exponential) {
    checkCopyMove(
        "Exponential", [] { return ExponentialDistribution::create(2.5).unwrap(); },
        [] { return ExponentialDistribution::create(0.4).unwrap(); }, {0.1, 0.5, 1.0, 3.0});
}

TEST(CopyMoveCache, Uniform) {
    checkCopyMove(
        "Uniform", [] { return UniformDistribution::create(-2.0, 3.0).unwrap(); },
        [] { return UniformDistribution::create(0.0, 1.0).unwrap(); }, {0.1, 0.5, 0.9});
}

TEST(CopyMoveCache, Poisson) {
    checkCopyMove(
        "Poisson", [] { return PoissonDistribution::create(7.5).unwrap(); },
        [] { return PoissonDistribution::create(1.0).unwrap(); }, {0.0, 2.0, 7.0, 12.0});
}

TEST(CopyMoveCache, Discrete) {
    checkCopyMove(
        "Discrete", [] { return DiscreteDistribution::create(2, 20).unwrap(); },
        [] { return DiscreteDistribution::create(0, 5).unwrap(); }, {2.0, 3.0, 5.0});
}

// The three shapes from #163: moderate, small (alpha < 1), and large with a tiny rate.
TEST(CopyMoveCache, Gamma) {
    checkCopyMove(
        "Gamma(2, 1)", [] { return GammaDistribution::create(2.0, 1.0).unwrap(); },
        [] { return GammaDistribution::create(0.5, 3.0).unwrap(); }, {0.2, 1.0, 2.5, 6.0});
    checkCopyMove(
        "Gamma(0.5, 3)", [] { return GammaDistribution::create(0.5, 3.0).unwrap(); },
        [] { return GammaDistribution::create(2.0, 1.0).unwrap(); }, {0.01, 0.1, 0.5, 2.0});
    checkCopyMove(
        "Gamma(1e4, 1e-3)", [] { return GammaDistribution::create(1e4, 1e-3).unwrap(); },
        [] { return GammaDistribution::create(2.0, 1.0).unwrap(); }, {9.8e6, 1e7, 1.02e7});
}

TEST(CopyMoveCache, ChiSquared) {
    checkCopyMove(
        "ChiSquared", [] { return ChiSquaredDistribution::create(5.0).unwrap(); },
        [] { return ChiSquaredDistribution::create(1.0).unwrap(); }, {0.2, 1.0, 4.0, 9.0});
}

TEST(CopyMoveCache, Erlang) {
    checkCopyMove(
        "Erlang", [] { return ErlangDistribution::create(3, 2.0).unwrap(); },
        [] { return ErlangDistribution::create(1, 0.5).unwrap(); }, {0.2, 1.0, 2.5, 6.0});
}

TEST(CopyMoveCache, InverseGamma) {
    checkCopyMove(
        "InverseGamma", [] { return InverseGammaDistribution::create(3.0, 2.0).unwrap(); },
        [] { return InverseGammaDistribution::create(1.0, 1.0).unwrap(); }, {0.2, 0.7, 1.5, 4.0});
}

TEST(CopyMoveCache, StudentT) {
    checkCopyMove(
        "StudentT", [] { return StudentTDistribution::create(4.0).unwrap(); },
        [] { return StudentTDistribution::create(30.0).unwrap(); }, {-3.0, -0.5, 0.0, 2.0});
}

TEST(CopyMoveCache, Cauchy) {
    checkCopyMove(
        "Cauchy", [] { return CauchyDistribution::create(1.0, 2.0).unwrap(); },
        [] { return CauchyDistribution::create(-3.0, 0.5).unwrap(); }, {-3.0, 0.0, 1.0, 5.0});
}

TEST(CopyMoveCache, Beta) {
    checkCopyMove(
        "Beta", [] { return BetaDistribution::create(2.0, 5.0).unwrap(); },
        [] { return BetaDistribution::create(0.5, 0.5).unwrap(); }, {0.05, 0.3, 0.6, 0.9});
}

TEST(CopyMoveCache, FisherF) {
    checkCopyMove(
        "FisherF", [] { return FDistribution::create(5.0, 10.0).unwrap(); },
        [] { return FDistribution::create(1.0, 1.0).unwrap(); }, {0.2, 1.0, 2.5, 5.0});
}

TEST(CopyMoveCache, LogNormal) {
    checkCopyMove(
        "LogNormal", [] { return LogNormalDistribution::create(1.0, 0.5).unwrap(); },
        [] { return LogNormalDistribution::create(0.0, 1.0).unwrap(); }, {0.5, 1.5, 3.0, 8.0});
}

TEST(CopyMoveCache, Pareto) {
    checkCopyMove(
        "Pareto", [] { return ParetoDistribution::create(3.0, 1.5).unwrap(); },
        [] { return ParetoDistribution::create(1.0, 1.0).unwrap(); }, {1.6, 2.0, 4.0, 10.0});
}

TEST(CopyMoveCache, Weibull) {
    checkCopyMove(
        "Weibull", [] { return WeibullDistribution::create(1.5, 2.0).unwrap(); },
        [] { return WeibullDistribution::create(0.7, 0.5).unwrap(); }, {0.2, 1.0, 2.0, 4.0});
}

TEST(CopyMoveCache, Rayleigh) {
    checkCopyMove(
        "Rayleigh", [] { return RayleighDistribution::create(2.0).unwrap(); },
        [] { return RayleighDistribution::create(0.5).unwrap(); }, {0.2, 1.0, 2.0, 5.0});
}

TEST(CopyMoveCache, VonMises) {
    checkCopyMove(
        "VonMises", [] { return VonMisesDistribution::create(0.5, 3.0).unwrap(); },
        [] { return VonMisesDistribution::create(-1.0, 0.5).unwrap(); }, {-2.0, 0.0, 0.5, 2.5});
}

TEST(CopyMoveCache, Laplace) {
    checkCopyMove(
        "Laplace", [] { return LaplaceDistribution::create(1.0, 2.0).unwrap(); },
        [] { return LaplaceDistribution::create(-2.0, 0.5).unwrap(); }, {-3.0, 0.0, 1.0, 4.0});
}

TEST(CopyMoveCache, Logistic) {
    checkCopyMove(
        "Logistic", [] { return LogisticDistribution::create(1.0, 2.0).unwrap(); },
        [] { return LogisticDistribution::create(-2.0, 0.5).unwrap(); }, {-3.0, 0.0, 1.0, 4.0});
}

TEST(CopyMoveCache, Gumbel) {
    checkCopyMove(
        "Gumbel", [] { return GumbelDistribution::create(1.0, 2.0).unwrap(); },
        [] { return GumbelDistribution::create(-2.0, 0.5).unwrap(); }, {-2.0, 0.0, 1.0, 4.0});
}

TEST(CopyMoveCache, HalfNormal) {
    checkCopyMove(
        "HalfNormal", [] { return HalfNormalDistribution::create(2.0).unwrap(); },
        [] { return HalfNormalDistribution::create(0.5).unwrap(); }, {0.1, 0.5, 1.5, 4.0});
}

TEST(CopyMoveCache, TruncatedNormal) {
    checkCopyMove(
        "TruncatedNormal",
        [] { return TruncatedNormalDistribution::create(1.0, 2.0, -1.0, 4.0).unwrap(); },
        [] { return TruncatedNormalDistribution::create(0.0, 1.0, -0.5, 3.0).unwrap(); },
        {-0.4, 0.5, 1.5, 2.9});
}

TEST(CopyMoveCache, Binomial) {
    checkCopyMove(
        "Binomial", [] { return BinomialDistribution::create(20, 0.3).unwrap(); },
        [] { return BinomialDistribution::create(5, 0.8).unwrap(); }, {0.0, 2.0, 4.0, 5.0});
}

TEST(CopyMoveCache, Bernoulli) {
    checkCopyMove(
        "Bernoulli", [] { return BernoulliDistribution::create(0.2).unwrap(); },
        [] { return BernoulliDistribution::create(0.9).unwrap(); }, {0.0, 1.0});
}

TEST(CopyMoveCache, NegativeBinomial) {
    checkCopyMove(
        "NegativeBinomial", [] { return NegativeBinomialDistribution::create(4.0, 0.3).unwrap(); },
        [] { return NegativeBinomialDistribution::create(1.0, 0.8).unwrap(); },
        {0.0, 2.0, 6.0, 15.0});
}

TEST(CopyMoveCache, Geometric) {
    checkCopyMove(
        "Geometric", [] { return GeometricDistribution::create(0.2).unwrap(); },
        [] { return GeometricDistribution::create(0.9).unwrap(); }, {0.0, 1.0, 3.0, 8.0});
}
