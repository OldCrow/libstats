// tests/test_discrete_pmf_accuracy.cpp
//
// Poisson, Binomial, NegativeBinomial and Geometric log-pmfs at large counts against mpmath
// (issue #172). Formed as k·log λ − λ − lgamma(k + 1) and its binomial analogues, the terms of
// size n·log n cancelled to a result of order one: 2e-9 relative in the pmf at counts ~1e5. The
// pmfs are now Stirling errors plus deviances around the means (detail::poisson_log_pmf,
// detail::binomial_log_pmf), which leaves the problem's own conditioning: a relative ε in a
// parameter moves the log-pmf by about (x − mean)/σ·√x·ε, so the budget is absolute,
// 16ε·(1 + |log pmf| + √x), on the log-pmf and relative on the pmf.
//
// Two-sided per the regression-guard rule: scalar and every batch strategy, rows at the mean and
// at ±5σ for counts 1e3, 1e5 and 1e7 (Poisson: 1e6, its parameter limit). References: mpmath at
// dps 60 from lgamma, at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr double kEps = 0x1p-52;
constexpr std::size_t kN = 69;  // 8*8+5: the batch body and tail on every tier width

double budgetFor(double x, double logpmf) {
    return 16 * kEps * (1.0 + std::fabs(logpmf) + std::sqrt(x));
}

template <typename Dist>
void expectPmf(const std::string& what, const Dist& d, double x, double logpmf) {
    const double budget = budgetFor(x, logpmf);
    const double got_log = d.getLogProbability(x);
    ASSERT_TRUE(std::isfinite(got_log)) << what << " logpmf = " << got_log;
    EXPECT_LE(std::fabs(got_log - logpmf), budget)
        << what << " logpmf = " << got_log << ", want " << logpmf;
    const double want = std::exp(logpmf);
    EXPECT_LE(std::fabs(d.getProbability(x) - want) / want, budget + 2 * kEps)
        << what << " pmf = " << d.getProbability(x) << ", want " << want;

    using Strategy = detail::PerformanceHint::PreferredStrategy;
    for (Strategy s :
         {Strategy::FORCE_SCALAR, Strategy::FORCE_VECTORIZED, Strategy::FORCE_PARALLEL}) {
        const detail::PerformanceHint hint{s, std::nullopt};
        const std::string tag = what + " batch strategy " + std::to_string(static_cast<int>(s));
        std::vector<double> xs(kN, x), out(kN);
        d.getLogProbability(std::span<const double>(xs), std::span<double>(out), hint);
        for (std::size_t i : {std::size_t{0}, kN - 1})
            EXPECT_LE(std::fabs(out[i] - logpmf), budget) << tag << " logpmf = " << out[i];
        d.getProbability(std::span<const double>(xs), std::span<double>(out), hint);
        for (std::size_t i : {std::size_t{0}, kN - 1})
            EXPECT_LE(std::fabs(out[i] - want) / want, budget + 2 * kEps)
                << tag << " pmf = " << out[i];
    }
}

std::string at(const char* name, double a, double b, double x) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s(%g, %g) at %.17g", name, a, b, x);
    return buf;
}

}  // namespace

TEST(DiscretePmfAccuracy, Poisson) {
    struct Row {
        double lambda, k, logpmf;
    };
    constexpr Row kRows[] = {
        {1e3, 1000.0, -4.3728995060262968242},
        {1e3, 842.0, -1.7483754600105560532e+1},
        {1e3, 1158.0, -1.6318326382054672597e+1},
        {1e5, 100000.0, -6.6754020990231202824},
        {1e5, 98419.0, -1.9231628031164135491e+1},
        {1e5, 101581.0, -1.9115702578617884532e+1},
        // λ = 1e6 is MAX_POISSON_LAMBDA.
        {1e6, 1000000.0, -7.8266938955201431272},
        {1e6, 995000.0, -2.0345073198466498577e+1},
        {1e6, 1005000.0, -2.0308406260130049129e+1},
    };
    for (const Row& r : kRows)
        expectPmf(at("poisson", r.lambda, 0, r.k), PoissonDistribution::create(r.lambda).unwrap(),
                  r.k, r.logpmf);
}

TEST(DiscretePmfAccuracy, Binomial) {
    struct Row {
        int n;
        double p, k, logpmf;
    };
    constexpr Row kRows[] = {
        {1000, 0.3, 300.0, -3.5928057905186981179},
        {1000, 0.3, 228.0, -1.6515069237058022792e+1},
        {1000, 0.3, 372.0, -1.5504258633690484586e+1},
        {100000, 0.3, 30000.0, -5.8950805264780875785},
        {100000, 0.3, 29275.0, -1.8461421916917576013e+1},
        {100000, 0.3, 30725.0, -1.8359995401187662421e+1},
        {10000000, 0.3, 3000000.0, -8.1976625159007047993},
        {10000000, 0.3, 2992754.0, -2.0703806408734724244e+1},
        {10000000, 0.3, 3007246.0, -2.0693684002943260526e+1},
    };
    for (const Row& r : kRows)
        expectPmf(at("binomial", r.n, r.p, r.k), BinomialDistribution::create(r.n, r.p).unwrap(),
                  r.k, r.logpmf);
}

TEST(DiscretePmfAccuracy, NegativeBinomialAndGeometric) {
    struct Row {
        double r, p, k, logpmf;
    };
    constexpr Row kRows[] = {
        {1000.0, 0.5, 1000.0, -4.7195147629705055907},
        {1000.0, 0.5, 776.0, -1.8697159505521376179e+1},
        {1000.0, 0.5, 1224.0, -1.6173371703012622475e+1},
        {1e5, 0.5, 100000.0, -7.0219761059697596013},
        {1e5, 0.5, 97764.0, -1.9645878558289455455e+1},
        {1e5, 0.5, 102236.0, -1.9399887304339640631e+1},
        {1e7, 0.5, 10000000.0, -9.3245599614638052906},
        {1e7, 0.5, 9977639.0, -2.1837233628580338383e+1},
        {1e7, 0.5, 10022361.0, -2.1812635679753776575e+1},
        {1234.5, 0.2, 4938.0, -5.9760862032711370177},  // real r
        {1234.5, 0.2, 4152.0, -1.9673073133879389766e+1},
        {1234.5, 0.2, 5724.0, -1.7541680005627951193e+1},
    };
    for (const Row& r : kRows)
        expectPmf(at("negative_binomial", r.r, r.p, r.k),
                  NegativeBinomialDistribution::create(r.r, r.p).unwrap(), r.k, r.logpmf);
    expectPmf(at("geometric", 1e-6, 0, 1e6), GeometricDistribution::create(1e-6).unwrap(), 1e6,
              -1.4815511057964607438e+1);
    expectPmf(at("geometric", 1e-6, 0, 5e6), GeometricDistribution::create(1e-6).unwrap(), 5e6,
              -1.8815513057965940591e+1);
}

// The kernels' domain edges, exactly. binomial_log_pmf on its Stirling path (xa + xb ≥ 20) took
// pa = 0 or 1 to a zero mean and returned 0·∞ = NaN, which a concurrent Binomial::setP(0) reached
// (test_snapshot_consistency).
TEST(DiscretePmfAccuracy, KernelEdges) {
    constexpr double kNegInf = -std::numeric_limits<double>::infinity();
    EXPECT_EQ(detail::binomial_log_pmf(5, 95, 0.0), kNegInf);
    EXPECT_EQ(detail::binomial_log_pmf(95, 5, 1.0), kNegInf);
    EXPECT_EQ(detail::binomial_log_pmf(0, 100, 0.0), 0.0);
    EXPECT_EQ(detail::binomial_log_pmf(100, 0, 1.0), 0.0);
    EXPECT_EQ(detail::binomial_log_pmf(3, 4, 0.0), kNegInf);  // the direct path, below 20
    EXPECT_EQ(detail::binomial_log_pmf(4, 3, 1.0), kNegInf);
    EXPECT_EQ(detail::poisson_log_pmf(0, 0.0), 0.0);
    EXPECT_EQ(detail::poisson_log_pmf(5, 0.0), kNegInf);
    EXPECT_EQ(detail::poisson_log_pmf(50, 0.0), kNegInf);
}
