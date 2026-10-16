// Survival-tail guards: getSurvival(x) = P(X > x) formed as the complement itself, not 1 − F.
//
// The DistributionBase default, 1 − getCumulativeProbability(x), returns exactly 0 wherever F
// rounds to 1 and loses relative precision as F nears 1. Each distribution below overrides it;
// every deep-tail case here returned 0 (or a visibly wrong value) through the default, as
// recorded beside it. References: mpmath at 60 digits (closed forms, betainc, or quad for von
// Mises). Correctness only: no timing label, so CI's -LE "timing|benchmark" pass runs it.
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

using namespace stats;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();

double relErr(double got, double want) {
    return std::fabs(got - want) / std::fabs(want);
}

// x stepped n ulps toward −inf.
double below(double x, int n) {
    for (int i = 0; i < n; ++i)
        x = std::nextafter(x, -kInf);
    return x;
}

// Where S is the large side, S + F stays within a few ulps of 1.
template <typename D>
void expectComplementary(const D& d, double x) {
    const double s = d.getSurvival(x);
    const double f = d.getCumulativeProbability(x);
    EXPECT_NEAR(s + f, 1.0, 4 * kEps) << "x = " << x << ", S = " << s << ", F = " << f;
}

}  // namespace

// Old: 0 at both points (S = 1 − F, F rounds to 1).
TEST(SurvivalTails, Cauchy) {
    EXPECT_LT(relErr(CauchyDistribution(1e8, 1e6).getSurvival(5.64e22), 5.643792308223249e-18),
              1e-14);
    EXPECT_LT(relErr(CauchyDistribution(0.0, 1.0).getSurvival(1e20), 3.1830988618379067e-21),
              1e-14);
    EXPECT_EQ(CauchyDistribution(0.0, 1.0).getSurvival(0.0), 0.5);
    expectComplementary(CauchyDistribution(0.0, 1.0), -3.0);
    expectComplementary(CauchyDistribution(0.0, 1.0), 0.25);
}

// Old: 0 at both points.
TEST(SurvivalTails, Gumbel) {
    EXPECT_LT(relErr(GumbelDistribution(0.0, 1.0).getSurvival(46.05), 1.0017033088663171e-20),
              1e-13);
    EXPECT_LT(relErr(GumbelDistribution(2.0, 3.0).getSurvival(300.0), 7.24571861380548e-44), 1e-13);
    expectComplementary(GumbelDistribution(0.0, 1.0), -1.0);
}

// Old: 0 at both points.
TEST(SurvivalTails, Laplace) {
    EXPECT_LT(relErr(LaplaceDistribution(0.0, 1.0).getSurvival(40.0), 2.1241771276457944e-18),
              1e-14);
    EXPECT_LT(relErr(LaplaceDistribution(5.0, 0.1).getSurvival(9.0), 2.124177127645799e-18), 1e-13);
    expectComplementary(LaplaceDistribution(0.0, 1.0), -0.5);
    expectComplementary(LaplaceDistribution(0.0, 1.0), 0.5);
}

// Old: 0 at both points.
TEST(SurvivalTails, Logistic) {
    EXPECT_LT(relErr(LogisticDistribution(0.0, 1.0).getSurvival(40.0), 4.248354255291589e-18),
              1e-14);
    EXPECT_LT(relErr(LogisticDistribution(1.0, 2.0).getSurvival(200.0), 6.133368390286092e-44),
              1e-13);
    expectComplementary(LogisticDistribution(0.0, 1.0), -2.0);
}

// Old: 0 at both points.
TEST(SurvivalTails, HalfNormal) {
    EXPECT_LT(relErr(HalfNormalDistribution(1.0).getSurvival(9.0), 2.257176811907681e-19), 1e-14);
    EXPECT_LT(relErr(HalfNormalDistribution(2.0).getSurvival(30.0), 7.341932398625502e-51), 1e-13);
    EXPECT_EQ(HalfNormalDistribution(1.0).getSurvival(-1.0), 1.0);
    expectComplementary(HalfNormalDistribution(1.0), 0.3);
}

// Old: 0 at both points.
TEST(SurvivalTails, LogNormal) {
    EXPECT_LT(relErr(LogNormalDistribution(0.0, 1.0).getSurvival(5183.8), 5.981252345049083e-18),
              1e-13);
    EXPECT_LT(relErr(LogNormalDistribution(1.0, 0.5).getSurvival(1e6), 3.4416754770828724e-145),
              1e-12);
    // Subnormal S: within one unit of the subnormal grid. libm's erfc is a few units off there
    // and ½erfc rounds twice, which returned 0 at the second point (reference 3.8e-324).
    const double tiny = std::numeric_limits<double>::denorm_min();
    EXPECT_NEAR(LogNormalDistribution(0.0, 1.0).getSurvival(3e16), 2.823174306e-315, tiny);
    EXPECT_EQ(LogNormalDistribution(0.0, 0.01).getSurvival(1.4692339448720562), tiny);
    EXPECT_EQ(LogNormalDistribution(0.0, 1.0).getSurvival(8e16), 0.0);
    EXPECT_EQ(LogNormalDistribution(0.0, 1.0).getSurvival(0.0), 1.0);
    expectComplementary(LogNormalDistribution(0.0, 1.0), 0.5);
}

// Old: 0.024 relative at b − 2 ulp of (−1e-3, 1e-3); 0.2 at b − 1 ulp of (0.1, 0.7).
TEST(SurvivalTails, Uniform) {
    EXPECT_LT(relErr(UniformDistribution(-1e-3, 1e-3).getSurvival(below(1e-3, 2)),
                     2.1684043449710089e-16),
              2 * kEps);
    EXPECT_LT(
        relErr(UniformDistribution(0.1, 0.7).getSurvival(below(0.7, 1)), 1.8503717077085943e-16),
        2 * kEps);
    EXPECT_EQ(UniformDistribution(0.0, 1.0).getSurvival(-1.0), 1.0);
    EXPECT_EQ(UniformDistribution(0.0, 1.0).getSurvival(2.0), 0.0);
    expectComplementary(UniformDistribution(0.1, 0.7), 0.2);
}

// Old: 9.5e-11 relative at k = 999999 of (−1e6, 1e6); 2.7e-8 at 999999999.5 of (0, 1e9).
TEST(SurvivalTails, Discrete) {
    EXPECT_LT(
        relErr(DiscreteDistribution(-1000000, 1000000).getSurvival(999999.0), 4.99999750000125e-07),
        2 * kEps);
    EXPECT_LT(relErr(DiscreteDistribution(0, 1000000000).getSurvival(999999999.5), 9.99999999e-10),
              2 * kEps);
    EXPECT_EQ(DiscreteDistribution(0, 9).getSurvival(9.0), 0.0);
    EXPECT_EQ(DiscreteDistribution(0, 9).getSurvival(-0.5), 1.0);
    expectComplementary(DiscreteDistribution(0, 9), 3.5);
}

// Old: 0 at all three points.
TEST(SurvivalTails, StudentT) {
    EXPECT_LT(
        relErr(StudentTDistribution(100.0).getSurvival(9.999999999999998), 4.950844492297114e-17),
        1e-13);
    EXPECT_LT(relErr(StudentTDistribution(3.0).getSurvival(1e6), 1.1026577908396146e-18), 1e-13);
    EXPECT_LT(relErr(StudentTDistribution(1.5).getSurvival(1e30), 3.770852432016246e-46), 1e-13);
    expectComplementary(StudentTDistribution(5.0), -1.0);
}

// Old: 0 at the first two points; Beta(30, 2) at 0.9999 lost 1.2e-11 relative to 1 − F.
TEST(SurvivalTails, Beta) {
    EXPECT_LT(relErr(BetaDistribution(0.5, 1000.0).getSurvival(0.0492), 9.782854830259111e-24),
              1e-13);
    EXPECT_LT(relErr(BetaDistribution(2.0, 3.0).getSurvival(0.999999), 3.9999970003450675e-18),
              1e-13);
    EXPECT_LT(relErr(BetaDistribution(30.0, 2.0).getSurvival(0.9999), 4.641019432706218e-06),
              1e-13);
    EXPECT_EQ(BetaDistribution(2.0, 3.0).getSurvival(0.0), 1.0);
    EXPECT_EQ(BetaDistribution(2.0, 3.0).getSurvival(1.0), 0.0);
    expectComplementary(BetaDistribution(2.0, 3.0), 0.2);
}

// Old: 0 at M_PI and at 3.0 (κ = 50). The double M_PI lies below π, so S(M_PI) is the mass on
// (M_PI, π), not 0; −M_PI wraps to the same t.
TEST(SurvivalTails, VonMises) {
    const double s_pi = 1.9490839671742587e-17;
    EXPECT_LT(relErr(VonMisesDistribution(0.0, 1e-6).getSurvival(std::numbers::pi), s_pi), 1e-13);
    EXPECT_LT(relErr(VonMisesDistribution(0.0, 1e-6).getSurvival(-std::numbers::pi), s_pi), 1e-13);
    EXPECT_LT(relErr(VonMisesDistribution(0.0, 50.0).getSurvival(3.0), 1.7715594808454948e-44),
              1e-13);
    EXPECT_LT(relErr(VonMisesDistribution(0.0, 2.0).getSurvival(3.1), 0.0003932261765160266),
              1e-13);
    EXPECT_EQ(VonMisesDistribution(0.0, 2.0).getSurvival(kInf), 0.0);
    EXPECT_EQ(VonMisesDistribution(0.0, 2.0).getSurvival(-kInf), 1.0);
    expectComplementary(VonMisesDistribution(0.0, 2.0), -1.0);
    expectComplementary(VonMisesDistribution(0.0, 0.0), -1.0);
}

// NaN in, NaN out, for every override.
TEST(SurvivalTails, NaNPropagates) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(CauchyDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(GumbelDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(LaplaceDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(LogisticDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(HalfNormalDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(LogNormalDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(UniformDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(DiscreteDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(StudentTDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(BetaDistribution().getSurvival(nan)));
    EXPECT_TRUE(std::isnan(VonMisesDistribution().getSurvival(nan)));
}
