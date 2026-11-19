// tests/test_corvus_adoption_guards.cpp
//
// Regression guards for defects the corvus adoption (v3.0.0) removed. Each
// case is a value, or a monotonicity property, that v2.5.0's own special
// functions got wrong by more than the budget asserted here, so each fails
// against a v2.5.0 build and passes on the adoption. Shown so on the Ivy
// Bridge machine (AVX tier, GCC 13.3) on 2026-11-19; the failing margins are
// in PLAN.md beside the disposition table.
//
//   #136  detail::erf_inv in its extreme-tail band: wrong values (7.59 for
//         5.4726 at 1 - 1e-14) and non-monotone.
//   #126  beta_i at b >= 1e9: about 1e-6 absolute through the lgamma
//         cancellation in its prefix.
//   #270  the same loss seen through NegativeBinomial with a tiny p.
//   #248  Student-t CDF just below nu = 40.
//   #266  (CDF half) the shape-20 prefactor switch in the Gamma and Beta CDFs.
//   #271  Gamma CDF at tiny shape.
//   #247  Beta CDF non-monotone at shapes <= 1e-3.
//   #262  the survival at exact dyadic ties, Binomial(n, 1/2) and NB(r, 1/2).
//   #214  Gamma quantile non-monotone between adjacent p near 1/2.
//
// Not here, because no case could be made to fail against v2.5.0 on this
// machine: #141 and #113 (Gamma(1e4), ChiSquared(1e5) and Student-t(30) CDFs
// already read within 5 eps there; tests/test_incomplete_gamma_beta_accuracy.cpp
// is their guard) and #126's Geometric form (already closed-form in v2.5.0).
//
// References are mpmath at 60 digits unless a closed form is given. Budgets
// are in units of DBL_EPSILON relative to the reference and leave the
// adoption several times its measured error; they are far below what v2.5.0
// reached, which is the point.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/core/math_utils.h"
#include "libstats/libstats.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>

using namespace stats;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

// |value - ref| in units of DBL_EPSILON * |ref|.
[[nodiscard]] double errEps(double value, double ref) {
    return std::fabs(value - ref) / (kEps * std::fabs(ref));
}

// The double k representable steps above x (k may be negative); x finite and positive.
[[nodiscard]] double stepped(double x, std::int64_t k) {
    std::int64_t bits = 0;
    std::memcpy(&bits, &x, sizeof bits);
    bits += k;
    double out = 0.0;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

}  // namespace

// #136. The arguments are the doubles nearest 1 - 1e-10 ... 1 - 1e-15.
TEST(CorvusAdoptionGuards, ErfInvExtremeTailValues) {
    struct Row {
        double x;
        double ref;
    };
    static constexpr Row kRows[] = {
        {0x1.ffffffff24190p-1, 4.572824958544925},  {0x1.ffffffffea028p-1, 4.812924058944833},
        {0x1.fffffffffdcd1p-1, 5.042031898572696},  {0x1.ffffffffffc7bp-1, 5.261483331372676},
        {0x1.fffffffffffa6p-1, 5.4725525174145355}, {0x1.ffffffffffff7p-1, 5.675915739744713},
    };
    for (const Row& row : kRows) {
        EXPECT_LE(errEps(detail::erf_inv(row.x), row.ref), 8.0)
            << "erf_inv(" << row.x << ") = " << detail::erf_inv(row.x);
        EXPECT_LE(errEps(detail::erf_inv(-row.x), -row.ref), 8.0) << "odd symmetry at " << row.x;
    }
}

TEST(CorvusAdoptionGuards, ErfInvMonotoneInTheLastDoublesBelowOne) {
    // The 20,000 doubles just below 1 cover the whole band the issue reports.
    double x = std::nextafter(1.0, 0.0);
    double above = detail::erf_inv(x);
    int decreases = 0;
    for (int i = 0; i < 20000; ++i) {
        x = std::nextafter(x, 0.0);
        const double here = detail::erf_inv(x);
        if (here > above)
            ++decreases;
        above = here;
    }
    EXPECT_EQ(decreases, 0);
}

// #126. I_p(1, k + 1) = 1 - (1 - p)^(k + 1), the issue's own gate.
TEST(CorvusAdoptionGuards, BetaIAtHugeSecondShapeMatchesClosedForm) {
    for (double k : {1e8, 1e9, 1e10}) {
        for (double scale : {0.01, 0.1, 0.5, 1.0, 2.0, 5.0}) {
            const double p = scale / k;
            const double ref = -std::expm1((k + 1.0) * std::log1p(-p));
            EXPECT_LE(errEps(detail::beta_i(p, 1.0, k + 1.0), ref), 8.0) << "k=" << k << " p=" << p;
        }
    }
}

// #270.
TEST(CorvusAdoptionGuards, NegativeBinomialCdfAtTinyP) {
    auto a = NegativeBinomialDistribution::create(3.0, 1e-20).unwrap();
    EXPECT_LE(errEps(a.getCumulativeProbability(2.67e20), 0.4989982386108019), 8.0);
    auto b = NegativeBinomialDistribution::create(3.0, 1e-12).unwrap();
    EXPECT_LE(errEps(b.getCumulativeProbability(2.67e12), 0.4989982386116252), 8.0);
}

// #248. The row sits just below the nu = 40 branch; v2.5.0 reads 333 eps here.
TEST(CorvusAdoptionGuards, StudentTCdfJustBelowTheShapeBranch) {
    auto t40 = StudentTDistribution::create(39.999999999997975).unwrap();
    EXPECT_LE(errEps(t40.getCumulativeProbability(-1.5993735651810874), 0.058803856564873705),
              32.0);
}

// #266, CDF half. Shape 19 is below the Stirling-prefactor switch at 20. Gamma is (shape, rate).
TEST(CorvusAdoptionGuards, CdfBelowThePrefactorShapeSwitch) {
    auto gamma = GammaDistribution::create(19.0, 1.0).unwrap();
    const double x = 19.999999999999922;
    EXPECT_LE(errEps(gamma.getCumulativeProbability(x), 0.6185780505528387), 8.0);
    EXPECT_LE(gamma.getCumulativeProbability(x),
              gamma.getCumulativeProbability(std::nextafter(x, 100.0)));

    auto beta = BetaDistribution::create(19.0, 1e15).unwrap();
    EXPECT_LE(errEps(beta.getCumulativeProbability(1.8666666666666312e-14), 0.4999020048898538),
              8.0);
}

// #271.
TEST(CorvusAdoptionGuards, GammaCdfAtTinyShape) {
    auto gamma = GammaDistribution::create(1e-4, 1.0).unwrap();
    EXPECT_LE(errEps(gamma.getCumulativeProbability(1e-12), 0.9972982674968738), 4.0);
}

// #247.
TEST(CorvusAdoptionGuards, BetaCdfMonotoneAcrossAdjacentDoublesAtTinyShapes) {
    int cdf_decreases = 0;
    int survival_increases = 0;
    for (double a : {1e-3, 1e-6, 1e-10}) {
        for (double b : {1e-3, 1e-6, 0.5}) {
            auto beta = BetaDistribution::create(a, b).unwrap();
            for (double x0 : {0.1, 0.3, 0.5, 0.7, 0.9}) {
                double f_prev = beta.getCumulativeProbability(x0);
                double s_prev = beta.getSurvival(x0);
                for (std::int64_t k = 1; k <= 1000; ++k) {
                    const double x = stepped(x0, k);
                    const double f = beta.getCumulativeProbability(x);
                    const double s = beta.getSurvival(x);
                    if (f < f_prev)
                        ++cdf_decreases;
                    if (s > s_prev)
                        ++survival_increases;
                    f_prev = f;
                    s_prev = s;
                }
            }
        }
    }
    EXPECT_EQ(cdf_decreases, 0);
    EXPECT_EQ(survival_increases, 0);
}

// #262. At p = 1/2 every CDF and survival value is a dyadic rational and exact in a double.
TEST(CorvusAdoptionGuards, SurvivalExactAtDyadicTies) {
    auto nb = NegativeBinomialDistribution::create(2.0, 0.5).unwrap();
    EXPECT_EQ(nb.getSurvival(7.0), 0.01953125);  // 10 / 512

    for (int n = 1; n <= 40; ++n) {
        auto bin = BinomialDistribution::create(n, 0.5).unwrap();
        // Exact tail sums of C(n, j) / 2^n, built downward from j = n.
        double coeff = 1.0;  // C(n, n)
        double tail = 0.0;   // sum over j > k of C(n, j)
        const double scale = std::ldexp(1.0, -n);
        for (int k = n - 1; k >= 0; --k) {
            tail += coeff;  // adds C(n, k + 1)
            coeff = coeff * static_cast<double>(k + 1) / static_cast<double>(n - k);  // C(n, k)
            const double ref = tail * scale;
            EXPECT_LE(errEps(bin.getSurvival(static_cast<double>(k)), ref), 1.0)
                << "Binomial(" << n << ", 1/2) survival at k=" << k;
        }
    }
}

TEST(CorvusAdoptionGuards, QuantileOfCdfReturnsTheCountAtDyadicTies) {
    for (int n = 1; n <= 40; ++n) {
        auto bin = BinomialDistribution::create(n, 0.5).unwrap();
        for (int k = 0; k < n; ++k) {
            const double f = bin.getCumulativeProbability(static_cast<double>(k));
            EXPECT_EQ(bin.getQuantile(f), static_cast<double>(k))
                << "Binomial(" << n << ", 1/2), k=" << k;
        }
    }
    for (double r : {1.0, 2.0, 5.0, 20.0}) {
        auto nb = NegativeBinomialDistribution::create(r, 0.5).unwrap();
        for (int k = 0; k < 30; ++k) {
            const double f = nb.getCumulativeProbability(static_cast<double>(k));
            EXPECT_EQ(nb.getQuantile(f), static_cast<double>(k)) << "NB(" << r << ", 1/2), k=" << k;
        }
    }
}

// #214, the half near p = 1/2: Q must not decrease between adjacent p.
TEST(CorvusAdoptionGuards, GammaQuantileMonotoneBetweenAdjacentPNearOneHalf) {
    int decreases = 0;
    for (double shape : {0.004, 0.05, 2.5}) {
        auto gamma = GammaDistribution::create(shape, 1.0).unwrap();
        for (double p0 : {0.49999999999999767, 0.5, 0.25}) {
            double q_prev = gamma.getQuantile(p0);
            for (std::int64_t k = 1; k <= 2000; ++k) {
                const double q = gamma.getQuantile(stepped(p0, k));
                if (q < q_prev)
                    ++decreases;
                q_prev = q;
            }
        }
    }
    EXPECT_EQ(decreases, 0);
}
