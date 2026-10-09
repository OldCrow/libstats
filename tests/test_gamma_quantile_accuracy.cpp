// tests/test_gamma_quantile_accuracy.cpp
//
// Gamma quantiles against mpmath (issue #160): GammaDistribution::getQuantile,
// the Erlang and ChiSquared quantiles that delegate to it, and
// detail::inverse_chi_squared_cdf and detail::gamma_inverse_cdf. Before v2.4.2
// the distribution solved P(α, βx) = p by Newton on the linear CDF from a
// Wilson-Hilferty seed and was 5e139 relative off at p = 1e-300; the two
// detail inverses bisected to an absolute 1e-8 in p.
//
// Two-sided per the regression-guard rule: each row must be finite and within
// the achievable-accuracy law (lawBudget below), or exactly 0 where the true
// quantile is below half the smallest subnormal.
//
// References: Gamma(α, 1) quantiles by bisection in log x on the small-side
// residual (P below the median, Q above it), mpmath at dps 60, evaluated at the
// double nearest each literal p.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

using namespace stats;

namespace {

struct Row {
    double alpha;
    double p;
    double x;  // Gamma(alpha, 1) quantile, mpmath; 0 where it underflows a double
};

constexpr double kUpper = 1 - 1e-15;
constexpr double kLawFactor = 8.0;  // the worst row uses 0.32 of it (Zen 4, MSVC)

constexpr Row kRows[] = {
    {0.01, 1e-300, 0.0},  // 5.7e-30001
    {0.01, 1e-100, 0.0},  // 5.7e-10001
    {0.01, 1e-15, 0.0},   // 5.7e-1501
    {0.01, 0.3, 2.9174171917458686172e-53},
    {0.01, 0.7, 1.8309524563808334366e-16},
    {0.01, kUpper, 2.6654705498139439301e+1},
    // The upper tail at small shape, where Q = 1 − P cancelled: 500x the budget at α = 1e-4.
    {1e-4, 0.99995, 5.5325010273648020406e-1},
    {1e-3, 0.9995, 5.5350736918717457462e-1},
    {1e-3, 0.9999, 1.501028147278442429},
    {0.01, 0.995, 5.5606758767600336574e-1},
    {2.5, 1e-300, 1.6167038902915641898e-120},
    {2.5, 1e-100, 1.6167038902915641865e-40},
    {2.5, 1e-15, 1.6167046370724630017e-6},
    {2.5, 0.3, 1.499954066379953105},
    {2.5, 0.7, 3.032214992077452192},
    {2.5, kUpper, 3.9818831701263427637e+1},
    {1e4, 1e-300, 6.7376871915903292373e+3},
    {1e4, 1e-100, 8.0204643833717386077e+3},
    {1e4, 1e-15, 9.2264285779217920269e+3},
    {1e4, 0.3, 9.9473192620195836341e+3},
    {1e4, 0.7, 1.0052197405331651789e+4},
    {1e4, kUpper, 1.0814955460936724432e+4},
};

constexpr double kEps = 0x1p-52;

// The achievable-accuracy law (tests/test_lognormal_cdf_accuracy.cpp, law_budget): a tail
// probability q is reachable only to a relative |ln q|·2^-52, and the quantile turns that into
// a relative error in x of the same times κ = d(ln x)/d(ln q) = q/(x·pdf(x)) — 1/α in the deep
// lower tail, a hundredfold at α = 0.01. κ is evaluated at the reference x.
double lawBudget(double alpha, double p, double x) {
    if (x == 0.0)
        return 0.0;  // an underflow row: expectQuantile wants exactly 0
    const double q = std::min(p, 1.0 - p);
    const auto unit = GammaDistribution::create(alpha, 1.0).unwrap();
    const double kappa = q / (x * unit.getProbability(x));
    return kLawFactor * kEps * (1.0 + std::max(1.0, std::fabs(std::log(q))) * kappa);
}

void expectQuantile(const std::string& what, double got, double want, double budget) {
    if (want == 0.0) {
        EXPECT_EQ(got, 0.0) << what << ": the true quantile underflows";
        return;
    }
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got;
    EXPECT_LE(std::fabs(got - want) / want, budget) << what << " = " << got << ", want " << want;
}

std::string caseLabel(const char* name, double a, double p) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "%s(%g).quantile(%.17g)", name, a, p);
    return buf;
}

}  // namespace

TEST(GammaQuantileAccuracy, UnitRate) {
    for (const Row& r : kRows) {
        const auto d = GammaDistribution::create(r.alpha, 1.0).unwrap();
        expectQuantile(caseLabel("Gamma", r.alpha, r.p), d.getQuantile(r.p), r.x,
                       lawBudget(r.alpha, r.p, r.x));
    }
}

TEST(GammaQuantileAccuracy, RateScalesTheAnswer) {
    // Rate 4 is a power of two, so the reference scales exactly.
    for (const Row& r : kRows) {
        const auto d = GammaDistribution::create(r.alpha, 4.0).unwrap();
        expectQuantile(caseLabel("Gamma rate 4", r.alpha, r.p), d.getQuantile(r.p), r.x / 4.0,
                       lawBudget(r.alpha, r.p, r.x));
    }
}

TEST(GammaQuantileAccuracy, DetailGammaInverseCdf) {
    for (const Row& r : kRows)
        expectQuantile(caseLabel("gamma_inverse_cdf", r.alpha, r.p),
                       detail::gamma_inverse_cdf(r.p, r.alpha, 1.0), r.x,
                       lawBudget(r.alpha, r.p, r.x));
}

TEST(GammaQuantileAccuracy, ChiSquaredAndErlangDelegate) {
    // Chi-squared(k) = Gamma(k/2, scale 2); Erlang(k, rate λ) = Gamma(k, rate λ).
    struct Delegated {
        double k;
        double p;
        double x;  // Gamma(k/2 or k, 1) quantile
    };
    constexpr Delegated kChi[] = {
        {5.0, 1e-300, 1.6167038902915641898e-120},
        {1.0, 1e-15, 7.8539816339744843168e-31},
        {100.0, kUpper, 1.283187707744315709e+2},
    };
    for (const auto& c : kChi) {
        const auto d = ChiSquaredDistribution::create(c.k).unwrap();
        expectQuantile(caseLabel("ChiSquared", c.k, c.p), d.getQuantile(c.p), 2.0 * c.x,
                       lawBudget(0.5 * c.k, c.p, c.x));
        expectQuantile(caseLabel("inverse_chi_squared_cdf", c.k, c.p),
                       detail::inverse_chi_squared_cdf(c.p, c.k), 2.0 * c.x,
                       lawBudget(0.5 * c.k, c.p, c.x));
    }
    const auto erlang = ErlangDistribution::create(3, 2.0).unwrap();
    expectQuantile("Erlang(3, 2).quantile(1e-100)", erlang.getQuantile(1e-100),
                   8.4343266530174924847e-34 / 2.0,
                   lawBudget(3.0, 1e-100, 8.4343266530174924847e-34));
}

TEST(GammaQuantileAccuracy, Edges) {
    const auto d = GammaDistribution::create(2.5, 1.0).unwrap();
    EXPECT_EQ(d.getQuantile(0.0), 0.0);
    EXPECT_EQ(d.getQuantile(1.0), std::numeric_limits<double>::infinity());
    EXPECT_TRUE(std::isnan(detail::gamma_p_inv(2.5, std::nan(""))));
    EXPECT_TRUE(std::isnan(detail::gamma_p_inv(-1.0, 0.5)));
}

// ---------------------------------------------------------------------------------------------
// #223: the upper tail at shape below ~0.004. Unfixed, the solver counted an underflowed tail
// (log Q = -inf at an overshoot) as a sighting of the f < 0 side and ended after two real
// evaluations: Gamma(0.001338, 0.9328).Q(0.99991893) = 1.89678, F of it 0.99990936. References:
// mpmath at the double nearest each p, Gamma(alpha, 1).
// ---------------------------------------------------------------------------------------------
TEST(GammaQuantileAccuracy, SmallShapeUpperTail) {
    constexpr Row kSmall[] = {
        {1e-3, 0.9, 9.821659644066882004576e-47},
        {1e-3, 0.9999, 1.501028147278442428997},
        {1e-3, 0.99991893441051749, 1.643979324753423276912},
        {1e-3, 0.999999, 5.120025083764955183661},
        {1e-3, 0.9999999999, 13.45459543385349244652},
        {1e-3, kUpper, 24.40223745447778092045},
        {0.001338322665, 0.9, 3.627586974584746235226e-35},
        {0.001338322665, 0.9999, 1.70071918857214635116},
        {0.001338322665, 0.99991893441051749, 1.848163801228761507739},
        {0.001338322665, 0.999999, 5.370622664769253734861},
        {0.001338322665, 0.9999999999, 13.72823039249922034341},
        {0.001338322665, kUpper, 24.68390006402349043988},
        {0.004, 0.9, 2.048197239726874248315e-12},
        {0.004, 0.9999, 2.502701061793707275841},
        {0.004, 0.99991893441051749, 2.663194619805208500086},
        {0.004, 0.999999, 6.327899360533214306612},
        {0.004, 0.9999999999, 14.76361089473275438826},
        {0.004, kUpper, 25.74838223827471475642},
    };
    for (const Row& r : kSmall) {
        const auto d = GammaDistribution::create(r.alpha, 1.0).unwrap();
        expectQuantile(caseLabel("Gamma", r.alpha, r.p), d.getQuantile(r.p), r.x,
                       lawBudget(r.alpha, r.p, r.x));
    }
    // The reproduction's own instance: Q non-decreasing in p over the upper tail, and F(Q(p))
    // within 4 ulps of p — F is formed as 1 − Q, so it carries a few ulps of its own; the
    // unfixed point was 1e-5 off.
    const auto d = GammaDistribution::create(0.001338322665, 0.932830774).unwrap();
    double prev = 0.0;
    for (int i = 0; i <= 300; ++i) {
        const double q = std::pow(10.0, -1.0 - 14.0 * i / 300.0);  // 1e-1 .. 1e-15
        const double p = 1.0 - q;
        const double x = d.getQuantile(p);
        ASSERT_TRUE(std::isfinite(x)) << "p = " << p;
        EXPECT_GE(x, prev) << "Q not monotone at p = " << p;
        prev = x;
        EXPECT_NEAR(d.getCumulativeProbability(x), p, 9e-16) << "p = " << p << ", Q = " << x;
    }
}

// ---------------------------------------------------------------------------------------------
// #214: p below 1e-300, where the tail underflows a double but the quantile does not. Unfixed,
// the residual was log of an underflowed P, so Newton bisected on -inf and stopped on noise:
// Gamma(1000).Q(5e-324) = 310.23 (F = 3e-211) against 218.26. References: mpmath at the double
// nearest each p.
// ---------------------------------------------------------------------------------------------
TEST(GammaQuantileAccuracy, SubnormalP) {
    constexpr double kDenormMin = 4.9406564584124654e-324;
    constexpr double kDblMin = 2.2250738585072014e-308;
    constexpr Row kDeep[] = {
        {2.5, 1e-305, 1.616703890291564171207e-122},
        {2.5, 1e-310, 1.616703890291562197956e-124},
        {2.5, 1e-320, 1.616696690879892674495e-128},
        {2.5, kDblMin, 1.404650428233620193792e-123},
        {2.5, kDenormMin, 7.693861180581284968062e-130},
        {1000.0, 1e-305, 230.4482772402535225192},
        {1000.0, 1e-310, 227.0351370192861905185},
        {1000.0, 1e-320, 220.4023309141524485676},
        {1000.0, kDblMin, 228.629282961407008988},
        {1000.0, kDenormMin, 218.2642471413576461088},
        {1e5, 1e-305, 88647.0893115270708381},
        {1e5, 1e-310, 88557.65163841593244452},
        {1e5, 1e-320, 88381.10219290181632749},
        {1e5, kDblMin, 88599.54135110867087206},
        {1e5, kDenormMin, 88323.39377026785200909},
    };
    for (const Row& r : kDeep) {
        const auto d = GammaDistribution::create(r.alpha, 1.0).unwrap();
        expectQuantile(caseLabel("Gamma", r.alpha, r.p), d.getQuantile(r.p), r.x,
                       lawBudget(r.alpha, r.p, r.x));
    }
    // ChiSquared(1000) = Gamma(500, scale 2): the issue's reproduction.
    struct ChiRow {
        double p, x;
    };
    constexpr ChiRow kChi[] = {
        {1e-305, 100.6522790442206617454},    {1e-310, 98.11217486562840713564},
        {1e-320, 93.24221021413898176691},    {kDblMin, 99.29570276561132336866},
        {kDenormMin, 91.6912853493057332716},
    };
    const auto chi = ChiSquaredDistribution::create(1000.0).unwrap();
    for (const ChiRow& r : kChi)
        expectQuantile(caseLabel("ChiSquared", 1000.0, r.p), chi.getQuantile(r.p), r.x,
                       lawBudget(500.0, r.p, r.x / 2.0));
    // Monotone over the subnormal range, Gamma, ChiSquared and Erlang.
    const double ps[] = {kDenormMin, 1e-320, 1e-310, kDblMin, 1e-305, 1e-300, 1e-299};
    const auto gamma = GammaDistribution::create(1000.0, 1.0).unwrap();
    const auto erlang = ErlangDistribution::create(1000, 1.0).unwrap();
    double qg = 0.0, qc = 0.0, qe = 0.0;
    for (double p : ps) {
        EXPECT_GE(gamma.getQuantile(p), qg) << "Gamma(1000) at p = " << p;
        EXPECT_GE(chi.getQuantile(p), qc) << "ChiSquared(1000) at p = " << p;
        EXPECT_GE(erlang.getQuantile(p), qe) << "Erlang(1000) at p = " << p;
        qg = gamma.getQuantile(p);
        qc = chi.getQuantile(p);
        qe = erlang.getQuantile(p);
    }
}

// ---------------------------------------------------------------------------------------------
// #226: large shape. Unfixed, the incomplete-gamma series hit its iteration cap from
// alpha ~ 1e10 (Gamma(1e16).Q(0.1) returned the point where F = 0.5) and took 1e6 terms per
// evaluation below it (4.6 s per quantile at alpha ~ 1e299). The CDF is now Temme's expansion
// there, and from alpha = 1e16 the quantile is the Wilson-Hilferty value, exact to double.
// References: Newton on the quadrature oracle (1e12 up) or the Kummer series (below), mpmath,
// at the double nearest each p.
// ---------------------------------------------------------------------------------------------
TEST(GammaQuantileAccuracy, LargeShape) {
    constexpr double kDenormMin = 4.9406564584124654e-324;
    constexpr Row kLarge[] = {
        {1e6, kDenormMin, 962023.9263240446037883215},
        {1e6, 1e-300, 963408.6539398657030362041},
        {1e6, 1e-15, 992079.3306128912828065577},
        {1e6, 0.1, 998718.6627499802811253291},
        {1e6, 0.5, 999999.6666666864197602979},
        {1e6, 0.9, 1001281.765499620957506265},
        {1e6, kUpper, 1007962.145687055725856382},
        {1e8, kDenormMin, 99615818.70014412507089959},
        {1e8, 1e-300, 99629986.0588641649926391},
        {1e8, 1e-15, 99920607.23382324715277552},
        {1e8, 0.1, 99987184.69848843142722886},
        {1e8, 0.5, 99999999.66666666686419753},
        {1e8, 0.9, 100012815.7297611785840363},
        {1e8, kUpper, 100079435.13495765980336},
        {1e12, kDenormMin, 999961533087.2950469365016},
        {1e12, 1e-300, 999962953360.8616816612345},
        {1e12, 1e-15, 999992058675.362138498835},
        {1e12, 0.1, 999998718448.6485803953072},
        {1e12, 0.5, 999999999999.6666666666667},
        {1e12, 0.9, 1000001281551.779669214793},
        {1e12, kUpper, 1000007941465.176275195735},
        {1e16, kDenormMin, 9999996153259931.199314609},
        {1e16, 1e-300, 9999996295290827.226314096},
        {1e16, 1e-15, 9999999205865488.071222061},
        {1e16, 0.1, 9999999871844843.65966476},
        {1e16, 0.5, 9999999999999999.666666667},
        {1e16, 0.9, 10000000128155156.76858485},
        {1e16, kUpper, 10000000794144469.43044485},
        {1e20, kDenormMin, 99999999615325944321.4703},
        {1e20, 1e-300, 99999999629529037463.55046},
        {1e20, 1e-15, 99999999920586546758.97835},
        {1e20, 0.1, 99999999987184484344.76812},
        {1e20, 0.5, 99999999999999999999.66667},
        {1e20, 0.9, 100000000012815515655.6601},
        {1e20, kUpper, 100000000079414444894.8486},
    };
    for (const Row& r : kLarge) {
        const auto d = GammaDistribution::create(r.alpha, 1.0).unwrap();
        expectQuantile(caseLabel("Gamma", r.alpha, r.p), d.getQuantile(r.p), r.x,
                       lawBudget(r.alpha, r.p, r.x));
    }
    // Beyond ~1e32 the central quantiles round to alpha itself (x = alpha(1 + z/sqrt(alpha))),
    // and the CDF resolves nothing finer; unfixed, 1e100 gave 1.0000000000000111e100 and 1e299
    // gave alpha/e after 4.6 s. 300 calls bound the time: they take microseconds here.
    const auto t0 = std::chrono::steady_clock::now();
    for (double alpha : {1e32, 1e100, 1e299}) {
        const auto d = GammaDistribution::create(alpha, 1.0).unwrap();
        for (double p : {0.1, 0.5, 0.9})
            for (int i = 0; i < 100 / 3; ++i)
                EXPECT_EQ(d.getQuantile(p), alpha) << "alpha = " << alpha << ", p = " << p;
    }
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    EXPECT_LT(seconds, 1.0) << "300 huge-shape quantiles";
}
