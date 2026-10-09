// tests/test_beta_quantile_accuracy.cpp
//
// Beta quantiles against mpmath (issue #137, defect-hunt D2): BetaDistribution::getQuantile and
// detail::inverse_beta_i, which it delegates to. Before the fix the detail inverse ran Newton on
// the linear CDF to an absolute 1e-8 from a start clamped into [1e-8, 1 − 1e-8]: Beta(1, 1).Q(p)
// was 1e-8 for every p below 1e-8, Beta(25, 1000).Q(6.57e-5) was 0.9999999999 (true 0.01005),
// Beta(2, 3).Q(1 − 1.7e-6) was 1.4e-5 off in x. The solver now runs Newton in the logit on the
// log of the small-side probability (the #160 gamma_p_inv pattern), with the leading term
// x^a/(a·B) as the answer where the root underflows the logit.
//
// Two-sided per the regression-guard rule: each row must be finite and within the
// achievable-accuracy law (lawBudget below) of the mpmath value, on the side nearer the root
// (x below the median, 1 − x above it), or exactly 0 / exactly 1 where the true quantile is
// closer to that endpoint than half the smallest subnormal / half an ulp of 1.
//
// References: x with I_x(a, b) = p by bisection in log x (log(1 − x) for p > ½, on the reflected
// problem I_{1−x}(b, a) = 1 − p), mpmath betainc at dps 50, evaluated at the double nearest each
// literal p; both x and 1 − x are recorded to 25 digits so the near-1 rows keep their precision.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <string>

using namespace stats;

namespace {

struct Row {
    double a;
    double b;
    double p;
    double x;     // Beta(a, b) quantile, mpmath; 0 where it underflows a double
    double comp;  // 1 − x, mpmath; 0 where x rounds to 1
};

constexpr double kUpper = 1 - 1e-15;
constexpr double kTiny = 4.9406564584124654e-324;  // the smallest subnormal
constexpr double kLawFactor = 8.0;  // the worst row uses 0.34 of it (Kaby Lake, AppleClang)

constexpr Row kRows[] = {
    // The D2 reproductions.
    {2, 3, 0.9999982705701526, 0.992424013054514820258662, 0.007575986945485179741337983},
    {2, 3, 3.489e-4, 0.007664767935227900120065979, 0.992335232064772099879934},
    {2, 3, 1e-100, 4.082482904638630204470435e-51, 1.0},
    {2, 3, 1e-300, 4.082482904638630214813797e-151, 1.0},
    {2, 3, kTiny, 9.07437459590876841323533e-163, 1.0},
    {2, 3, 0.3, 0.2723839420751053410308959, 0.7276160579248946589691041},
    {2, 3, 0.7, 0.5084047548725843654733989, 0.4915952451274156345266011},
    {2, 3, kUpper, 0.9999937020636601583465921, 6.297936339841653407885171e-06},
    {3, 2, 1.7294298474399e-06, 0.00757598694548314868340276, 0.9924240130545168513165972},
    // Beta(1, 1) is Uniform: the quantile is p itself, down to the smallest subnormal.
    {1, 1, 1e-15, 1.0000000000000000777054e-15, kUpper},
    {1, 1, 1e-100, 1.0000000000000000199919e-100, 1.0},
    {1, 1, 1e-300, 1.000000000000000025059092e-300, 1.0},
    {1, 1, kTiny, kTiny, 1.0},
    {1, 1, 0.3, 0.2999999999999999888977698, 0.7000000000000000111022302},
    {1, 1, 0.7, 0.699999999999999955591079, 0.300000000000000044408921},
    {1, 1, kUpper, 0.9999999999999990007992778, 9.992007221626408863812685e-16},
    // Large shapes, both orders (the reflection must be exact).
    {25, 1000, 6.57e-5, 0.01004772277780727647525115, 0.9899522772221927235247488},
    {25, 1000, 1e-300, 1.005671958261379747294551e-14, 0.9999999999999899432804174},
    {25, 1000, 1e-15, 0.002814379459574917520086894, 0.9971856205404250824799131},
    {25, 1000, 0.5, 0.02408147520966657224914693, 0.9759185247903334277508531},
    {25, 1000, kUpper, 0.08270099059502927013514374, 0.9172990094049707298648563},
    {1000, 25, 1e-15, 0.9173000032603581645220905, 0.08269999673964183547790952},
    {1000, 25, 0.5, 0.9759185247903334277508531, 0.02408147520966657224914693},
    {1000, 25, 0.9999343, 0.989952277222193254038175, 0.01004772277780674596182503},
    {1000, 25, kUpper, 0.9971857214462382700359172, 0.002814278553761729964082764},
    {1000, 1000, 1e-15, 0.4118975910769509029452314, 0.5881024089230490970547686},
    {1000, 1000, 0.3, 0.4941364927279814918481411, 0.5058635072720185081518589},
    {1000, 1000, 0.5, 0.5, 0.5},
    {1000, 1000, kUpper, 0.5881034917731916847208879, 0.4118965082268083152791121},
    // Small shapes: U-shaped and the deep tails, where the law's κ = 1/a amplifies.
    {0.5, 0.5, 1e-300, 0.0, 1.0},  // 2.5e-600
    {0.5, 0.5, 1e-100, 2.467401100272339753364694e-200, 1.0},
    {0.5, 0.5, 1e-15, 2.467401100272340038169401e-30, 1.0},
    {0.5, 0.5, 0.3, 0.2061073738537634213069226, 0.7938926261462365786930774},
    {0.5, 0.5, kUpper, 1.0, 2.463458398528700447570016e-30},
    {0.01, 0.01, 1e-15, 0.0, 1.0},  // 1.2e-1470
    {0.01, 0.01, 0.3, 6.428119935718612424247993e-23, 1.0},
    {0.01, 0.01, 0.49, 0.1156816550148711815136767, 0.8843183449851288184863233},
    {0.01, 0.01, 0.51, 0.8843183449851288184863233, 0.1156816550148711815136767},
    {0.01, 0.01, kUpper, 1.0, 0.0},  // 1 − 1.2e-1470
    {1e-3, 1e-3, 0.25, 9.317319597711542184556118e-302, 1.0},
    {1e-3, 1e-3, 0.75, 1.0, 0.0},
    {7, 7, 1e-200, 9.25758121938355153167227e-30, 1.0},
    {7, 7, 0.4, 0.4655767542089523359840921, 0.5344232457910476640159079},
    // Large shapes at subnormal and deep-tail p (the CI identity test's gap): the prefactor
    // x^a(1 − x)^b/B underflows a double long before the tail does, so a residual on log of the
    // formed I was −inf across the whole left side and the root landed at x = 1 or in the bulk
    // (Beta(1e4, 1e4).Q(4.9e-324) = 1, Beta(1e5, 2e5).Q(4.9e-324) = 0.48). The residual now
    // keeps the prefactor in the log. References: betainc up to shape 1000; above that the exact
    // binomial identity I_x(a, b) = P(Bin(a + b − 1, x) ≥ a), summed in log form at dps 50.
    {1e4, 1e4, kTiny, 0.3664728485760300136033065, 0.6335271514239699863966935},
    {1e4, 1e4, 1e-320, 0.3671361563855558811131094, 0.6328638436144441188868906},
    {1e4, 1e4, 1e-300, 0.3712325170640976973873224, 0.6287674829359023026126776},
    {1e4, 1e4, 1e-100, 0.4252095419236987713644189, 0.5747904580763012286355811},
    {1e5, 2e5, kTiny, 0.3008112212173126993026988, 0.6991887787826873006973012},
    {1e5, 2e5, 1e-320, 0.3009756631976626246916782, 0.6990243368023373753083218},
    {1e5, 2e5, 1e-300, 0.3019899126288922453411011, 0.6980100873711077546588989},
    {1e5, 2e5, 1e-100, 0.3151977375817438757396669, 0.6848022624182561242603331},
    {25, 1000, kTiny, 1.175459653232399320140254e-15, 0.9999999999999988245403468},
    {25, 1000, 1e-320, 1.593881930717016187099259e-15, 0.9999999999999984061180693},
    {1000, 25, kTiny, 0.4307467158407831535245606, 0.5692532841592168464754394},
    {1000, 25, 1e-320, 0.4340998674088249034617041, 0.5659001325911750965382959},
    {1000, 1000, kTiny, 0.1384383837250824810455866, 0.8615616162749175189544134},
    {1000, 1000, 1e-300, 0.1476444413346902467967858, 0.8523555586653097532032142},
    {1000, 1000, 1e-100, 0.2749725424386269575582672, 0.7250274575613730424417328},
    // One shape below 1, both orders: the root is on the far side from the mass.
    {5, 0.1, 1e-15, 0.002099654497455523446605129, 0.9979003455025444765533949},
    {5, 0.1, 0.5, 0.9998698504475908024680978, 0.0001301495524091975319022318},
    {5, 0.1, 0.9, 0.9999999999866789912774595, 1.332100872254053653100671e-11},
    {5, 0.1, kUpper, 1.0, 0.0},
    {0.1, 5, 1e-15, 1.332100872189554475091222e-151, 1.0},
    {0.1, 5, 0.5, 0.0001301495524091975319022318, 0.9998698504475908024680978},
    {0.1, 5, kUpper, 0.9979006811454769089122247, 0.002099318854523091087775277},
};

// Beta(½, 1e6) and its mirror: beta_i itself is 2e-11 relative off at the median there (its
// continued fraction at b = 1e6; a CDF limit, not the solver's), so these rows get a flat budget
// that still catches the pre-fix 1e-8 collapse (Q(1e-100) was 1e-8, 2e196 times too large).
constexpr Row kLargeShapeRows[] = {
    {0.5, 1e6, 1e-300, 0.0, 1.0},  // 7.9e-607
    {0.5, 1e6, 1e-100, 7.853983597470137340674191e-207, 1.0},
    {0.5, 1e6, 0.5, 2.274682425559406873926327e-07, 0.9999997725317574440593126},
    {0.5, 1e6, kUpper, 3.221550860802348707361684e-05, 0.9999677844913919765129264},
    {1e6, 0.5, 1e-15, 0.9999677852790914167431062, 3.221472090858325689375305e-05},
    {1e6, 0.5, 0.5, 0.9999997725317574440593126, 2.274682425559406873926327e-07},
    {1e6, 0.5, kUpper, 1.0, 7.841433584899885806700773e-37},
};
constexpr double kLargeShapeBudget = 1e-9;

constexpr double kEps = 0x1p-52;

double g_worst_usage = 0.0;  // worst |error| / allowance over every row, printed by BudgetUsage

// The achievable-accuracy law (tests/test_gamma_quantile_accuracy.cpp): a tail probability q
// is reachable only to a relative |ln q|·2^-52, and the quantile turns that into a relative
// error in the small-side distance s (x, or 1 − x above the median) of the same times
// κ = d(ln s)/d(ln q) = q/(s·pdf(x)) — 1/a in the deep lower tail, a hundredfold at a = 0.01.
double lawBudget(const Row& r) {
    if (r.x == 0.0 || r.comp == 0.0)
        return 0.0;  // an endpoint row: expectQuantile wants exactly 0 or exactly 1
    const bool upper = r.p > 0.5;
    const double q = upper ? 1.0 - r.p : r.p;
    const double s = upper ? r.comp : r.x;
    const double x = upper ? 1.0 - r.comp : r.x;
    const auto d = BetaDistribution::create(r.a, r.b).unwrap();
    const double kappa = q / (s * d.getProbability(x));
    EXPECT_TRUE(std::isfinite(kappa)) << "kappa at Beta(" << r.a << ", " << r.b << ") x=" << x;
    return kLawFactor * kEps * (1.0 + std::max(1.0, std::fabs(std::log(q))) * kappa);
}

// The error is measured on the small side: x below the median, 1 − x above it. One ulp of the
// returned double — |x|·2^-52, or 2^-53 for an x next to 1 — is representation, not solver.
void expectQuantile(const std::string& what, double got, const Row& r, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got;
    if (r.x == 0.0) {
        EXPECT_EQ(got, 0.0) << what << ": the true quantile underflows";
        return;
    }
    if (r.comp == 0.0) {
        EXPECT_EQ(got, 1.0) << what << ": the true quantile rounds to 1";
        return;
    }
    const bool upper = r.p > 0.5;
    const double want = upper ? r.comp : r.x;
    const double have = upper ? 1.0 - got : got;
    const double ulp = (upper ? 0.5 : r.x) * kEps;
    if (budget > 0.0)  // solver error past the ulp allowance, as a fraction of the law budget
        g_worst_usage = std::max(g_worst_usage, (std::fabs(have - want) - ulp) / (budget * want));
    EXPECT_LE(std::fabs(have - want), budget * want + ulp)
        << what << " = " << got << ", want " << (upper ? 1.0 - r.comp : r.x) << " (small side "
        << have << " vs " << want << ", rel " << std::fabs(have - want) / want << ", budget "
        << budget << ")";
}

std::string caseLabel(const char* name, double a, double b, double p) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s(%g, %g).quantile(%.17g)", name, a, b, p);
    return buf;
}

}  // namespace

TEST(BetaQuantileAccuracy, Distribution) {
    for (const Row& r : kRows) {
        const auto d = BetaDistribution::create(r.a, r.b).unwrap();
        expectQuantile(caseLabel("Beta", r.a, r.b, r.p), d.getQuantile(r.p), r, lawBudget(r));
    }
}

TEST(BetaQuantileAccuracy, DetailInverseBetaI) {
    for (const Row& r : kRows)
        expectQuantile(caseLabel("inverse_beta_i", r.a, r.b, r.p),
                       detail::inverse_beta_i(r.p, r.a, r.b), r, lawBudget(r));
}

TEST(BetaQuantileAccuracy, LargeShapeWithinCdfLimit) {
    for (const Row& r : kLargeShapeRows) {
        const auto d = BetaDistribution::create(r.a, r.b).unwrap();
        expectQuantile(caseLabel("Beta", r.a, r.b, r.p), d.getQuantile(r.p), r, kLargeShapeBudget);
    }
}

TEST(BetaQuantileAccuracy, RoundTripAndMonotone) {
    // F(Q(p)) returns p to the CDF's own accuracy wherever x carries the information (x not
    // within a few ulp of an endpoint), and Q is monotone across the switch to the reflected
    // problem at p = ½.
    const double shapes[][2] = {{2, 3}, {0.5, 0.5}, {25, 1000}, {5, 0.1}, {1, 1}};
    for (const auto& s : shapes) {
        const auto d = BetaDistribution::create(s[0], s[1]).unwrap();
        double prev = -1.0;
        for (double p = 1e-12; p < 1.0; p = std::min(p * 10.0, p + 0.05)) {
            const double x = d.getQuantile(p);
            EXPECT_GE(x, prev) << "non-monotone at Beta(" << s[0] << ", " << s[1] << ") p=" << p;
            prev = x;
            if (x > 1e-300 && 1.0 - x > 1e-12) {
                // Half an ulp of x moves F by pdf(x)·ulp(x)/2 — representation, not solver.
                const double back = d.getCumulativeProbability(x);
                const double ulp_shift = d.getProbability(x) * x * kEps;
                EXPECT_NEAR(back, p, 1e-12 * std::min(p, 1.0 - p) + ulp_shift + 1e-15)
                    << "F(Q(p)) at Beta(" << s[0] << ", " << s[1] << ") p=" << p;
            }
        }
    }
}

TEST(BetaQuantileAccuracy, Edges) {
    const auto d = BetaDistribution::create(2.0, 3.0).unwrap();
    EXPECT_EQ(d.getQuantile(0.0), 0.0);
    EXPECT_EQ(d.getQuantile(1.0), 1.0);
    EXPECT_EQ(detail::inverse_beta_i(0.5, 4.0, 4.0), 0.5);  // symmetric median, exact
    EXPECT_TRUE(std::isnan(detail::inverse_beta_i(std::nan(""), 2.0, 3.0)));
    EXPECT_TRUE(std::isnan(detail::inverse_beta_i(0.5, -1.0, 3.0)));
    EXPECT_TRUE(std::isnan(detail::inverse_beta_i(0.5, 2.0, 0.0)));
}

TEST(BetaQuantileAccuracy, BudgetUsage) {
    // Runs after the row tests (gtest keeps definition order within a suite): how much of the
    // allowance the worst row used, so a platform that lands near the edge is visible.
    std::printf("worst row used %.3f of its allowance\n", g_worst_usage);
    EXPECT_GT(g_worst_usage, 0.0);
}

TEST(BetaQuantileAccuracy, FisherFDelegateConsistent) {
    // FisherF(d1, d2) solves its own bisection in log x on the steered CDF, not inverse_beta_i;
    // this pins that X = (d2/d1)·Y/(1 − Y) with Y the Beta(d1/2, d2/2) quantile stays within
    // budget of the mpmath transform (recorded in the same run as the Beta references).
    struct FRow {
        double d1, d2, p, x;
    };
    constexpr FRow kF[] = {
        {4, 6, 0.9999982705701526, 196.4940053742974499919795},
        {4, 6, 1e-300, 6.123724356957945322220696e-151},
        {4, 6, 0.3, 0.5615267951588166749815364},
        {50, 2000, 6.57e-5, 0.4059881676721305650536456},
        {50, 2000, kUpper, 3.606282782259859429298598},
        {1, 1, 1e-100, 2.467401100272339753364694e-200},
        {10, 0.2, 0.5, 153.649372116769739713977},
        {0.2, 10, kUpper, 23767.24905307853083059087},
    };
    for (const auto& f : kF) {
        const auto d = FDistribution::create(f.d1, f.d2).unwrap();
        const double got = d.getQuantile(f.p);
        ASSERT_TRUE(std::isfinite(got));
        EXPECT_LE(std::fabs(got - f.x) / f.x, 1e-12)
            << "F(" << f.d1 << ", " << f.d2 << ").quantile(" << f.p << ") = " << got << ", want "
            << f.x;
    }
}
