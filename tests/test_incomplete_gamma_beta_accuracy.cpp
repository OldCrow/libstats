// tests/test_incomplete_gamma_beta_accuracy.cpp
//
// CDF accuracy of the distributions built on detail::gamma_p / gamma_q /
// beta_i, at large shape and against mpmath (issue #166). Before v2.4.2 the
// series and continued fractions stopped at DEFAULT_TOLERANCE = 1e-8 and at
// fixed caps of 100 (beta) and 1000 (gamma) iterations, and the gamma
// prefactor exp(−x + a·log x − lgamma(a)) cancelled terms of size a·log x:
// chi-squared k = 1e5 at its median was 8e-6 relative off.
//
// Two-sided: every row must be finite and within budget, scalar and batch.
// References: mpmath at dps 50, evaluated at the double nearest each literal.

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr std::size_t kN = 69;  // 8*8+5: SIMD body and scalar tail on every tier width

template <typename Dist>
void expectCdf(const std::string& name, const Dist& d, double x, double want, double budget) {
    const double got = d.getCumulativeProbability(x);
    ASSERT_TRUE(std::isfinite(got)) << name << " cdf(" << x << ") = " << got;
    EXPECT_LE(std::fabs(got - want) / want, budget)
        << name << " scalar cdf(" << x << ") = " << got << ", want " << want;

    std::vector<double> xs(kN, x), out(kN);
    const detail::PerformanceHint simd{detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED,
                                       std::nullopt};
    d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out), simd);
    for (std::size_t i : {std::size_t{0}, kN - 1})
        EXPECT_LE(std::fabs(out[i] - want) / want, budget)
            << name << " batch cdf(" << x << ")[" << i << "] = " << out[i] << ", want " << want;
}

}  // namespace

// The #166 gate row: P(5e4, x/2) at the median, 8.3e-6 relative off before v2.4.2.
TEST(IncompleteGammaBetaAccuracy, ChiSquaredLargeK) {
    const auto d = ChiSquaredDistribution::create(1e5).unwrap();
    expectCdf("ChiSquared(1e5)", d, 99999.33, 0.49999702574336407182, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, GammaLargeShape) {
    const auto d = GammaDistribution::create(1e4, 1e-3).unwrap();
    expectCdf("Gamma(1e4, 1e-3)", d, 9.95e6, 0.30941788486118332678, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, InverseGammaLargeShape) {
    // CDF = Q(alpha, beta/x): the continued-fraction side.
    const auto d = InverseGammaDistribution::create(1e4, 1e-3).unwrap();
    expectCdf("InverseGamma(1e4, 1e-3)", d, 1e-7, 0.49867019166004216388, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, BetaLargeShape) {
    const auto d = BetaDistribution::create(1e4, 1e4).unwrap();
    expectCdf("Beta(1e4, 1e4)", d, 0.498166, 0.30197493918611573649, 1e-12);
}

TEST(IncompleteGammaBetaAccuracy, FisherFLargeDf) {
    const auto d = FDistribution::create(1e4, 1e4).unwrap();
    expectCdf("F(1e4, 1e4)", d, 0.99749, 0.45000266703024268636, 1e-12);
}

// Q(a, x) for a < ½ and x ≤ a + 1, where 1 − P(a, x) cancelled about log₁₀(1/a) digits: 4e4·ε
// at a = 1e-4. InverseGamma's CDF is Q(α, β/x), the public route to it.
TEST(IncompleteGammaBetaAccuracy, SmallShapeUpperTail) {
    struct Row {
        double a, x, q;
    };
    constexpr Row kRows[] = {
        {1e-4, 0.5, 5.5980292957401715717e-5},   {1e-3, 1.0, 2.1960835758555639629e-4},
        {1e-3, 0.5, 5.6006665647074988868e-4},   {1e-2, 0.9, 2.6263432520511505494e-3},
        {0.3, 1.2, 6.3394719569259104781e-2},    {0.49, 1.45, 8.6164475718001852214e-2},
        {1e-3, 1e-200, 3.686788709141712445e-1},
    };
    constexpr double kBudget = 16 * 0x1p-52;
    for (const Row& r : kRows) {
        const double got = detail::gamma_q(r.a, r.x);
        EXPECT_LE(std::fabs(got - r.q) / r.q, kBudget)
            << "gamma_q(" << r.a << ", " << r.x << ") = " << got << ", want " << r.q;
    }
    const auto d = InverseGammaDistribution::create(1e-3, 1.0).unwrap();
    expectCdf("InverseGamma(1e-3, 1)", d, 2.0, 5.6006665647074988868e-4, kBudget);
}

// Small shapes, where the 1e-8 stop cost 1e-9 to 1e-8.
TEST(IncompleteGammaBetaAccuracy, SmallShapes) {
    const auto g = GammaDistribution::create(2.0, 1.0).unwrap();
    expectCdf("Gamma(2, 1)", g, 1.5, 0.44217459962892542767, 1e-14);
    const auto t = StudentTDistribution::create(5.0).unwrap();
    expectCdf("StudentT(5)", t, 1.3, 0.87484968291466138803, 1e-14);
}

// #226: P and Q at large shape go through Temme's uniform expansion from shape 1e4 near the
// median (|eta| <= 0.6). Unfixed, the series and continued fraction needed ~9*sqrt(a) terms
// there and were cut off by their iteration cap from a ~ 1e10: P(1e12, 1e12) = 0.341,
// P(1e16, 1e16) = 0.004. Rows at z = eta*sqrt(a/2) in {-20, -5, -1, 0, 1, 5, 20}; references
// by quadrature of the density in (x - a)/sqrt(a) with the Gaussian factor taken out (mpmath,
// 60+ digits), cross-checked against the Kummer series and Legendre fraction at 1e4 and 1e6.
// Two-sided: within 8x the accuracy law |ln P|*2^-52 (the worst row uses 0.9 of the law: the
// rounding of the exponent z^2 at z^2 = 680).
TEST(IncompleteGammaBetaAccuracy, LargeShapeTemme) {
    struct Row {
        double a, x, p, q;
    };
    constexpr Row kRows[] = {
        {1e4, 7431.7132610186081, 2.9715457620697385217e-176, 1.0},
        {1e4, 9309.46074626275, 7.8752880197169511462e-13, 0.9999999999992124712},
        {1e4, 9859.2445232723694, 0.07914054663470428154, 0.92085945336529571846},
        {1e4, 10000, 0.50132980833995520038, 0.49867019166004479962},
        {1e4, 10142.088807098011, 0.92183788036008210372, 0.078162119639917896279},
        {1e4, 10723.870735365766, 0.99999999999924941601, 7.5058398781267141638e-13},
        {1e4, 13101.146604240574, 1.0, 2.4603765115548522697e-176},
        {1e6, 971981.76450531359, 2.723581870787843244e-176, 1.0},
        {1e6, 992945.58902461035, 7.7057999542630340417e-13, 0.99999999999922942},
        {1e6, 998586.45302571135, 0.078698541713635244608, 0.92130145828636475539},
        {1e6, 1000000, 0.50013298076087259124, 0.49986701923912740876},
        {1e6, 1001414.8803075923, 0.92139930007105196999, 0.078600699928948030008},
        {1e6, 1007087.7442902045, 0.99999999999923311367, 7.6688632853390038063e-13},
        {1e6, 1028551.5640873392, 1.0, 2.6726439177248717902e-176},
        {1e8, 99717423.891314402, 2.7004814146509295401e-176, 1.0},
        {1e8, 99929305.98756583, 7.6891461280105444995e-13, 0.99999999999923108539},
        {1e8, 99985858.531035081, 0.078654495786906436935, 0.92134550421309356307},
        {1e8, 100000000, 0.50001329807601411987, 0.49998670192398588013},
        {1e8, 100014142.80229825, 0.92135528839057918725, 0.078644711609420812751},
        {1e8, 100070727.34576732, 0.99999999999923145475, 7.6854524692567366917e-13},
        {1e8, 100283109.44197153, 1.0, 2.6953877986504993106e-176},
        {1e12, 999971715995.41858, 2.6979582742747806784e-176, 1.0},
        {1e12, 999992928948.85474, 7.6873174374165035561e-13, 0.99999999999923126826},
        {1e12, 999998585787.10425, 0.07864965243937777489, 0.92135034756062222511},
        {1e12, 1000000000000, 0.50000013298076013381, 0.49999986701923986619},
        {1e12, 1000001414214.229, 0.9213504453904540962, 0.078649554609545903796},
        {1e12, 1000007071084.4785, 0.99999999999923127195, 7.6872805053387371944e-13},
        {1e12, 1000028284537.9148, 1.0, 2.6979073350238455512e-176},
        {1e16, 9999997171573142, 2.6979331212487486661e-176, 1.0},
        {1e16, 9999999292893236, 7.6872994448625559167e-13, 0.99999999999923127006},
        {1e16, 9999999858578644, 0.078649603384215433294, 0.92135039661578456671},
        {1e16, 10000000000000000, 0.50000000132980760134, 0.49999999867019239866},
        {1e16, 10000000141421356, 0.9213503956373668155, 0.078649604362633184502},
        {1e16, 10000000707106798, 0.99999999999923127013, 7.6872987061308036841e-13},
        {1e16, 10000002828427392, 1.0, 2.6979321025260599021e-176},
        {1e20, 9.999999971715729e+19, 2.6979495558420096755e-176, 1.0},
        {1e20, 9.9999999929289327e+19, 7.6873249810148231808e-13, 0.9999999999992312675},
        {1e20, 9.9999999985857872e+19, 0.078649713529652462554, 0.92135028647034753745},
        {1e20, 1e+20, 0.50000000001329807601, 0.49999999998670192399},
        {1e20, 1.0000000001414213e+20, 0.92135028646056337027, 0.078649713539436629727},
        {1e20, 1.0000000007071067e+20, 0.99999999999923126748, 7.6873251620046759608e-13},
        {1e20, 1.0000000028284271e+20, 1.0, 2.6979536256679663293e-176},
    };
    constexpr double kEps = 0x1p-52;
    const auto law = [](double v) { return 8.0 * kEps * std::max(1.0, std::fabs(std::log(v))); };
    for (const Row& r : kRows) {
        const double p = detail::gamma_p(r.a, r.x);
        const double q = detail::gamma_q(r.a, r.x);
        EXPECT_LE(std::fabs(p - r.p) / r.p, law(r.p)) << "P(" << r.a << ", " << r.x << ") = " << p;
        EXPECT_LE(std::fabs(q - r.q) / r.q, law(r.q)) << "Q(" << r.a << ", " << r.x << ") = " << q;
    }
    // Through the distribution, scalar and batch, at the median rows.
    expectCdf("Gamma(1e12)", GammaDistribution::create(1e12, 1.0).unwrap(), 1e12,
              0.50000013298076013381, 8 * kEps);
    expectCdf("Gamma(1e16)", GammaDistribution::create(1e16, 1.0).unwrap(), 1e16,
              0.50000000132980760134, 8 * kEps);
}

// The hunt-3 defects in the incomplete beta and Fisher-F (2026-10-10). Each row failed before the
// fix; references from mpmath at dps 60 at the double nearest each literal (closed forms where
// one shape is 1 or an integer, the gamma limit where d₂ or d₁ is at the top of the range).
namespace {
constexpr double kEpsilon = 0x1p-52;

void expectRel(const std::string& what, double got, double want, double budget) {
    ASSERT_TRUE(std::isfinite(got)) << what << " = " << got;
    EXPECT_LE(std::fabs(got - want) / want, budget) << what << " = " << got << ", want " << want;
}
}  // namespace

// F1: the direct-form prefactor took log(1 − x) from a rounded x, b·ε/4 relative, and past
// b·x = O(1) with x < ε a CDF outside [0, 1]. The quantile solver held the exact logs, so I(Q(p))
// missed p. Binomial's 1 − p beside an exact p, and Fisher-F's y beside ybar, carried the same.
TEST(IncompleteGammaBetaAccuracy, BetaLogComplementPrefactor) {
    expectCdf("Beta(1, 1e6)", BetaDistribution::create(1.0, 1e6).unwrap(), 1.5e-6,
              0.77687009087311017, 16 * kEpsilon);  // was 1.2e-11 off
    expectCdf("Beta(1, 4.19e46)", BetaDistribution::create(1.0, 4.19e46).unwrap(), 3.65e-47,
              0.78332353874716522, 256 * kEpsilon);     // was 3.6
    EXPECT_EQ(detail::beta_i(1e-18, 1.0, 1e200), 1.0);  // was NaN
    const auto beta = BetaDistribution::create(1.0, 1e6).unwrap();
    const double q = beta.getQuantile(0.3);
    EXPECT_LE(std::fabs(beta.getCumulativeProbability(q) - 0.3), 4 * kEpsilon * 0.3)
        << "Beta(1, 1e6): I(Q(0.3)) = " << beta.getCumulativeProbability(q);  // was 9.8e-12 off
    expectCdf("Binomial(1e9, 1e-7)", BinomialDistribution::create(1000000000, 1e-7).unwrap(), 100.0,
              0.52656219852999868, 64 * kEpsilon);  // was 4.2e-9 off
    expectCdf("Binomial(2e9, 3e-9)", BinomialDistribution::create(2000000000, 3e-9).unwrap(), 5.0,
              0.44567964112367654, 256 * kEpsilon);  // was 1.9e-8 off
    expectCdf("F(1, 1e18)", FDistribution::create(1.0, 1e18).unwrap(), 2.0, 0.84270079294971487,
              16 * kEpsilon);  // was 2.29
    expectCdf("F(2, 1e300)", FDistribution::create(2.0, 1e300).unwrap(), 2.0, 0.86466471676338731,
              1e-12);  // was 0
}

// F8: the extreme-argument path's leading term assumed z·(a + b) ≪ 1; at d₂ ~ 1e308 it is the
// gamma limit, and lbeta of two shapes at DBL_MAX was NaN.
TEST(IncompleteGammaBetaAccuracy, FisherFExtremeShapes) {
    const auto f = FDistribution::create(2.0, 1.6e308).unwrap();
    expectCdf("F(2, 1.6e308)", f, 1.5, 0.77686983985157017, 16 * kEpsilon);  // was 1.4999999999998
    expectRel("F(2, 1.6e308).sf(1.5)", f.getSurvivalProbability(1.5), 0.22313016014842983,
              16 * kEpsilon);  // was -0.5
    expectCdf("F(2, 1e308)", FDistribution::create(2.0, 1e308).unwrap(), 1.5, 0.77686983985157017,
              1e-13);  // was NaN
    const double big = std::numeric_limits<double>::max();
    const double tiny = std::numeric_limits<double>::denorm_min();
    const auto g = FDistribution::create(big, big).unwrap();
    EXPECT_EQ(g.getCumulativeProbability(tiny), 0.0);  // all three were NaN
    EXPECT_EQ(g.getCumulativeProbability(big), 1.0);
    EXPECT_EQ(g.getSurvivalProbability(big), 0.0);
    EXPECT_EQ(g.getSurvivalProbability(tiny), 1.0);
}

// F5: the CDF and survival chose by the branch point and complemented above it, losing the small
// side where a shape is below 1 (or below 1e-6 in the extreme path), and decided the branch on a y
// rounded to 1.
TEST(IncompleteGammaBetaAccuracy, FisherFSmallSide) {
    expectCdf("F(2, 3.2e-15)", FDistribution::create(2.0, 3.2e-15).unwrap(), 4.62e-11,
              1.6433225726814347e-14, 16 * kEpsilon);  // was 1.6320e-14
    expectCdf("F(2, 6.2e-262)", FDistribution::create(2.0, 6.2e-262).unwrap(), 2.13e-260,
              1.3157451371267393e-261, 16 * kEpsilon);  // was 0
    expectRel("F(4.03e-152, 2).sf(2.63e124)",
              FDistribution::create(4.03e-152, 2.0).unwrap().getSurvivalProbability(2.63e124),
              1.265516308179236e-150, 16 * kEpsilon);  // was 0
    expectRel("F(9.99e-48, 2).sf(3.39e-275)",
              FDistribution::create(9.9907101850319212e-48, 2.0)
                  .unwrap()
                  .getSurvivalProbability(3.3877725165904521e-275),
              3.7010900783002e-45, 16 * kEpsilon);  // was 0
    expectCdf("F(9.09e220, 2)", FDistribution::create(9.0896886444491038e+220, 2.0).unwrap(),
              0.060285167009089161, 6.2516937918432809e-8, 1e-12);  // was 0
    const auto f = FDistribution::create(0.001, 1e300).unwrap();
    EXPECT_LE(std::fabs(f.getCumulativeProbability(1.0) - 0.99649386209850102557), 4 * kEpsilon)
        << "F(0.001, 1e300).cdf(1) = " << f.getCumulativeProbability(1.0);  // was 9.5e-15 off
}

// F2: the Stirling prefactor past (a + b)²·ε > min(a, b) lost x − x₀ to the rounding of
// x₀ = a/(a + b) next to 1, and the continued fraction's Lentz guard replaced its O(1/a) terms
// from a ~ 1e30; the switch to that form jumped ~5e6·ε. Integer second shape: closed forms.
TEST(IncompleteGammaBetaAccuracy, BetaShapeRatioPrefactor) {
    struct Row {
        const char* name;
        double x, a, want;
    };
    constexpr Row kRows[] = {
        {"beta_i(1 - 2.6e-10, 1e11, 20)", 0.99999999973999998, 1e11,
         0.096820780352030053118},  // was 1.7e9·ε off
        {"beta_i(1 - 8.7e-13, 3e13, 20)", 0.99999999999913336, 3e13,
         0.0968465159838704448},  // was 6.9e11·ε off
    };
    for (const Row& r : kRows)
        expectRel(r.name, detail::beta_i(r.x, r.a, 20.0), r.want, 16 * kEpsilon);
    const auto f = FDistribution::create(40.0, 2e200).unwrap();
    expectRel("F(40, 2e200).sf(1.2)", f.getSurvivalProbability(1.2), 0.18026051417019201,
              64 * kEpsilon);  // was 1
    expectCdf("F(40, 2e200)", f, 0.8, 0.18775147166316223, 64 * kEpsilon);
    expectRel("F(50, 1e20).Q(0.5)", FDistribution::create(50.0, 1e20).unwrap().getQuantile(0.5),
              0.98669873467953706, 1e-13);  // was 163.84
}

// IEEE edges: an infinite shape reached the beta quantile solver (4.9e-324), and Student's t at
// ν = 4.9e-324 halved ν to 0 and returned NaN; its CDF tends to ½ at every finite t as ν → 0.
TEST(IncompleteGammaBetaAccuracy, BetaAndStudentTEdges) {
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isnan(detail::inverse_beta_i(0.5, inf, 2.0)));
    EXPECT_TRUE(std::isnan(detail::inverse_beta_i(0.5, 2.0, inf)));
    EXPECT_TRUE(std::isfinite(detail::inverse_beta_i(0.5, 3.0, 2.0)));
    const double nu = std::numeric_limits<double>::denorm_min();
    for (double t : {-1e300, -1.0, 1.0, 1e300})
        EXPECT_EQ(detail::t_cdf(t, nu), 0.5) << "t_cdf(" << t << ", 4.9e-324)";
    const auto st = StudentTDistribution::create(nu).unwrap();
    EXPECT_EQ(st.getCumulativeProbability(-1.0), 0.5);
    EXPECT_EQ(st.getCumulativeProbability(1.0), 0.5);
    EXPECT_EQ(detail::inverse_t_cdf(0.25, nu), -inf);
}
