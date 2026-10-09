// Contract gates from the v2.5.0 architecture review (AR) and defect hunt (DH), 2026-10-07.
//
// Each gate was run against the unfixed code first and seen to fail there; the comment above
// each TEST names the defect and the failure it caught. Correctness only: no timing assertion,
// so this binary carries no "timing" label and runs in CI's -LE "timing|benchmark" pass.
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <math.h>  // signgam (POSIX)
#include <numbers>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace stats;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kPi = std::numbers::pi;

double relErr(double got, double want) {
    return std::fabs(got - want) / std::fabs(want);
}

std::uint64_t bitsOf(double v) {
    std::uint64_t u;
    std::memcpy(&u, &v, sizeof u);
    return u;
}

}  // namespace

//==============================================================================
// F1 (AR D1/D8): copy- and move-assignment must not leave the lock-free atomic
// parameter copy stale. Unfixed: Exponential/Gaussian/Poisson stale after both,
// Uniform/Discrete after move-assign (and every moved-from source).
//==============================================================================

namespace {
template <class D, class G>
void checkAssignRefreshesAtomics(D a, D b, G atomicGet, G lockedGet) {
    // Warm both objects so atomicParamsValid_ is true before the assignment.
    (void)a.getProbability(0.5);
    (void)a.getVariance();
    (void)b.getProbability(0.5);
    (void)b.getVariance();

    D c = a;
    (void)c.getProbability(0.5);
    (void)c.getVariance();
    c = b;
    EXPECT_EQ(atomicGet(c), lockedGet(b)) << "copy-assign left the atomic parameter stale";

    D m = a;
    (void)m.getProbability(0.5);
    (void)m.getVariance();
    D src = b;
    (void)src.getProbability(0.5);
    (void)src.getVariance();
    m = std::move(src);
    EXPECT_EQ(atomicGet(m), lockedGet(b)) << "move-assign left the atomic parameter stale";
    // The moved-from source was reset: its atomic getter must agree with its locked one.
    EXPECT_EQ(atomicGet(src), lockedGet(src)) << "moved-from source kept a stale atomic";
}
}  // namespace

TEST(ContractGates, F1_AssignmentRefreshesAtomicParameters) {
    using E = ExponentialDistribution;
    checkAssignRefreshesAtomics<E>(
        E(2.0), E(5.0), +[](const E& d) { return d.getLambdaAtomic(); },
        +[](const E& d) { return d.getLambda(); });
    using G = GaussianDistribution;
    checkAssignRefreshesAtomics<G>(
        G(1.0, 2.0), G(7.0, 3.0), +[](const G& d) { return d.getMeanAtomic(); },
        +[](const G& d) { return d.getMean(); });
    checkAssignRefreshesAtomics<G>(
        G(1.0, 2.0), G(7.0, 3.0), +[](const G& d) { return d.getStandardDeviationAtomic(); },
        +[](const G& d) { return d.getStandardDeviation(); });
    using P = PoissonDistribution;
    checkAssignRefreshesAtomics<P>(
        P(2.0), P(9.0), +[](const P& d) { return d.getLambdaAtomic(); },
        +[](const P& d) { return d.getLambda(); });
    using U = UniformDistribution;
    checkAssignRefreshesAtomics<U>(
        U(0.0, 1.0), U(3.0, 9.0), +[](const U& d) { return d.getUpperBoundAtomic(); },
        +[](const U& d) { return d.getUpperBound(); });
    checkAssignRefreshesAtomics<U>(
        U(0.0, 1.0), U(3.0, 9.0), +[](const U& d) { return d.getLowerBoundAtomic(); },
        +[](const U& d) { return d.getLowerBound(); });
    using Di = DiscreteDistribution;
    checkAssignRefreshesAtomics<Di>(
        Di(0, 3), Di(5, 9),
        +[](const Di& d) { return static_cast<double>(d.getUpperBoundAtomic()); },
        +[](const Di& d) { return static_cast<double>(d.getUpperBound()); });
    checkAssignRefreshesAtomics<Di>(
        Di(0, 3), Di(5, 9),
        +[](const Di& d) { return static_cast<double>(d.getLowerBoundAtomic()); },
        +[](const Di& d) { return static_cast<double>(d.getLowerBound()); });
}

//==============================================================================
// F2 (AR D2): TruncatedNormal operator>> found "a=" inside "sigma=".
// Unfixed: "(mu=0, sigma=1, a=-1, b=2)" read back with a = 1, no failbit.
//==============================================================================

TEST(ContractGates, F2_TruncatedNormalStreamRoundTripAndFailbit) {
    for (const double a : {-1.0, 0.0, -2.5, 10.0, -std::numeric_limits<double>::infinity()}) {
        TruncatedNormalDistribution d(0.25, 1.5, a, 12.0);
        std::ostringstream os;
        os << d;
        TruncatedNormalDistribution e;
        std::istringstream is(os.str());
        is >> e;
        ASSERT_FALSE(is.fail()) << os.str();
        EXPECT_EQ(e.getMu(), d.getMu()) << os.str();
        EXPECT_EQ(e.getSigma(), d.getSigma()) << os.str();
        EXPECT_EQ(e.getLowerBound(), d.getLowerBound()) << os.str();
        EXPECT_EQ(e.getUpperBound(), d.getUpperBound()) << os.str();
    }
    // Malformed: each must set failbit and leave the object unchanged.
    for (const char* bad : {
             "TruncatedNormalDistribution(mu=0, sigma=1, b=2)",        // no a=; sigma= holds one
             "TruncatedNormalDistribution(mu=0, sigma=1, a=1x, b=2)",  // trailing garbage
             "TruncatedNormalDistribution(mu=0, sigma=1, a=-1, b=2",   // unterminated
             "TruncatedNormalDistribution(mu=0, sigma=1, a=-1, a=0, b=2)",  // duplicate key
             "TruncatedNormalDistribution(mu=0, sigma=1, c=-1, b=2)",       // unknown key
             "TruncatedNormalDistribution(mu=0, sigma=1, a=3, b=2)",        // invalid window
         }) {
        TruncatedNormalDistribution e(0.0, 1.0, -2.0, 2.0);
        std::istringstream is(bad);
        is >> e;
        EXPECT_TRUE(is.fail()) << bad;
        EXPECT_EQ(e.getLowerBound(), -2.0) << bad;
        EXPECT_EQ(e.getUpperBound(), 2.0) << bad;
    }
}

//==============================================================================
// F3 (AR D3): getQuantile(NaN) is NaN in all 27, and never throws. Unfixed: 7
// returned a finite value (Beta 1, Bernoulli 1, Binomial n, Discrete b, Poisson
// INT_MAX, Geometric/NegBin 2^53) and 4 threw. Two-sided: Q(0.5) stays a number.
//==============================================================================

namespace {
template <class D>
void checkNaNQuantile(const char* name, const D& d) {
    double q = 0.0;
    EXPECT_NO_THROW(q = d.getQuantile(kNaN)) << name;
    EXPECT_TRUE(std::isnan(q)) << name << " getQuantile(NaN) = " << q;
    EXPECT_FALSE(std::isnan(d.getQuantile(0.5))) << name;
    EXPECT_THROW((void)d.getQuantile(1.5), std::invalid_argument) << name;
}
}  // namespace

TEST(ContractGates, F3_QuantileOfNaNIsNaN_All27) {
    checkNaNQuantile("Bernoulli", BernoulliDistribution(0.3));
    checkNaNQuantile("Beta", BetaDistribution(2.0, 3.0));
    checkNaNQuantile("Binomial", BinomialDistribution(10, 0.3));
    checkNaNQuantile("Cauchy", CauchyDistribution(0.5, 2.0));
    checkNaNQuantile("ChiSquared", ChiSquaredDistribution(3.0));
    checkNaNQuantile("Discrete", DiscreteDistribution(0, 6));
    checkNaNQuantile("Erlang", ErlangDistribution(3, 2.0));
    checkNaNQuantile("Exponential", ExponentialDistribution(2.0));
    checkNaNQuantile("FisherF", FDistribution(3.0, 7.0));
    checkNaNQuantile("Gamma", GammaDistribution(2.0, 3.0));
    checkNaNQuantile("Gaussian", GaussianDistribution(1.0, 2.0));
    checkNaNQuantile("Geometric", GeometricDistribution(0.3));
    checkNaNQuantile("Gumbel", GumbelDistribution(0.5, 2.0));
    checkNaNQuantile("HalfNormal", HalfNormalDistribution(2.0));
    checkNaNQuantile("InverseGamma", InverseGammaDistribution(2.0, 3.0));
    checkNaNQuantile("Laplace", LaplaceDistribution(0.5, 2.0));
    checkNaNQuantile("Logistic", LogisticDistribution(0.5, 2.0));
    checkNaNQuantile("LogNormal", LogNormalDistribution(0.5, 1.5));
    checkNaNQuantile("NegBinomial", NegativeBinomialDistribution(3.0, 0.4));
    checkNaNQuantile("Pareto", ParetoDistribution(3.0, 1.5));
    checkNaNQuantile("Poisson", PoissonDistribution(3.5));
    checkNaNQuantile("Rayleigh", RayleighDistribution(2.0));
    checkNaNQuantile("StudentT", StudentTDistribution(5.0));
    checkNaNQuantile("TruncNormal", TruncatedNormalDistribution(0.0, 1.0, -1.0, 2.0));
    checkNaNQuantile("Uniform", UniformDistribution(-1.0, 3.0));
    checkNaNQuantile("VonMises", VonMisesDistribution(0.5, 2.0));
    checkNaNQuantile("Weibull", WeibullDistribution(2.0, 3.0));
}

//==============================================================================
// F4 (AR D4): Binomial fit throws on non-finite, negative or out-of-range data,
// as the other 26 do. Unfixed: {1, 2, NaN, 3} silently fit n = 3, p = 2/3.
//==============================================================================

TEST(ContractGates, F4_BinomialFitRejectsInvalidData) {
    const double inf = std::numeric_limits<double>::infinity();
    for (const auto& data :
         {std::vector<double>{1.0, 2.0, kNaN, 3.0}, std::vector<double>{1.0, 2.0, inf},
          std::vector<double>{1.0, -1.0, 3.0}, std::vector<double>{1.0, 3e9, 2.0}}) {
        BinomialDistribution d(10, 0.3);
        EXPECT_THROW(d.fit(data), std::invalid_argument);
        EXPECT_EQ(d.getN(), 10);  // unchanged
    }
    BinomialDistribution ok(10, 0.3);
    EXPECT_NO_THROW(ok.fit(std::vector<double>{1.0, 2.0, 3.0, 2.0, 4.0}));
}

//==============================================================================
// F5 (AR D5): Cauchy batch pdf/logpdf check sizes before the empty-input return.
// Unfixed: empty values with a non-empty results span returned silently.
//==============================================================================

TEST(ContractGates, F5_CauchyBatchSizeCheckPrecedesEmptyReturn) {
    CauchyDistribution c(0.5, 2.0);
    std::vector<double> in, out(3);
    EXPECT_THROW(c.getProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    EXPECT_THROW(c.getLogProbability(std::span<const double>(in), std::span<double>(out)),
                 std::invalid_argument);
    std::vector<double> none;
    EXPECT_NO_THROW(c.getProbability(std::span<const double>(in), std::span<double>(none)));
}

//==============================================================================
// F6 (AR A11): getSurvival is virtual, with an accurate complement where the
// family has one, so the hazard stays finite inside the support. Unfixed:
// Exponential(1) S(40) = 0 and hazard +inf (true 1); Gaussian S(9) = 0.
//==============================================================================

TEST(ContractGates, F6_SurvivalAndHazardInUpperTail) {
    const ExponentialDistribution e(1.0);
    const DistributionBase& eb = e;
    EXPECT_LT(relErr(eb.getSurvival(40.0), 4.248354255291589e-18), 1e-14);
    EXPECT_LT(relErr(eb.getHazard(40.0), 1.0), 1e-14);

    const GaussianDistribution g(0.0, 1.0);
    const DistributionBase& gb = g;
    EXPECT_LT(relErr(gb.getSurvival(9.0), 1.1285884059538406e-19), 1e-12);
    EXPECT_TRUE(std::isfinite(gb.getHazard(9.0)));
    EXPECT_GT(gb.getHazard(9.0), 9.0);  // Mills ratio: h(x) > x

    const WeibullDistribution w(2.0, 1.0);  // S = exp(-x^2), h = 2x
    const DistributionBase& wb = w;
    EXPECT_LT(relErr(wb.getSurvival(6.0), 2.3195228302435696e-16), 1e-13);
    EXPECT_LT(relErr(wb.getHazard(6.0), 12.0), 1e-13);

    const RayleighDistribution r(1.0);  // S = exp(-x^2/2), h = x
    const DistributionBase& rb = r;
    EXPECT_LT(relErr(rb.getSurvival(9.0), 2.576757109154981e-18), 1e-13);
    EXPECT_LT(relErr(rb.getHazard(9.0), 9.0), 1e-13);

    const ParetoDistribution p(1.0, 2.0);  // S = x^-2, h = 2/x
    const DistributionBase& pb = p;
    EXPECT_LT(relErr(pb.getSurvival(1e9), 1e-18), 1e-14);
    EXPECT_LT(relErr(pb.getHazard(1e9), 2e-9), 1e-14);

    // The families that already computed the complement now expose it.
    const InverseGammaDistribution ig(3.0, 2.0);
    EXPECT_EQ(static_cast<const DistributionBase&>(ig).getSurvival(1e4),
              ig.getSurvivalProbability(1e4));
    EXPECT_GT(ig.getSurvivalProbability(1e4), 0.0);
    const FDistribution f(5.0, 7.0);
    EXPECT_EQ(static_cast<const DistributionBase&>(f).getSurvival(1e6),
              f.getSurvivalProbability(1e6));
    EXPECT_GT(f.getSurvivalProbability(1e6), 0.0);

    // Edges: S(-inf) = 1, S(+inf) = 0, S(NaN) = NaN, below the support S = 1.
    for (const DistributionBase* d : {&eb, &gb, &wb, &rb, &pb}) {
        EXPECT_EQ(d->getSurvival(-std::numeric_limits<double>::infinity()), 1.0);
        EXPECT_EQ(d->getSurvival(std::numeric_limits<double>::infinity()), 0.0);
        EXPECT_TRUE(std::isnan(d->getSurvival(kNaN)));
    }
    EXPECT_EQ(eb.getSurvival(-1.0), 1.0);
    EXPECT_EQ(wb.getSurvival(-1.0), 1.0);
    EXPECT_EQ(rb.getSurvival(-1.0), 1.0);
    EXPECT_EQ(pb.getSurvival(0.5), 1.0);
}

//==============================================================================
// F8 (AR D6): von Mises takes the uniform shortcut at kappa == 0 only. Unfixed:
// kappa < 1e-10 was treated as uniform, so F(pi/2) = 0.75 exactly at kappa = 1e-11
// (true 0.75 + kappa/(2 pi)) and the circular variance was 1 (true 1 - kappa/2).
// The kappa -> 0 paths must stay accurate down to the smallest kappa.
//==============================================================================

TEST(ContractGates, F8_VonMisesTinyKappaIsNotUniform) {
    for (const double kappa : {1e-9, 1e-11, 1e-13, 1e-20, 1e-300}) {
        const VonMisesDistribution v(0.0, kappa);
        const double shift = kappa / (2.0 * kPi);  // first Fourier term; O(kappa^2) beyond
        EXPECT_NEAR(v.getCumulativeProbability(kPi / 2), 0.75 + shift, 2.3e-16) << kappa;
        EXPECT_NEAR(v.getCumulativeProbability(-kPi / 2), 0.25 - shift, 1.2e-16) << kappa;
        EXPECT_NEAR(v.getCumulativeProbability(0.0), 0.5, 1.2e-16) << kappa;
        EXPECT_NEAR(v.getProbability(0.0), (1.0 + kappa) / (2.0 * kPi), 1e-16) << kappa;
        EXPECT_NEAR(v.getCircularVariance(), 1.0 - kappa / 2.0, 2.3e-16) << kappa;
        EXPECT_NEAR(v.getQuantile(0.75 + shift), kPi / 2, 1e-12) << kappa;
    }
    // Two-sided at kappa = 1e-11: the shift is ~7000 ulps of 0.75 and must be there.
    const VonMisesDistribution v(0.0, 1e-11);
    EXPECT_NE(v.getCumulativeProbability(kPi / 2), 0.75);
    EXPECT_NE(v.getCircularVariance(), 1.0);
    // kappa == 0 exactly is still uniform.
    const VonMisesDistribution u(0.0, 0.0);
    EXPECT_EQ(u.getCumulativeProbability(kPi / 2), 0.75);
    EXPECT_EQ(u.getCircularVariance(), 1.0);
}

//==============================================================================
// F13 (DH D4): Geometric CDF by the closed form -expm1((k+1) log1p(-p)). Unfixed:
// rel 2.1e-7 at p = 1e-12 (incomplete beta), non-monotone, Q(F(k)) = k + 111033.
// References: 1 - (1 - p)^(k+1) in 50-digit decimal, p the double 1e-12.
//==============================================================================

TEST(ContractGates, F13_GeometricCdfClosedForm) {
    const GeometricDistribution g(1e-12);
    EXPECT_LT(relErr(g.getCumulativeProbability(6700641196083.0), 0.9987698771002426), 4e-16);
    EXPECT_LT(relErr(g.getCumulativeProbability(1.3e12), 0.7274682069664371), 4e-16);
    EXPECT_LT(relErr(g.getCumulativeProbability(1e11), 0.0951625819649905), 4e-16);
    const double k = 6700641196083.0;
    EXPECT_LE(g.getCumulativeProbability(k), g.getCumulativeProbability(k + 111032.0));
    EXPECT_EQ(g.getQuantile(g.getCumulativeProbability(k)), k);
    // NegativeBinomial(1, p) is the same distribution and takes the same path.
    const NegativeBinomialDistribution nb(1.0, 1e-12);
    EXPECT_EQ(nb.getCumulativeProbability(k), g.getCumulativeProbability(k));
    // Batch agrees with scalar bit for bit.
    std::vector<double> xs{0.0, 1.0, 1e11, 1.3e12, k}, out(xs.size());
    g.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out));
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_EQ(bitsOf(out[i]), bitsOf(g.getCumulativeProbability(xs[i]))) << xs[i];
    // Small k, ordinary p: exact against the direct product.
    const GeometricDistribution h(0.3);
    EXPECT_LT(relErr(h.getCumulativeProbability(2.0), 1.0 - 0.7 * 0.7 * 0.7), 4e-16);
}

//==============================================================================
// F14 (DH D6): Rayleigh and HalfNormal reject sigma outside the range where
// sigma^2 is finite and normal, as Gaussian rejects sigma > MAX_STANDARD_DEVIATION.
// Unfixed: Rayleigh(1e-155) gave cdf(sigma) = 1 scalar, NaN batch.
//==============================================================================

TEST(ContractGates, F14_ScaleRangeRayleighHalfNormal) {
    for (const double s : {1e-155, 1e-200, 1e-300, 1e155, 1e300}) {
        EXPECT_THROW(RayleighDistribution{s}, std::invalid_argument) << s;
        EXPECT_THROW(HalfNormalDistribution{s}, std::invalid_argument) << s;
        EXPECT_FALSE(RayleighDistribution::create(s).isOk()) << s;
        EXPECT_FALSE(HalfNormalDistribution::create(s).isOk()) << s;
    }
    // The bound is Gaussian's: MAX_STANDARD_DEVIATION accepted, just above it rejected.
    const double hi = detail::MAX_STANDARD_DEVIATION;
    EXPECT_NO_THROW(RayleighDistribution{hi});
    EXPECT_NO_THROW(HalfNormalDistribution{hi});
    EXPECT_THROW(RayleighDistribution{std::nextafter(hi, 2 * hi)}, std::invalid_argument);
    EXPECT_THROW(HalfNormalDistribution{std::nextafter(hi, 2 * hi)}, std::invalid_argument);
    RayleighDistribution setr(1.0);
    EXPECT_THROW(setr.setSigma(1e-160), std::invalid_argument);
    EXPECT_EQ(setr.getSigma(), 1.0);
    // Inside the range, at both ends, scalar and batch are accurate and agree.
    for (const double s : {2e-154, 1e-100, 1.0, 1e5, hi}) {
        const RayleighDistribution r(s);
        const HalfNormalDistribution h(s);
        std::vector<double> xs(16, s), rc(16), rp(16), hp(16), hc(16);
        r.getCumulativeProbability(std::span<const double>(xs), std::span<double>(rc));
        r.getProbability(std::span<const double>(xs), std::span<double>(rp));
        h.getProbability(std::span<const double>(xs), std::span<double>(hp));
        h.getCumulativeProbability(std::span<const double>(xs), std::span<double>(hc));
        EXPECT_LT(relErr(r.getCumulativeProbability(s), -std::expm1(-0.5)), 1e-14) << s;
        EXPECT_LT(relErr(rc[0], -std::expm1(-0.5)), 1e-13) << s;
        EXPECT_LT(relErr(r.getProbability(s) * s, std::exp(-0.5)), 1e-13) << s;  // log-space pdf
        EXPECT_LT(relErr(rp[0] * s, std::exp(-0.5)), 1e-13) << s;
        const double hpdf = std::sqrt(2.0 / kPi) * std::exp(-0.5);  // times sigma
        EXPECT_LT(relErr(h.getProbability(s) * s, hpdf), 1e-13) << s;
        EXPECT_LT(relErr(hp[0] * s, hpdf), 1e-13) << s;
        EXPECT_LT(relErr(hc[0], std::erf(1.0 / std::sqrt(2.0))), 1e-13) << s;
    }
}

//==============================================================================
// F15 (DH D7): Discrete quantile is min{k : F(k) >= p} exactly, ties included.
// Unfixed: Discrete(0, 9).getQuantile(F(0)) = 1, Discrete(-5, 5) Q(F(-5)) = -4.
//==============================================================================

TEST(ContractGates, F15_DiscreteQuantileTies) {
    for (const auto& [a, b] : std::vector<std::pair<int, int>>{
             {0, 9}, {-5, 5}, {0, 6}, {1, 100}, {-1000, 2000}, {0, 1}, {3, 3}}) {
        const DiscreteDistribution d(a, b);
        for (int k = a; k <= b; ++k) {
            const double f = d.getCumulativeProbability(static_cast<double>(k));
            EXPECT_EQ(d.getQuantile(f), static_cast<double>(k)) << a << ".." << b << " k=" << k;
            if (k < b) {
                EXPECT_EQ(d.getQuantile(std::nextafter(f, 2.0)), static_cast<double>(k + 1))
                    << a << ".." << b << " k=" << k;
            }
            if (k > a) {
                const double below = std::nextafter(f, -1.0);
                const double qb = d.getQuantile(below);
                EXPECT_GE(d.getCumulativeProbability(qb), below);
                EXPECT_LT(d.getCumulativeProbability(qb - 1.0), below);
            }
        }
        EXPECT_EQ(d.getQuantile(0.0), static_cast<double>(a));
        EXPECT_EQ(d.getQuantile(1.0), static_cast<double>(b));
    }
}

//==============================================================================
// F16 (DH D9): TruncatedNormal never has a subnormal normaliser Z, so the CDF is
// never NaN where the pdf is finite. Unfixed: TN(0, 1, -40, -38).cdf(-39) = NaN.
//==============================================================================

TEST(ContractGates, F16_TruncatedNormalNoSubnormalZ) {
    int accepted = 0;
    for (double lower = 35.0; lower <= 40.0; lower += 0.01) {
        try {
            const TruncatedNormalDistribution t(0.0, 1.0, -lower - 2.0, -lower);
            ++accepted;
            EXPECT_GE(t.getNormalizationConstant(), std::numeric_limits<double>::min()) << lower;
            for (const double x : {-lower - 1.5, -lower - 1.0, -lower - 0.01}) {
                const double c = t.getCumulativeProbability(x);
                EXPECT_TRUE(std::isfinite(c) && c >= 0.0 && c <= 1.0) << lower << " x=" << x;
                EXPECT_TRUE(std::isfinite(t.getProbability(x))) << lower << " x=" << x;
            }
        } catch (const std::invalid_argument&) {
        }
    }
    EXPECT_GT(accepted, 100);  // windows nearer than ~37.5 sd still construct
    EXPECT_THROW(TruncatedNormalDistribution(0.0, 1.0, -40.0, -38.0), std::invalid_argument);
}

//==============================================================================
// F17 (DH D10): NegativeBinomial/Geometric cdf(x) = 1 for huge finite x, scalar
// and batch. Unfixed: cdf(1e300) and cdf(DBL_MAX) were NaN.
//==============================================================================

TEST(ContractGates, F17_NegBinCdfAtHugeCount) {
    const double big[] = {1e17, 1e300, std::numeric_limits<double>::max()};
    for (const auto& nb :
         {NegativeBinomialDistribution(5.0, 0.5), NegativeBinomialDistribution(0.3, 1e-6),
          NegativeBinomialDistribution(1e5, 0.9), NegativeBinomialDistribution(1.0, 0.25)}) {
        for (const double x : big)
            EXPECT_EQ(nb.getCumulativeProbability(x), 1.0) << nb.toString() << " x=" << x;
        std::vector<double> xs(64, 1e300), out(64);
        xs[7] = std::numeric_limits<double>::max();
        nb.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out));
        for (const double v : out)
            EXPECT_EQ(v, 1.0) << nb.toString();
    }
    const GeometricDistribution g(0.3);
    for (const double x : big)
        EXPECT_EQ(g.getCumulativeProbability(x), 1.0) << x;
}

//==============================================================================
// F18 (DH D13): Gamma fit throws on one point or zero-variance data, as the
// majority does for insufficient data. Unfixed: alpha = beta = NaN, no throw.
//==============================================================================

TEST(ContractGates, F18_GammaFitDegenerateData) {
    for (const auto& data :
         {std::vector<double>{2.5}, std::vector<double>{1.5, 1.5}, std::vector<double>(30, 1.0)}) {
        GammaDistribution d(2.0, 1.0);
        EXPECT_THROW(d.fit(data), std::invalid_argument) << data.size();
        EXPECT_EQ(d.getAlpha(), 2.0);
        EXPECT_EQ(d.getBeta(), 1.0);
    }
    GammaDistribution ok(2.0, 1.0);
    EXPECT_NO_THROW(ok.fit(std::vector<double>{1.0, 2.0}));
    EXPECT_TRUE(std::isfinite(ok.getAlpha()) && ok.getAlpha() > 0.0);
}

//==============================================================================
// F19 (DH D14, AR D11): operator<< writes max_digits10 so operator>> restores
// every parameter bit for bit, all 27. Unfixed: 6 fixed decimals or 6
// significant digits, so 1/3 and pi*1e-7 did not round-trip (8 printed 0.000000).
//==============================================================================

namespace {
template <class D, class... Getters>
void checkStreamRoundTrip(const char* name, const D& d, Getters... getters) {
    std::ostringstream os;
    os << d;
    D e;
    std::istringstream is(os.str());
    is >> e;
    ASSERT_FALSE(is.fail()) << name << ": " << os.str();
    int i = 0;
    (
        [&] {
            EXPECT_EQ(bitsOf(static_cast<double>(getters(e))),
                      bitsOf(static_cast<double>(getters(d))))
                << name << " parameter " << i << ": " << os.str();
            ++i;
        }(),
        ...);
}
}  // namespace

TEST(ContractGates, F19_StreamRoundTripIsExact_All27) {
    const double t = 1.0 / 3.0, s = 2.0 / 7.0, tiny = kPi * 1e-7;
    checkStreamRoundTrip("Bernoulli", BernoulliDistribution(t),
                         [](const BernoulliDistribution& d) { return d.getP(); });
    checkStreamRoundTrip(
        "Beta", BetaDistribution(10 * t, 10 * s),
        [](const BetaDistribution& d) { return d.getAlpha(); },
        [](const BetaDistribution& d) { return d.getBeta(); });
    checkStreamRoundTrip(
        "Binomial", BinomialDistribution(10, t),
        [](const BinomialDistribution& d) { return d.getN(); },
        [](const BinomialDistribution& d) { return d.getP(); });
    checkStreamRoundTrip(
        "Cauchy", CauchyDistribution(t, tiny),
        [](const CauchyDistribution& d) { return d.getX0(); },
        [](const CauchyDistribution& d) { return d.getGamma(); });
    checkStreamRoundTrip("ChiSquared", ChiSquaredDistribution(10 * t),
                         [](const ChiSquaredDistribution& d) { return d.getK(); });
    checkStreamRoundTrip(
        "Discrete", DiscreteDistribution(-5, 9),
        [](const DiscreteDistribution& d) { return d.getLowerBound(); },
        [](const DiscreteDistribution& d) { return d.getUpperBound(); });
    checkStreamRoundTrip(
        "Erlang", ErlangDistribution(3, t), [](const ErlangDistribution& d) { return d.getK(); },
        [](const ErlangDistribution& d) { return d.getLambda(); });
    checkStreamRoundTrip("Exponential", ExponentialDistribution(t),
                         [](const ExponentialDistribution& d) { return d.getLambda(); });
    checkStreamRoundTrip(
        "FisherF", FDistribution(10 * t, 10 * s), [](const FDistribution& d) { return d.getD1(); },
        [](const FDistribution& d) { return d.getD2(); });
    checkStreamRoundTrip(
        "Gamma", GammaDistribution(10 * t, s),
        [](const GammaDistribution& d) { return d.getAlpha(); },
        [](const GammaDistribution& d) { return d.getBeta(); });
    checkStreamRoundTrip(
        "Gaussian", GaussianDistribution(t, tiny),
        [](const GaussianDistribution& d) { return d.getMean(); },
        [](const GaussianDistribution& d) { return d.getStandardDeviation(); });
    checkStreamRoundTrip("Geometric", GeometricDistribution(t),
                         [](const GeometricDistribution& d) { return d.getP(); });
    checkStreamRoundTrip(
        "Gumbel", GumbelDistribution(t, tiny),
        [](const GumbelDistribution& d) { return d.getMu(); },
        [](const GumbelDistribution& d) { return d.getBeta(); });
    checkStreamRoundTrip("HalfNormal", HalfNormalDistribution(tiny),
                         [](const HalfNormalDistribution& d) { return d.getSigma(); });
    checkStreamRoundTrip(
        "InverseGamma", InverseGammaDistribution(10 * t, s),
        [](const InverseGammaDistribution& d) { return d.getAlpha(); },
        [](const InverseGammaDistribution& d) { return d.getBeta(); });
    checkStreamRoundTrip(
        "Laplace", LaplaceDistribution(t, tiny),
        [](const LaplaceDistribution& d) { return d.getMu(); },
        [](const LaplaceDistribution& d) { return d.getB(); });
    checkStreamRoundTrip(
        "Logistic", LogisticDistribution(t, tiny),
        [](const LogisticDistribution& d) { return d.getMu(); },
        [](const LogisticDistribution& d) { return d.getS(); });
    checkStreamRoundTrip(
        "LogNormal", LogNormalDistribution(t, s),
        [](const LogNormalDistribution& d) { return d.getMu(); },
        [](const LogNormalDistribution& d) { return d.getSigma(); });
    checkStreamRoundTrip(
        "NegBinomial", NegativeBinomialDistribution(10 * t, tiny),
        [](const NegativeBinomialDistribution& d) { return d.getR(); },
        [](const NegativeBinomialDistribution& d) { return d.getP(); });
    checkStreamRoundTrip(
        "Pareto", ParetoDistribution(tiny, 10 * t),
        [](const ParetoDistribution& d) { return d.getScale(); },
        [](const ParetoDistribution& d) { return d.getAlpha(); });
    checkStreamRoundTrip("Poisson", PoissonDistribution(10 * t),
                         [](const PoissonDistribution& d) { return d.getLambda(); });
    checkStreamRoundTrip("Rayleigh", RayleighDistribution(tiny),
                         [](const RayleighDistribution& d) { return d.getSigma(); });
    checkStreamRoundTrip("StudentT", StudentTDistribution(10 * t),
                         [](const StudentTDistribution& d) { return d.getNu(); });
    checkStreamRoundTrip(
        "TruncNormal", TruncatedNormalDistribution(t, s, -s, 5 * t),
        [](const TruncatedNormalDistribution& d) { return d.getMu(); },
        [](const TruncatedNormalDistribution& d) { return d.getSigma(); },
        [](const TruncatedNormalDistribution& d) { return d.getLowerBound(); },
        [](const TruncatedNormalDistribution& d) { return d.getUpperBound(); });
    checkStreamRoundTrip(
        "Uniform", UniformDistribution(t, 10 * s),
        [](const UniformDistribution& d) { return d.getLowerBound(); },
        [](const UniformDistribution& d) { return d.getUpperBound(); });
    checkStreamRoundTrip(
        "VonMises", VonMisesDistribution(t, 10 * s),
        [](const VonMisesDistribution& d) { return d.getMu(); },
        [](const VonMisesDistribution& d) { return d.getKappa(); });
    checkStreamRoundTrip(
        "Weibull", WeibullDistribution(10 * t, tiny),
        [](const WeibullDistribution& d) { return d.getShape(); },
        [](const WeibullDistribution& d) { return d.getScale(); });
}

//==============================================================================
// #173 (AR C1): no lgamma call writes the global signgam (POSIX: MT-unsafe).
// Unfixed: detail::lgamma was std::lgamma, and 33 call sites called it directly;
// each wrote signgam. MSVC has no signgam (and its lgamma writes none).
//==============================================================================

#if !defined(_WIN32)
TEST(ContractGates, Issue173_LgammaLeavesSigngamAlone) {
    constexpr int kSentinel = 12345;
    signgam = kSentinel;
    volatile double sink = 0.0;
    sink = sink + detail::lgamma(-0.5);  // Gamma(-0.5) < 0: libm would write -1
    sink = sink + detail::lgamma(7.25);
    EXPECT_EQ(signgam, kSentinel) << "detail::lgamma wrote signgam";

    signgam = kSentinel;
    const auto exercise = [&](const DistributionBase& d, double x) {
        sink = sink + d.getProbability(x) + d.getLogProbability(x) + d.getCumulativeProbability(x) +
               d.getQuantile(0.3) + d.getMean() + d.getVariance() + d.getSkewness() +
               d.getKurtosis() + d.getEntropy();
    };
    exercise(BinomialDistribution(10, 0.3), 3.0);
    exercise(BinomialDistribution(100000, 0.3), 30000.0);
    exercise(PoissonDistribution(3.5), 2.0);
    exercise(PoissonDistribution(1e5), 1e5 + 7.0);
    exercise(NegativeBinomialDistribution(3.5, 0.4), 4.0);
    exercise(GammaDistribution(2.5, 1.5), 1.2);
    exercise(GammaDistribution(50.0, 1.0), 49.0);
    exercise(BetaDistribution(2.5, 3.5), 0.4);
    exercise(BetaDistribution(30.0, 40.0), 0.4);
    exercise(WeibullDistribution(1.7, 2.0), 1.1);
    exercise(StudentTDistribution(4.5), 0.7);
    exercise(ChiSquaredDistribution(3.5), 2.0);
    exercise(FDistribution(4.0, 9.0), 1.3);
    exercise(InverseGammaDistribution(3.0, 2.0), 0.8);
    exercise(ErlangDistribution(4, 2.0), 1.5);
    exercise(GeometricDistribution(0.3), 2.0);
    {
        WeibullDistribution w(1.7, 2.0);
        std::vector<double> data{0.5, 1.0, 1.5, 2.0, 3.0, 4.5};
        w.fit(data);
        GammaDistribution g(2.0, 1.0);
        g.fit(data);
        BetaDistribution b(2.0, 2.0);
        b.fit(std::vector<double>{0.1, 0.4, 0.5, 0.7, 0.9});
        sink = sink + w.getMean() + g.getMean() + b.getMean();
    }
    EXPECT_EQ(signgam, kSentinel) << "a distribution path called a signgam-writing lgamma";
    (void)sink;
}
#endif

//==============================================================================
// F20 (DH mutation addendum): accuracy gates on the large-count log-pmfs that
// fail when either accuracy fix is reverted. Mutant A (log1pmx below 1/2 by the
// direct log1p(t) - t, pre-b5a47df) and mutant D (#172 two_prod without the fma
// residual) both passed the whole suite; here A errs 3.5e-13..7e-13 and D
// 2e-13..5e-13 at these points, against <= 9.3e-15 on head. References: log-gamma
// by Stirling's series in 40-digit decimal, p the double 0.3.
//==============================================================================

TEST(ContractGates, F20_LargeCountPmfAccuracy) {
    constexpr double kTol = 2.5e-14;
    const BinomialDistribution b(1000000, 0.3);
    const struct {
        double k, pmf;
    } binom[] = {
        {302750.0, 1.3563110862792082e-11}, {297250.0, 1.2803418342626395e-11},
        {298625.0, 9.631376597314238e-06},  {290835.0, 3.7315880094871274e-91},
        {293126.0, 7.347284302178108e-53},  {306874.0, 1.936477835512476e-52},
        {296334.0, 1.0274527495709011e-17},
    };
    for (const auto& r : binom)
        EXPECT_LT(relErr(b.getProbability(r.k), r.pmf), kTol) << "Binomial(1e6, 0.3) k=" << r.k;
    const PoissonDistribution p(100000.0);
    EXPECT_LT(relErr(p.getProbability(96838.0), 1.4603428444776278e-25), kTol);
}

//==============================================================================
// F5, second half (AR D5): Cauchy and InverseGamma assert no overlap on the
// caller's spans; their delegates see a scratch buffer, so the dispatcher's own
// assert could not. Debug builds only: the assert compiles away under NDEBUG.
//==============================================================================

#if !defined(NDEBUG) && GTEST_HAS_DEATH_TEST
TEST(ContractGatesDeathTest, F5_OverlapAssertOnCallerSpans) {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    std::vector<double> buf(64, 1.5);
    const std::span<const double> in(buf.data(), 32);
    const std::span<double> out(buf.data() + 16, 32);
    const CauchyDistribution c(0.5, 2.0);
    EXPECT_DEATH(c.getProbability(in, out), "overlap");
    EXPECT_DEATH(c.getLogProbability(in, out), "overlap");
    const InverseGammaDistribution ig(3.0, 2.0);
    EXPECT_DEATH(ig.getLogProbability(in, out), "overlap");
    EXPECT_DEATH(ig.getProbability(in, out), "overlap");
}
#endif

//==============================================================================
// F21 (#227): an owner that delegates (ChiSquared, Erlang, InverseGamma -> Gamma;
// FisherF -> Beta; Bernoulli -> Binomial; Geometric -> NegativeBinomial) must
// reject every value its delegate rejects, and after every accepted set must
// behave bit for bit like a freshly constructed owner with the same parameters.
// Unfixed: ChiSquared k = 4.94e-324 (k/2 rounds to 0) and FisherF d1 or d2 =
// 4.94e-324 were accepted, the delegate kept its old parameters, and a fresh
// owner with the new parameters could not be constructed at all.
//==============================================================================

namespace {
constexpr double kDenormMin = std::numeric_limits<double>::denorm_min();

const std::vector<double>& delegateProbes() {
    static const std::vector<double> probes{-1.0,
                                            -0.0,
                                            0.0,
                                            kDenormMin,
                                            2.0 * kDenormMin,
                                            3.0 * kDenormMin,
                                            std::numeric_limits<double>::min(),
                                            1e-300,
                                            0.25,
                                            0.5,
                                            std::nextafter(1.0, 0.0),
                                            1.0,
                                            std::nextafter(1.0, 2.0),
                                            2.0,
                                            7.5,
                                            1e300,
                                            std::numeric_limits<double>::max(),
                                            std::numeric_limits<double>::infinity(),
                                            kNaN};
    return probes;
}

// One settable parameter of one delegating class.
template <class D>
struct DelegateParam {
    const char* name;
    std::function<VoidResult(D&, double)> trySet;
    std::function<void(D&, double)> set;
    // Would the delegate accept the parameters `d` would have after setting v?
    std::function<bool(const D&, double)> delegateAccepts;
};

template <class D>
void checkDelegateAgreement(const char* cls, const D& base,
                            const std::vector<DelegateParam<D>>& params,
                            const std::function<std::vector<double>(const D&)>& getParams,
                            const std::function<D(const std::vector<double>&)>& fresh,
                            const std::function<double(const D&)>& probe) {
    for (const auto& p : params) {
        for (double v : delegateProbes()) {
            SCOPED_TRACE(std::string(cls) + "::" + p.name + "(" + std::to_string(v) + ")");
            const bool delegateOk = p.delegateAccepts(base, v);
            D d = base;
            const auto before = getParams(d);
            const auto r = p.trySet(d, v);
            if (r.isError()) {
                EXPECT_EQ(getParams(d), before) << "a rejected set changed the parameters";
            } else {
                EXPECT_TRUE(delegateOk) << "owner accepted a value its delegate rejects";
                try {
                    const D f = fresh(getParams(d));
                    EXPECT_EQ(bitsOf(probe(d)), bitsOf(probe(f)))
                        << "owner and delegate disagree after an accepted set";
                } catch (const std::exception& e) {
                    ADD_FAILURE() << "a fresh owner rejects parameters the setter accepted: "
                                  << e.what();
                }
            }
            D e = base;
            bool threw = false;
            try {
                p.set(e, v);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            EXPECT_EQ(threw, r.isError()) << "throwing setter and trySet disagree";
            EXPECT_EQ(getParams(e), getParams(d));
        }
    }
}
}  // namespace

TEST(ContractGates, F21_DelegateAgreement_ChiSquared) {
    {
        using D = ChiSquaredDistribution;
        checkDelegateAgreement<D>(
            "ChiSquared", D(3.0),
            {{"k", [](D& d, double v) { return d.trySetK(v); }, [](D& d, double v) { d.setK(v); },
              [](const D&, double v) { return validateGammaParameters(v / 2.0, 0.5).isOk(); }}},
            [](const D& d) { return std::vector<double>{d.getK()}; },
            [](const std::vector<double>& p) { return D(p[0]); },
            [](const D& d) { return d.getMean(); });
        // The issue's own quantity: the delegate's mean is k.
        D d(3.0);
        EXPECT_TRUE(d.trySetK(2.0 * kDenormMin).isOk());
        EXPECT_EQ(d.getMean(), 2.0 * kDenormMin);
        D s(3.0);
        std::istringstream in("ChiSquaredDistribution(k=4.9406564584124654e-324)");
        in >> s;
        EXPECT_TRUE(in.fail()) << "operator>> accepted a k the delegate rejects";
        EXPECT_EQ(s.getK(), 3.0);
        EXPECT_EQ(s.getMean(), 3.0);
        D fitted(3.0);
        EXPECT_THROW(fitted.fit(std::vector<double>{kDenormMin, kDenormMin}),
                     std::invalid_argument);
        EXPECT_EQ(fitted.getK(), 3.0);
        EXPECT_EQ(fitted.getMean(), 3.0);
        // Last: unfixed, create() reached Gamma's throwing constructor from a noexcept path and
        // called std::terminate.
        EXPECT_THROW(D{kDenormMin}, std::invalid_argument);
        EXPECT_TRUE(D::create(kDenormMin).isError());
    }
}

TEST(ContractGates, F21_DelegateAgreement_FisherF) {
    {
        using D = FDistribution;
        const auto betaOk = [](double d1, double d2) {
            return validateBetaParameters(d1 * 0.5, d2 * 0.5).isOk();
        };
        checkDelegateAgreement<D>(
            "FisherF", D(5.0, 7.0),
            {{"d1", [](D& d, double v) { return d.trySetD1(v); },
              [](D& d, double v) { d.setD1(v); },
              [&](const D& d, double v) { return betaOk(v, d.getD2()); }},
             {"d2", [](D& d, double v) { return d.trySetD2(v); },
              [](D& d, double v) { d.setD2(v); },
              [&](const D& d, double v) { return betaOk(d.getD1(), v); }},
             {"params", [](D& d, double v) { return d.trySetParameters(v, v); },
              [](D& d, double v) { d.setParameters(v, v); },
              [&](const D&, double v) { return betaOk(v, v); }}},
            [](const D& d) { return std::vector<double>{d.getD1(), d.getD2()}; },
            [](const std::vector<double>& p) { return D(p[0], p[1]); },
            [](const D& d) {
                // beta_ is observable only through sample(); the Beta sampler does not return
                // for extreme shapes, so sample where it terminates. Outside that range the
                // rejection side above still applies.
                const auto ok = [](double v) { return v >= 1e-2 && v <= 1e8; };
                if (!ok(d.getD1()) || !ok(d.getD2()))
                    return 0.0;
                std::mt19937 rng(42);
                return d.sample(rng);
            });
        D s(5.0, 7.0);
        std::istringstream in("FDistribution(d1=4.9406564584124654e-324,d2=7)");
        in >> s;
        EXPECT_TRUE(in.fail()) << "operator>> accepted a d1 the delegate rejects";
        EXPECT_EQ(s.getD1(), 5.0);
        EXPECT_THROW((D{kDenormMin, 1.0}), std::invalid_argument);
        EXPECT_TRUE(D::create(kDenormMin, 1.0).isError());
        EXPECT_TRUE(D::create(1.0, kDenormMin).isError());
    }
}

TEST(ContractGates, F21_DelegateAgreement_Erlang) {
    {
        using D = ErlangDistribution;
        const auto gammaOk = [](double k, double l) {
            return validateGammaParameters(k, l).isOk();
        };
        // k is an int: the probes map through a saturating cast (negative, 0, 1, 2, 7, INT_MAX).
        const auto toK = [](double v) {
            if (!(v > 0.0))
                return std::isnan(v) ? 0 : -1;
            return v >= 2147483647.0 ? 2147483647 : static_cast<int>(v);
        };
        checkDelegateAgreement<D>(
            "Erlang", D(3, 2.0),
            {{"k", [&](D& d, double v) { return d.trySetK(toK(v)); },
              [&](D& d, double v) { d.setK(toK(v)); },
              [&](const D& d, double v) { return toK(v) >= 1 && gammaOk(toK(v), d.getLambda()); }},
             {"lambda", [](D& d, double v) { return d.trySetLambda(v); },
              [](D& d, double v) { d.setLambda(v); },
              [&](const D& d, double v) { return gammaOk(d.getK(), v); }}},
            [](const D& d) {
                return std::vector<double>{static_cast<double>(d.getK()), d.getLambda()};
            },
            [](const std::vector<double>& p) { return D(static_cast<int>(p[0]), p[1]); },
            [](const D& d) { return d.getLogProbability(0.75); });
    }
}

TEST(ContractGates, F21_DelegateAgreement_InverseGamma) {
    {
        using D = InverseGammaDistribution;
        checkDelegateAgreement<D>(
            "InverseGamma", D(3.0, 2.0),
            {{"alpha", [](D& d, double v) { return d.trySetAlpha(v); },
              [](D& d, double v) { d.setAlpha(v); },
              [](const D& d, double v) { return validateGammaParameters(v, d.getBeta()).isOk(); }},
             {"beta", [](D& d, double v) { return d.trySetBeta(v); },
              [](D& d, double v) { d.setBeta(v); },
              [](const D& d, double v) {
                  return validateGammaParameters(d.getAlpha(), v).isOk();
              }}},
            [](const D& d) { return std::vector<double>{d.getAlpha(), d.getBeta()}; },
            [](const std::vector<double>& p) { return D(p[0], p[1]); },
            [](const D& d) { return d.getLogProbability(0.75); });
    }
}

TEST(ContractGates, F21_DelegateAgreement_Bernoulli) {
    {
        using D = BernoulliDistribution;
        checkDelegateAgreement<D>(
            "Bernoulli", D(0.3),
            {{"p", [](D& d, double v) { return d.trySetP(v); }, [](D& d, double v) { d.setP(v); },
              [](const D&, double v) { return validateBinomialParameters(1, v).isOk(); }}},
            [](const D& d) { return std::vector<double>{d.getP()}; },
            [](const std::vector<double>& p) { return D(p[0]); },
            [](const D& d) { return d.getProbability(1.0); });
    }
}

TEST(ContractGates, F21_DelegateAgreement_Geometric) {
    {
        using D = GeometricDistribution;
        checkDelegateAgreement<D>(
            "Geometric", D(0.3),
            {{"p", [](D& d, double v) { return d.trySetP(v); }, [](D& d, double v) { d.setP(v); },
              [](const D&, double v) {
                  return validateNegativeBinomialParameters(1.0, v).isOk();
              }}},
            [](const D& d) { return std::vector<double>{d.getP()}; },
            [](const std::vector<double>& p) { return D(p[0]); },
            [](const D& d) { return d.getLogProbability(2.0); });
    }
}
