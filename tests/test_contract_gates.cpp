// Contract gates from the v2.5.0 architecture review (AR) and defect hunt (DH), 2026-10-07.
//
// Each gate was run against the unfixed code first and seen to fail there; the comment above
// each TEST names the defect and the failure it caught. Correctness only: no timing assertion,
// so this binary carries no "timing" label and runs in CI's -LE "timing|benchmark" pass.
#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <math.h>  // signgam (POSIX)
#include <numbers>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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

// One-sample Kolmogorov-Smirnov on u = F(x), with Stephens' finite-n correction on the
// asymptotic Kolmogorov survival. Returns the p-value.
double ksPValue(std::vector<double> u) {
    std::sort(u.begin(), u.end());
    const double n = static_cast<double>(u.size());
    double D = 0.0;
    for (std::size_t i = 0; i < u.size(); ++i) {
        const double i0 = static_cast<double>(i);
        D = std::max(D, std::max(u[i] - i0 / n, (i0 + 1.0) / n - u[i]));
    }
    const double lam = D * (std::sqrt(n) + 0.12 + 0.11 / std::sqrt(n));
    if (lam < 1e-3)
        return 1.0;
    double s = 0.0;
    for (int k = 1; k < 200; ++k) {
        const double t = 2.0 * ((k % 2) ? 1.0 : -1.0) * std::exp(-2.0 * k * k * lam * lam);
        s += t;
        if (std::fabs(t) < 1e-18)
            break;
    }
    return std::min(1.0, std::max(0.0, s));
}

// Runs fn on a worker thread; if it has not returned within `seconds`, the defect under
// test is a hang (DH2 S1), which no assertion can report, so the process exits non-zero.
// The worker is left running: _Exit skips the destructors it would otherwise race.
template <typename Fn>
void runOrHang(Fn&& fn, int seconds, const char* what) {
    std::atomic<bool> done{false};
    std::thread worker([&] {
        fn();
        done.store(true, std::memory_order_release);
    });
    for (int i = 0; i < seconds * 100 && !done.load(std::memory_order_acquire); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if (!done.load(std::memory_order_acquire)) {
        std::fprintf(stderr, "HANG: %s did not return within %d s\n", what, seconds);
        std::fflush(stderr);
        std::_Exit(3);
    }
    worker.join();
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
    // The double F(k) = 0.99876987710024256462 lies above the exact F(k) = 0.99876987710024256243
    // (mpmath), and below F(k + 1), so its exact quantile is k + 1; the survival-side search
    // (p > ½) returns it, where the CDF-side search returned k on F's rounding.
    EXPECT_EQ(g.getQuantile(g.getCumulativeProbability(k)), k + 1.0);
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

//==============================================================================
// F22 (#224): the Gamma and Erlang CDFs are 1 where x·rate overflows to +inf,
// scalar and under every batch strategy, and detail::gamma_p / gamma_q take
// x = +inf as 1 and 0. Unfixed: gamma_p(a, inf) was NaN (the continued
// fraction's prefactor is -inf + ...), so Gamma(100, 100).cdf(DBL_MAX) and
// Erlang(1, 1e300).cdf(1e18) were NaN while cdf(+inf) itself was guarded.
//==============================================================================

TEST(ContractGates, F22_GammaCdfOneWhereRateTimesXOverflows) {
    constexpr double kMax = std::numeric_limits<double>::max();
    EXPECT_EQ(detail::gamma_p(2.5, std::numeric_limits<double>::infinity()), 1.0);
    EXPECT_EQ(detail::gamma_q(2.5, std::numeric_limits<double>::infinity()), 0.0);

    const GammaDistribution g(100.0, 100.0);
    EXPECT_EQ(g.getCumulativeProbability(kMax), 1.0);
    EXPECT_EQ(g.getCumulativeProbability(1e307), 1.0);
    const ErlangDistribution e(1, 1e300);
    EXPECT_EQ(e.getCumulativeProbability(1e18), 1.0);
    EXPECT_EQ(e.getCumulativeProbability(kMax), 1.0);

    // Batch, every strategy: the overflow elements among ordinary ones.
    std::vector<double> xs(69, 1.0);  // 8*8+5: SIMD body and scalar tail on every tier width
    xs[0] = kMax;
    xs[1] = 1e307;
    xs[68] = kMax;
    using PS = detail::PerformanceHint::PreferredStrategy;
    for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL}) {
        std::vector<double> out(xs.size(), -1.0);
        g.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out),
                                   detail::PerformanceHint{s, std::nullopt});
        EXPECT_EQ(out[0], 1.0) << static_cast<int>(s);
        EXPECT_EQ(out[1], 1.0) << static_cast<int>(s);
        EXPECT_EQ(out[68], 1.0) << static_cast<int>(s);
        EXPECT_EQ(out[2], g.getCumulativeProbability(1.0)) << static_cast<int>(s);
        std::vector<double> eout(xs.size(), -1.0);
        e.getCumulativeProbability(std::span<const double>(xs), std::span<double>(eout),
                                   detail::PerformanceHint{s, std::nullopt});
        EXPECT_EQ(eout[0], 1.0) << static_cast<int>(s);
        EXPECT_EQ(eout[2], 1.0) << static_cast<int>(s);  // 1e300 * 1 = 1e300: Q(1, 1e300) = 0
    }
}

//==============================================================================
// DH2 S1: von Mises sample() never returned for kappa in [1e-9, ~1.5e-8). The Best-Fisher
// constants were formed as rho = (tau - sqrt(2 tau)) / (2 kappa) with tau = 1 + sqrt(1 + 4
// kappa^2), which is 2 - 2 = 0 exactly once 4 kappa^2 < epsilon: r = +inf, the acceptance test
// compared NaN, and the rejection loop never accepted. Just above, rho carried only a few bits and
// the draws were biased. Both overloads. Now rho = 2 kappa / (tau + sqrt(2 tau)) (no cancellation),
// so every kappa terminates and the draws follow the library CDF.
//
// Fixed seed. KS of u = F(x) at 6 kappa x 2 overloads = 12 tests, rejected at p < 1e-4: a
// false alarm once in ~800 runs of the suite if the sampler is right.
//==============================================================================

TEST(ContractGates, DH2_S1_VonMisesSamplerTerminatesAndFollowsCdf) {
    for (const double kappa : {1e-12, 1e-9, 1e-8, 1e-6, 1e-3, 1.0}) {
        const VonMisesDistribution v(0.3, kappa);
        std::vector<double> xs, xv;
        runOrHang(
            [&] {
                std::mt19937 rng(20261009);
                xs.reserve(4000);
                for (int i = 0; i < 4000; ++i)
                    xs.push_back(v.sample(rng));
                xv = v.sample(rng, 4000);
            },
            20, "VonMises sample at tiny kappa");
        for (const auto* draws : {&xs, &xv}) {
            ASSERT_EQ(draws->size(), 4000u);
            std::vector<double> u;
            u.reserve(draws->size());
            for (const double x : *draws) {
                ASSERT_TRUE(std::isfinite(x)) << kappa;
                ASSERT_GT(x, -kPi) << kappa;
                ASSERT_LE(x, kPi) << kappa;
                u.push_back(v.getCumulativeProbability(x));
            }
            EXPECT_GT(ksPValue(u), 1e-4)
                << "kappa " << kappa << (draws == &xs ? " scalar" : " vector");
        }
    }
}

//==============================================================================
// DH3: von Mises sample() at large kappa. The Best-Fisher sampler returned acos(f) and accepted on
// c = kappa (r - f), with f within 1/kappa of 1 near the mean, where doubles are 1.1e-16 apart. The
// draws sat on a lattice of relative spacing ~1.1e-16 kappa near theta = 1/sqrt(kappa): 59
// repeated values in 20000 draws at kappa = 1e9, 9062 at 1e12, KS sqrt(n) D = 6.97 at 1e14 and
// 21 at 1e15. From kappa ~ 6.5e15, r = (1 + rho^2)/(2 rho) rounded to 1, c = 0, every draw was
// rejected and sample() never returned. Now: Best-Fisher in half-angle form from kappa = 1e4,
// a normal draw with sigma = 1/sqrt(kappa) from 1e15 (where the von Mises law and the normal
// differ by < 4.4e-17 in CDF), the textbook form (unchanged streams) below 1e4. Both overloads.
//==============================================================================

TEST(ContractGates, DH3_VonMisesSamplerTerminatesAtHugeKappa) {
    for (const double kappa : {6.5e15, 1e16, 1e300, std::numeric_limits<double>::max()}) {
        const VonMisesDistribution v(0.3, kappa);
        std::vector<double> xs, xv;
        runOrHang(
            [&] {
                std::mt19937 rng(20261010);
                for (int i = 0; i < 1000; ++i)
                    xs.push_back(v.sample(rng));
                xv = v.sample(rng, 1000);
            },
            20, "VonMises sample at huge kappa");
        ASSERT_EQ(xs.size(), 1000u);
        ASSERT_EQ(xv.size(), 1000u);
        // In range, and within 10 standard deviations of the mean (the normal draw is at most
        // ~9.4 sigma: Box-Muller's u1 is at least 2^-64).
        for (const auto* draws : {&xs, &xv}) {
            for (const double x : *draws) {
                ASSERT_TRUE(std::isfinite(x)) << kappa;
                ASSERT_GT(x, -kPi) << kappa;
                ASSERT_LE(x, kPi) << kappa;
                ASSERT_LE(std::fabs(x - 0.3), 10.0 / std::sqrt(kappa) + 1e-16) << kappa;
            }
        }
    }
}

// KS of u = F(x) against the library CDF at the defect's kappa and on both sides of the two new
// switches (1e4, 1e15): 8 kappa x 2 overloads = 16 tests at n = 20000, rejected at p < 1e-4, a
// false alarm once in ~600 runs of the suite if the seed were varied (it is fixed). Above kappa =
// 1000 the CDF is the wrapped normal, 0.0435/kappa absolute off (4e-6 at 1e4), far below the KS
// resolution 1/sqrt(n) = 7e-3. At mu = 0
// the draws are theta itself, at full relative precision, so a correct sampler repeats a value in
// 20000 draws with probability ~1e-7; the lattice repeated 59 times at kappa = 1e9.
TEST(ContractGates, DH3_VonMisesSamplerFollowsCdfAtLargeKappa) {
    for (const double kappa : {9999.0, 1e4, 1e9, 1e12, 1e14, 9.99e14, 1e15, 1e16}) {
        for (const double mu : {0.3, 0.0}) {
            const VonMisesDistribution v(mu, kappa);
            std::vector<double> xs, xv;
            runOrHang(
                [&] {
                    std::mt19937 rng(20261010);
                    xs.reserve(20000);
                    for (int i = 0; i < 20000; ++i)
                        xs.push_back(v.sample(rng));
                    xv = v.sample(rng, 20000);
                },
                20, "VonMises sample at large kappa");
            for (const auto* draws : {&xs, &xv}) {
                const char* which = (draws == &xs ? " scalar" : " vector");
                ASSERT_EQ(draws->size(), 20000u);
                if (mu == 0.0) {
                    std::vector<double> sorted = *draws;
                    std::sort(sorted.begin(), sorted.end());
                    const auto last = std::unique(sorted.begin(), sorted.end());
                    const auto repeats = static_cast<std::size_t>(sorted.end() - last);
                    EXPECT_EQ(repeats, 0u) << "kappa " << kappa << which;
                    continue;
                }
                std::vector<double> u;
                u.reserve(draws->size());
                for (const double x : *draws) {
                    ASSERT_TRUE(std::isfinite(x)) << kappa;
                    ASSERT_GT(x, -kPi) << kappa;
                    ASSERT_LE(x, kPi) << kappa;
                    u.push_back(v.getCumulativeProbability(x));
                }
                EXPECT_GT(ksPValue(u), 1e-4) << "kappa " << kappa << which;
            }
        }
    }
}

//==============================================================================
// DH2 M1: von Mises getEntropy() had both signs wrong, in the code and in the header formula:
// log(2 pi) - log I0(kappa) + kappa I1/I0. The entropy is log(2 pi I0(kappa)) - kappa I1/I0.
// kappa = 1 returned 2.0484 (true 1.6274), kappa = 100 returned 4.557 (true -0.881); only
// kappa -> 0 agreed. References: mpmath, dps 50.
//==============================================================================

TEST(ContractGates, DH2_M1_VonMisesEntropy) {
    struct Row {
        double kappa, h, tol;
    };
    // Tolerance: the Tier 2 (A&S 9.8) Bessel polynomials carry ~2e-7 relative in I0 and I1,
    // so log I0 and kappa * A(kappa) inherit it below the asymptotic cut (kappa < 50); from
    // the cut the scaled asymptotic series are accurate to double on both tiers.
    for (const Row& r :
         {Row{0.01, 1.8378520668780868, 1e-6}, Row{0.5, 1.7781769793044258, 1e-6},
          Row{1.0, 1.6274014590199896, 1e-6}, Row{10.0, 0.29485088997958145, 5e-6},
          Row{100.0, -0.88112754416494736, 1e-9}, Row{700.0, -1.8562441080454707, 1e-9},
          Row{1e4, -3.1862066509081582, 1e-9}, Row{1e6, -5.4888164957772768, 1e-9},
          Row{1e8, -7.7914018362715100, 1e-9}}) {
        const VonMisesDistribution v(0.0, r.kappa);
        EXPECT_NEAR(v.getEntropy(), r.h, r.tol) << r.kappa;
    }
    EXPECT_EQ(VonMisesDistribution(0.0, 0.0).getEntropy(), detail::LN_2PI);
}

//==============================================================================
// DH2 M2: Gamma getEntropy() formed alpha - log beta + lgamma(alpha) + (1 - alpha) psi(alpha)
// directly, which cancels at alpha ln alpha: 7.5e-7 relative at 1e10, 15.238 for 15.2344 at
// 1e12, -64 for 19.84 at 1e16, 0 at 1e20; ChiSquared(1e17) gave 0 for 21.34. Alpha >= 100
// now takes the Stirling form 1/2 log(2 pi e alpha) - 1/(3 alpha) - ... - log beta.
// ChiSquared and Erlang delegate. References: mpmath, dps 50.
// From the switch on the tolerance is a few ulps of h (2026-10-11): the former flat 2e-12 is
// ~550 ulps at h = 3.7, and a sign flip of the alpha^-5 coefficient -- 9.5e-13 at alpha = 100 --
// passed it.
//==============================================================================

TEST(ContractGates, DH2_M2_GammaEntropyLargeShape) {
    struct Row {
        double alpha, h;
    };
    // 4 ulps of the expected value: the reference is rounded once (1/2 ulp), the - log beta
    // shift adds a rounding, and the Stirling form a few more.
    const auto fewUlps = [](double h) {
        return 4.0 * std::numeric_limits<double>::epsilon() * std::fabs(h);
    };
    for (const Row& r :
         {Row{20.0, 2.8999283345986621}, Row{50.0, 3.3682499483781684},
          Row{100.0, 3.7181819485047462}, Row{1e3, 4.8724827560179718},
          Row{1e4, 6.0240753850260863}, Row{1e6, 8.3266934788533931}, Row{1e8, 10.629278901847522},
          Row{1e10, 12.931863998141568}, Row{1e12, 15.234449091168614},
          Row{1e16, 19.839619277157038}, Row{1e20, 24.44478946314513}}) {
        // Below the switch the direct form is unchanged and carries ~alpha log alpha eps.
        const double hb = r.h - std::log(4.0);
        const double tol = r.alpha < 100.0 ? 1e-11 : fewUlps(r.h);
        const double tolb = r.alpha < 100.0 ? 1e-11 : fewUlps(hb);
        EXPECT_NEAR(GammaDistribution(r.alpha, 1.0).getEntropy(), r.h, tol) << r.alpha;
        // - log beta is additive: beta = 4 shifts by -log 4.
        EXPECT_NEAR(GammaDistribution(r.alpha, 4.0).getEntropy(), hb, tolb) << r.alpha;
    }
    EXPECT_NEAR(ChiSquaredDistribution(1e17).getEntropy(), 21.337485413934034,
                fewUlps(21.337485413934034));
    EXPECT_NEAR(ErlangDistribution(1000000, 2.0).getEntropy(), 8.3266934788533931 - std::log(2.0),
                fewUlps(8.3266934788533931 - std::log(2.0)));
    // The series' own tail: 1e300 is 1/2 log(2 pi e alpha) to double.
    EXPECT_NEAR(GammaDistribution(1e300, 1.0).getEntropy(),
                0.5 * std::log(2.0 * kPi * std::exp(1.0)) + 0.5 * std::log(1e300), 1e-12);
}

//==============================================================================
// DH2 F2: Weibull fit was not scale-invariant. The method-of-moments seed used sum(x^2), the
// Newton iterate exp(k log x) underflowed or overflowed with the scale, and lambda was
// bounded by MAX_DISTRIBUTION_PARAMETER: Weibull(3, 1e-50) data fitted (1, 1) silently (the
// reset defaults, the #207 class), Weibull(3, 1e30) data threw. Now the seed comes from the
// variance of log x and the sums are formed on x / max(x), so the fit is invariant to the
// data scale across the finite range; constant data throws.
//==============================================================================

TEST(ContractGates, DH2_F2_WeibullFitScaleInvariant) {
    std::mt19937 rng(42);
    const WeibullDistribution truth(3.0, 1.0);
    const std::vector<double> base = truth.sample(rng, 2000);
    WeibullDistribution ref(1.0, 1.0);
    ref.fit(base);
    EXPECT_NEAR(ref.getShape(), 3.0, 0.15);
    EXPECT_NEAR(ref.getScale(), 1.0, 0.03);
    for (const double s : {1e-300, 1e-100, 1e-30, 1e-7, 1e7, 1e30, 1e100, 1e300}) {
        std::vector<double> scaled(base.size());
        for (std::size_t i = 0; i < base.size(); ++i)
            scaled[i] = base[i] * s;
        WeibullDistribution f(1.0, 1.0);
        ASSERT_NO_THROW(f.fit(scaled)) << s;
        EXPECT_LT(relErr(f.getShape(), ref.getShape()), 1e-9) << s;
        EXPECT_LT(relErr(f.getScale(), ref.getScale() * s), 1e-9) << s;
    }
    // Constant data: the shape is undefined. An exception, never a silent default.
    WeibullDistribution c(2.0, 5.0);
    EXPECT_THROW(c.fit(std::vector<double>(50, 2.5)), std::invalid_argument);
    EXPECT_EQ(c.getShape(), 2.0);
    EXPECT_EQ(c.getScale(), 5.0);
}

//==============================================================================
// DH2 S2: Uniform(-1e308, 1e308) is a valid instance whose width b - a overflows. sample()
// returned +inf (a + U (b - a)), getEntropy() +inf, the quantile +inf, the CDF 0 everywhere
// (1 / inf = 0 as the cached 1/width) and the log-density -inf. Everything width-dependent is
// now formed from the half-width b/2 - a/2, which cannot overflow; at ordinary widths the
// results are bit-identical (halving and doubling are exact).
//==============================================================================

TEST(ContractGates, DH2_S2_UniformWidthOverflow) {
    const UniformDistribution u(-1e308, 1e308);
    std::mt19937 rng(7);
    for (int i = 0; i < 1000; ++i) {
        const double x = u.sample(rng);
        ASSERT_TRUE(std::isfinite(x)) << i;
        ASSERT_GE(x, -1e308);
        ASSERT_LE(x, 1e308);
    }
    const auto xs = u.sample(rng, 1000);
    for (const double x : xs) {
        ASSERT_TRUE(std::isfinite(x));
        ASSERT_GE(x, -1e308);
        ASSERT_LE(x, 1e308);
    }
    EXPECT_NEAR(u.getEntropy(), 709.88935582272602, 1e-12);  // log(2e308), mpmath
    EXPECT_EQ(u.getCumulativeProbability(0.0), 0.5);
    EXPECT_EQ(u.getCumulativeProbability(5e307), 0.75);
    EXPECT_EQ(u.getQuantile(0.75), 5e307);
    EXPECT_EQ(u.getQuantile(0.5), 0.0);
    EXPECT_LT(relErr(u.getProbability(0.0), 5e-309), 1e-10);  // subnormal, representable
    EXPECT_NEAR(u.getLogProbability(0.0), -709.88935582272602, 1e-12);
    EXPECT_EQ(u.getMean(), 0.0);
    EXPECT_EQ(u.getMedian(), 0.0);
    // Batch CDF under every strategy: the linear form must not overflow at x - a.
    std::vector<double> in(69, 5e307), out(69, -1.0);
    in[1] = -1e308;
    in[2] = 1e308;
    using PS = detail::PerformanceHint::PreferredStrategy;
    for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL}) {
        u.getCumulativeProbability(std::span<const double>(in), std::span<double>(out),
                                   detail::PerformanceHint{s, std::nullopt});
        EXPECT_EQ(out[0], 0.75) << static_cast<int>(s);
        EXPECT_EQ(out[1], 0.0) << static_cast<int>(s);
        EXPECT_EQ(out[2], 1.0) << static_cast<int>(s);
        EXPECT_EQ(out[68], 0.75) << static_cast<int>(s);
        u.getLogProbability(std::span<const double>(in), std::span<double>(out),
                            detail::PerformanceHint{s, std::nullopt});
        EXPECT_NEAR(out[0], -709.88935582272602, 1e-12) << static_cast<int>(s);
    }
    // Ordinary widths: the half-width form reproduces the direct one exactly.
    const UniformDistribution o(-3.0, 7.5);
    EXPECT_EQ(o.getCumulativeProbability(1.2), (1.2 - (-3.0)) / 10.5);  // DH3 U1: a quotient
    // lo + p·w, as before the fix; the compiler may contract it to an FMA (AArch64 does).
    const double q = o.getQuantile(0.3);
    EXPECT_TRUE(q == -3.0 + 0.3 * 10.5 || q == std::fma(0.3, 10.5, -3.0)) << q;
    EXPECT_EQ(o.getEntropy(), std::log(10.5));
}

//==============================================================================
// #216 class (DH2): a CDF argument formed as x * rate, rate / x or x / x_m before the special
// function under- or overflowed where the mass is not small. Gamma(0.005, 0.5).cdf(5e-324)
// returned 0 (mass 0.0242; ChiSquared(0.01) is the same instance), Gamma(0.01, 0.01) 0 for
// 5.6e-4, InverseGamma(0.01, 1e-20).cdf(1e308) 1 for 0.99947 (beta / x underflowed to 0),
// Pareto(1e-6, 0.01).cdf(1e303) 1 for 0.99919 ((x - x_m) / x_m overflowed to +inf). The
// arguments are now taken in log space where they leave the normal range. References:
// mpmath, dps 50. Weibull, LogNormal, Exponential, Erlang, Rayleigh were checked and do not
// carry the pattern (log-space already, or the true mass is itself below DBL_MIN).
//==============================================================================

TEST(ContractGates, DH2_Issue216_CdfArgumentUnderOverflow) {
    constexpr double kDenormMin = std::numeric_limits<double>::denorm_min();
    constexpr double kMax = std::numeric_limits<double>::max();
    using PS = detail::PerformanceHint::PreferredStrategy;
    const auto batchAll = [](const auto& d, double x, double want, double tol, const char* who) {
        std::vector<double> xs(69, x), out(69, -1.0);
        xs[3] = 1.0;  // an ordinary element among the extreme ones
        for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL}) {
            d.getCumulativeProbability(std::span<const double>(xs), std::span<double>(out),
                                       detail::PerformanceHint{s, std::nullopt});
            EXPECT_LT(relErr(out[0], want), tol) << who << " strategy " << static_cast<int>(s);
            EXPECT_LT(relErr(out[68], want), tol) << who << " strategy " << static_cast<int>(s);
            // The ordinary element agrees with the scalar path to the vector kernels' ulp.
            EXPECT_LT(relErr(out[3], d.getCumulativeProbability(1.0)), 1e-14) << who;
        }
    };

    const GammaDistribution g1(0.005, 0.5);
    EXPECT_LT(relErr(g1.getCumulativeProbability(kDenormMin), 0.0241661948617129), 1e-13);
    batchAll(g1, kDenormMin, 0.0241661948617129, 1e-13, "Gamma(0.005,0.5)");
    const GammaDistribution g2(0.01, 0.01);
    EXPECT_LT(relErr(g2.getCumulativeProbability(kDenormMin), 5.6157674581888378e-4), 1e-13);
    const GammaDistribution g3(0.5, 0.5);
    EXPECT_LT(relErr(g3.getCumulativeProbability(kDenormMin), 1.7735048886036273e-162), 1e-13);
    EXPECT_LT(relErr(g3.getCumulativeProbability(1e-300), 7.9788456080286536e-151), 1e-13);
    const GammaDistribution g4(1e-3, 1e-3);
    EXPECT_LT(relErr(g4.getCumulativeProbability(1e-310), 0.48668764850593823), 1e-13);
    EXPECT_LT(relErr(ChiSquaredDistribution(0.01).getCumulativeProbability(kDenormMin),
                     0.0241661948617129),
              1e-13);
    // Unchanged where the argument is normal: rate 1 at the same x was already right.
    EXPECT_LT(relErr(GammaDistribution(0.005, 1.0).getCumulativeProbability(kDenormMin),
                     0.024250093812704406),
              1e-12);

    const InverseGammaDistribution ig(0.01, 1e-20);
    EXPECT_LT(relErr(ig.getCumulativeProbability(1e308), 0.9994721977110211), 1e-13);
    EXPECT_LT(relErr(ig.getSurvivalProbability(1e308), 5.2780228897890254e-4), 1e-13);
    batchAll(ig, 1e308, 0.9994721977110211, 1e-13, "InverseGamma(0.01,1e-20)");

    const ParetoDistribution pa(1e-6, 0.01);
    EXPECT_LT(relErr(pa.getCumulativeProbability(1e303), 0.9991871694838359), 1e-13);
    EXPECT_LT(relErr(pa.getCumulativeProbability(kMax), 0.99927980045211548), 1e-13);
    EXPECT_LT(relErr(pa.getSurvival(kMax), 7.2019954788452102e-4), 1e-13);
    batchAll(pa, 1e303, 0.9991871694838359, 1e-13, "Pareto(1e-6,0.01)");
    batchAll(pa, kMax, 0.99927980045211548, 1e-13, "Pareto(1e-6,0.01)");
}

//==============================================================================
// DH3 contract round, 2026-10-10. References: mpmath, dps 50-120 (the truncated-normal moments
// by tn_moments in the hunt oracle). Each gate failed on the unfixed code, as noted per block.
//==============================================================================

namespace {

// Every batch strategy's result for one repeated input, beside the scalar call. 69 elements:
// past the SIMD threshold, with a scalar tail.
template <typename D, typename Op>
void expectAllStrategies(const D& d, double x, double want, double tol, Op op, const char* who) {
    using PS = detail::PerformanceHint::PreferredStrategy;
    std::vector<double> in(69, x), out(69, -1.0);
    for (PS s :
         {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL, PS::MAXIMIZE_THROUGHPUT}) {
        op(d, std::span<const double>(in), std::span<double>(out),
           detail::PerformanceHint{s, std::nullopt});
        for (std::size_t i : {std::size_t{0}, std::size_t{68}}) {
            if (tol == 0.0 || std::isinf(want)) {
                EXPECT_EQ(out[i], want) << who << " strategy " << static_cast<int>(s);
            } else {
                EXPECT_LT(relErr(out[i], want), tol) << who << " strategy " << static_cast<int>(s);
            }
        }
    }
}

const auto kCdfBatch = [](const auto& d, std::span<const double> in, std::span<double> out,
                          const detail::PerformanceHint& h) {
    d.getCumulativeProbability(in, out, h);
};
const auto kPdfBatch = [](const auto& d, std::span<const double> in, std::span<double> out,
                          const detail::PerformanceHint& h) { d.getProbability(in, out, h); };
const auto kLogPdfBatch = [](const auto& d, std::span<const double> in, std::span<double> out,
                             const detail::PerformanceHint& h) { d.getLogProbability(in, out, h); };

}  // namespace

// DH3 U1: Uniform at extreme and near-unit bounds. Unfixed: U(-DBL_MAX, -1e308).getMean() =
// -inf (a + b overflowed with a finite width); U(0, 3e154).getVariance() = inf for 7.5e307
// ((b - a)^2 overflowed); F(b) < 1 wherever w * fl(1/w) != 1: 0.99999999999999989 for
// U(-DBL_MAX, 0) (1/w subnormal) and for U(0, 49); U(0, 1e-320).cdf(5e-321) = +inf and the
// survival -inf (1/w = inf); and the unit-interval fast path took U(0, 1 + 1e-9) and
// U(1e-9, 1) for [0, 1] at tolerance 1e-8: F(b) = 1.000000001, F(a) = 1e-9, density 1, while
// the batch kernels used the true bounds.
TEST(ContractGates, DH3_U1_UniformExtremeBounds) {
    constexpr double kMax = std::numeric_limits<double>::max();
    const UniformDistribution m(-kMax, -1e308);
    EXPECT_EQ(m.getMean(), -1.398846567431158e308);  // (a + b)/2, as getMedian forms it
    EXPECT_EQ(m.getMean(), m.getMedian());
    EXPECT_EQ(m.getMidpoint(), m.getMedian());
    EXPECT_EQ(m.getVariance(), std::numeric_limits<double>::infinity());  // 5.3e614
    EXPECT_LT(relErr(UniformDistribution(0.0, 3e154).getVariance(), 7.5e307), 1e-15);
    EXPECT_EQ(UniformDistribution(-1e300, -9e299).getVariance(), (1e299 * 1e299) / 12.0);

    struct Row {
        double a, b, x, f;
    };
    for (const Row& r : {Row{-kMax, 0.0, 0.0, 1.0}, Row{-kMax, 0.0, -kMax, 0.0},
                         Row{-8e307, 8e307, 8e307, 1.0}, Row{-8e307, 8e307, 4e307, 0.75},
                         Row{0.0, 1e-320, 5e-321, 0.5}, Row{0.0, 1e-320, 1e-320, 1.0},
                         Row{0.0, 49.0, 49.0, 1.0}, Row{0.0, 1.0 + 1e-9, 1.0 + 1e-9, 1.0},
                         Row{1e-9, 1.0, 1e-9, 0.0}, Row{-3.0, 7.5, 1.2, 4.2 / 10.5}}) {
        const UniformDistribution u(r.a, r.b);
        EXPECT_EQ(u.getCumulativeProbability(r.x), r.f) << r.a << " " << r.b << " " << r.x;
        EXPECT_EQ(u.getSurvival(r.x), 1.0 - r.f) << r.a << " " << r.b << " " << r.x;
        expectAllStrategies(u, r.x, r.f, 0.0, kCdfBatch, "Uniform cdf");
    }
    const UniformDistribution nu(1e-9, 1.0);
    const double dens = 1.0 / (1.0 - 1e-9);
    EXPECT_EQ(nu.getProbability(0.5), dens);
    EXPECT_EQ(nu.getLogProbability(0.5), -std::log(1.0 - 1e-9));
    expectAllStrategies(nu, 0.5, dens, 0.0, kPdfBatch, "Uniform(1e-9, 1) pdf");
}

// DH3 T1: TruncatedNormal pdf/logpdf where sigma^2 over- or underflows (both sigma accepted).
// Unfixed: -(x - mu)^2 / (2 sigma^2) gave -0 * inf or 0 * (-inf), NaN under every strategy
// (TN(1e300, 1e299, 1.1e300, 1.5e300).pdf(1.2e300), TN(0, 1e-170, -1e-170, 1e-170).pdf(0)); the
// variance sigma^2 * c2 was +inf for a finite 7e306 at sigma = 1e160.
TEST(ContractGates, DH3_T1_TruncatedNormalSigmaSquaredRange) {
    struct Row {
        double mu, sigma, a, b, x, logpdf;
    };
    for (const Row& r : {Row{1e300, 1e299, 1.1e300, 1.5e300, 1.2e300, -689.5508578866559},
                         Row{0.0, 1e-170, -1e-170, 1e-170, 0.0, 390.90224242208524},
                         Row{0.0, 1e-170, -1e-170, 1e-170, 5e-171, 390.77724242208524}}) {
        const TruncatedNormalDistribution t(r.mu, r.sigma, r.a, r.b);
        EXPECT_LT(std::fabs(t.getLogProbability(r.x) - r.logpdf), 1e-12 * std::fabs(r.logpdf))
            << r.sigma;
        EXPECT_LT(relErr(t.getProbability(r.x), std::exp(r.logpdf)), 1e-12) << r.sigma;
        expectAllStrategies(t, r.x, r.logpdf, 1e-14, kLogPdfBatch, "TN logpdf");
        expectAllStrategies(t, r.x, std::exp(r.logpdf), 1e-12, kPdfBatch, "TN pdf");
    }
    // sigma^2 = 1e320 overflows; c2 = 8.3e-14 for the window (0, 1e-6 sigma).
    const TruncatedNormalDistribution w(0.0, 1e160, 0.0, 1e154);
    EXPECT_LT(relErr(w.getVariance(), 8.333333333333055e306), 1e-12);
}

// DH3 G1: Gamma density where the rate-scaled argument beta * x overflows at shapes >= 20.
// Unfixed: Gamma(100, 1e300).pdf(1e300) formed (inf - alpha)/alpha and log(inf) - inf: NaN
// under every strategy (shape 5 was already 0). Gamma(1.5e308, 2) at 1e308 has a finite
// log-density, -6.85e306, also NaN.
TEST(ContractGates, DH3_G1_GammaDensityRateOverflow) {
    const GammaDistribution g(100.0, 1e300);
    EXPECT_EQ(g.getProbability(1e300), 0.0);
    EXPECT_EQ(g.getLogProbability(1e300), -std::numeric_limits<double>::infinity());
    expectAllStrategies(g, 1e300, 0.0, 0.0, kPdfBatch, "Gamma(100, 1e300) pdf");
    expectAllStrategies(g, 1e300, -std::numeric_limits<double>::infinity(), 0.0, kLogPdfBatch,
                        "Gamma(100, 1e300) logpdf");
    const GammaDistribution h(1.5e308, 2.0);
    EXPECT_LT(relErr(h.getLogProbability(1e308), -6.847689132232862e306), 1e-13);
    expectAllStrategies(h, 1e308, -6.847689132232862e306, 1e-13, kLogPdfBatch,
                        "Gamma(1.5e308, 2) logpdf");
}

// DH3 W1: Weibull moments where Gamma(1 + n/k) overflows or the scale's powers do. Unfixed:
// the variance inf - inf = NaN from k = 0.0117 down (Weibull(0.01, 1e-100) is 7.9e174), the
// skewness and kurtosis NaN at (0.02, 1), (0.01, 1e-100) and (2, 1e200) (Gamma(1 + n/k),
// lambda^n or the variance overflowed), the kurtosis -3 at (0.05, 1e-100) and NaN at
// (0.5, 1e-100) (lambda^4 = 0). Where the true value overflows (+inf rows) +inf is right.
TEST(ContractGates, DH3_W1_WeibullMomentOverflow) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    const WeibullDistribution a(0.01, 1e-100);
    EXPECT_LT(relErr(a.getVariance(), 7.886578673647731e174), 1e-12);
    EXPECT_LT(relErr(a.getMean(), 9.332621544394325e57), 1e-12);
    EXPECT_LT(relErr(a.getSkewness(), 1.3818792121546423e52), 1e-12);
    EXPECT_LT(relErr(a.getKurtosis(), 1.0295250013541385e119), 1e-12);
    EXPECT_EQ(WeibullDistribution(0.01, 1.0).getVariance(), kInf);  // 7.9e374
    const WeibullDistribution b(0.003, 1e-100);
    EXPECT_EQ(b.getMean(), kInf);  // 7.2e597
    EXPECT_EQ(b.getVariance(), kInf);
    EXPECT_LT(relErr(b.getSkewness(), 1.8781803558133598e175), 1e-12);
    EXPECT_EQ(b.getKurtosis(), kInf);  // 5.2e399
    const WeibullDistribution c(0.02, 1.0);
    EXPECT_LT(relErr(c.getSkewness(), 6.337061601553519e25), 1e-12);
    EXPECT_LT(relErr(c.getKurtosis(), 9.054851465610302e58), 1e-12);
    EXPECT_LT(relErr(WeibullDistribution(0.05, 1e-100).getKurtosis(), 1.0750720873489566e23),
              1e-12);
    const WeibullDistribution d(2.0, 1e200);
    EXPECT_EQ(d.getVariance(), kInf);  // 2.1e399
    EXPECT_LT(relErr(d.getSkewness(), 0.6311106578189372), 1e-13);
    EXPECT_LT(relErr(d.getKurtosis(), 0.24508930068763807), 1e-13);
    EXPECT_LT(relErr(WeibullDistribution(0.5, 1e-100).getKurtosis(), 84.72), 1e-13);
}

// DH3 G2: Gamma entropy below the Stirling switch took psi from detail::digamma, whose
// asymptotic series stops at 1/(252 x^6) after lifting x past 6: 1.2e-8 absolute at alpha = 6,
// 6.3e-10 at 0.5 (relative 6.9e-9).
TEST(ContractGates, DH3_G2_GammaEntropySmallShape) {
    struct Row {
        double alpha, h;
    };
    for (const Row& r : {Row{0.5, 0.09060992991398835}, Row{3.0, 1.8475785103630111},
                         Row{5.6, 2.218086627766992}, Row{6.0, 2.2569034006230435},
                         Row{20.0, 2.899928334598662}, Row{50.0, 3.3682499483781685}}) {
        // From alpha = 20 the direct form's own alpha log alpha eps cancellation dominates.
        const double tol = r.alpha < 10.0 ? 2e-14 : 6e-14;
        EXPECT_NEAR(GammaDistribution(r.alpha, 1.0).getEntropy(), r.h, tol) << r.alpha;
        EXPECT_NEAR(GammaDistribution(r.alpha, 4.0).getEntropy(), r.h - std::log(4.0), tol)
            << r.alpha;
    }
}

// DH3 T2/T3: TruncatedNormal on narrow windows and just past the closed-form switch. Unfixed:
// log Z from the erfc difference Q(alpha) - Q(beta) lost 1/(alpha * width) of its arguments'
// rounding, so the entropy was off by 7.4e-10 at (20, 20 + 1.41e-6) and 1.3e-9 at
// (37, 37 + 1e-6), and the log-density by as much; the kurtosis took the closed forms from
// c2 = 0.05 max(1, m^2), 8e-13 absolute at (-1.5, -0.5513).
TEST(ContractGates, DH3_T2_TruncatedNormalNarrowWindowsAndKurtosis) {
    struct Row {
        double a, b, h, kurt, logpdf_mid;
    };
    for (const Row& r :
         {Row{20.0, 20.0 + 1.41e-6, -13.471920854015579, -1.1999999998681492, 13.471920853949428},
          Row{37.0, 37.000001, -13.815510560546073, -1.1999999997731028, 13.815510560432031},
          Row{-3.0, -2.999999, -13.81551055782487, -1.1999999999984743, 13.815510557824163},
          Row{0.0, 1e-6, -13.815510557964274, -1.1999999999999658, 13.815510557964316},
          Row{-1.5, -0.5513167019494862, -0.08954339525050585, -1.0235359908519184,
              0.05162480839418757},
          Row{1.182179, 2.924669, 0.20611090042918168, 0.4103303796461589, -0.8802914185414576},
          Row{-3.0, -1.23396, 0.19623333155808514, 0.5233648939064417, -0.927231948877003}}) {
        const TruncatedNormalDistribution t(0.0, 1.0, r.a, r.b);
        EXPECT_NEAR(t.getEntropy(), r.h, 1e-14) << r.a << " " << r.b;
        EXPECT_NEAR(t.getKurtosis(), r.kurt, 3e-14) << r.a << " " << r.b;
        EXPECT_NEAR(t.getLogProbability(0.5 * (r.a + r.b)), r.logpdf_mid, 1e-13)
            << r.a << " " << r.b;
        // sigma = 4 (exact scaling of the bounds) shifts the entropy by log 4.
        const TruncatedNormalDistribution s(0.0, 4.0, 4.0 * r.a, 4.0 * r.b);
        EXPECT_NEAR(s.getEntropy(), r.h + std::log(4.0), 1e-14) << r.a << " " << r.b;
    }
}

// DH3: detail::digamma lifted x past 6 and stopped its series at 1/(252x⁶) (truncation
// 1/(240x⁸) = 2.5e-9 at 6; ψ(1) was −0.57721566727); detail::trigamma stopped at 1/(30x⁹)
// (2.1e-10 at 6). Their consumers (Gamma, InverseGamma, Beta, Fisher-F, Student-t entropies;
// the Gamma, Beta, Student-t and NegativeBinomial fits) carried those errors. Both now lift
// past 10 and run through B₁₄ (first omitted term < 1e-16). References: mpmath, dps 40, at
// the doubles shown; rows straddle the old (6) and new (10) shift points.
TEST(ContractGates, DH3_D1_DigammaTrigammaSeries) {
    struct Row {
        double x, psi, psi1;
    };
    const Row rows[] = {
        {0.001, -1000.5755719318102797, 1000001.6425331958273},
        {0.5, -1.9635100260214234794, 4.9348022005446793094},
        {1.0, -0.57721566490153286061, 1.6449340668482264365},
        {2.5, 0.70315664064524318723, 0.49035775610023486497},
        {5.5, 1.6110931485817511237, 0.19934238698962765913},
        {5.999999999999999, 1.7061176684318003117, 0.18132295573711535448},
        {6.0, 1.7061176684318004727, 0.18132295573711532536},
        {9.5, 2.1977378764029495332, 0.11099728846909903237},
        {9.999999999999998, 2.2517525890667209208, 0.10516633568168576575},
        {10.0, 2.2517525890667211076, 0.10516633568168574612},
        {37.25, 3.6041690730056271675, 0.0272092058039555752},
        {1000.0, 6.9072551956488120521, 0.0010005001666666333334},
    };
    constexpr double kEps = std::numeric_limits<double>::epsilon();
    for (const auto& r : rows) {
        // 16ε of max(1, |ψ|): the recurrence sums up to ~10 terms of size 1/x.
        EXPECT_NEAR(detail::digamma(r.x), r.psi, 16.0 * kEps * std::max(1.0, std::abs(r.psi)))
            << "digamma(" << r.x << ")";
        EXPECT_NEAR(detail::trigamma(r.x), r.psi1, 16.0 * kEps * r.psi1)
            << "trigamma(" << r.x << ")";
    }
}

//==============================================================================
// Hunt-3 continuation, 2026-10-11. References: mpmath, dps 60 (von Mises by log-concave tail
// quadrature of e^{−2κ sin²(s/2)}; the truncated normal and Cauchy in closed form). Each gate
// failed on the unfixed code as noted.
//==============================================================================

// von Mises above κ = 1000 took a wrapped normal in t, 0.5·(1 + erf(z/√2)): 0 for every
// F ≲ 1e-20 (κ = 1e8 at F = 1e-20 read 0, at μ = 0, 2.5, −3, scalar and batch) and ~0.04/κ
// absolute centrally (κ = 1000 + ulp: 2e-5), while the quantile solved the exact law, so
// cdf(quantile(p)) missed p by 7e-2 at p = 1e-10 and returned 0 at 1e-20. The exact masses now
// run to κ = 1e20 and the normal limit in w = 2√κ·sin(t/2) beyond; both sides agree. At
// κ ≥ 9e307 the quantile overflowed 2κ and returned −π for every p ≤ 0.1.
TEST(ContractGates, H3_VonMisesCdfAboveKappa1000) {
    struct Row {
        double mu, kappa, x, want;
    };
    const double k1 = std::nextafter(1000.0, 2000.0);
    const Row rows[] = {
        {0.0, 1e8, -0.0009262340432470076, 9.9999999999999305692e-21},
        {2.5, 1e8, 2.499073765956753, 9.9999999998109139939e-21},
        {-3.0, 1e8, -3.000926234043247, 9.9999999998109139939e-21},
        {0.0, k1, -0.03162277660168379, 0.15869560139334777757},
        {0.0, k1, -0.09486832980505137, 0.0013565622038255720177},
        {0.0, k1, -0.5, 1.7917438606926490745e-55},
    };
    for (const auto& r : rows) {
        const VonMisesDistribution d(r.mu, r.kappa);
        EXPECT_LT(relErr(d.getCumulativeProbability(r.x), r.want), 1e-13)
            << "mu " << r.mu << " kappa " << r.kappa << " x " << r.x;
        expectAllStrategies(d, r.x, r.want, 1e-13, kCdfBatch, "von Mises CDF");
    }
    // Round trips, with the condition number w² ≤ 1400 of the deepest p: 1e-12.
    for (double kappa : {k1, 1e4, 1e8, 1e16, 1e19, 1e21, 1e100, 1.7e308}) {
        const VonMisesDistribution d(0.0, kappa);
        for (double p : {1e-300, 1e-20, 1e-10, 0.1, 0.3}) {
            const double q = d.getQuantile(p);
            EXPECT_LT(q, 0.0) << kappa << " " << p;
            EXPECT_LT(relErr(d.getCumulativeProbability(q), p), 1e-12) << kappa << " " << p;
            if (p >= 0.1) {  // 1 - p exact enough to mirror
                const double qu = d.getQuantile(1.0 - p);
                EXPECT_NEAR(qu, -q, 1e-14 * std::fabs(q)) << kappa << " " << p;
            }
        }
    }
}

// TruncatedNormal on a left-half-infinite window formed F = Φ(ξ)/Φ(β) from ½·erfc(−ξ/√2),
// subnormal from ξ = −37.5: TN(2, 1, −∞, −28).cdf(−37.296) = 0 for 1e-140, at −36.107 off by
// 4.5e-8. And the erfc argument u·INV_SQRT_2 (one ulp below 1/√2, then rounded) put a constant
// 1.18e-13 relative error into Z = Q(α) at α = 30 and 9.8e-14 at α = 37, read straight through
// near the bound: TN(0, 0.5, 15, ∞) and TN(0, 1e-3, 0.037, ∞) there; 1.7e-13 at (2, 1, −∞, −28)
// in the bulk. The arguments are now carried to two doubles and the deep left tail is a ratio
// of scaled tails.
TEST(ContractGates, H3_TruncatedNormalHalfInfiniteWindows) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    struct Row {
        double mean, sd, a, b, x, want, tol;
    };
    const Row rows[] = {
        {2.0, 1.0, -kInf, -28.0, -37.296118031184605, 1.0000000000002739413e-140, 1e-12},
        {2.0, 1.0, -kInf, -28.0, -36.10699230682169, 1.0000000000000115448e-120, 1e-12},
        {2.0, 1.0, -kInf, -28.0, -28.01, 0.74053488116785746286, 2e-15},
        {0.0, 0.5, 15.0, kInf, 15.0000000000005, 2.998257985503900142e-11, 2e-14},
        {0.0, 1e-3, 0.037, kInf, 0.037000000000001, 3.6997392834801391916e-11, 2e-14},
    };
    for (const auto& r : rows) {
        const TruncatedNormalDistribution d(r.mean, r.sd, r.a, r.b);
        EXPECT_LT(relErr(d.getCumulativeProbability(r.x), r.want), r.tol)
            << r.mean << " " << r.sd << " " << r.a << " " << r.b << " x " << r.x;
        expectAllStrategies(d, r.x, r.want, r.tol, kCdfBatch, "truncated normal CDF");
    }
}

// TruncatedNormal inherited S = 1 − F, so the upper tail was capped by F's ulp: 0 for S below
// ~1e-17 (TN(0, 1, 30, ∞).S(38) = 5.9e-119, TN(0, 1, 30, 40).S(39.9) = 4e-151, TN(0, 1, −2, 2)
// one ulp below b = 1.3e-17), a few bits above it (TN(0, 1, −∞, 1) one ulp below b: 1.1e-16
// for 3.2e-17; TN(0, 1, −2, 2).S(1.9999999999) 1.3e-6 off). S is now the CDF kernel on the
// reflected window (TN(0, 1, 37, 37 + 1e-6) one ulp below b: 5.6e-5 off). Two CDF defects
// the reflection exposed are fixed with it: a same-tail window inside the near-bound band took
// Z from an erfc difference good to Q(α)/Z of its terms (F 6e-11 off mid-window at (0, 1e-6)),
// and the band scale φ(α)/Z underflowed past |α| = 38.5 though Z did not
// (TN(0, 1, −40, −30).F(−39.999) = 0 for 3e-154; S(39.999) on (30, 40) likewise). Expected
// values: mpmath, dps 80, regime-split erfc pieces.
TEST(ContractGates, A11_TruncatedNormalSurvivalUpperTail) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kEps = std::numeric_limits<double>::epsilon();
    struct Row {
        double mean, sd, a, b, x, want;
    };
    const Row rows[] = {
        // Right-half-infinite, deep in the tail.
        {0.0, 1.0, 30.0, kInf, 38.0, 5.88057181019683653519e-119},
        {0.0, 0.5, 15.0, kInf, 19.0, 5.88057181019683653519e-119},
        // Right-side window: bulk, past erfc underflow, and the band at b where φ(β) underflows.
        {0.0, 1.0, 30.0, 40.0, 35.0, 2.29259484692683400263e-71},
        {0.0, 1.0, 30.0, 40.0, 39.9, 3.98333273030939322093e-151},
        {0.0, 1.0, 30.0, 40.0, 39.999, 3.04262601322290708215e-154},
        {0.0, 1.0, 37.0, 38.0, 37.999, 1.95322809323307731838e-18},
        // Narrow windows.
        {0.0, 1.0, 0.0, 1e-6, 9.999999999999997e-07, 2.11758236813504508271e-16},
        {0.0, 1.0, 37.0, 37.000001, 37.00000099999999, 7.105295925943542207e-9},
        {0.0, 1.0, 37.0, 37.000001, 37.0000005, 4.99995374999949309008e-1},
        // Central window, one ulp and 1e-10 below b.
        {0.0, 1.0, -2.0, 2.0, 1.9999999999999998, 1.25598807160274718619e-17},
        {0.0, 1.0, -2.0, 2.0, 1.9999999999, 5.65646787983577780651e-12},
        // Left-half-infinite, one ulp below b.
        {0.0, 1.0, -kInf, 1.0, 0.9999999999999999, 3.19300109618201740862e-17},
        // S near 1 (1 − F was already right here; the complement must stay so).
        {0.0, 1.0, -3.0, 3.0, -2.9999999999, 9.99999999999555615374e-1},
        {0.0, 1.0, 30.0, kInf, 30.0000000001, 9.99999996996727139051e-1},
        {0.0, 1.0, -kInf, -30.0, -30.1, 9.50625464334053235438e-1},
    };
    for (const auto& r : rows) {
        const TruncatedNormalDistribution d(r.mean, r.sd, r.a, r.b);
        const DistributionBase& db = d;
        const double s = db.getSurvival(r.x);
        // |ln S|·ε-class: four times the law.
        const double tol = 4.0 * kEps * (1.0 + std::fabs(std::log(r.want)));
        EXPECT_LT(relErr(s, r.want), tol)
            << r.mean << " " << r.sd << " " << r.a << " " << r.b << " x " << r.x << " S " << s;
        EXPECT_LE(std::fabs((s + d.getCumulativeProbability(r.x)) - 1.0), 4.0 * kEps) << r.x;
    }

    // The lower band of a left-tail window, where φ(α) underflows: the CDF, every strategy.
    const TruncatedNormalDistribution left(0.0, 1.0, -40.0, -30.0);
    EXPECT_LT(relErr(left.getCumulativeProbability(-39.999), 3.04262601322290708215e-154), 1e-13);
    expectAllStrategies(left, -39.999, 3.04262601322290708215e-154, 1e-13, kCdfBatch,
                        "truncated normal CDF");

    // A finite hazard deep in the tail: h(x) > ξ (Mills ratio) at TN(0, 1, 30, ∞), x = 38.
    const TruncatedNormalDistribution rh(0.0, 1.0, 30.0, kInf);
    const DistributionBase& rhb = rh;
    EXPECT_TRUE(std::isfinite(rhb.getHazard(38.0)));
    EXPECT_GT(rhb.getHazard(38.0), 38.0);

    // Edges: 1 at and below a, 0 at and above b, ±∞, NaN.
    const TruncatedNormalDistribution c(0.0, 1.0, -2.0, 2.0);
    const DistributionBase& cb = c;
    EXPECT_EQ(cb.getSurvival(-2.0), 1.0);
    EXPECT_EQ(cb.getSurvival(-3.0), 1.0);
    EXPECT_EQ(cb.getSurvival(2.0), 0.0);
    EXPECT_EQ(cb.getSurvival(3.0), 0.0);
    EXPECT_EQ(cb.getSurvival(-kInf), 1.0);
    EXPECT_EQ(cb.getSurvival(kInf), 0.0);
    EXPECT_TRUE(std::isnan(cb.getSurvival(kNaN)));
    EXPECT_EQ(rhb.getSurvival(kInf), 0.0);
    EXPECT_EQ(rhb.getSurvival(30.0), 1.0);

    // Monotone, bit-stepped across the switches of the deep-tail windows: the band edge at b
    // (and its sliver), the scaled-ratio switch where Q(ξ) leaves the normal range
    // (ξ = 37.629), and x = μ + σ·37.629 on the half-infinite window.
    struct Mono {
        double mean, sd, a, b, x;
    };
    const Mono mono[] = {
        {0.0, 1.0, 30.0, 40.0, 39.99375},
        {0.0, 1.0, 30.0, 40.0, 37.629223185902724},
        {0.0, 1.0, 37.0, 38.0, 37.99342105263158},
        {0.0, 1.0, 37.0, 38.0, 37.629223185902724},
        {0.0, 1.0, 30.0, kInf, 37.629223185902724},
        {0.0, 0.5, 15.0, kInf, 18.814611592951362},
        {0.0, 1.0, 0.0, 1e-6, 1e-6},
    };
    for (const auto& m : mono) {
        const TruncatedNormalDistribution d(m.mean, m.sd, m.a, m.b);
        double x = m.x;
        for (int i = 0; i < 2000; ++i)
            x = std::nextafter(x, -kInf);
        double prev = d.getSurvival(x);
        for (int i = 0; i < 4000; ++i) {
            x = std::nextafter(x, kInf);
            const double s = d.getSurvival(x);
            ASSERT_LE(s, prev) << m.a << " " << m.b << " x " << x;
            prev = s;
        }
    }
}

// Cauchy's density was Student-t(1)'s exp(ln c − log1p(z²)) times 1/γ: below DBL_MIN·γ the
// product went through a subnormal and kept its few bits (Cauchy(0, 1e-6) at ∓3.18e153:
// 3.141763e-314 for 3.141593e-314, 5.4e-5), and the exponential carried |ln f|·ε elsewhere
// (3.7e-14 at x = 1e94). It is now c/(1 + z²), c = 1/(πγ), rounded once.
TEST(ContractGates, H3_CauchyDensityIntoSubnormals) {
    const CauchyDistribution d(0.0, 1e-6);
    for (double sgn : {-1.0, 1.0}) {
        const double x = sgn * 3.1830988618379068e+153;
        // Half an ulp of 3.1e-314 is 7.9e-11 of it.
        EXPECT_LT(relErr(d.getProbability(x), 3.1415926535897930054e-314), 1e-10) << x;
        expectAllStrategies(d, x, 3.1415926535897930054e-314, 1e-10, kPdfBatch, "Cauchy PDF");
        EXPECT_LT(relErr(d.getProbability(sgn * 1e94), 3.1830988618379064426e-195), 1e-15);
        expectAllStrategies(d, sgn * 1e94, 3.1830988618379064426e-195, 1e-15, kPdfBatch,
                            "Cauchy PDF");
    }
}

// DH3: six derived constants were 1 ulp off because `1.0 / X` and `PI / k` round twice
// (INV_SQRT_2 low by 1 ulp put ~z²·ulp into every erfc(z·INV_SQRT_2): 1.4e-13 at z = 30).
// Every derived constant must equal the correctly rounded value (mpmath, dps 50).
TEST(ContractGates, DH3_C1_DerivedConstantsCorrectlyRounded) {
    struct Row {
        const char* name;
        double value, expected;
    };
    const Row rows[] = {
        {"ONE_THIRD", detail::ONE_THIRD, 0x1.5555555555555p-2},
        {"ONE_SIXTH", detail::ONE_SIXTH, 0x1.5555555555555p-3},
        {"ONE_TWELFTH", detail::ONE_TWELFTH, 0x1.5555555555555p-4},
        {"E_INV", detail::E_INV, 0x1.78b56362cef38p-2},
        {"PHI_INV", detail::PHI_INV, 0x1.3c6ef372fe950p-1},
        {"PI_INV", detail::PI_INV, 0x1.45f306dc9c883p-2},
        {"INV_LN2", detail::INV_LN2, 0x1.71547652b82fep+0},
        {"INV_SQRT_2PI", detail::INV_SQRT_2PI, 0x1.9884533d43651p-2},
        {"INV_SQRT_2", detail::INV_SQRT_2, 0x1.6a09e667f3bcdp-1},
        {"INV_SQRT_3", detail::INV_SQRT_3, 0x1.279a74590331cp-1},
        {"TWO_PI", detail::TWO_PI, 0x1.921fb54442d18p+2},
        {"PI_OVER_2", detail::PI_OVER_2, 0x1.921fb54442d18p+0},
        {"PI_OVER_3", detail::PI_OVER_3, 0x1.0c152382d7366p+0},
        {"PI_OVER_4", detail::PI_OVER_4, 0x1.921fb54442d18p-1},
        {"PI_OVER_6", detail::PI_OVER_6, 0x1.0c152382d7366p-1},
        {"THREE_PI_OVER_2", detail::THREE_PI_OVER_2, 0x1.2d97c7f3321d2p+2},
        {"FOUR_PI", detail::FOUR_PI, 0x1.921fb54442d18p+3},
        {"NEG_HALF_LN_2PI", detail::NEG_HALF_LN_2PI, -0x1.d67f1c864beb5p-1},
    };
    for (const auto& r : rows)
        EXPECT_EQ(r.value, r.expected) << r.name;
}

// DH3: Poisson's CDF and survival rounded x to the nearest count, so Poisson(3).cdf(2.6)
// returned F(3) = 0.647 where a right-continuous step function gives F(2) = 0.423, as
// Binomial, NegativeBinomial and Geometric do. They floor now; the PMF's documented rounding
// is unchanged.
TEST(ContractGates, DH3_P1_PoissonCdfFloorsNonIntegerX) {
    const auto d = PoissonDistribution::create(3.0).unwrap();
    for (double x : {2.0, 2.4, 2.5, 2.6, 2.999999999999999}) {
        EXPECT_EQ(d.getCumulativeProbability(x), d.getCumulativeProbability(2.0)) << x;
        EXPECT_EQ(d.getSurvival(x), d.getSurvival(2.0)) << x;
    }
    EXPECT_NE(d.getCumulativeProbability(3.0), d.getCumulativeProbability(2.0));
    EXPECT_EQ(d.getCumulativeProbability(0.7), d.getCumulativeProbability(0.0));

    const std::vector<double> in = {2.0, 2.5, 2.6, 2.999999999999999, 3.0, 0.7};
    std::vector<double> out(in.size());
    using PS = detail::PerformanceHint::PreferredStrategy;
    for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL}) {
        d.getCumulativeProbability(std::span<const double>(in), std::span<double>(out),
                                   detail::PerformanceHint{s, std::nullopt});
        for (size_t i = 0; i < in.size(); ++i)
            EXPECT_EQ(out[i], d.getCumulativeProbability(in[i]))
                << "x = " << in[i] << ", strategy " << static_cast<int>(s);
    }
}

// InverseGamma at a huge shape took β/x rounded, and the shape amplifies that rounding:
// d log f / d log(β/x) = α − β/x, ~3e9 at α = 1e16 thirty standard deviations out. Half an ulp of
// β/x moved the density 1.6e-7, the CDF 7.4e-8 and the survival function 1.5e-7 relative, though
// x is exact (Gamma(1e16, 1) at the mirrored points is clean: there βx is x). The roundings are
// now carried to first order. β = 3 also takes the delegate's β·(1/x) rounding, so it guards the
// log-density correction's coupling to that product (the last row). References: mpmath at 50
// digits (the last row 60); the 1e-12 leaves room for the incomplete gamma's own error at this
// shape (1.5e-13 at an exact argument).
TEST(ContractGates, H4_InverseGammaReciprocalRoundingAtHugeShape) {
    enum class Op { Pdf, LogPdf, Cdf, Sf };
    struct Row {
        double beta, x;
        Op op;
        double want, tol;
    };
    const Row rows[] = {
        {1.0, 9.999996688805569e-17, Op::Pdf, 3.3142420801521426481e-215, 1e-12},
        {1.0, 9.999999073766057e-17, Op::LogPdf, 11.447632596656798001, 1e-14},
        {1.0, 1.0000003e-16, Op::LogPdf, -395.65671674644297681, 1e-14},
        {1.0, 9.999999073766057e-17, Op::Cdf, 1.0000007525316864383e-20, 1e-12},
        {1.0, 1.0000003e-16, Op::Sf, 4.9075979729434451753e-198, 1e-12},
        {3.0, 2.9999997221298169e-16, Op::LogPdf, 10.3490202699329653, 1e-14},
        {3.0, 2.9999997221298169e-16, Op::Cdf, 1.0000007140421337226e-20, 1e-12},
        // The delegate's product β·fl(1/x) rounds here (by 1e-16 relative; at the β = 3 row
        // above it is exact), so this row fails if the correction and Gamma's own product
        // correction (H5) both carry that rounding, or neither does.
        {3.0, 3.000000900000288e-16, Op::LogPdf, -396.75561709247441498, 1e-14},
    };
    for (const auto& r : rows) {
        const InverseGammaDistribution d(1e16, r.beta);
        switch (r.op) {
            case Op::Pdf:
                EXPECT_LT(relErr(d.getProbability(r.x), r.want), r.tol) << r.beta << " " << r.x;
                expectAllStrategies(d, r.x, r.want, r.tol, kPdfBatch, "inverse gamma pdf");
                break;
            case Op::LogPdf:
                EXPECT_LT(relErr(d.getLogProbability(r.x), r.want), r.tol) << r.beta << " " << r.x;
                expectAllStrategies(d, r.x, r.want, r.tol, kLogPdfBatch, "inverse gamma logpdf");
                break;
            case Op::Cdf:
                EXPECT_LT(relErr(d.getCumulativeProbability(r.x), r.want), r.tol)
                    << r.beta << " " << r.x;
                expectAllStrategies(d, r.x, r.want, r.tol, kCdfBatch, "inverse gamma cdf");
                break;
            case Op::Sf:
                EXPECT_LT(relErr(d.getSurvivalProbability(r.x), r.want), r.tol)
                    << r.beta << " " << r.x;
                break;
        }
    }
}

// Gamma at a huge shape took βx rounded, and the shape amplifies that rounding: d log f / d log(βx)
// = α − βx, ~3e9 at α = 1e16 thirty standard deviations out, so at a non-dyadic rate half an ulp
// of βx moved the log density 1.5e-7 absolute and the CDF 1.5e-7 relative, though x is exact.
// The product's rounding is now carried to first order (fma residual). Erlang delegates to Gamma;
// at k = 2e9 the amplification is ~1e6. Each batch strategy must equal the scalar path bit for bit
// (the PDF's vector exp aside). References: mpmath at 60 digits, the CDF by quadrature.
TEST(ContractGates, H5_GammaRateProductRoundingAtHugeShape) {
    using PS = detail::PerformanceHint::PreferredStrategy;
    enum class Op { Pdf, LogPdf, Cdf };
    struct Row {
        double alpha, beta, x;
        Op op;
        double want, tol;
    };
    const Row rows[] = {
        {1e16, 3.0, 3333334333333333.5, Op::LogPdf, -468.24091743850908859, 1e-14},
        {1e16, 3.0, 3333334333333333.5, Op::Pdf, 4.4213343173073725796e-204, 1e-11},
        {1e16, 3.0, 3333332333333333.5, Op::Cdf, 4.9062730810522310327e-198, 1e-12},
        {2e9, 3.0, 667113880.2621666, Op::LogPdf, -460.3280584265334858, 1e-14},
        {2e9, 3.0, 666219453.0711666, Op::Cdf, 4.0118712405962559055e-198, 1e-12},
    };
    for (const auto& r : rows) {
        const GammaDistribution g(r.alpha, r.beta);
        // Erlang's k is an int: from the rows below INT_MAX only, Gamma(1e16, 3) otherwise.
        const ErlangDistribution e(r.alpha < 2147483647.0 ? static_cast<int>(r.alpha) : 1, r.beta);
        std::vector<double> in(69, r.x), out(69, -1.0);
        switch (r.op) {
            case Op::Pdf:
                EXPECT_LT(relErr(g.getProbability(r.x), r.want), r.tol) << r.alpha << " " << r.x;
                expectAllStrategies(g, r.x, r.want, r.tol, kPdfBatch, "gamma pdf");
                break;
            case Op::LogPdf:
                EXPECT_LT(relErr(g.getLogProbability(r.x), r.want), r.tol) << r.alpha << " " << r.x;
                if (r.alpha < 2147483647.0) {
                    EXPECT_LT(relErr(e.getLogProbability(r.x), r.want), r.tol) << r.x;
                }
                for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL,
                             PS::MAXIMIZE_THROUGHPUT}) {
                    g.getLogProbability(std::span<const double>(in), std::span<double>(out),
                                        detail::PerformanceHint{s, std::nullopt});
                    EXPECT_EQ(out[0], g.getLogProbability(r.x)) << static_cast<int>(s);
                    EXPECT_EQ(out[68], g.getLogProbability(r.x)) << static_cast<int>(s);
                }
                break;
            case Op::Cdf:
                EXPECT_LT(relErr(g.getCumulativeProbability(r.x), r.want), r.tol)
                    << r.alpha << " " << r.x;
                if (r.alpha < 2147483647.0) {
                    EXPECT_LT(relErr(e.getCumulativeProbability(r.x), r.want), r.tol) << r.x;
                }
                for (PS s : {PS::FORCE_SCALAR, PS::FORCE_VECTORIZED, PS::FORCE_PARALLEL,
                             PS::MAXIMIZE_THROUGHPUT}) {
                    g.getCumulativeProbability(std::span<const double>(in), std::span<double>(out),
                                               detail::PerformanceHint{s, std::nullopt});
                    EXPECT_EQ(out[0], g.getCumulativeProbability(r.x)) << static_cast<int>(s);
                    EXPECT_EQ(out[68], g.getCumulativeProbability(r.x)) << static_cast<int>(s);
                }
                break;
        }
    }
}

// Gamma, ChiSquared and Erlang inherited S = 1 − F, which keeps S only to the ulp of F near 1:
// below ~1e-16 it is 0 (Gamma(2, 1).getSurvival(45) for 1.3e-18, a hazard of +inf for 45/46),
// and at a huge shape the upper tail loses relative accuracy (Gamma(1e4, 1) at 10500: 4.3e-7 to
// ~1e-10 relative). S = Q(α, βx) is now formed directly, with βx's rounding carried as the CDF
// carries it, mirrored (a non-dyadic rate at α = 1e16 amplifies half an ulp of βx to 1.5e-7
// relative; the last 1e16 row, S = 1.3e-3, passed as 1 − F and fails without the carry).
// ChiSquared and Erlang delegate, so they are called through DistributionBase&. References:
// mpmath at 60 digits, above shape 2e4 by quadrature (tools/accuracy_vs_mpmath.py).
TEST(ContractGates, A11_GammaFamilySurvivalUpperTail) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kEps = std::numeric_limits<double>::epsilon();
    enum class Cls { Gamma, ChiSquared, Erlang };
    struct Row {
        Cls cls;
        double alpha, beta, x, want, tol;  // tol 0: the |ln S|·ε law, four times
    };
    const Row rows[] = {
        {Cls::Gamma, 2.0, 1.0, 45.0, 1.3167585470527210765e-18, 0.0},
        {Cls::Gamma, 0.001, 1.0, 40.0, 1.0412296580236109492e-22, 0.0},
        {Cls::Gamma, 5.5, 0.7, 90.0, 1.1201010674905907736e-21, 0.0},
        {Cls::Gamma, 25.0, 3.0, 40.0, 1.22494070437302878e-26, 0.0},
        {Cls::Gamma, 40.0, 3.0, 30.0, 1.1480803801482780499e-9, 0.0},
        {Cls::Gamma, 1e4, 1.0, 10500.0, 4.2758724550596472692e-7, 1e-13},
        {Cls::Gamma, 1e5, 1.0, 102500.0, 2.2204960520410972022e-15, 1e-13},
        {Cls::Gamma, 1e5, 1.0, 105000.0, 7.2049596948349205796e-55, 1e-12},
        {Cls::Gamma, 1e16, 1.0, 1.00000009e16, 1.1285911472299371638e-19, 1e-12},
        {Cls::Gamma, 1e16, 3.0, 3333334333333333.5, 4.9071548126574352892e-198, 1e-12},
        {Cls::Gamma, 1e16, 3.0, 3333333433333333.5, 0.0013498981276534780595, 1e-12},
        // S near 1, where 1 − F was already right; the complement must stay so (the second in
        // the #216 range, βx below DBL_MIN).
        {Cls::Gamma, 3.0, 0.7, 0.01, 0.9999999431326196149, 0.0},
        {Cls::Gamma, 0.005, 0.5, 5e-324, 0.9758338051382870999, 0.0},
        {Cls::Gamma, 25.0, 3.0, 8.0, 0.55400122307499568648, 0.0},
        // χ²(k) = Gamma(k/2, ½); Erlang(k, λ) = Gamma(k, λ).
        {Cls::ChiSquared, 5.0, 0.5, 150.0, 3.7274850550625096223e-27, 0.0},
        {Cls::ChiSquared, 1.5, 0.5, 100.0, 1.5541594313896049214e-21, 0.0},
        {Cls::Erlang, 3.0, 0.7, 100.0, 1.0022108784225741856e-27, 0.0},
        {Cls::Erlang, 40.0, 3.0, 30.0, 1.1480803801482780499e-9, 0.0},
    };
    for (const auto& r : rows) {
        const GammaDistribution g(r.alpha, r.beta);
        const ChiSquaredDistribution c(r.cls == Cls::ChiSquared ? 2.0 * r.alpha : 1.0);
        const ErlangDistribution e(r.cls == Cls::Erlang ? static_cast<int>(r.alpha) : 1,
                                   r.cls == Cls::Erlang ? r.beta : 1.0);
        const DistributionBase* pick = &g;
        if (r.cls == Cls::ChiSquared)
            pick = &c;
        else if (r.cls == Cls::Erlang)
            pick = &e;
        const DistributionBase& d = *pick;
        const double s = d.getSurvival(r.x);
        const double tol = r.tol > 0.0 ? r.tol : 4.0 * kEps * (1.0 + std::fabs(std::log(r.want)));
        EXPECT_LT(relErr(s, r.want), tol) << static_cast<int>(r.cls) << " " << r.alpha << " "
                                          << r.beta << " x " << r.x << " S " << s;
        EXPECT_LE(std::fabs((s + d.getCumulativeProbability(r.x)) - 1.0), 4.0 * kEps) << r.x;
    }

    // A finite hazard where S was 0: h = f/S = 45/46 at Gamma(2, 1), x = 45.
    const GammaDistribution g21(2.0, 1.0);
    const DistributionBase& g21b = g21;
    EXPECT_LT(relErr(g21b.getHazard(45.0), 45.0 / 46.0), 1e-14);

    // Edges: 1 at and below 0, 0 at +∞, NaN; 0 where βx overflows (Gamma(2, 1e300) at 1e300).
    const GammaDistribution g(5.5, 0.7);
    const DistributionBase& gb = g;
    EXPECT_EQ(gb.getSurvival(0.0), 1.0);
    EXPECT_EQ(gb.getSurvival(-0.0), 1.0);
    EXPECT_EQ(gb.getSurvival(-3.0), 1.0);
    EXPECT_EQ(gb.getSurvival(-kInf), 1.0);
    EXPECT_EQ(gb.getSurvival(kInf), 0.0);
    EXPECT_TRUE(std::isnan(gb.getSurvival(kNaN)));
    const GammaDistribution wide(2.0, 1e300);
    EXPECT_EQ(static_cast<const DistributionBase&>(wide).getSurvival(1e300), 0.0);

    // Monotone across the switches, S down and F up, in steps of 1e-13 relative in x: the
    // series / continued fraction switch at βx = α + 1 (at shapes below and above
    // STIRLING_PREFACTOR_SHAPE, under kSmallShapeQ, at a non-dyadic rate), Q = DBL_MIN and
    // P = DBL_MIN where the rate carry stops (α = 25, β = 3), the Temme zone's lower edge
    // (α = 1e4, η = −0.6), the carry at α = 1e16, and βx = DBL_MIN. Not bit-stepped: one ulp of x
    // moves ln S by (βx·f/S)·ε, and the incomplete gamma's own rounding is |ln S|·ε-class, the
    // same order in the tails (~100 ulp of S at Q = DBL_MIN) and larger in the bulk below α = 20
    // (tens of ulp), so a bit-stepped S, like F, zig-zags at rounding level everywhere; a step of
    // 1e-13 moves S by ~10x that rounding.
    struct Mono {
        double alpha, beta, x;
    };
    const Mono mono[] = {
        {0.3, 1.0, 1.3},
        {5.5, 0.7, 6.5 / 0.7},
        {19.5, 1.0, 20.5},
        {25.0, 3.0, 26.0 / 3.0},
        {1e3, 1.0, 1001.0},
        {25.0, 3.0, 271.5011491074834},       // Q = DBL_MIN
        {25.0, 3.0, 1.6764798954741084e-12},  // P = DBL_MIN
        {1e4, 1.0, 5135.05541020432},
        {1e16, 3.0, 3333334333333333.5},
        {0.005, 0.5, 2.0 * std::numeric_limits<double>::min()},
    };
    // The step scales with log y's own rounding: near βx = DBL_MIN, log x ≈ −708 carries ~1e-13,
    // and a step moves F by α·h·F while either path's noise is ~α·|ln βx|·ε·F, so a fixed 1e-13
    // equalled the noise at α = 0.005 there and MSVC's libm stepped F down 4 ulp across the
    // log-space seam (CI on 47bea80). h = max(1e-13, 1000·|ln βx|·ε) keeps the step ~1000x the
    // noise on every row (1.6e-10 at the DBL_MIN row; 1e-13 in the bulk).
    for (const auto& m : mono) {
        const GammaDistribution d(m.alpha, m.beta);
        const double h = std::max(1e-13, 1000.0 * std::numeric_limits<double>::epsilon() *
                                             std::abs(std::log(m.beta * m.x)));
        double prev = d.getSurvival(m.x * (1.0 - 200.0 * h));
        double prev_f = d.getCumulativeProbability(m.x * (1.0 - 200.0 * h));
        for (int k = -199; k <= 200; ++k) {
            const double x = m.x * (1.0 + k * h);
            const double s = d.getSurvival(x);
            const double f = d.getCumulativeProbability(x);
            ASSERT_LE(s, prev) << m.alpha << " " << m.beta << " x " << x;
            ASSERT_GE(f, prev_f) << m.alpha << " " << m.beta << " x " << x;
            prev = s;
            prev_f = f;
        }
    }
}
