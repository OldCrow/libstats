/**
 * @file test_distribution_identities.cpp
 * @brief Oracle-free identity and round-trip checks over all 27 distributions.
 *
 * Every check compares the library with itself, so no reference values are needed:
 *   range     CDF in [0, 1] and monotone on a grid between the 1e-12 quantiles.
 *   qf        Q(F(x)) = x within the conditioning of the CDF (continuous); Q(F(k)) = k (discrete).
 *   fq        F(Q(p)) = p within the rounding of Q(p) (continuous); discrete:
 *             Q(p) = min{k : F(k) >= p} on the library's own CDF (Decided #116/#170).
 *   qmono     Q non-decreasing in p, down to the subnormal p; qnan: Q(p) is never NaN.
 *   deriv     pdf against a Richardson central difference of the CDF; pmf against F(k) - F(k-1).
 *   logpdf    logpdf against log(pdf).
 *   batch     scalar against the FORCE_VECTORIZED batch path.
 *   edge      F at +-1e18, +-1e300, +-DBL_MAX in [0, 1]; Q(NaN) is NaN.
 * plus identities between independent implementations (CrossClass tests).
 *
 * The instances are fixed adversarial parameter sets (the regime boundaries the code branches on,
 * and the cases that found the 2026-10-07 defects) plus two seeded pseudo-random instances per
 * family. The generator is mt19937_64 with its raw output mapped to [0, 1) here, so the instances
 * are the same on every standard library.
 *
 * Budgets are conditioning-based; each constant below states what it covers. A failure names the
 * check, the instance and the worst cases. Defects open past this release are listed in
 * knownIssue(); a test whose only failures are known ones is skipped with their tags, as in
 * test_concurrency_stress.cpp. Remove an entry when its fix lands (its "[ known ]" lines vanish).
 */

#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <gtest/gtest.h>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace {

using P = std::array<double, 4>;
constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kPi = std::acos(-1.0);

// ---------------------------------------------------------------------------------------------
// Budgets
// ---------------------------------------------------------------------------------------------
// Relative error allowed in a CDF value, in units of eps. Covers the CDF's own accuracy law
// (tails ~1e-13) and, for F near 1, the half-ulp of F itself: F is returned as F, not as 1 - F,
// so near 1 its absolute error is eps/2 however accurate the algorithm is.
constexpr double kCdfRelUlps = 1024.0;
// Ulps of x (or of Q(p)) a quantile may be off and still pass. Q cannot be closer than an ulp of
// its result; the iterative solvers stop within a few hundred ulps of relative tolerance.
constexpr double kQuantileUlps = 256.0;
// Relative pdf against dF/dx floor (the Richardson estimate carries its own error terms on top).
constexpr double kDerivRel = 1e-7;
// von Mises: the pdf uses the Tier-2 Bessel normaliser (4.7e-7 relative on libc++, documented in
// ACCURACY_CHARACTERIZATION) and the CDF a half-ulp one, so pdf and dF/dx differ by up to ~5e-7.
constexpr double kDerivRelVonMises = 2e-6;
// von Mises kappa > 1000: the CDF is the documented wrapped-normal approximation, "~0.04/kappa"
// absolute (von_mises.h); measured 0.0435/kappa at p = 0.1, so the budget is 0.1/kappa. Round trips
// are scored in F-space against that; qf and deriv are skipped.
constexpr double kVonMisesApproxKappa = 1000.0;
constexpr double kVonMisesApproxAbs = 0.1;
// Scalar against vectorised batch: the SIMD kernels are documented few-ulp; 1e-11 relative leaves
// a wide margin and still catches a wrong branch. Below kBatchAbsFloor values are compared
// absolutely (a kernel may flush a result near the underflow threshold earlier than libm).
constexpr double kBatchRel = 1e-11;
constexpr double kBatchAbsFloor = 1e-290;

// ---------------------------------------------------------------------------------------------
// Deterministic generator: mt19937_64's output sequence is fixed by the standard; the
// distributions in <random> are not, so they are not used.
// ---------------------------------------------------------------------------------------------
struct Rng {
    std::mt19937_64 g;
    explicit Rng(std::uint64_t seed) : g(seed) {}
    double uni() { return static_cast<double>(g() >> 11) * 0x1.0p-53; }
    double logUni(double lo, double hi) {
        return std::exp(std::log(lo) + uni() * (std::log(hi) - std::log(lo)));
    }
};
std::uint64_t fnv1a(const std::string& s) {
    std::uint64_t h = 1469598103934665603ULL;
    for (char c : s)
        h = (h ^ static_cast<unsigned char>(c)) * 1099511628211ULL;
    return h;
}

std::string fmt(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}
double ulp(double x) {
    x = std::fabs(x);
    if (!std::isfinite(x))
        return kInf;
    return std::nextafter(x, kInf) - x;
}

// Resolution of a quantile at q: kQuantileUlps ulps of q, and at least the resolution of a solver
// that bisects on ln q and returns exp(t) (InverseGamma and others): |q| * ulp(ln|q|), which is
// ~500 ulps of q at q ~ 1e300.
double quantileResolution(double q) {
    const double aq = std::fabs(q);
    double r = kQuantileUlps * ulp(q);
    if (aq > 0 && std::isfinite(aq))
        r = std::max(r, 4 * aq * ulp(std::log(aq)));
    return r;
}

// ---------------------------------------------------------------------------------------------
// Known open defects, by issue number. A failure they explain prints
// "[ known ]" and does not fail the test.
// ---------------------------------------------------------------------------------------------
struct Where {
    std::string family;
    P par;
    std::string check;
    double v;  // p for fq/qmono, x otherwise
};

std::string knownIssue(const Where& w) {
    const auto& f = w.family;
    const bool roundTrip = w.check == "fq" || w.check == "qmono" || w.check == "qf";
    // #214: Gamma-family quantile seed fails below p ~ 1e-300: Q non-monotone in p, F(Q(p)) up to
    // 100 orders above p. Student-t shares the symptom at p = denorm_min (defect hunt D3).
    if ((f == "gamma" || f == "chi_squared" || f == "erlang" || f == "student_t") &&
        (w.check == "fq" || w.check == "qmono") && w.v < 1e-300)
        return "214";
    // #215: von Mises mu != 0: Q(p) for p <~ 1e-16 is mu + pi (wrapped), where the CDF wraps x - mu
    // onto +pi and returns 1.
    if (f == "von_mises" && w.par[0] != 0.0 && (w.check == "fq" || w.check == "qmono") &&
        w.v < 1e-15)
        return "215";
    // #113: beta_i at a shape ratio >= 1e5 is ~1e-11 relative (FisherF and Beta).
    if (f == "fisher_f" || f == "beta") {
        const double a = w.par[0], b = w.par[1];
        if (std::max(a, b) / std::min(a, b) >= 1e5 && (roundTrip || w.check == "deriv"))
            return "113";
    }
    // #126: Binomial at n >= 1e9 takes beta_i at b ~ n, which loses ~1e-6 absolute there: pmf vs
    // F(k) - F(k-1) at the median differs by 3.6e-7 relative under MSVC (CI, 2026-10-08).
    if (f == "binomial" && w.par[0] >= 1e9 && (roundTrip || w.check == "deriv"))
        return "126";
    // #126: NegativeBinomial with tiny p (beta_i at x = p, b ~ 1/p) is ~1e-10 relative at r = 3.
    // r = 1 (Geometric) takes the closed form (#202) and is checked.
    if (f == "negative_binomial" && w.par[0] != 1.0 && w.par[1] <= 1e-4 &&
        (roundTrip || w.check == "deriv"))
        return "126";
    // Found by this test on 2026-10-07 (#223-#225; disposition pending):
    // #223: Gamma quantile in the upper tail for shape below ~0.004 stops far from the root:
    // Gamma(0.001338, 0.9328).Q(0.9999189344) = 1.89678, F of it 0.9999094; Q not monotone.
    if (f == "gamma" && w.par[0] < 0.004 && (roundTrip || w.check == "fq"))
        return "223";
    // #224: Gamma/Erlang CDF is NaN where x * rate overflows: Gamma(100, 100).cdf(DBL_MAX),
    // Erlang(1, 1e300).cdf(1e18).
    if ((f == "gamma" || f == "erlang") && w.check == "edge" && std::fabs(w.v) >= 1e18)
        return "224";
    // #225: TruncatedNormal(0, 1, -inf, 0).Q(denorm_min) is NaN.
    if (f == "truncated_normal" && w.check == "qnan" && std::isinf(w.par[2]) && w.par[3] == 0.0)
        return "225";
    return {};
}

// ---------------------------------------------------------------------------------------------
// Failure collection: one ADD_FAILURE per (check, instance) with the worst examples.
// ---------------------------------------------------------------------------------------------
struct Report {
    struct Item {
        int count = 0;
        double worst = 0.0;
        std::vector<std::pair<double, std::string>> ex;
    };
    std::map<std::string, Item> fails;  // "check | instance"
    std::map<std::string, int> known;   // "tag check | instance"

    void add(const Where& w, const std::string& inst, double severity, const std::string& detail) {
        const std::string tag = knownIssue(w);
        if (!tag.empty()) {
            known["#" + tag + " " + w.check + " | " + inst]++;
            return;
        }
        auto& it = fails[w.check + " | " + inst];
        it.count++;
        it.worst = std::max(it.worst, severity);
        it.ex.emplace_back(severity, detail);
        std::sort(it.ex.begin(), it.ex.end(), [](auto& a, auto& b) { return a.first > b.first; });
        if (it.ex.size() > 3)
            it.ex.resize(3);
    }

    // Reports, then skips the test if known defects were all that failed.
    void finish() {
        for (const auto& [k, it] : fails) {
            std::ostringstream o;
            o << "[" << k << "] " << it.count << " failure(s), worst err/budget "
              << std::setprecision(3) << it.worst;
            for (const auto& e : it.ex)
                o << "\n    " << e.second;
            ADD_FAILURE() << o.str();
        }
        std::vector<std::string> lines;
        for (const auto& [k, n] : known) {
            std::printf("[ known    ] %s (%d)\n", k.c_str(), n);
            lines.push_back(k + " (" + std::to_string(n) + ")");
        }
        if (!lines.empty() && !::testing::Test::HasFailure()) {
            std::string list;
            for (const auto& l : lines)
                list += "\n  " + l;
            GTEST_SKIP() << "only known open defects failed:" << list;
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Per-instance checks
// ---------------------------------------------------------------------------------------------
struct Inst {
    std::string family;
    P par;
    std::string name;
    bool discrete = false;
    bool circular = false;    // von Mises: CDF at wrap(x - mu); grid on (mu - pi, mu + pi)
    double approxAbsF = 0.0;  // documented absolute CDF approximation (von Mises kappa > 1000)
    double derivRel = kDerivRel;
};

template <typename D>
void checkInstance(const D& d, const Inst& in, Report& r, Rng& rng) {
    const auto F = [&](double x) { return d.getCumulativeProbability(x); };
    const auto f = [&](double x) { return d.getProbability(x); };
    const auto lf = [&](double x) { return d.getLogProbability(x); };
    const auto Q = [&](double p) { return d.getQuantile(p); };
    const auto W = [&](const std::string& check, double v) {
        return Where{in.family, in.par, check, v};
    };
    const double lo = d.getSupportLowerBound(), hi = d.getSupportUpperBound();
    const double mu = in.par[0];

    // Grid
    std::vector<double> xs;
    if (in.circular) {
        for (int i = 1; i < 400; ++i)
            xs.push_back(mu - kPi + 2 * kPi * i / 400.0);
    } else {
        double qlo = Q(1e-12), qhi = Q(1.0 - 1e-12);
        double mean = d.getMean(), var = d.getVariance();
        const double sd = (std::isfinite(var) && var > 0) ? std::sqrt(var) : 1.0;
        if (!std::isfinite(mean))
            mean = 0.0;
        if (!std::isfinite(qlo))
            qlo = std::max(lo, mean - 40 * sd);
        if (!std::isfinite(qhi))
            qhi = std::min(hi, mean + 40 * sd);
        const int n = in.discrete ? 300 : 400;
        for (int i = 0; i <= n; ++i) {
            const double x = qlo + (qhi - qlo) * i / n;
            xs.push_back(in.discrete ? std::floor(x) : x);
        }
        for (int i = 0; i < 80; ++i) {
            double p = std::pow(10.0, -14.0 * rng.uni());
            if (rng.uni() < 0.5)
                p = 1.0 - p;
            const double x = Q(p);
            if (std::isfinite(x))
                xs.push_back(x);
        }
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

    // range: F in [0, 1] and monotone. A step down of a few ulps is a rounding nit, not a contract
    // break; 8 eps relative separates the two.
    {
        double prev = 0.0, prevX = -kInf;
        for (double x : xs) {
            const double Fx = F(x);
            if (!(Fx >= 0.0 && Fx <= 1.0))
                r.add(W("range", x), in.name, 1.0, "F(" + fmt(x) + ") = " + fmt(Fx));
            else if (Fx < prev - 8 * kEps * prev)
                r.add(W("range", x), in.name, (prev - Fx) / (8 * kEps * prev),
                      "F(" + fmt(x) + ") = " + fmt(Fx) + " < F(" + fmt(prevX) + ") = " + fmt(prev));
            if (Fx >= 0.0 && Fx <= 1.0) {
                prev = Fx;
                prevX = x;
            }
        }
    }

    // qf: Q(F(x)) = x
    for (double x : xs) {
        const double Fx = F(x);
        if (!(Fx > 0.0 && Fx < 1.0))
            continue;
        const double q = Q(Fx);
        if (in.discrete) {
            const double Fprev = F(x - 1);
            if (Fprev < Fx && q != x)
                r.add(W("qf", x), in.name, std::fabs(q - x),
                      "k=" + fmt(x) + " F(k)=" + fmt(Fx) + " Q(F(k))=" + fmt(q));
            continue;
        }
        if (in.approxAbsF > 0.0)
            continue;  // scored in F-space by fq
        const double err =
            in.circular ? std::fabs(std::remainder(q - x, 2 * kPi)) : std::fabs(q - x);
        // dx = dF / f with dF the CDF's error, plus Q's own error in ulps of x.
        const double fx = f(x);
        // A subnormal F carries an absolute error of denorm_min.
        const double dF = kCdfRelUlps * (kEps * Fx + std::numeric_limits<double>::denorm_min());
        const double budget =
            (fx > 0 && std::isfinite(fx) ? dF / fx : kInf) +
            std::max(quantileResolution(x), in.circular ? kQuantileUlps * ulp(kPi) : 0.0);
        if (!(err <= budget))
            r.add(W("qf", x), in.name, err / budget,
                  "x=" + fmt(x) + " F(x)=" + fmt(Fx) + " Q(F(x))=" + fmt(q) + " err=" + fmt(err) +
                      " budget=" + fmt(budget));
    }

    // fq and qmono
    std::vector<double> ps;
    for (int e = 1; e <= 300; e += 7)
        ps.push_back(std::pow(10.0, -e));
    for (double p : {0.5, 0.3, 0.7, 0.9, 0.99, 0.2, 0.1, 0.01, 1e-3, 1.0 - 1e-6, 1.0 - 1e-10,
                     1.0 - 1e-15, 1e-301, 1e-305, 1e-310, 1e-320, DBL_MIN, 5e-324})
        ps.push_back(p);
    for (int i = 0; i < 60; ++i) {
        const double p = std::pow(10.0, -16.0 * rng.uni());
        ps.push_back(rng.uni() < 0.5 ? p : 1.0 - p);
    }
    std::sort(ps.begin(), ps.end());
    ps.erase(std::unique(ps.begin(), ps.end()), ps.end());
    double prevQ = -kInf, prevP = 0.0;
    // von Mises: Q is wrapped to (-pi, pi]; unwrap it to t = q - mu in [-pi, pi] on the side its p
    // implies (t <= 0 below the median), since q - mu rounds onto either end near +-pi.
    const auto unwrapT = [&](double q, double p) {
        double t = std::remainder(q - mu, 2 * kPi);
        if (p < 0.5 && t > 0)
            t -= 2 * kPi;
        if (p > 0.5 && t < 0)
            t += 2 * kPi;
        return t;
    };
    for (double p : ps) {
        double q = Q(p);
        if (std::isnan(q)) {
            r.add(W("qnan", p), in.name, 1.0, "Q(" + fmt(p) + ") = NaN");
            continue;
        }
        // qmono: consecutive p (a step down of 16 ulps is a rounding nit), unwrapped for von Mises.
        const double qu = in.circular ? mu + unwrapT(q, p) : q;
        if (qu < prevQ - 16 * ulp(prevQ))
            r.add(W("qmono", prevP), in.name, 1.0,
                  "Q(" + fmt(prevP) + ") = " + fmt(prevQ) + " > Q(" + fmt(p) + ") = " + fmt(qu));
        prevQ = qu;
        prevP = p;
        if (std::isinf(q)) {
            if ((q > 0 && std::isfinite(hi)) || (q < 0 && std::isfinite(lo)))
                r.add(W("fq", p), in.name, 1.0,
                      "Q(" + fmt(p) + ") = " + fmt(q) + " on finite support");
            continue;
        }
        const double Fq = F(q);
        if (in.discrete) {
            if (!(Fq >= p))
                r.add(W("fq", p), in.name, (p - Fq) / p,
                      "p=" + fmt(p) + " Q=" + fmt(q) + " F(Q)=" + fmt(Fq) + " < p");
            else if (q - 1 >= lo && !(F(q - 1) < p))
                r.add(W("fq", p), in.name, 1.0,
                      "p=" + fmt(p) + " Q=" + fmt(q) + " not minimal: F(Q-1)=" + fmt(F(q - 1)));
            continue;
        }
        // Budget: F's own error at p, plus the change of F over kQuantileUlps ulps of q (from the
        // CDF itself, so it holds where q is subnormal and the pdf is 0 or unrepresentable).
        // The spread counts only where F is monotone across [qa, q, qb], so a CDF that jumps (a
        // wrap, a garbage value) cannot widen its own budget.
        double spread = 0.0;
        if (!in.circular) {
            const double s = quantileResolution(q);
            const double Fa = F(std::max(q - s, lo)), Fb = F(std::min(q + s, hi));
            if (Fa <= Fq && Fq <= Fb)
                spread = Fb - Fa;
        } else {
            // q is a wrapped angle, resolved to an ulp of pi; F(mu + t) re-rounds x - mu near
            // +-pi, so the spread comes from the (bounded, positive) density instead.
            spread = 4 * kQuantileUlps * ulp(kPi) * f(q);
        }
        double budget = kCdfRelUlps * kEps * p + spread + in.approxAbsF;
        if (p < 1e-300)
            budget = std::max(budget, 1e-300);  // F near the underflow threshold
        const double err = std::fabs(Fq - p);
        if (!(err <= budget))
            r.add(W("fq", p), in.name, err / budget,
                  "p=" + fmt(p) + " Q(p)=" + fmt(q) + " F(Q)=" + fmt(Fq) + " err=" + fmt(err) +
                      " budget=" + fmt(budget));
    }

    // deriv and logpdf
    const double countMean = in.discrete ? d.getMean() : 0.0;
    for (std::size_t i = 0; i < xs.size(); i += 3) {
        const double x = xs[i];
        const double fx = f(x), lfx = lf(x);
        if (fx >= 1e-290 && fx <= 1e290) {
            const double l = std::log(fx);
            const double tol = std::max(1e-12, 4e-14 * std::max(1.0, std::fabs(l)));
            if (!(std::fabs(lfx - l) <= tol))
                r.add(W("logpdf", x), in.name, std::fabs(lfx - l) / tol,
                      "x=" + fmt(x) + " pdf=" + fmt(fx) + " logpdf=" + fmt(lfx) +
                          " log(pdf)=" + fmt(l));
        } else if (fx == 0.0 && lfx > -700.0) {
            r.add(W("logpdf", x), in.name, 1.0, "x=" + fmt(x) + " pdf=0 logpdf=" + fmt(lfx));
        } else if (std::isnan(fx) != std::isnan(lfx)) {
            r.add(W("logpdf", x), in.name, 1.0,
                  "x=" + fmt(x) + " pdf=" + fmt(fx) + " logpdf=" + fmt(lfx));
        }

        const double Fx = F(x);
        if (in.discrete) {
            // pmf(k) = F(k) - F(k-1); the difference carries both CDF values' error. A count
            // CDF (incomplete beta/gamma prefactor) is conditioned ~|k - mean| times eps relative
            // (Binomial(1e6, 0.3) is documented at 280x the bare law; n = 2e9 measures 2e5 eps at
            // |k - mean| = 1.5e5), so that factor scales the budget.
            if (Fx > 1e-280 && Fx < 0.9) {
                const double diff = Fx - F(x - 1);
                const double cond = 1.0 + std::fabs(x - countMean);
                const double tol = 1e-12 * fx + 2 * kCdfRelUlps * kEps * Fx * cond;
                if (!(std::fabs(fx - diff) <= tol))
                    r.add(W("deriv", x), in.name, std::fabs(fx - diff) / tol,
                          "k=" + fmt(x) + " pmf=" + fmt(fx) + " F(k)-F(k-1)=" + fmt(diff));
            }
            continue;
        }
        if (in.approxAbsF > 0.0 || !(fx > 1e-280 && std::isfinite(fx)) ||
            !(Fx > 1e-6 && Fx < 1 - 1e-6))
            continue;
        // Step: a thousandth of the distance over which F changes by its own size. Where 1024 ulps
        // of x exceed a hundredth of that distance the arguments are too coarse to resolve the
        // derivative (a scale far below ulp(location), e.g. Cauchy(-7, 1e-300)): skip.
        const double local = std::min(Fx, 1 - Fx) / fx;
        double h = 1e-3 * local;
        if (1024 * ulp(x) > 1e-2 * local)
            continue;
        h = std::max(h, 1024 * ulp(x));
        const double x1 = x - h, x2 = x + h, x3 = x - h / 2, x4 = x + h / 2;
        if (!in.circular && (x1 <= lo || x2 >= hi))
            continue;
        if (in.circular && std::fabs(x - mu) + h >= kPi)
            continue;
        const double F1 = F(x1), F2 = F(x2), F3 = F(x3), F4 = F(x4);
        const double d1 = (F2 - F1) / (x2 - x1), d2 = (F4 - F3) / (x4 - x3);
        const double est = (4 * d2 - d1) / 3;
        const double round = 4 * kCdfRelUlps * kEps * F2 / (x4 - x3);
        const double tol = in.derivRel * fx + 2 * std::fabs(d2 - d1) + round;
        if (!(std::fabs(fx - est) <= tol))
            r.add(W("deriv", x), in.name, std::fabs(fx - est) / tol,
                  "x=" + fmt(x) + " pdf=" + fmt(fx) + " dF/dx=" + fmt(est) + " tol=" + fmt(tol));
    }

    // batch: scalar against FORCE_VECTORIZED
    {
        std::vector<double> rx = xs;
        std::shuffle(rx.begin(), rx.end(), rng.g);
        if (rx.size() > 257)
            rx.resize(257);
        std::vector<double> out(rx.size());
        stats::detail::PerformanceHint hint;
        hint.strategy = stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
        // LogNormal: the kernels' few-ulp log x carries an absolute error ~|log x| eps into
        // z = (log x - mu) / sigma, which the density amplifies by |z| (conditioning of
        // log x - mu, not a kernel defect; LogNormal(100, 1e-3) needs ~1e-9 relative).
        // Weibull likewise: z = log x - log lambda carries |log x| eps absolute, and
        // t = exp(k z) multiplies it by k (1 + t) in pdf and cdf.
        const auto extra = [&](double x) {
            if (!(x > 0))
                return 0.0;
            const double lx = std::max(1.0, std::fabs(std::log(x)));
            if (in.family == "lognormal") {
                const double z = (std::log(x) - in.par[0]) / in.par[1];
                return 64 * kEps * lx / in.par[1] * (1 + std::fabs(z));
            }
            if (in.family == "weibull") {
                const double t = std::pow(x / in.par[1], in.par[0]);
                return 64 * kEps * lx * in.par[0] * (1 + t);
            }
            return 0.0;
        };
        for (int m = 0; m < 3; ++m) {
            const char* mname = m == 0 ? "pdf" : m == 1 ? "logpdf" : "cdf";
            if (m == 0)
                d.getProbability(std::span<const double>(rx), std::span<double>(out), hint);
            else if (m == 1)
                d.getLogProbability(std::span<const double>(rx), std::span<double>(out), hint);
            else
                d.getCumulativeProbability(std::span<const double>(rx), std::span<double>(out),
                                           hint);
            for (std::size_t i = 0; i < rx.size(); ++i) {
                const double s = m == 0 ? f(rx[i]) : m == 1 ? lf(rx[i]) : F(rx[i]);
                const double b = out[i];
                if (s == b || (std::isnan(s) && std::isnan(b)))
                    continue;
                const double rel = kBatchRel + extra(rx[i]);
                const double tol = m == 1 ? rel * std::max(1.0, std::fabs(s))
                                          : std::max(rel * std::fabs(s), kBatchAbsFloor);
                const double err = std::fabs(s - b);
                if (std::isnan(s) || std::isnan(b) || !(err <= tol))
                    r.add(W("batch", rx[i]), in.name, std::isnan(err) ? 1.0 : err / tol,
                          std::string(mname) + " x=" + fmt(rx[i]) + " scalar=" + fmt(s) +
                              " batch=" + fmt(b));
            }
        }
    }

    // edge: F at huge finite x stays a probability, in scalar and batch.
    {
        const std::vector<double> ex = {1e18, 1e300, DBL_MAX, -1e18, -1e300, -DBL_MAX};
        std::vector<double> out(ex.size());
        stats::detail::PerformanceHint hint;
        hint.strategy = stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
        d.getCumulativeProbability(std::span<const double>(ex), std::span<double>(out), hint);
        for (std::size_t i = 0; i < ex.size(); ++i) {
            const double s = F(ex[i]);
            if (!(s >= 0.0 && s <= 1.0) || !(out[i] >= 0.0 && out[i] <= 1.0))
                r.add(W("edge", ex[i]), in.name, 1.0,
                      "F(" + fmt(ex[i]) + ") scalar=" + fmt(s) + " batch=" + fmt(out[i]));
        }
    }
}

// Q(NaN) is NaN in every class (one rule: NaN in, NaN out).
template <typename D>
void checkQuantileNaN(const D& d, const Inst& in, Report& r) {
    double q = 0.0;
    std::string what;
    try {
        q = d.getQuantile(kNaN);
    } catch (const std::exception& e) {
        what = std::string(" (threw: ") + e.what() + ")";
    }
    if (!what.empty() || !std::isnan(q))
        r.add(Where{in.family, in.par, "edge", kNaN}, in.name, 1.0, "Q(NaN) = " + fmt(q) + what);
}

template <typename D, typename Make, typename Random>
void runFamily(const std::string& family, Make make, std::vector<P> instances, Random random,
               bool discrete) {
    Rng rng(fnv1a(family));
    for (int i = 0; i < 2; ++i)
        instances.push_back(random(rng));
    Report r;
    bool nanChecked = false;
    for (const auto& par : instances) {
        Inst in;
        in.family = family;
        in.par = par;
        in.discrete = discrete;
        std::ostringstream tag;
        tag << family << "(" << std::setprecision(10) << par[0] << "," << par[1];
        if (family == "truncated_normal")
            tag << "," << par[2] << "," << par[3];
        tag << ")";
        in.name = tag.str();
        if (family == "von_mises") {
            in.circular = true;
            in.derivRel = kDerivRelVonMises;
            if (par[1] > kVonMisesApproxKappa)
                in.approxAbsF = kVonMisesApproxAbs / par[1];
        }
        // An instance outside the computable range may be rejected; one the factory accepts must
        // satisfy every check (the D6 / D9 contract).
        std::optional<D> d;
        try {
            d.emplace(make(par));
        } catch (const std::exception& e) {
            std::printf("[ rejected ] %s: %s\n", in.name.c_str(), e.what());
            continue;
        }
        checkInstance(*d, in, r, rng);
        if (!nanChecked) {
            checkQuantileNaN(*d, in, r);
            nanChecked = true;
        }
    }
    r.finish();
}

// ---------------------------------------------------------------------------------------------
// Families. Each list: the regime boundaries the code branches on, plus the instances that found
// the 2026-10-07 defects (named in the comment).
// ---------------------------------------------------------------------------------------------
using namespace stats;

TEST(DistributionIdentities, Gaussian) {
    runFamily<GaussianDistribution>(
        "gaussian", [](const P& p) { return GaussianDistribution(p[0], p[1]); },
        {{0, 1},
         {3.5, 0.7},
         {1e3, 1e-3},
         {-1e-8, 1e-12},
         {0, 1e100},
         {13668.12633, 9.564126947e-09}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-9, 1e9)}; }, false);
}
TEST(DistributionIdentities, LogNormal) {
    runFamily<LogNormalDistribution>(
        "lognormal", [](const P& p) { return LogNormalDistribution(p[0], p[1]); },
        {{0, 1}, {1.3, 0.25}, {-5, 5}, {100, 1e-3}},
        [](Rng& g) { return P{g.logUni(1e-3, 20), g.logUni(1e-4, 30)}; }, false);
}
TEST(DistributionIdentities, Exponential) {
    runFamily<ExponentialDistribution>(
        "exponential", [](const P& p) { return ExponentialDistribution(p[0]); },
        {{1}, {0.37}, {1e-300}, {1e300}}, [](Rng& g) { return P{g.logUni(1e-12, 1e12)}; }, false);
}
TEST(DistributionIdentities, Uniform) {
    runFamily<UniformDistribution>(
        "uniform", [](const P& p) { return UniformDistribution(p[0], p[1]); },
        {{0, 1}, {-3, 7}, {1e300, 1.5e300}, {-1e-300, 1e-300}},
        [](Rng& g) { return P{g.logUni(1e-3, 1e3), g.logUni(2e3, 1e6)}; }, false);
}
TEST(DistributionIdentities, Gamma) {
    // Stirling switch at shape 20; the p < 1e-300 seed failure is N1.
    runFamily<GammaDistribution>(
        "gamma", [](const P& p) { return GammaDistribution(p[0], p[1]); },
        {{2, 1},
         {19.999, 1},
         {20, 1},
         {20.001, 1},
         {0.5, 1},
         {0.999, 1},
         {1.001, 1},
         {100, 100},
         {1e3, 1e-6},
         {1e6, 1},
         {1e8, 1},
         {0.01, 0.01},
         {1e4, 1e-3}},
        [](Rng& g) { return P{g.logUni(1e-3, 1e7), g.logUni(1e-5, 1e5)}; }, false);
}
TEST(DistributionIdentities, StudentT) {
    runFamily<StudentTDistribution>(
        "student_t", [](const P& p) { return StudentTDistribution(p[0]); },
        {{1}, {2}, {2.5}, {3}, {4}, {30}, {1e3}, {1e4}, {1e8}, {0.5}, {1.001}, {1e6}},
        [](Rng& g) { return P{g.logUni(0.1, 1e9)}; }, false);
}
TEST(DistributionIdentities, Cauchy) {
    runFamily<CauchyDistribution>(
        "cauchy", [](const P& p) { return CauchyDistribution(p[0], p[1]); },
        {{0, 1}, {2, 0.5}, {-7, 1e-300}, {1e300, 1}, {0, 1e-6}, {1e8, 1e6}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-6, 1e6)}; }, false);
}
TEST(DistributionIdentities, VonMises) {
    // mu != 0 (N3 at p <~ 1e-16); kappa astride the Tier-2 and wrapped-normal switches.
    runFamily<VonMisesDistribution>(
        "von_mises", [](const P& p) { return VonMisesDistribution(p[0], p[1]); },
        {{0, 1},
         {1, 1},
         {-2.5, 5},
         {3, 50},
         {0, 44},
         {0, 46},
         {0, 999},
         {0, 1001},
         {0, 1e4},
         {0, 1e-17},
         {0, 1e-9},
         {0, 0.312},
         {3.1, 2},
         {-3.1, 100},
         {0, 100}},
        [](Rng& g) { return P{-3.14 + 6.28 * g.uni(), g.logUni(1e-8, 1e3)}; }, false);
}
TEST(DistributionIdentities, Binomial) {
    runFamily<BinomialDistribution>(
        "binomial", [](const P& p) { return BinomialDistribution(static_cast<int>(p[0]), p[1]); },
        {{20, 0.5},
         {19, 0.5},
         {21, 0.5},
         {25, 1e-6},
         {25, 1 - 1e-6},
         {1e6, 1e-7},
         {1e6, 0.999999},
         {2e9, 0.5},
         {7, 0.1},
         {1, 0.5},
         {1e6, 0.3}},
        [](Rng& g) { return P{std::floor(g.logUni(1, 1e8)), g.logUni(1e-6, 1 - 1e-6)}; }, true);
}
TEST(DistributionIdentities, NegativeBinomial) {
    // cdf(1e300) NaN (D10); r != 1 with p <= 1e-4 is NB-r.
    runFamily<NegativeBinomialDistribution>(
        "negative_binomial", [](const P& p) { return NegativeBinomialDistribution(p[0], p[1]); },
        {{5, 0.5},
         {19.5, 0.5},
         {20, 0.5},
         {25, 0.999},
         {1e5, 0.5},
         {1e5, 1e-4},
         {0.5, 0.5},
         {1e-3, 0.999},
         {3, 1e-6},
         {0.01, 0.9},
         {1e4, 0.01},
         {1, 1e-12}},
        [](Rng& g) { return P{g.logUni(1e-2, 1e6), g.logUni(1e-3, 1 - 1e-5)}; }, true);
}
TEST(DistributionIdentities, Geometric) {
    // p = 1e-12: CDF 3e-5 relative and non-monotone over large k (D4).
    runFamily<GeometricDistribution>(
        "geometric", [](const P& p) { return GeometricDistribution(p[0]); },
        {{0.3}, {0.05}, {0.999999}, {1e-12}, {0.5}, {1}, {1e-6}, {2.243476974e-09}},
        [](Rng& g) { return P{g.logUni(1e-9, 1 - 1e-9)}; }, true);
}
TEST(DistributionIdentities, Beta) {
    // Quantile: 1e-8 absolute Newton tolerance and floor, wrong end at Beta(25,1000) (D2).
    runFamily<BetaDistribution>(
        "beta", [](const P& p) { return BetaDistribution(p[0], p[1]); },
        {{2, 3},
         {20, 2},
         {2, 20},
         {19.99, 20.01},
         {20, 20},
         {1e3, 1},
         {1, 1e3},
         {0.5, 0.5},
         {1, 1},
         {1e5, 2e5},
         {1e6, 1e6},
         {1e-3, 1e3},
         {25, 1e3},
         {0.01, 0.01},
         {1e4, 1e4}},
        [](Rng& g) { return P{g.logUni(1e-2, 1e4), g.logUni(1e-2, 1e4)}; }, false);
}
TEST(DistributionIdentities, ChiSquared) {
    runFamily<ChiSquaredDistribution>(
        "chi_squared", [](const P& p) { return ChiSquaredDistribution(p[0]); },
        {{3}, {1}, {2}, {39.9}, {40}, {40.1}, {1e3}, {1e7}, {0.5}, {0.01}, {1e5}},
        [](Rng& g) { return P{g.logUni(1e-2, 1e6)}; }, false);
}
TEST(DistributionIdentities, Laplace) {
    runFamily<LaplaceDistribution>(
        "laplace", [](const P& p) { return LaplaceDistribution(p[0], p[1]); },
        {{0, 1}, {2, 0.5}, {-7, 1e-300}, {1e300, 1}, {0, 1e-6}, {1e8, 1e6}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-6, 1e6)}; }, false);
}
TEST(DistributionIdentities, Pareto) {
    runFamily<ParetoDistribution>(
        "pareto", [](const P& p) { return ParetoDistribution(p[0], p[1]); },
        {{1, 2},
         {1, 1e-3},
         {1, 1e3},
         {1e-300, 1},
         {1e300, 2},
         {2, 1},
         {2, 3},
         {1, 1e-9},
         {1e-6, 0.01},
         {1e6, 100}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-3, 1e3)}; }, false);
}
TEST(DistributionIdentities, Rayleigh) {
    // sigma^2 under/overflows beyond 1e+-154 (D6): reject, or compute.
    runFamily<RayleighDistribution>(
        "rayleigh", [](const P& p) { return RayleighDistribution(p[0]); },
        {{1}, {0.37}, {1e-300}, {1e300}, {1e-155}, {1e155}, {1e-6}, {1e6}},
        [](Rng& g) { return P{g.logUni(1e-12, 1e12)}; }, false);
}
TEST(DistributionIdentities, Weibull) {
    runFamily<WeibullDistribution>(
        "weibull", [](const P& p) { return WeibullDistribution(p[0], p[1]); },
        {{1.5, 1},
         {0.5, 1},
         {1, 1},
         {1 + 1e-9, 1},
         {2, 1},
         {1e3, 1},
         {1e-3, 1},
         {3, 1e300},
         {3, 1e-300},
         {0.999, 2},
         {0.01, 1e-3},
         {100, 1e4}},
        [](Rng& g) { return P{g.logUni(1e-3, 1e4), g.logUni(1e-6, 1e6)}; }, false);
}
TEST(DistributionIdentities, Logistic) {
    runFamily<LogisticDistribution>(
        "logistic", [](const P& p) { return LogisticDistribution(p[0], p[1]); },
        {{0, 1}, {2, 0.5}, {-7, 1e-300}, {1e300, 1}, {0, 1e-6}, {1e8, 1e6}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-6, 1e6)}; }, false);
}
TEST(DistributionIdentities, Gumbel) {
    runFamily<GumbelDistribution>(
        "gumbel", [](const P& p) { return GumbelDistribution(p[0], p[1]); },
        {{0, 1}, {2, 0.5}, {-7, 1e-300}, {1e300, 1}, {0, 1e-6}, {1e8, 1e6}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e6), g.logUni(1e-6, 1e6)}; }, false);
}
TEST(DistributionIdentities, Erlang) {
    runFamily<ErlangDistribution>(
        "erlang", [](const P& p) { return ErlangDistribution(static_cast<int>(p[0]), p[1]); },
        {{2, 1},
         {19, 1},
         {20, 1},
         {21, 1},
         {1, 1e300},
         {3, 1e-300},
         {100000, 1},
         {1, 1e-3},
         {10000, 1e-3}},
        [](Rng& g) { return P{std::floor(g.logUni(1, 1e5)), g.logUni(1e-5, 1e5)}; }, false);
}
TEST(DistributionIdentities, FisherF) {
    // One huge df: beta_i at shape ratio >= 1e5 (D5-corvus).
    runFamily<FDistribution>(
        "fisher_f", [](const P& p) { return FDistribution(p[0], p[1]); },
        {{5, 10},
         {1, 1},
         {2, 2},
         {1, 1e6},
         {1e6, 1},
         {40, 40},
         {39.9, 40.1},
         {4, 4},
         {1e-2, 1e2},
         {3, 7},
         {0.01, 0.01},
         {1e4, 1e4}},
        [](Rng& g) { return P{g.logUni(1e-1, 1e4), g.logUni(1e-1, 1e4)}; }, false);
}
TEST(DistributionIdentities, InverseGamma) {
    runFamily<InverseGammaDistribution>(
        "inverse_gamma", [](const P& p) { return InverseGammaDistribution(p[0], p[1]); },
        {{3, 2},
         {20, 1},
         {19.999, 1},
         {1, 1},
         {0.5, 1e-3},
         {1e6, 1e6},
         {2, 1e300},
         {2, 1e-300},
         {0.01, 0.01},
         {1e4, 1e-3}},
        [](Rng& g) { return P{g.logUni(1e-3, 1e7), g.logUni(1e-5, 1e5)}; }, false);
}
TEST(DistributionIdentities, HalfNormal) {
    // sigma^2 under/overflows beyond 1e+-154 (D6): reject, or compute.
    runFamily<HalfNormalDistribution>(
        "half_normal", [](const P& p) { return HalfNormalDistribution(p[0]); },
        {{1}, {0.37}, {1e-300}, {1e300}, {1e-155}, {1e155}, {1e-6}, {1e6}},
        [](Rng& g) { return P{g.logUni(1e-12, 1e12)}; }, false);
}
TEST(DistributionIdentities, Bernoulli) {
    runFamily<BernoulliDistribution>(
        "bernoulli", [](const P& p) { return BernoulliDistribution(p[0]); },
        {{0.5}, {0.3}, {1e-300}, {1 - 1e-16}, {0}, {1}, {1e-6}},
        [](Rng& g) { return P{g.logUni(1e-9, 1 - 1e-9)}; }, true);
}
TEST(DistributionIdentities, Poisson) {
    runFamily<PoissonDistribution>(
        "poisson", [](const P& p) { return PoissonDistribution(p[0]); },
        {{4},
         {19.9},
         {20},
         {20.1},
         {1e3},
         {1e7},
         {1e9},
         {0.5},
         {1e-9},
         {1e-300},
         {1e15},
         {1e-3},
         {1e5}},
        [](Rng& g) { return P{g.logUni(1e-6, 1e8)}; }, true);
}
TEST(DistributionIdentities, Discrete) {
    // Q(p) at p = F(k) exactly: the tie (D7).
    runFamily<DiscreteDistribution>(
        "discrete",
        [](const P& p) {
            return DiscreteDistribution(static_cast<int>(p[0]), static_cast<int>(p[1]));
        },
        {{0, 9},
         {-5, 5},
         {7, 7},
         {1, 2},
         {-2147483647.0, 2147483647.0},
         {3, 2e9},
         {0, 1},
         {-1e6, 1e6}},
        [](Rng& g) {
            const double a = std::floor(g.logUni(1, 1e3));
            return P{a, a + std::floor(g.logUni(2, 1e6))};
        },
        true);
}
TEST(DistributionIdentities, TruncatedNormal) {
    // alpha = (a - mu)/sigma at the Hermite roots 0, +-1, sqrt(3): near-lower series stops on a
    // zero term (D1). (-40, -38): subnormal normaliser, CDF NaN (D9).
    runFamily<TruncatedNormalDistribution>(
        "truncated_normal",
        [](const P& p) { return TruncatedNormalDistribution(p[0], p[1], p[2], p[3]); },
        {{0, 1, -2, 2},
         {0, 2, 10, 12},
         {1e3, 1e2, -kInf, kInf},
         {0, 1, 0, kInf},
         {0, 1, -kInf, 0},
         {0, 1, 1, kInf},
         {0, 1, -1, 1},
         {0, 1, std::sqrt(3.0), kInf},
         {0, 1, 8, 9},
         {0, 1, -40, -38},
         {5, 1e-3, 4.999, 5.001},
         {0, 1, 3, 3.0000001}},
        [](Rng& g) {
            const double a = -3 + 6 * g.uni();
            return P{0.0, 1.0, a, a + g.logUni(0.1, 10)};
        },
        false);
}

// ---------------------------------------------------------------------------------------------
// Cross-class identities between independent code paths
// ---------------------------------------------------------------------------------------------
// exp(-lambda x) evaluated by two classes forms its argument by different roundings (lambda * x
// against x / scale); an argument error of eps |lambda x| is a relative error of the same size in
// the value, about 1e-10 at lambda x ~ 5e5 (a libm-dependent last bit: macos-15 CI, 2026-10-08).
constexpr double kExpArgCond = 4 * std::numeric_limits<double>::epsilon();

struct Cross {
    Report r;
    // extraRel: the reference formula's own conditioning at x (a cancelling difference), added to
    // the relative tolerance.
    void cmp(const std::string& id, const std::function<double(double)>& A,
             const std::function<double(double)>& B, const std::vector<double>& xs, double tol,
             const std::function<double(double)>& extraRel = nullptr) {
        for (double x : xs) {
            const double a = A(x), b = B(x);
            const double err = std::fabs(a - b);
            const double rel = tol + (extraRel ? extraRel(x) : 0.0);
            const double bound = rel * std::max({std::fabs(a), std::fabs(b), 1e-300});
            if (std::isnan(a) != std::isnan(b) || (!std::isnan(a) && !(err <= bound)))
                r.add(Where{"cross", {}, "cross", x}, id, err / bound,
                      "x=" + fmt(x) + " lhs=" + fmt(a) + " rhs=" + fmt(b));
        }
    }
};

std::vector<double> logGrid(Rng& g, int n, double lo, double hi) {
    std::vector<double> v;
    for (int i = 0; i < n; ++i)
        v.push_back(g.logUni(lo, hi));
    return v;
}
std::vector<double> probGrid(Rng& g, int n, double maxExp) {
    std::vector<double> v;
    for (int i = 0; i < n; ++i) {
        const double p = std::pow(10.0, -maxExp * g.uni());
        v.push_back(g.uni() < 0.5 ? p : 1 - p);
    }
    return v;
}

TEST(DistributionIdentitiesCrossClass, ContinuousSpecialCases) {
    Rng g(fnv1a("cross-continuous"));
    Cross c;
    const auto xs = logGrid(g, 200, 1e-6, 1e6);
    std::vector<double> xsym;
    for (double x : xs) {
        xsym.push_back(x);
        xsym.push_back(-x);
    }
    const auto unit = probGrid(g, 200, 15);
    // Cauchy(0,1) = StudentT(1)
    {
        CauchyDistribution ca(0.0, 1.0);
        StudentTDistribution t(1.0);
        c.cmp(
            "cauchy=t1 cdf", [&](double x) { return ca.getCumulativeProbability(x); },
            [&](double x) { return t.getCumulativeProbability(x); }, xsym, 1e-13);
        c.cmp(
            "cauchy=t1 pdf", [&](double x) { return ca.getProbability(x); },
            [&](double x) { return t.getProbability(x); }, xsym, 1e-13);
        c.cmp(
            "cauchy=t1 quantile", [&](double p) { return ca.getQuantile(p); },
            [&](double p) { return t.getQuantile(p); }, unit, 1e-11);
    }
    // Exponential(l) = Gamma(1, l) = Weibull(1, 1/l)
    for (double l : {0.3, 1.0, 7.0, 1e3, 1e-3}) {
        ExponentialDistribution e(l);
        GammaDistribution ga(1.0, l);
        WeibullDistribution w(1.0, 1.0 / l);
        std::vector<double> x;
        for (double v : xs)
            x.push_back(v / l);
        const std::string s = " l=" + fmt(l);
        c.cmp(
            "exp=gamma1 cdf" + s, [&](double v) { return e.getCumulativeProbability(v); },
            [&](double v) { return ga.getCumulativeProbability(v); }, x, 1e-13);
        c.cmp(
            "exp=gamma1 pdf" + s, [&](double v) { return e.getProbability(v); },
            [&](double v) { return ga.getProbability(v); }, x, 1e-13,
            [l](double v) { return kExpArgCond * l * v; });
        c.cmp(
            "exp=weibull1 cdf" + s, [&](double v) { return e.getCumulativeProbability(v); },
            [&](double v) { return w.getCumulativeProbability(v); }, x, 1e-13);
        c.cmp(
            "exp=weibull1 pdf" + s, [&](double v) { return e.getProbability(v); },
            [&](double v) { return w.getProbability(v); }, x, 1e-13,
            [l](double v) { return kExpArgCond * l * v; });
        c.cmp(
            "exp=gamma1 quantile" + s, [&](double p) { return e.getQuantile(p); },
            [&](double p) { return ga.getQuantile(p); }, unit, 1e-11);
    }
    // Gamma(a, 1/2) = ChiSquared(2a); integer a: Erlang(a, 1/2). Astride the Stirling switch.
    for (double a : {1.0, 2.0, 5.0, 19.0, 20.0, 21.0, 100.0, 1000.0}) {
        GammaDistribution ga(a, 0.5);
        ChiSquaredDistribution ch(2 * a);
        ErlangDistribution er(static_cast<int>(a), 0.5);
        std::vector<double> x;
        for (double p : probGrid(g, 100, 15))
            x.push_back(ga.getQuantile(p));
        const std::string s = " a=" + fmt(a);
        c.cmp(
            "chisq=gamma cdf" + s, [&](double v) { return ch.getCumulativeProbability(v); },
            [&](double v) { return ga.getCumulativeProbability(v); }, x, 1e-13);
        c.cmp(
            "chisq=gamma pdf" + s, [&](double v) { return ch.getProbability(v); },
            [&](double v) { return ga.getProbability(v); }, x, 1e-13);
        c.cmp(
            "erlang=gamma cdf" + s, [&](double v) { return er.getCumulativeProbability(v); },
            [&](double v) { return ga.getCumulativeProbability(v); }, x, 1e-13);
        c.cmp(
            "erlang=gamma logpdf" + s, [&](double v) { return er.getLogProbability(v); },
            [&](double v) { return ga.getLogProbability(v); }, x, 1e-13);
        c.cmp(
            "chisq=gamma quantile" + s, [&](double p) { return ch.getQuantile(p); },
            [&](double p) { return ga.getQuantile(p); }, unit, 1e-11);
        c.cmp(
            "erlang=gamma quantile" + s, [&](double p) { return er.getQuantile(p); },
            [&](double p) { return ga.getQuantile(p); }, unit, 1e-11);
    }
    // Rayleigh(s) = Weibull(2, s sqrt 2)
    for (double s : {0.5, 1.0, 30.0}) {
        RayleighDistribution ra(s);
        WeibullDistribution w(2.0, s * std::sqrt(2.0));
        std::vector<double> x;
        for (double v : xs)
            x.push_back(v * s);
        const std::string t = " s=" + fmt(s);
        c.cmp(
            "rayleigh=weibull2 cdf" + t, [&](double v) { return ra.getCumulativeProbability(v); },
            [&](double v) { return w.getCumulativeProbability(v); }, x, 1e-12);
        c.cmp(
            "rayleigh=weibull2 pdf" + t, [&](double v) { return ra.getProbability(v); },
            [&](double v) { return w.getProbability(v); }, x, 1e-12);
        c.cmp(
            "rayleigh=weibull2 quantile" + t, [&](double p) { return ra.getQuantile(p); },
            [&](double p) { return w.getQuantile(p); }, unit, 1e-11);
    }
    // LogNormal(mu, s) at x = Gaussian(mu, s) at log x
    for (double s : {0.1, 1.0, 5.0}) {
        LogNormalDistribution ln(0.3, s);
        GaussianDistribution ga(0.3, s);
        const std::string t = " s=" + fmt(s);
        c.cmp(
            "lognormal=gauss(log) cdf" + t,
            [&](double v) { return ln.getCumulativeProbability(v); },
            [&](double v) { return ga.getCumulativeProbability(std::log(v)); }, xs, 1e-12);
        c.cmp(
            "lognormal=gauss(log) quantile" + t,
            [&](double p) { return std::log(ln.getQuantile(p)); },
            [&](double p) { return ga.getQuantile(p); }, unit, 1e-10);
    }
    // InverseGamma(a, b): pdf(x) = GammaRate(a, b).pdf(1/x) / x^2; 1 - F(x) = GammaRate.F(1/x),
    // scored in the IG upper tail only (where 1 - F does not cancel).
    for (auto [a, b] :
         std::vector<std::pair<double, double>>{{3.0, 2.0}, {0.5, 1.0}, {25.0, 1.0}, {1e3, 1e3}}) {
        InverseGammaDistribution ig(a, b);
        GammaDistribution ga(a, b);
        std::vector<double> x;
        for (int i = 0; i < 100; ++i)
            x.push_back(ig.getQuantile(1 - std::pow(10.0, -6.0 * g.uni())));
        const std::string t = " a=" + fmt(a);
        c.cmp(
            "invgamma=gamma(1/x) pdf" + t, [&](double v) { return ig.getProbability(v); },
            [&](double v) { return ga.getProbability(1.0 / v) / (v * v); }, x, 1e-12);
        c.cmp(
            "invgamma=gamma(1/x) upper cdf" + t,
            [&](double v) { return 1.0 - ig.getCumulativeProbability(v); },
            [&](double v) { return ga.getCumulativeProbability(1.0 / v); }, x, 1e-9);
    }
    // FisherF(d1, d2) at x = Beta(d1/2, d2/2) at y = d1 x / (d1 x + d2). Scored for y <= 1/2:
    // above, 1 - y cancels in the map itself (not in either class).
    for (auto [d1, d2] : std::vector<std::pair<double, double>>{
             {5.0, 10.0}, {1.0, 1.0}, {50.0, 60.0}, {1e3, 3.0}}) {
        FDistribution fd(d1, d2);
        BetaDistribution be(d1 / 2, d2 / 2);
        std::vector<double> x;
        for (double y : probGrid(g, 100, 10))
            if (y <= 0.5)
                x.push_back(d2 * y / (d1 * (1 - y)));
        const std::string t = " d=" + fmt(d1) + "," + fmt(d2);
        c.cmp(
            "fisherf=beta cdf" + t, [&](double v) { return fd.getCumulativeProbability(v); },
            [&](double v) { return be.getCumulativeProbability(d1 * v / (d1 * v + d2)); }, x,
            1e-10);
        c.cmp(
            "fisherf=beta pdf" + t, [&](double v) { return fd.getProbability(v); },
            [&](double v) {
                const double y = d1 * v / (d1 * v + d2);
                return be.getProbability(y) * d1 * d2 / ((d1 * v + d2) * (d1 * v + d2));
            },
            x, 1e-10);
    }
    // StudentT(nu): for t < 0, F(t) = 1/2 Beta(nu/2, 1/2).F(nu / (nu + t^2))
    for (double nu : {1.0, 2.0, 3.5, 10.0, 100.0, 1e4}) {
        StudentTDistribution st(nu);
        BetaDistribution be(nu / 2, 0.5);
        std::vector<double> x;
        for (int i = 0; i < 100; ++i)
            x.push_back(-g.logUni(1e-3, 1e6));
        c.cmp(
            "studentt=beta lower cdf nu=" + fmt(nu),
            [&](double v) { return st.getCumulativeProbability(v); },
            [&](double v) { return 0.5 * be.getCumulativeProbability(nu / (nu + v * v)); }, x,
            1e-10);
    }
    // Beta(1,1) = Uniform(0,1); Beta reflection F_{a,b}(x) = 1 - F_{b,a}(1 - x)
    {
        BetaDistribution b11(1.0, 1.0);
        UniformDistribution u(0.0, 1.0);
        c.cmp(
            "beta11=uniform cdf", [&](double v) { return b11.getCumulativeProbability(v); },
            [&](double v) { return u.getCumulativeProbability(v); }, unit, 1e-14);
        c.cmp(
            "beta11=uniform pdf", [&](double v) { return b11.getProbability(v); },
            [&](double v) { return u.getProbability(v); }, unit, 1e-14);
        for (auto [a, b] : std::vector<std::pair<double, double>>{
                 {2.0, 7.0}, {0.5, 3.0}, {30.0, 2.0}, {1e3, 1e3}, {25.0, 1e3}}) {
            BetaDistribution ab(a, b), ba(b, a);
            // Grid from Gaussian-free fixed points in (0,1), so a Beta quantile defect cannot move
            // it.
            std::vector<double> x;
            const double m = a / (a + b), sd = std::sqrt(a * b / ((a + b) * (a + b) * (a + b + 1)));
            for (int i = -20; i <= 20; ++i) {
                const double v = m + 0.15 * i * sd;
                if (v > 0 && v < 1)
                    x.push_back(v);
            }
            const std::string t = " " + fmt(a) + "," + fmt(b);
            c.cmp(
                "beta reflection pdf" + t, [&](double v) { return ab.getProbability(v); },
                [&](double v) { return ba.getProbability(1.0 - v); }, x, 1e-12);
            c.cmp(
                "beta reflection cdf" + t, [&](double v) { return ab.getCumulativeProbability(v); },
                [&](double v) { return 1.0 - ba.getCumulativeProbability(1.0 - v); }, x, 1e-11);
        }
    }
    // Symmetric families: F(-d) + F(d) = 1 in the bulk
    {
        std::vector<double> ds;
        for (int i = 1; i <= 100; ++i)
            ds.push_back(0.05 * i);
        auto sym = [&](const std::string& id, const std::function<double(double)>& Fn) {
            c.cmp(
                "symmetry " + id, [&](double v) { return Fn(-v) + Fn(v); },
                [](double) { return 1.0; }, ds, 1e-14);
        };
        GaussianDistribution ga(0.0, 1.0);
        LaplaceDistribution la(0.0, 1.0);
        LogisticDistribution lo(0.0, 1.0);
        CauchyDistribution ca(0.0, 1.0);
        StudentTDistribution st(3.0);
        VonMisesDistribution vm(0.0, 2.0);
        sym("gaussian", [&](double v) { return ga.getCumulativeProbability(v); });
        sym("laplace", [&](double v) { return la.getCumulativeProbability(v); });
        sym("logistic", [&](double v) { return lo.getCumulativeProbability(v); });
        sym("cauchy", [&](double v) { return ca.getCumulativeProbability(v); });
        sym("studentt3", [&](double v) { return st.getCumulativeProbability(v); });
        sym("vonmises(0,2)", [&](double v) { return vm.getCumulativeProbability(v); });
    }
    c.r.finish();
}

TEST(DistributionIdentitiesCrossClass, TruncatedNormalAgainstGaussian) {
    // HalfNormal(s) = TN(0, s, 0, inf); Gaussian = TN(-inf, inf); windows against the Gaussian
    // formula written with the small-side tails, at alpha on Hermite roots (D1) and off them.
    Rng g(fnv1a("cross-tn"));
    Cross c;
    const auto unit = probGrid(g, 200, 15);
    GaussianDistribution n01(0.0, 1.0);
    const auto Phi = [&](double z) { return n01.getCumulativeProbability(z); };
    {
        HalfNormalDistribution h(1.5);
        TruncatedNormalDistribution tn(0.0, 1.5, 0.0, kInf);
        std::vector<double> x;
        for (int i = 1; i <= 200; ++i)
            x.push_back(1.5 * 4.0 * i / 200.0);
        for (double v : logGrid(g, 50, 1e-8, 0.4))
            x.push_back(v);
        c.cmp(
            "halfnormal=truncnormal cdf", [&](double v) { return h.getCumulativeProbability(v); },
            [&](double v) { return tn.getCumulativeProbability(v); }, x, 1e-12);
        c.cmp(
            "halfnormal=truncnormal pdf", [&](double v) { return h.getProbability(v); },
            [&](double v) { return tn.getProbability(v); }, x, 1e-12);
        c.cmp(
            "halfnormal=truncnormal quantile", [&](double p) { return h.getQuantile(p); },
            [&](double p) { return tn.getQuantile(p); }, unit, 1e-10);
    }
    {
        GaussianDistribution ga(0.7, 2.0);
        TruncatedNormalDistribution tg(0.7, 2.0, -kInf, kInf);
        std::vector<double> x;
        for (double v : logGrid(g, 100, 1e-3, 30)) {
            x.push_back(0.7 + v);
            x.push_back(0.7 - v);
        }
        c.cmp(
            "gauss=truncnormal(inf) cdf", [&](double v) { return ga.getCumulativeProbability(v); },
            [&](double v) { return tg.getCumulativeProbability(v); }, x, 1e-12);
        c.cmp(
            "gauss=truncnormal(inf) pdf", [&](double v) { return ga.getProbability(v); },
            [&](double v) { return tg.getProbability(v); }, x, 1e-12);
        c.cmp(
            "gauss=truncnormal(inf) quantile", [&](double p) { return ga.getQuantile(p); },
            [&](double p) { return tg.getQuantile(p); }, unit, 1e-10);
    }
    // Two-sided windows [a, b] around 0: F(x) = (Phi(x) - Phi(a)) / (Phi(b) - Phi(a)), each Phi
    // on its small side so nothing cancels beyond the difference itself (a < x < 0 or 0 < x < b
    // with a and b of opposite sign keeps the difference well conditioned).
    for (auto [a, b] :
         std::vector<std::pair<double, double>>{{-1.0, 1.0}, {-3.0, 3.0}, {-2.0, 0.5}}) {
        TruncatedNormalDistribution tn(0.0, 1.0, a, b);
        const double Z = Phi(b) - Phi(a);
        std::vector<double> x;
        for (int i = 1; i < 200; ++i)
            x.push_back(a + (b - a) * i / 200.0);
        const std::string t = " [" + fmt(a) + "," + fmt(b) + "]";
        c.cmp(
            "truncnormal window cdf" + t, [&](double v) { return tn.getCumulativeProbability(v); },
            [&](double v) { return (Phi(v) - Phi(a)) / Z; }, x, 1e-11,
            [&](double v) { return 16 * kEps * (Phi(v) + Phi(a)) / std::fabs(Phi(v) - Phi(a)); });
        c.cmp(
            "truncnormal window pdf" + t, [&](double v) { return tn.getProbability(v); },
            [&](double v) { return n01.getProbability(v) / Z; }, x, 1e-12);
    }
    // One-sided upper windows [a, inf), a > 0: F(x) = (Phi(-a) - Phi(-x)) / Phi(-a).
    for (double a : {1.0, std::sqrt(3.0), 0.5, 2.5}) {
        TruncatedNormalDistribution tn(0.0, 1.0, a, kInf);
        const double Za = Phi(-a);
        std::vector<double> x;
        for (int i = 1; i <= 200; ++i)
            x.push_back(a + 4.0 * i / 200.0);
        for (double v : logGrid(g, 50, 1e-8, 0.25))
            x.push_back(a + v);
        c.cmp(
            "truncnormal upper cdf a=" + fmt(a),
            [&](double v) { return tn.getCumulativeProbability(v); },
            [&](double v) { return (Za - Phi(-v)) / Za; }, x, 1e-11,
            [&](double v) { return 16 * kEps * (Za + Phi(-v)) / std::fabs(Za - Phi(-v)); });
    }
    c.r.finish();
}

TEST(DistributionIdentitiesCrossClass, DiscreteSpecialCases) {
    Rng g(fnv1a("cross-discrete"));
    Cross c;
    // Bernoulli(p) = Binomial(1, p); Geometric(p) = NegativeBinomial(1, p)
    for (double p : {0.3, 1e-6, 1 - 1e-6, 0.5}) {
        BernoulliDistribution be(p);
        BinomialDistribution bi(1, p);
        const std::vector<double> k = {0.0, 1.0};
        const std::string t = " p=" + fmt(p);
        c.cmp(
            "bernoulli=binomial1 pmf" + t, [&](double v) { return be.getProbability(v); },
            [&](double v) { return bi.getProbability(v); }, k, 1e-14);
        c.cmp(
            "bernoulli=binomial1 cdf" + t, [&](double v) { return be.getCumulativeProbability(v); },
            [&](double v) { return bi.getCumulativeProbability(v); }, k, 1e-14);
        GeometricDistribution ge(p);
        NegativeBinomialDistribution nb(1.0, p);
        std::vector<double> ks;
        for (int i = 0; i < 40; ++i)
            ks.push_back(std::floor(std::log1p(-g.uni()) / std::log1p(-std::min(p, 0.999))));
        c.cmp(
            "geometric=negbin1 pmf" + t, [&](double v) { return ge.getProbability(v); },
            [&](double v) { return nb.getProbability(v); }, ks, 1e-12);
        c.cmp(
            "geometric=negbin1 cdf" + t, [&](double v) { return ge.getCumulativeProbability(v); },
            [&](double v) { return nb.getCumulativeProbability(v); }, ks, 1e-12);
    }
    // Geometric closed form: F(k) = 1 - (1-p)^(k+1) = -expm1((k+1) log1p(-p)) (D4)
    for (double p : {0.3, 1e-3, 1e-6, 1e-9, 1e-12}) {
        GeometricDistribution ge(p);
        std::vector<double> ks;
        for (double u : {1e-3, 0.1, 0.5, 0.9, 0.99, 0.999})
            ks.push_back(std::floor(std::log1p(-u) / std::log1p(-p)));
        c.cmp(
            "geometric closed-form cdf p=" + fmt(p),
            [&](double k) { return ge.getCumulativeProbability(k); },
            [&](double k) { return -std::expm1((k + 1) * std::log1p(-p)); }, ks, 1e-12);
    }
    // Binomial complement F_{n,p}(k) = 1 - F_{n,1-p}(n-k-1), pmf reflection
    for (auto [n, p] :
         std::vector<std::pair<int, double>>{{20, 0.3}, {1000, 0.01}, {1000000, 0.4}}) {
        BinomialDistribution bp(n, p), bq(n, 1 - p);
        std::vector<double> ks;
        const double m = n * p, sd = std::sqrt(n * p * (1 - p));
        for (int i = -10; i <= 10; ++i) {
            const double k = std::floor(m + 0.3 * i * sd);
            if (k >= 0 && k < n)
                ks.push_back(k);
        }
        const std::string t = " n=" + std::to_string(n);
        c.cmp(
            "binomial complement" + t, [&](double k) { return bp.getCumulativeProbability(k); },
            [&](double k) { return 1.0 - bq.getCumulativeProbability(n - k - 1); }, ks, 1e-11);
        c.cmp(
            "binomial pmf reflection" + t, [&](double k) { return bp.getProbability(k); },
            [&](double k) { return bq.getProbability(n - k); }, ks, 1e-12);
    }
    c.r.finish();
}

}  // namespace
