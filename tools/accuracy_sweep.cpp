/**
 * @file accuracy_sweep.cpp
 * @brief Issue #46: deterministic bit-exact accuracy sweep across all 27 distributions.
 *
 * Evaluates the scalar AND batch (span/SIMD) code paths of every distribution's
 * pdf/logpdf/cdf/quantile methods over fixed, support-aware characterization grids
 * and dumps the results as bit-exact hex to a CSV consumed by a sibling mpmath
 * oracle (tools/accuracy_vs_mpmath.py, owned by a different agent). This tool
 * replaces the issue's original pylibstats approach -- pylibstats pins to a
 * *released* libstats and would characterize the wrong code; this binary always
 * characterizes the code actually checked out.
 *
 * Determinism is the load-bearing property: no randomness anywhere. The x/k
 * grids are built from each distribution's OWN getQuantile()/getMean()/
 * getVariance(), which makes grid construction support-aware for free (a
 * quantile near p=0 or p=1 lands deep in whatever tail the support allows)
 * without hand-deriving per-family tail formulas.
 *
 * Output contract (do not change without updating the sibling tool):
 *   # libstats accuracy_sweep v1
 *   # commit=<short sha>  isa=<SIMDPolicy::getLevelString()>  date=YYYY-MM-DD
 *   dist,method,p1_bits,p2_bits,x_bits,scalar_bits,batch_bits
 *   ...
 *   # skipped_quantile <dist> <n>          (only emitted when n > 0)
 *
 * Usage: accuracy_sweep <output.csv>
 */

// Use tool_utils.h for the consolidated (LIBSTATS_FULL_INTERFACE) libstats.h include,
// which pulls in all 27 distribution headers.
#include "libstats/platform/simd_policy.h"
#include "tool_utils.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

//==============================================================================
// Bit-pattern formatting
//==============================================================================

std::uint64_t bitsOf(double v) {
    std::uint64_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

std::string hex64(std::uint64_t bits) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << bits;
    return oss.str();
}

std::string fmtBits(double v) {
    return hex64(bitsOf(v));
}

//==============================================================================
// Row / CSV plumbing
//==============================================================================

struct Row {
    std::string dist;
    std::string method;
    std::string p1;
    std::string p2;
    std::string x;
    std::string scalar;
    std::string batch;

    std::string toCsvLine() const {
        std::string line;
        line.reserve(dist.size() + method.size() + p1.size() + p2.size() + x.size() +
                     scalar.size() + batch.size() + 8);
        line += dist;
        line += ',';
        line += method;
        line += ',';
        line += p1;
        line += ',';
        line += p2;
        line += ',';
        line += x;
        line += ',';
        line += scalar;
        line += ',';
        line += batch;
        return line;
    }
};

Row makeRow(const std::string& dist, const std::string& method, double p1, double p2, bool hasP2,
            double x, double scalar, double batch, bool hasBatch) {
    Row r;
    r.dist = dist;
    r.method = method;
    r.p1 = fmtBits(p1);
    r.p2 = hasP2 ? fmtBits(p2) : std::string("0x0");
    r.x = fmtBits(x);
    r.scalar = fmtBits(scalar);
    r.batch = hasBatch ? fmtBits(batch) : std::string("-");
    return r;
}

// Fixed, dependency-ordered dist name list (matches the issue #46 contract and
// drives both the summary table and the skip-count trailer, so both keep a
// deterministic dist order regardless of sweep call order).
// The v2.4.0 eight are APPENDED, never interleaved: the sweep's row order is
// the sweepAll() call order, and the #46 comparison baseline (63 violations on
// the 6063-row grid) is only meaningful if every pre-v2.4.0 row keeps its
// byte-for-byte identity and its CSV line number, which the appendix table of
// docs/ACCURACY_CHARACTERIZATION.md indexes by.
constexpr std::array<const char*, 27> kDistNames = {"gaussian",    "lognormal",
                                                    "exponential", "uniform",
                                                    "poisson",     "gamma",
                                                    "discrete",    "student_t",
                                                    "cauchy",      "von_mises",
                                                    "binomial",    "negative_binomial",
                                                    "geometric",   "beta",
                                                    "chi_squared", "laplace",
                                                    "pareto",      "rayleigh",
                                                    "weibull",     "logistic",
                                                    "gumbel",      "erlang",
                                                    "fisher_f",    "inverse_gamma",
                                                    "half_normal", "truncated_normal",
                                                    "bernoulli"};

class Sink {
   public:
    Sink() {
        for (const char* name : kDistNames) {
            counts_.emplace_back(name, 0);
            skips_.emplace_back(name, 0);
        }
    }

    void addRow(Row row) {
        bump(counts_, row.dist);
        rows_.push_back(std::move(row));
    }

    void addSkip(const std::string& dist) { bump(skips_, dist); }

    const std::vector<Row>& rows() const { return rows_; }
    const std::vector<std::pair<std::string, int>>& counts() const { return counts_; }
    const std::vector<std::pair<std::string, int>>& skips() const { return skips_; }

   private:
    static void bump(std::vector<std::pair<std::string, int>>& v, const std::string& dist) {
        for (auto& [name, n] : v) {
            if (name == dist) {
                ++n;
                return;
            }
        }
        v.emplace_back(dist, 1);
    }

    std::vector<Row> rows_;
    std::vector<std::pair<std::string, int>> counts_;
    std::vector<std::pair<std::string, int>> skips_;
};

//==============================================================================
// Deterministic grids (no randomness anywhere)
//==============================================================================

// Probability grid used to *construct* the continuous x-grid via getQuantile(p).
// Log-spaced deep into both tails (down to p=1e-300, i.e. F~1e-300 in the left
// tail / 1-F~1e-300 in the right tail where the support allows it reaching that
// far) plus a denser cluster around the center. Symmetric about p=0.5 by
// construction. ~57 points.
const std::vector<double>& xConstructionPGrid() {
    static const std::vector<double> grid = [] {
        std::vector<double> lower;  // all values strictly < 0.5
        for (int e : {300, 280, 260, 240, 220, 200, 180, 160, 140, 120, 100, 80, 60, 40, 20}) {
            lower.push_back(std::pow(10.0, -e));
        }
        for (double e : {15.0, 10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.7, 0.52}) {
            lower.push_back(std::pow(10.0, -e));
        }
        for (double p : {0.35, 0.4, 0.45}) {
            lower.push_back(p);
        }
        // Strictly ascending by construction: both exponent lists descend
        // (so the values ascend), and 10^-0.52 ~ 0.302 < 0.35. Assert
        // rather than sort+unique -- GCC 13's -Werror=strict-overflow
        // fires a pointer-wraparound false positive when it constexpr-
        // expands std::sort inside this immediately-invoked initializer.
        for (std::size_t i = 1; i < lower.size(); ++i) {
            assert(lower[i - 1] < lower[i]);
        }

        std::vector<double> full;
        full.reserve(lower.size() * 2 + 1);
        for (double p : lower) {
            full.push_back(p);
        }
        full.push_back(0.5);
        for (auto it = lower.rbegin(); it != lower.rend(); ++it) {
            full.push_back(1.0 - *it);
        }
        return full;
    }();
    return grid;
}

// Probability grid for method=quantile rows (the contract's fixed 15-point list).
const std::vector<double>& quantilePGrid() {
    static const std::vector<double> grid = {1e-300, 1e-15, 1e-10,      1e-6,        1e-3,
                                             0.01,   0.1,   0.25,       0.5,         0.75,
                                             0.9,    0.99,  1.0 - 1e-6, 1.0 - 1e-10, 1.0 - 1e-15};
    return grid;
}

// Boundary epsilon offsets applied to both-sides-bounded supports (uniform, beta),
// per the issue's grid spec.
const std::array<double, 3>& boundaryEpsilons() {
    static const std::array<double, 3> eps = {1e-15, 1e-9, 1e-4};
    return eps;
}

// k-landmark multipliers (in units of standard deviation, both directions around
// the mean) used to build the discrete integer grid. Reaches far enough into the
// tail (mean +/- 16000*sd) that cdf reaches 1-1e-12 for every discrete instance
// used below; landmarks are clamped to the distribution's own support bounds.
const std::array<double, 23>& discreteTMultipliers() {
    static const std::array<double, 23> t = {0.5, 1,   1.5, 2,   3,    4,    5,    6,
                                             8,   10,  15,  20,  30,   50,   75,   100,
                                             150, 250, 400, 650, 1000, 4000, 16000};
    return t;
}

//==============================================================================
// Generic grid builders (templated on the concrete distribution type; every
// distribution exposes the same DistributionInterface surface used here)
//==============================================================================

// Builds the continuous x-grid for one instance: quantile-derived points spanning
// both tails as deep as the support allows, plus explicit near-boundary points
// for both-sides-bounded supports (uniform, beta). Sorted, deduplicated.
template <typename Dist>
std::vector<double> buildContinuousXGrid(Dist& dist, bool boundedBothSides) {
    std::vector<double> xs;
    for (double p : xConstructionPGrid()) {
        try {
            double x = dist.getQuantile(p);
            if (std::isfinite(x)) {
                xs.push_back(x);
            }
        } catch (...) {
            // Support doesn't reach this deep at this p for this instance; skip.
        }
    }
    if (boundedBothSides) {
        double lo = dist.getSupportLowerBound();
        double hi = dist.getSupportUpperBound();
        // Finiteness guard, added with the v2.4.0 eight: truncated_normal is the
        // first distribution whose support bounds are RUNTIME parameters, and its
        // doubly-infinite instance would otherwise push +/-inf in here as ordinary
        // grid points, duplicating the specials block. A no-op for uniform and
        // beta, whose bounds are finite for every swept instance (verified by
        // diffing the pre-change CSV against the head of the post-change one).
        for (double eps : boundaryEpsilons()) {
            if (std::isfinite(lo)) {
                xs.push_back(lo + eps);
            }
            if (std::isfinite(hi)) {
                xs.push_back(hi - eps);
            }
        }
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
}

// Builds the discrete k-grid for one instance from the distribution's own
// mean/variance/support -- landmark points (0, 1, 2, mean, mean +/- t*sd for a
// deterministic multiplier ladder, and the support edges), not sampling.
template <typename Dist>
std::vector<double> buildDiscreteKGrid(Dist& dist) {
    double lo = dist.getSupportLowerBound();
    double hi = dist.getSupportUpperBound();
    double mean = dist.getMean();
    double var = dist.getVariance();
    double sd = (std::isfinite(var) && var > 0.0) ? std::sqrt(var) : 1.0;
    if (!std::isfinite(mean)) {
        mean = 0.0;
    }

    auto clamp = [&](double k) {
        if (std::isfinite(lo) && k < lo) {
            k = lo;
        }
        if (std::isfinite(hi) && k > hi) {
            k = hi;
        }
        return k;
    };

    std::vector<double> ks;
    auto add = [&](double k) { ks.push_back(clamp(std::floor(k + 0.5))); };

    add(0.0);
    add(1.0);
    add(2.0);
    add(mean);
    for (double t : discreteTMultipliers()) {
        add(mean - t * sd);
        add(mean + t * sd);
    }
    if (std::isfinite(lo)) {
        add(lo);
        add(lo + 1.0);
        add(lo + 2.0);
    }
    if (std::isfinite(hi)) {
        add(hi);
        add(hi - 1.0);
        add(hi - 2.0);
    }

    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    return ks;
}

//==============================================================================
// Generic row emission (identical method surface across all 19 distributions)
//==============================================================================

template <typename Dist>
void emitPdfLogpdfCdfRows(Dist& dist, const std::string& distName, double p1, double p2, bool hasP2,
                          std::vector<double> xs, bool includeSpecials, Sink& sink) {
    if (includeSpecials) {
        // At the FRONT, not the back: the batch kernels process floor(n/W)*W lanes in
        // the vector body and hand the remainder to scalar libm. With 43 finite x the
        // three specials at the tail were evaluated by libm on AVX-512 (43 mod 8 = 3),
        // so the sweep never saw what the kernels do with NaN/inf (review 2026-08-21,
        // NV2 -- this is how lognormal's batch cdf(NaN) = 1 escaped #102).
        xs.insert(xs.begin(), {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()});
    }

    using ScalarFn = double (Dist::*)(double) const;
    using BatchFn = void (Dist::*)(std::span<const double>, std::span<double>,
                                   const stats::detail::PerformanceHint&) const;

    struct Method {
        const char* name;
        ScalarFn scalarFn;
        BatchFn batchFn;
    };
    const std::array<Method, 3> methods = {
        Method{"pdf", static_cast<ScalarFn>(&Dist::getProbability),
               static_cast<BatchFn>(&Dist::getProbability)},
        Method{"logpdf", static_cast<ScalarFn>(&Dist::getLogProbability),
               static_cast<BatchFn>(&Dist::getLogProbability)},
        Method{"cdf", static_cast<ScalarFn>(&Dist::getCumulativeProbability),
               static_cast<BatchFn>(&Dist::getCumulativeProbability)},
    };

    stats::detail::PerformanceHint hint;
    hint.strategy = stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;

    std::vector<double> batchOut(xs.size());
    for (const auto& m : methods) {
        std::fill(batchOut.begin(), batchOut.end(), 0.0);
        (dist.*(m.batchFn))(std::span<const double>(xs), std::span<double>(batchOut), hint);
        for (std::size_t i = 0; i < xs.size(); ++i) {
            double scalar = (dist.*(m.scalarFn))(xs[i]);
            sink.addRow(makeRow(distName, m.name, p1, p2, hasP2, xs[i], scalar, batchOut[i],
                                /*hasBatch=*/true));
        }
    }
}

template <typename Dist>
void emitQuantileRows(Dist& dist, const std::string& distName, double p1, double p2, bool hasP2,
                      Sink& sink) {
    for (double p : quantilePGrid()) {
        try {
            double q = dist.getQuantile(p);
            sink.addRow(
                makeRow(distName, "quantile", p1, p2, hasP2, p, q, 0.0, /*hasBatch=*/false));
        } catch (...) {
            sink.addSkip(distName);
        }
    }
}

// Drives one continuous distribution across its 3 parameter instances.
template <typename Dist, typename Factory>
void sweepContinuous(const std::string& distName, Factory factory,
                     const std::array<std::pair<double, double>, 3>& instances, bool hasP2,
                     bool boundedBothSides, Sink& sink) {
    for (auto [p1, p2] : instances) {
        Dist dist = factory(p1, p2);
        std::vector<double> xs = buildContinuousXGrid(dist, boundedBothSides);
        emitPdfLogpdfCdfRows(dist, distName, p1, p2, hasP2, xs, /*includeSpecials=*/true, sink);
        emitQuantileRows(dist, distName, p1, p2, hasP2, sink);
    }
}

// Drives one FOUR-parameter continuous distribution (truncated_normal is the
// only one) across its 3 parameter instances.
//
// The CSV carries exactly two parameter columns and widening it is not an
// option -- p3_bits/p4_bits would rewrite all 6063 pre-v2.4.0 rows, which the
// characterization baseline forbids. So p1_bits/p2_bits keep their documented
// meaning ("the two constructor parameters in DECLARATION order", here mean and
// standardDeviation) and the oracle recovers the truncation window from a small
// instance table keyed on that pair -- which is why the three instances below
// must have PAIRWISE-DISTINCT (mean, sd). tools/accuracy_vs_mpmath.py's
// _TN_WINDOWS is the other half of this contract and raises rather than guesses
// on an unknown key, so the two tables cannot drift apart silently.
template <typename Dist, typename Factory>
void sweepContinuous4(const std::string& distName, Factory factory,
                      const std::array<std::array<double, 4>, 3>& instances, bool boundedBothSides,
                      Sink& sink) {
    for (const auto& inst : instances) {
        Dist dist = factory(inst[0], inst[1], inst[2], inst[3]);
        std::vector<double> xs = buildContinuousXGrid(dist, boundedBothSides);
        emitPdfLogpdfCdfRows(dist, distName, inst[0], inst[1], /*hasP2=*/true, xs,
                             /*includeSpecials=*/true, sink);
        emitQuantileRows(dist, distName, inst[0], inst[1], /*hasP2=*/true, sink);
    }
}

// Drives one discrete distribution across its 3 parameter instances.
template <typename Dist, typename Factory>
void sweepDiscrete(const std::string& distName, Factory factory,
                   const std::array<std::pair<double, double>, 3>& instances, bool hasP2,
                   Sink& sink) {
    for (auto [p1, p2] : instances) {
        Dist dist = factory(p1, p2);
        std::vector<double> ks = buildDiscreteKGrid(dist);
        // Specials included since #102: skipping them here is why no sweep on any
        // ISA could see the discrete-family batch NaN victims (poisson, discrete,
        // binomial, negative_binomial, geometric).
        emitPdfLogpdfCdfRows(dist, distName, p1, p2, hasP2, ks, /*includeSpecials=*/true, sink);
        emitQuantileRows(dist, distName, p1, p2, hasP2, sink);
    }
}

//==============================================================================
// Per-distribution instance tables (typical / small-boundary / stressed) and
// factories. Parameter values are chosen within each header's documented valid
// range (see the @param Doxygen comments checked against directly).
//==============================================================================

//==============================================================================
// Switch-band instances (v2.5.0 oracle hardening, 2026-10-11). The three
// instances per distribution above sit at typical / small / stressed
// parameters and never reach most of the kernels' internal switches; a grid
// that never visits a band cannot see a defect in it (the unswept-band probe
// found the von Mises tail collapse above kappa = 1000 that way). Each
// instance below sits on one side of a switch, its neighbour on the other.
// Rows are APPENDED after every pre-existing row (same CSV contract), on a
// thinner x grid than the base instances plus points at the switch itself,
// so the oracle's runtime stays bounded.
//==============================================================================

// 23 probabilities: 11 tail decades either side of the median, plus 0.5.
const std::vector<double>& bandPGrid() {
    static const std::vector<double> grid = [] {
        const std::vector<double> lower = {1e-300, 1e-240, 1e-180, 1e-120, 1e-60, 1e-20,
                                           1e-10,  1e-6,   1e-3,   0.1,    0.3};
        std::vector<double> full(lower);
        full.push_back(0.5);
        for (auto it = lower.rbegin(); it != lower.rend(); ++it) {
            full.push_back(1.0 - *it);
        }
        return full;
    }();
    return grid;
}

// x = Q(p) over bandPGrid(), the boundary epsilons where the support is
// bounded, and `extra` (non-finite values dropped); sorted, unique.
template <typename Dist>
std::vector<double> buildBandXGrid(Dist& dist, bool boundedBothSides,
                                   const std::vector<double>& extra) {
    std::vector<double> xs;
    for (double p : bandPGrid()) {
        try {
            const double x = dist.getQuantile(p);
            if (std::isfinite(x)) {
                xs.push_back(x);
            }
        } catch (...) {
        }
    }
    if (boundedBothSides) {
        const double lo = dist.getSupportLowerBound();
        const double hi = dist.getSupportUpperBound();
        for (double eps : boundaryEpsilons()) {
            if (std::isfinite(lo)) {
                xs.push_back(lo + eps);
            }
            if (std::isfinite(hi)) {
                xs.push_back(hi - eps);
            }
        }
    }
    for (double e : extra) {
        if (std::isfinite(e)) {
            xs.push_back(e);
        }
    }
    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
}

template <typename Dist>
void emitBandInstance(Dist& dist, const std::string& name, double p1, double p2, bool hasP2,
                      bool boundedBothSides, const std::vector<double>& extra, Sink& sink) {
    std::vector<double> xs = buildBandXGrid(dist, boundedBothSides, extra);
    emitPdfLogpdfCdfRows(dist, name, p1, p2, hasP2, xs, /*includeSpecials=*/false, sink);
    emitQuantileRows(dist, name, p1, p2, hasP2, sink);
}

// x and its k nearest doubles on each side.
std::vector<double> neighbours(double x, int k) {
    std::vector<double> v{x};
    double lo = x;
    double hi = x;
    for (int i = 0; i < k; ++i) {
        lo = std::nextafter(lo, -std::numeric_limits<double>::infinity());
        hi = std::nextafter(hi, std::numeric_limits<double>::infinity());
        v.push_back(lo);
        v.push_back(hi);
    }
    return v;
}

void append(std::vector<double>& v, const std::vector<double>& w) {
    v.insert(v.end(), w.begin(), w.end());
}

// lambda with sign(lambda - 1) * sqrt(2 (lambda - 1 - log lambda)) = eta: Temme's
// uniform variable, so x = a * lambdaOfEta(eta) puts a gamma point at a chosen
// eta (the library's Temme band is |eta| <= 0.6 from a = 1e4). Bisection.
double lambdaOfEta(double eta) {
    const double target = 0.5 * eta * eta;
    double lo = eta < 0.0 ? 1e-300 : 1.0;
    double hi = eta < 0.0 ? 1.0 : 1e6;
    for (int i = 0; i < 400; ++i) {
        const double mid =
            eta < 0.0 ? std::exp(0.5 * (std::log(lo) + std::log(hi))) : 0.5 * (lo + hi);
        const double f = mid - 1.0 - std::log(mid);
        const bool moveLo = eta < 0.0 ? (f > target) : (f < target);
        (moveLo ? lo : hi) = mid;
    }
    return 0.5 * (lo + hi);
}

// x wrapped to (-pi, pi], as VonMisesDistribution wraps.
double wrapToPi(double x) {
    constexpr double pi = std::numbers::pi;
    double t = std::fmod(x, 2.0 * pi);
    if (t <= -pi) {
        t += 2.0 * pi;
    }
    if (t > pi) {
        t -= 2.0 * pi;
    }
    return t;
}

void sweepSwitchBands(Sink& sink) {
    using stats::BetaDistribution;
    using stats::BinomialDistribution;
    using stats::ChiSquaredDistribution;
    using stats::ErlangDistribution;
    using stats::FDistribution;
    using stats::GammaDistribution;
    using stats::InverseGammaDistribution;
    using stats::NegativeBinomialDistribution;
    using stats::PoissonDistribution;
    using stats::StudentTDistribution;
    using stats::TruncatedNormalDistribution;
    using stats::VonMisesDistribution;
    const double inf = std::numeric_limits<double>::infinity();
    const auto below = [](double x) { return std::nextafter(x, 0.0); };
    const auto above = [&](double x) { return std::nextafter(x, inf); };

    // gamma(a, 1): the Stirling prefactor from a = 20, the Temme expansion
    // from a = 1e4 (|eta| <= 0.6), Wilson-Hilferty quantiles from a = 1e16; the
    // series / continued-fraction switch at x = a + 1 and the eta = +-0.6 Temme
    // edges get their own points.
    for (double a : {19.9, 20.0, 20.1, 100.0, 1e3, 9999.0, 1e4, 10001.0, 1e5, 1e8, below(1e16),
                     1e16, above(1e16)}) {
        GammaDistribution d(a, 1.0);
        std::vector<double> extra;
        append(extra, neighbours(a + 1.0, 1));
        extra.push_back(a - 1.0);  // the mode
        for (double eta : {-3.0, -1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 3.0}) {
            const double x = a * lambdaOfEta(eta);
            if (std::fabs(eta) == 0.6) {
                append(extra, neighbours(x, 1));
            } else {
                extra.push_back(x);
            }
        }
        emitBandInstance(d, "gamma", a, 1.0, true, false, extra, sink);
    }
    // chi_squared(k) = gamma(k/2, 1/2), erlang(k, 1) and inverse_gamma(a, 1)
    // ride the same kernels: both sides of a = 20, 1e4 and 1e16.
    for (double k : {39.8, 40.0, 40.2, 19998.0, 2e4, 20002.0, below(2e16), 2e16}) {
        ChiSquaredDistribution d(k);
        emitBandInstance(d, "chi_squared", k, 0.0, false, false, {}, sink);
    }
    for (int k : {19, 20, 9999, 10000}) {
        ErlangDistribution d(k, 1.0);
        emitBandInstance(d, "erlang", static_cast<double>(k), 1.0, true, false, {}, sink);
    }
    for (double a : {19.9, 20.0, 9999.0, 1e4, below(1e16), 1e16}) {
        InverseGammaDistribution d(a, 1.0);
        emitBandInstance(d, "inverse_gamma", a, 1.0, true, false, {}, sink);
    }

    // beta: the tail-integral path (a >= 1, b < 1) against b = 1, the tiny-shape
    // path below 1e-3, a huge shape against a small one in both orders. Points
    // on the continued fraction's branch point x_b = (a + 1)/(a + b + 2) and its
    // mirror, and at 1 - 2^-k / 2^-10k where 1 - x and x are exact.
    for (auto [a, b] : std::vector<std::pair<double, double>>{{10.0, 0.1},
                                                              {1e3, 0.5},
                                                              {2.0, below(1e-3)},
                                                              {2.0, 1e-3},
                                                              {0.5, below(1e-3)},
                                                              {0.5, 1e-3},
                                                              {3.0, below(1.0)},
                                                              {3.0, 1.0},
                                                              {0.5, 1e3}}) {
        BetaDistribution d(a, b);
        std::vector<double> extra;
        append(extra, neighbours((a + 1.0) / (a + b + 2.0), 1));
        append(extra, neighbours(1.0 - (b + 1.0) / (a + b + 2.0), 1));
        for (int k : {1, 3, 8, 20, 40, 53}) {
            extra.push_back(1.0 - std::ldexp(1.0, -k));
            extra.push_back(std::ldexp(1.0, -10 * k));
        }
        emitBandInstance(d, "beta", a, b, true, true, extra, sink);
    }
    // fisher_f(d1, d2) is beta(d1/2, d2/2) in y = d1 x / (d1 x + d2).
    for (auto [d1, d2] : std::vector<std::pair<double, double>>{
             {2.0, 0.2}, {20.0, 1.0}, {2.0, below(2e-3)}, {2.0, 2e-3}}) {
        FDistribution d(d1, d2);
        emitBandInstance(d, "fisher_f", d1, d2, true, false, {}, sink);
    }

    // von_mises(mu, kappa): kappa above the exact-CDF range, mu != 0 (the
    // shift-and-wrap path), points at fixed offsets from mu and at the antimode.
    for (double kappa : {1e3, 1e4, 1e8}) {
        for (double mu : {0.0, 2.5, -3.0}) {
            VonMisesDistribution d(mu, kappa);
            std::vector<double> extra;
            for (double t :
                 {-3.0, -2.0, -1.0, -0.1, -0.01, -1e-3, 1e-3, 0.01, 0.1, 1.0, 2.0, 3.0}) {
                extra.push_back(wrapToPi(mu + t));
            }
            append(extra, neighbours(wrapToPi(mu + std::numbers::pi), 1));
            emitBandInstance(d, "von_mises", mu, kappa, true, false, extra, sink);
        }
    }

    // student_t(nu): the batch switch at (nu + 1)/2 = 16, the Stirling / BGRAT
    // switch at nu = 40, huge nu up to where x = nu/(nu + t^2) rounds to 1
    // (~4e17). Points at u = t^2/nu = 1 and on the (a + 1) u = 1.5
    // tail/central switch, both signs.
    for (double nu : {20.0, 30.9, 31.0, 39.9, 40.0, 100.0, 1e3, 1e5, 1e7, 1e11, 1e17, below(4e17),
                      4e17, 1e18}) {
        StudentTDistribution d(nu);
        std::vector<double> extra;
        const double s1 = std::sqrt(nu);
        const double s2 = std::sqrt(1.5 / (0.5 * nu + 1.0) * nu);
        for (double s : {s1, -s1, s2, -s2}) {
            append(extra, neighbours(s, 1));
        }
        for (double t : {-1e3, -100.0, -38.0, 10.0, 38.0}) {
            extra.push_back(t);
        }
        emitBandInstance(d, "student_t", nu, 0.0, false, false, extra, sink);
    }

    // truncated_normal: half-infinite windows, both sides, in the tail and
    // straddling the mean. The (mean, sd) pairs are the oracle's window keys
    // (see sweepContinuous4): distinct from each other and from the three base
    // instances, and listed in _TN_WINDOWS in tools/accuracy_vs_mpmath.py.
    // Points just inside the finite bound.
    for (const auto& w : std::vector<std::array<double, 4>>{{0.0, 1.5, 7.5, inf},
                                                            {0.0, 0.5, 15.0, inf},
                                                            {4.0, 1.0, 2.0, inf},
                                                            {1.0, 1.0, -inf, -4.0},
                                                            {2.0, 1.0, -inf, -28.0},
                                                            {3.0, 1.0, -inf, 5.0},
                                                            {0.0, 0.25, -inf, 0.0}}) {
        TruncatedNormalDistribution d(w[0], w[1], w[2], w[3]);
        std::vector<double> extra;
        for (double e : {1e-12, 1e-6, 1e-3, 0.1}) {
            extra.push_back(std::isfinite(w[2]) ? w[2] + e * w[1] : w[3] - e * w[1]);
        }
        emitBandInstance(d, "truncated_normal", w[0], w[1], true, true, extra, sink);
    }

    // poisson(lambda): CDF Q(k + 1, lambda) through the gamma bands, k near
    // lambda (and near a = 1e4).
    for (double lam : {20.0, 1e3, 9999.0, 1e4}) {
        PoissonDistribution d(lam);
        std::vector<double> extra;
        for (double k : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
            extra.push_back(std::floor(lam) + k);
        }
        emitBandInstance(d, "poisson", lam, 0.0, false, false, extra, sink);
    }
    // binomial(n, p): CDF I_{1-p}(n - k, k + 1), both shapes large at n = 1e4.
    for (auto [n, p] : std::vector<std::pair<double, double>>{
             {40.0, 0.5}, {1000.0, 0.3}, {10000.0, 0.5}, {1000000.0, 1e-5}}) {
        BinomialDistribution d(static_cast<int>(n), p);
        emitBandInstance(d, "binomial", n, p, true, false, {}, sink);
    }
    // negative_binomial(r, p): CDF I_p(r, k + 1), r across the tiny-shape
    // switch (1e-3), a = 1 and the Stirling prefactor (20).
    for (auto [r, p] : std::vector<std::pair<double, double>>{{below(1e-3), 0.5},
                                                              {1e-3, 0.5},
                                                              {0.5, 0.3},
                                                              {below(1.0), 0.3},
                                                              {1.0, 0.3},
                                                              {below(20.0), 0.5},
                                                              {20.0, 0.5},
                                                              {1e4, 0.5}}) {
        NegativeBinomialDistribution d(r, p);
        emitBandInstance(d, "negative_binomial", r, p, true, false, {}, sink);
    }

    // gamma, erlang and inverse_gamma at a non-dyadic rate: the product beta*x
    // (beta/x for inverse_gamma) rounds, and a large shape amplifies that
    // rounding by ~(alpha - beta*x); at rate 1 it never shows. Points 30
    // standard deviations either side of the mean as well.
    for (auto [a, b] : std::vector<std::pair<double, double>>{{1e4, 0.3}, {1e16, 3.0}}) {
        GammaDistribution d(a, b);
        std::vector<double> extra;
        for (double z : {-30.0, 30.0}) {
            extra.push_back((a + z * std::sqrt(a)) / b);
        }
        emitBandInstance(d, "gamma", a, b, true, false, extra, sink);
    }
    for (auto [k, lam] : std::vector<std::pair<int, double>>{{10000, 0.3}, {2000000000, 3.0}}) {
        ErlangDistribution d(k, lam);
        std::vector<double> extra;
        for (double z : {-30.0, 30.0}) {
            extra.push_back((k + z * std::sqrt(static_cast<double>(k))) / lam);
        }
        emitBandInstance(d, "erlang", static_cast<double>(k), lam, true, false, extra, sink);
    }
    for (double a : {1e4, 1e16}) {
        InverseGammaDistribution d(a, 3.0);
        std::vector<double> extra;
        for (double z : {-30.0, 30.0}) {
            extra.push_back(3.0 / (a + z * std::sqrt(a)));
        }
        emitBandInstance(d, "inverse_gamma", a, 3.0, true, false, extra, sink);
    }
}

void sweepAll(Sink& sink) {
    using stats::BernoulliDistribution;
    using stats::BetaDistribution;
    using stats::BinomialDistribution;
    using stats::CauchyDistribution;
    using stats::ChiSquaredDistribution;
    using stats::DiscreteDistribution;
    using stats::ErlangDistribution;
    using stats::ExponentialDistribution;
    using stats::FDistribution;
    using stats::GammaDistribution;
    using stats::GaussianDistribution;
    using stats::GeometricDistribution;
    using stats::GumbelDistribution;
    using stats::HalfNormalDistribution;
    using stats::InverseGammaDistribution;
    using stats::LaplaceDistribution;
    using stats::LogisticDistribution;
    using stats::LogNormalDistribution;
    using stats::NegativeBinomialDistribution;
    using stats::ParetoDistribution;
    using stats::PoissonDistribution;
    using stats::RayleighDistribution;
    using stats::StudentTDistribution;
    using stats::TruncatedNormalDistribution;
    using stats::UniformDistribution;
    using stats::VonMisesDistribution;
    using stats::WeibullDistribution;

    // gaussian(mean, sigma): sigma > 0
    sweepContinuous<GaussianDistribution>(
        "gaussian", [](double a, double b) { return GaussianDistribution(a, b); },
        {{{0.0, 1.0}, {1e-3, 1e-3}, {1e6, 1e4}}}, true, false, sink);

    // lognormal(mu, sigma): sigma > 0
    sweepContinuous<LogNormalDistribution>(
        "lognormal", [](double a, double b) { return LogNormalDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 0.01}, {0.0, 2.0}}}, true, false, sink);

    // exponential(lambda): lambda > 0
    sweepContinuous<ExponentialDistribution>(
        "exponential", [](double a, double) { return ExponentialDistribution(a); },
        {{{1.0, 0.0}, {1e-3, 0.0}, {1e6, 0.0}}}, false, false, sink);

    // uniform(a, b): b > a
    sweepContinuous<UniformDistribution>(
        "uniform", [](double a, double b) { return UniformDistribution(a, b); },
        {{{0.0, 1.0}, {-1e-3, 1e-3}, {-1e8, 1e8}}}, true, true, sink);

    // gamma(alpha, beta): both > 0 (beta is a rate parameter)
    sweepContinuous<GammaDistribution>(
        "gamma", [](double a, double b) { return GammaDistribution(a, b); },
        {{{2.0, 1.0}, {0.01, 0.01}, {1e4, 1e-3}}}, true, false, sink);

    // student_t(nu): nu > 0
    sweepContinuous<StudentTDistribution>(
        "student_t", [](double a, double) { return StudentTDistribution(a); },
        {{{5.0, 0.0}, {1.001, 0.0}, {1e6, 0.0}}}, false, false, sink);

    // cauchy(x0, gamma): gamma > 0
    sweepContinuous<CauchyDistribution>(
        "cauchy", [](double a, double b) { return CauchyDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 1e-6}, {1e8, 1e6}}}, true, false, sink);

    // von_mises(mu, kappa): kappa >= 0
    sweepContinuous<VonMisesDistribution>(
        "von_mises", [](double a, double b) { return VonMisesDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 1e-6}, {0.0, 100.0}}}, true, false, sink);

    // beta(alpha, beta): both > 0; support fixed at [0, 1]
    sweepContinuous<BetaDistribution>(
        "beta", [](double a, double b) { return BetaDistribution(a, b); },
        {{{2.0, 3.0}, {0.01, 0.01}, {1e4, 1e4}}}, true, true, sink);

    // chi_squared(k): k > 0
    sweepContinuous<ChiSquaredDistribution>(
        "chi_squared", [](double a, double) { return ChiSquaredDistribution(a); },
        {{{3.0, 0.0}, {0.01, 0.0}, {1e5, 0.0}}}, false, false, sink);

    // laplace(mu, b): b > 0
    sweepContinuous<LaplaceDistribution>(
        "laplace", [](double a, double b) { return LaplaceDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 1e-6}, {1e8, 1e6}}}, true, false, sink);

    // pareto(scale=x_m, alpha): both > 0
    sweepContinuous<ParetoDistribution>(
        "pareto", [](double a, double b) { return ParetoDistribution(a, b); },
        {{{1.0, 2.0}, {1e-6, 0.01}, {1e6, 100.0}}}, true, false, sink);

    // rayleigh(sigma): sigma > 0
    sweepContinuous<RayleighDistribution>(
        "rayleigh", [](double a, double) { return RayleighDistribution(a); },
        {{{1.0, 0.0}, {1e-6, 0.0}, {1e6, 0.0}}}, false, false, sink);

    // weibull(shape, scale): both > 0
    sweepContinuous<WeibullDistribution>(
        "weibull", [](double a, double b) { return WeibullDistribution(a, b); },
        {{{1.5, 1.0}, {0.01, 1e-3}, {100.0, 1e4}}}, true, false, sink);

    // --- Discrete distributions ---

    // poisson(lambda): lambda > 0
    sweepDiscrete<PoissonDistribution>(
        "poisson", [](double a, double) { return PoissonDistribution(a); },
        {{{4.0, 0.0}, {1e-3, 0.0}, {1e5, 0.0}}}, false, sink);

    // discrete(a, b): int bounds, b > a; p1/p2 store the double bits of the ints used
    sweepDiscrete<DiscreteDistribution>(
        "discrete",
        [](double a, double b) {
            return DiscreteDistribution(static_cast<int>(a), static_cast<int>(b));
        },
        {{{0.0, 9.0}, {0.0, 1.0}, {-1000000.0, 1000000.0}}}, true, sink);

    // binomial(n, p): n positive int, p in [0,1]; p1 stores double bits of the int n used
    sweepDiscrete<BinomialDistribution>(
        "binomial", [](double a, double b) { return BinomialDistribution(static_cast<int>(a), b); },
        {{{20.0, 0.5}, {1.0, 0.5}, {1000000.0, 0.3}}}, true, sink);

    // negative_binomial(r, p): r > 0 (real-valued), p in (0,1]
    sweepDiscrete<NegativeBinomialDistribution>(
        "negative_binomial", [](double a, double b) { return NegativeBinomialDistribution(a, b); },
        {{{5.0, 0.5}, {0.01, 0.9}, {10000.0, 0.01}}}, true, sink);

    // geometric(p): p in (0,1]
    sweepDiscrete<GeometricDistribution>(
        "geometric", [](double a, double) { return GeometricDistribution(a); },
        {{{0.3, 0.0}, {1.0, 0.0}, {1e-6, 0.0}}}, false, sink);

    // ------------------------------------------------------------------------
    // v2.4.0 distributions (#54/#55/#56/#57). APPENDED, never interleaved --
    // see the kDistNames comment: every row above must keep its byte identity
    // and its CSV line number.
    // ------------------------------------------------------------------------

    // logistic(mu, s): s > 0 (SCALE). Laplace's instance shape: same location-
    // scale family geometry, so the same three points stress the same seams
    // (degenerate scale, then a large location where mu + s*logit(p) cancels).
    sweepContinuous<LogisticDistribution>(
        "logistic", [](double a, double b) { return LogisticDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 1e-6}, {1e8, 1e6}}}, true, false, sink);

    // gumbel(mu, beta): beta > 0 (SCALE). Same location-scale ladder, but the
    // family is ASYMMETRIC -- the two tails are exp(-e^-z) and 1-exp(-e^-z),
    // so the grid probes two structurally different tail laws per instance.
    sweepContinuous<GumbelDistribution>(
        "gumbel", [](double a, double b) { return GumbelDistribution(a, b); },
        {{{0.0, 1.0}, {0.0, 1e-6}, {1e8, 1e6}}}, true, false, sink);

    // erlang(k, lambda): k >= 1 INTEGER shape, lambda > 0 (RATE -- Gamma's
    // convention, not a scale). k=1 is the exponential special case; k=10000
    // is the large-shape regime where the oracle's lower-gamma series stops
    // converging and the upper-gamma continued fraction takes over.
    sweepContinuous<ErlangDistribution>(
        "erlang", [](double a, double b) { return ErlangDistribution(static_cast<int>(a), b); },
        {{{2.0, 1.0}, {1.0, 1e-3}, {10000.0, 1e-3}}}, true, false, sink);

    // fisher_f(d1, d2): both > 0, real-valued. CDF is I_y(d1/2, d2/2) at
    // y = d1*x/(d1*x + d2). (1e4, 1e4) puts BOTH incomplete-beta parameters at
    // 5000 -- exactly the min(a,b) >= 5000 regime where mp.betainc hangs and
    // the oracle's Lentz continued fraction is the only usable reference.
    sweepContinuous<FDistribution>(
        "fisher_f", [](double a, double b) { return FDistribution(a, b); },
        {{{5.0, 10.0}, {0.01, 0.01}, {1e4, 1e4}}}, true, false, sink);

    // inverse_gamma(alpha, beta): alpha > 0 (shape), beta > 0 (SCALE -- scipy's
    // invgamma(a, scale=beta); it is passed unchanged to a Gamma delegate whose
    // same-named parameter is a RATE, which is consistent only because the
    // reciprocal is taken on the variate, never on the parameter). Gamma's
    // instance ladder, since the CDF is Q(alpha, beta/x) -- the same incomplete
    // gamma read from the opposite tail.
    sweepContinuous<InverseGammaDistribution>(
        "inverse_gamma", [](double a, double b) { return InverseGammaDistribution(a, b); },
        {{{3.0, 2.0}, {0.01, 0.01}, {1e4, 1e-3}}}, true, false, sink);

    // half_normal(sigma): sigma > 0. Rayleigh's single-scale ladder; CDF is
    // erf(x/(sigma*sqrt(2))) and the quantile is the erf_inv path #136 files
    // against, so the extreme-p rows here are expected to be the loud ones.
    sweepContinuous<HalfNormalDistribution>(
        "half_normal", [](double a, double) { return HalfNormalDistribution(a); },
        {{{1.0, 0.0}, {1e-6, 0.0}, {1e6, 0.0}}}, false, false, sink);

    // truncated_normal(mean, sd, lower, upper). Instances must have pairwise-
    // distinct (mean, sd) -- see sweepContinuous4's comment. Windows:
    //   (0, 1)      -> [-2, 2]    symmetric, straddling: Z from the erf
    //                             difference branch (opposite signs, no
    //                             cancellation).
    //   (0, 2)      -> [10, 12]   SAME-TAIL window at alpha=5, beta=6 sigma:
    //                             Z = Q(5) - Q(6) ~ 2.86e-7 from the
    //                             right-tail branch. This is the #57 seam --
    //                             a Phi-difference reference cancels here and
    //                             the oracle must be built survival-form. The
    //                             sigma=2 realization keeps (mean, sd) unique
    //                             while holding the STANDARDIZED window fixed
    //                             at the brief's (5, 6), so Z is unchanged.
    //   (1e3, 1e2)  -> (-inf,+inf) degenerate: reduces exactly to the plain
    //                             Gaussian, and is the reason
    //                             buildContinuousXGrid grew its finiteness
    //                             guard. The factory accepts +/-inf bounds
    //                             (error_handling.h: "either bound may be
    //                             +/-infinity ... and is allowed"), so no
    //                             finite-window substitution was needed.
    // boundedBothSides=true: the near-boundary epsilon points are exactly where
    // a truncated density is most worth probing; the guard drops them for the
    // infinite instance rather than emitting +/-inf twice.
    sweepContinuous4<TruncatedNormalDistribution>(
        "truncated_normal",
        [](double m, double s, double lo, double hi) {
            return TruncatedNormalDistribution(m, s, lo, hi);
        },
        {{{{0.0, 1.0, -2.0, 2.0}},
          {{0.0, 2.0, 10.0, 12.0}},
          {{1e3, 1e2, -std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()}}}},
        true, sink);

    // --- v2.4.0 discrete ---

    // bernoulli(p): p in [0, 1] -- validation is deliberately CLOSED, so p=0
    // and p=1 are legal; 1e-6 / 1-1e-6 stay just inside without relying on it.
    sweepDiscrete<BernoulliDistribution>(
        "bernoulli", [](double a, double) { return BernoulliDistribution(a); },
        {{{0.5, 0.0}, {1e-6, 0.0}, {1.0 - 1e-6, 0.0}}}, false, sink);

    // Switch-band instances, appended after every row above.
    sweepSwitchBands(sink);
}

//==============================================================================
// Environment metadata (commit / isa / date)
//==============================================================================

std::string getGitCommitShort() {
#if defined(_WIN32)
    FILE* pipe = _popen("git rev-parse --short HEAD 2>NUL", "r");
#else
    FILE* pipe = popen("git rev-parse --short HEAD 2>/dev/null", "r");
#endif
    if (!pipe) {
        return "unknown";
    }
    std::string result;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result.empty() ? "unknown" : result;
}

std::string getDateStamp() {
    std::time_t now = std::time(nullptr);
    std::tm tmv{};
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tmv, "%Y-%m-%d");
    return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: accuracy_sweep <output.csv>\n";
        return 1;
    }
    const std::string outPath = argv[1];

    Sink sink;
    sweepAll(sink);

    std::ofstream out(outPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::cerr << "Failed to open output file: " << outPath << "\n";
        return 1;
    }

    out << "# libstats accuracy_sweep v1\n";
    out << "# commit=" << getGitCommitShort()
        << "  isa=" << stats::arch::simd::SIMDPolicy::getLevelString()
        << "  date=" << getDateStamp() << "\n";
    out << "dist,method,p1_bits,p2_bits,x_bits,scalar_bits,batch_bits\n";
    for (const auto& row : sink.rows()) {
        out << row.toCsvLine() << "\n";
    }
    for (const auto& [distName, n] : sink.skips()) {
        if (n > 0) {
            out << "# skipped_quantile " << distName << " " << n << "\n";
        }
    }
    out.close();

    std::size_t total = sink.rows().size();
    std::cout << "accuracy_sweep: wrote " << total << " rows to " << outPath << "\n";
    std::cout << "Per-distribution row counts:\n";
    for (const auto& [distName, n] : sink.counts()) {
        std::cout << "  " << distName << ": " << n << "\n";
    }
    bool anySkips = false;
    for (const auto& [distName, n] : sink.skips()) {
        if (n > 0) {
            if (!anySkips) {
                std::cout << "Skipped quantile rows (getQuantile threw):\n";
                anySkips = true;
            }
            std::cout << "  " << distName << ": " << n << "\n";
        }
    }
    if (!anySkips) {
        std::cout << "No quantile rows skipped.\n";
    }

    return 0;
}
