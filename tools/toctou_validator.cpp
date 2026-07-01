/**
 * @file toctou_validator.cpp
 * @brief TOCTOU Race Condition Validator for libstats distributions
 *
 * Stress-tests each distribution's probability methods for time-of-check to
 * time-of-use (TOCTOU) cache inconsistencies by running concurrent writer threads
 * (cycling between two parameter sets) and reader threads (checking PDF/CDF
 * invariants) simultaneously.
 *
 * @par Violation categories
 * - HARD  — NaN, ±Inf, negative PDF, or CDF outside [0,1].  Always a bug.
 * - MIXED — PDF/CDF result inconsistent with ALL known-valid parameter sets,
 *           indicating a partial read of a cache whose entries don't match the
 *           current parameters (the canonical TOCTOU symptom).
 *
 * @par Detection strategy
 * For each distribution, two parameter sets P1 and P2 are chosen so that their
 * PDF at a common test point x differs by ≥5×.  The baseline values pdf_p1 and
 * pdf_p2 are measured single-threaded.  A reader observation v is classified as
 * MIXED if it is inconsistent with both (i.e. |v − pdf_p1| > tol_p1 and
 * |v − pdf_p2| > tol_p2, where tol_pi = max(|pdf_pi| × 0.10, 1e-10)).
 *
 * @par Usage
 * @code
 * ./build/tools/toctou_validator [--duration-ms N] [--reader-threads N] [--quick]
 * @endcode
 *
 * Exit code: 0 = all distributions pass; 1 = at least one violation detected.
 */

#include "tool_utils.h"

#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/cauchy.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/pareto.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/rayleigh.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace stats;
using namespace std::chrono_literals;

namespace {

constexpr int DEFAULT_DURATION_MS  = 300;   // per distribution
constexpr int QUICK_DURATION_MS    = 50;
constexpr int DEFAULT_READER_THREADS = 4;
constexpr double CONSISTENT_REL_TOL = 0.10; // 10 % relative tolerance

// ─── Result ───────────────────────────────────────────────────────────────────

struct RaceResult {
    long long total_reads    = 0;
    long long hard_violations = 0;
    long long mixed_violations = 0;
    std::string first_hard;
    std::string first_mixed;

    bool pass() const { return hard_violations == 0 && mixed_violations == 0; }
};

// ─── Core race harness ────────────────────────────────────────────────────────
//
// Runs one writer (alternating P1/P2 as fast as possible) and N readers
// (continuously sampling PDF/CDF and checking invariants).
//
// check_fn(dist, msg) → 0 = ok | 1 = hard violation | 2 = mixed violation
//   On violation it sets msg to a human-readable description.

template<typename Dist>
RaceResult run_race(
    Dist& dist,
    std::function<void(Dist&)> write_p1,
    std::function<void(Dist&)> write_p2,
    std::function<int(const Dist&, std::string&)> check_fn,
    int duration_ms,
    int n_readers)
{
    constexpr auto rlx = std::memory_order_relaxed;

    std::atomic<bool>      stop{false};
    std::atomic<long long> hard{0}, mixed{0}, total{0};
    std::atomic<bool>      logged_hard{false}, logged_mixed{false};
    std::mutex             log_mu;
    std::string            first_hard, first_mixed;

    // Single writer: alternates P1 → P2 at full speed
    std::thread writer([&] {
        while (!stop.load(rlx)) {
            write_p1(dist);
            write_p2(dist);
        }
    });

    // Readers: sample PDF/CDF and classify each observation
    std::vector<std::thread> readers;
    readers.reserve(static_cast<size_t>(n_readers));
    for (int r = 0; r < n_readers; ++r) {
        readers.emplace_back([&] {
            std::string msg;
            while (!stop.load(rlx)) {
                total.fetch_add(1, rlx);
                int code = check_fn(dist, msg);
                if (code == 1) {
                    hard.fetch_add(1, rlx);
                    if (!logged_hard.exchange(true, rlx)) {
                        std::lock_guard<std::mutex> lg(log_mu);
                        first_hard = msg;
                    }
                } else if (code == 2) {
                    mixed.fetch_add(1, rlx);
                    if (!logged_mixed.exchange(true, rlx)) {
                        std::lock_guard<std::mutex> lg(log_mu);
                        first_mixed = msg;
                    }
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
    stop.store(true, std::memory_order_relaxed);
    writer.join();
    for (auto& t : readers) t.join();

    RaceResult r;
    r.total_reads     = total.load();
    r.hard_violations = hard.load();
    r.mixed_violations = mixed.load();
    r.first_hard  = first_hard;
    r.first_mixed = first_mixed;
    return r;
}

// ─── Checker builder ──────────────────────────────────────────────────────────
//
// Returns a check_fn suitable for run_race given baseline (single-threaded)
// pdf values at test_x for P1 and P2.

template<typename Dist>
std::function<int(const Dist&, std::string&)>
make_checker(double test_x, double pdf_p1, double pdf_p2, double cdf_test_x)
{
    // per-set absolute tolerances
    const double tol1 = std::max(std::abs(pdf_p1) * CONSISTENT_REL_TOL, 1e-10);
    const double tol2 = std::max(std::abs(pdf_p2) * CONSISTENT_REL_TOL, 1e-10);
    const double cdf_tol = 1e-10;

    return [=](const Dist& dist, std::string& msg) -> int {
        // --- PDF check ---
        const double pdf = dist.getProbability(test_x);

        if (!std::isfinite(pdf)) {
            std::ostringstream os;
            os << "PDF(" << test_x << ") = " << pdf << " (non-finite)";
            msg = os.str();
            return 1;  // hard
        }
        if (pdf < -1e-13) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(4)
               << "PDF(" << test_x << ") = " << pdf << " (negative)";
            msg = os.str();
            return 1;  // hard
        }

        const bool ok_p1 = (std::abs(pdf - pdf_p1) <= tol1);
        const bool ok_p2 = (std::abs(pdf - pdf_p2) <= tol2);
        if (!ok_p1 && !ok_p2) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(4)
               << "PDF(" << test_x << ")=" << pdf
               << " inconsistent with P1=" << pdf_p1
               << " (tol=" << tol1 << ") and P2=" << pdf_p2
               << " (tol=" << tol2 << ")";
            msg = os.str();
            return 2;  // mixed state
        }

        // --- CDF check ---
        const double cdf = dist.getCumulativeProbability(test_x);
        if (!std::isfinite(cdf) || cdf < -cdf_tol || cdf > 1.0 + cdf_tol) {
            std::ostringstream os;
            os << std::scientific << std::setprecision(4)
               << "CDF(" << test_x << ") = " << cdf << " (out of [0,1] or non-finite)";
            msg = os.str();
            return 1;  // hard
        }

        return 0;  // ok
    };
}

// ─── Per-distribution race specifications ─────────────────────────────────────
//
// Each returns a RaceResult.  Baseline PDF values are computed single-threaded
// before the concurrent phase, so the tool is self-calibrating.

RaceResult race_gaussian(int dur, int nr) {
    // P1: N(0,1), P2: N(0,0.2) — PDF(0) differs by 5×
    auto d = GaussianDistribution::create(0.0, 1.0).value;
    const double pdf_p1 = GaussianDistribution::create(0.0, 1.0) .value.getProbability(0.0);
    const double pdf_p2 = GaussianDistribution::create(0.0, 0.2) .value.getProbability(0.0);
    const double cdf_p1 = GaussianDistribution::create(0.0, 1.0) .value.getCumulativeProbability(0.0);
    auto check = make_checker<GaussianDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<GaussianDistribution>(
        d,
        [](GaussianDistribution& x){ x.trySetParameters(0.0, 1.0);  },
        [](GaussianDistribution& x){ x.trySetParameters(0.0, 0.2);  },
        check, dur, nr);
}

RaceResult race_exponential(int dur, int nr) {
    // P1: Exp(1), P2: Exp(10) — PDF(0.5) differs by ~11×
    auto d = ExponentialDistribution::create(1.0).value;
    const double pdf_p1 = ExponentialDistribution::create(1.0) .value.getProbability(0.5);
    const double pdf_p2 = ExponentialDistribution::create(10.0).value.getProbability(0.5);
    const double cdf_p1 = ExponentialDistribution::create(1.0) .value.getCumulativeProbability(0.5);
    auto check = make_checker<ExponentialDistribution>(0.5, pdf_p1, pdf_p2, cdf_p1);
    return run_race<ExponentialDistribution>(
        d,
        [](ExponentialDistribution& x){ x.trySetLambda(1.0);  },
        [](ExponentialDistribution& x){ x.trySetLambda(10.0); },
        check, dur, nr);
}

RaceResult race_uniform(int dur, int nr) {
    // P1: U(0,1), P2: U(0,10) — PDF(0.5) differs by 10×
    auto d = UniformDistribution::create(0.0, 1.0).value;
    const double pdf_p1 = UniformDistribution::create(0.0, 1.0) .value.getProbability(0.5);
    const double pdf_p2 = UniformDistribution::create(0.0, 10.0).value.getProbability(0.5);
    const double cdf_p1 = UniformDistribution::create(0.0, 1.0) .value.getCumulativeProbability(0.5);
    auto check = make_checker<UniformDistribution>(0.5, pdf_p1, pdf_p2, cdf_p1);
    return run_race<UniformDistribution>(
        d,
        [](UniformDistribution& x){ x.trySetParameters(0.0, 1.0);  },
        [](UniformDistribution& x){ x.trySetParameters(0.0, 10.0); },
        check, dur, nr);
}

RaceResult race_gamma(int dur, int nr) {
    // P1: Gamma(1,1)=Exp(1), P2: Gamma(8,1) — PDF(1) differs by ~5000×
    auto d = GammaDistribution::create(1.0, 1.0).value;
    const double pdf_p1 = GammaDistribution::create(1.0, 1.0).value.getProbability(1.0);
    const double pdf_p2 = GammaDistribution::create(8.0, 1.0).value.getProbability(1.0);
    const double cdf_p1 = GammaDistribution::create(1.0, 1.0).value.getCumulativeProbability(1.0);
    auto check = make_checker<GammaDistribution>(1.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<GammaDistribution>(
        d,
        [](GammaDistribution& x){ x.trySetParameters(1.0, 1.0); },
        [](GammaDistribution& x){ x.trySetParameters(8.0, 1.0); },
        check, dur, nr);
}

RaceResult race_beta(int dur, int nr) {
    // P1: Beta(1,1)=Uniform, P2: Beta(5,2) — PDF(0.1) differs by ~370×
    auto d = BetaDistribution::create(1.0, 1.0).value;
    const double pdf_p1 = BetaDistribution::create(1.0, 1.0).value.getProbability(0.1);
    const double pdf_p2 = BetaDistribution::create(5.0, 2.0).value.getProbability(0.1);
    const double cdf_p1 = BetaDistribution::create(1.0, 1.0).value.getCumulativeProbability(0.1);
    auto check = make_checker<BetaDistribution>(0.1, pdf_p1, pdf_p2, cdf_p1);
    return run_race<BetaDistribution>(
        d,
        [](BetaDistribution& x){ x.trySetParameters(1.0, 1.0); },
        [](BetaDistribution& x){ x.trySetParameters(5.0, 2.0); },
        check, dur, nr);
}

RaceResult race_lognormal(int dur, int nr) {
    // P1: LN(0,1), P2: LN(0,0.1) — PDF(1) differs by 10×
    auto d = LogNormalDistribution::create(0.0, 1.0).value;
    const double pdf_p1 = LogNormalDistribution::create(0.0, 1.0) .value.getProbability(1.0);
    const double pdf_p2 = LogNormalDistribution::create(0.0, 0.1) .value.getProbability(1.0);
    const double cdf_p1 = LogNormalDistribution::create(0.0, 1.0) .value.getCumulativeProbability(1.0);
    auto check = make_checker<LogNormalDistribution>(1.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<LogNormalDistribution>(
        d,
        [](LogNormalDistribution& x){ x.trySetParameters(0.0, 1.0);  },
        [](LogNormalDistribution& x){ x.trySetParameters(0.0, 0.1);  },
        check, dur, nr);
}

RaceResult race_pareto(int dur, int nr) {
    // P1: Pareto(1,2), P2: Pareto(1,10) — PDF(2) differs by ~51×
    auto d = ParetoDistribution::create(1.0, 2.0).value;
    const double pdf_p1 = ParetoDistribution::create(1.0, 2.0) .value.getProbability(2.0);
    const double pdf_p2 = ParetoDistribution::create(1.0, 10.0).value.getProbability(2.0);
    const double cdf_p1 = ParetoDistribution::create(1.0, 2.0) .value.getCumulativeProbability(2.0);
    auto check = make_checker<ParetoDistribution>(2.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<ParetoDistribution>(
        d,
        [](ParetoDistribution& x){ x.trySetParameters(1.0, 2.0);  },
        [](ParetoDistribution& x){ x.trySetParameters(1.0, 10.0); },
        check, dur, nr);
}

RaceResult race_weibull(int dur, int nr) {
    // P1: Weibull(1,1)=Exp(1), P2: Weibull(10,1) — PDF(0.5) differs by ~31×
    auto d = WeibullDistribution::create(1.0, 1.0).value;
    const double pdf_p1 = WeibullDistribution::create(1.0,  1.0).value.getProbability(0.5);
    const double pdf_p2 = WeibullDistribution::create(10.0, 1.0).value.getProbability(0.5);
    const double cdf_p1 = WeibullDistribution::create(1.0,  1.0).value.getCumulativeProbability(0.5);
    auto check = make_checker<WeibullDistribution>(0.5, pdf_p1, pdf_p2, cdf_p1);
    return run_race<WeibullDistribution>(
        d,
        [](WeibullDistribution& x){ x.trySetParameters(1.0,  1.0); },
        [](WeibullDistribution& x){ x.trySetParameters(10.0, 1.0); },
        check, dur, nr);
}

RaceResult race_rayleigh(int dur, int nr) {
    // P1: Rayleigh(1), P2: Rayleigh(3) — PDF(0.5) differs by ~8×
    auto d = RayleighDistribution::create(1.0).value;
    const double pdf_p1 = RayleighDistribution::create(1.0).value.getProbability(0.5);
    const double pdf_p2 = RayleighDistribution::create(3.0).value.getProbability(0.5);
    const double cdf_p1 = RayleighDistribution::create(1.0).value.getCumulativeProbability(0.5);
    auto check = make_checker<RayleighDistribution>(0.5, pdf_p1, pdf_p2, cdf_p1);
    return run_race<RayleighDistribution>(
        d,
        [](RayleighDistribution& x){ x.trySetSigma(1.0); },
        [](RayleighDistribution& x){ x.trySetSigma(3.0); },
        check, dur, nr);
}

RaceResult race_von_mises(int dur, int nr) {
    // P1: VM(0,0.1)≈uniform, P2: VM(0,10) — PDF(0) differs by ~8×
    auto d = VonMisesDistribution::create(0.0, 0.1).value;
    const double pdf_p1 = VonMisesDistribution::create(0.0, 0.1) .value.getProbability(0.0);
    const double pdf_p2 = VonMisesDistribution::create(0.0, 10.0).value.getProbability(0.0);
    const double cdf_p1 = VonMisesDistribution::create(0.0, 0.1) .value.getCumulativeProbability(0.0);
    auto check = make_checker<VonMisesDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<VonMisesDistribution>(
        d,
        [](VonMisesDistribution& x){ x.trySetKappa(0.1);  },
        [](VonMisesDistribution& x){ x.trySetKappa(10.0); },
        check, dur, nr);
}

RaceResult race_student_t(int dur, int nr) {
    // P1: t(1)=Cauchy, P2: t(100)≈Normal — PDF(5) differs by ~8000×
    auto d = StudentTDistribution::create(1.0).value;
    const double pdf_p1 = StudentTDistribution::create(1.0)  .value.getProbability(5.0);
    const double pdf_p2 = StudentTDistribution::create(100.0).value.getProbability(5.0);
    const double cdf_p1 = StudentTDistribution::create(1.0)  .value.getCumulativeProbability(5.0);
    auto check = make_checker<StudentTDistribution>(5.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<StudentTDistribution>(
        d,
        [](StudentTDistribution& x){ x.trySetNu(1.0);   },
        [](StudentTDistribution& x){ x.trySetNu(100.0); },
        check, dur, nr);
}

RaceResult race_chi_squared(int dur, int nr) {
    // P1: χ²(2), P2: χ²(20) — PDF(2) differs substantially
    auto d = ChiSquaredDistribution::create(2.0).value;
    const double pdf_p1 = ChiSquaredDistribution::create(2.0) .value.getProbability(2.0);
    const double pdf_p2 = ChiSquaredDistribution::create(20.0).value.getProbability(2.0);
    const double cdf_p1 = ChiSquaredDistribution::create(2.0) .value.getCumulativeProbability(2.0);
    auto check = make_checker<ChiSquaredDistribution>(2.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<ChiSquaredDistribution>(
        d,
        [](ChiSquaredDistribution& x){ x.trySetK(2.0);  },
        [](ChiSquaredDistribution& x){ x.trySetK(20.0); },
        check, dur, nr);
}

RaceResult race_poisson(int dur, int nr) {
    // P1: Poisson(0.1), P2: Poisson(5) — P(0) differs by e^{4.9}≈134×
    auto d = PoissonDistribution::create(0.1).value;
    const double pdf_p1 = PoissonDistribution::create(0.1).value.getProbability(0.0);
    const double pdf_p2 = PoissonDistribution::create(5.0).value.getProbability(0.0);
    const double cdf_p1 = PoissonDistribution::create(0.1).value.getCumulativeProbability(0.0);
    auto check = make_checker<PoissonDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<PoissonDistribution>(
        d,
        [](PoissonDistribution& x){ x.trySetLambda(0.1); },
        [](PoissonDistribution& x){ x.trySetLambda(5.0); },
        check, dur, nr);
}

RaceResult race_discrete(int dur, int nr) {
    // P1: Disc[1,2] P(1)=0.5, P2: Disc[1,10] P(1)=0.1 — 5× difference
    auto d = DiscreteDistribution::create(1, 2).value;
    const double pdf_p1 = DiscreteDistribution::create(1, 2) .value.getProbability(1.0);
    const double pdf_p2 = DiscreteDistribution::create(1, 10).value.getProbability(1.0);
    const double cdf_p1 = DiscreteDistribution::create(1, 2) .value.getCumulativeProbability(1.0);
    auto check = make_checker<DiscreteDistribution>(1.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<DiscreteDistribution>(
        d,
        [](DiscreteDistribution& x){ x.trySetBounds(1, 2);  },
        [](DiscreteDistribution& x){ x.trySetBounds(1, 10); },
        check, dur, nr);
}

RaceResult race_binomial(int dur, int nr) {
    // P1: B(10,0.1) P(0)=(0.9)^10≈0.349, P2: B(10,0.9) P(0)=(0.1)^10≈1e-10 — huge ratio
    auto d = BinomialDistribution::create(10, 0.1).value;
    const double pdf_p1 = BinomialDistribution::create(10, 0.1).value.getProbability(0.0);
    const double pdf_p2 = BinomialDistribution::create(10, 0.9).value.getProbability(0.0);
    const double cdf_p1 = BinomialDistribution::create(10, 0.1).value.getCumulativeProbability(0.0);
    auto check = make_checker<BinomialDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<BinomialDistribution>(
        d,
        [](BinomialDistribution& x){ x.trySetP(0.1); },
        [](BinomialDistribution& x){ x.trySetP(0.9); },
        check, dur, nr);
}

RaceResult race_negative_binomial(int dur, int nr) {
    // P1: NB(1,0.1) P(0)=0.1, P2: NB(1,0.9) P(0)=0.9 — 9× ratio
    auto d = NegativeBinomialDistribution::create(1.0, 0.1).value;
    const double pdf_p1 = NegativeBinomialDistribution::create(1.0, 0.1).value.getProbability(0.0);
    const double pdf_p2 = NegativeBinomialDistribution::create(1.0, 0.9).value.getProbability(0.0);
    const double cdf_p1 = NegativeBinomialDistribution::create(1.0, 0.1).value.getCumulativeProbability(0.0);
    auto check = make_checker<NegativeBinomialDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<NegativeBinomialDistribution>(
        d,
        [](NegativeBinomialDistribution& x){ x.trySetP(0.1); },
        [](NegativeBinomialDistribution& x){ x.trySetP(0.9); },
        check, dur, nr);
}

RaceResult race_geometric(int dur, int nr) {
    // P1: Geo(0.1) P(0)=0.1, P2: Geo(0.9) P(0)=0.9 — 9× ratio
    auto d = GeometricDistribution::create(0.1).value;
    const double pdf_p1 = GeometricDistribution::create(0.1).value.getProbability(0.0);
    const double pdf_p2 = GeometricDistribution::create(0.9).value.getProbability(0.0);
    const double cdf_p1 = GeometricDistribution::create(0.1).value.getCumulativeProbability(0.0);
    auto check = make_checker<GeometricDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<GeometricDistribution>(
        d,
        [](GeometricDistribution& x){ x.trySetP(0.1); },
        [](GeometricDistribution& x){ x.trySetP(0.9); },
        check, dur, nr);
}

RaceResult race_laplace(int dur, int nr) {
    // P1: Laplace(0,1), P2: Laplace(0,0.2) — PDF(0) differs by 5×
    auto d = LaplaceDistribution::create(0.0, 1.0).value;
    const double pdf_p1 = LaplaceDistribution::create(0.0, 1.0).value.getProbability(0.0);
    const double pdf_p2 = LaplaceDistribution::create(0.0, 0.2).value.getProbability(0.0);
    const double cdf_p1 = LaplaceDistribution::create(0.0, 1.0).value.getCumulativeProbability(0.0);
    auto check = make_checker<LaplaceDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<LaplaceDistribution>(
        d,
        [](LaplaceDistribution& x){ x.trySetParameters(0.0, 1.0); },
        [](LaplaceDistribution& x){ x.trySetParameters(0.0, 0.2); },
        check, dur, nr);
}

RaceResult race_cauchy(int dur, int nr) {
    // P1: Cauchy(0,1), P2: Cauchy(0,0.2) — PDF(0) differs by 5×
    auto d = CauchyDistribution::create(0.0, 1.0).value;
    const double pdf_p1 = CauchyDistribution::create(0.0, 1.0).value.getProbability(0.0);
    const double pdf_p2 = CauchyDistribution::create(0.0, 0.2).value.getProbability(0.0);
    const double cdf_p1 = CauchyDistribution::create(0.0, 1.0).value.getCumulativeProbability(0.0);
    auto check = make_checker<CauchyDistribution>(0.0, pdf_p1, pdf_p2, cdf_p1);
    return run_race<CauchyDistribution>(
        d,
        [](CauchyDistribution& x){ x.trySetParameters(0.0, 1.0); },
        [](CauchyDistribution& x){ x.trySetParameters(0.0, 0.2); },
        check, dur, nr);
}

// ─── Distribution registry ────────────────────────────────────────────────────

struct DistEntry {
    std::string name;
    std::function<RaceResult(int dur, int nr)> run;
    std::string params;  // human-readable P1 / P2 description
};

std::vector<DistEntry> build_registry() {
    return {
        {"Gaussian",         race_gaussian,         "P1=N(0,1)  P2=N(0,0.2)  x=0.0"},
        {"Exponential",      race_exponential,       "P1=Exp(1)  P2=Exp(10)   x=0.5"},
        {"Uniform",          race_uniform,           "P1=U(0,1)  P2=U(0,10)   x=0.5"},
        {"Gamma",            race_gamma,             "P1=Γ(1,1)  P2=Γ(8,1)    x=1.0"},
        {"Beta",             race_beta,              "P1=B(1,1)  P2=B(5,2)    x=0.1"},
        {"LogNormal",        race_lognormal,         "P1=LN(0,1) P2=LN(0,0.1) x=1.0"},
        {"Pareto",           race_pareto,            "P1=Pa(1,2) P2=Pa(1,10)  x=2.0"},
        {"Weibull",          race_weibull,           "P1=W(1,1)  P2=W(10,1)   x=0.5"},
        {"Rayleigh",         race_rayleigh,          "P1=R(1)    P2=R(3)      x=0.5"},
        {"VonMises",         race_von_mises,         "P1=VM(0,.1)P2=VM(0,10)  x=0.0"},
        {"StudentT",         race_student_t,         "P1=t(1)    P2=t(100)    x=5.0"},
        {"ChiSquared",       race_chi_squared,       "P1=χ²(2)   P2=χ²(20)    x=2.0"},
        {"Poisson",          race_poisson,           "P1=Poi(.1) P2=Poi(5)    x=0.0"},
        {"Discrete",         race_discrete,          "P1=[1,2]   P2=[1,10]    x=1.0"},
        {"Binomial",         race_binomial,          "P1=B(10,.1)P2=B(10,.9)  x=0.0"},
        {"NegativeBinomial", race_negative_binomial, "P1=NB(1,.1)P2=NB(1,.9)  x=0.0"},
        {"Geometric",        race_geometric,         "P1=Geo(.1) P2=Geo(.9)   x=0.0"},
        {"Laplace",          race_laplace,           "P1=L(0,1)  P2=L(0,.2)   x=0.0"},
        {"Cauchy",           race_cauchy,            "P1=C(0,1)  P2=C(0,.2)   x=0.0"},
    };
}

// ─── Main ─────────────────────────────────────────────────────────────────────

void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [options]\n\n"
        "Options:\n"
        "  --duration-ms N     ms per distribution (default " << DEFAULT_DURATION_MS << ")\n"
        "  --reader-threads N  concurrent reader threads (default " << DEFAULT_READER_THREADS << ")\n"
        "  --quick             50 ms per distribution (CI mode)\n"
        "  --help              print this message\n\n"
        "Exit code: 0 = all pass, 1 = any violation detected\n";
}

}  // namespace

int main(int argc, char** argv) {
    int duration_ms   = DEFAULT_DURATION_MS;
    int reader_threads = DEFAULT_READER_THREADS;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            duration_ms = QUICK_DURATION_MS;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--duration-ms" && i + 1 < argc) {
            duration_ms = std::stoi(argv[++i]);
        } else if (arg == "--reader-threads" && i + 1 < argc) {
            reader_threads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    stats::detail::detail::displayToolHeader(
        "TOCTOU Race Condition Validator",
        "Concurrent writer/reader stress test for all 19 distributions");

    std::cout << "Configuration: " << duration_ms << " ms/distribution"
              << ", " << reader_threads << " reader thread(s)"
              << ", 1 writer thread\n\n";

    const auto registry = build_registry();

    // Table header
    const int COL_NAME = 20, COL_READS = 14, COL_HARD = 8, COL_MIXED = 8, COL_STATUS = 8;
    stats::detail::detail::ColumnFormatter fmt({COL_NAME, COL_READS, COL_HARD, COL_MIXED, COL_STATUS});
    std::cout << fmt.formatRow({"Distribution", "Reads", "Hard ✗", "Mixed ✗", "Status"}) << "\n";
    std::cout << fmt.getSeparator() << "\n";

    long long total_violations = 0;
    long long grand_reads      = 0;
    int       failures         = 0;

    for (const auto& entry : registry) {
        std::cout << std::left << std::setw(COL_NAME) << entry.name << std::flush;

        const RaceResult r = entry.run(duration_ms, reader_threads);
        grand_reads      += r.total_reads;
        total_violations += r.hard_violations + r.mixed_violations;

        // Format reads with commas
        std::string reads_str;
        {
            std::ostringstream os;
            long long v = r.total_reads;
            std::string s = std::to_string(v);
            int ins = static_cast<int>(s.size()) % 3;
            for (int i = 0; i < static_cast<int>(s.size()); ++i) {
                if (i > 0 && (i - ins) % 3 == 0) os << ',';
                os << s[static_cast<size_t>(i)];
            }
            reads_str = os.str();
        }

        std::cout << std::right << std::setw(COL_READS) << reads_str
                  << std::right << std::setw(COL_HARD)  << r.hard_violations
                  << std::right << std::setw(COL_MIXED) << r.mixed_violations;

        if (r.pass()) {
            std::cout << "  PASS ✓\n";
        } else {
            std::cout << "  FAIL ✗\n";
            ++failures;
            if (!r.first_hard.empty())
                std::cout << "  [HARD]  " << r.first_hard << "\n";
            if (!r.first_mixed.empty())
                std::cout << "  [MIXED] " << r.first_mixed << "\n";
        }
    }

    std::cout << fmt.getSeparator() << "\n";

    if (failures == 0) {
        std::cout << "\n✓ All " << registry.size() << " distributions PASS"
                  << " (" << grand_reads << " total reads, 0 violations)\n\n";
    } else {
        std::cout << "\n✗ " << failures << " of " << registry.size()
                  << " distributions FAIL — " << total_violations
                  << " violation(s) in " << grand_reads << " reads\n\n";
    }

    std::cout << "Notes:\n"
              << "  HARD  = NaN / Inf / negative PDF / CDF outside [0,1]\n"
              << "  MIXED = PDF inconsistent with all valid parameter states\n"
              << "          (indicates stale cache combined with new parameters)\n"
              << "  Both require TOCTOU window to be exploitable under your\n"
              << "  OS scheduler.  Zero violations confirm the snapshot-under-lock\n"
              << "  fix is effective on this platform.\n\n";

    return (failures > 0) ? 1 : 0;
}
