// table_ab_bench: AUTO against forced VECTORIZED and PARALLEL for every distribution's batch
// pdf/logpdf/cdf at sizes straddling the dispatch-table thresholds. Built against two trees that
// differ only in dispatch_thresholds.h, run back to back, it shows per (distribution, op, n)
// whether AUTO under each table reaches the best forced strategy (PLAN.md R8, the 2026-10-14 row
// proposal). Parameters and input ranges follow tools/strategy_profile.cpp, so the rows map onto
// the R8 bundles.
//
// Build (T = a tree with build-release/libstats.a):
//   clang++ -std=c++20 -O2 -DNDEBUG -I$T/include -I$T/build-release/generated \
//     tools/bench/table_ab_bench.cpp $T/build-release/libstats.a -lpthread -o table_ab_<tag>
// Run: table_ab_old > old.csv; table_ab_new > new.csv (same CSV columns as v242_cost_bench).
#include "libstats/distributions/bernoulli.h"
#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/cauchy.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/erlang.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/fisher_f.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/gumbel.h"
#include "libstats/distributions/half_normal.h"
#include "libstats/distributions/inverse_gamma.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/logistic.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/pareto.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/rayleigh.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/truncated_normal.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;
using Strategy = stats::detail::PerformanceHint::PreferredStrategy;

volatile double sink = 0.0;
constexpr double kPi = 3.14159265358979323846;
constexpr std::size_t kMinBlock = 200'000;  // elements per timed block
constexpr std::size_t kSizes[] = {1024,  2048,  4096,   8192,   10000,
                                  25000, 50000, 100000, 200000, 500000};

template <class F>
double nsPer(std::size_t n, F&& f) {
    const std::size_t reps = std::max<std::size_t>(1, (kMinBlock + n - 1) / n);
    double best = 1e300;
    for (int r = 0; r < 5; ++r) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i)
            f();
        const auto t1 = clk::now();
        best =
            std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * n));
    }
    return best;
}

void row(const char* name, const char* op, std::size_t n, const char* strategy, double ns) {
    std::printf("%s,%s,%zu,%s,%.2f\n", name, op, n, strategy, ns);
    std::fflush(stdout);
}

struct Forced {
    const char* name;
    Strategy strategy;
};
constexpr Forced kStrategies[] = {{"auto", Strategy::AUTO},
                                  {"vectorized", Strategy::FORCE_VECTORIZED},
                                  {"parallel", Strategy::FORCE_PARALLEL}};

std::mt19937_64 rng(42);

std::vector<double> uniformInputs(double lo, double hi, bool integer) {
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<double> x(kSizes[sizeof kSizes / sizeof kSizes[0] - 1]);
    for (auto& v : x)
        v = integer ? std::round(u(rng)) : u(rng);
    return x;
}

template <class D>
void bench(const char* name, const D& d, const std::vector<double>& x) {
    std::vector<double> y(x.size());
    for (std::size_t n : kSizes) {
        const std::span<const double> xs(x.data(), n);
        const std::span<double> ys(y.data(), n);
        for (const Forced& f : kStrategies) {
            const stats::detail::PerformanceHint hint{f.strategy, std::nullopt};
            row(name, "pdf", n, f.name, nsPer(n, [&] {
                    d.getProbability(xs, ys, hint);
                    sink = sink + y[n / 2];
                }));
            row(name, "logpdf", n, f.name, nsPer(n, [&] {
                    d.getLogProbability(xs, ys, hint);
                    sink = sink + y[n / 2];
                }));
            row(name, "cdf", n, f.name, nsPer(n, [&] {
                    d.getCumulativeProbability(xs, ys, hint);
                    sink = sink + y[n / 2];
                }));
        }
    }
}

}  // namespace

int main() {
    std::printf("case,op,n,strategy,ns_per_element\n");
    bench("Uniform", stats::UniformDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-0.5, 1.5, false));
    bench("Gaussian", stats::GaussianDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-4.0, 4.0, false));
    bench("Exponential", stats::ExponentialDistribution::create(1.0).unwrap(),
          uniformInputs(0.0, 6.0, false));
    bench("Discrete", stats::DiscreteDistribution::create(0, 10).unwrap(),
          uniformInputs(0.0, 10.0, true));
    bench("Poisson", stats::PoissonDistribution::create(3.5).unwrap(),
          uniformInputs(0.0, 12.0, true));
    bench("Gamma", stats::GammaDistribution::create(2.0, 1.0).unwrap(),
          uniformInputs(0.05, 8.0, false));
    bench("StudentT", stats::StudentTDistribution::create(5.0).unwrap(),
          uniformInputs(-6.0, 6.0, false));
    bench("Beta", stats::BetaDistribution::create(2.0, 5.0).unwrap(),
          uniformInputs(-0.1, 1.1, false));
    bench("ChiSquared", stats::ChiSquaredDistribution::create(4.0).unwrap(),
          uniformInputs(0.05, 14.0, false));
    bench("LogNormal", stats::LogNormalDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(0.05, 8.0, false));
    bench("Pareto", stats::ParetoDistribution::create(1.0, 2.0).unwrap(),
          uniformInputs(1.0, 10.0, false));
    bench("Weibull", stats::WeibullDistribution::create(2.0, 1.0).unwrap(),
          uniformInputs(0.0, 3.0, false));
    bench("Rayleigh", stats::RayleighDistribution::create(1.0).unwrap(),
          uniformInputs(0.0, 4.0, false));
    bench("VonMises", stats::VonMisesDistribution::create(0.0, 2.0).unwrap(),
          uniformInputs(-kPi, kPi, false));
    bench("Binomial", stats::BinomialDistribution::create(20, 0.5).unwrap(),
          uniformInputs(0.0, 20.0, true));
    bench("NegBinomial", stats::NegativeBinomialDistribution::create(5.0, 0.5).unwrap(),
          uniformInputs(0.0, 25.0, true));
    bench("Geometric", stats::GeometricDistribution::create(0.5).unwrap(),
          uniformInputs(0.0, 10.0, true));
    bench("Laplace", stats::LaplaceDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-5.0, 5.0, false));
    bench("Cauchy", stats::CauchyDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-10.0, 10.0, false));
    bench("Logistic", stats::LogisticDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-10.0, 10.0, false));
    bench("Gumbel", stats::GumbelDistribution::create(0.0, 1.0).unwrap(),
          uniformInputs(-3.0, 10.0, false));
    bench("HalfNormal", stats::HalfNormalDistribution::create(1.0).unwrap(),
          uniformInputs(0.0, 4.0, false));
    bench("TruncatedNormal",
          stats::TruncatedNormalDistribution::create(0.0, 1.0, -2.0, 2.0).unwrap(),
          uniformInputs(-2.5, 2.5, false));
    bench("Bernoulli", stats::BernoulliDistribution::create(0.5).unwrap(),
          uniformInputs(0.0, 1.0, true));
    bench("Erlang", stats::ErlangDistribution::create(2, 1.0).unwrap(),
          uniformInputs(0.05, 8.0, false));
    bench("FisherF", stats::FDistribution::create(5.0, 10.0).unwrap(),
          uniformInputs(0.0, 5.0, false));
    bench("InverseGamma", stats::InverseGammaDistribution::create(3.0, 2.0).unwrap(),
          uniformInputs(0.05, 4.0, false));
    return sink == 12345.678 ? 1 : 0;
}
