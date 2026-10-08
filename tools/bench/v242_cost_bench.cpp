// tools/bench/v242_cost_bench.cpp — what the v2.4.2 accuracy fixes cost, v2.4.1 against v2.4.2.
//
// Standalone, not part of the CMake build: compile this one file against each version and run
// the two binaries back to back on a quiet machine, then join their CSVs with v242_compare.py.
// tools/bench/distributions_bench.cpp (dev/v2.5.0-corvus) samples only small shapes; this covers
// the paths v2.4.2 changed:
//   - Gamma at and around α = 20, where the density and the SIMD batch switch to Stirling form
//     and to the scalar loop; Student-t at ν ≥ 31 and Beta at shapes ≥ 17, the other two scalar
//     fall-backs; the incomplete gamma/beta 3ε stop and √a caps (#166) through every CDF.
//   - von Mises tail CDF quadrature and Newton quantile at κ = 1, 30, 300.
//   - Poisson, Binomial and NegativeBinomial pmfs at large counts (#172).
//   - The quantile solvers (#159, #160, #170, #171).
// Small-shape rows are controls: they should not move.
//
// Output, one CSV row per measurement:
//   case,op,n,strategy,ns_per_element
// op is pdf/logpdf/cdf for batches at n = 1e3, 1e4, 1e5 under each forced strategy and AUTO;
// "call_pdf", "call_logpdf", "call_cdf" (n = 1e4 scalar calls) and "quantile" (n = 1e3 calls)
// carry strategy "call". Each figure is the minimum of 7 timed blocks of at least 1e5 elements.
// LIBSTATS_BENCH_WARMUP_SECONDS=N spins N seconds first; use it on Zen 4, whose boost drops
// ~1.5× some 8–15 s into a run (docs/VALIDATION_HISTORY.md, "Zen 4 frequency scaling").
//
// Build: a Release libstats_static in each tree; OLD = a worktree at tag v2.4.1, NEW = this
// branch.
//   macOS / Linux (build-release from the release preset):
//     clang++ -std=c++20 -O2 -DNDEBUG -I$T/include -I$T/build-release/generated \
//       tools/bench/v242_cost_bench.cpp $T/build-release/libstats.a -lpthread -o bench_<ver>
//   Windows (vcvars64; the flags and definitions the library's own TUs use):
//     cl /std:c++20 /O2 /Ob2 /DNDEBUG /EHsc /MD /utf-8 /DNOMINMAX /D_USE_MATH_DEFINES
//        /D_CRT_SECURE_NO_WARNINGS /DLIBSTATS_HAS_SSE2=1 /DLIBSTATS_HAS_AVX=1 /DLIBSTATS_HAS_AVX2=1
//        /DLIBSTATS_HAS_AVX512=1 /I %T%\include /I %T%\build\generated
//        tools\bench\v242_cost_bench.cpp /Fe:bench_<ver>.exe
//        /link %T%\build\Release\stats_static.lib
// Run:  bench_v241 > v241.csv; bench_v242 > v242.csv; python v242_compare.py v241.csv v242.csv

#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/negative_binomial.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/von_mises.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

constexpr std::size_t kMinBlock = 100'000;  // elements per timed block
constexpr std::size_t kCalls = 10'000;
constexpr std::size_t kQuantiles = 1'000;

// ns per element of f, which processes n elements per call: the minimum over 7 blocks of
// ceil(kMinBlock / n) calls each.
template <class F>
double nsPer(std::size_t n, F&& f) {
    const std::size_t reps = std::max<std::size_t>(1, (kMinBlock + n - 1) / n);
    double best = 1e300;
    for (int r = 0; r < 7; ++r) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i)
            f();
        const auto t1 = clk::now();
        best =
            std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * n));
    }
    return best;
}

void row(const std::string& name, const char* op, std::size_t n, const char* strategy, double ns) {
    std::printf("%s,%s,%zu,%s,%.2f\n", name.c_str(), op, n, strategy, ns);
    std::fflush(stdout);
}

struct Forced {
    const char* name;
    Strategy strategy;
};
constexpr Forced kStrategies[] = {{"auto", Strategy::AUTO},
                                  {"scalar", Strategy::FORCE_SCALAR},
                                  {"vectorized", Strategy::FORCE_VECTORIZED},
                                  {"parallel", Strategy::FORCE_PARALLEL}};

// x uniform on [lo, hi] (rounded for a discrete distribution); p uniform on (1e-6, 1 − 1e-6).
template <class D>
void bench(const std::string& name, const D& d, double lo, double hi, bool integer) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> ux(lo, hi);
    std::vector<double> x(100'000);
    for (auto& v : x)
        v = integer ? std::round(ux(rng)) : ux(rng);
    std::uniform_real_distribution<double> up(1e-6, 1 - 1e-6);
    std::vector<double> p(kQuantiles);
    for (auto& v : p)
        v = up(rng);
    std::vector<double> y(x.size());

    for (std::size_t n : {std::size_t{1'000}, std::size_t{10'000}, std::size_t{100'000}}) {
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

    row(name, "call_pdf", kCalls, "call", nsPer(kCalls, [&] {
            double a = 0;
            for (std::size_t i = 0; i < kCalls; ++i)
                a += d.getProbability(x[i]);
            sink = sink + a;
        }));
    row(name, "call_logpdf", kCalls, "call", nsPer(kCalls, [&] {
            double a = 0;
            for (std::size_t i = 0; i < kCalls; ++i)
                a += d.getLogProbability(x[i]);
            sink = sink + a;
        }));
    row(name, "call_cdf", kCalls, "call", nsPer(kCalls, [&] {
            double a = 0;
            for (std::size_t i = 0; i < kCalls; ++i)
                a += d.getCumulativeProbability(x[i]);
            sink = sink + a;
        }));
    row(name, "quantile", kQuantiles, "call", nsPer(kQuantiles, [&] {
            double a = 0;
            for (std::size_t i = 0; i < kQuantiles; ++i)
                a += d.getQuantile(p[i]);
            sink = sink + a;
        }));
}

// An environment variable's value, empty when unset. The Windows CRT deprecates std::getenv.
std::string envValue(const char* name) {
#ifdef _WIN32
    char* buf = nullptr;
    std::size_t len = 0;
    std::string value;
    if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
        value = buf;
    std::free(buf);
    return value;
#else
    const char* s = std::getenv(name);
    return s ? s : "";
#endif
}

void warmup() {
    const std::string s = envValue("LIBSTATS_BENCH_WARMUP_SECONDS");
    const double secs = s.empty() ? 0.0 : std::atof(s.c_str());
    if (secs <= 0.0)
        return;
    volatile double x = 1.0;
    const auto end = clk::now() + std::chrono::duration<double>(secs);
    while (clk::now() < end)
        for (int i = 0; i < 1000; ++i)
            x = x * 1.0000001 + 1e-9;
}

std::string tag(const char* dist, double a) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s(%g)", dist, a);
    return buf;
}

std::string tag(const char* dist, double a, double b) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%s(%g;%g)", dist, a, b);
    return buf;
}

}  // namespace

int main() {
    warmup();
    std::printf("case,op,n,strategy,ns_per_element\n");

    // Gamma(α, 1) over mean ± 4 sd: the α = 20 switch, a control below, a large shape above.
    for (double alpha : {2.5, 10.0, 19.9, 20.0, 1000.0}) {
        const double sd = std::sqrt(alpha);
        bench(tag("gamma", alpha), stats::GammaDistribution::create(alpha, 1.0).unwrap(),
              std::max(1e-3, alpha - 4 * sd), alpha + 4 * sd, false);
    }
    // Small-shape upper tail, where gamma_q is now formed directly (review M-1).
    bench(tag("gamma", 0.01), stats::GammaDistribution::create(0.01, 1.0).unwrap(), 1e-6, 3.0,
          false);

    // Student-t: SIMD up to ν = 30 ((ν + 1)/2 ≤ 16), scalar above; BGRAT tails at large ν.
    for (double nu : {7.0, 30.0, 1000.0, 1e6})
        bench(tag("student_t", nu), stats::StudentTDistribution::create(nu).unwrap(), -6.0, 6.0,
              false);

    // Beta over mean ± 4 sd: SIMD while |shape − 1| ≤ 16.
    for (double s : {2.0, 25.0, 1000.0}) {
        const double sd = std::sqrt(s * s / ((2 * s) * (2 * s) * (2 * s + 1)));
        bench(tag("beta", s, s), stats::BetaDistribution::create(s, s).unwrap(),
              std::max(1e-6, 0.5 - 4 * sd), std::min(1 - 1e-6, 0.5 + 4 * sd), false);
    }

    // von Mises over the whole circle: tail quadrature wherever F < ¼ or F > ¾.
    for (double kappa : {1.0, 30.0, 300.0})
        bench(tag("von_mises", kappa), stats::VonMisesDistribution::create(0.0, kappa).unwrap(),
              -kPi, kPi, false);

    // Discrete pmfs over mean ± 4 sd: the lgamma form below counts of 20, Stirling above.
    for (double lambda : {12.0, 1e3, 1e5}) {
        const double sd = std::sqrt(lambda);
        bench(tag("poisson", lambda), stats::PoissonDistribution::create(lambda).unwrap(),
              std::max(0.0, lambda - 4 * sd), lambda + 4 * sd, true);
    }
    for (int n : {50, 1000, 100000}) {
        const double mean = 0.3 * n;
        const double sd = std::sqrt(mean * 0.7);
        bench(tag("binomial", n, 0.3), stats::BinomialDistribution::create(n, 0.3).unwrap(),
              std::max(0.0, mean - 4 * sd), mean + 4 * sd, true);
    }
    for (double r : {5.0, 1000.0}) {
        const double mean = r;  // r(1 − p)/p at p = ½
        const double sd = std::sqrt(2 * r);
        bench(tag("negative_binomial", r, 0.5),
              stats::NegativeBinomialDistribution::create(r, 0.5).unwrap(),
              std::max(0.0, mean - 4 * sd), mean + 4 * sd, true);
    }
    return sink == 12345.678 ? 1 : 0;
}
