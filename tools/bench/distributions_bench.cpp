// Same-source microbenchmark, compiled against v2.4.1 and dev/v2.5.0-corvus.
// Reports ns/element (min of 7 runs) for batch cdf/pdf/logpdf at 1e6 elements
// (auto dispatch), scalar cdf at 1e5, and quantile at 1e4.
#include "libstats/distributions/beta.h"
#include "libstats/distributions/binomial.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/lognormal.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/distributions/weibull.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <span>
#include <vector>
using clk = std::chrono::steady_clock;
static double sink = 0.0;
template <class F>
double ns_per(std::size_t n, F&& f) {
    double best = 1e300;
    for (int r = 0; r < 7; ++r) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / n);
    }
    return best;
}
template <class D>
void bench(const char* name, D& d, double lo, double hi, bool integer) {
    const std::size_t N = 1'000'000, NS = 100'000, NQ = 10'000;
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(lo, hi);
    std::vector<double> x(N), y(N), p(NQ);
    for (auto& v : x) {
        v = u(rng);
        if (integer)
            v = std::round(v);
    }
    std::uniform_real_distribution<double> up(1e-6, 1 - 1e-6);
    for (auto& v : p)
        v = up(rng);
    std::span<const double> xs(x);
    std::span<double> ys(y);
    double cdf_b = ns_per(N, [&] {
        d.getCumulativeProbability(xs, ys);
        sink += y[N / 2];
    });
    double pdf_b = ns_per(N, [&] {
        d.getProbability(xs, ys);
        sink += y[N / 2];
    });
    double lpdf_b = ns_per(N, [&] {
        d.getLogProbability(xs, ys);
        sink += y[N / 2];
    });
    double cdf_s = ns_per(NS, [&] {
        double a = 0;
        for (std::size_t i = 0; i < NS; ++i)
            a += d.getCumulativeProbability(x[i]);
        sink += a;
    });
    double pdf_s = ns_per(NS, [&] {
        double a = 0;
        for (std::size_t i = 0; i < NS; ++i)
            a += d.getProbability(x[i]);
        sink += a;
    });
    double lpdf_s = ns_per(NS, [&] {
        double a = 0;
        for (std::size_t i = 0; i < NS; ++i)
            a += d.getLogProbability(x[i]);
        sink += a;
    });
    double q = ns_per(NQ, [&] {
        double a = 0;
        for (std::size_t i = 0; i < NQ; ++i)
            a += d.getQuantile(p[i]);
        sink += a;
    });
    std::printf("%-12s %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f %9.1f\n", name, pdf_b, lpdf_b, cdf_b,
                pdf_s, lpdf_s, cdf_s, q);
}
// LIBSTATS_BENCH_WARMUP_SECONDS=N: spin N seconds before the first measurement, so every row is
// taken in the sustained-frequency regime. Off by default. Needed on the Zen 4 box, where boost
// steps down ~1.5x some 8-15 s into a run and lands on different rows of the two binaries.
static void warmup() {
    const char* s = std::getenv("LIBSTATS_BENCH_WARMUP_SECONDS");
    const double secs = s ? std::atof(s) : 0.0;
    if (secs <= 0.0)
        return;
    volatile double x = 1.0;
    const auto end = clk::now() + std::chrono::duration<double>(secs);
    while (clk::now() < end)
        for (int i = 0; i < 1000; ++i)
            x = x * 1.0000001 + 1e-9;
    std::printf("(warm-up %.0f s)\n", secs);
}
int main() {
    warmup();
    std::printf("%-12s %9s %9s %9s %9s %9s %9s %9s\n", "ns/elem", "pdf@1e6", "lpdf@1e6", "cdf@1e6",
                "pdf-sc", "lpdf-sc", "cdf-sc", "quantile");
    auto g = stats::GammaDistribution::create(2.5, 1.5);
    bench("gamma", *g, 0.0, 20.0, false);
    auto b = stats::BetaDistribution::create(2.0, 5.0);
    bench("beta", *b, 0.0, 1.0, false);
    auto po = stats::PoissonDistribution::create(12.0);
    bench("poisson", *po, 0.0, 40.0, true);
    auto t = stats::StudentTDistribution::create(7.0);
    bench("student_t", *t, -8.0, 8.0, false);
    auto bi = stats::BinomialDistribution::create(50, 0.3);
    bench("binomial", *bi, 0.0, 50.0, true);
    auto n = stats::GaussianDistribution::create(0.0, 1.0);
    bench("gaussian", *n, -5.0, 5.0, false);
    auto ln = stats::LogNormalDistribution::create(0.0, 1.0);
    bench("lognormal", *ln, 0.01, 10.0, false);
    auto vm = stats::VonMisesDistribution::create(0.0, 3.0);
    bench("von_mises", *vm, -3.14, 3.14, false);
    auto w = stats::WeibullDistribution::create(1.5, 2.0);
    bench("weibull", *w, 0.01, 6.0, false);
    auto e = stats::ExponentialDistribution::create(1.3);
    bench("exponential", *e, 0.0, 6.0, false);
    std::printf("(sink %g)\n", sink);
}
