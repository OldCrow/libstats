// VectorOps::vector_exp/log/cos/sin/erf at 1e3 and 1e6 elements, ns/element.
//
// Timing (#320): each cell runs its own work for 0.15 s, then is timed for at least 0.25 s in
// blocks, and reports the fastest block (sustained_timing.h). The earlier minimum of nine calls
// put the whole n = 1000 row inside the cold start of the process: it read 0.72x between two
// binaries whose kernels were the same.
#include "libstats/platform/simd.h"
#include "sustained_timing.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using clk = std::chrono::steady_clock;
static double sink;
template <class F>
double ns_per(std::size_t n, F&& f) {
    return bench_timing::timeCell(f).min_ns / static_cast<double>(n);
}
// LIBSTATS_BENCH_WARMUP_SECONDS=N: spin N seconds before the first measurement, so every row is
// taken in the sustained-frequency regime. Off by default. Needed on the Zen 4 box, where boost
// steps down ~1.5x some 8-15 s into a run and lands on different rows of the two binaries.
static void warmup() {
    const std::string s = bench_timing::envValue("LIBSTATS_BENCH_WARMUP_SECONDS");
    const double secs = s.empty() ? 0.0 : std::atof(s.c_str());
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
    using V = stats::arch::simd::VectorOps;
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> u(0.05, 20.0), ue(-3.0, 3.0);
    for (std::size_t n : {std::size_t(1000), std::size_t(1000000)}) {
        std::vector<double> x(n), xe(n), y(n);
        for (auto& v : x)
            v = u(rng);
        for (auto& v : xe)
            v = ue(rng);
        std::printf("n=%-8zu exp %5.2f  log %5.2f  cos %5.2f  sin %5.2f  erf %5.2f\n", n,
                    ns_per(n,
                           [&] {
                               V::vector_exp(xe.data(), y.data(), n);
                               sink += y[n / 2];
                           }),
                    ns_per(n,
                           [&] {
                               V::vector_log(x.data(), y.data(), n);
                               sink += y[n / 2];
                           }),
                    ns_per(n,
                           [&] {
                               V::vector_cos(x.data(), y.data(), n);
                               sink += y[n / 2];
                           }),
                    ns_per(n,
                           [&] {
                               V::vector_sin(x.data(), y.data(), n);
                               sink += y[n / 2];
                           }),
                    ns_per(n, [&] {
                        V::vector_erf(xe.data(), y.data(), n);
                        sink += y[n / 2];
                    }));
    }
    std::printf("(sink %g)\n", sink);
}
