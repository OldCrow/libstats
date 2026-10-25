#include "libstats/platform/simd.h"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>
using clk = std::chrono::steady_clock;
static double sink;
template <class F>
double ns_per(std::size_t n, F&& f) {
    double b = 1e300;
    for (int r = 0; r < 9; ++r) {
        auto t0 = clk::now();
        f();
        auto t1 = clk::now();
        b = std::min(b, std::chrono::duration<double, std::nano>(t1 - t0).count() / n);
    }
    return b;
}
int main() {
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
