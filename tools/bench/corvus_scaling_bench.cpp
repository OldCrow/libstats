// corvus per-call scaling: ns/element vs span length, min of 9 runs.
#include <chrono>
#include <corvus/corvus.h>
#include <cstdio>
#include <random>
#include <span>
#include <vector>
using clk = std::chrono::steady_clock;
static double sink;
template <class F>
double ns_per_elem(std::size_t n, std::size_t reps, F&& f) {
    double best = 1e300;
    for (int r = 0; r < 9; ++r) {
        auto t0 = clk::now();
        for (std::size_t k = 0; k < reps; ++k)
            f();
        auto t1 = clk::now();
        best =
            std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / (reps * n));
    }
    return best;
}
int main() {
    std::printf("corvus %s\n", corvus::active_target());
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> ux(0.05, 20.0), ub(0.01, 0.99), up(0.001, 0.999);
    const std::size_t Ns[] = {1, 2, 4, 8, 32, 256, 4096, 65536};
    std::printf("%8s %10s %10s %10s %10s %10s %10s\n", "n", "erf", "exp", "lgamma", "gamma_p",
                "beta_p", "gamma_p_inv");
    for (std::size_t n : Ns) {
        std::vector<double> a(n, 2.5), b(n, 5.0), x(n), xb(n), p(n), out(n);
        for (auto& v : x)
            v = ux(rng);
        for (auto& v : xb)
            v = ub(rng);
        for (auto& v : p)
            v = up(rng);
        const std::size_t reps = std::max<std::size_t>(1, 200000 / n);
        std::span<const double> A(a), B(b), X(x), XB(xb), P(p);
        std::span<double> O(out);
        double e = ns_per_elem(n, reps, [&] {
            corvus::erf(XB, O);
            sink += out[0];
        });
        double ex = ns_per_elem(n, reps, [&] {
            corvus::exp(XB, O);
            sink += out[0];
        });
        double lg = ns_per_elem(n, reps, [&] {
            corvus::lgamma(X, O);
            sink += out[0];
        });
        double gp = ns_per_elem(n, reps, [&] {
            corvus::gamma_p(A, X, O);
            sink += out[0];
        });
        double bp = ns_per_elem(n, reps, [&] {
            corvus::beta_p(A, B, XB, O);
            sink += out[0];
        });
        double gi = ns_per_elem(n, reps, [&] {
            corvus::gamma_p_inv(A, P, O);
            sink += out[0];
        });
        std::printf("%8zu %10.1f %10.1f %10.1f %10.1f %10.1f %10.1f\n", n, e, ex, lg, gp, bp, gi);
    }
    std::printf("(sink %g)\n", sink);
}
