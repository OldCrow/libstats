#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <cstdio>
#include <span>
#include <vector>
using namespace stats;
int main() {
    // Large kappa: f(mu) = 1/(2 pi I0(k) e^{-k}) ~ sqrt(k/(2 pi)) * (1 + 1/(8k) + ...)
    for (double k : {1e3, 1e5, 1e8, 1e10, 1e13, 1e15, 1e16, 1e20, 1e100}) {
        const auto d = VonMisesDistribution::create(0.0, k).unwrap();
        const double want = std::sqrt(k / (2 * M_PI)) /
                            (1 + 1 / (8 * k) + 9 / (128 * k * k) + 225 / (3072 * k * k * k));
        std::vector<double> x{0.0, 1e-9}, r(2);
        d.getProbability(std::span<const double>(x), std::span<double>(r));
        std::printf(
            "k=%-8g pdf(0)=%.17g want %.17g rel %.2e  logpdf(0)=%.17g want %.17g  batch %.17g\n", k,
            d.getProbability(0.0), want, std::fabs(d.getProbability(0.0) - want) / want,
            d.getLogProbability(0.0), std::log(want), r[0]);
    }
}
