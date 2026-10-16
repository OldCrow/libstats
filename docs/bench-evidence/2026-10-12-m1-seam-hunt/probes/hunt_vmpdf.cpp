#include "seam.h"
int main() {
    for (double k : {10.0, 1e3, 1e5, 1e8, 1e10, 1e13, 1e15, 1e16, 1e17, 1e20, 1e25}) {
        auto d = VonMisesDistribution::create(0.0, k).unwrap();
        // exact: pdf(0)=1/(2 pi I0e(k)) ~ sqrt(k/2pi)/(1+1/(8k)+...)
        double t = 1 / k;
        double ex =
            std::sqrt(k / (2 * M_PI)) / (1 + t * (0.125 + t * (0.0703125 + t * 0.0732421875)));
        double x1 = 1 / std::sqrt(k);
        double ex1 = ex * std::exp(k * (std::cos(x1) - 1));
        printf(
            "k=%g pdf(0)=%.17g exact=%.17g rel=%.2e | pdf(1/sqrt k)=%.17g exact=%.17g rel=%.2e | "
            "logpdf(0)=%.17g\n",
            k, d.getProbability(0.0), ex, d.getProbability(0.0) / ex - 1, d.getProbability(x1), ex1,
            d.getProbability(x1) / ex1 - 1, d.getLogProbability(0.0));
    }
    auto d = VonMisesDistribution::create(0.0, 0.5).unwrap();
    printf("VM(0,0.5) CDF(-M_PI)=%.17g CDF(nextup)=%.17g CDF(M_PI)=%.17g\n",
           d.getCumulativeProbability(-M_PI),
           d.getCumulativeProbability(std::nextafter(-M_PI, 0.0)),
           d.getCumulativeProbability(M_PI));
}
