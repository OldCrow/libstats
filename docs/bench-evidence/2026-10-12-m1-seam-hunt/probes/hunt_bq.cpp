#include "seam.h"
int main() {
    for (double a : {1e-3, 1e-6, 1e-10, 1e-15, 1e-20, 1e-30, 1e-100}) {
        auto d = BetaDistribution::create(a, a).unwrap();
        printf("a=%g\n", a);
        double p = 0.5;
        for (int k = 0; k < 12; ++k) {
            p = std::nextafter(p, 0.0);
            if (k % 2)
                continue;
            double q = d.getQuantile(p);
            printf("  p=0.5-%2d ulp Q=%.17g F(Q)=%.17g F(Q)-p=%.3g\n", k + 1, q,
                   d.getCumulativeProbability(q), d.getCumulativeProbability(q) - p);
        }
        for (double pp : {0.5 - 1e-14, 0.4999, 0.49, 0.3}) {
            double q = d.getQuantile(pp);
            printf("  p=%.17g Q=%.17g F(Q)-p=%.3g\n", pp, q, d.getCumulativeProbability(q) - pp);
        }
    }
}
