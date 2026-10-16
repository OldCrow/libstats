#include "seam.h"
int main() {
    for (double a : {2.0, 0.5, 30.0})
        for (double m : {0.3, 1.0, 3.0})
            for (double B0 : {1e100, 1.003e100}) {
                const double x = m * a * 1e-100;
                walk(
                    fmt("Beta CF b~%g CDF(b)", B0), fmt("Beta(%g,*) x=%g", a, x),
                    [&](double b) {
                        return BetaDistribution::create(a, b).unwrap().getCumulativeProbability(x);
                    },
                    B0, +1, true, {}, 48, 30);
                walk(
                    fmt("NegBin CF k~%g CDF(k)", B0), fmt("NB(%g,*) ", a, 0),
                    [&](double k) {
                        return NegativeBinomialDistribution::create(a, m * 1e-100)
                            .unwrap()
                            .getCumulativeProbability(k);
                    },
                    B0, +1, true, {}, 48, 30);
            }
    report();
}
