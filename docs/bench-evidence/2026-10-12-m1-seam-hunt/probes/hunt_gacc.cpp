#include "seam.h"
int main() {
    for (int a = 2; a <= 40; ++a)
        for (double f : {0.8, 1.0, 1.05, 1.3}) {
            double x = a * f + (f == 1.05 ? 1 : 0);
            auto d = GammaDistribution::create(a, 1.0).unwrap();
            printf("%d %.17g %.17g %.17g\n", a, x, d.getCumulativeProbability(x), d.getSurvival(x));
        }
}
