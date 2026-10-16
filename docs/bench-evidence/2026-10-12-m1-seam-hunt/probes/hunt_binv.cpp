#include "seam.h"

#include <cstdlib>
int main(int c, char** v) {
    int n = atoi(v[1]);
    int k = atoi(v[2]);
    double p = atof(v[3]);
    for (int i = 0; i < 3; ++i)
        p = std::nextafter(p, 0.0);
    for (int i = 0; i < 7; ++i, p = std::nextafter(p, 1.0)) {
        auto d = BinomialDistribution::create(n, p).unwrap();
        printf("%.17g %.17g %.17g\n", p, d.getCumulativeProbability(k), d.getSurvival(k));
    }
}
