#include "seam.h"

#include <cstdlib>
// usage: bval a b x0 n  -> prints x, F for n neighbours each side
int main(int c, char** v) {
    double a = atof(v[1]), b = atof(v[2]), x = atof(v[3]);
    int n = atoi(v[4]);
    auto d = BetaDistribution::create(a, b).unwrap();
    for (int i = 0; i < n; ++i)
        x = std::nextafter(x, 0.0);
    for (int i = 0; i < 2 * n + 1; ++i, x = std::nextafter(x, 1.0))
        printf("%.17g %.17g\n", x, d.getCumulativeProbability(x));
}
