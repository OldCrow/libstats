#include "seam.h"

#include <cstdlib>
int main(int c, char** v) {
    double a = atof(v[1]), b = atof(v[2]), x = atof(v[3]);
    for (int i = 0; i < 3; ++i)
        a = std::nextafter(a, 0.0);
    for (int i = 0; i < 6; ++i, a = std::nextafter(a, 1e9))
        printf("%.17g %.17g\n", a, BetaDistribution::create(a, b).unwrap().getLogProbability(x));
}
