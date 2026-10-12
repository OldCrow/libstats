#include "seam.h"

#include <cstdlib>
int main(int c, char** v) {
    double nu = atof(v[1]);
    auto d = StudentTDistribution::create(nu).unwrap();
    for (int i = 2; i < c; ++i) {
        double t = atof(v[i]);
        for (int k = 0; k < 3; ++k)
            t = std::nextafter(t, -1e9);
        for (int k = 0; k < 7; ++k, t = std::nextafter(t, 1e9))
            printf("%.17g %.17g\n", t, d.getCumulativeProbability(t));
    }
}
