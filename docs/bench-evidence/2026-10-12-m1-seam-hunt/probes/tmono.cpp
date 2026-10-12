#define LIBSTATS_FULL_INTERFACE
#include "libstats/libstats.h"

#include <cmath>
#include <cstdio>
using namespace stats;
int main() {
    for (double nu : {100.0, 1000.0, 30.0}) {
        const auto d = StudentTDistribution::create(nu).unwrap();
        for (double t0 :
             {-std::sqrt(nu) * 1.0001, -10.0, -std::sqrt(nu) * 2, -std::sqrt(nu) * 0.9}) {
            double t = t0, prev = d.getCumulativeProbability(t), worst = 0, avgstep = 0;
            int down = 0;
            for (int i = 0; i < 2000; ++i) {
                t = std::nextafter(t, 0.0);
                const double f = d.getCumulativeProbability(t);
                const double ulp = std::nextafter(prev, 1.0) - prev;
                if (f < prev) {
                    ++down;
                    worst = std::fmax(worst, (prev - f) / ulp);
                }
                avgstep += (f - prev) / ulp;
                prev = f;
            }
            std::printf("nu %g t0 %.6g: down %d/2000, worst %.0f ulp, mean step %.1f ulp\n", nu, t0,
                        down, worst, avgstep / 2000);
        }
    }
}
