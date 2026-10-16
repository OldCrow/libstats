#include "seam.h"
int main() {
    // gamma_p_inv final-correction switch |t|*slope = 16 (lower tail: slope ~ a, so t = -16/a)
    for (double a : {0.03, 0.1, 0.5, 1.0, 2.0, 5.0}) {
        auto d = GammaDistribution::create(a, 1.0).unwrap();
        const double t = -16.0 / a;
        const double p = std::exp(a * t - std::lgamma(a + 1));
        if (p > 1e-300)
            qSeamC("Gamma Q |t|*slope=16", fmt("Gamma(%g,1)", a), d, p);
    }
    // F extreme: leading-term limit log(a+b) - |log r| = log 2^-60, and the p<=q pick at P(s,
    // lambda) = 1/2
    for (double d1 : {0.5, 1.0, 2.0, 5.0})
        for (double d2 : {1e300, 1e308}) {
            auto f = FDistribution::create(d1, d2).unwrap();
            const double a = d1 / 2, b = d2 / 2;
            const double x = d2 / (d1 * (a + b) * std::ldexp(1.0, 60));
            cdfSeamC("F extreme leading-term limit", fmt("F(%g,%g)", d1, d2), f, x, true, 0,
                     INFINITY);
            const double xm = bisectD(
                1e-300, 1e300, [&](double x) { return f.getCumulativeProbability(x) >= 0.5; });
            cdfSeamC("F extreme P(s,lambda)=1/2 pick", fmt("F(%g,%g)", d1, d2), f, xm, true, 0,
                     INFINITY);
        }
    report();
}
