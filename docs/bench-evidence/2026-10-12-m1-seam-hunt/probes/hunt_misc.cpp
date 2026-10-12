#include "seam.h"
int main() {
    for (double x : {19.999999999999865, 19.999999999999869, 19.99999999999997, 20.0,
                     20.000000000000004, 13.0, 17.5})
        printf("lbeta %.17g %.17g\n", x, detail::lbeta(x, 0.5));
    for (double d : {1e4, 1e6, 1e8, 1e10}) {
        auto f = FDistribution::create(d, d).unwrap();
        printf("Flogpdf %g %.17g %.17g\n", d, f.getLogProbability(1.0), f.getLogProbability(0.9));
    }
    auto f = FDistribution::create(0.1, 0.001).unwrap();
    printf("F(0.1,0.001) SF(DBL_MAX)=%.17g CDF(DBL_MAX)=%.17g Q(0.49)=%g Q(0.3)=%g\n",
           f.getSurvival(1.7976931348623157e308),
           f.getCumulativeProbability(1.7976931348623157e308), f.getQuantile(0.49),
           f.getQuantile(0.3));
}
