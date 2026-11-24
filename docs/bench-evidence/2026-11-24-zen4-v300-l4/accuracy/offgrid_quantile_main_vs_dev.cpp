// The 14 off-grid quantile rows over 100 eps: print each library's quantile, exact bits.
#include "libstats/distributions/fisher_f.h"
#include "libstats/distributions/inverse_gamma.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/weibull.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
struct Row { const char* dist; std::uint64_t p1, p2, p; };
static double d(std::uint64_t b) { double x; std::memcpy(&x, &b, 8); return x; }
int main() {
    const Row rows[] = {
#include "qrows.inc"
    };
    for (const Row& r : rows) {
        const std::string n = r.dist; const double a = d(r.p1), b = d(r.p2), p = d(r.p);
        double q = 0;
        if (n == "student_t") q = stats::StudentTDistribution(a).getQuantile(p);
        else if (n == "fisher_f") q = stats::FDistribution(a, b).getQuantile(p);
        else if (n == "inverse_gamma") q = stats::InverseGammaDistribution(a, b).getQuantile(p);
        else q = stats::WeibullDistribution(a, b).getQuantile(p);
        std::uint64_t bits; std::memcpy(&bits, &q, 8);
        std::printf("%s,%.17g,%.17g,%.17g,%.17g,0x%016llx\n", r.dist, a, b, p, q, static_cast<unsigned long long>(bits));
    }
}
