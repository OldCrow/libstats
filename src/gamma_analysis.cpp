#include "libstats/stats/analysis/gamma_analysis.h"

#include "libstats/core/math_utils.h"
#include "libstats/core/statistical_constants.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace stats::analysis::gamma {

std::tuple<double, double, bool>
normalApproximationTest(const std::vector<double>& data, double significance_level) {
    if (data.empty())
        throw std::invalid_argument("Data vector cannot be empty");
    if (significance_level <= 0.0 || significance_level >= 1.0)
        throw std::invalid_argument("Significance level must be in (0, 1)");

    const std::size_t n = data.size();
    const double nd  = static_cast<double>(n);
    const double sum  = std::accumulate(data.begin(), data.end(), 0.0);
    const double sum2 = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    const double mean = sum / nd;
    const double var  = sum2 / nd - mean * mean;

    // Method-of-moments: α̂ = mean²/var, β̂ = mean/var
    const double alpha_hat = (var > 0.0) ? (mean * mean / var) : 1.0;
    const double beta_hat  = (var > 0.0 && mean > 0.0) ? (mean / var) : 1.0;

    // For large α, Gamma(α,β) ≈ N(α/β, α/(n·β²)).
    // CI for the DATA MEAN x̄ under the normal approximation:
    //   x̄ ± z · sqrt(α̂/(n·β̂²))
    const double ci_sd  = std::sqrt(alpha_hat / (nd * beta_hat * beta_hat));
    const double z_crit = detail::inverse_normal_cdf(1.0 - significance_level / 2.0);
    const double mu_hat = alpha_hat / beta_hat;  // = sample mean by MoM construction

    const double lower = mu_hat - z_crit * ci_sd;
    const double upper = mu_hat + z_crit * ci_sd;

    // Valid if: (a) α̂ is large enough for the normal approximation (rule of thumb: ≥ 30)
    //           (b) the observed mean lies within the normal-approximation CI for the mean.
    const bool valid = (alpha_hat >= 30.0) && (mean >= lower && mean <= upper);
    return {lower, upper, valid};
}

}  // namespace stats::analysis::gamma
