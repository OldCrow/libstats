/**
 * @file test_bessel_tier.cpp
 * @brief Guards for #97 — the Bessel tier every TU compiles, and the domain
 *        agreement between the two tiers.
 *
 * These are build-plumbing guards rather than distribution tests, which is why
 * they live in their own binary. The first version of them was appended to
 * test_von_mises_enhanced, which carries the "timing" label — and CI runs
 * `ctest -LE "timing|benchmark"`, so they never ran there. A guard that does
 * not run guards nothing; this file is deliberately unlabelled so it is part of
 * the standard correctness suite.
 */

#include "libstats/core/bessel.h"

#include <gtest/gtest.h>

TEST(BesselTier, MatchesTheBuild) {
    // Two-sided on purpose. Asserting only "Tier 2 is within 1.6e-7" would pass
    // on a Tier 1 build as well, so a silent revert to a target-scoped
    // definition would go unnoticed — which is exactly how #97 survived. So
    // decide INDEPENDENTLY of libstats whether this compiler has the C++17
    // special math functions, then require libstats to have reached the same
    // conclusion in this TU.
#if defined(__cpp_lib_math_special_functions)
    constexpr bool compiler_has_it = true;
#else
    constexpr bool compiler_has_it = false;
#endif
#if defined(LIBSTATS_HAS_CXX17_BESSEL)
    constexpr bool libstats_selected_it = true;
#else
    constexpr bool libstats_selected_it = false;
#endif

    if (compiler_has_it) {
        EXPECT_TRUE(libstats_selected_it)
            << "this TU has std::cyl_bessel_i but did not get "
               "LIBSTATS_HAS_CXX17_BESSEL — libstats_config.h is not reaching every TU, so "
               "some are compiling a different Bessel tier than the library ships (#97)";
    }

    // And the tier that was selected must behave like the tier it claims to be.
    constexpr double kI0_1 = 1.2660658777520084;   // mpmath, dps 50
    constexpr double kI1_1 = 0.56515910399248503;  // mpmath, dps 50
    const double tol = libstats_selected_it ? 1e-14 : 1.6e-7;
    EXPECT_NEAR(stats::detail::bessel_i0(1.0), kI0_1, tol);
    EXPECT_NEAR(stats::detail::bessel_i1(1.0), kI1_1, tol);
}

TEST(BesselTier, NegativeArgumentsAreSymmetric) {
    // libstdc++ throws std::domain_error for a negative argument to
    // std::cyl_bessel_i, and these helpers are noexcept — so before the #97 fix
    // this was std::terminate on GCC/Linux and merely wrong on tiers that fold
    // through fabs. MSVC does not throw, which is why Windows never saw it.
    // I₀ is even, I₁ is odd, and both tiers must agree on that.
    for (const double x : {0.5, 1.0, 3.0, 12.0}) {
        EXPECT_DOUBLE_EQ(stats::detail::bessel_i0(-x), stats::detail::bessel_i0(x));
        EXPECT_DOUBLE_EQ(stats::detail::bessel_i1(-x), -stats::detail::bessel_i1(x));
        EXPECT_DOUBLE_EQ(stats::detail::log_bessel_i0(-x), stats::detail::log_bessel_i0(x));
    }
    // Past the log I0 asymptotic seam as well.
    EXPECT_DOUBLE_EQ(stats::detail::log_bessel_i0(-800.0), stats::detail::log_bessel_i0(800.0));
}
