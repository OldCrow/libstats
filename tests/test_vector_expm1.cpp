// VectorOps::vector_expm1: the generic expm1 (Taylor polynomial below |x| = ½, vector_exp less 1
// above).
//
// Accuracy against std::expm1 across magnitudes and both signs, the IEEE edges (±0 with its sign,
// ±inf, NaN, overflow), in-place calls (VectorOps kernels may alias) and sizes around the
// 256-element block and the SIMD threshold.

#include "libstats/platform/simd.h"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

using stats::arch::simd::VectorOps;

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Log-spaced magnitudes from 1e-300 to 700, both signs, and points either side of ½.
std::vector<double> sweepInputs() {
    std::vector<double> xs;
    for (int e = -300; e <= 2; ++e) {
        const double m = std::pow(10.0, e);
        for (double f : {1.0, 2.3, 4.9}) {
            if (f * m <= 700.0) {
                xs.push_back(f * m);
                xs.push_back(-f * m);
            }
        }
    }
    for (double x :
         {0.4999999, 0.5, 0.5000001, -0.4999999, -0.5, -0.5000001, 30.0, -30.0, 700.0, -700.0})
        xs.push_back(x);
    return xs;
}

}  // namespace

TEST(VectorExpm1, AccurateAcrossMagnitudes) {
    const std::vector<double> xs = sweepInputs();
    std::vector<double> out(xs.size());
    VectorOps::vector_expm1(xs.data(), out.data(), xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double want = std::expm1(xs[i]);
        // vector_exp's error times at most 2.5 above ½, with margin; the polynomial is ~1 ulp.
        EXPECT_LE(std::fabs(out[i] - want), 8 * kEps * std::fabs(want))
            << "expm1(" << xs[i] << ") = " << out[i] << ", want " << want;
    }
}

TEST(VectorExpm1, IeeeEdges) {
    const std::vector<double> xs = {0.0,  -0.0,  kInf, -kInf,
                                    kNaN, 710.0, -1e3, std::numeric_limits<double>::denorm_min()};
    std::vector<double> in(xs);
    in.resize(64, 0.25);  // a SIMD-sized call, so the edges go through the dispatched path
    std::vector<double> out(in.size());
    VectorOps::vector_expm1(in.data(), out.data(), in.size());
    EXPECT_EQ(out[0], 0.0);
    EXPECT_FALSE(std::signbit(out[0]));
    EXPECT_EQ(out[1], 0.0);
    EXPECT_TRUE(std::signbit(out[1]));
    EXPECT_EQ(out[2], kInf);
    EXPECT_EQ(out[3], -1.0);
    EXPECT_TRUE(std::isnan(out[4]));
    EXPECT_EQ(out[5], kInf);
    EXPECT_EQ(out[6], -1.0);
    EXPECT_EQ(out[7], std::numeric_limits<double>::denorm_min());
}

TEST(VectorExpm1, InPlaceMatchesOutOfPlace) {
    const std::vector<double> xs = sweepInputs();
    std::vector<double> out(xs.size());
    VectorOps::vector_expm1(xs.data(), out.data(), xs.size());
    std::vector<double> inplace(xs);
    VectorOps::vector_expm1(inplace.data(), inplace.data(), inplace.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_EQ(inplace[i], out[i]) << "at x = " << xs[i];
}

TEST(VectorExpm1, SizesAroundBlockAndThreshold) {
    for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{255},
                          std::size_t{256}, std::size_t{257}, std::size_t{513}}) {
        std::vector<double> xs(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
            xs[i] = -2.0 + 4.0 * static_cast<double>(i) / static_cast<double>(n);
        VectorOps::vector_expm1(xs.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            const double want = std::expm1(xs[i]);
            EXPECT_LE(std::fabs(out[i] - want), 8 * kEps * std::fabs(want))
                << "n = " << n << ", x = " << xs[i];
        }
    }
}
