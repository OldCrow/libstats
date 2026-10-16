// VectorOps::vector_log1p: the generic log1p over each tier's vector_log.
//
// Accuracy against std::log1p across magnitudes and both signs, the IEEE edges (±0 with its sign,
// −1, below −1, ±inf, NaN), in-place calls (VectorOps kernels may alias) and sizes around the
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

// Log-spaced magnitudes from 1e-300 to 1e300, both signs, negatives kept above −1.
std::vector<double> sweepInputs() {
    std::vector<double> xs;
    for (int e = -300; e <= 300; e += 3) {
        const double m = std::pow(10.0, e);
        xs.push_back(m);
        xs.push_back(1.7 * m);
        if (m < 1.0) {
            xs.push_back(-m);
            xs.push_back(-0.7 * m);
        }
    }
    for (double x : {-0.999999, -0.75, -0.5, -0.25, -1e-8, 0.25, 0.5, 1.0, 1.5, 3.0})
        xs.push_back(x);
    return xs;
}

}  // namespace

TEST(VectorLog1p, AccurateAcrossMagnitudes) {
    const std::vector<double> xs = sweepInputs();
    std::vector<double> out(xs.size());
    VectorOps::vector_log1p(xs.data(), out.data(), xs.size());
    for (std::size_t i = 0; i < xs.size(); ++i) {
        const double want = std::log1p(xs[i]);
        // vector_log's error plus the correction's ulp, with margin.
        EXPECT_LE(std::fabs(out[i] - want), 4 * kEps * std::fabs(want))
            << "log1p(" << xs[i] << ") = " << out[i] << ", want " << want;
    }
}

TEST(VectorLog1p, IeeeEdges) {
    const std::vector<double> xs = {0.0,
                                    -0.0,
                                    -1.0,
                                    -2.0,
                                    kInf,
                                    -kInf,
                                    kNaN,
                                    std::numeric_limits<double>::denorm_min(),
                                    -std::numeric_limits<double>::denorm_min()};
    // Pad to a SIMD-sized call so the edges go through the dispatched path.
    std::vector<double> in(xs);
    in.resize(64, 0.5);
    std::vector<double> out(in.size());
    VectorOps::vector_log1p(in.data(), out.data(), in.size());
    EXPECT_EQ(out[0], 0.0);
    EXPECT_FALSE(std::signbit(out[0]));
    EXPECT_EQ(out[1], 0.0);
    EXPECT_TRUE(std::signbit(out[1]));
    EXPECT_EQ(out[2], -kInf);
    EXPECT_TRUE(std::isnan(out[3]));
    EXPECT_EQ(out[4], kInf);
    EXPECT_TRUE(std::isnan(out[5]));
    EXPECT_TRUE(std::isnan(out[6]));
    EXPECT_EQ(out[7], std::numeric_limits<double>::denorm_min());
    EXPECT_EQ(out[8], -std::numeric_limits<double>::denorm_min());
}

TEST(VectorLog1p, InPlaceMatchesOutOfPlace) {
    const std::vector<double> xs = sweepInputs();
    std::vector<double> out(xs.size());
    VectorOps::vector_log1p(xs.data(), out.data(), xs.size());
    std::vector<double> inplace(xs);
    VectorOps::vector_log1p(inplace.data(), inplace.data(), inplace.size());
    for (std::size_t i = 0; i < xs.size(); ++i)
        EXPECT_EQ(inplace[i], out[i]) << "at x = " << xs[i];
}

TEST(VectorLog1p, SizesAroundBlockAndThreshold) {
    for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{255},
                          std::size_t{256}, std::size_t{257}, std::size_t{513}}) {
        std::vector<double> xs(n), out(n);
        for (std::size_t i = 0; i < n; ++i)
            xs[i] = -0.9 + 1.8 * static_cast<double>(i) / static_cast<double>(n);
        VectorOps::vector_log1p(xs.data(), out.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            const double want = std::log1p(xs[i]);
            EXPECT_LE(std::fabs(out[i] - want), 4 * kEps * std::fabs(want) + 4 * kEps * 1e-300)
                << "n = " << n << ", x = " << xs[i];
        }
    }
}
