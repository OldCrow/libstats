/**
 * @file simd_verification.cpp
 * @brief Architecture-Agnostic SIMD Operations Verification Tool
 *
 * This tool verifies that SIMD operations produce identical results to scalar operations
 * on the actual system it runs on, regardless of architecture (x86, ARM, etc.).
 *
 * Features:
 * - Tests the actual active SIMD level (AVX512, AVX2, AVX, SSE2, NEON, or Scalar)
 * - Validates vectorized vs scalar mathematical equivalence for all six distributions
 * - Tests edge cases (NaN, infinity, subnormals, zeros)
 * - Performance vs accuracy analysis using actual system capabilities
 * - Architecture-agnostic design works on Intel, AMD, Apple Silicon, ARM, etc.
 * - Uses libstats' built-in SIMD detection and capabilities
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

// Additional standard library includes for SIMD verification
#include "libstats/distributions/beta.h"
#include "libstats/distributions/chi_squared.h"
#include "libstats/distributions/discrete.h"
#include "libstats/distributions/exponential.h"
#include "libstats/distributions/gamma.h"
#include "libstats/distributions/gaussian.h"
#include "libstats/distributions/poisson.h"
#include "libstats/distributions/student_t.h"
#include "libstats/distributions/uniform.h"
#include "libstats/distributions/geometric.h"
#include "libstats/distributions/laplace.h"
#include "libstats/distributions/cauchy.h"
#include "libstats/distributions/von_mises.h"
#include "libstats/platform/simd.h"

#include <algorithm>   // for std::max, std::min, std::count_if, std::clamp
#include <chrono>      // for timing operations
#include <cmath>       // for mathematical functions, std::abs
#include <cstddef>     // for size_t
#include <functional>  // for std::function
#include <iomanip>     // for std::setprecision, std::scientific
#include <iostream>    // for std::cout
#include <limits>      // for std::numeric_limits
#include <map>         // for std::map
#include <random>      // for std::mt19937, random distributions
#include <span>        // for std::span
#include <sstream>     // for std::ostringstream
#include <string>      // for std::string, to_string
#include <vector>      // for std::vector

using namespace stats;
using namespace stats::detail;

namespace {
constexpr int VERIFICATION_SEED = 12345;
constexpr size_t TEST_SIZE = 1024;  // Size for correctness tests
constexpr int TEST_ITERATIONS = 5;
constexpr double TOLERANCE_NORMAL = 1e-14;   // Normal numerical precision
constexpr double TOLERANCE_RELAXED = 1e-12;  // Relaxed for complex operations
constexpr double TOLERANCE_ERF_APPROX =
    1e-13;  // musl rational polynomial (measured max ~2.2e-16; must stay > TOLERANCE_NORMAL so
            // absolute-only mode only fires for erf-derived ops)
constexpr double TOLERANCE_COS = 1e-9;  // 7-term Horner polynomial (max error ~1e-10, 10x headroom)
constexpr double TOLERANCE_VONMISES =
    5e-10;  // VonMises batch uses vector_cos; abs error floor ~1e-10

// Edge case test values that are architecture-independent
const std::vector<double> EDGE_CASES = {0.0,
                                        -0.0,
                                        std::numeric_limits<double>::min(),
                                        std::numeric_limits<double>::max(),
                                        std::numeric_limits<double>::denorm_min(),
                                        std::numeric_limits<double>::infinity(),
                                        -std::numeric_limits<double>::infinity(),
                                        std::numeric_limits<double>::quiet_NaN(),
                                        1.0,
                                        -1.0,
                                        1e-100,
                                        1e100,
                                        -1e-100,
                                        -1e100,
                                        0.5,
                                        -0.5,
                                        2.0,
                                        -2.0,
                                        std::numeric_limits<double>::epsilon(),
                                        1.0 + std::numeric_limits<double>::epsilon(),
                                        1.0 - std::numeric_limits<double>::epsilon()};
}  // namespace

struct VerificationResult {
    std::string distribution_name;
    std::string operation_name;
    size_t test_size;
    bool correctness_passed;
    double max_difference;
    double avg_difference;
    size_t failed_comparisons;
    double scalar_time_ns;
    double simd_time_ns;
    double speedup_ratio;
    std::string error_details;
    std::string simd_level_used;
};

class SIMDVerifier {
   private:
    std::mt19937 rng_;
    std::vector<VerificationResult> results_;
    std::string active_simd_level_;

   public:
    SIMDVerifier() : rng_(VERIFICATION_SEED) {
        // Get the actual active SIMD level from libstats
        active_simd_level_ = stats::arch::simd::VectorOps::get_active_simd_level();
    }

    void runVerification() {
        stats::detail::detail::displayToolHeader(
            "SIMD Verification Tool",
            "Validates SIMD operations correctness and performance on actual system architecture");

        // Display system SIMD capabilities
        displaySystemSIMDInfo();

        // Test all distributions with different operations (same order as v1.4.0)
        testUniformDistribution();
        testGaussianDistribution();
        testExponentialDistribution();
        testDiscreteDistribution();
        testPoissonDistribution();
        testGammaDistribution();
        testChiSquaredDistribution();
        testStudentTDistribution();
        testBetaDistribution();

        // Test edge cases (same order as v1.4.0 to preserve rng_ state)
        testEdgeCases();

        // New v1.5.0 coverage: placed AFTER all existing tests so that the shared
        // rng_ state (and hence test data) for the original 54 tests is identical
        // to v1.4.0 baselines. Both new functions are rng_-independent.
        testVonMisesDistribution();  // uses local RNG

        // New v2.0.0 distributions — placed after all existing tests to preserve
        // rng_ state for the established baselines.
        testGeometricDistribution();  // uses local RNG (integer outputs)
        testLaplaceDistribution();     // uses deterministic linspace data
        testCauchyDistribution();      // uses deterministic linspace data

        testPrimitiveVectorOps();    // uses deterministic linspace data

        // Analyze and report results
        analyzeResults();
    }

   private:
    void displaySystemSIMDInfo() {
        stats::detail::detail::subsectionHeader("System SIMD Capabilities");

        const auto& features = stats::arch::get_features();
        std::cout << "Active SIMD Level: " << active_simd_level_ << "\n";
        std::cout << "Architecture: " << stats::detail::detail::getActiveArchitecture() << "\n";

        // Display available SIMD features based on architecture
        stats::detail::detail::ColumnFormatter formatter({20, 10, 30});
        std::cout << formatter.formatRow({"SIMD Feature", "Available", "Description"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";

#ifdef __x86_64__
        std::cout << formatter.formatRow(
                         {"AVX-512", features.avx512f ? "Yes" : "No", "512-bit vectors"})
                  << "\n";
        std::cout << formatter.formatRow(
                         {"AVX2", features.avx2 ? "Yes" : "No", "256-bit integer vectors"})
                  << "\n";
        std::cout << formatter.formatRow(
                         {"AVX", features.avx ? "Yes" : "No", "256-bit floating-point vectors"})
                  << "\n";
        std::cout << formatter.formatRow({"SSE2", features.sse2 ? "Yes" : "No", "128-bit vectors"})
                  << "\n";
        std::cout << formatter.formatRow({"FMA", features.fma ? "Yes" : "No", "Fused multiply-add"})
                  << "\n";
#elif defined(__aarch64__) || defined(__arm__)
        std::cout << formatter.formatRow(
                         {"NEON", features.neon ? "Yes" : "No", "ARM SIMD instructions"})
                  << "\n";
        std::cout << formatter.formatRow({"FMA", features.fma ? "Yes" : "No", "Fused multiply-add"})
                  << "\n";
#endif

        std::cout << "\nOptimal SIMD block size: " << stats::arch::get_optimal_simd_block_size()
                  << " elements\n";
        std::cout << "Memory alignment: " << stats::arch::get_optimal_alignment() << " bytes\n\n";
    }

    void testLaplaceDistribution() {
        stats::detail::detail::subsectionHeader("Laplace Distribution (standalone)");
        // Standard Laplace(0,1): LogPDF uses fabs + vector_exp pipeline.
        // Correctness: VECTORIZED vs SCALAR should match to TOLERANCE_NORMAL.
        auto dist = stats::LaplaceDistribution::create(0.0, 1.0).value;

        std::vector<double> test_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            test_data[i] = -5.0 + 10.0 * static_cast<double>(i) /
                           static_cast<double>(TEST_SIZE - 1);

        verifyOperation(
            dist, test_data, "PDF", "Laplace",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "LogPDF", "Laplace",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getLogProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getLogProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "CDF", "Laplace",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getCumulativeProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getCumulativeProbability(std::span<const double>(data), std::span<double>(output));
            });
    }

    void testCauchyDistribution() {
        stats::detail::detail::subsectionHeader("Cauchy Distribution (delegates to StudentT(nu=1))");
        // Standard Cauchy(0,1): batch ops transform input then delegate to StudentT.
        // Correctness: VECTORIZED vs SCALAR should match to TOLERANCE_NORMAL.
        auto dist = stats::CauchyDistribution::create(0.0, 1.0).value;

        std::vector<double> test_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            test_data[i] = -10.0 + 20.0 * static_cast<double>(i) /
                           static_cast<double>(TEST_SIZE - 1);

        verifyOperation(
            dist, test_data, "PDF", "Cauchy",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "LogPDF", "Cauchy",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getLogProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getLogProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "CDF", "Cauchy",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getCumulativeProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getCumulativeProbability(std::span<const double>(data), std::span<double>(output));
            });
    }

    void testGeometricDistribution() {
        stats::detail::detail::subsectionHeader("Geometric Distribution (delegates to NegBinomial)");
        // Geometric(p=0.5): PMF(k) = 0.5^(k+1), inputs are non-negative integers.
        // Batch correctness: scalar vs auto-dispatch must match exactly (discrete,
        // no SIMD approximation).
        auto dist = stats::GeometricDistribution::create(0.5).value;

        // Generate integer test data {0, 1, 2, ..., TEST_SIZE-1} mod 20
        std::vector<double> test_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            test_data[i] = static_cast<double>(i % 20);

        verifyOperation(
            dist, test_data, "PMF", "Geometric",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "LogPDF", "Geometric",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getLogProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getLogProbability(std::span<const double>(data), std::span<double>(output));
            });

        verifyOperation(
            dist, test_data, "CDF", "Geometric",
            [](const auto& d, const auto& data, auto& output) {
                for (size_t i = 0; i < data.size(); ++i)
                    output[i] = d.getCumulativeProbability(data[i]);
            },
            [](const auto& d, const auto& data, auto& output) {
                d.getCumulativeProbability(std::span<const double>(data), std::span<double>(output));
            });
    }

    void testPrimitiveVectorOps() {
        stats::detail::detail::subsectionHeader("Primitive Vector Operations");
        std::cout << "Direct SIMD vs scalar benchmarks for each vector op (authoritative speedup "
                     "signal)\n";

        using VectorOps = stats::arch::simd::VectorOps;

        // All inputs are deterministic linspace — no rng_ consumption.

        // vector_exp: linspace over [-500, 500] (clamped to [-708, 709] by the implementation)
        std::vector<double> exp_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            exp_data[i] =
                -500.0 + 1000.0 * static_cast<double>(i) / static_cast<double>(TEST_SIZE - 1);
        verifyVectorOp(
            exp_data, "VectorExp",
            [](const std::vector<double>& in, std::vector<double>& out) {
                for (size_t i = 0; i < in.size(); ++i)
                    out[i] = std::exp(in[i]);
            },
            [](const std::vector<double>& in, std::vector<double>& out) {
                VectorOps::vector_exp(in.data(), out.data(), in.size());
            },
            TOLERANCE_NORMAL);

        // vector_log: linspace over (1e-6, 1000]
        std::vector<double> log_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            log_data[i] = 1e-6 + (1000.0 - 1e-6) * static_cast<double>(i) /
                                     static_cast<double>(TEST_SIZE - 1);
        verifyVectorOp(
            log_data, "VectorLog",
            [](const std::vector<double>& in, std::vector<double>& out) {
                for (size_t i = 0; i < in.size(); ++i)
                    out[i] = std::log(in[i]);
            },
            [](const std::vector<double>& in, std::vector<double>& out) {
                VectorOps::vector_log(in.data(), out.data(), in.size());
            },
            TOLERANCE_NORMAL);

        // vector_erf: linspace over [-8, 8] covers all five regions:
        //   R1 |x|<0.84375, R2 0.84375-1.25, R3 1.25-2.857, R4 2.857-6, R5 |x|>=6.
        // Region boundaries (0.84375, 1.25, 2.857143, 6.0) are naturally sampled.
        std::vector<double> erf_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            erf_data[i] = -8.0 + 16.0 * static_cast<double>(i) / static_cast<double>(TEST_SIZE - 1);
        verifyVectorOp(
            erf_data, "VectorErf",
            [](const std::vector<double>& in, std::vector<double>& out) {
                for (size_t i = 0; i < in.size(); ++i)
                    out[i] = std::erf(in[i]);
            },
            [](const std::vector<double>& in, std::vector<double>& out) {
                VectorOps::vector_erf(in.data(), out.data(), in.size());
            },
            TOLERANCE_ERF_APPROX, /*absolute_only=*/true);

        // vector_cos: linspace over [-4π, +4π] to exercise range reduction.
        // Horner polynomial error is absolute (not relative near zero), so use absolute_only=true.
        std::vector<double> cos_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            cos_data[i] =
                -4.0 * PI + 8.0 * PI * static_cast<double>(i) / static_cast<double>(TEST_SIZE - 1);
        verifyVectorOp(
            cos_data, "VectorCos",
            [](const std::vector<double>& in, std::vector<double>& out) {
                for (size_t i = 0; i < in.size(); ++i)
                    out[i] = std::cos(in[i]);
            },
            [](const std::vector<double>& in, std::vector<double>& out) {
                VectorOps::vector_cos(in.data(), out.data(), in.size());
            },
            TOLERANCE_COS, /*absolute_only=*/true);
    }

    // Benchmark a single vector op: scalar loop vs SIMD call.
    // Produces one VerificationResult row in the summary table.
    template <typename ScalarFunc, typename SIMDFunc>
    void verifyVectorOp(const std::vector<double>& test_data, const std::string& op_name,
                        ScalarFunc scalar_func, SIMDFunc simd_func, double tolerance,
                        bool absolute_only = false) {
        std::vector<double> scalar_results(test_data.size());
        std::vector<double> simd_results(test_data.size());

        // Warm up
        scalar_func(test_data, scalar_results);
        simd_func(test_data, simd_results);

        // Time scalar
        auto scalar_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < TEST_ITERATIONS; ++iter)
            scalar_func(test_data, scalar_results);
        auto scalar_end = std::chrono::high_resolution_clock::now();
        auto scalar_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(scalar_end - scalar_start)
                .count() /
            TEST_ITERATIONS;

        // Time SIMD
        auto simd_start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < TEST_ITERATIONS; ++iter)
            simd_func(test_data, simd_results);
        auto simd_end = std::chrono::high_resolution_clock::now();
        auto simd_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(simd_end - simd_start).count() /
            TEST_ITERATIONS;

        VerificationResult result;
        result.distribution_name = op_name;  // reuse field for op name
        result.operation_name = "---";
        result.test_size = test_data.size();
        result.scalar_time_ns = static_cast<double>(scalar_time);
        result.simd_time_ns = static_cast<double>(simd_time);
        result.speedup_ratio = static_cast<double>(scalar_time) / static_cast<double>(simd_time);
        result.simd_level_used = active_simd_level_;

        // Analyze correctness with caller-supplied tolerance.
        // When absolute_only=false (VectorExp, VectorLog), track relative error
        // in max_difference: absolute diff is meaningless when outputs span many
        // orders of magnitude (e.g. exp(500)~5e+217 makes abs diff ~1e+201 even
        // for a sub-ULP SIMD result). Pass/fail already uses relative tolerance;
        // the stored metric should match.
        result.max_difference = 0.0;
        result.avg_difference = 0.0;
        result.failed_comparisons = 0;
        double sum_diff = 0.0;
        size_t valid = 0;
        std::ostringstream err;
        for (size_t i = 0; i < scalar_results.size(); ++i) {
            double s = scalar_results[i], v = simd_results[i];
            if (std::isnan(s) && std::isnan(v))
                continue;
            if (std::isinf(s) && std::isinf(v) && (s > 0) == (v > 0))
                continue;
            double diff = std::abs(s - v);
            if (std::isfinite(diff)) {
                // Use relative error for non-absolute-only ops; guard against
                // near-zero denominators (denormals, underflowed exp results).
                double display_diff =
                    (!absolute_only && std::abs(s) > 1e-300) ? diff / std::abs(s) : diff;
                result.max_difference = std::max(result.max_difference, display_diff);
                sum_diff += display_diff;
                valid++;
            }
            bool bad = absolute_only ? (std::isfinite(diff) && diff > tolerance)
                                     : ((std::abs(s) > tolerance) ? (diff / std::abs(s) > tolerance)
                                                                  : (diff > tolerance));
            if (bad) {
                result.failed_comparisons++;
                if (result.failed_comparisons <= 3)
                    err << "[" << i << "] scalar=" << s << " simd=" << v << " diff=" << diff
                        << "; ";
            }
        }
        result.avg_difference = (valid > 0) ? sum_diff / static_cast<double>(valid) : 0.0;
        result.correctness_passed = (result.failed_comparisons == 0);
        result.error_details = err.str();
        if (result.failed_comparisons > 3)
            result.error_details +=
                "... (+" + std::to_string(result.failed_comparisons - 3) + " more)";

        results_.push_back(result);

        std::cout << "  " << op_name << ": ";
        if (result.correctness_passed)
            std::cout << "\u2713 PASS";
        else
            std::cout << "\u2717 FAIL";
        // Label distinguishes relative (VectorExp/Log) from absolute (VectorErf/Cos).
        const char* diff_label = absolute_only ? "max_abs" : "max_rel";
        std::cout << " (" << diff_label << "=" << std::scientific << std::setprecision(2)
                  << result.max_difference << ", speedup=" << std::fixed << std::setprecision(1)
                  << result.speedup_ratio << "x)\n";
        if (!result.correctness_passed && !result.error_details.empty())
            std::cout << "    Error: " << result.error_details << "\n";
    }

    void testVonMisesDistribution() {
        stats::detail::detail::subsectionHeader("VonMises Distribution SIMD Verification");
        // kappa=2: unimodal, exercises vector_cos in both LogPDF and PDF batch paths.
        // Uses a local RNG (seed VERIFICATION_SEED+1) so that calling this after testEdgeCases()
        // does not alter rng_ or the test data of any previously-run test.
        auto dist = stats::VonMisesDistribution::create(0.0, 2.0).value;

        std::mt19937 local_rng(VERIFICATION_SEED + 1);
        std::uniform_real_distribution<double> angle_dist(-PI, PI);
        std::vector<double> test_data(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i)
            test_data[i] = angle_dist(local_rng);

        verifyDistributionOperations(dist, test_data, "VonMises");
    }

    void testUniformDistribution() {
        stats::detail::detail::subsectionHeader("Uniform Distribution SIMD Verification");
        auto dist = stats::UniformDistribution::create(0.0, 1.0).value;

        // Test data around the distribution range
        auto test_data = generateTestData(-0.5, 1.5, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Uniform");
    }

    void testGaussianDistribution() {
        stats::detail::detail::subsectionHeader("Gaussian Distribution SIMD Verification");
        auto dist = stats::GaussianDistribution::create(0.0, 1.0).value;

        // Test data with wider range for Gaussian
        auto test_data = generateTestData(-5.0, 5.0, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Gaussian");
    }

    void testExponentialDistribution() {
        stats::detail::detail::subsectionHeader("Exponential Distribution SIMD Verification");
        auto dist = stats::ExponentialDistribution::create(1.0).value;

        // Test data for exponential (positive values)
        auto test_data = generateTestData(0.0, 10.0, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Exponential");
    }

    void testDiscreteDistribution() {
        stats::detail::detail::subsectionHeader("Discrete Distribution SIMD Verification");
        auto dist = stats::DiscreteDistribution::create(0, 10).value;

        // Test data with integer and near-integer values
        auto test_data = generateIntegerTestData(-2, 12, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Discrete");
    }

    void testPoissonDistribution() {
        stats::detail::detail::subsectionHeader("Poisson Distribution SIMD Verification");
        auto dist = stats::PoissonDistribution::create(3.0).value;

        // Test data with non-negative integer and near-integer values
        auto test_data = generateIntegerTestData(0, 15, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Poisson");
    }

    void testGammaDistribution() {
        stats::detail::detail::subsectionHeader("Gamma Distribution SIMD Verification");
        auto dist = stats::GammaDistribution::create(2.0, 1.0).value;

        // Test data for gamma (positive values)
        auto test_data = generateTestData(0.0, 20.0, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Gamma");
    }

    void testStudentTDistribution() {
        stats::detail::detail::subsectionHeader("StudentT Distribution SIMD Verification");
        // nu=3: finite variance (3), good SIMD test — full real-line domain, no fixup needed
        auto dist = stats::StudentTDistribution::create(3.0).value;

        // Full real line: test data spans negative and positive values
        auto test_data = generateTestData(-10.0, 10.0, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "StudentT");
    }

    void testBetaDistribution() {
        stats::detail::detail::subsectionHeader("Beta Distribution SIMD Verification");
        // alpha=2, beta=3: unimodal, interior-heavy, exercises the two-log SIMD pipeline
        auto dist = stats::BetaDistribution::create(2.0, 3.0).value;

        // Strictly interior (0.01, 0.99): avoids boundary fixup path for a clean SIMD benchmark
        auto test_data = generateTestData(0.01, 0.99, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "Beta");
    }

    void testChiSquaredDistribution() {
        stats::detail::detail::subsectionHeader("ChiSquared Distribution SIMD Verification");
        // k=2: analytically tractable (Exp(1/2)), good SIMD test case
        auto dist = stats::ChiSquaredDistribution::create(2.0).value;

        // Positive values only; chi-squared support is (0, +inf)
        auto test_data = generateTestData(0.0, 20.0, TEST_SIZE);

        verifyDistributionOperations(dist, test_data, "ChiSquared");
    }

    template <typename Distribution>
    void verifyDistributionOperations(const Distribution& dist,
                                      const std::vector<double>& test_data,
                                      const std::string& dist_name) {
        // Test PDF operation
        verifyOperation(
            dist, test_data, "PDF", dist_name,
            [](const auto& d, const auto& data, auto& output) {
                // Scalar version - element by element
                for (size_t i = 0; i < data.size(); ++i) {
                    output[i] = d.getProbability(data[i]);
                }
            },
            [](const auto& d, const auto& data, auto& output) {
                // SIMD version: force-vectorized span batch call (TOOL-2)
                stats::detail::PerformanceHint hint;
                hint.strategy =
                    stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
                std::span<const double> input_span(data);
                std::span<double> output_span(output);
                d.getProbability(input_span, output_span, hint);
            });

        // Test LogPDF operation
        verifyOperation(
            dist, test_data, "LogPDF", dist_name,
            [](const auto& d, const auto& data, auto& output) {
                // Scalar version
                for (size_t i = 0; i < data.size(); ++i) {
                    output[i] = d.getLogProbability(data[i]);
                }
            },
            [](const auto& d, const auto& data, auto& output) {
                // SIMD version: force-vectorized span batch call (TOOL-2)
                stats::detail::PerformanceHint hint;
                hint.strategy =
                    stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
                std::span<const double> input_span(data);
                std::span<double> output_span(output);
                d.getLogProbability(input_span, output_span, hint);
            });

        // Test CDF operation
        verifyOperation(
            dist, test_data, "CDF", dist_name,
            [](const auto& d, const auto& data, auto& output) {
                // Scalar version
                for (size_t i = 0; i < data.size(); ++i) {
                    output[i] = d.getCumulativeProbability(data[i]);
                }
            },
            [](const auto& d, const auto& data, auto& output) {
                // SIMD version: force-vectorized span batch call (TOOL-2)
                stats::detail::PerformanceHint hint;
                hint.strategy =
                    stats::detail::PerformanceHint::PreferredStrategy::FORCE_VECTORIZED;
                std::span<const double> input_span(data);
                std::span<double> output_span(output);
                d.getCumulativeProbability(input_span, output_span, hint);
            });
    }

    template <typename Distribution, typename ScalarFunc, typename SIMDFunc>
    void verifyOperation(const Distribution& dist, const std::vector<double>& test_data,
                         const std::string& operation_name, const std::string& dist_name,
                         ScalarFunc scalar_func, SIMDFunc simd_func) {
        std::vector<double> scalar_results(test_data.size());
        std::vector<double> simd_results(test_data.size());

        // Warm up both versions
        scalar_func(dist, test_data, scalar_results);
        simd_func(dist, test_data, simd_results);

        // Time scalar execution
        auto scalar_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_ITERATIONS; ++i) {
            scalar_func(dist, test_data, scalar_results);
        }
        auto scalar_end = std::chrono::high_resolution_clock::now();
        auto scalar_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(scalar_end - scalar_start)
                .count() /
            TEST_ITERATIONS;

        // Time SIMD execution
        auto simd_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_ITERATIONS; ++i) {
            simd_func(dist, test_data, simd_results);
        }
        auto simd_end = std::chrono::high_resolution_clock::now();
        auto simd_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(simd_end - simd_start).count() /
            TEST_ITERATIONS;

        // Analyze differences
        VerificationResult result;
        result.distribution_name = dist_name;
        result.operation_name = operation_name;
        result.test_size = test_data.size();
        result.scalar_time_ns = static_cast<double>(scalar_time);
        result.simd_time_ns = static_cast<double>(simd_time);
        result.speedup_ratio = static_cast<double>(scalar_time) / static_cast<double>(simd_time);
        result.simd_level_used = active_simd_level_;

        analyzeDifferences(scalar_results, simd_results, result);

        results_.push_back(result);

        // Print immediate results
        std::cout << "  " << operation_name << ": ";
        if (result.correctness_passed) {
            std::cout << "✓ PASS";
        } else {
            std::cout << "✗ FAIL";
        }
        std::cout << " (max_diff=" << std::scientific << std::setprecision(2)
                  << result.max_difference;
        std::cout << ", speedup=" << std::fixed << std::setprecision(1) << result.speedup_ratio
                  << "x)\n";

        if (!result.correctness_passed && !result.error_details.empty()) {
            std::cout << "    Error: " << result.error_details << "\n";
        }
    }

    void analyzeDifferences(const std::vector<double>& scalar_results,
                            const std::vector<double>& simd_results, VerificationResult& result) {
        result.max_difference = 0.0;
        result.avg_difference = 0.0;
        result.failed_comparisons = 0;
        double sum_differences = 0.0;
        size_t valid_comparisons = 0;  // Track number of non-special-case comparisons
        std::ostringstream error_stream;

        // Per-distribution tolerance table keyed by exact (dist_name, op_name) pairs.
        // Uses == rather than find() so a name change does not silently fall through
        // to TOLERANCE_NORMAL — any rename is immediately visible as a table miss.
        struct ToleranceEntry { const char* dist; const char* op; double tol; };
        static const ToleranceEntry kToleranceTable[] = {
            // Gaussian CDF routes through vector_erf (musl polynomial, max ~2.2e-16)
            { "Gaussian", "CDF",    TOLERANCE_ERF_APPROX },
            // Beta LogPDF: two vector_log calls accumulate ~1 ULP rounding vs scalar
            { "Beta",     "LogPDF", TOLERANCE_RELAXED     },
            // VonMises PDF/LogPDF route through vector_cos (7-term Horner, ~1e-10 floor)
            { "VonMises", "PDF",    TOLERANCE_VONMISES    },
            { "VonMises", "LogPDF", TOLERANCE_VONMISES    },
        };
        double default_tolerance = TOLERANCE_NORMAL;
        for (const auto& e : kToleranceTable) {
            if (result.distribution_name == e.dist && result.operation_name == e.op) {
                default_tolerance = e.tol;
                break;
            }
        }

        for (size_t i = 0; i < scalar_results.size(); ++i) {
            double scalar_val = scalar_results[i];
            double simd_val = simd_results[i];

            // Handle special cases (NaN, infinity) BEFORE computing difference
            if (std::isnan(scalar_val) && std::isnan(simd_val)) {
                // Both NaN - consider equal, skip difference calculation
                continue;
            }
            if (std::isinf(scalar_val) && std::isinf(simd_val)) {
                // Both infinite with same sign - consider equal
                if ((scalar_val > 0) == (simd_val > 0)) {
                    // Skip difference calculation to avoid inf-inf=nan
                    continue;
                }
                // Different signs of infinity - this is an error
                // Fall through to error handling
            }

            // NOW compute difference after handling special cases
            double diff = std::abs(scalar_val - simd_val);

            // Only update statistics if diff is finite
            if (std::isfinite(diff)) {
                result.max_difference = std::max(result.max_difference, diff);
                sum_differences += diff;
                valid_comparisons++;
            }

            // Select tolerance: erf-derived for Gaussian CDF, relaxed for near-zero values
            double tolerance = default_tolerance;
            if (std::abs(scalar_val) < 1e-100 || std::abs(simd_val) < 1e-100) {
                tolerance = TOLERANCE_RELAXED;
            }

            bool is_error = false;
            if (tolerance >= TOLERANCE_ERF_APPROX || tolerance == TOLERANCE_RELAXED ||
                tolerance == TOLERANCE_VONMISES) {
                // Gaussian CDF (erf approx), Beta LogPDF (two-log rounding), and VonMises
                // PDF/LogPDF (vector_cos pipeline) all use absolute tolerance only:
                // relative error is misleading near zero crossings and distribution tails.
                is_error = (diff > tolerance);
            } else if (std::abs(scalar_val) > tolerance) {
                double relative_error = diff / std::abs(scalar_val);
                if (relative_error > tolerance) {
                    is_error = true;
                }
            } else if (diff > tolerance) {
                // For values close to zero, use absolute tolerance
                is_error = true;
            }

            if (is_error) {
                result.failed_comparisons++;
                if (result.failed_comparisons <= 3) {  // Only report first few
                    error_stream << "Index " << i << ": scalar=" << scalar_val
                                 << ", simd=" << simd_val << ", diff=" << diff << "; ";
                }
            }
        }

        // Calculate average only from valid comparisons
        result.avg_difference = (valid_comparisons > 0)
                                    ? sum_differences / static_cast<double>(valid_comparisons)
                                    : 0.0;
        result.correctness_passed = (result.failed_comparisons == 0);
        result.error_details = error_stream.str();

        if (result.failed_comparisons > 3) {
            result.error_details +=
                "... (+" + std::to_string(result.failed_comparisons - 3) + " more)";
        }
    }

    void testEdgeCases() {
        stats::detail::detail::subsectionHeader("Edge Cases Testing");

        std::cout << "Testing distributions with edge case values (NaN, infinity, etc.)\n";

        // Test each distribution with edge case values
        std::vector<std::pair<std::string, std::function<void()>>> edge_tests = {
            {"Uniform",
             [this]() { testDistributionEdgeCases(stats::Uniform(0.0, 1.0), "Uniform"); }},
            {"Gaussian",
             [this]() { testDistributionEdgeCases(stats::Gaussian(0.0, 1.0), "Gaussian"); }},
            {"Exponential",
             [this]() { testDistributionEdgeCases(stats::Exponential(1.0), "Exponential"); }},
            {"Discrete",
             [this]() { testDistributionEdgeCases(stats::Discrete(0, 10), "Discrete"); }},
            {"Poisson", [this]() { testDistributionEdgeCases(stats::Poisson(3.0), "Poisson"); }},
            {"Gamma", [this]() { testDistributionEdgeCases(stats::Gamma(2.0, 1.0), "Gamma"); }},
            {"Beta",
             [this]() {
                 // alpha=2, beta=2: symmetric, avoids alpha=1 or beta=1 boundary ambiguity
                 testDistributionEdgeCases(stats::Beta(2.0, 2.0), "Beta");
             }},
            {"StudentT", [this]() { testDistributionEdgeCases(stats::StudentT(3.0), "StudentT"); }},
            {"ChiSquared", [this]() {
                 // Use k=4 (alpha=2) to avoid the alpha=1 x=0 boundary case
                 // while the batch fixup returns 0. Same reasoning as Gamma using alpha=2.
                 testDistributionEdgeCases(stats::ChiSquared(4.0), "ChiSquared");
             }}};

        for (const auto& test : edge_tests) {
            std::cout << "  Testing " << test.first << " edge cases...\n";
            test.second();
        }
    }

    template <typename Distribution>
    void testDistributionEdgeCases(const Distribution& dist, const std::string& dist_name) {
        // Create test data that includes edge cases plus some normal values
        std::vector<double> edge_test_data = EDGE_CASES;

        // Pad with normal values to reach minimum batch size for SIMD
        auto normal_data = generateTestData(-10.0, 10.0, 100);
        edge_test_data.insert(edge_test_data.end(), normal_data.begin(), normal_data.end());

        // Test operations with edge cases
        verifyDistributionOperations(dist, edge_test_data, dist_name + "_EdgeCases");
    }

    std::vector<double> generateTestData(double min_val, double max_val, size_t count) {
        std::vector<double> data;
        data.reserve(count);

        std::uniform_real_distribution<double> uniform_dist(min_val, max_val);
        std::normal_distribution<double> normal_dist((min_val + max_val) / 2,
                                                     (max_val - min_val) / 6);

        // Mix of uniform and normal distributions for comprehensive testing
        for (size_t i = 0; i < count; ++i) {
            if (i % 3 == 0) {
                data.push_back(uniform_dist(rng_));
            } else {
                double val = normal_dist(rng_);
                data.push_back(std::clamp(val, min_val, max_val));
            }
        }

        return data;
    }

    std::vector<double> generateIntegerTestData(int min_val, int max_val, size_t count) {
        std::vector<double> data;
        data.reserve(count);

        std::uniform_int_distribution<int> int_dist(min_val, max_val);
        std::uniform_real_distribution<double> offset_dist(-0.4, 0.4);

        // Mix of exact integers and near-integers
        for (size_t i = 0; i < count; ++i) {
            int base_val = int_dist(rng_);
            if (i % 4 == 0) {
                // Exact integer
                data.push_back(static_cast<double>(base_val));
            } else {
                // Near integer with small offset
                data.push_back(base_val + offset_dist(rng_));
            }
        }

        return data;
    }

    void analyzeResults() {
        stats::detail::detail::sectionHeader("SIMD Verification Analysis");

        // Summary statistics
        size_t total_tests = results_.size();
        size_t passed_tests = static_cast<size_t>(std::count_if(
            results_.begin(), results_.end(), [](const auto& r) { return r.correctness_passed; }));

        std::cout << "\n=== Summary ===\n";
        std::cout << "SIMD Level Tested: " << active_simd_level_ << "\n";
        std::cout << "Total tests: " << total_tests << "\n";
        std::cout << "Passed: " << passed_tests << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * static_cast<double>(passed_tests) / static_cast<double>(total_tests))
                  << "%)\n";
        std::cout << "Failed: " << (total_tests - passed_tests) << "\n\n";

        // Detailed results table
        // Max Err / Avg Err: absolute for distribution and VectorErf/VectorCos rows;
        // relative (dimensionless) for VectorExp and VectorLog rows.
        stats::detail::detail::ColumnFormatter formatter({18, 10, 8, 12, 12, 10, 8});
        std::cout << formatter.formatRow({"Distribution", "Operation", "Status", "Max Err",
                                          "Avg Err", "Speedup", "Errors"})
                  << "\n";
        std::cout << formatter.getSeparator() << "\n";

        for (const auto& result : results_) {
            std::string status = result.correctness_passed ? "PASS" : "FAIL";
            // Use scientific notation for diff values (fixed format gives "0.00" for tiny values)
            auto format_diff = [](double v) -> std::string {
                if (v < 1e-15)
                    return "~0";
                std::ostringstream oss;
                oss << std::scientific << std::setprecision(1) << v;
                return oss.str();
            };
            std::string max_diff_str = format_diff(result.max_difference);
            std::string avg_diff_str = format_diff(result.avg_difference);
            std::string speedup_str =
                stats::detail::detail::formatDouble(result.speedup_ratio, 1) + "x";
            std::string errors_str = std::to_string(result.failed_comparisons);

            // Truncate long distribution names for better table formatting
            std::string dist_name = result.distribution_name;
            if (dist_name.length() > 17) {
                dist_name = dist_name.substr(0, 14) + "...";
            }

            std::cout << formatter.formatRow({dist_name, result.operation_name, status,
                                              max_diff_str, avg_diff_str, speedup_str, errors_str})
                      << "\n";
        }

        // Failed tests details
        auto failed_tests = std::count_if(results_.begin(), results_.end(),
                                          [](const auto& r) { return !r.correctness_passed; });

        if (failed_tests > 0) {
            stats::detail::detail::subsectionHeader("Failed Tests Details");
            for (const auto& result : results_) {
                if (!result.correctness_passed) {
                    std::cout << "❌ " << result.distribution_name << "::" << result.operation_name
                              << "\n";
                    std::cout << "   Max difference: " << std::scientific << result.max_difference
                              << "\n";
                    std::cout << "   Failed comparisons: " << result.failed_comparisons << "/"
                              << result.test_size << "\n";
                    if (!result.error_details.empty()) {
                        std::cout << "   Sample errors: " << result.error_details << "\n";
                    }
                    std::cout << "\n";
                }
            }
        }

        // Performance analysis
        // Partition results into two populations that must not be aggregated together:
        //   - Distribution tests: end-to-end batch pipeline speedup (PDF/LogPDF/CDF)
        //   - Primitive vector ops: raw intrinsic-level speedup vs std::exp/log/erf/cos
        // Mixing them produces a meaningless composite that changes with test composition.
        stats::detail::detail::subsectionHeader("Performance Analysis");

        // Per-operation-type geometric mean speedups.
        // Partitioning by PDF/LogPDF/CDF reveals meaningful structure:
        //   LogPDF: log-space paths, exp-dominated, typically highest speedup
        //   PDF:    exp + normalization, moderate speedup
        //   CDF:    erf/incomplete special functions, lowest speedup
        // Geometric mean is correct for ratios: insensitive to which tests
        // dominate wall-clock time and symmetric across orders of magnitude.
        struct OpStats {
            double log_sum = 0.0;
            size_t count = 0;
            double min_sp = std::numeric_limits<double>::max();
            double max_sp = 0.0;
        };
        std::map<std::string, OpStats> op_stats;  // keyed by operation_name

        for (const auto& result : results_) {
            // Primitive ops (operation_name == "---") are excluded from the distribution
            // suite geometric mean — they are reported individually below.
            if (result.operation_name != "---" && result.speedup_ratio > 0.0) {
                auto& s = op_stats[result.operation_name];
                s.log_sum += std::log(result.speedup_ratio);
                ++s.count;
                s.min_sp = std::min(s.min_sp, result.speedup_ratio);
                s.max_sp = std::max(s.max_sp, result.speedup_ratio);
            }
        }

        std::cout << "Distribution suite speedup geometric means (" << active_simd_level_ << "):\n";
        // Print in canonical order: PDF, LogPDF, CDF
        double overall_log_sum = 0.0;
        size_t overall_count = 0;
        for (const std::string& op :
             {std::string("PDF"), std::string("LogPDF"), std::string("CDF")}) {
            auto it = op_stats.find(op);
            if (it == op_stats.end() || it->second.count == 0)
                continue;
            const auto& s = it->second;
            double gm = std::exp(s.log_sum / static_cast<double>(s.count));
            std::cout << "  " << std::left << std::setw(8) << op << std::fixed
                      << std::setprecision(1) << gm << "x"
                      << "  (range " << s.min_sp << "x\u2013" << s.max_sp << "x,  n=" << s.count
                      << ")\n";
            overall_log_sum += s.log_sum;
            overall_count += s.count;
        }
        double overall_geomean =
            (overall_count > 0) ? std::exp(overall_log_sum / static_cast<double>(overall_count))
                                : 0.0;

        // Primitive vector op speedups — reported individually, not aggregated
        std::cout << "\nPrimitive vector op speedups:\n";
        for (const auto& result : results_) {
            if (result.operation_name == "---") {
                std::cout << "  " << result.distribution_name << ": " << std::fixed
                          << std::setprecision(1) << result.speedup_ratio << "x";
                if (!result.correctness_passed)
                    std::cout << "  \u26a0\ufe0f ACCURACY FAIL";
                std::cout << "\n";
            }
        }

        // Architecture-specific performance expectation — distribution suite only
        double expected_min_speedup = 1.5;  // Conservative baseline
        if (active_simd_level_ == "AVX-512") {
            expected_min_speedup = 2.0;  // Phase 4 target; rises after native transcendentals
        } else if (active_simd_level_ == "AVX2" || active_simd_level_ == "AVX") {
            expected_min_speedup = 2.5;
        } else if (active_simd_level_ == "SSE2" || active_simd_level_ == "NEON") {
            expected_min_speedup = 1.5;  // Phase 3 target; rises after native transcendentals
        }

        // Recommendations
        stats::detail::detail::subsectionHeader("Recommendations");
        if (passed_tests == total_tests) {
            std::cout << "\u2705 All SIMD operations are producing correct results.\n";
            std::cout << "\u2705 " << active_simd_level_
                      << " optimizations are working correctly.\n";
        } else {
            std::cout << "\u26a0\ufe0f  Some SIMD operations are not producing identical results "
                         "to scalar "
                         "versions.\n";
            std::cout
                << "\u26a0\ufe0f  Review failed tests above and consider adjusting tolerance or "
                   "implementation.\n";
        }

        if (overall_geomean < expected_min_speedup) {
            std::cout << "\u26a0\ufe0f  Distribution suite geometric mean speedup (" << std::fixed
                      << std::setprecision(2) << overall_geomean
                      << "x) is below expected (>=" << expected_min_speedup << "x).\n";
            std::cout
                << "   Consider profiling individual operations for optimization opportunities.\n";
        } else {
            std::cout << "\u2705 " << active_simd_level_
                      << " distribution suite performance is meeting expectations.\n";
        }

        std::cout << "\nNote: Small numerical differences are expected due to floating-point "
                     "precision.\n";
        std::cout << "      SIMD operations may use different rounding or instruction sequences.\n";
        std::cout
            << "      Focus on fixing tests with large relative errors or systematic issues.\n";
    }
};

int main() {
    return stats::detail::detail::runTool("SIMD Verification", []() {
        SIMDVerifier verifier;
        verifier.runVerification();
    });
}
