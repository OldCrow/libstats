#pragma once

/**
 * @file tests/include/basic_test_runner.h
 * @brief Shared template for Tests 6 and 8 in distribution basic test files.
 *
 * Tests 1–5 and 7 contain significant distribution-specific content
 * (parameter names, known values, MLE formulas) and remain per-distribution.
 *
 * Test 6 (Auto-dispatch Batch Operations) is ~95% generic: only the
 * representative input values and random-data range vary.
 *
 * Test 8 (Error Handling) is 100% generic in structure: only the
 * invalid-parameter scenarios vary.
 *
 * Usage in a per-distribution file:
 * @code
 *   #include "include/basic_test_runner.h"
 *   // ...
 *   BasicDistConfig cfg{
 *       "Gaussian",
 *       {-2.5, -1.2, 0.3, 1.8, 2.1},
 *       -3.0, 3.0,
 *       {{"negative sigma", [] { return GaussianDistribution::create(0.0,-1.0).isError(); }}}
 *   };
 *   auto dist = GaussianDistribution::create(0.0, 1.0).unwrap();
 *   runBatchTests(cfg, dist);   // Test 6
 *   // ... Test 7 (per-distribution) ...
 *   runErrorTests(cfg);         // Test 8
 * @endcode
 */

#include "fixtures.h"

#include <chrono>
#include <cstddef>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace stats {
namespace tests {

//==============================================================================
// Configuration
//==============================================================================

/**
 * @brief Per-distribution configuration for the shared basic test runner.
 */
struct BasicDistConfig {
    /// Distribution display name (e.g. "Gaussian").
    std::string name;

    /// Small set of representative domain-appropriate input values (5–7 elements).
    /// Used in the small-batch batch-vs-scalar correctness check.
    std::vector<double> small_test_values;

    /// Lower bound for the uniform random data range used in the large-batch section.
    double large_data_lo = 0.0;

    /// Upper bound for the uniform random data range used in the large-batch section.
    double large_data_hi = 1.0;

    /// Tolerance for PDF and LogPDF batch-vs-scalar comparison.
    double pdf_tolerance = 1e-12;

    /// Tolerance for CDF batch-vs-scalar comparison (may be relaxed for discrete CDFs).
    double cdf_tolerance = 1e-12;

    /**
     * @brief Describes one invalid-parameter scenario for Test 8.
     *
     * @p produces_error must return true if the distribution correctly
     * rejects the invalid input (i.e. create() or trySet() returns isError()).
     */
    struct InvalidScenario {
        std::string description;
        std::function<bool()> produces_error;
    };

    /// List of invalid-parameter scenarios exercised in Test 8.
    std::vector<InvalidScenario> invalid_scenarios = {};
};

//==============================================================================
// Test 6: Auto-dispatch Batch Operations
//==============================================================================

/**
 * @brief Run Test 6 (Auto-dispatch Batch Operations) for any distribution.
 *
 * Verifies:
 *   1. Small-batch PDF, LogPDF, CDF via span API match scalar calls element-wise.
 *   2. Large-batch auto-dispatch PDF matches a scalar loop on the same inputs.
 *
 * Throws std::runtime_error on any correctness failure so the test process
 * exits with a non-zero code.
 *
 * @tparam Dist Distribution type.
 * @param cfg   Per-distribution configuration.
 * @param dist  Constructed distribution instance to test (by const-ref to avoid mutation).
 */
template <typename Dist>
void runBatchTests(const BasicDistConfig& cfg, const Dist& dist) {
    using namespace stats::tests::fixtures;
    using std::cout;
    using std::endl;

    BasicTestFormatter::printTestStart(6, "Auto-dispatch Batch Operations");
    cout << "Batch PDF/LogPDF/CDF via auto-dispatch; verify against scalar calls." << endl;

    // ---- Small batch correctness ----
    const auto& xs = cfg.small_test_values;
    const std::size_t n_small = xs.size();

    std::vector<double> pdf_b(n_small), lpdf_b(n_small), cdf_b(n_small);
    dist.getProbability(std::span<const double>(xs), std::span<double>(pdf_b));
    dist.getLogProbability(std::span<const double>(xs), std::span<double>(lpdf_b));
    dist.getCumulativeProbability(std::span<const double>(xs), std::span<double>(cdf_b));

    bool small_ok = true;
    for (std::size_t i = 0; i < n_small; ++i) {
        if (std::abs(pdf_b[i] - dist.getProbability(xs[i])) > cfg.pdf_tolerance ||
            std::abs(lpdf_b[i] - dist.getLogProbability(xs[i])) > cfg.pdf_tolerance ||
            std::abs(cdf_b[i] - dist.getCumulativeProbability(xs[i])) > cfg.cdf_tolerance) {
            small_ok = false;
            break;
        }
    }
    cout << "Small batch matches scalar (n=" << n_small << "): " << (small_ok ? "PASS" : "FAIL")
         << endl;

    // ---- Large batch correctness ----
    constexpr std::size_t N = 5000;
    std::vector<double> large_in(N), large_out(N), large_scl(N);
    {
        std::mt19937 gen(42);
        std::uniform_real_distribution<double> rng(cfg.large_data_lo, cfg.large_data_hi);
        for (auto& v : large_in)
            v = rng(gen);
    }
    dist.getProbability(std::span<const double>(large_in), std::span<double>(large_out));
    for (std::size_t i = 0; i < N; ++i)
        large_scl[i] = dist.getProbability(large_in[i]);

    // Measure auto-dispatch throughput while we have the timing infrastructure handy
    const auto t0 = std::chrono::high_resolution_clock::now();
    dist.getProbability(std::span<const double>(large_in), std::span<double>(large_out));
    const auto t1 = std::chrono::high_resolution_clock::now();
    const std::int64_t batch_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    bool large_ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        if (std::abs(large_out[i] - large_scl[i]) > cfg.pdf_tolerance) {
            large_ok = false;
            break;
        }
    }
    cout << "Large batch matches scalar (n=" << N << "): " << (large_ok ? "PASS" : "FAIL") << endl;
    cout << "Auto-dispatch PDF throughput (n=" << N << "): " << batch_us << " μs" << endl;

    if (!small_ok || !large_ok)
        throw std::runtime_error("Batch correctness test failed: " + cfg.name);

    BasicTestFormatter::printTestSuccess("Batch operation tests passed");
    BasicTestFormatter::printNewline();
}

//==============================================================================
// Test 8: Error Handling
//==============================================================================

/**
 * @brief Run Test 8 (Error Handling) for any distribution.
 *
 * Iterates over all @p cfg.invalid_scenarios, calls @p produces_error(), and
 * reports the result. Throws std::runtime_error if any scenario fails to
 * produce an error.
 *
 * @param cfg Per-distribution configuration.
 */
inline void runErrorTests(const BasicDistConfig& cfg) {
    using namespace stats::tests::fixtures;
    using std::cout;
    using std::endl;

    BasicTestFormatter::printTestStart(8, "Error Handling");

    bool all_ok = true;
    for (const auto& s : cfg.invalid_scenarios) {
        const bool ok = s.produces_error();
        if (ok)
            BasicTestFormatter::printTestSuccess(s.description + " rejected");
        else {
            BasicTestFormatter::printTestError(s.description + " was NOT rejected — test failed");
            all_ok = false;
        }
    }

    if (!all_ok)
        throw std::runtime_error("Error handling test failed: " + cfg.name);

    BasicTestFormatter::printTestSuccess("All error handling tests passed");
    BasicTestFormatter::printNewline();
}

}  // namespace tests
}  // namespace stats
