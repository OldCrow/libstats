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
#include <cmath>
#include <algorithm>
#include <limits>

#include "../include/libstats.h"
#include "../include/distributions/uniform.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/exponential.h"
#include "../include/distributions/discrete.h"
#include "../include/distributions/poisson.h"
#include "../include/distributions/gamma.h"
#include "../include/platform/simd.h"
#include "tool_utils.h"

using namespace libstats;
using namespace libstats::tools;

namespace {
    constexpr int VERIFICATION_SEED = 12345;
    constexpr size_t TEST_SIZE = 1024;        // Size for correctness tests
    [[maybe_unused]] constexpr size_t LARGE_TEST_SIZE = 65536; // Size for performance tests - reserved for future benchmarking features
    constexpr int TEST_ITERATIONS = 5;
    constexpr double TOLERANCE_NORMAL = 1e-14;   // Normal numerical precision
    constexpr double TOLERANCE_RELAXED = 1e-12; // Relaxed for complex operations
    
    // Edge case test values that are architecture-independent
    const std::vector<double> EDGE_CASES = {
        0.0, -0.0,
        std::numeric_limits<double>::min(),
        std::numeric_limits<double>::max(),
        std::numeric_limits<double>::denorm_min(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        1.0, -1.0,
        1e-100, 1e100, -1e-100, -1e100,
        0.5, -0.5, 2.0, -2.0,
        std::numeric_limits<double>::epsilon(),
        1.0 + std::numeric_limits<double>::epsilon(),
        1.0 - std::numeric_limits<double>::epsilon()
    };
}

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
        active_simd_level_ = libstats::simd::VectorOps::get_active_simd_level();
    }
    
    void runVerification() {
        system_info::displayToolHeader("SIMD Verification Tool", 
                                       "Validates SIMD operations correctness and performance on actual system architecture");
        
        // Display system SIMD capabilities
        displaySystemSIMDInfo();
        
        // Test all distributions with different operations
        testUniformDistribution();
        testGaussianDistribution();
        testExponentialDistribution();
        testDiscreteDistribution();
        testPoissonDistribution();
        testGammaDistribution();
        
        // Test edge cases
        testEdgeCases();
        
        // Analyze and report results
        analyzeResults();
    }

private:
    void displaySystemSIMDInfo() {
        display::subsectionHeader("System SIMD Capabilities");
        
        const auto& features = libstats::cpu::get_features();
        std::cout << "Active SIMD Level: " << active_simd_level_ << "\n";
        std::cout << "Architecture: " << system_info::getActiveArchitecture() << "\n";
        
        // Display available SIMD features based on architecture
        table::ColumnFormatter formatter({20, 10, 30});
        std::cout << formatter.formatRow({"SIMD Feature", "Available", "Description"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        #ifdef __x86_64__
        std::cout << formatter.formatRow({"AVX-512", features.avx512f ? "Yes" : "No", "512-bit vectors"}) << "\n";
        std::cout << formatter.formatRow({"AVX2", features.avx2 ? "Yes" : "No", "256-bit integer vectors"}) << "\n";
        std::cout << formatter.formatRow({"AVX", features.avx ? "Yes" : "No", "256-bit floating-point vectors"}) << "\n";
        std::cout << formatter.formatRow({"SSE2", features.sse2 ? "Yes" : "No", "128-bit vectors"}) << "\n";
        std::cout << formatter.formatRow({"FMA", features.fma ? "Yes" : "No", "Fused multiply-add"}) << "\n";
        #elif defined(__aarch64__) || defined(__arm__)
        std::cout << formatter.formatRow({"NEON", features.neon ? "Yes" : "No", "ARM SIMD instructions"}) << "\n";
        std::cout << formatter.formatRow({"FMA", features.fma ? "Yes" : "No", "Fused multiply-add"}) << "\n";
        #endif
        
        std::cout << "\nOptimal SIMD block size: " << libstats::constants::platform::get_optimal_simd_block_size() << " elements\n";
        std::cout << "Memory alignment: " << libstats::constants::platform::get_optimal_alignment() << " bytes\n\n";
    }
    
    void testUniformDistribution() {
        display::subsectionHeader("Uniform Distribution SIMD Verification");
        auto dist = libstats::UniformDistribution::create(0.0, 1.0).value;
        
        // Test data around the distribution range
        auto test_data = generateTestData(-0.5, 1.5, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Uniform");
    }
    
    void testGaussianDistribution() {
        display::subsectionHeader("Gaussian Distribution SIMD Verification");
        auto dist = libstats::GaussianDistribution::create(0.0, 1.0).value;
        
        // Test data with wider range for Gaussian
        auto test_data = generateTestData(-5.0, 5.0, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Gaussian");
    }
    
    void testExponentialDistribution() {
        display::subsectionHeader("Exponential Distribution SIMD Verification");
        auto dist = libstats::ExponentialDistribution::create(1.0).value;
        
        // Test data for exponential (positive values)
        auto test_data = generateTestData(0.0, 10.0, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Exponential");
    }
    
    void testDiscreteDistribution() {
        display::subsectionHeader("Discrete Distribution SIMD Verification");
        auto dist = libstats::DiscreteDistribution::create(0, 10).value;
        
        // Test data with integer and near-integer values
        auto test_data = generateIntegerTestData(-2, 12, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Discrete");
    }
    
    void testPoissonDistribution() {
        display::subsectionHeader("Poisson Distribution SIMD Verification");
        auto dist = libstats::PoissonDistribution::create(3.0).value;
        
        // Test data with non-negative integer and near-integer values
        auto test_data = generateIntegerTestData(0, 15, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Poisson");
    }
    
    void testGammaDistribution() {
        display::subsectionHeader("Gamma Distribution SIMD Verification");
        auto dist = libstats::GammaDistribution::create(2.0, 1.0).value;
        
        // Test data for gamma (positive values)
        auto test_data = generateTestData(0.0, 20.0, TEST_SIZE);
        
        verifyDistributionOperations(dist, test_data, "Gamma");
    }
    
    template<typename Distribution>
    void verifyDistributionOperations(const Distribution& dist, 
                                      const std::vector<double>& test_data,
                                      const std::string& dist_name) {
        // Test PDF operation
        verifyOperation(dist, test_data, "PDF", dist_name, 
                       [](const auto& d, const auto& data, auto& output) {
                           // Scalar version - element by element
                           for (size_t i = 0; i < data.size(); ++i) {
                               output[i] = d.getProbability(data[i]);
                           }
                       },
                       [](const auto& d, const auto& data, auto& output) {
                           // SIMD version - using explicit strategy for SIMD verification
                           std::span<const double> input_span(data);
                           std::span<double> output_span(output);
                           d.getProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::SIMD_BATCH);
                       });
                       
        // Test LogPDF operation
        verifyOperation(dist, test_data, "LogPDF", dist_name,
                       [](const auto& d, const auto& data, auto& output) {
                           // Scalar version
                           for (size_t i = 0; i < data.size(); ++i) {
                               output[i] = d.getLogProbability(data[i]);
                           }
                       },
                       [](const auto& d, const auto& data, auto& output) {
                           // SIMD version - using explicit strategy for SIMD verification
                           std::span<const double> input_span(data);
                           std::span<double> output_span(output);
                           d.getLogProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::SIMD_BATCH);
                       });
                       
        // Test CDF operation
        verifyOperation(dist, test_data, "CDF", dist_name,
                       [](const auto& d, const auto& data, auto& output) {
                           // Scalar version
                           for (size_t i = 0; i < data.size(); ++i) {
                               output[i] = d.getCumulativeProbability(data[i]);
                           }
                       },
                       [](const auto& d, const auto& data, auto& output) {
                           // SIMD version - using explicit strategy for SIMD verification
                           std::span<const double> input_span(data);
                           std::span<double> output_span(output);
                           d.getCumulativeProbabilityWithStrategy(input_span, output_span, libstats::performance::Strategy::SIMD_BATCH);
                       });
    }
    
    template<typename Distribution, typename ScalarFunc, typename SIMDFunc>
    void verifyOperation(const Distribution& dist,
                        const std::vector<double>& test_data, 
                        const std::string& operation_name,
                        const std::string& dist_name,
                        ScalarFunc scalar_func,
                        SIMDFunc simd_func) {
        
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
        auto scalar_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            scalar_end - scalar_start).count() / TEST_ITERATIONS;
            
        // Time SIMD execution
        auto simd_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < TEST_ITERATIONS; ++i) {
            simd_func(dist, test_data, simd_results);
        }
        auto simd_end = std::chrono::high_resolution_clock::now();
        auto simd_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
            simd_end - simd_start).count() / TEST_ITERATIONS;
        
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
        std::cout << " (max_diff=" << std::scientific << std::setprecision(2) << result.max_difference;
        std::cout << ", speedup=" << std::fixed << std::setprecision(1) << result.speedup_ratio << "x)\n";
        
        if (!result.correctness_passed && !result.error_details.empty()) {
            std::cout << "    Error: " << result.error_details << "\n";
        }
    }
    
    void analyzeDifferences(const std::vector<double>& scalar_results,
                           const std::vector<double>& simd_results,
                           VerificationResult& result) {
        
        result.max_difference = 0.0;
        result.avg_difference = 0.0;
        result.failed_comparisons = 0;
        double sum_differences = 0.0;
        std::ostringstream error_stream;
        
        for (size_t i = 0; i < scalar_results.size(); ++i) {
            double scalar_val = scalar_results[i];
            double simd_val = simd_results[i];
            double diff = std::abs(scalar_val - simd_val);
            
            // Handle special cases (NaN, infinity)
            if (std::isnan(scalar_val) && std::isnan(simd_val)) {
                // Both NaN - consider equal
                continue;
            }
            if (std::isinf(scalar_val) && std::isinf(simd_val)) {
                // Both infinite with same sign - consider equal
                if ((scalar_val > 0) == (simd_val > 0)) {
                    continue;
                }
            }
            
            // Calculate relative error for non-zero values
            double tolerance = TOLERANCE_NORMAL;
            
            // Use relaxed tolerance for complex operations or very small numbers
            if (std::abs(scalar_val) < 1e-100 || std::abs(simd_val) < 1e-100) {
                tolerance = TOLERANCE_RELAXED;
            }
            
            bool is_error = false;
            if (std::abs(scalar_val) > tolerance) {
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
                if (result.failed_comparisons <= 3) { // Only report first few
                    error_stream << "Index " << i << ": scalar=" << scalar_val 
                               << ", simd=" << simd_val << ", diff=" << diff << "; ";
                }
            }
            
            result.max_difference = std::max(result.max_difference, diff);
            sum_differences += diff;
        }
        
        result.avg_difference = sum_differences / static_cast<double>(scalar_results.size());
        result.correctness_passed = (result.failed_comparisons == 0);
        result.error_details = error_stream.str();
        
        if (result.failed_comparisons > 3) {
            result.error_details += "... (+" + std::to_string(result.failed_comparisons - 3) + " more)";
        }
    }
    
    void testEdgeCases() {
        display::subsectionHeader("Edge Cases Testing");
        
        std::cout << "Testing distributions with edge case values (NaN, infinity, etc.)\n";
        
        // Test each distribution with edge case values
        std::vector<std::pair<std::string, std::function<void()>>> edge_tests = {
            {"Uniform", [this]() { testDistributionEdgeCases(libstats::Uniform(0.0, 1.0), "Uniform"); }},
            {"Gaussian", [this]() { testDistributionEdgeCases(libstats::Gaussian(0.0, 1.0), "Gaussian"); }},
            {"Exponential", [this]() { testDistributionEdgeCases(libstats::Exponential(1.0), "Exponential"); }},
            {"Discrete", [this]() { testDistributionEdgeCases(libstats::Discrete(0, 10), "Discrete"); }},
            {"Poisson", [this]() { testDistributionEdgeCases(libstats::Poisson(3.0), "Poisson"); }},
            {"Gamma", [this]() { testDistributionEdgeCases(libstats::Gamma(2.0, 1.0), "Gamma"); }}
        };
        
        for (const auto& test : edge_tests) {
            std::cout << "  Testing " << test.first << " edge cases...\n";
            test.second();
        }
    }
    
    template<typename Distribution>
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
        display::sectionHeader("SIMD Verification Analysis");
        
        // Summary statistics
        size_t total_tests = results_.size();
        size_t passed_tests = static_cast<size_t>(std::count_if(results_.begin(), results_.end(),
                                          [](const auto& r) { return r.correctness_passed; }));
        
        std::cout << "\n=== Summary ===\n";
        std::cout << "SIMD Level Tested: " << active_simd_level_ << "\n";
        std::cout << "Total tests: " << total_tests << "\n";
        std::cout << "Passed: " << passed_tests << " (" 
                  << std::fixed << std::setprecision(1) 
                  << (100.0 * static_cast<double>(passed_tests) / static_cast<double>(total_tests)) << "%)\n";
        std::cout << "Failed: " << (total_tests - passed_tests) << "\n\n";
        
        // Detailed results table
        table::ColumnFormatter formatter({18, 10, 8, 12, 12, 10, 8});
        std::cout << formatter.formatRow({"Distribution", "Operation", "Status", "Max Diff", "Avg Diff", "Speedup", "Errors"}) << "\n";
        std::cout << formatter.getSeparator() << "\n";
        
        for (const auto& result : results_) {
            std::string status = result.correctness_passed ? "PASS" : "FAIL";
            std::string max_diff_str = (result.max_difference < 1e-15) ? "~0" : 
                                      format::formatDouble(result.max_difference, 2);
            std::string avg_diff_str = (result.avg_difference < 1e-15) ? "~0" : 
                                      format::formatDouble(result.avg_difference, 2);
            std::string speedup_str = format::formatDouble(result.speedup_ratio, 1) + "x";
            std::string errors_str = std::to_string(result.failed_comparisons);
            
            // Truncate long distribution names for better table formatting
            std::string dist_name = result.distribution_name;
            if (dist_name.length() > 17) {
                dist_name = dist_name.substr(0, 14) + "...";
            }
            
            std::cout << formatter.formatRow({
                dist_name,
                result.operation_name, 
                status,
                max_diff_str,
                avg_diff_str,
                speedup_str,
                errors_str
            }) << "\n";
        }
        
        // Failed tests details
        auto failed_tests = std::count_if(results_.begin(), results_.end(),
                                        [](const auto& r) { return !r.correctness_passed; });
        
        if (failed_tests > 0) {
            display::subsectionHeader("Failed Tests Details");
            for (const auto& result : results_) {
                if (!result.correctness_passed) {
                    std::cout << "❌ " << result.distribution_name << "::" << result.operation_name << "\n";
                    std::cout << "   Max difference: " << std::scientific << result.max_difference << "\n";
                    std::cout << "   Failed comparisons: " << result.failed_comparisons << "/" << result.test_size << "\n";
                    if (!result.error_details.empty()) {
                        std::cout << "   Sample errors: " << result.error_details << "\n";
                    }
                    std::cout << "\n";
                }
            }
        }
        
        // Performance analysis
        display::subsectionHeader("Performance Analysis");
        double total_scalar_time = 0, total_simd_time = 0;
        double min_speedup = std::numeric_limits<double>::max();
        double max_speedup = 0;
        
        for (const auto& result : results_) {
            total_scalar_time += result.scalar_time_ns;
            total_simd_time += result.simd_time_ns;
            min_speedup = std::min(min_speedup, result.speedup_ratio);
            max_speedup = std::max(max_speedup, result.speedup_ratio);
        }
        
        double overall_speedup = total_scalar_time / total_simd_time;
        
        std::cout << "Overall " << active_simd_level_ << " speedup: " << std::fixed << std::setprecision(2) << overall_speedup << "x\n";
        std::cout << "Speedup range: " << std::setprecision(1) << min_speedup << "x to " << max_speedup << "x\n";
        
        // Architecture-specific performance expectations
        double expected_min_speedup = 1.5; // Conservative baseline
        if (active_simd_level_ == "AVX-512") {
            expected_min_speedup = 4.0;
        } else if (active_simd_level_ == "AVX2" || active_simd_level_ == "AVX") {
            expected_min_speedup = 2.5;
        } else if (active_simd_level_ == "SSE2" || active_simd_level_ == "NEON") {
            expected_min_speedup = 2.0;
        }
        
        // Recommendations
        display::subsectionHeader("Recommendations");
        if (passed_tests == total_tests) {
            std::cout << "✅ All SIMD operations are producing correct results.\n";
            std::cout << "✅ " << active_simd_level_ << " optimizations are working correctly.\n";
        } else {
            std::cout << "⚠️  Some SIMD operations are not producing identical results to scalar versions.\n";
            std::cout << "⚠️  Review failed tests above and consider adjusting tolerance or implementation.\n";
        }
        
        if (overall_speedup < expected_min_speedup) {
            std::cout << "⚠️  Overall " << active_simd_level_ << " speedup (" << overall_speedup 
                      << "x) is below expected performance (>=" << expected_min_speedup << "x).\n";
            std::cout << "   Consider profiling individual operations for optimization opportunities.\n";
        } else {
            std::cout << "✅ " << active_simd_level_ << " performance is meeting expectations.\n";
        }
        
        std::cout << "\nNote: Small numerical differences are expected due to floating-point precision.\n";
        std::cout << "      SIMD operations may use different rounding or instruction sequences.\n";
        std::cout << "      Focus on fixing tests with large relative errors or systematic issues.\n";
    }
};

int main() {
    return tool_utils::runTool("SIMD Verification", []() {
        SIMDVerifier verifier;
        verifier.runVerification();
    });
}
