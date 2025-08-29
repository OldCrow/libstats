/**
 * @file test_simd_policy.cpp
 * @brief Comprehensive test suite for platform/simd_policy.h centralized SIMD abstraction
 *
 * Tests the SIMDPolicy class for centralized SIMD decision making
 * with command-line options for selective testing:
 * --all/-a           Test all SIMD policy components (default)
 * --policy/-p        Test core policy decisions (shouldUseSIMD)
 * --detection/-d     Test SIMD level detection
 * --thresholds/-t    Test threshold calculations
 * --alignment/-A     Test alignment requirements
 * --capabilities/-c  Test capability reporting
 * --consistency/-C   Test cross-platform consistency
 * --stress/-s        Run stress tests with various data sizes
 * --help/-h          Show this help
 */

#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd_policy.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace stats::arch::simd;

//==============================================================================
// COMMAND-LINE ARGUMENT PARSING
//==============================================================================

struct TestOptions {
    bool test_all = false;
    bool test_policy = false;
    bool test_detection = false;
    bool test_thresholds = false;
    bool test_alignment = false;
    bool test_capabilities = false;
    bool test_consistency = false;
    bool test_stress = false;
    bool show_help = false;
};

void print_help() {
    std::cout << "Usage: test_simd_policy [options]\n\n";
    std::cout << "Test platform/simd_policy.h centralized SIMD abstraction:\n\n";
    std::cout << "Options:\n";
    std::cout << "  --all/-a           Test all SIMD policy components (default)\n";
    std::cout << "  --policy/-p        Test core policy decisions (shouldUseSIMD)\n";
    std::cout << "  --detection/-d     Test SIMD level detection\n";
    std::cout << "  --thresholds/-t    Test threshold calculations\n";
    std::cout << "  --alignment/-A     Test alignment requirements\n";
    std::cout << "  --capabilities/-c  Test capability reporting\n";
    std::cout << "  --consistency/-C   Test cross-platform consistency\n";
    std::cout << "  --stress/-s        Run stress tests with various data sizes\n";
    std::cout << "  --help/-h          Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  test_simd_policy                    # Test all components\n";
    std::cout << "  test_simd_policy --policy --detection # Test core policy and detection\n";
    std::cout << "  test_simd_policy -t -A              # Test thresholds and alignment\n";
    std::cout << "  test_simd_policy --stress            # Run stress tests\n";
}

TestOptions parse_arguments(int argc, char* argv[]) {
    TestOptions options;
    bool any_specific_test = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--all" || arg == "-a") {
            options.test_all = true;
        } else if (arg == "--policy" || arg == "-p") {
            options.test_policy = true;
            any_specific_test = true;
        } else if (arg == "--detection" || arg == "-d") {
            options.test_detection = true;
            any_specific_test = true;
        } else if (arg == "--thresholds" || arg == "-t") {
            options.test_thresholds = true;
            any_specific_test = true;
        } else if (arg == "--alignment" || arg == "-A") {
            options.test_alignment = true;
            any_specific_test = true;
        } else if (arg == "--capabilities" || arg == "-c") {
            options.test_capabilities = true;
            any_specific_test = true;
        } else if (arg == "--consistency" || arg == "-C") {
            options.test_consistency = true;
            any_specific_test = true;
        } else if (arg == "--stress" || arg == "-s") {
            options.test_stress = true;
            any_specific_test = true;
        } else if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return options;
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            std::cerr << "Use --help/-h for usage information.\n";
            options.show_help = true;
            return options;
        }
    }

    // If no specific tests requested, default to all
    if (!any_specific_test && !options.test_all) {
        options.test_all = true;
    }

    return options;
}

//==============================================================================
// HELPER FUNCTIONS
//==============================================================================

std::string level_to_string(SIMDPolicy::Level level) {
    switch (level) {
        case SIMDPolicy::Level::None:
            return "None";
        case SIMDPolicy::Level::NEON:
            return "NEON";
        case SIMDPolicy::Level::SSE2:
            return "SSE2";
        case SIMDPolicy::Level::AVX:
            return "AVX";
        case SIMDPolicy::Level::AVX2:
            return "AVX2";
        case SIMDPolicy::Level::AVX512:
            return "AVX512";
        default:
            return "Unknown";
    }
}

bool is_power_of_2(std::size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

//==============================================================================
// TEST FUNCTIONS
//==============================================================================

void test_core_policy_decisions() {
    std::cout << "[Core Policy Decision Tests]\n";

    // Test 1: Basic shouldUseSIMD functionality
    {
        // Very small data should not use SIMD
        bool small_decision = SIMDPolicy::shouldUseSIMD(1);
        [[maybe_unused]] bool tiny_decision = SIMDPolicy::shouldUseSIMD(2);

        // Large data should use SIMD (if available)
        bool large_decision = SIMDPolicy::shouldUseSIMD(10000);
        bool huge_decision = SIMDPolicy::shouldUseSIMD(1000000);

        // Get the threshold to understand the cutoff
        std::size_t threshold = SIMDPolicy::getMinThreshold();

        // Test threshold behavior
        [[maybe_unused]] bool below_threshold = SIMDPolicy::shouldUseSIMD(threshold - 1);
        [[maybe_unused]] bool at_threshold = SIMDPolicy::shouldUseSIMD(threshold);
        [[maybe_unused]] bool above_threshold = SIMDPolicy::shouldUseSIMD(threshold + 1);

        // Validate logical consistency
        if (threshold > 1) {
            assert(!below_threshold ||
                   threshold <= 1);  // Below threshold should be false (unless threshold is 1)
        }
        assert(at_threshold ||
               SIMDPolicy::getBestLevel() ==
                   SIMDPolicy::Level::None);  // At threshold should be true unless no SIMD
        assert(above_threshold ||
               SIMDPolicy::getBestLevel() ==
                   SIMDPolicy::Level::None);  // Above threshold should be true unless no SIMD

        std::cout << "   ✓ Basic shouldUseSIMD decisions passed\n";
        std::cout << "     Threshold: " << threshold << " elements\n";
        std::cout << "     Small(1): " << (small_decision ? "SIMD" : "Scalar") << "\n";
        std::cout << "     Large(10K): " << (large_decision ? "SIMD" : "Scalar") << "\n";
        std::cout << "     Huge(1M): " << (huge_decision ? "SIMD" : "Scalar") << "\n";
    }

    // Test 2: Policy consistency across different data sizes
    {
        std::vector<std::size_t> test_sizes = {1,   2,   4,   8,    16,   32,   64,
                                               128, 256, 512, 1024, 4096, 16384};
        std::vector<bool> decisions;

        for (auto size : test_sizes) {
            decisions.push_back(SIMDPolicy::shouldUseSIMD(size));
        }

        // Check that decisions are monotonic (once true, should stay true for larger sizes)
        bool found_simd = false;
        for (size_t i = 0; i < decisions.size(); ++i) {
            if (decisions[i]) {
                found_simd = true;
            } else if (found_simd) {
                // Once we decide to use SIMD, we shouldn't go back to scalar for larger sizes
                assert(false && "SIMD policy should be monotonic");
            }
        }

        std::cout << "   ✓ Policy monotonicity test passed\n";
    }

    std::cout << "\n";
}

void test_simd_level_detection() {
    std::cout << "[SIMD Level Detection Tests]\n";

    // Test 1: Basic level detection
    {
        [[maybe_unused]] SIMDPolicy::Level detected_level = SIMDPolicy::getBestLevel();
        std::string level_str = SIMDPolicy::getLevelString();

        // Level should be valid
        assert(detected_level >= SIMDPolicy::Level::None);
        assert(detected_level <= SIMDPolicy::Level::AVX512);

        // Level string should not be empty and should match
        assert(!level_str.empty());
        assert(level_str == level_to_string(detected_level));

        std::cout << "   ✓ Basic level detection passed\n";
        std::cout << "     Detected level: " << level_str << "\n";
    }

    // Test 2: Level consistency with runtime CPU detection
    {
        SIMDPolicy::Level policy_level = SIMDPolicy::getBestLevel();

        // Compare with direct CPU detection
        bool cpu_supports_avx512 = stats::arch::supports_avx512();
        bool cpu_supports_avx2 = stats::arch::supports_avx2();
        bool cpu_supports_avx = stats::arch::supports_avx();
        bool cpu_supports_sse2 = stats::arch::supports_sse2();
        bool cpu_supports_neon = stats::arch::supports_neon();

        // Validate consistency
        if (policy_level == SIMDPolicy::Level::AVX512) {
            assert(cpu_supports_avx512);
        } else if (policy_level == SIMDPolicy::Level::AVX2) {
            assert(cpu_supports_avx2);
            assert(!cpu_supports_avx512);  // Should choose highest available
        } else if (policy_level == SIMDPolicy::Level::AVX) {
            assert(cpu_supports_avx);
            assert(!cpu_supports_avx2 && !cpu_supports_avx512);
        } else if (policy_level == SIMDPolicy::Level::SSE2) {
            assert(cpu_supports_sse2);
            assert(!cpu_supports_avx && !cpu_supports_avx2 && !cpu_supports_avx512);
        } else if (policy_level == SIMDPolicy::Level::NEON) {
            assert(cpu_supports_neon);
            // NEON systems typically don't have x86 SIMD
            assert(!cpu_supports_sse2 && !cpu_supports_avx);
        }

        std::cout << "   ✓ Level consistency with CPU detection passed\n";
        std::cout << "     CPU AVX512: " << (cpu_supports_avx512 ? "YES" : "NO") << "\n";
        std::cout << "     CPU AVX2: " << (cpu_supports_avx2 ? "YES" : "NO") << "\n";
        std::cout << "     CPU AVX: " << (cpu_supports_avx ? "YES" : "NO") << "\n";
        std::cout << "     CPU SSE2: " << (cpu_supports_sse2 ? "YES" : "NO") << "\n";
        std::cout << "     CPU NEON: " << (cpu_supports_neon ? "YES" : "NO") << "\n";
    }

    // Test 3: Cache refresh functionality
    {
        [[maybe_unused]] SIMDPolicy::Level level_before = SIMDPolicy::getBestLevel();
        SIMDPolicy::refreshCache();
        [[maybe_unused]] SIMDPolicy::Level level_after = SIMDPolicy::getBestLevel();

        // Should be the same after refresh (CPU didn't change)
        assert(level_before == level_after);

        std::cout << "   ✓ Cache refresh functionality passed\n";
    }

    std::cout << "\n";
}

void test_threshold_calculations() {
    std::cout << "[Threshold Calculation Tests]\n";

    // Test 1: Basic threshold properties
    {
        std::size_t threshold = SIMDPolicy::getMinThreshold();

        // Threshold should be reasonable
        assert(threshold > 0);
        assert(threshold <= 10000);  // Should be reasonable, not excessive

        // Should be consistent with policy decisions
        if (threshold > 1) {
            assert(!SIMDPolicy::shouldUseSIMD(threshold - 1));
        }
        assert(SIMDPolicy::shouldUseSIMD(threshold) ||
               SIMDPolicy::getBestLevel() == SIMDPolicy::Level::None);
        assert(SIMDPolicy::shouldUseSIMD(threshold + 100) ||
               SIMDPolicy::getBestLevel() == SIMDPolicy::Level::None);

        std::cout << "   ✓ Basic threshold properties passed\n";
        std::cout << "     Minimum threshold: " << threshold << " elements\n";
    }

    // Test 2: Threshold relationship to SIMD level
    {
        SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
        std::size_t threshold = SIMDPolicy::getMinThreshold();

        // Generally, more powerful SIMD should have similar or lower thresholds
        // (since setup cost is amortized better)
        if (level == SIMDPolicy::Level::None) {
            // No SIMD means threshold is irrelevant, but should still be valid
            assert(threshold > 0);
        } else {
            // SIMD available, threshold should be reasonable for the level
            if (level >= SIMDPolicy::Level::AVX2) {
                assert(threshold <= 64);  // AVX2+ should have low threshold
            } else if (level >= SIMDPolicy::Level::AVX || level == SIMDPolicy::Level::NEON) {
                assert(threshold <= 128);  // AVX/NEON should have moderate threshold
            } else {
                assert(threshold <= 256);  // SSE2 might have higher threshold
            }
        }

        std::cout << "   ✓ Threshold-to-level relationship passed\n";
        std::cout << "     Level: " << level_to_string(level) << ", Threshold: " << threshold
                  << "\n";
    }

    std::cout << "\n";
}

void test_alignment_requirements() {
    std::cout << "[Alignment Requirement Tests]\n";

    // Test 1: Basic alignment properties
    {
        std::size_t alignment = SIMDPolicy::getOptimalAlignment();

        // Alignment should be a power of 2
        assert(is_power_of_2(alignment));

        // Alignment should be reasonable (between 4 and 64 bytes typically)
        assert(alignment >= 4);
        assert(alignment <= 128);

        std::cout << "   ✓ Basic alignment properties passed\n";
        std::cout << "     Optimal alignment: " << alignment << " bytes\n";
    }

    // Test 2: Alignment relationship to SIMD level
    {
        SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
        std::size_t alignment = SIMDPolicy::getOptimalAlignment();

        // Validate alignment matches expected values for each SIMD level
        switch (level) {
            case SIMDPolicy::Level::AVX512:
                assert(alignment >= 64);  // AVX-512 typically wants 64-byte alignment
                break;
            case SIMDPolicy::Level::AVX2:
            case SIMDPolicy::Level::AVX:
                assert(alignment >= 32);  // AVX typically wants 32-byte alignment
                break;
            case SIMDPolicy::Level::SSE2:
            case SIMDPolicy::Level::NEON:
                assert(alignment >= 16);  // SSE2/NEON typically want 16-byte alignment
                break;
            case SIMDPolicy::Level::None:
                assert(alignment >= 4);  // Even scalar should have some alignment
                break;
        }

        std::cout << "   ✓ Alignment-to-level relationship passed\n";
        std::cout << "     Level: " << level_to_string(level) << ", Alignment: " << alignment
                  << " bytes\n";
    }

    std::cout << "\n";
}

void test_capability_reporting() {
    std::cout << "[Capability Reporting Tests]\n";

    // Test 1: Basic capability string
    {
        std::string capability_str = SIMDPolicy::getCapabilityString();

        // Should not be empty
        assert(!capability_str.empty());

        // Should contain some expected information
        SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
        std::string level_str = level_to_string(level);
        assert(capability_str.find(level_str) != std::string::npos);

        std::cout << "   ✓ Basic capability reporting passed\n";
        std::cout << "     Capabilities: " << capability_str << "\n";
    }

    // Test 2: Block size reporting
    {
        std::size_t block_size = SIMDPolicy::getOptimalBlockSize();

        // Block size should be reasonable
        assert(block_size > 0);
        assert(block_size <= 32);  // Should not be excessive

        // Block size should match SIMD level
        SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
        switch (level) {
            case SIMDPolicy::Level::AVX512:
                assert(block_size >= 8);  // 512-bit / 64-bit = 8 doubles
                break;
            case SIMDPolicy::Level::AVX2:
            case SIMDPolicy::Level::AVX:
                assert(block_size >= 4);  // 256-bit / 64-bit = 4 doubles
                break;
            case SIMDPolicy::Level::SSE2:
            case SIMDPolicy::Level::NEON:
                assert(block_size >= 2);  // 128-bit / 64-bit = 2 doubles
                break;
            case SIMDPolicy::Level::None:
                assert(block_size >= 1);  // Scalar processing
                break;
        }

        std::cout << "   ✓ Block size reporting passed\n";
        std::cout << "     Optimal block size: " << block_size << " elements\n";
    }

    std::cout << "\n";
}

void test_cross_platform_consistency() {
    std::cout << "[Cross-Platform Consistency Tests]\n";

    // Test 1: API consistency
    {
        // All API calls should work without throwing
        [[maybe_unused]] bool decision = SIMDPolicy::shouldUseSIMD(100);
        [[maybe_unused]] SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
        [[maybe_unused]] std::size_t threshold = SIMDPolicy::getMinThreshold();
        [[maybe_unused]] std::size_t block_size = SIMDPolicy::getOptimalBlockSize();
        [[maybe_unused]] std::size_t alignment = SIMDPolicy::getOptimalAlignment();
        std::string level_str = SIMDPolicy::getLevelString();
        std::string capability_str = SIMDPolicy::getCapabilityString();

        // Basic sanity checks
        assert(threshold > 0);
        assert(block_size > 0);
        assert(alignment > 0);
        assert(!level_str.empty());
        assert(!capability_str.empty());

        std::cout << "   ✓ API consistency test passed\n";
        std::cout << "     All APIs callable and return valid values\n";
    }

    // Test 2: Logical consistency
    {
        // Test that the policy is internally consistent
        std::size_t threshold = SIMDPolicy::getMinThreshold();
        SIMDPolicy::Level level = SIMDPolicy::getBestLevel();

        // If we have SIMD capability, decisions should make sense
        if (level != SIMDPolicy::Level::None) {
            // Should use SIMD for data above threshold
            assert(SIMDPolicy::shouldUseSIMD(threshold * 2));
            assert(SIMDPolicy::shouldUseSIMD(threshold * 10));

            // Should generally not use SIMD for very small data (unless threshold is 1)
            if (threshold > 1) {
                assert(!SIMDPolicy::shouldUseSIMD(1));
            }
        } else {
            // No SIMD capability - should never recommend SIMD
            assert(!SIMDPolicy::shouldUseSIMD(1));
            assert(!SIMDPolicy::shouldUseSIMD(1000));
            assert(!SIMDPolicy::shouldUseSIMD(1000000));
        }

        std::cout << "   ✓ Logical consistency test passed\n";
    }

    std::cout << "\n";
}

void test_stress_conditions() {
    std::cout << "[Stress Test Conditions]\n";

    // Test 1: Extreme data sizes
    {
        std::vector<std::size_t> extreme_sizes = {
            0, 1, 2, SIZE_MAX / 2, SIZE_MAX - 1
            // Note: Avoiding SIZE_MAX itself to prevent overflow in calculations
        };

        for (auto size : extreme_sizes) {
            // Should not crash or throw
            try {
                bool decision = SIMDPolicy::shouldUseSIMD(size);
                (void)decision;  // Suppress unused variable warning
            } catch (...) {
                assert(false && "shouldUseSIMD should not throw for any size_t value");
            }
        }

        std::cout << "   ✓ Extreme data size stress test passed\n";
    }

    // Test 2: Rapid repeated calls
    {
        const int num_calls = 10000;
        [[maybe_unused]] SIMDPolicy::Level initial_level = SIMDPolicy::getBestLevel();

        for (int i = 0; i < num_calls; ++i) {
            // Mix of different operations
            [[maybe_unused]] bool decision = SIMDPolicy::shouldUseSIMD(i % 1000 + 1);
            [[maybe_unused]] SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
            [[maybe_unused]] std::size_t threshold = SIMDPolicy::getMinThreshold();

            // Results should be consistent
            assert(level == initial_level);
            assert(threshold == SIMDPolicy::getMinThreshold());  // Should be cached/consistent

            (void)decision;  // Suppress unused variable warning
        }

        std::cout << "   ✓ Rapid repeated calls stress test passed\n";
    }

    // Test 3: Cache refresh stress
    {
        const int num_refreshes = 100;
        [[maybe_unused]] SIMDPolicy::Level initial_level = SIMDPolicy::getBestLevel();

        for (int i = 0; i < num_refreshes; ++i) {
            SIMDPolicy::refreshCache();
            [[maybe_unused]] SIMDPolicy::Level level = SIMDPolicy::getBestLevel();
            assert(level == initial_level);  // Should remain consistent
        }

        std::cout << "   ✓ Cache refresh stress test passed\n";
    }

    std::cout << "\n";
}

//==============================================================================
// MAIN FUNCTION
//==============================================================================

int main(int argc, char* argv[]) {
    TestOptions options = parse_arguments(argc, argv);

    if (options.show_help) {
        print_help();
        return 0;
    }

    std::cout << "Testing platform/simd_policy.h centralized SIMD abstraction...\n\n";

    int tests_run = 0;
    int tests_passed = 0;

    try {
        // Run selected tests
        if (options.test_all || options.test_policy) {
            test_core_policy_decisions();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_detection) {
            test_simd_level_detection();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_thresholds) {
            test_threshold_calculations();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_alignment) {
            test_alignment_requirements();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_capabilities) {
            test_capability_reporting();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_consistency) {
            test_cross_platform_consistency();
            tests_run++;
            tests_passed++;
        }

        if (options.test_all || options.test_stress) {
            test_stress_conditions();
            tests_run++;
            tests_passed++;
        }

        // Print summary
        std::cout << "=== Test Summary ===\n";
        std::cout << "Tests run: " << tests_run << "\n";
        std::cout << "Tests passed: " << tests_passed << "\n";

        if (tests_passed == tests_run && tests_run > 0) {
            std::cout << "✓ All SIMD policy tests passed!\n";
            return 0;
        } else if (tests_run == 0) {
            std::cout << "No tests were run. Use --help for usage information.\n";
            return 1;
        } else {
            std::cout << "✗ Some tests failed!\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << "\n";
        return 1;
    }
}
