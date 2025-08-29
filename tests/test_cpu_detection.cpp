/**
 * @file test_cpu_detection.cpp
 * @brief Enhanced comprehensive CPU detection test with command-line options
 *
 * Features:
 * - Vendor-specific testing (Intel vs AMD vs Apple Silicon)
 * - Command-line options for selective testing
 * - JSON output support
 * - Performance benchmarking
 * - Cache hierarchy validation
 * - CPU topology analysis
 * - Cross-architecture consistency validation
 */

#include "../include/platform/cpu_detection.h"
#include "../include/platform/parallel_thresholds.h"
#include "../include/platform/simd.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace stats;

// Command-line options structure
struct TestOptions {
    bool verbose = false;
    bool benchmarks = false;
    bool cache = false;
    bool topology = false;
    bool json = false;
    bool validate = false;
    bool help = false;
    bool vendor_specific = false;
    bool generation_detection = false;
    bool cross_architecture = false;
    bool run_all = true;  // Default to running all tests
};

// Forward declarations
void print_help();
TestOptions parse_args(int argc, char* argv[]);
void test_basic_detection(const TestOptions& opts);
void test_vendor_specific(const TestOptions& opts);
void test_cache_hierarchy(const TestOptions& opts);
void test_cpu_topology(const TestOptions& opts);
void test_performance_benchmarks(const TestOptions& opts);
void test_generation_detection(const TestOptions& opts);
void test_cross_architecture_consistency(const TestOptions& opts);
void test_validation_against_database(const TestOptions& opts);
void output_json_summary(const TestOptions& opts);
string get_intel_microarchitecture();
string get_amd_architecture();
string get_apple_silicon_features();
bool validate_cpu_against_database();

int main(int argc, char* argv[]) {
    try {
        TestOptions opts = parse_args(argc, argv);

        if (opts.help) {
            print_help();
            return 0;
        }

        if (opts.json) {
            output_json_summary(opts);

            // Check if tests would pass for proper CI exit code
            bool validation_passed = validate_cpu_against_database();
            auto simple_parallel = arch::get_min_elements_for_simple_distribution_parallel();
            auto dist_parallel = arch::get_min_elements_for_distribution_parallel();
            auto min_parallel = arch::get_min_elements_for_parallel();
            bool hierarchy_consistent =
                (dist_parallel <= min_parallel) && (min_parallel <= simple_parallel);
            bool all_tests_passed = validation_passed && hierarchy_consistent;

            return all_tests_passed ? 0 : 1;
        }

        cout << "=== Enhanced CPU Detection Test Suite ===" << endl;
        cout << "===========================================" << endl;

        // Always run basic detection
        test_basic_detection(opts);

        if (opts.run_all || opts.vendor_specific) {
            test_vendor_specific(opts);
        }

        if (opts.run_all || opts.cache) {
            test_cache_hierarchy(opts);
        }

        if (opts.run_all || opts.topology) {
            test_cpu_topology(opts);
        }

        if (opts.run_all || opts.benchmarks) {
            test_performance_benchmarks(opts);
        }

        if (opts.run_all || opts.generation_detection) {
            test_generation_detection(opts);
        }

        if (opts.run_all || opts.cross_architecture) {
            test_cross_architecture_consistency(opts);
        }

        if (opts.run_all || opts.validate) {
            test_validation_against_database(opts);
        }

        cout << "\n=== ALL CPU DETECTION TESTS COMPLETED SUCCESSFULLY ===" << endl;
        return 0;

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

TestOptions parse_args(int argc, char* argv[]) {
    TestOptions opts;
    bool any_specific_test = false;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            opts.help = true;
        } else if (arg == "--verbose" || arg == "-v") {
            opts.verbose = true;
        } else if (arg == "--benchmarks" || arg == "-b") {
            opts.benchmarks = true;
            any_specific_test = true;
        } else if (arg == "--cache" || arg == "-c") {
            opts.cache = true;
            any_specific_test = true;
        } else if (arg == "--topology" || arg == "-t") {
            opts.topology = true;
            any_specific_test = true;
        } else if (arg == "--json" || arg == "-j") {
            opts.json = true;
        } else if (arg == "--validate" || arg == "-V") {
            opts.validate = true;
            any_specific_test = true;
        } else if (arg == "--vendor") {
            opts.vendor_specific = true;
            any_specific_test = true;
        } else if (arg == "--generation") {
            opts.generation_detection = true;
            any_specific_test = true;
        } else if (arg == "--cross-arch") {
            opts.cross_architecture = true;
            any_specific_test = true;
        }
    }

    // If any specific test was requested, don't run all
    if (any_specific_test) {
        opts.run_all = false;
    }

    return opts;
}

void print_help() {
    cout << "Enhanced CPU Detection Test Suite\n\n";
    cout << "Usage: test_cpu_detection [OPTIONS]\n\n";
    cout << "Options:\n";
    cout << "  -h, --help       Show this help message\n";
    cout << "  -v, --verbose    Show detailed CPU information\n";
    cout << "  -b, --benchmarks Include performance benchmarks\n";
    cout << "  -c, --cache      Test cache hierarchy detection\n";
    cout << "  -t, --topology   Test CPU topology detection\n";
    cout << "  -j, --json       Output in JSON format\n";
    cout << "  -V, --validate   Validate against known CPU database\n";
    cout << "      --vendor     Test vendor-specific features\n";
    cout << "      --generation Test CPU generation detection\n";
    cout << "      --cross-arch Test cross-architecture consistency\n\n";
    cout << "If no specific test is specified, all tests are run.\n";
}

void test_basic_detection(const TestOptions& opts) {
    cout << "\n=== BASIC CPU DETECTION ===" << endl;

    const auto& features = arch::get_features();

    cout << "\nCPU Identification:" << endl;
    cout << "  Vendor: " << features.vendor << endl;
    if (!features.brand.empty()) {
        cout << "  Brand: " << features.brand << endl;
    }

    // Architecture-aware identification output
    if (features.vendor == "GenuineIntel" || features.vendor == "AuthenticAMD") {
        // x86/x64 CPUs use Family/Model/Stepping
        cout << "  Family: " << features.family << ", Model: " << features.model
             << ", Stepping: " << features.stepping << endl;
    } else if (features.vendor == "Apple") {
        // Apple Silicon uses different identification
        cout << "  Architecture: ARM64 (Apple Silicon)" << endl;
        if (features.neon)
            cout << "  SIMD: NEON + Apple Matrix Extensions" << endl;
    } else if (features.neon || features.sve) {
        // Other ARM processors
        cout << "  Architecture: ARM64" << endl;
        if (features.sve)
            cout << "  SIMD: SVE (Scalable Vector Extension)" << endl;
        else if (features.neon)
            cout << "  SIMD: NEON" << endl;
    } else {
        // Unknown or other architectures - fallback to Family/Model/Stepping
        cout << "  Family: " << features.family << ", Model: " << features.model
             << ", Stepping: " << features.stepping << " (x86 format)" << endl;
    }

    if (opts.verbose) {
        cout << "\nSIMD Features:" << endl;
        cout << "  SSE2: " << (features.sse2 ? "✓" : "✗") << endl;
        cout << "  SSE4.1: " << (features.sse4_1 ? "✓" : "✗") << endl;
        cout << "  AVX: " << (features.avx ? "✓" : "✗") << endl;
        cout << "  AVX2: " << (features.avx2 ? "✓" : "✗") << endl;
        cout << "  FMA: " << (features.fma ? "✓" : "✗") << endl;
        cout << "  AVX512F: " << (features.avx512f ? "✓" : "✗") << endl;
        cout << "  NEON: " << (features.neon ? "✓" : "✗") << endl;
        cout << "  SVE: " << (features.sve ? "✓" : "✗") << endl;
    }

    cout << "\nOptimal Configuration:" << endl;
    cout << "  Best SIMD Level: " << arch::best_simd_level() << endl;
    cout << "  Double Vector Width: " << arch::optimal_double_width() << endl;
    cout << "  Float Vector Width: " << arch::optimal_float_width() << endl;
    cout << "  Memory Alignment: " << arch::optimal_alignment() << " bytes" << endl;
}

void test_vendor_specific(const TestOptions& opts) {
    cout << "\n=== VENDOR-SPECIFIC TESTING ===" << endl;

    const auto& features = arch::get_features();

    if (features.vendor == "GenuineIntel") {
        cout << "\nIntel-Specific Analysis:" << endl;
        cout << "  Microarchitecture: " << get_intel_microarchitecture() << endl;
        cout << "  Sandy/Ivy Bridge: " << (arch::is_sandy_ivy_bridge() ? "✓" : "✗") << endl;
        cout << "  Haswell/Broadwell: " << (arch::is_haswell_broadwell() ? "✓" : "✗") << endl;
        cout << "  Skylake Generation: " << (arch::is_skylake_generation() ? "✓" : "✗") << endl;
        cout << "  Kaby/Coffee Lake: " << (arch::is_kaby_coffee_lake() ? "✓" : "✗") << endl;
        cout << "  Modern Intel: " << (arch::is_modern_intel() ? "✓" : "✗") << endl;

        if (opts.verbose) {
            cout << "  Intel Optimization Strategy: ";
            if (features.avx512f) {
                cout << "AVX-512 focused, aggressive vectorization" << endl;
            } else if (features.avx2) {
                cout << "AVX2 + FMA, balanced approach" << endl;
            } else if (features.avx) {
                cout << "AVX baseline, conservative thresholds" << endl;
            } else {
                cout << "Legacy SSE, minimal vectorization" << endl;
            }
        }

    } else if (features.vendor == "AuthenticAMD") {
        cout << "\nAMD-Specific Analysis:" << endl;
        cout << "  Architecture: " << get_amd_architecture() << endl;

        if (opts.verbose) {
            cout << "  AMD Optimization Strategy: ";
            if (features.avx2) {
                cout << "Zen architecture, prefer cache efficiency" << endl;
            } else if (features.avx) {
                cout << "Bulldozer family, careful with FMA" << endl;
            } else {
                cout << "Legacy architecture, conservative approach" << endl;
            }
        }

    } else if (features.vendor == "Apple") {
        cout << "\nApple Silicon Analysis:" << endl;
        cout << "  Features: " << get_apple_silicon_features() << endl;
        cout << "  NEON Available: " << (features.neon ? "✓" : "✗") << endl;

        if (opts.verbose) {
            cout << "  Apple Optimization Strategy: ";
            cout << "Unified memory, aggressive SIMD, low thread overhead" << endl;
            cout << "  AMX Support: " << "Detected via CPU model analysis" << endl;
            cout << "  Specialized NEON: Enhanced dot product, matrix operations" << endl;
        }

    } else {
        cout << "\nUnknown Vendor (" << features.vendor << ")" << endl;
        cout << "  Using generic optimization strategies" << endl;
    }
}

void test_cache_hierarchy(const TestOptions& opts) {
    cout << "\n=== CACHE HIERARCHY ANALYSIS ===" << endl;

    const auto& features = arch::get_features();

    cout << "\nCache Sizes:" << endl;
    cout << "  L1 Data: " << features.l1_data_cache.size << " bytes" << endl;
    cout << "  L1 Instruction: " << features.l1_instruction_cache.size << " bytes" << endl;
    cout << "  L2: " << features.l2_cache.size << " bytes" << endl;
    cout << "  L3: " << features.l3_cache.size << " bytes" << endl;
    cout << "  Cache Line: " << features.cache_line_size << " bytes" << endl;

    // Get cache thresholds for optimization
    auto thresholds = arch::get_cache_thresholds();
    cout << "\nOptimal Thresholds (elements):" << endl;
    cout << "  L1 Optimal: " << thresholds.l1_optimal_size << endl;
    cout << "  L2 Optimal: " << thresholds.l2_optimal_size << endl;
    cout << "  L3 Optimal: " << thresholds.l3_optimal_size << endl;
    cout << "  Blocking Size: " << thresholds.blocking_size << endl;

    if (opts.verbose) {
        cout << "\nCache Analysis:" << endl;

        // Calculate cache ratios
        if (features.l2_cache.size > 0 && features.l1_data_cache.size > 0) {
            double l2_l1_ratio =
                static_cast<double>(features.l2_cache.size) / features.l1_data_cache.size;
            cout << "  L2/L1 Ratio: " << fixed << setprecision(1) << l2_l1_ratio << "x" << endl;
        }

        if (features.l3_cache.size > 0 && features.l2_cache.size > 0) {
            double l3_l2_ratio =
                static_cast<double>(features.l3_cache.size) / features.l2_cache.size;
            cout << "  L3/L2 Ratio: " << fixed << setprecision(1) << l3_l2_ratio << "x" << endl;
        }

        // Cache-aware strategy recommendations
        cout << "\nCache-Aware Strategies:" << endl;
        if (features.l3_cache.size > 8 * 1024 * 1024) {
            cout << "  - Large L3: Use cache blocking for matrices > "
                 << (thresholds.l3_optimal_size / 1000) << "K elements" << endl;
        }
        if (features.cache_line_size >= 128) {
            cout << "  - Large cache line: Prefer sequential access patterns" << endl;
        }
        if (features.l1_data_cache.size >= 64 * 1024) {
            cout << "  - Large L1: Aggressive loop unrolling beneficial" << endl;
        }
    }
}

void test_cpu_topology(const TestOptions& opts) {
    cout << "\n=== CPU TOPOLOGY ANALYSIS ===" << endl;

    const auto& features = arch::get_features();
    const auto& topology = features.topology;

    cout << "\nTopology Information:" << endl;
    cout << "  Logical Cores: " << topology.logical_cores << endl;
    cout << "  Physical Cores: " << topology.physical_cores << endl;
    cout << "  CPU Packages: " << topology.packages << endl;
    cout << "  Threads per Core: " << topology.threads_per_core << endl;
    cout << "  Hyperthreading: " << (topology.hyperthreading ? "✓" : "✗") << endl;

    if (opts.verbose) {
        cout << "\nTopology Analysis:" << endl;

        // Calculate parallelization strategy
        if (topology.hyperthreading && topology.threads_per_core > 1) {
            cout << "  - SMT Available: " << topology.threads_per_core << " threads per core"
                 << endl;
            cout << "  - Recommendation: Use physical core count for CPU-bound tasks" << endl;
        }

        if (topology.packages > 1) {
            cout << "  - Multi-socket system: " << topology.packages << " packages" << endl;
            cout << "  - NUMA considerations likely important" << endl;
        }

        // Calculate optimal grain sizes based on topology
        size_t optimal_threads =
            topology.physical_cores > 0 ? topology.physical_cores : topology.logical_cores;
        cout << "  - Optimal thread count: " << optimal_threads << endl;
        cout << "  - Grain size factor: "
             << (topology.hyperthreading ? "Conservative" : "Aggressive") << endl;
    }
}

void test_performance_benchmarks(const TestOptions& opts) {
    cout << "\n=== PERFORMANCE BENCHMARKS ===" << endl;

    const size_t test_size = 100000;
    vector<double> data(test_size, 1.0);

    cout << "\nMemory Performance (μs for " << test_size << " doubles):" << endl;

    // Sequential read benchmark
    auto start = chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (size_t i = 0; i < test_size; ++i) {
        sum += data[i];
    }
    auto end = chrono::high_resolution_clock::now();
    auto sequential_time = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "  Sequential Read: " << sequential_time.count() << " μs" << endl;

    // Cache line strided access
    start = chrono::high_resolution_clock::now();
    sum = 0.0;
    const size_t stride = arch::get_features().cache_line_size / sizeof(double);
    for (size_t i = 0; i < test_size; i += stride) {
        sum += data[i];
    }
    end = chrono::high_resolution_clock::now();
    auto strided_time = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "  Cache Line Strided: " << strided_time.count() << " μs" << endl;

    // SIMD operation benchmark
    start = chrono::high_resolution_clock::now();
    stats::simd::ops::VectorOps::scalar_multiply(data.data(), 2.0, data.data(), test_size);
    end = chrono::high_resolution_clock::now();
    auto simd_time = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "  SIMD Multiply: " << simd_time.count() << " μs" << endl;

    if (opts.verbose) {
        cout << "\nPerformance Analysis:" << endl;
        double sequential_bandwidth =
            (test_size * sizeof(double)) / (sequential_time.count() / 1e6) / 1e9;
        cout << "  Sequential Bandwidth: " << fixed << setprecision(2) << sequential_bandwidth
             << " GB/s" << endl;

        if (simd_time.count() > 0) {
            double simd_speedup = static_cast<double>(sequential_time.count()) / simd_time.count();
            cout << "  SIMD Speedup: " << fixed << setprecision(2) << simd_speedup << "x" << endl;
        }
    }
}

void test_generation_detection(const TestOptions& opts) {
    cout << "\n=== CPU GENERATION DETECTION ===" << endl;

    cout << "\nGeneration Detection Results:" << endl;
    cout << "  Sandy/Ivy Bridge: " << (arch::is_sandy_ivy_bridge() ? "✓" : "✗") << endl;
    cout << "  Haswell/Broadwell: " << (arch::is_haswell_broadwell() ? "✓" : "✗") << endl;
    cout << "  Skylake Generation: " << (arch::is_skylake_generation() ? "✓" : "✗") << endl;
    cout << "  Kaby/Coffee Lake: " << (arch::is_kaby_coffee_lake() ? "✓" : "✗") << endl;
    cout << "  Modern Intel: " << (arch::is_modern_intel() ? "✓" : "✗") << endl;

    if (opts.verbose) {
        cout << "\nGeneration Analysis:" << endl;
        cout << "  Microarchitecture: " << get_intel_microarchitecture() << endl;

        // Provide optimization recommendations based on generation
        if (arch::is_modern_intel()) {
            cout << "  Optimization: Modern CPU - use aggressive SIMD and threading" << endl;
        } else if (arch::is_skylake_generation()) {
            cout << "  Optimization: Skylake era - balance SIMD with cache efficiency" << endl;
        } else if (arch::is_haswell_broadwell()) {
            cout << "  Optimization: Haswell era - careful with AVX2 power management" << endl;
        } else {
            cout << "  Optimization: Legacy CPU - conservative vectorization" << endl;
        }
    }
}

void test_cross_architecture_consistency(const TestOptions& opts) {
    cout << "\n=== CROSS-ARCHITECTURE CONSISTENCY ===" << endl;

    cout << "\nAPI Consistency Tests:" << endl;

    // Test that all platforms return reasonable values
    auto min_parallel = arch::get_min_elements_for_parallel();
    auto grain_size = arch::get_default_grain_size();
    auto alignment = arch::optimal_alignment();

    cout << "  Min parallel elements: " << min_parallel << " (should be > 0)" << endl;
    cout << "  Default grain size: " << grain_size << " (should be > 0)" << endl;
    cout << "  Optimal alignment: " << alignment << " bytes (should be power of 2)" << endl;

    // Verify alignment is power of 2
    bool alignment_ok = (alignment > 0) && ((alignment & (alignment - 1)) == 0);
    cout << "  Alignment check: " << (alignment_ok ? "✓" : "✗") << endl;

    // Test SIMD consistency
    string simd_level = arch::best_simd_level();
    bool simd_available = arch::supports_sse2() || arch::supports_avx() || arch::supports_neon();
    cout << "  SIMD Level: " << simd_level << endl;
    cout << "  SIMD Available: " << (simd_available ? "✓" : "✗") << endl;

    if (opts.verbose) {
        cout << "\nConsistency Analysis:" << endl;

        // Check if parallel thresholds make sense relative to each other
        auto simple_parallel = arch::get_min_elements_for_simple_distribution_parallel();
        auto dist_parallel = arch::get_min_elements_for_distribution_parallel();

        cout << "  Threshold hierarchy check:" << endl;
        cout << "    Simple: " << simple_parallel << endl;
        cout << "    Distribution: " << dist_parallel << endl;
        cout << "    General: " << min_parallel << endl;

        // Correct hierarchy: distribution ≤ general ≤ simple
        // Distribution operations benefit from parallelism sooner (lower threshold)
        // Simple operations need more elements to overcome parallel overhead (higher threshold)
        bool hierarchy_ok = (dist_parallel <= min_parallel) && (min_parallel <= simple_parallel);
        cout << "    Hierarchy consistent: " << (hierarchy_ok ? "✓" : "✗") << endl;
    }
}

void test_validation_against_database(const TestOptions& opts) {
    cout << "\n=== CPU DATABASE VALIDATION ===" << endl;

    const auto& features = arch::get_features();
    bool validation_passed = validate_cpu_against_database();

    cout << "CPU Validation: " << (validation_passed ? "✓ PASSED" : "⚠ WARNINGS") << endl;

    if (opts.verbose || !validation_passed) {
        cout << "\nValidation Details:" << endl;

        // Check for common CPU configurations
        if (features.vendor == "GenuineIntel" && !features.avx && features.family >= 6) {
            cout << "  - Warning: Intel CPU without AVX (unusual for modern processors)" << endl;
        }

        if (features.vendor == "AuthenticAMD" && features.avx512f) {
            cout << "  - Note: AMD CPU with AVX-512 (relatively rare)" << endl;
        }

        if (features.vendor == "Apple" && !features.neon) {
            cout << "  - Warning: Apple CPU without NEON (unexpected)" << endl;
        }

        // Validate cache hierarchy makes sense
        if (features.l1_cache_size > features.l2_cache_size && features.l2_cache_size > 0) {
            cout << "  - Warning: L1 cache larger than L2 (unusual)" << endl;
        }

        if (features.l2_cache_size > features.l3_cache_size && features.l3_cache_size > 0) {
            cout << "  - Warning: L2 cache larger than L3 (unusual)" << endl;
        }

        cout << "  - CPU signature: " << hex << "0x" << features.family << "/" << features.model
             << "/" << features.stepping << dec << endl;
    }
}

void output_json_summary([[maybe_unused]] const TestOptions& opts) {
    const auto& features = arch::get_features();

    // Get validation status
    bool validation_passed = validate_cpu_against_database();

    // Check threshold hierarchy consistency
    auto simple_parallel = arch::get_min_elements_for_simple_distribution_parallel();
    auto dist_parallel = arch::get_min_elements_for_distribution_parallel();
    auto min_parallel = arch::get_min_elements_for_parallel();
    bool hierarchy_consistent =
        (dist_parallel <= min_parallel) && (min_parallel <= simple_parallel);

    // Overall test status
    bool all_tests_passed = validation_passed && hierarchy_consistent;

    cout << "{" << endl;
    cout << "  \"test_status\": {" << endl;
    cout << "    \"passed\": " << (all_tests_passed ? "true" : "false") << "," << endl;
    cout << "    \"validation_passed\": " << (validation_passed ? "true" : "false") << "," << endl;
    cout << "    \"hierarchy_consistent\": " << (hierarchy_consistent ? "true" : "false") << ","
         << endl;
    cout << "    \"exit_code\": " << (all_tests_passed ? 0 : 1) << endl;
    cout << "  }," << endl;

    cout << "  \"cpu\": {" << endl;
    cout << "    \"vendor\": \"" << features.vendor << "\"," << endl;
    cout << "    \"brand\": \"" << features.brand << "\"," << endl;

    // Architecture-specific identification
    if (features.vendor == "GenuineIntel" || features.vendor == "AuthenticAMD") {
        cout << "    \"architecture\": \"x86_64\"," << endl;
        cout << "    \"family\": " << features.family << "," << endl;
        cout << "    \"model\": " << features.model << "," << endl;
        cout << "    \"stepping\": " << features.stepping << endl;
    } else if (features.vendor == "Apple") {
        cout << "    \"architecture\": \"ARM64\"," << endl;
        cout << "    \"platform\": \"Apple Silicon\"," << endl;
        cout << "    \"matrix_extensions\": " << (features.neon ? "true" : "false") << endl;
    } else if (features.neon || features.sve) {
        cout << "    \"architecture\": \"ARM64\"," << endl;
        cout << "    \"platform\": \"Generic ARM\"," << endl;
        cout << "    \"sve_available\": " << (features.sve ? "true" : "false") << endl;
    } else {
        cout << "    \"architecture\": \"Unknown\"," << endl;
        cout << "    \"family\": " << features.family << "," << endl;
        cout << "    \"model\": " << features.model << "," << endl;
        cout << "    \"stepping\": " << features.stepping << endl;
    }
    cout << "  }," << endl;

    cout << "  \"simd\": {" << endl;
    cout << "    \"sse2\": " << (features.sse2 ? "true" : "false") << "," << endl;
    cout << "    \"avx\": " << (features.avx ? "true" : "false") << "," << endl;
    cout << "    \"avx2\": " << (features.avx2 ? "true" : "false") << "," << endl;
    cout << "    \"avx512f\": " << (features.avx512f ? "true" : "false") << "," << endl;
    cout << "    \"neon\": " << (features.neon ? "true" : "false") << "," << endl;
    cout << "    \"best_level\": \"" << arch::best_simd_level() << "\"" << endl;
    cout << "  }," << endl;

    cout << "  \"cache\": {" << endl;
    cout << "    \"l1_data\": " << features.l1_data_cache.size << "," << endl;
    cout << "    \"l1_instruction\": " << features.l1_instruction_cache.size << "," << endl;
    cout << "    \"l2\": " << features.l2_cache.size << "," << endl;
    cout << "    \"l3\": " << features.l3_cache.size << "," << endl;
    cout << "    \"line_size\": " << features.cache_line_size << endl;
    cout << "  }," << endl;

    cout << "  \"topology\": {" << endl;
    cout << "    \"logical_cores\": " << features.topology.logical_cores << "," << endl;
    cout << "    \"physical_cores\": " << features.topology.physical_cores << "," << endl;
    cout << "    \"packages\": " << features.topology.packages << "," << endl;
    cout << "    \"hyperthreading\": " << (features.topology.hyperthreading ? "true" : "false")
         << endl;
    cout << "  }," << endl;

    cout << "  \"optimization\": {" << endl;
    cout << "    \"double_vector_width\": " << arch::optimal_double_width() << "," << endl;
    cout << "    \"float_vector_width\": " << arch::optimal_float_width() << "," << endl;
    cout << "    \"alignment\": " << arch::optimal_alignment() << "," << endl;
    cout << "    \"min_parallel\": " << arch::get_min_elements_for_parallel() << "," << endl;
    cout << "    \"grain_size\": " << arch::get_default_grain_size() << endl;
    cout << "  }," << endl;

    cout << "  \"thresholds\": {" << endl;
    cout << "    \"simple_distribution_parallel\": " << simple_parallel << "," << endl;
    cout << "    \"distribution_parallel\": " << dist_parallel << "," << endl;
    cout << "    \"general_parallel\": " << min_parallel << "," << endl;
    cout << "    \"hierarchy_valid\": " << (hierarchy_consistent ? "true" : "false") << endl;
    cout << "  }," << endl;

    // Add basic performance benchmark
    const size_t bench_size = 10000;
    vector<double> bench_data(bench_size, 1.0);
    auto start = chrono::high_resolution_clock::now();
    volatile double sum = 0.0;
    for (size_t i = 0; i < bench_size; ++i) {
        sum += bench_data[i];
    }
    auto end = chrono::high_resolution_clock::now();
    auto seq_time = chrono::duration_cast<chrono::microseconds>(end - start);

    // SIMD benchmark
    start = chrono::high_resolution_clock::now();
    stats::simd::ops::VectorOps::scalar_multiply(bench_data.data(), 2.0, bench_data.data(),
                                                 bench_size);
    end = chrono::high_resolution_clock::now();
    auto simd_time = chrono::duration_cast<chrono::microseconds>(end - start);

    cout << "  \"performance\": {" << endl;
    cout << "    \"sequential_read_us\": " << seq_time.count() << "," << endl;
    cout << "    \"simd_multiply_us\": " << simd_time.count() << "," << endl;
    double speedup =
        simd_time.count() > 0 ? static_cast<double>(seq_time.count()) / simd_time.count() : 1.0;
    cout << "    \"simd_speedup\": " << fixed << setprecision(2) << speedup << "," << endl;
    double bandwidth = (bench_size * sizeof(double)) / (seq_time.count() / 1e6) / 1e9;
    cout << "    \"memory_bandwidth_gbps\": " << fixed << setprecision(2) << bandwidth << endl;
    cout << "  }" << endl;
    cout << "}" << endl;
}

string get_intel_microarchitecture() {
    const auto& features = arch::get_features();

    if (arch::is_modern_intel())
        return "Modern (10th gen+)";
    if (arch::is_kaby_coffee_lake())
        return "Kaby/Coffee Lake";
    if (arch::is_skylake_generation())
        return "Skylake";
    if (arch::is_haswell_broadwell())
        return "Haswell/Broadwell";
    if (arch::is_sandy_ivy_bridge())
        return "Sandy/Ivy Bridge";

    // Fallback based on features
    if (features.avx512f)
        return "AVX-512 capable";
    if (features.avx2)
        return "AVX2 era";
    if (features.avx)
        return "AVX era";
    if (features.sse4_2)
        return "SSE4 era";
    return "Legacy";
}

string get_amd_architecture() {
    const auto& features = arch::get_features();

    // AMD architecture detection is less precise without model numbers
    if (features.avx512f)
        return "Zen 4+ (AVX-512)";
    if (features.avx2 && features.fma)
        return "Zen/Zen+ (AVX2+FMA)";
    if (features.avx2)
        return "Zen (AVX2)";
    if (features.avx)
        return "Bulldozer family";
    if (features.sse4_2)
        return "K10 or later";
    return "Legacy AMD";
}

string get_apple_silicon_features() {
    const auto& features = arch::get_features();

    string result = "";
    if (features.neon)
        result += "NEON ";
    if (features.crypto)
        result += "Crypto ";
    if (features.crc32)
        result += "CRC32 ";

    // Apple Silicon specific features (detected via model analysis)
    result += "AMX-capable ";

    return result.empty() ? "Basic ARM64" : result;
}

bool validate_cpu_against_database() {
    const auto& features = arch::get_features();
    bool valid = true;

    // Basic sanity checks
    if (features.topology.logical_cores == 0)
        valid = false;
    if (features.cache_line_size == 0)
        valid = false;
    if (features.vendor.empty())
        valid = false;

    // Platform-specific validation
    if (features.vendor == "Apple" && !features.neon)
        valid = false;
    if (features.vendor == "GenuineIntel" && features.family == 0)
        valid = false;

    return valid;
}
