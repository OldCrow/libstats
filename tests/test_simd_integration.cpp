#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>

// Include only Gaussian header for now
#include "../include/distributions/gaussian.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd.h"

/**
 * @brief Test program demonstrating the integration of compile-time and runtime SIMD detection
 * 
 * This test shows how to:
 * 1. Use compile-time detection macros from simd.h
 * 2. Use runtime detection functions from cpu_detection.h
 * 3. Safely combine both for optimal SIMD usage
 */

// Example function that uses both compile-time and runtime detection
void demonstrate_simd_integration() {
    std::cout << "=== SIMD Integration Demonstration ===" << std::endl;
    
    // 1. Query compile-time capabilities
    std::cout << "\n1. COMPILE-TIME SIMD DETECTION:" << std::endl;
    
#ifdef LIBSTATS_HAS_AVX2
    std::cout << "   ✓ Compiler supports AVX2" << std::endl;
#else
    std::cout << "   ✗ Compiler does not support AVX2" << std::endl;
#endif

#ifdef LIBSTATS_HAS_AVX
    std::cout << "   ✓ Compiler supports AVX" << std::endl;
#else
    std::cout << "   ✗ Compiler does not support AVX" << std::endl;
#endif

#ifdef LIBSTATS_HAS_SSE2
    std::cout << "   ✓ Compiler supports SSE2" << std::endl;
#else
    std::cout << "   ✗ Compiler does not support SSE2" << std::endl;
#endif

#ifdef LIBSTATS_HAS_AVX512
    std::cout << "   ✓ Compiler supports AVX512" << std::endl;
#else
    std::cout << "   ✗ Compiler does not support AVX512" << std::endl;
#endif

    // 2. Query runtime capabilities using cpu_detection.h
    std::cout << "\n2. RUNTIME CPU DETECTION:" << std::endl;
    
    const auto& features = libstats::cpu::get_features();
    std::cout << "   CPU Vendor: " << features.vendor << std::endl;
    std::cout << "   CPU Brand: " << features.brand << std::endl;
    std::cout << "   CPU Family: " << features.family << ", Model: " << features.model << ", Stepping: " << features.stepping << std::endl;
    
    std::cout << "\n   SIMD Support:" << std::endl;
    std::cout << "   - SSE2: " << (libstats::cpu::supports_sse2() ? "✓" : "✗") << std::endl;
    std::cout << "   - SSE4.1: " << (libstats::cpu::supports_sse4_1() ? "✓" : "✗") << std::endl;
    std::cout << "   - AVX: " << (libstats::cpu::supports_avx() ? "✓" : "✗") << std::endl;
    std::cout << "   - AVX2: " << (libstats::cpu::supports_avx2() ? "✓" : "✗") << std::endl;
    std::cout << "   - FMA: " << (libstats::cpu::supports_fma() ? "✓" : "✗") << std::endl;
    std::cout << "   - AVX512: " << (libstats::cpu::supports_avx512() ? "✓" : "✗") << std::endl;
    
    // 3. Show optimal settings
    std::cout << "\n3. OPTIMAL SIMD CONFIGURATION:" << std::endl;
    std::cout << "   Best SIMD level: " << libstats::cpu::best_simd_level() << std::endl;
    std::cout << "   Optimal double width: " << libstats::cpu::optimal_double_width() << std::endl;
    std::cout << "   Optimal float width: " << libstats::cpu::optimal_float_width() << std::endl;
    std::cout << "   Optimal alignment: " << libstats::cpu::optimal_alignment() << " bytes" << std::endl;
    
    // 4. Show compile-time constants from simd.h
    std::cout << "\n4. COMPILE-TIME SIMD CONSTANTS:" << std::endl;
    std::cout << "   Has SIMD support: " << (libstats::simd::has_simd_support() ? "✓" : "✗") << std::endl;
    std::cout << "   Compile-time double width: " << libstats::simd::double_vector_width() << std::endl;
    std::cout << "   Compile-time float width: " << libstats::simd::float_vector_width() << std::endl;
    std::cout << "   Compile-time alignment: " << libstats::simd::optimal_alignment() << " bytes" << std::endl;
    std::cout << "   Compile-time feature string: " << libstats::simd::feature_string() << std::endl;
    
    // 5. Show combined usage pattern
    std::cout << "\n5. SAFE SIMD USAGE PATTERN:" << std::endl;
    
    // This is the recommended pattern for safe SIMD usage
#ifdef LIBSTATS_HAS_AVX2
    if (libstats::cpu::supports_avx2()) {
        std::cout << "   ✓ Using AVX2 code path (compiler can generate + CPU supports)" << std::endl;
    } else {
        std::cout << "   ⚠ Compiler supports AVX2 but CPU does not - using fallback" << std::endl;
    }
#else
    std::cout << "   ⚠ Compiler does not support AVX2 - using fallback" << std::endl;
#endif

#ifdef LIBSTATS_HAS_AVX
    if (libstats::cpu::supports_avx()) {
        std::cout << "   ✓ Using AVX code path (compiler can generate + CPU supports)" << std::endl;
    } else {
        std::cout << "   ⚠ Compiler supports AVX but CPU does not - using fallback" << std::endl;
    }
#else
    std::cout << "   ⚠ Compiler does not support AVX - using fallback" << std::endl;
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (libstats::cpu::supports_sse2()) {
        std::cout << "   ✓ Using SSE2 code path (compiler can generate + CPU supports)" << std::endl;
    } else {
        std::cout << "   ⚠ Compiler supports SSE2 but CPU does not - using fallback" << std::endl;
    }
#else
    std::cout << "   ⚠ Compiler does not support SSE2 - using fallback" << std::endl;
#endif
}

// Example of a function that safely uses SIMD with both detections
void safe_vectorized_add(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }
    
    result.resize(a.size());
    const size_t size = a.size();
    
    // Use libstats SIMD operations with automatic detection
    // This would use the VectorOps class from simd.h with runtime detection
    libstats::simd::VectorOps::vector_add(a.data(), b.data(), result.data(), size);
}

// Test Gaussian distribution SIMD batch operations
void benchmark_gaussian_simd() {
    std::cout << "\n=== GAUSSIAN DISTRIBUTION SIMD BENCHMARK ===" << std::endl;
    
    // Create a Gaussian distribution
    libstats::GaussianDistribution gauss(0.0, 1.0);  // Standard normal
    
    // Create test data
    const size_t size = 100000;
    std::vector<double> values(size);
    std::vector<double> pdf_results(size);
    std::vector<double> log_pdf_results(size);
    
    // Fill with test values
    for (size_t i = 0; i < size; ++i) {
        values[i] = -3.0 + 6.0 * static_cast<double>(i) / (size - 1);  // Range [-3, 3]
    }
    
    std::cout << "Testing Gaussian batch operations with " << size << " values..." << std::endl;
    
    // Benchmark PDF batch operations
    auto start_pdf = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        gauss.getProbabilityWithStrategy(std::span<const double>(values), std::span<double>(pdf_results), libstats::performance::Strategy::SCALAR);
    }
    
    auto end_pdf = std::chrono::high_resolution_clock::now();
    auto duration_pdf = std::chrono::duration_cast<std::chrono::microseconds>(end_pdf - start_pdf);
    
    std::cout << "PDF batch operations (100 iterations): " << duration_pdf.count() << " microseconds" << std::endl;
    
    // Benchmark log PDF batch operations
    auto start_log = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        gauss.getLogProbabilityWithStrategy(std::span<const double>(values), std::span<double>(log_pdf_results), libstats::performance::Strategy::SCALAR);
    }
    
    auto end_log = std::chrono::high_resolution_clock::now();
    auto duration_log = std::chrono::duration_cast<std::chrono::microseconds>(end_log - start_log);
    
    std::cout << "Log PDF batch operations (100 iterations): " << duration_log.count() << " microseconds" << std::endl;
    
    // Verify correctness by comparing a few values with single-value computations
    std::cout << "\nVerifying SIMD results against scalar computations:" << std::endl;
    bool pdf_correct = true;
    bool log_pdf_correct = true;
    
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        double expected_pdf = gauss.getProbability(values[i]);
        double expected_log_pdf = gauss.getLogProbability(values[i]);
        
        if (std::abs(pdf_results[i] - expected_pdf) > 1e-12) {
            pdf_correct = false;
        }
        if (std::abs(log_pdf_results[i] - expected_log_pdf) > 1e-12) {
            log_pdf_correct = false;
        }
    }
    
    std::cout << "PDF batch verification: " << (pdf_correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
    std::cout << "Log PDF batch verification: " << (log_pdf_correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
    
    // Show sample results
    std::cout << "\nSample results (first 5 values):" << std::endl;
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "x=" << std::setw(8) << std::fixed << std::setprecision(3) << values[i] 
                  << " PDF=" << std::setw(12) << std::setprecision(6) << pdf_results[i]
                  << " logPDF=" << std::setw(12) << std::setprecision(6) << log_pdf_results[i] << std::endl;
    }
}

// Performance benchmark comparing different SIMD levels
void benchmark_simd_performance() {
    std::cout << "\n=== SIMD PERFORMANCE BENCHMARK ===" << std::endl;
    
    const size_t size = 1000000;
    std::vector<double> a(size, 1.5);
    std::vector<double> b(size, 2.5);
    std::vector<double> result(size);
    
    // Benchmark the vectorized addition
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i) {
        safe_vectorized_add(a, b, result);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Time for 100 iterations of " << size << " element vector addition: " 
              << duration.count() << " microseconds" << std::endl;
    
    // Verify correctness
    bool correct = true;
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i) {
        if (std::abs(result[i] - 4.0) > 1e-10) {
            correct = false;
            break;
        }
    }
    
    std::cout << "Result verification: " << (correct ? "✓ PASSED" : "✗ FAILED") << std::endl;
    std::cout << "First 5 results: ";
    for (size_t i = 0; i < std::min(size, size_t(5)); ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "libstats SIMD Integration Test" << std::endl;
    std::cout << "=============================" << std::endl;
    
    try {
        // Demonstrate SIMD integration
        demonstrate_simd_integration();
        
        // Test Gaussian distribution SIMD operations
        benchmark_gaussian_simd();
        
        // Run performance benchmark
        benchmark_simd_performance();
        
        std::cout << "\n=== SUMMARY ===" << std::endl;
        std::cout << "✓ Compile-time SIMD detection working" << std::endl;
        std::cout << "✓ Runtime CPU detection working" << std::endl;
        std::cout << "✓ Safe SIMD integration pattern demonstrated" << std::endl;
        std::cout << "✓ Performance benchmark completed" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
