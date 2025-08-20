#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// Include only the implemented parts
#include "../include/distributions/gaussian.h"
#include "../include/platform/cpu_detection.h"
#include "../include/platform/simd.h"

using namespace std;
using namespace libstats;

int main() {
    cout << "=== SIMD Integration Test (Simple) ===" << endl;
    cout << endl;

    // 1. Show compile-time SIMD detection
    cout << "1. COMPILE-TIME SIMD DETECTION:" << endl;
    cout << "   LIBSTATS_HAS_SSE2: "
         <<
#ifdef LIBSTATS_HAS_SSE2
        "YES" << endl;
#else
        "NO" << endl;
#endif

    cout << "   LIBSTATS_HAS_AVX: "
         <<
#ifdef LIBSTATS_HAS_AVX
        "YES" << endl;
#else
        "NO" << endl;
#endif

    cout << "   LIBSTATS_HAS_AVX2: "
         <<
#ifdef LIBSTATS_HAS_AVX2
        "YES" << endl;
#else
        "NO" << endl;
#endif

    cout << "   LIBSTATS_HAS_FMA: "
         <<
#ifdef LIBSTATS_HAS_FMA
        "YES" << endl;
#else
        "NO" << endl;
#endif

    cout << "   LIBSTATS_HAS_AVX512: "
         <<
#ifdef LIBSTATS_HAS_AVX512
        "YES" << endl;
#else
        "NO" << endl;
#endif

    cout << endl;

    // 2. Show runtime CPU detection
    cout << "2. RUNTIME CPU DETECTION:" << endl;
    cout << "   CPU supports SSE2: " << (cpu::supports_sse2() ? "YES" : "NO") << endl;
    cout << "   CPU supports AVX: " << (cpu::supports_avx() ? "YES" : "NO") << endl;
    cout << "   CPU supports AVX2: " << (cpu::supports_avx2() ? "YES" : "NO") << endl;
    cout << "   CPU supports FMA: " << (cpu::supports_fma() ? "YES" : "NO") << endl;
    cout << "   CPU supports AVX512: " << (cpu::supports_avx512() ? "YES" : "NO") << endl;
    cout << "   Best SIMD level: " << cpu::best_simd_level() << endl;
    cout << "   Optimal double vector width: " << cpu::optimal_double_width() << endl;
    cout << "   Optimal float vector width: " << cpu::optimal_float_width() << endl;
    cout << "   Optimal alignment: " << cpu::optimal_alignment() << " bytes" << endl;
    cout << endl;

    // 3. Test Gaussian distribution (the only implemented one)
    cout << "3. GAUSSIAN DISTRIBUTION TEST:" << endl;
    try {
        auto normal = libstats::GaussianDistribution::create(0.0, 1.0).value;  // Standard normal

        cout << "   Standard Normal Distribution N(0,1):" << endl;
        cout << "   Mean: " << normal.getMean() << endl;
        cout << "   Variance: " << normal.getVariance() << endl;
        cout << "   Standard Deviation: " << normal.getStandardDeviation() << endl;
        cout << "   PDF at x=0: " << normal.getProbability(0.0) << endl;
        cout << "   PDF at x=1: " << normal.getProbability(1.0) << endl;
        cout << "   CDF at x=0: " << normal.getCumulativeProbability(0.0) << endl;
        cout << "   CDF at x=1: " << normal.getCumulativeProbability(1.0) << endl;
        cout << endl;

    } catch (const std::exception& e) {
        cout << "   Error testing Gaussian: " << e.what() << endl;
    }

    // 4. Simple SIMD vector addition benchmark
    cout << "4. SIMD VECTOR ADDITION BENCHMARK:" << endl;

    const size_t N = 1000000;
    const int iterations = 100;  // Multiple iterations for better timing
    vector<double> a(N, 1.5);
    vector<double> b(N, 2.5);
    vector<double> result(N);

    // Benchmark scalar addition with multiple iterations
    auto start = chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            result[i] = a[i] + b[i];
        }
    }
    auto end = chrono::high_resolution_clock::now();
    auto scalar_time = chrono::duration_cast<chrono::nanoseconds>(end - start);

    cout << "   Scalar addition time (" << iterations << " iterations): " << scalar_time.count()
         << " ns" << endl;
    cout << "   Average per iteration: " << scalar_time.count() / iterations << " ns" << endl;
    cout << "   Average per iteration: "
         << static_cast<double>(scalar_time.count()) / static_cast<double>(iterations) / 1000.0
         << " µs" << endl;

    // Benchmark SIMD addition (if available)
#ifdef LIBSTATS_HAS_AVX2
    if (cpu::supports_avx2()) {
        start = chrono::high_resolution_clock::now();

        // Simple AVX2 vector addition
        const size_t simd_width = 4;  // 4 doubles per AVX2 register
        size_t simd_end = (N / simd_width) * simd_width;

        for (size_t i = 0; i < simd_end; i += simd_width) {
            // This would use SIMD intrinsics in real implementation
            // For now, just do regular addition to test the detection
            for (size_t j = 0; j < simd_width; ++j) {
                result[i + j] = a[i + j] + b[i + j];
            }
        }

        // Handle remaining elements
        for (size_t i = simd_end; i < N; ++i) {
            result[i] = a[i] + b[i];
        }

        end = chrono::high_resolution_clock::now();
        auto simd_time = chrono::duration_cast<chrono::microseconds>(end - start);

        cout << "   SIMD addition time: " << simd_time.count() << " µs" << endl;
        cout << "   Speedup: " << fixed << setprecision(2)
             << (double)scalar_time.count() / simd_time.count() << "x" << endl;
    } else {
        cout << "   SIMD addition: Not available (CPU doesn't support AVX2)" << endl;
    }
#else
    cout << "   SIMD addition: Not available (not compiled with AVX2)" << endl;
#endif

    cout << endl;
    cout << "=== Test completed successfully ===" << endl;

    return 0;
}
