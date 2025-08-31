// Use focused header for benchmark infrastructure
#include "../include/platform/benchmark.h"

// Standard library includes
#include <algorithm>  // for std::fill
#include <chrono>     // for std::chrono::microseconds, std::chrono::milliseconds
#include <cstddef>    // for std::size_t
#include <iostream>   // for std::cout, std::endl
#include <thread>     // for std::this_thread::sleep_for
#include <vector>     // for std::vector

using namespace stats;

int main() {
    std::cout << "=== Benchmark System Test ===\n\n";

    // Create a benchmark suite
    Benchmark bench(true, 10, 5);  // Enable warmup, 10 iterations, 5 warmup runs

    // Test 1: Simple computation benchmark
    bench.addTest("Simple Loop", []() {
        volatile int sum = 0;
        for (int i = 0; i < 1000; ++i) {
            sum = sum + i;
        }
    });

    // Test 2: Vector operations benchmark
    bench.addTest("Vector Operations", []() {
        std::vector<double> data(1000);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<double>(i) * 2.5;
        }

        // Compute sum
        volatile double sum = 0.0;
        for (double val : data) {
            sum = sum + val;
        }
    });

    // Test 3: Memory allocation benchmark
    bench.addTest("Memory Allocation", []() {
        std::vector<int> temp(500);
        std::fill(temp.begin(), temp.end(), 42);
    });

    // Test 4: Sleep test (should be consistent timing)
    bench.addTest(
        "Sleep Test", []() { std::this_thread::sleep_for(std::chrono::microseconds(100)); },
        5);  // Only 5 iterations for sleep test

    std::cout << "Running benchmarks...\n";

    // Run all benchmarks
    auto results = bench.runAll();

    // Print results
    bench.printResults();

    // Test StatsBenchmarkUtils
    std::cout << "\n=== Testing StatsBenchmarkUtils ===\n";

    auto testVectors = StatsBenchmarkUtils::createTestVectors(100, 1000, 5);
    std::cout << "Created " << testVectors.size() << " test vectors\n";

    Benchmark statsBench(false, 5);  // No warmup for quick test
    StatsBenchmarkUtils::benchmarkBasicStats(testVectors, statsBench);

    auto statsResults = statsBench.runAll();
    statsBench.printResults();

    // Test Timer class directly
    std::cout << "\n=== Testing Timer Class ===\n";

    Timer timer;
    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    double elapsed = timer.stop();

    std::cout << "Timer test: " << elapsed << "s (expected ~0.05s)\n";

    if (elapsed >= 0.045 && elapsed <= 0.055) {
        std::cout << "✓ Timer accuracy test passed\n";
    } else {
        std::cout << "⚠ Timer accuracy test: timing might be off\n";
    }

    std::cout << "\n🎉 Benchmark system tests completed!\n";

    return 0;
}
