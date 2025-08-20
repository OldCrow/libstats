#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace BasicTestUtilities {

/**
 * Standardized basic test framework for distribution implementations
 * Provides consistent test structure and output formatting across all distributions
 */
class StandardizedBasicTest {
   public:
    static void printTestHeader(const std::string& distributionName) {
        std::cout << "Testing " << distributionName << "Distribution Implementation" << std::endl;
        std::cout << std::string(40 + distributionName.length(), '=') << std::endl << std::endl;
    }

    static void printTestStart(int testNumber, const std::string& testName) {
        std::cout << "Test " << testNumber << ": " << testName << std::endl;
    }

    static void printTestSuccess(const std::string& message = "") {
        std::cout << "✅ " << (message.empty() ? "Test passed successfully" : message) << std::endl;
    }

    static void printTestError(const std::string& message) {
        std::cout << "❌ " << message << std::endl;
    }

    static void printProperty(const std::string& name, double value, int precision = 6) {
        std::cout << name << ": " << std::fixed << std::setprecision(precision) << value
                  << std::endl;
    }

    static void printPropertyInt(const std::string& name, int value) {
        std::cout << name << ": " << value << std::endl;
    }

    static void printSamples(const std::vector<double>& samples,
                             const std::string& prefix = "Samples", int precision = 3) {
        std::cout << prefix << ": ";
        for (double sample : samples) {
            std::cout << std::fixed << std::setprecision(precision) << sample << " ";
        }
        std::cout << std::endl;
    }

    static void printIntegerSamples(const std::vector<int>& samples,
                                    const std::string& prefix = "Integer samples") {
        std::cout << prefix << ": ";
        for (int sample : samples) {
            std::cout << sample << " ";
        }
        std::cout << std::endl;
    }

    static void printBatchResults(const std::vector<double>& results, const std::string& prefix,
                                  int precision = 4) {
        std::cout << prefix << ": ";
        for (double result : results) {
            std::cout << std::fixed << std::setprecision(precision) << result << " ";
        }
        std::cout << std::endl;
    }

    static void printLargeBatchValidation(double firstValue, double lastValue,
                                          const std::string& testType) {
        std::cout << "Large batch " << testType << " (first): " << std::fixed
                  << std::setprecision(6) << firstValue << std::endl;
        std::cout << "All values equal: " << (firstValue == lastValue ? "YES" : "NO") << std::endl;
    }

    static void printCompletionMessage(const std::string& distributionName) {
        std::cout << "\n🎉 All " << distributionName << "Distribution tests completed successfully!"
                  << std::endl;
    }

    static void printSummaryHeader() { std::cout << "\n=== SUMMARY ===" << std::endl; }

    static void printSummaryItem(const std::string& item) {
        std::cout << "✓ " << item << std::endl;
    }

    static void printNewline() { std::cout << std::endl; }

    // Helper function to validate samples are within expected range
    static bool validateSamplesInRange(const std::vector<double>& samples, double minVal,
                                       double maxVal) {
        return std::all_of(samples.begin(), samples.end(),
                           [minVal, maxVal](double x) { return x >= minVal && x <= maxVal; });
    }

    // Helper function to compute sample statistics
    static double computeSampleMean(const std::vector<double>& samples) {
        if (samples.empty())
            return 0.0;
        double sum = 0.0;
        for (double sample : samples) {
            sum += sample;
        }
        return sum / static_cast<double>(samples.size());
    }

    static double computeSampleVariance(const std::vector<double>& samples) {
        if (samples.size() < 2)
            return 0.0;
        double mean = computeSampleMean(samples);
        double sumSquaredDiffs = 0.0;
        for (double sample : samples) {
            double diff = sample - mean;
            sumSquaredDiffs += diff * diff;
        }
        return sumSquaredDiffs / static_cast<double>(samples.size() - 1);
    }

    // Helper function to check if two values are approximately equal
    static bool approxEqual(double a, double b, double tolerance = 1e-10) {
        return std::abs(a - b) < tolerance;
    }

    // Standard test data generators
    static std::vector<double> generateUniformTestData() {
        return {0.1, 0.3, 0.7, 0.2, 0.9, 0.4, 0.8, 0.6, 0.15, 0.85};
    }

    static std::vector<double> generateGaussianTestData() {
        return {0.5, 1.2, 0.8, -0.3, 0.9, -0.5, 1.1, 0.2, -0.8, 1.5};
    }

    static std::vector<double> generateExponentialTestData() {
        return {0.5, 1.2, 0.8, 2.1, 0.3, 1.5, 0.9, 1.8, 0.7, 1.1};
    }

    static std::vector<double> generateDiscreteTestData() { return {1, 2, 3, 4, 5, 6, 1, 2, 3, 4}; }

    static std::vector<double> generatePoissonTestData() {
        return {2, 1, 4, 3, 2, 5, 1, 3, 2, 4, 3, 2, 1, 4, 3};
    }

    static std::vector<double> generateGammaTestData() {
        return {0.8, 1.5, 2.1, 0.9, 1.2, 2.8, 1.1, 1.8, 0.7, 2.3};
    }
};

}  // namespace BasicTestUtilities
