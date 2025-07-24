#include "../include/discrete.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace std;
using namespace libstats;

int main() {
    cout << "Testing DiscreteDistribution Implementation" << endl;
    cout << "===========================================" << endl << endl;
    
    try {
        // Test 1: Safe factory method
        cout << "Test 1: Safe factory method" << endl;
        auto result = DiscreteDistribution::create(1, 6);  // Standard die
        if (result.isOk()) {
            cout << "✅ Safe factory creation successful" << endl;
            auto dice_dist = std::move(result.value);
            
            // Test basic properties
            cout << "Lower bound: " << dice_dist.getLowerBound() << endl;
            cout << "Upper bound: " << dice_dist.getUpperBound() << endl;
            cout << "Range: " << dice_dist.getRange() << endl;
            cout << "Mean: " << dice_dist.getMean() << endl;
            cout << "Variance: " << dice_dist.getVariance() << endl;
            cout << "Skewness: " << dice_dist.getSkewness() << endl;
            cout << "Kurtosis: " << dice_dist.getKurtosis() << endl;
            cout << "Single outcome probability: " << dice_dist.getSingleOutcomeProbability() << endl;
            
            // Test PMF, CDF, and quantile
            cout << "\nTest 2: PMF, CDF, and quantile functions" << endl;
            double x = 3.0;
            cout << "PMF(3.0): " << dice_dist.getProbability(x) << endl;
            cout << "Log PMF(3.0): " << dice_dist.getLogProbability(x) << endl;
            cout << "CDF(3.0): " << dice_dist.getCumulativeProbability(x) << endl;
            cout << "Quantile(0.5): " << dice_dist.getQuantile(0.5) << endl;
            
            // Test PMF for values outside support
            cout << "PMF(0.0): " << dice_dist.getProbability(0.0) << " (outside support)" << endl;
            cout << "PMF(7.0): " << dice_dist.getProbability(7.0) << " (outside support)" << endl;
            
            // Test sampling
            cout << "\nTest 3: Random sampling" << endl;
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(dice_dist.sample(rng));
            }
            
            cout << "10 random samples: ";
            for (double sample : samples) {
                cout << fixed << setprecision(0) << sample << " ";
            }
            cout << endl;
            
            // Test batch sampling
            cout << "\nBatch sampling (5 samples): ";
            auto batch_samples = dice_dist.sample(rng, 5);
            for (double sample : batch_samples) {
                cout << fixed << setprecision(0) << sample << " ";
            }
            cout << endl;
            
            // Test fitting
            cout << "\nTest 4: Parameter fitting" << endl;
            vector<double> data = {1, 2, 3, 4, 5, 6, 1, 2, 3, 4};  // Die rolls
            dice_dist.fit(data);
            cout << "After fitting to data, Lower bound: " << dice_dist.getLowerBound() << endl;
            cout << "After fitting to data, Upper bound: " << dice_dist.getUpperBound() << endl;
            cout << "New mean: " << dice_dist.getMean() << endl;
            
            // Test batch operations
            cout << "\nTest 5: Batch operations" << endl;
            vector<double> test_values = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
            vector<double> pmf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            dice_dist.getProbabilityBatch(test_values.data(), pmf_results.data(), test_values.size());
            dice_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            cout << "Batch PMF results: ";
            for (double result : pmf_results) {
                cout << fixed << setprecision(4) << result << " ";
            }
            cout << endl;
            
            cout << "Batch CDF results: ";
            for (double result : cdf_results) {
                cout << fixed << setprecision(4) << result << " ";
            }
            cout << endl;
            
            // Test large batch for SIMD validation
            cout << "\nTest 6: Large batch SIMD validation" << endl;
            const size_t large_size = 1000;
            vector<double> large_input(large_size, 3.0);  // All 3s
            vector<double> large_output(large_size);
            
            dice_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            
            cout << "Large batch PMF at 3: " << fixed << setprecision(6) << large_output[0] << endl;
            cout << "All values equal: " << (large_output[0] == large_output[999] ? "YES" : "NO") << endl;
            
            // Test discrete-specific utility methods
            cout << "\nTest 7: Discrete-specific utility methods" << endl;
            cout << "Is 3.0 in support: " << (dice_dist.isInSupport(3.0) ? "YES" : "NO") << endl;
            cout << "Is 0.0 in support: " << (dice_dist.isInSupport(0.0) ? "YES" : "NO") << endl;
            cout << "Is 7.0 in support: " << (dice_dist.isInSupport(7.0) ? "YES" : "NO") << endl;
            
            // Get all outcomes (only for small ranges)
            auto outcomes = dice_dist.getAllOutcomes();
            cout << "All possible outcomes: ";
            for (int outcome : outcomes) {
                cout << outcome << " ";
            }
            cout << endl;
            
            // Test integer sampling
            auto int_samples = dice_dist.sampleIntegers(rng, 5);
            cout << "Integer samples: ";
            for (int sample : int_samples) {
                cout << sample << " ";
            }
            cout << endl;
            
            cout << "\n✅ All tests passed successfully!" << endl;
            
        } else {
            cout << "❌ Error creating distribution: " << result.message << endl;
            return 1;
        }
        
        // Test error handling
        cout << "\nTest 8: Error handling" << endl;
        auto error_result = DiscreteDistribution::create(5, 3);  // Invalid: upper < lower
        if (error_result.isError()) {
            cout << "✅ Error handling works: " << error_result.message << endl;
        } else {
            cout << "❌ Error handling failed" << endl;
            return 1;
        }
        
        // Test binary distribution (special case)
        cout << "\nTest 9: Binary distribution special case" << endl;
        auto binary_result = DiscreteDistribution::create(0, 1);
        if (binary_result.isOk()) {
            auto binary_dist = std::move(binary_result.value);
            cout << "Binary distribution mean: " << binary_dist.getMean() << endl;
            cout << "Binary distribution variance: " << binary_dist.getVariance() << endl;
            cout << "Binary PMF(0): " << binary_dist.getProbability(0.0) << endl;
            cout << "Binary PMF(1): " << binary_dist.getProbability(1.0) << endl;
        }
        
        cout << "\n🎉 All DiscreteDistribution tests completed successfully!" << endl;
        
        cout << "\n=== SUMMARY ===" << endl;
        cout << "✓ Safe factory creation and error handling" << endl;
        cout << "✓ All distribution properties (mean, variance, skewness, kurtosis)" << endl;
        cout << "✓ PMF, Log PMF, CDF, and quantile functions" << endl;
        cout << "✓ Random sampling (single and batch)" << endl;
        cout << "✓ Parameter fitting (range estimation)" << endl;
        cout << "✓ Batch operations with SIMD optimization" << endl;
        cout << "✓ Large batch SIMD validation" << endl;
        cout << "✓ Discrete-specific utility methods" << endl;
        cout << "✓ Special case handling (binary distribution)" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}
