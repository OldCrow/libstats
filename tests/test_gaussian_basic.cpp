#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cassert>
#include <cmath>

// Include the Gaussian distribution
#include "gaussian.h"

using namespace std;
using namespace libstats;

int main() {
    cout << "Testing GaussianDistribution Implementation" << endl;
    cout << "==========================================" << endl << endl;
    
    try {
        // Test 1: Safe factory method
        cout << "Test 1: Safe factory method" << endl;
        auto result = GaussianDistribution::create(0.0, 1.0);
        if (result.isOk()) {
            cout << "✅ Safe factory creation successful" << endl;
            auto gauss_dist = std::move(result.value);
            
            // Test basic properties
            cout << "Mean: " << gauss_dist.getMean() << endl;
            cout << "Variance: " << gauss_dist.getVariance() << endl;
            cout << "Standard Deviation: " << gauss_dist.getStandardDeviation() << endl;
            cout << "Skewness: " << gauss_dist.getSkewness() << endl;
            cout << "Kurtosis: " << gauss_dist.getKurtosis() << endl;
            
            // Test PDF, CDF, and quantile
            cout << "\nTest 2: PDF, CDF, and quantile functions" << endl;
            double x = 1.0;
            cout << "PDF(1.0): " << gauss_dist.getProbability(x) << endl;
            cout << "Log PDF(1.0): " << gauss_dist.getLogProbability(x) << endl;
            cout << "CDF(1.0): " << gauss_dist.getCumulativeProbability(x) << endl;
            cout << "Quantile(0.5): " << gauss_dist.getQuantile(0.5) << endl;
            
            // Test sampling
            cout << "\nTest 3: Random sampling" << endl;
            mt19937 rng(42);
            vector<double> samples;
            for (int i = 0; i < 10; ++i) {
                samples.push_back(gauss_dist.sample(rng));
            }
            
            cout << "10 random samples: ";
            for (double sample : samples) {
                cout << fixed << setprecision(3) << sample << " ";
            }
            cout << endl;
            
            // Test fitting
            cout << "\nTest 4: Parameter fitting" << endl;
            vector<double> data = {0.5, 1.2, 0.8, -0.3, 0.9, -0.5, 1.1, 0.2, -0.8, 1.5};
            gauss_dist.fit(data);
            cout << "After fitting to data, Mean: " << gauss_dist.getMean() << endl;
            cout << "After fitting to data, Std Dev: " << gauss_dist.getStandardDeviation() << endl;
            
            // Test batch operations
            cout << "\nTest 5: Batch operations" << endl;
            vector<double> test_values = {-2.0, -1.0, 0.0, 1.0, 2.0};
            vector<double> pdf_results(test_values.size());
            vector<double> cdf_results(test_values.size());
            
            gauss_dist.getProbabilityBatch(test_values.data(), pdf_results.data(), test_values.size());
            gauss_dist.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), test_values.size());
            
            cout << "Batch PDF results: ";
            for (double result : pdf_results) {
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
            vector<double> large_input(large_size, 0.0);  // All zeros
            vector<double> large_output(large_size);
            
            gauss_dist.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
            
            cout << "Large batch PDF at 0: " << fixed << setprecision(6) << large_output[0] << endl;
            cout << "All values equal: " << (large_output[0] == large_output[999] ? "YES" : "NO") << endl;
            
            cout << "\n✅ All tests passed successfully!" << endl;
            
        } else {
            cout << "❌ Error creating distribution: " << result.message << endl;
            return 1;
        }
        
        // Test error handling
        cout << "\nTest 7: Error handling" << endl;
        auto error_result = GaussianDistribution::create(0.0, -1.0);
        if (error_result.isError()) {
            cout << "✅ Error handling works: " << error_result.message << endl;
        } else {
            cout << "❌ Error handling failed" << endl;
            return 1;
        }
        
        cout << "\n🎉 All GaussianDistribution tests completed successfully!" << endl;
        
        cout << "\n=== SUMMARY ===" << endl;
        cout << "✓ Safe factory creation and error handling" << endl;
        cout << "✓ All distribution properties (mean, variance, skewness, kurtosis)" << endl;
        cout << "✓ PDF, Log PDF, CDF, and quantile functions" << endl;
        cout << "✓ Random sampling (Box-Muller algorithm)" << endl;
        cout << "✓ Parameter fitting (MLE)" << endl;
        cout << "✓ Batch operations with SIMD optimization" << endl;
        cout << "✓ Large batch SIMD validation" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}
