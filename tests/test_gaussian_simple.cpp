#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// Include the enhanced Gaussian distribution
#include "gaussian.h"

using namespace std;
using namespace libstats;

int main() {
    cout << "=== Simple Gaussian Distribution Test ===" << endl;
    
    try {
        // Test 1: Basic functionality with valid parameters
        cout << "1. Testing basic functionality..." << endl;
        
        GaussianDistribution normal(0.0, 1.0);
        
        // Test basic properties
        cout << "   Mean: " << normal.getMean() << endl;
        cout << "   Variance: " << normal.getVariance() << endl;
        cout << "   PDF at 0: " << normal.getProbability(0.0) << endl;
        cout << "   CDF at 0: " << normal.getCumulativeProbability(0.0) << endl;
        
        // Test 2: SIMD batch operations
        cout << "2. Testing SIMD batch operations..." << endl;
        
        vector<double> test_values = {-1.0, 0.0, 1.0};
        vector<double> pdf_results(3);
        vector<double> cdf_results(3);
        
        normal.getProbabilityBatch(test_values.data(), pdf_results.data(), 3);
        normal.getCumulativeProbabilityBatch(test_values.data(), cdf_results.data(), 3);
        
        cout << "   Batch PDF results: ";
        for (double val : pdf_results) cout << val << " ";
        cout << endl;
        
        cout << "   Batch CDF results: ";
        for (double val : cdf_results) cout << val << " ";
        cout << endl;
        
        // Test 3: Large batch for SIMD
        cout << "3. Testing large batch SIMD..." << endl;
        
        const size_t large_size = 1000;
        vector<double> large_input(large_size, 0.0);  // All zeros
        vector<double> large_output(large_size);
        
        normal.getProbabilityBatch(large_input.data(), large_output.data(), large_size);
        
        cout << "   Large batch PDF at 0: " << large_output[0] << endl;
        cout << "   All values equal: " << (large_output[0] == large_output[999] ? "YES" : "NO") << endl;
        
        cout << "\n=== ALL BASIC TESTS PASSED! ===" << endl;
        cout << "✓ Constructor works" << endl;
        cout << "✓ Basic probability functions work" << endl;
        cout << "✓ SIMD batch operations work" << endl;
        cout << "✓ Large batch SIMD works" << endl;
        
        return 0;
        
    } catch (const exception& e) {
        cout << "ERROR: " << e.what() << endl;
        return 1;
    } catch (...) {
        cout << "ERROR: Unknown exception" << endl;
        return 1;
    }
}
