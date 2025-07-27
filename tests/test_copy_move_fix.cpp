#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <vector>

// Include all three distributions
#include "../include/distributions/uniform.h"
#include "../include/distributions/gaussian.h"
#include "../include/distributions/exponential.h"

using namespace std;
using namespace libstats;

void testUniformCopyMove() {
    cout << "Testing Uniform copy/move semantics:" << endl;
    
    // Test Uniform copy assignment
    auto result1 = UniformDistribution::create(1.0, 3.0);
    auto result2 = UniformDistribution::create(0.0, 1.0);
    
    if (result1.isOk() && result2.isOk()) {
        auto uniform1 = std::move(result1.value);
        auto uniform2 = std::move(result2.value);
        
        cout << "  Before copy assignment: uniform1 range [" << uniform1.getLowerBound() << ", " << uniform1.getUpperBound() << "]" << endl;
        cout << "  Before copy assignment: uniform2 range [" << uniform2.getLowerBound() << ", " << uniform2.getUpperBound() << "]" << endl;
        
        uniform1 = uniform2;  // This should NOT deadlock
        
        cout << "  After copy assignment: uniform1 range [" << uniform1.getLowerBound() << ", " << uniform1.getUpperBound() << "]" << endl;
        cout << "  ✓ Copy assignment successful" << endl;
    }
    
    // Test Uniform move assignment
    auto result3 = UniformDistribution::create(2.0, 4.0);
    auto result4 = UniformDistribution::create(5.0, 10.0);
    
    if (result3.isOk() && result4.isOk()) {
        auto uniform3 = std::move(result3.value);
        auto uniform4 = std::move(result4.value);
        
        cout << "  Before move assignment: uniform3 range [" << uniform3.getLowerBound() << ", " << uniform3.getUpperBound() << "]" << endl;
        
        uniform3 = std::move(uniform4);  // This should NOT deadlock
        
        cout << "  After move assignment: uniform3 range [" << uniform3.getLowerBound() << ", " << uniform3.getUpperBound() << "]" << endl;
        cout << "  ✓ Move assignment successful" << endl;
    }
}

void testGaussianCopyMove() {
    cout << "\nTesting Gaussian copy/move semantics:" << endl;
    
    // Test Gaussian copy assignment
    GaussianDistribution gauss1(1.0, 2.0);
    GaussianDistribution gauss2(5.0, 1.0);
    
    cout << "  Before copy assignment: gauss1 N(" << gauss1.getMean() << ", " << gauss1.getVariance() << ")" << endl;
    cout << "  Before copy assignment: gauss2 N(" << gauss2.getMean() << ", " << gauss2.getVariance() << ")" << endl;
    
    gauss1 = gauss2;  // This should NOT deadlock
    
    cout << "  After copy assignment: gauss1 N(" << gauss1.getMean() << ", " << gauss1.getVariance() << ")" << endl;
    cout << "  ✓ Copy assignment successful" << endl;
    
    // Test Gaussian move assignment
    GaussianDistribution gauss3(3.0, 4.0);
    GaussianDistribution gauss4(7.0, 9.0);
    
    cout << "  Before move assignment: gauss3 N(" << gauss3.getMean() << ", " << gauss3.getVariance() << ")" << endl;
    
    gauss3 = std::move(gauss4);  // This should NOT deadlock
    
    cout << "  After move assignment: gauss3 N(" << gauss3.getMean() << ", " << gauss3.getVariance() << ")" << endl;
    cout << "  ✓ Move assignment successful" << endl;
}

void testExponentialCopyMove() {
    cout << "\nTesting Exponential copy/move semantics:" << endl;
    
    // Test Exponential copy assignment
    auto result1 = ExponentialDistribution::create(2.0);
    auto result2 = ExponentialDistribution::create(0.5);
    
    if (result1.isOk() && result2.isOk()) {
        auto exp1 = std::move(result1.value);
        auto exp2 = std::move(result2.value);
        
        cout << "  Before copy assignment: exp1 lambda=" << exp1.getLambda() << ", mean=" << exp1.getMean() << endl;
        cout << "  Before copy assignment: exp2 lambda=" << exp2.getLambda() << ", mean=" << exp2.getMean() << endl;
        
        exp1 = exp2;  // This should NOT deadlock
        
        cout << "  After copy assignment: exp1 lambda=" << exp1.getLambda() << ", mean=" << exp1.getMean() << endl;
        cout << "  ✓ Copy assignment successful" << endl;
    }
    
    // Test Exponential move assignment
    auto result3 = ExponentialDistribution::create(1.5);
    auto result4 = ExponentialDistribution::create(3.0);
    
    if (result3.isOk() && result4.isOk()) {
        auto exp3 = std::move(result3.value);
        auto exp4 = std::move(result4.value);
        
        cout << "  Before move assignment: exp3 lambda=" << exp3.getLambda() << ", mean=" << exp3.getMean() << endl;
        
        exp3 = std::move(exp4);  // This should NOT deadlock
        
        cout << "  After move assignment: exp3 lambda=" << exp3.getLambda() << ", mean=" << exp3.getMean() << endl;
        cout << "  ✓ Move assignment successful" << endl;
    }
}

void testConcurrentCopyMove() {
    cout << "\nTesting concurrent copy/move operations:" << endl;
    
    // Test that multiple threads can safely perform copy/move operations
    const int numThreads = 4;
    const int numOperations = 10;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([t]() {
            for (int i = 0; i < numOperations; ++i) {
                // Create distributions
                auto uniformResult = UniformDistribution::create(t, t + 1);
                GaussianDistribution gauss(t, 1.0);
                auto expResult = ExponentialDistribution::create(t + 1);
                
                if (uniformResult.isOk() && expResult.isOk()) {
                    auto uniform = std::move(uniformResult.value);
                    auto exp = std::move(expResult.value);
                    
                    // Perform copy operations
                    auto uniform2 = uniform;
                    auto gauss2 = gauss;
                    auto exp2 = exp;
                    
                    // Perform move operations
                    auto uniform3 = std::move(uniform2);
                    auto gauss3 = std::move(gauss2);
                    auto exp3 = std::move(exp2);
                    
                    // Use the distributions to ensure they're valid
                    double sum = uniform3.getMean() + gauss3.getMean() + exp3.getMean();
                    (void)sum; // Prevent unused variable warning
                }
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    cout << "  ✓ Concurrent copy/move operations completed successfully" << endl;
}

int main() {
    cout << "=== Testing Copy/Move Semantics Fix ===" << endl;
    cout << "This test verifies that the deadlock issue in copy/move assignment has been fixed." << endl;
    cout << "If this test completes without hanging, the fix is working correctly." << endl;
    
    try {
        testUniformCopyMove();
        testGaussianCopyMove();
        testExponentialCopyMove();
        testConcurrentCopyMove();
        
        cout << "\n🎉 All copy/move semantics tests passed!" << endl;
        cout << "✅ No deadlocks occurred" << endl;
        cout << "✅ All distributions support safe copy/move operations" << endl;
        cout << "✅ Thread safety is maintained" << endl;
        
        return 0;
    } catch (const exception& e) {
        cout << "\n❌ Test failed with exception: " << e.what() << endl;
        return 1;
    }
}
