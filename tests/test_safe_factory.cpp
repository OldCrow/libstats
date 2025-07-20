#include <iostream>
#include <cassert>
#include "../include/gaussian.h"

using namespace libstats;

void testSafeFactory() {
    std::cout << "=== Testing Safe Factory Methods ===" << std::endl;
    
    // Test successful creation
    std::cout << "1. Testing valid parameters:" << std::endl;
    auto result = GaussianDistribution::create(0.0, 1.0);
    assert(result.isOk());
    std::cout << "   ✓ Valid parameters accepted" << std::endl;
    
    auto& dist = result.value;
    assert(std::abs(dist.getMean() - 0.0) < 1e-10);
    assert(std::abs(dist.getStandardDeviation() - 1.0) < 1e-10);
    std::cout << "   ✓ Parameters correctly set" << std::endl;
    
    // Test basic functionality
    double pdf = dist.getProbability(0.0);
    assert(std::abs(pdf - 0.39894228040143268) < 1e-10); // 1/sqrt(2π)
    std::cout << "   ✓ Basic functionality works" << std::endl;
    
    // Test error handling with invalid parameters
    std::cout << "2. Testing invalid parameters:" << std::endl;
    
    auto badResult1 = GaussianDistribution::create(0.0, 0.0);
    assert(badResult1.isError());
    std::cout << "   ✓ Zero standard deviation rejected: " << badResult1.message << std::endl;
    
    auto badResult2 = GaussianDistribution::create(0.0, -1.0);
    assert(badResult2.isError());
    std::cout << "   ✓ Negative standard deviation rejected: " << badResult2.message << std::endl;
    
    auto badResult3 = GaussianDistribution::create(std::numeric_limits<double>::infinity(), 1.0);
    assert(badResult3.isError());
    std::cout << "   ✓ Infinite mean rejected: " << badResult3.message << std::endl;
    
    auto badResult4 = GaussianDistribution::create(0.0, std::numeric_limits<double>::quiet_NaN());
    assert(badResult4.isError());
    std::cout << "   ✓ NaN standard deviation rejected: " << badResult4.message << std::endl;
    
    // Test trySetParameters
    std::cout << "3. Testing safe parameter setting:" << std::endl;
    
    auto tryResult = dist.trySetParameters(5.0, 2.0);
    assert(tryResult.isOk());
    assert(std::abs(dist.getMean() - 5.0) < 1e-10);
    assert(std::abs(dist.getStandardDeviation() - 2.0) < 1e-10);
    std::cout << "   ✓ Valid parameter update succeeded" << std::endl;
    
    auto tryBadResult = dist.trySetParameters(0.0, -1.0);
    assert(tryBadResult.isError());
    // Parameters should remain unchanged
    assert(std::abs(dist.getMean() - 5.0) < 1e-10);
    assert(std::abs(dist.getStandardDeviation() - 2.0) < 1e-10);
    std::cout << "   ✓ Invalid parameter update rejected: " << tryBadResult.message << std::endl;
    
    // Test parameter validation
    std::cout << "4. Testing parameter validation:" << std::endl;
    
    auto validationResult = dist.validateCurrentParameters();
    assert(validationResult.isOk());
    std::cout << "   ✓ Current parameters validated as correct" << std::endl;
    
    std::cout << "=== All Safe Factory Tests Passed! ===" << std::endl;
}

int main() {
    try {
        testSafeFactory();
        std::cout << "\n✓ All tests passed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
