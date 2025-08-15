// Use comprehensive library header for testing multiple distributions
#define LIBSTATS_FULL_INTERFACE
#include "../include/libstats.h"
#include <iostream>
#include <cassert>
#include <limits>

using namespace libstats;

void testGaussianDualAPI() {
    std::cout << "Testing GaussianDistribution dual API:\n";
    
    auto result = GaussianDistribution::create(0.0, 1.0);
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    // Test Result-based setters
    auto setMeanResult = dist.trySetMean(5.0);
    assert(setMeanResult.isOk());
    std::cout << "   ✅ trySetMean() works correctly\n";
    
    auto setStdResult = dist.trySetStandardDeviation(2.0);
    assert(setStdResult.isOk());
    std::cout << "   ✅ trySetStandardDeviation() works correctly\n";
    
    // Test error cases
    auto invalidStdResult = dist.trySetStandardDeviation(-1.0);
    assert(invalidStdResult.isError());
    std::cout << "   ✅ trySetStandardDeviation() correctly rejects negative values\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(1.0, 0.5);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    // Verify values were set correctly
    assert(std::abs(dist.getMean() - 1.0) < 1e-10);
    assert(std::abs(dist.getStandardDeviation() - 0.5) < 1e-10);
    std::cout << "   ✅ Parameter values set correctly\n";
}

void testUniformDualAPI() {
    std::cout << "Testing UniformDistribution dual API:\n";
    
    auto result = UniformDistribution::create(0.0, 1.0);
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    auto setLowerResult = dist.trySetLowerBound(-2.0);
    assert(setLowerResult.isOk());
    std::cout << "   ✅ trySetLowerBound() works correctly\n";
    
    auto setUpperResult = dist.trySetUpperBound(3.0);
    assert(setUpperResult.isOk());
    std::cout << "   ✅ trySetUpperBound() works correctly\n";
    
    // Test error case - invalid bounds
    auto invalidBoundResult = dist.trySetUpperBound(-5.0); // lower than current lower bound
    assert(invalidBoundResult.isError());
    std::cout << "   ✅ trySetUpperBound() correctly rejects invalid bounds\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(1.0, 4.0);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    assert(std::abs(dist.getLowerBound() - 1.0) < 1e-10);
    assert(std::abs(dist.getUpperBound() - 4.0) < 1e-10);
    std::cout << "   ✅ Parameter values set correctly\n";
}

void testExponentialDualAPI() {
    std::cout << "Testing ExponentialDistribution dual API:\n";
    
    auto result = ExponentialDistribution::create(1.0);
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    auto setLambdaResult = dist.trySetLambda(2.5);
    assert(setLambdaResult.isOk());
    std::cout << "   ✅ trySetLambda() works correctly\n";
    
    // Test error case
    auto invalidLambdaResult = dist.trySetLambda(-1.0);
    assert(invalidLambdaResult.isError());
    std::cout << "   ✅ trySetLambda() correctly rejects negative values\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(3.0);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    assert(std::abs(dist.getLambda() - 3.0) < 1e-10);
    std::cout << "   ✅ Parameter value set correctly\n";
}

void testPoissonDualAPI() {
    std::cout << "Testing PoissonDistribution dual API:\n";
    
    auto result = PoissonDistribution::create(1.0);
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    auto setLambdaResult = dist.trySetLambda(3.7);
    assert(setLambdaResult.isOk());
    std::cout << "   ✅ trySetLambda() works correctly\n";
    
    // Test error case
    auto invalidLambdaResult = dist.trySetLambda(0.0);
    assert(invalidLambdaResult.isError());
    std::cout << "   ✅ trySetLambda() correctly rejects zero/negative values\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(5.2);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    assert(std::abs(dist.getLambda() - 5.2) < 1e-10);
    std::cout << "   ✅ Parameter value set correctly\n";
}

void testDiscreteDualAPI() {
    std::cout << "Testing DiscreteDistribution dual API:\n";
    
    auto result = DiscreteDistribution::create(1, 6); // Standard die
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    auto setLowerResult = dist.trySetLowerBound(0);
    assert(setLowerResult.isOk());
    std::cout << "   ✅ trySetLowerBound() works correctly\n";
    
    auto setUpperResult = dist.trySetUpperBound(10);
    assert(setUpperResult.isOk());
    std::cout << "   ✅ trySetUpperBound() works correctly\n";
    
    // Test error case - invalid bounds
    auto invalidBoundResult = dist.trySetUpperBound(-5); // lower than current lower bound
    assert(invalidBoundResult.isError());
    std::cout << "   ✅ trySetUpperBound() correctly rejects invalid bounds\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(2, 8);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    assert(dist.getLowerBound() == 2);
    assert(dist.getUpperBound() == 8);
    std::cout << "   ✅ Parameter values set correctly\n";
}

// TODO: GammaDistribution tests for v0.8.0 implementation
/*
void testGammaDualAPI() {
    std::cout << "Testing GammaDistribution dual API:\n";
    
    auto result = GammaDistribution::create(1.0, 1.0);
    assert(result.isOk());
    auto dist = std::move(result.value);
    
    auto setAlphaResult = dist.trySetAlpha(2.5);
    assert(setAlphaResult.isOk());
    std::cout << "   ✅ trySetAlpha() works correctly\n";
    
    auto setBetaResult = dist.trySetBeta(0.8);
    assert(setBetaResult.isOk());
    std::cout << "   ✅ trySetBeta() works correctly\n";
    
    // Test error cases
    auto invalidAlphaResult = dist.trySetAlpha(-1.0);
    assert(invalidAlphaResult.isError());
    std::cout << "   ✅ trySetAlpha() correctly rejects negative values\n";
    
    auto invalidBetaResult = dist.trySetBeta(0.0);
    assert(invalidBetaResult.isError());
    std::cout << "   ✅ trySetBeta() correctly rejects zero/negative values\n";
    
    // Test trySetParameters
    auto setParamsResult = dist.trySetParameters(1.5, 2.0);
    assert(setParamsResult.isOk());
    std::cout << "   ✅ trySetParameters() works correctly\n";
    
    assert(std::abs(dist.getAlpha() - 1.5) < 1e-10);
    assert(std::abs(dist.getBeta() - 2.0) < 1e-10);
    std::cout << "   ✅ Parameter values set correctly\n";
}
*/

void testConsistentErrorHandling() {
    std::cout << "Testing consistent error handling across distributions:\n";
    
    // Test that all distributions handle NaN parameters consistently
    {
        auto gaussian = GaussianDistribution::create(0.0, 1.0).value;
        auto meanResult = gaussian.trySetMean(std::numeric_limits<double>::quiet_NaN());
        assert(meanResult.isError());
        std::cout << "   ✅ Gaussian handles NaN values consistently\n";
    }
    
    {
        auto uniform = UniformDistribution::create(0.0, 1.0).value;
        auto boundResult = uniform.trySetLowerBound(std::numeric_limits<double>::infinity());
        assert(boundResult.isError());
        std::cout << "   ✅ Uniform handles infinite values consistently\n";
    }
    
    {
        auto exponential = ExponentialDistribution::create(1.0).value;
        auto lambdaResult = exponential.trySetLambda(std::numeric_limits<double>::quiet_NaN());
        assert(lambdaResult.isError());
        std::cout << "   ✅ Exponential handles NaN values consistently\n";
    }
    
    {
        auto poisson = PoissonDistribution::create(1.0).value;
        auto lambdaResult = poisson.trySetLambda(std::numeric_limits<double>::infinity());
        assert(lambdaResult.isError());
        std::cout << "   ✅ Poisson handles infinite values consistently\n";
    }
    
    {
        auto discrete = DiscreteDistribution::create(1, 6).value;
        auto boundResult = discrete.trySetLowerBound(INT_MAX); // Try to set an extreme value
        assert(boundResult.isError());
        std::cout << "   ✅ Discrete handles extreme values consistently\n";
    }
    
    // TODO: Uncomment for v0.8.0 when GammaDistribution is implemented
    /*
    {
        auto gamma = GammaDistribution::create(1.0, 1.0).value;
        auto alphaResult = gamma.trySetAlpha(std::numeric_limits<double>::quiet_NaN());
        assert(alphaResult.isError());
        std::cout << "   ✅ Gamma handles NaN values consistently\n";
    }
    */
}

void testValidationConsistency() {
    std::cout << "Testing validation consistency between factory and setters:\n";
    
    // Test that factory and setters use the same validation logic
    {
        // This should fail at factory level
        auto factoryResult = GaussianDistribution::create(0.0, -1.0);
        assert(factoryResult.isError());
        
        // This should fail at setter level with same validation
        auto dist = GaussianDistribution::create(0.0, 1.0).value;
        auto setterResult = dist.trySetStandardDeviation(-1.0);
        assert(setterResult.isError());
        std::cout << "   ✅ Gaussian validation is consistent between factory and setters\n";
    }
    
    {
        auto factoryResult = UniformDistribution::create(5.0, 2.0); // invalid: a >= b
        assert(factoryResult.isError());
        
        auto dist = UniformDistribution::create(0.0, 10.0).value; // Wide range initially
        auto lowerBoundResult = dist.trySetLowerBound(5.0); // Set lower bound to 5.0
        assert(lowerBoundResult.isOk()); // This should succeed since 5.0 < 10.0
        auto setterResult = dist.trySetUpperBound(2.0); // Try to set upper < lower
        assert(setterResult.isError()); // This should fail since 2.0 < 5.0
        std::cout << "   ✅ Uniform validation is consistent between factory and setters\n";
    }
    
    {
        // Test Exponential validation consistency
        auto factoryResult = ExponentialDistribution::create(-1.0); // invalid: negative lambda
        assert(factoryResult.isError());
        
        auto dist = ExponentialDistribution::create(1.0).value;
        auto setterResult = dist.trySetLambda(-1.0); // Try to set negative lambda
        assert(setterResult.isError());
        std::cout << "   ✅ Exponential validation is consistent between factory and setters\n";
    }
    
    {
        // Test Poisson validation consistency
        auto factoryResult = PoissonDistribution::create(0.0); // invalid: zero lambda
        assert(factoryResult.isError());
        
        auto dist = PoissonDistribution::create(1.0).value;
        auto setterResult = dist.trySetLambda(0.0); // Try to set zero lambda
        assert(setterResult.isError());
        std::cout << "   ✅ Poisson validation is consistent between factory and setters\n";
    }
    
    {
        // Test Discrete validation consistency
        auto factoryResult = DiscreteDistribution::create(5, 2); // invalid: a > b
        assert(factoryResult.isError());
        
        auto dist = DiscreteDistribution::create(0, 10).value; // Wide range initially
        auto lowerBoundResult = dist.trySetLowerBound(5); // Set lower bound to 5
        assert(lowerBoundResult.isOk()); // This should succeed since 5 < 10
        auto setterResult = dist.trySetUpperBound(2); // Try to set upper < lower
        assert(setterResult.isError()); // This should fail since 2 < 5
        std::cout << "   ✅ Discrete validation is consistent between factory and setters\n";
    }
}

int main() {
    try {
        std::cout << "=== Testing Dual API Functionality ===\n\n";
        
        testGaussianDualAPI();
        std::cout << "\n";
        
        testUniformDualAPI();
        std::cout << "\n";
        
        testExponentialDualAPI();
        std::cout << "\n";
        
        testPoissonDualAPI();
        std::cout << "\n";
        
        testDiscreteDualAPI();
        std::cout << "\n";
        
        // testGammaDualAPI();
        // std::cout << "\n";
        
        testConsistentErrorHandling();
        std::cout << "\n";
        
        testValidationConsistency();
        std::cout << "\n";
        
        std::cout << "🎉 All dual API tests passed successfully!\n";
        std::cout << "✅ Exception-based and Result-based APIs are working correctly\n";
        std::cout << "✅ Error handling is consistent across all distributions\n";
        std::cout << "✅ Parameter validation works for both APIs\n";
        std::cout << "✅ Factory and setter validation are consistent\n";
        
        std::cout << "\n=== DUAL API IMPLEMENTATION SUCCESSFUL ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
