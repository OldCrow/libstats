/**
 * @file test_uniform_simple.cpp
 * @brief Simple unit tests for UniformDistribution class
 * 
 * This file contains basic functionality tests for the UniformDistribution
 * class, including constructor, basic probability methods, and parameter
 * validation. These tests ensure the distribution works correctly for
 * simple use cases.
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include "../include/uniform.h"
#include "../include/constants.h"

void testBasicFunctionality() {
    std::cout << "Testing basic functionality..." << std::endl;
    
    // Test default constructor (U(0,1))
    libstats::UniformDistribution uniform;
    
    // Test basic properties
    assert(uniform.getLowerBound() == 0.0);
    assert(uniform.getUpperBound() == 1.0);
    assert(uniform.getMean() == 0.5);
    assert(std::abs(uniform.getVariance() - 1.0/12.0) < 1e-10);
    assert(uniform.getSkewness() == 0.0);
    assert(uniform.getKurtosis() == -1.2);
    assert(uniform.getWidth() == 1.0);
    assert(uniform.getMidpoint() == 0.5);
    
    // Test distribution properties
    assert(!uniform.isDiscrete());
    assert(uniform.getNumParameters() == 2);
    assert(uniform.getDistributionName() == "Uniform");
    assert(uniform.getSupportLowerBound() == 0.0);
    assert(uniform.getSupportUpperBound() == 1.0);
    
    std::cout << "✓ Basic functionality tests passed" << std::endl;
}

void testProbabilityMethods() {
    std::cout << "Testing probability methods..." << std::endl;
    
    // Test U(0,1) - unit interval
    libstats::UniformDistribution uniform(0.0, 1.0);
    
    // Test PDF
    assert(uniform.getProbability(0.5) == 1.0);  // Inside support
    assert(uniform.getProbability(0.0) == 1.0);  // At lower bound
    assert(uniform.getProbability(1.0) == 1.0);  // At upper bound
    assert(uniform.getProbability(-0.1) == 0.0); // Below support
    assert(uniform.getProbability(1.1) == 0.0);  // Above support
    
    // Test log-PDF
    assert(uniform.getLogProbability(0.5) == 0.0);  // log(1) = 0
    assert(uniform.getLogProbability(-0.1) == libstats::constants::probability::NEGATIVE_INFINITY);
    assert(uniform.getLogProbability(1.1) == libstats::constants::probability::NEGATIVE_INFINITY);
    
    // Test CDF
    assert(uniform.getCumulativeProbability(0.0) == 0.0);
    assert(uniform.getCumulativeProbability(0.5) == 0.5);
    assert(uniform.getCumulativeProbability(1.0) == 1.0);
    assert(uniform.getCumulativeProbability(-0.1) == 0.0);
    assert(uniform.getCumulativeProbability(1.1) == 1.0);
    
    // Test quantile
    assert(uniform.getQuantile(0.0) == 0.0);
    assert(uniform.getQuantile(0.5) == 0.5);
    assert(uniform.getQuantile(1.0) == 1.0);
    
    // Test U(2,5) - general case
    libstats::UniformDistribution uniform2(2.0, 5.0);
    
    // Test PDF
    assert(std::abs(uniform2.getProbability(3.0) - 1.0/3.0) < 1e-10);
    assert(uniform2.getProbability(1.0) == 0.0);
    assert(uniform2.getProbability(6.0) == 0.0);
    
    // Test log-PDF
    assert(std::abs(uniform2.getLogProbability(3.0) - std::log(1.0/3.0)) < 1e-10);
    
    // Test CDF
    assert(uniform2.getCumulativeProbability(2.0) == 0.0);
    assert(std::abs(uniform2.getCumulativeProbability(3.5) - 0.5) < 1e-10);
    assert(uniform2.getCumulativeProbability(5.0) == 1.0);
    
    // Test quantile
    assert(uniform2.getQuantile(0.0) == 2.0);
    assert(std::abs(uniform2.getQuantile(0.5) - 3.5) < 1e-10);
    assert(uniform2.getQuantile(1.0) == 5.0);
    
    std::cout << "✓ Probability methods tests passed" << std::endl;
}

void testParameterSetters() {
    std::cout << "Testing parameter setters..." << std::endl;
    
    libstats::UniformDistribution uniform(0.0, 1.0);
    
    // Test individual setters
    uniform.setLowerBound(-2.0);
    assert(uniform.getLowerBound() == -2.0);
    assert(uniform.getUpperBound() == 1.0);
    
    uniform.setUpperBound(3.0);
    assert(uniform.getLowerBound() == -2.0);
    assert(uniform.getUpperBound() == 3.0);
    
    // Test bounds setter
    uniform.setBounds(10.0, 20.0);
    assert(uniform.getLowerBound() == 10.0);
    assert(uniform.getUpperBound() == 20.0);
    assert(uniform.getWidth() == 10.0);
    assert(uniform.getMidpoint() == 15.0);
    
    // Test that cache is updated
    assert(std::abs(uniform.getMean() - 15.0) < 1e-10);
    assert(std::abs(uniform.getVariance() - 100.0/12.0) < 1e-10);
    
    std::cout << "✓ Parameter setters tests passed" << std::endl;
}

void testSafeFactory() {
    std::cout << "Testing safe factory methods..." << std::endl;
    
    // Test successful creation
    auto result = libstats::UniformDistribution::create(1.0, 3.0);
    assert(result.isOk());
    assert(result.value.getLowerBound() == 1.0);
    assert(result.value.getUpperBound() == 3.0);
    
    // Test failed creation (invalid parameters)
    auto result2 = libstats::UniformDistribution::create(5.0, 2.0);  // a > b
    assert(result2.isError());
    
    // Test parameter validation
    libstats::UniformDistribution uniform(0.0, 1.0);
    auto validation = uniform.validateCurrentParameters();
    assert(validation.isOk());
    
    // Test trySetParameters
    auto setResult = uniform.trySetParameters(2.0, 4.0);
    assert(setResult.isOk());
    assert(uniform.getLowerBound() == 2.0);
    assert(uniform.getUpperBound() == 4.0);
    
    auto setResult2 = uniform.trySetParameters(8.0, 6.0);  // invalid
    assert(setResult2.isError());
    
    std::cout << "✓ Safe factory tests passed" << std::endl;
}

void testSampling() {
    std::cout << "Testing sampling..." << std::endl;
    
    libstats::UniformDistribution uniform(0.0, 1.0);
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    
    // Test single sample
    double sample = uniform.sample(rng);
    assert(sample >= 0.0 && sample <= 1.0);
    
    // Test multiple samples
    std::vector<double> samples;
    for (int i = 0; i < 1000; ++i) {
        samples.push_back(uniform.sample(rng));
    }
    
    // Check all samples are in range
    for (double s : samples) {
        assert(s >= 0.0 && s <= 1.0);
    }
    
    // Basic statistical checks
    double sum = 0.0;
    for (double s : samples) {
        sum += s;
    }
    double mean = sum / samples.size();
    assert(std::abs(mean - 0.5) < 0.1);  // Should be close to 0.5
    
    std::cout << "✓ Sampling tests passed" << std::endl;
}

void testFitting() {
    std::cout << "Testing parameter fitting..." << std::endl;
    
    // Create test data from known distribution
    std::vector<double> data = {1.2, 2.3, 1.8, 2.9, 1.5, 2.1, 2.7, 1.9, 2.4, 1.6};
    
    libstats::UniformDistribution uniform;
    uniform.fit(data);
    
    // Check fitted parameters
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    
    // Fitted bounds should be slightly outside the data range
    assert(uniform.getLowerBound() < min_val);
    assert(uniform.getUpperBound() > max_val);
    
    std::cout << "✓ Fitting tests passed" << std::endl;
}

void testComparison() {
    std::cout << "Testing comparison operators..." << std::endl;
    
    libstats::UniformDistribution uniform1(1.0, 3.0);
    libstats::UniformDistribution uniform2(1.0, 3.0);
    libstats::UniformDistribution uniform3(2.0, 4.0);
    
    // Test equality
    assert(uniform1 == uniform2);
    assert(!(uniform1 == uniform3));
    
    // Test inequality
    assert(!(uniform1 != uniform2));
    assert(uniform1 != uniform3);
    
    std::cout << "✓ Comparison tests passed" << std::endl;
}

void testStringRepresentation() {
    std::cout << "Testing string representation..." << std::endl;
    
    libstats::UniformDistribution uniform(2.0, 5.0);
    std::string str = uniform.toString();
    
    // Check that string contains the parameters
    assert(str.find("2") != std::string::npos);
    assert(str.find("5") != std::string::npos);
    assert(str.find("Uniform") != std::string::npos);
    
    std::cout << "✓ String representation tests passed" << std::endl;
}

void testReset() {
    std::cout << "Testing reset functionality..." << std::endl;
    
    libstats::UniformDistribution uniform(5.0, 10.0);
    
    // Verify initial state
    assert(uniform.getLowerBound() == 5.0);
    assert(uniform.getUpperBound() == 10.0);
    
    // Reset to defaults
    uniform.reset();
    
    // Verify reset to U(0,1)
    assert(uniform.getLowerBound() == 0.0);
    assert(uniform.getUpperBound() == 1.0);
    
    std::cout << "✓ Reset tests passed" << std::endl;
}

int main() {
    std::cout << "Running Uniform Distribution Simple Tests..." << std::endl;
    
    try {
        testBasicFunctionality();
        testProbabilityMethods();
        testParameterSetters();
        testSafeFactory();
        testSampling();
        testFitting();
        testComparison();
        testStringRepresentation();
        testReset();
        
        std::cout << "\n✅ All Uniform distribution simple tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
