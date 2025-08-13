/**
 * @file error_handling_demo.cpp
 * @brief Comprehensive demonstration of error handling and safe usage patterns
 * 
 * This demo showcases proper error handling techniques, parameter validation,
 * edge case handling, and defensive programming practices when using libstats.
 * Essential for writing robust statistical applications.
 * 
 * Topics covered:
 * - Parameter validation and range checking
 * - Exception handling with try-catch blocks
 * - Edge case handling (extreme values, boundary conditions)
 * - Input sanitization and validation
 * - Graceful degradation and fallback strategies
 * - Numerical stability considerations
 * - Memory management and resource safety
 */

#include "libstats.h"

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

void print_subsection(const std::string& title) {
    std::cout << "\n" << std::string(50, '-') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '-') << std::endl;
}

void demonstrate_parameter_validation() {
    print_separator("1. Parameter Validation and Range Checking");
    
    std::cout << "Demonstrating proper parameter validation when creating distributions:\n" << std::endl;
    
    print_subsection("Valid Parameter Examples");
    
    // Use factory methods instead of constructors for safe creation
    auto uniform_result = libstats::UniformDistribution::create(0.0, 10.0);
    if (uniform_result.isOk()) {
        std::cout << "✅ Uniform(0.0, 10.0): Valid parameters (a < b)" << std::endl;
    } else {
        std::cout << "❌ Unexpected error: " << uniform_result.message << std::endl;
    }
    
    auto exp_result = libstats::ExponentialDistribution::create(1.5);
    if (exp_result.isOk()) {
        std::cout << "✅ Exponential(1.5): Valid rate parameter (λ > 0)" << std::endl;
    } else {
        std::cout << "❌ Unexpected error: " << exp_result.message << std::endl;
    }
    
    auto gaussian_result = libstats::GaussianDistribution::create(5.0, 2.0);
    if (gaussian_result.isOk()) {
        std::cout << "✅ Gaussian(5.0, 2.0): Valid parameters (σ > 0)" << std::endl;
    } else {
        std::cout << "❌ Unexpected error: " << gaussian_result.message << std::endl;
    }
    
    auto poisson_result = libstats::PoissonDistribution::create(7.5);
    if (poisson_result.isOk()) {
        std::cout << "✅ Poisson(7.5): Valid rate parameter (λ > 0)" << std::endl;
    } else {
        std::cout << "❌ Unexpected error: " << poisson_result.message << std::endl;
    }
    
    print_subsection("Invalid Parameter Handling");
    
    // Test invalid uniform distribution (a >= b) using factory method
    std::cout << "\n🔍 Testing Uniform distribution with invalid parameters:" << std::endl;
    auto invalid_uniform_result = libstats::UniformDistribution::create(10.0, 5.0);  // a > b
    if (invalid_uniform_result.isOk()) {
        std::cout << "❌ ERROR: Should have rejected invalid parameters for Uniform(10.0, 5.0)" << std::endl;
    } else {
        std::cout << "✅ Correctly rejected invalid parameters: " << invalid_uniform_result.message << std::endl;
    }
    
    // Test zero rate parameter using factory method
    std::cout << "\n🔍 Testing Exponential distribution with zero rate:" << std::endl;
    auto zero_exp_result = libstats::ExponentialDistribution::create(0.0);
    if (zero_exp_result.isOk()) {
        std::cout << "❌ ERROR: Should have rejected zero rate parameter" << std::endl;
    } else {
        std::cout << "✅ Correctly rejected zero rate: " << zero_exp_result.message << std::endl;
    }
    
    // Test negative standard deviation using factory method
    std::cout << "\n🔍 Testing Gaussian distribution with negative standard deviation:" << std::endl;
    auto negative_gaussian_result = libstats::GaussianDistribution::create(0.0, -1.0);
    if (negative_gaussian_result.isOk()) {
        std::cout << "❌ ERROR: Should have rejected negative standard deviation" << std::endl;
    } else {
        std::cout << "✅ Correctly rejected negative σ: " << negative_gaussian_result.message << std::endl;
    }
    
    // Test negative Poisson rate using factory method
    std::cout << "\n🔍 Testing Poisson distribution with negative rate:" << std::endl;
    auto negative_poisson_result = libstats::PoissonDistribution::create(-2.0);
    if (negative_poisson_result.isOk()) {
        std::cout << "❌ ERROR: Should have rejected negative rate parameter" << std::endl;
    } else {
        std::cout << "✅ Correctly rejected negative λ: " << negative_poisson_result.message << std::endl;
    }
    
    std::cout << "\n💡 Best Practice: Always validate parameters before creating distributions!" << std::endl;
}

void demonstrate_safe_creation_patterns() {
    print_separator("2. Safe Distribution Creation Patterns");
    
    std::cout << "Demonstrating defensive programming patterns for robust code:\n" << std::endl;
    
    // Safe factory function example using libstats factory methods
    auto safe_create_uniform = [](double a, double b) -> std::unique_ptr<libstats::Uniform> {
        // Use libstats factory method for safe creation
        auto result = libstats::UniformDistribution::create(a, b);
        if (result.isOk()) {
            // Move the distribution from the result
            return std::make_unique<libstats::Uniform>(std::move(result.value));
        } else {
            std::cerr << "Failed to create Uniform(" << a << ", " << b << "): " << result.message << std::endl;
            return nullptr;
        }
    };
    
    std::cout << "🛡️ Safe factory function example:" << std::endl;
    
    // Test with valid parameters
    if (auto dist = safe_create_uniform(0.0, 10.0)) {
        std::cout << "✅ Successfully created Uniform(0.0, 10.0)" << std::endl;
        std::cout << "   Mean: " << std::fixed << std::setprecision(3) << dist->getMean() << std::endl;
    }
    
    // Test with invalid parameters
    if (auto dist = safe_create_uniform(10.0, 5.0)) {
        std::cout << "❌ ERROR: Should not have created invalid distribution" << std::endl;
    } else {
        std::cout << "✅ Properly rejected invalid parameters" << std::endl;
    }
    
    // Test with NaN parameters
    if (auto dist = safe_create_uniform(std::numeric_limits<double>::quiet_NaN(), 5.0)) {
        std::cout << "❌ ERROR: Should not have created distribution with NaN" << std::endl;
    } else {
        std::cout << "✅ Properly rejected NaN parameters" << std::endl;
    }
    
    // Parameter validation helper
    auto validate_positive = [](double value, const std::string& param_name) {
        if (std::isnan(value)) {
            throw std::invalid_argument(param_name + " cannot be NaN");
        }
        if (std::isinf(value)) {
            throw std::invalid_argument(param_name + " cannot be infinite");
        }
        if (value <= 0.0) {
            throw std::invalid_argument(param_name + " must be positive");
        }
    };
    
    std::cout << "\n🔍 Parameter validation helper example:" << std::endl;
    
    try {
        double rate = 2.5;
        validate_positive(rate, "Exponential rate");
        libstats::Exponential exp_dist(rate);
        std::cout << "✅ Exponential distribution created with validated rate: " << rate << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ Validation failed: " << e.what() << std::endl;
    }
    
    try {
        double invalid_rate = -1.0;
        validate_positive(invalid_rate, "Exponential rate");
        libstats::Exponential exp_dist(invalid_rate);
        std::cout << "❌ ERROR: Should not reach here" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✅ Properly caught invalid rate: " << e.what() << std::endl;
    }
}

void demonstrate_numerical_edge_cases() {
    print_separator("3. Numerical Edge Cases and Stability");
    
    std::cout << "Testing behavior with extreme values and edge cases:\n" << std::endl;
    
    print_subsection("Extreme Parameter Values");
    
    // Very small standard deviation
    std::cout << "🔬 Testing very small standard deviation:" << std::endl;
    try {
        libstats::Gaussian tiny_std(0.0, 1e-10);
        std::cout << "✅ Created Gaussian with σ = 1e-10" << std::endl;
        std::cout << "   PDF at mean: " << std::scientific << std::setprecision(3) 
                  << tiny_std.getProbability(0.0) << std::endl;
        std::cout << "   CDF at mean: " << std::fixed << std::setprecision(6) 
                  << tiny_std.getCumulativeProbability(0.0) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠️  Exception with tiny std dev: " << e.what() << std::endl;
    }
    
    // Very large standard deviation
    std::cout << "\n🌊 Testing very large standard deviation:" << std::endl;
    try {
        libstats::Gaussian large_std(0.0, 1e6);
        std::cout << "✅ Created Gaussian with σ = 1e6" << std::endl;
        std::cout << "   PDF at mean: " << std::scientific << std::setprecision(3) 
                  << large_std.getProbability(0.0) << std::endl;
        std::cout << "   CDF at x = 1e6: " << std::fixed << std::setprecision(6) 
                  << large_std.getCumulativeProbability(1e6) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠️  Exception with large std dev: " << e.what() << std::endl;
    }
    
    // Very high Poisson rate
    std::cout << "\n🚀 Testing high Poisson rate (λ = 1000):" << std::endl;
    try {
        libstats::Poisson high_rate(1000.0);
        std::cout << "✅ Created Poisson with λ = 1000" << std::endl;
        std::cout << "   Mean: " << high_rate.getMean() << std::endl;
        std::cout << "   P(X = 1000): " << std::scientific << std::setprecision(3) 
                  << high_rate.getProbability(1000) << std::endl;
        std::cout << "   P(X ≤ 1000): " << std::fixed << std::setprecision(6) 
                  << high_rate.getCumulativeProbability(1000) << std::endl;
    } catch (const std::exception& e) {
        std::cout << "⚠️  Exception with high rate: " << e.what() << std::endl;
    }
    
    print_subsection("Boundary Value Testing");
    
    libstats::Uniform unit_uniform(0.0, 1.0);
    
    std::cout << "🎯 Testing uniform distribution boundary values:" << std::endl;
    std::cout << "   PDF at x = 0.0 (lower bound): " << unit_uniform.getProbability(0.0) << std::endl;
    std::cout << "   PDF at x = 1.0 (upper bound): " << unit_uniform.getProbability(1.0) << std::endl;
    std::cout << "   PDF at x = -0.001 (below range): " << unit_uniform.getProbability(-0.001) << std::endl;
    std::cout << "   PDF at x = 1.001 (above range): " << unit_uniform.getProbability(1.001) << std::endl;
    
    std::cout << "\n   CDF at x = 0.0: " << unit_uniform.getCumulativeProbability(0.0) << std::endl;
    std::cout << "   CDF at x = 1.0: " << unit_uniform.getCumulativeProbability(1.0) << std::endl;
    std::cout << "   CDF at x = -1.0: " << unit_uniform.getCumulativeProbability(-1.0) << std::endl;
    std::cout << "   CDF at x = 2.0: " << unit_uniform.getCumulativeProbability(2.0) << std::endl;
}

void demonstrate_sampling_safety() {
    print_separator("4. Safe Random Sampling Practices");
    
    std::cout << "Demonstrating safe sampling patterns and RNG management:\n" << std::endl;
    
    // RNG state management
    std::cout << "🎲 Random number generator state management:" << std::endl;
    
    std::mt19937 rng(42);  // Seeded for reproducibility
    libstats::Gaussian gaussian(0.0, 1.0);
    
    // Safe sampling with error handling
    auto safe_sample = [](auto& dist, std::mt19937& rng, int n_samples) -> std::vector<double> {
        std::vector<double> samples;
        samples.reserve(n_samples);
        
        try {
            for (int i = 0; i < n_samples; ++i) {
                double sample = dist.sample(rng);
                
                // Check for problematic values
                if (std::isnan(sample)) {
                    throw std::runtime_error("Sample is NaN");
                }
                if (std::isinf(sample)) {
                    throw std::runtime_error("Sample is infinite");
                }
                
                samples.push_back(sample);
            }
        } catch (const std::exception& e) {
            std::cerr << "Sampling error after " << samples.size() 
                      << " samples: " << e.what() << std::endl;
            // Return partial results
        }
        
        return samples;
    };
    
    auto samples = safe_sample(gaussian, rng, 10);
    std::cout << "✅ Generated " << samples.size() << " samples safely" << std::endl;
    
    // Batch sampling with validation
    std::cout << "\n📦 Batch sampling with validation:" << std::endl;
    try {
        auto batch_samples = gaussian.sample(rng, 1000);
        
        // Validate batch
        int nan_count = 0, inf_count = 0;
        double sum = 0.0;
        
        for (auto sample : batch_samples) {
            if (std::isnan(sample)) nan_count++;
            else if (std::isinf(sample)) inf_count++;
            else sum += sample;
        }
        
        std::cout << "   Total samples: " << batch_samples.size() << std::endl;
        std::cout << "   NaN samples: " << nan_count << std::endl;
        std::cout << "   Infinite samples: " << inf_count << std::endl;
        std::cout << "   Sample mean: " << std::fixed << std::setprecision(4) 
                  << sum / (batch_samples.size() - nan_count - inf_count) << std::endl;
        
        if (nan_count > 0 || inf_count > 0) {
            std::cout << "⚠️  Warning: Found problematic samples!" << std::endl;
        } else {
            std::cout << "✅ All samples are valid" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "❌ Batch sampling failed: " << e.what() << std::endl;
    }
    
    // RNG seeding best practices
    std::cout << "\n🌱 RNG seeding best practices:" << std::endl;
    
    // Method 1: Time-based seeding
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 time_rng(time_seed);
    std::cout << "✅ Time-based seeding for non-reproducible results" << std::endl;
    
    // Method 2: Random device seeding
    std::random_device rd;
    std::mt19937 random_rng(rd());
    std::cout << "✅ Random device seeding (hardware entropy when available)" << std::endl;
    
    // Method 3: Fixed seed for testing
    std::mt19937 test_rng(12345);
    std::cout << "✅ Fixed seed for reproducible testing" << std::endl;
}

void demonstrate_input_validation() {
    print_separator("5. Input Data Validation and Sanitization");
    
    std::cout << "Demonstrating input validation for distribution operations:\n" << std::endl;
    
    libstats::Gaussian gaussian(0.0, 1.0);
    libstats::Poisson poisson(5.0);
    
    // Input validation helper
    auto validate_input = [](double x, const std::string& context) -> bool {
        if (std::isnan(x)) {
            std::cout << "❌ Invalid input in " << context << ": NaN" << std::endl;
            return false;
        }
        if (std::isinf(x)) {
            std::cout << "❌ Invalid input in " << context << ": infinite" << std::endl;
            return false;
        }
        return true;
    };
    
    print_subsection("PDF/CDF Input Validation");
    
    std::vector<double> test_inputs = {
        0.0,                                    // Normal value
        std::numeric_limits<double>::quiet_NaN(), // NaN
        std::numeric_limits<double>::infinity(),   // Positive infinity
        -std::numeric_limits<double>::infinity(),  // Negative infinity
        1e308,                                    // Very large number
        -1e308                                    // Very small number
    };
    
    for (double x : test_inputs) {
        std::cout << "\n🔍 Testing input value: ";
        if (std::isnan(x)) std::cout << "NaN";
        else if (std::isinf(x)) std::cout << (x > 0 ? "+∞" : "-∞");
        else std::cout << x;
        std::cout << std::endl;
        
        if (validate_input(x, "PDF evaluation")) {
            try {
                double pdf_result = gaussian.getProbability(x);
                std::cout << "   PDF result: " << std::scientific << std::setprecision(3) << pdf_result << std::endl;
                
                double cdf_result = gaussian.getCumulativeProbability(x);
                std::cout << "   CDF result: " << std::fixed << std::setprecision(6) << cdf_result << std::endl;
            } catch (const std::exception& e) {
                std::cout << "   Exception: " << e.what() << std::endl;
            }
        }
    }
    
    print_subsection("Quantile Input Validation");
    
    std::vector<double> probability_inputs = {0.0, 0.25, 0.5, 0.75, 1.0, -0.1, 1.1, 
                                              std::numeric_limits<double>::quiet_NaN()};
    
    for (double p : probability_inputs) {
        std::cout << "\n🎯 Testing probability: ";
        if (std::isnan(p)) std::cout << "NaN";
        else std::cout << p;
        std::cout << std::endl;
        
        try {
            if (p >= 0.0 && p <= 1.0 && !std::isnan(p)) {
                double quantile = gaussian.getQuantile(p);
                std::cout << "✅ Quantile: " << std::fixed << std::setprecision(4) << quantile << std::endl;
            } else {
                std::cout << "❌ Invalid probability (must be in [0,1])" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << "❌ Exception: " << e.what() << std::endl;
        }
    }
}

void demonstrate_recovery_strategies() {
    print_separator("6. Error Recovery and Fallback Strategies");
    
    std::cout << "Demonstrating graceful error handling and recovery patterns:\n" << std::endl;
    
    // Fallback distribution creation using libstats factory methods
    auto create_distribution_with_fallback = [](double mean, double std_dev) -> std::unique_ptr<libstats::Gaussian> {
        // First try to create with requested parameters using factory method
        auto result = libstats::GaussianDistribution::create(mean, std_dev);
        if (result.isOk()) {
            return std::make_unique<libstats::Gaussian>(std::move(result.value));
        } else {
            std::cout << "⚠️  Failed to create Gaussian(" << mean << ", " << std_dev 
                      << "): " << result.message << std::endl;
            std::cout << "   Falling back to standard normal distribution" << std::endl;
            
            // Create fallback using factory method
            auto fallback = libstats::GaussianDistribution::create(0.0, 1.0);
            if (fallback.isOk()) {
                return std::make_unique<libstats::Gaussian>(std::move(fallback.value));
            } else {
                std::cerr << "❌ CRITICAL ERROR: Even fallback distribution failed!" << std::endl;
                return nullptr;
            }
        }
    };
    
    std::cout << "🛡️ Fallback distribution creation:" << std::endl;
    
    // Valid parameters
    if (auto dist = create_distribution_with_fallback(5.0, 2.0)) {
        std::cout << "✅ Created distribution with mean: " << dist->getMean() 
                  << ", std dev: " << libstats::getStandardDeviation(*dist) << std::endl;
    }
    
    // Invalid parameters (negative std dev)
    if (auto dist = create_distribution_with_fallback(5.0, -2.0)) {
        std::cout << "✅ Created fallback distribution with mean: " << dist->getMean() 
                  << ", std dev: " << libstats::getStandardDeviation(*dist) << std::endl;
    }
    
    print_subsection("Robust Statistical Computation");
    
    // Robust sample statistics with outlier handling
    auto compute_robust_stats = [](const std::vector<double>& data) {
        if (data.empty()) {
            throw std::invalid_argument("Empty data vector");
        }
        
        std::vector<double> clean_data;
        int nan_count = 0, inf_count = 0;
        
        // Filter out invalid values
        for (double val : data) {
            if (std::isnan(val)) {
                nan_count++;
            } else if (std::isinf(val)) {
                inf_count++;
            } else {
                clean_data.push_back(val);
            }
        }
        
        if (clean_data.empty()) {
            throw std::runtime_error("No valid data points after filtering");
        }
        
        // Compute statistics on clean data
        double sum = std::accumulate(clean_data.begin(), clean_data.end(), 0.0);
        double mean = sum / clean_data.size();
        
        double variance = 0.0;
        for (double val : clean_data) {
            variance += (val - mean) * (val - mean);
        }
        variance /= (clean_data.size() - 1);
        
        std::cout << "📊 Robust statistics computed:" << std::endl;
        std::cout << "   Original data size: " << data.size() << std::endl;
        std::cout << "   Valid data points: " << clean_data.size() << std::endl;
        std::cout << "   Filtered NaN: " << nan_count << std::endl;
        std::cout << "   Filtered Inf: " << inf_count << std::endl;
        std::cout << "   Mean: " << std::fixed << std::setprecision(4) << mean << std::endl;
        std::cout << "   Std Dev: " << std::sqrt(variance) << std::endl;
        
        return std::make_pair(mean, std::sqrt(variance));
    };
    
    // Test with contaminated data
    std::vector<double> contaminated_data = {
        1.0, 2.0, 3.0, 4.0, 5.0,  // Good data
        std::numeric_limits<double>::quiet_NaN(),  // NaN
        std::numeric_limits<double>::infinity(),   // Infinity
        6.0, 7.0, 8.0, 9.0, 10.0  // More good data
    };
    
    try {
        [[maybe_unused]] auto [robust_mean, robust_std] = compute_robust_stats(contaminated_data);
        std::cout << "✅ Successfully computed robust statistics" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "❌ Failed to compute statistics: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "=== libstats Error Handling and Safety Demo ===" << std::endl;
    std::cout << "Comprehensive guide to robust statistical programming" << std::endl;
    
    try {
        demonstrate_parameter_validation();
        demonstrate_safe_creation_patterns();
        demonstrate_numerical_edge_cases();
        demonstrate_sampling_safety();
        demonstrate_input_validation();
        demonstrate_recovery_strategies();
        
        print_separator("Best Practices Summary");
        std::cout << "✅ Always validate parameters before creating distributions" << std::endl;
        std::cout << "✅ Use try-catch blocks around distribution operations" << std::endl;
        std::cout << "✅ Check for NaN and infinity in inputs and outputs" << std::endl;
        std::cout << "✅ Implement fallback strategies for error recovery" << std::endl;
        std::cout << "✅ Validate probability inputs are in [0,1] range" << std::endl;
        std::cout << "✅ Use appropriate RNG seeding for your use case" << std::endl;
        std::cout << "✅ Filter invalid data before statistical computations" << std::endl;
        std::cout << "✅ Consider numerical stability with extreme parameter values" << std::endl;
        std::cout << "\n🛡️  Safe programming leads to reliable statistical applications!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Unexpected error in demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
