#pragma once

namespace libstats {
namespace constants {
namespace precision {
    // Basic precision and tolerance values
    inline constexpr double ZERO = 1.0e-30;
    inline constexpr double DEFAULT_TOLERANCE = 1.0e-8;
    inline constexpr double HIGH_PRECISION_TOLERANCE = 1.0e-12;
    inline constexpr double ULTRA_HIGH_PRECISION_TOLERANCE = 1.0e-15;
    inline constexpr double LOG_PROBABILITY_EPSILON = 1.0e-300;
    inline constexpr double MIN_STD_DEV = 1.0e-6;
    inline constexpr double HIGH_PRECISION_UPPER_BOUND = 1.0e12;
    inline constexpr double MAX_STANDARD_DEVIATION = 1.0e10;
    
    // Machine epsilon for various types
    inline constexpr double MACHINE_EPSILON = std::numeric_limits<double>::epsilon();
    inline constexpr float MACHINE_EPSILON_FLOAT = std::numeric_limits<float>::epsilon();
    inline constexpr long double MACHINE_EPSILON_LONG_DOUBLE = std::numeric_limits<long double>::epsilon();
    
    // Numerical differentiation step sizes
    inline constexpr double FORWARD_DIFF_STEP = 1.0e-8;
    inline constexpr double CENTRAL_DIFF_STEP = 1.0e-6;
    inline constexpr double NUMERICAL_DERIVATIVE_STEP = 1.0e-5;
    
    // Convergence criteria for numerical methods
    inline constexpr double NEWTON_RAPHSON_TOLERANCE = 1.0e-10;
    inline constexpr double BISECTION_TOLERANCE = 1.0e-12;
    inline constexpr double GRADIENT_DESCENT_TOLERANCE = 1.0e-9;
    inline constexpr double CONJUGATE_GRADIENT_TOLERANCE = 1.0e-10;
    
    // Maximum iterations for numerical methods
    inline constexpr std::size_t MAX_NEWTON_ITERATIONS = 100;
    inline constexpr std::size_t MAX_BISECTION_ITERATIONS = 1000;
    inline constexpr std::size_t MAX_GRADIENT_DESCENT_ITERATIONS = 10000;
    inline constexpr std::size_t MAX_CONJUGATE_GRADIENT_ITERATIONS = 1000;
    
    // Maximum iterations for special mathematical functions
    inline constexpr std::size_t MAX_BETA_ITERATIONS = 100;
    inline constexpr std::size_t MAX_GAMMA_SERIES_ITERATIONS = 1000;
    
    // Numerical integration tolerances
    inline constexpr double INTEGRATION_TOLERANCE = 1.0e-10;
    inline constexpr double ADAPTIVE_INTEGRATION_TOLERANCE = 1.0e-8;
    inline constexpr double MONTE_CARLO_INTEGRATION_TOLERANCE = 1.0e-6;
    
    // Maximum recursion depth for adaptive Simpson's rule
    inline constexpr int MAX_ADAPTIVE_SIMPSON_DEPTH = 15;
}
} // namespace constants
} // namespace libstats
