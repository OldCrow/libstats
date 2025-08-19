#pragma once

#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <climits>
#include <limits>

namespace libstats {

/**
 * @brief Error codes for distribution parameter validation
 * 
 * This enum replaces exception-based error handling to avoid ABI compatibility
 * issues with Homebrew LLVM libc++ on macOS. The specific issue is that exceptions
 * thrown from the library compiled with Homebrew LLVM cannot be safely caught
 * in applications, leading to segfaults during exception unwinding.
 */
enum class ValidationError {
    None = 0,           ///< No error
    InvalidMean,        ///< Mean parameter is NaN or infinite
    InvalidStdDev,      ///< Standard deviation is NaN, infinite, or non-positive
    InvalidRange,       ///< Parameter values are outside valid range
    InvalidParameter,   ///< Generic invalid parameter error
    UnknownError       ///< Catch-all for unexpected errors
};

/**
 * @brief Result type for operations that may fail
 * 
 * This provides a safe alternative to exceptions for error reporting.
 * @tparam T The type of the result value
 */
template<typename T>
struct Result {
    T value;
    ValidationError error_code;
    std::string message;
    
    /**
     * @brief Check if the result represents success
     * @return true if no error occurred
     */
    bool isOk() const noexcept {
        return error_code == ValidationError::None;
    }
    
    /**
     * @brief Check if the result represents an error
     * @return true if an error occurred
     */
    bool isError() const noexcept {
        return error_code != ValidationError::None;
    }
    
    /**
     * @brief Create a successful result
     * @param val The success value
     * @return Result representing success
     */
    static Result<T> ok(T val) noexcept {
        return {std::move(val), ValidationError::None, ""};
    }
    
    /**
     * @brief Create an error result
     * @param err The error code
     * @param msg Error message
     * @return Result representing an error
     */
    static Result<T> makeError(ValidationError err, const std::string& msg) noexcept {
        return {T{}, err, msg};
    }
};

/**
 * @brief Specialized result type for void operations
 */
using VoidResult = Result<bool>;

/**
 * @brief Convert ValidationError to human-readable string
 * @param error The error code
 * @return String description of the error
 */
inline std::string errorToString(ValidationError error) noexcept {
    switch (error) {
        case ValidationError::None:
            return "No error";
        case ValidationError::InvalidMean:
            return "Mean must be a finite number";
        case ValidationError::InvalidStdDev:
            return "Standard deviation must be a positive finite number";
        case ValidationError::InvalidRange:
            return "Parameter values are outside valid range";
        case ValidationError::InvalidParameter:
            return "Invalid parameter value";
        case ValidationError::UnknownError:
        default:
            return "Unknown error occurred";
    }
}

/**
 * @brief Validate Gaussian distribution parameters without throwing exceptions
 * @param mean Mean parameter
 * @param stdDev Standard deviation parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateGaussianParameters(double mean, double stdDev) noexcept {
    if (std::isnan(mean) || std::isinf(mean)) {
        return VoidResult::makeError(ValidationError::InvalidMean, 
                                "Mean must be a finite number");
    }
    
    if (std::isnan(stdDev) || std::isinf(stdDev) || stdDev <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidStdDev, 
                                "Standard deviation must be a positive finite number");
    }
    
    return VoidResult::ok(true);
}

/**
 * @brief Validate Exponential distribution parameters without throwing exceptions
 * @param lambda Rate parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateExponentialParameters(double lambda) noexcept {
    if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter, 
                                "Lambda (rate parameter) must be a positive finite number");
    }
    
    return VoidResult::ok(true);
}

/**
 * @brief Validate Uniform distribution parameters without throwing exceptions
 * @param a Lower bound parameter
 * @param b Upper bound parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateUniformParameters(double a, double b) noexcept {
    if (std::isnan(a) || std::isinf(a) || std::isnan(b) || std::isinf(b)) {
        return VoidResult::makeError(ValidationError::InvalidParameter, 
                                "Uniform distribution parameters must be finite numbers");
    }
    
    if (a >= b) {
        return VoidResult::makeError(ValidationError::InvalidRange, 
                                "Upper bound (b) must be strictly greater than lower bound (a)");
    }
    
    return VoidResult::ok(true);
}

/**
 * @brief Validate Discrete distribution parameters without throwing exceptions
 * @param a Lower bound parameter
 * @param b Upper bound parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateDiscreteParameters(int a, int b) noexcept {
    if (a > b) {
        return VoidResult::makeError(ValidationError::InvalidRange, 
                                "Upper bound (b) must be greater than or equal to lower bound (a)");
    }
    
    // Check for integer overflow in range calculation
    constexpr int int_max = std::numeric_limits<int>::max();
    constexpr int int_min = std::numeric_limits<int>::min();
    if (b > int_max - 1 || a < int_min + 1) {
        return VoidResult::makeError(ValidationError::InvalidRange, 
                                "Parameter range too large - risk of integer overflow");
    }
    
    // Additional safety check for very large ranges
    const long long range_check = static_cast<long long>(b) - static_cast<long long>(a) + 1;
    if (range_check > int_max) {
        return VoidResult::makeError(ValidationError::InvalidRange, 
                                "Parameter range exceeds maximum supported size");
    }
    
    return VoidResult::ok(true);
}

/**
 * @brief Validate Poisson distribution parameters without throwing exceptions
 * @param lambda Rate parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validatePoissonParameters(double lambda) noexcept {
    if (std::isnan(lambda) || std::isinf(lambda) || lambda <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter, 
                                "Lambda (rate parameter) must be a positive finite number");
    }
    
    // Check for practical upper limit to avoid numerical issues
    if (lambda > 1e6) {
        return VoidResult::makeError(ValidationError::InvalidRange, 
                                "Lambda too large for accurate Poisson computation");
    }
    
    return VoidResult::ok(true);
}

/**
 * @brief Validate Gamma distribution parameters without throwing exceptions
 * @param alpha Shape parameter
 * @param beta Rate parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateGammaParameters(double alpha, double beta) noexcept {
    if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter, 
                                "Alpha (shape parameter) must be a positive finite number");
    }
    
    if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter, 
                                "Beta (rate parameter) must be a positive finite number");
    }
    
    return VoidResult::ok(true);
}

} // namespace libstats
