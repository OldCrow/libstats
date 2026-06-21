#pragma once

#include <climits>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <variant>  // for std::monostate (VoidResult sentinel)

namespace stats {

/**
 * @brief Error codes for distribution parameter validation
 *
 * Used by Result<T> / VoidResult to report validation failures without
 * exceptions. See Result<T> documentation below for the v2.0.0 design rationale.
 */
enum class ValidationError {
    None = 0,          ///< No error
    InvalidMean,       ///< Mean parameter is NaN or infinite
    InvalidStdDev,     ///< Standard deviation is NaN, infinite, or non-positive
    InvalidRange,      ///< Parameter values are outside valid range
    InvalidParameter,  ///< Generic invalid parameter error
    UnknownError       ///< Catch-all for unexpected errors
};

/**
 * @brief Result type for operations that may fail.
 *
 * **v2.0.0 trajectory decision** (June 2026):
 * In v1.x, Result<T> was introduced as an ABI workaround: exceptions thrown
 * from the library compiled with Homebrew LLVM libc++ could not be safely
 * caught by applications linked against Apple libc++, causing segfaults during
 * stack unwinding. Removing Homebrew LLVM in v2.0.0 eliminates that constraint.
 *
 * Result<T> is **retained** in v2.0.0 as a deliberate design choice:
 * - Explicit error handling is easier to audit than hidden exception paths.
 * - Factory functions that validate parameters are naturally expressed as
 *   Result<Distribution>, keeping the hot-path constructors noexcept.
 * - No dependency on compiler exception-handling ABI.
 *
 * **v2.x decision point**: `std::expected<T, E>` (C++23) is available on
 * AppleClang 16 (Xcode 16, macOS 14 Sonoma) and GCC 12 / Clang 16.
 * When the project minimum is raised to macOS 14, a v2.x minor may introduce
 * `using Result = std::expected<T, std::string>` as a drop-in typedef —
 * the public API surface is already compatible with that substitution.
 * Do NOT migrate to std::expected in v2.0.0; validate baseline availability
 * first.
 *
 * @tparam T The type of the result value on success.
 */
template <typename T>
struct Result {
    T value;
    ValidationError error_code;
    std::string message;

    /**
     * @brief Check if the result represents success
     * @return true if no error occurred
     */
    bool isOk() const noexcept { return error_code == ValidationError::None; }

    /**
     * @brief Check if the result represents an error
     * @return true if an error occurred
     */
    bool isError() const noexcept { return error_code != ValidationError::None; }

    /**
     * @brief Create a successful result
     * @param val The success value
     * @return Result representing success
     */
    static Result<T> ok(T val) noexcept { return {std::move(val), ValidationError::None, ""}; }

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
 * @brief Specialized result type for void (no-value) operations.
 *
 * Uses std::monostate as the success sentinel value so the success
 * branch carries no meaningful payload. The canonical usage pattern is:
 * @code
 *   VoidResult::ok({})               // success
 *   VoidResult::makeError(code, msg) // failure
 *   if (result.isOk()) { ... }       // check result
 * @endcode
 *
 * **v2.x migration note**: v1.x used `Result<bool>` with `ok(true)` as
 * the sentinel. v2.0.0 uses `Result<std::monostate>` to make the
 * absence of a meaningful value explicit.
 */
using VoidResult = Result<std::monostate>;

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
        return VoidResult::makeError(ValidationError::InvalidMean, "Mean must be a finite number");
    }

    if (std::isnan(stdDev) || std::isinf(stdDev) || stdDev <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidStdDev,
                                     "Standard deviation must be a positive finite number");
    }

    return VoidResult::ok({});
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

    return VoidResult::ok({});
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
        return VoidResult::makeError(
            ValidationError::InvalidRange,
            "Upper bound (b) must be strictly greater than lower bound (a)");
    }

    return VoidResult::ok({});
}

/**
 * @brief Validate Discrete distribution parameters without throwing exceptions
 * @param a Lower bound parameter
 * @param b Upper bound parameter
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateDiscreteParameters(int a, int b) noexcept {
    if (a >= b) {
        return VoidResult::makeError(
            ValidationError::InvalidRange,
            "Upper bound (b) must be strictly greater than lower bound (a)");
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

    return VoidResult::ok({});
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

    return VoidResult::ok({});
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

    return VoidResult::ok({});
}

/**
 * @brief Validate Log-Normal distribution parameters without throwing exceptions
 * @param mu Location parameter (log-mean, any finite real)
 * @param sigma Scale parameter (log-stddev, must be positive)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateLogNormalParameters(double mu, double sigma) noexcept {
    if (std::isnan(mu) || std::isinf(mu)) {
        return VoidResult::makeError(ValidationError::InvalidMean,
                                     "Mu (log-mean) must be a finite real number");
    }
    if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidStdDev,
                                     "Sigma (log-stddev) must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Pareto distribution parameters without throwing exceptions
 * @param scale Scale parameter (minimum value x_m, must be positive)
 * @param alpha Shape parameter (must be positive)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateParetoParameters(double scale, double alpha) noexcept {
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Scale (minimum value) must be a positive finite number");
    }
    if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Alpha (shape) must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Weibull distribution parameters without throwing exceptions
 * @param shape Shape parameter k (must be positive)
 * @param scale Scale parameter λ (must be positive)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateWeibullParameters(double shape, double scale) noexcept {
    if (std::isnan(shape) || std::isinf(shape) || shape <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Shape (k) must be a positive finite number");
    }
    if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Scale (λ) must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Rayleigh distribution parameters without throwing exceptions
 * @param sigma Scale parameter σ (must be positive)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateRayleighParameters(double sigma) noexcept {
    if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Sigma (σ) must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Von Mises distribution parameters without throwing exceptions
 * @param mu Mean direction (must be finite)
 * @param kappa Concentration parameter (must be non-negative and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateVonMisesParameters(double mu, double kappa) noexcept {
    if (std::isnan(mu) || std::isinf(mu)) {
        return VoidResult::makeError(ValidationError::InvalidMean,
                                     "Mu (mean direction) must be a finite real number");
    }
    if (std::isnan(kappa) || std::isinf(kappa) || kappa < 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Kappa (concentration) must be a non-negative finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Binomial distribution parameters without throwing exceptions
 * @param n Number of trials (must be positive integer)
 * @param p Success probability (must be in [0, 1])
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateBinomialParameters(int n, double p) noexcept {
    if (n <= 0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Number of trials n must be a positive integer");
    }
    if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Success probability p must be in [0, 1]");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Negative Binomial distribution parameters without throwing exceptions
 * @param r Number of successes (must be positive)
 * @param p Success probability (must be in (0, 1])
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateNegativeBinomialParameters(double r, double p) noexcept {
    if (std::isnan(r) || std::isinf(r) || r <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Number of successes r must be a positive finite number");
    }
    if (std::isnan(p) || std::isinf(p) || p <= 0.0 || p > 1.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Success probability p must be in (0, 1]");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Beta distribution parameters without throwing exceptions
 * @param alpha Shape parameter α (must be positive and finite)
 * @param beta  Shape parameter β (must be positive and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateBetaParameters(double alpha, double beta) noexcept {
    if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Alpha (shape1) must be a positive finite number");
    }
    if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Beta (shape2) must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Chi-squared distribution parameters without throwing exceptions
 * @param k Degrees of freedom (must be positive and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateChiSquaredParameters(double k) noexcept {
    if (std::isnan(k) || std::isinf(k) || k <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Degrees of freedom k must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Student's t distribution parameters without throwing exceptions
 * @param nu Degrees of freedom ν (must be positive and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateStudentTParameters(double nu) noexcept {
    if (std::isnan(nu) || std::isinf(nu) || nu <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Degrees of freedom nu must be a positive finite number");
    }
    return VoidResult::ok({});
}

}  // namespace stats
