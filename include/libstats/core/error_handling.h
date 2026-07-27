#pragma once

#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <variant>  // std::variant (Result<T> storage) + std::monostate (VoidResult)

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
 * Implemented as a discriminated union (`std::variant<T, ErrorInfo>`) so that
 * the success and error paths are mutually exclusive and `makeError()` never
 * constructs `T`. This is the C++20 equivalent of C++23 `std::expected<T, E>`.
 *
 * **Migration from the v2.0.0-pre aggregate struct:**
 * | Old call site          | New call site               |
 * |------------------------|-----------------------------|
 * | `result.value`         | `*result`                   |
 * | `std::move(r.value)`   | `std::move(r).unwrap()`     |
 * | `result.error_code`    | `result.errorCode()`        |
 * | `result.message`       | `result.message()`          |
 *
 * **v2.x decision point**: when the project minimum is raised to macOS 14
 * (AppleClang 16), `Result<T>` can become a thin alias over
 * `std::expected<T, std::string>` with minimal call-site changes.
 *
 * @tparam T The type of the success value.
 */
template <typename T>
class Result {
    struct ErrorInfo {
        ValidationError code;
        std::string message;
        // Explicit constructor required: std::in_place_type uses direct-init,
        // which doesn't work for aggregates without a constructor.
        ErrorInfo(ValidationError c, std::string m) noexcept : code(c), message(std::move(m)) {}
    };

    std::variant<T, ErrorInfo> data_;

    // Private constructors — use the static factory methods.
    explicit Result(T val) : data_(std::in_place_type<T>, std::move(val)) {}
    explicit Result(ValidationError code, const std::string& msg)
        : data_(std::in_place_type<ErrorInfo>, code, msg) {}

   public:
    // -------------------------------------------------------------------------
    // Factory methods
    // -------------------------------------------------------------------------

    /** @brief Create a successful result containing @p val. */
    [[nodiscard]] static Result ok(T val) noexcept { return Result(std::move(val)); }

    /**
     * @brief Create an error result. T is **never** constructed.
     * @param code  Error category.
     * @param msg   Human-readable description.
     */
    [[nodiscard]] static Result makeError(ValidationError code, const std::string& msg) noexcept {
        return Result(code, msg);
    }

    // -------------------------------------------------------------------------
    // Status queries
    // -------------------------------------------------------------------------

    [[nodiscard]] bool isOk() const noexcept { return std::holds_alternative<T>(data_); }
    [[nodiscard]] bool isError() const noexcept { return !isOk(); }

    // -------------------------------------------------------------------------
    // Value access (only valid when isOk())
    // -------------------------------------------------------------------------

    /** @brief Dereference to the success value (lvalue ref). */
    [[nodiscard]] T& operator*() & { return std::get<T>(data_); }
    /** @brief Dereference to the success value (const lvalue ref). */
    [[nodiscard]] const T& operator*() const& { return std::get<T>(data_); }
    /** @brief Dereference to the success value (rvalue ref). */
    [[nodiscard]] T&& operator*() && { return std::get<T>(std::move(data_)); }

    /** @brief Arrow access to the success value's members. */
    [[nodiscard]] T* operator->() { return &std::get<T>(data_); }
    /** @brief Arrow access to the success value's members (const). */
    [[nodiscard]] const T* operator->() const { return &std::get<T>(data_); }

    /**
     * @brief Access or move the success value out.
     *
     * Three overloads cover all call patterns produced by the `.value` migration:
     * - `result.unwrap()` (lvalue) — returns T& (same as `*result`)
     * - `std::move(result).unwrap()` — returns T&& (moves value out of variant)
     * - `std::move(result.unwrap())` — lvalue overload returns T&, std::move produces T&&
     *
     * Undefined (std::bad_variant_access) if isError().
     */
    [[nodiscard]] T& unwrap() & { return std::get<T>(data_); }
    [[nodiscard]] const T& unwrap() const& { return std::get<T>(data_); }
    [[nodiscard]] T&& unwrap() && { return std::get<T>(std::move(data_)); }

    // -------------------------------------------------------------------------
    // Error access (only meaningful when isError())
    // -------------------------------------------------------------------------

    /** @brief Returns the error code, or ValidationError::None on success. */
    [[nodiscard]] ValidationError errorCode() const noexcept {
        if (const auto* e = std::get_if<ErrorInfo>(&data_))
            return e->code;
        return ValidationError::None;
    }

    /**
     * @brief Returns the error message, or an empty string on success.
     *
     * The returned reference is stable for the lifetime of this Result.
     */
    [[nodiscard]] const std::string& message() const noexcept {
        if (const auto* e = std::get_if<ErrorInfo>(&data_))
            return e->message;
        static const std::string empty;
        return empty;
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
    if (a > b) {
        return VoidResult::makeError(ValidationError::InvalidRange,
                                     "Upper bound (b) must be >= lower bound (a)");
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

/**
 * @brief Validate Geometric distribution parameters without throwing exceptions
 * @param p Success probability (must be in (0, 1])
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateGeometricParameters(double p) noexcept {
    if (std::isnan(p) || std::isinf(p) || p <= 0.0 || p > 1.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Success probability p must be in (0, 1]");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Laplace distribution parameters without throwing exceptions
 * @param mu Location parameter (must be finite)
 * @param b  Scale parameter (must be positive and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateLaplaceParameters(double mu, double b) noexcept {
    if (!std::isfinite(mu)) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Location parameter mu must be a finite number");
    }
    if (std::isnan(b) || std::isinf(b) || b <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Scale parameter b must be a positive finite number");
    }
    return VoidResult::ok({});
}

/**
 * @brief Validate Cauchy distribution parameters without throwing exceptions
 * @param x0    Location parameter (must be finite)
 * @param gamma Scale parameter (must be positive and finite)
 * @return VoidResult indicating success or failure
 */
inline VoidResult validateCauchyParameters(double x0, double gamma) noexcept {
    if (!std::isfinite(x0)) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Location parameter x0 must be a finite number");
    }
    if (std::isnan(gamma) || std::isinf(gamma) || gamma <= 0.0) {
        return VoidResult::makeError(ValidationError::InvalidParameter,
                                     "Scale parameter gamma must be a positive finite number");
    }
    return VoidResult::ok({});
}

}  // namespace stats
