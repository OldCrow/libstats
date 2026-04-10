// AVX-specific SIMD implementations
// This file is compiled ONLY with AVX flags to ensure safety
//
// Some algorithms and polynomial coefficients in this file are inspired by or
// derived from the SLEEF library (https://github.com/shibatch/sleef), which is
// licensed under the Boost Software License 1.0. The Boost license is fully
// compatible with our MIT License.

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC target("avx")
    #pragma GCC target("no-avx512f,no-avx2")
#elif defined(__clang__)
    #pragma clang attribute push(__attribute__((target("avx"))), apply_to = function)
#elif defined(_MSC_VER)
// MSVC doesn't need target pragmas - uses /arch flags in CMake
// and has different intrinsic handling
#endif

#include "libstats/common/simd_implementation_common.h"

#include <cmath>
#include <immintrin.h>  // AVX intrinsics
#include <limits>       // std::numeric_limits

namespace stats {
namespace simd {
namespace ops {

// All AVX functions use double-precision (64-bit) values
// AVX processes 4 doubles per 256-bit register

double VectorOps::dot_product_avx(const double* a, const double* b, std::size_t size) noexcept {
    // Runtime safety check - bail out if AVX not supported
    if (!stats::arch::supports_avx()) {
        return dot_product_fallback(a, b, size);
    }

    __m256d sum = _mm256_setzero_pd();
    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    // Process quartets of doubles
    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d prod = _mm256_mul_pd(va, vb);
        sum = _mm256_add_pd(sum, prod);
    }

    // Extract horizontal sum
    double result[4];
    _mm256_storeu_pd(result, sum);
    double final_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        final_sum += a[i] * b[i];
    }

    return final_sum;
}

void VectorOps::vector_add_avx(const double* a, const double* b, double* result,
                               std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_add_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorOps::vector_subtract_avx(const double* a, const double* b, double* result,
                                    std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_subtract_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorOps::vector_multiply_avx(const double* a, const double* b, double* result,
                                    std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_multiply_fallback(a, b, result, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vresult = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorOps::scalar_multiply_avx(const double* a, double scalar, double* result,
                                    std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return scalar_multiply_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_mul_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    // Handle remaining elements
    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

void VectorOps::scalar_add_avx(const double* a, double scalar, double* result,
                               std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return scalar_add_fallback(a, scalar, result, size);
    }

    __m256d vscalar = _mm256_set1_pd(scalar);
    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vresult = _mm256_add_pd(va, vscalar);
        _mm256_storeu_pd(&result[i], vresult);
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        result[i] = a[i] + scalar;
    }
}

// Transcendental function implementations

void VectorOps::vector_exp_avx(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_exp_fallback(input, output, size);
    }

    // SLEEF-inspired constants for exp computation - ultra high precision
    const __m256d ln2_inv = _mm256_set1_pd(1.4426950408889634073599246810019);  // 1/ln(2)
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180369123816490e+00);         // ln(2) high part
    const __m256d ln2_lo = _mm256_set1_pd(1.90821492927058770002e-10);          // ln(2) low part

    // Bounds for valid range.
    // exp_max: largest x for which exp(x) is finite (log of DBL_MAX).
    // exp_min: -708.0, chosen so that the range-reduction exponent n = round(x/ln2) satisfies
    //   n + 1023 >= 1 > 0, keeping the biased exponent non-negative for IEEE 754 bit manipulation.
    //   For x < -708, exp(x) < 2.2e-308 (smallest normal double), so clamping here is correct.
    //   Using -1000 was wrong: n + 1023 = -420 there, producing garbage from the bit shift.
    const __m256d exp_max = _mm256_set1_pd(709.782712893383996732223);
    const __m256d exp_min = _mm256_set1_pd(-708.0);

    // SLEEF polynomial coefficients for exp(s) where |s| < ln(2)/2
    // These provide < 1 ULP error for double precision
    const __m256d c1 = _mm256_set1_pd(0.1666666666666669072e+0);
    const __m256d c2 = _mm256_set1_pd(0.4166666666666602598e-1);
    const __m256d c3 = _mm256_set1_pd(0.8333333333314938210e-2);
    const __m256d c4 = _mm256_set1_pd(0.1388888888914497797e-2);
    const __m256d c5 = _mm256_set1_pd(0.1984126989855865850e-3);
    const __m256d c6 = _mm256_set1_pd(0.2480158687479686264e-4);
    const __m256d c7 = _mm256_set1_pd(0.2755723402025388239e-5);
    const __m256d c8 = _mm256_set1_pd(0.2755762628169491192e-6);
    const __m256d c9 = _mm256_set1_pd(0.2511210703042288022e-7);
    const __m256d c10 = _mm256_set1_pd(0.2081276378237164457e-8);

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Clamp to avoid overflow/underflow
        x = _mm256_min_pd(x, exp_max);
        x = _mm256_max_pd(x, exp_min);

        // Range reduction: x = n*ln(2) + r
        // n = round(x / ln(2))
        __m256d n_float = _mm256_mul_pd(x, ln2_inv);
        n_float = _mm256_round_pd(n_float, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        // Improved range reduction for higher precision
        // r = x - n*ln(2) using high-low decomposition
        __m256d r = _mm256_sub_pd(x, _mm256_mul_pd(n_float, ln2_hi));
        r = _mm256_sub_pd(r, _mm256_mul_pd(n_float, ln2_lo));

        // Compute exp(r) using SLEEF polynomial
        // exp(r) ≈ 1 + r + r²*(0.5 + r*P(r))
        // Using Horner's method for P(r)
        __m256d r2 = _mm256_mul_pd(r, r);

        // Evaluate polynomial P(r)
        __m256d poly = c10;
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c9);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c8);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c7);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c6);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c5);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c4);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c3);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c2);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, r), c1);

        // Complete: exp(r) = 1 + r + r²*(0.5 + r*P(r))
        poly = _mm256_mul_pd(poly, r);
        poly = _mm256_add_pd(poly, _mm256_set1_pd(0.5));
        poly = _mm256_mul_pd(poly, r2);
        poly = _mm256_add_pd(poly, r);
        poly = _mm256_add_pd(poly, _mm256_set1_pd(1.0));

        // Scale by 2^n
        // Convert n to integer for bit manipulation
        __m128i n_int_low = _mm256_cvtpd_epi32(n_float);

        // Create 2^n by manipulating exponent bits
        // Double precision: sign(1) | exponent(11) | mantissa(52)
        // 2^n = 1.0 * 2^n has exponent = 1023 + n
        __m128i bias = _mm_set1_epi32(1023);
        __m128i exp_bits = _mm_add_epi32(n_int_low, bias);

        // Shift to exponent position and convert back to double
        // Extract lower two elements [0,1] for conversion to 64-bit
        __m128i exp_bits_64_low = _mm_cvtepi32_epi64(exp_bits);
        // Extract upper two elements [2,3] by shuffling them to [0,1] first
        __m128i exp_bits_64_high = _mm_cvtepi32_epi64(_mm_shuffle_epi32(exp_bits, 0x0E));

        exp_bits_64_low = _mm_slli_epi64(exp_bits_64_low, 52);
        exp_bits_64_high = _mm_slli_epi64(exp_bits_64_high, 52);

        __m256d scale =
            _mm256_set_m128d(_mm_castsi128_pd(exp_bits_64_high), _mm_castsi128_pd(exp_bits_64_low));

        // Final result: exp(x) = exp(r) * 2^n
        __m256d result = _mm256_mul_pd(poly, scale);

        _mm256_storeu_pd(&output[i], result);
    }

    // Handle remaining elements with scalar fallback
    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::exp(input[i]);
    }
}

void VectorOps::vector_log_avx(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_log_fallback(input, output, size);
    }

    // Constants for log computation - exact SLEEF values
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d ln2_hi = _mm256_set1_pd(0.693147180559945286226764);        // ln(2) high part
    const __m256d ln2_lo = _mm256_set1_pd(2.319046813846299558417771e-17);    // ln(2) low part
    const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730950488016887242097);  // sqrt(2)
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d two = _mm256_set1_pd(2.0);

    // SLEEF exact polynomial coefficients from xlog_u1 for high accuracy
    // For the (m-1)/(m+1) transformation: log(m) = 2*atanh((m-1)/(m+1))
    const __m256d c1 = _mm256_set1_pd(0.6666666666667333541e+0);
    const __m256d c2 = _mm256_set1_pd(0.3999999999635251990e+0);
    const __m256d c3 = _mm256_set1_pd(0.2857142932794299317e+0);
    const __m256d c4 = _mm256_set1_pd(0.2222214519839380009e+0);
    const __m256d c5 = _mm256_set1_pd(0.1818605932937785996e+0);
    const __m256d c6 = _mm256_set1_pd(0.1525629051003428716e+0);
    const __m256d c7 = _mm256_set1_pd(0.1532076988502701353e+0);

    // Special value handling
    const __m256d zero = _mm256_setzero_pd();
    const __m256d neg_inf = _mm256_set1_pd(-std::numeric_limits<double>::infinity());
    const __m256d pos_inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Handle special cases: log(0) = -inf, log(+inf) = +inf, log(negative) = nan
        __m256d is_zero = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
        __m256d is_negative = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
        __m256d is_inf = _mm256_cmp_pd(x, pos_inf, _CMP_EQ_OQ);

        // SLEEF-inspired pure SIMD approach for exponent/mantissa extraction
        // Handle denormals by scaling
        const __m256d min_normal = _mm256_set1_pd(2.2250738585072014e-308);  // DBL_MIN
        __m256d is_denormal = _mm256_cmp_pd(x, min_normal, _CMP_LT_OQ);

        // Scale denormals by 2^54 to bring into normal range
        const __m256d scale_up = _mm256_set1_pd(18014398509481984.0);  // 2^54
        __m256d scaled_x = _mm256_blendv_pd(x, _mm256_mul_pd(x, scale_up), is_denormal);

        // Cast to integer for bit manipulation (process in two 128-bit halves)
        __m256i xi = _mm256_castpd_si256(scaled_x);

        // Extract the lower and upper 128-bit halves
        __m128i xi_low = _mm256_castsi256_si128(xi);
        __m128i xi_high = _mm256_extractf128_si256(xi, 1);

        // Extract exponent bits (bits 52-62) using shifts
        // Right shift by 52 to get exponent in lower bits
        __m128i exp_low = _mm_srli_epi64(xi_low, 52);
        __m128i exp_high = _mm_srli_epi64(xi_high, 52);

        // Mask to keep only exponent bits (11 bits)
        __m128i exp_mask = _mm_set1_epi64x(0x7FF);
        exp_low = _mm_and_si128(exp_low, exp_mask);
        exp_high = _mm_and_si128(exp_high, exp_mask);

        // Subtract bias (1023) to get actual exponent
        __m128i bias = _mm_set1_epi64x(1023);
        exp_low = _mm_sub_epi64(exp_low, bias);
        exp_high = _mm_sub_epi64(exp_high, bias);

        // Convert exponent to double
        // Need to convert 64-bit integers to doubles properly
        // Since we have 64-bit values in exp_low/exp_high, we need different conversion
        alignas(16) int64_t exp_low_arr[2];
        alignas(16) int64_t exp_high_arr[2];
        _mm_store_si128(reinterpret_cast<__m128i*>(exp_low_arr), exp_low);
        _mm_store_si128(reinterpret_cast<__m128i*>(exp_high_arr), exp_high);

        __m128d exp_low_d =
            _mm_set_pd(static_cast<double>(exp_low_arr[1]), static_cast<double>(exp_low_arr[0]));
        __m128d exp_high_d =
            _mm_set_pd(static_cast<double>(exp_high_arr[1]), static_cast<double>(exp_high_arr[0]));
        __m256d e = _mm256_set_m128d(exp_high_d, exp_low_d);

        // Adjust exponent for denormals
        e = _mm256_blendv_pd(e, _mm256_sub_pd(e, _mm256_set1_pd(54.0)), is_denormal);

        // Create mantissa by setting exponent to bias (1023)
        // Clear exponent bits and set to 0x3FF (1023)
        __m128i mantissa_mask = _mm_set1_epi64x(0x000FFFFFFFFFFFFF);
        __m128i mantissa_low = _mm_and_si128(xi_low, mantissa_mask);
        __m128i mantissa_high = _mm_and_si128(xi_high, mantissa_mask);

        __m128i exp_bias = _mm_set1_epi64x(0x3FF0000000000000);
        mantissa_low = _mm_or_si128(mantissa_low, exp_bias);
        mantissa_high = _mm_or_si128(mantissa_high, exp_bias);

        // Convert back to double (mantissa in [1, 2) range)
        __m128d m_low = _mm_castsi128_pd(mantissa_low);
        __m128d m_high = _mm_castsi128_pd(mantissa_high);
        __m256d m = _mm256_set_m128d(m_high, m_low);

        // Range reduction: if m > sqrt(2), divide by 2 and increment exponent
        __m256d needs_adjust = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);
        m = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), needs_adjust);
        e = _mm256_blendv_pd(e, _mm256_add_pd(e, one), needs_adjust);

        // Compute log(m) using SLEEF's (m-1)/(m+1) transformation
        // where xr = (m-1)/(m+1), log(m) = 2*xr + xr³*t where t is polynomial
        __m256d mp1 = _mm256_add_pd(m, one);
        __m256d mm1 = _mm256_sub_pd(m, one);
        __m256d xr = _mm256_div_pd(mm1, mp1);
        __m256d xr2 = _mm256_mul_pd(xr, xr);

        // Evaluate polynomial using exact SLEEF coefficients and order
        // t = c7 + xr²*(c6 + xr²*(c5 + xr²*(c4 + xr²*(c3 + xr²*(c2 + xr²*c1)))))
        __m256d t = c7;
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c6);
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c5);
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c4);
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c3);
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c2);
        t = _mm256_add_pd(_mm256_mul_pd(t, xr2), c1);

        // Compute log(m) = 2*xr + xr³*t
        __m256d xr3 = _mm256_mul_pd(xr, xr2);
        __m256d log_m = _mm256_mul_pd(xr3, t);
        log_m = _mm256_add_pd(_mm256_mul_pd(xr, two), log_m);

        // Final result: log(x) = log(m) + e*ln(2)
        // Use high-low decomposition for better accuracy
        __m256d result = _mm256_mul_pd(e, ln2_hi);
        result = _mm256_add_pd(result, _mm256_mul_pd(e, ln2_lo));
        result = _mm256_add_pd(result, log_m);

        // Handle special cases: log(0) = -inf, log(+inf) = +inf, log(negative) = nan
        result = _mm256_blendv_pd(result, neg_inf, is_zero);
        result = _mm256_blendv_pd(result, pos_inf, is_inf);
        result = _mm256_blendv_pd(result, _mm256_set1_pd(std::numeric_limits<double>::quiet_NaN()),
                                  is_negative);

        _mm256_storeu_pd(&output[i], result);
    }

    // Handle remaining elements with scalar fallback
    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::log(input[i]);
    }
}

// Scalar exponent version (matches header signature)
void VectorOps::vector_pow_avx(const double* base, double exponent, double* output,
                               std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_pow_fallback(base, exponent, output, size);
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    // For pow(x, y) = exp(y * log(x))
    // We'll use our SIMD log and exp functions
    alignas(32) double temp_log[AVX_DOUBLE_WIDTH];
    alignas(32) double temp_mul[AVX_DOUBLE_WIDTH];

    __m256d exp_val = _mm256_set1_pd(exponent);  // Broadcast scalar exponent

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        // Step 1: Compute log(base)
        vector_log_avx(&base[i], temp_log, AVX_DOUBLE_WIDTH);

        // Step 2: Multiply by exponent
        __m256d log_val = _mm256_load_pd(temp_log);
        __m256d product = _mm256_mul_pd(log_val, exp_val);
        _mm256_store_pd(temp_mul, product);

        // Step 3: Compute exp(y * log(x))
        vector_exp_avx(temp_mul, &output[i], AVX_DOUBLE_WIDTH);

        // Handle special cases
        __m256d base_val = _mm256_loadu_pd(&base[i]);
        __m256d result = _mm256_loadu_pd(&output[i]);

        // pow(0, y) = 0 for y > 0, 1 for y = 0, inf for y < 0
        __m256d zero = _mm256_setzero_pd();
        __m256d one = _mm256_set1_pd(1.0);
        __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

        __m256d base_is_zero = _mm256_cmp_pd(base_val, zero, _CMP_EQ_OQ);
        __m256d exp_is_zero = _mm256_cmp_pd(exp_val, zero, _CMP_EQ_OQ);
        __m256d exp_is_positive = _mm256_cmp_pd(exp_val, zero, _CMP_GT_OQ);

        // If base is 0
        result = _mm256_blendv_pd(result, one, _mm256_and_pd(base_is_zero, exp_is_zero));
        result = _mm256_blendv_pd(result, zero, _mm256_and_pd(base_is_zero, exp_is_positive));
        result = _mm256_blendv_pd(
            result, inf, _mm256_and_pd(base_is_zero, _mm256_cmp_pd(exp_val, zero, _CMP_LT_OQ)));

        // pow(1, y) = 1 for any y
        __m256d base_is_one = _mm256_cmp_pd(base_val, one, _CMP_EQ_OQ);
        result = _mm256_blendv_pd(result, one, base_is_one);

        // pow(x, 0) = 1 for any x != 0
        result = _mm256_blendv_pd(result, one, exp_is_zero);

        _mm256_storeu_pd(&output[i], result);
    }

    // Handle remaining elements with scalar fallback
    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::pow(base[i], exponent);
    }
}

// Element-wise power version (NEW - for vector exponents)
void VectorOps::vector_pow_elementwise_avx(const double* base, const double* exponent,
                                           double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        // Fallback to scalar implementation
        for (std::size_t i = 0; i < size; ++i) {
            output[i] = std::pow(base[i], exponent[i]);
        }
        return;
    }

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    // Temporary buffers for intermediate results
    alignas(32) double temp_log[AVX_DOUBLE_WIDTH];
    alignas(32) double temp_mul[AVX_DOUBLE_WIDTH];

    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d inf = _mm256_set1_pd(std::numeric_limits<double>::infinity());

    // For pow(x, y) = exp(y * log(x))
    // Use high-accuracy log and exp functions
    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        // Step 1: Compute log(base) using our high-accuracy function
        vector_log_avx(&base[i], temp_log, AVX_DOUBLE_WIDTH);

        // Step 2: Multiply log(base) by exponent
        __m256d exp_vec = _mm256_loadu_pd(&exponent[i]);
        __m256d log_vec = _mm256_load_pd(temp_log);
        __m256d product = _mm256_mul_pd(exp_vec, log_vec);
        _mm256_store_pd(temp_mul, product);

        // Step 3: Compute exp(y * log(x)) using our high-accuracy function
        vector_exp_avx(temp_mul, &output[i], AVX_DOUBLE_WIDTH);

        // Load results and inputs for special case handling
        __m256d base_vec = _mm256_loadu_pd(&base[i]);
        __m256d result = _mm256_loadu_pd(&output[i]);

        // === Step 4: Handle special cases for pow ===

        // Check base conditions
        __m256d base_is_zero = _mm256_cmp_pd(base_vec, zero, _CMP_EQ_OQ);
        __m256d base_is_one = _mm256_cmp_pd(base_vec, one, _CMP_EQ_OQ);

        // Check exponent conditions
        __m256d exp_is_zero = _mm256_cmp_pd(exp_vec, zero, _CMP_EQ_OQ);
        __m256d exp_is_positive = _mm256_cmp_pd(exp_vec, zero, _CMP_GT_OQ);
        __m256d exp_is_negative = _mm256_cmp_pd(exp_vec, zero, _CMP_LT_OQ);

        // pow(0, y) = 0 for y > 0, 1 for y = 0, inf for y < 0
        result = _mm256_blendv_pd(result, one, _mm256_and_pd(base_is_zero, exp_is_zero));
        result = _mm256_blendv_pd(result, zero, _mm256_and_pd(base_is_zero, exp_is_positive));
        result = _mm256_blendv_pd(result, inf, _mm256_and_pd(base_is_zero, exp_is_negative));

        // pow(1, y) = 1 for any y
        result = _mm256_blendv_pd(result, one, base_is_one);

        // pow(x, 0) = 1 for any x != 0
        result = _mm256_blendv_pd(result, one, _mm256_andnot_pd(base_is_zero, exp_is_zero));

        _mm256_storeu_pd(&output[i], result);
    }

    // Handle remaining elements with scalar fallback
    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::pow(base[i], exponent[i]);
    }
}

void VectorOps::vector_erf_avx(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_erf_fallback(input, output, size);
    }

    // Abramowitz & Stegun 7.1.26 approximation
    // Maximum error: 1.5×10^−7
    // erf(x) = 1 - 1/(1 + p*|x|)^n * exp(-x²) * P(t) for x ≥ 0

    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);

    // Coefficients for Abramowitz & Stegun approximation
    const __m256d a1 = _mm256_set1_pd(0.254829592);
    const __m256d a2 = _mm256_set1_pd(-0.284496736);
    const __m256d a3 = _mm256_set1_pd(1.421413741);
    const __m256d a4 = _mm256_set1_pd(-1.453152027);
    const __m256d a5 = _mm256_set1_pd(1.061405429);
    const __m256d p = _mm256_set1_pd(0.3275911);

    // For very small x approximation: erf(x) ≈ (2/sqrt(pi)) * x
    const __m256d two_over_sqrtpi = _mm256_set1_pd(1.12837916709551262756245475959);
    const __m256d thresh_small = _mm256_set1_pd(1e-8);
    const __m256d thresh_large = _mm256_set1_pd(6.0);

    constexpr std::size_t AVX_DOUBLE_WIDTH = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / AVX_DOUBLE_WIDTH) * AVX_DOUBLE_WIDTH;

    for (std::size_t i = 0; i < simd_end; i += AVX_DOUBLE_WIDTH) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Save sign and compute absolute value
        __m256d sign = _mm256_and_pd(x, sign_mask);
        __m256d abs_x = _mm256_andnot_pd(sign_mask, x);

        // Check for special cases
        __m256d is_small = _mm256_cmp_pd(abs_x, thresh_small, _CMP_LT_OQ);
        __m256d is_large = _mm256_cmp_pd(abs_x, thresh_large, _CMP_GE_OQ);

        // Compute t = 1 / (1 + p * |x|)
        __m256d t = _mm256_add_pd(one, _mm256_mul_pd(p, abs_x));
        t = _mm256_div_pd(one, t);

        // Evaluate polynomial using Horner's method
        // poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
        __m256d poly = a5;
        poly = _mm256_add_pd(a4, _mm256_mul_pd(t, poly));
        poly = _mm256_add_pd(a3, _mm256_mul_pd(t, poly));
        poly = _mm256_add_pd(a2, _mm256_mul_pd(t, poly));
        poly = _mm256_add_pd(a1, _mm256_mul_pd(t, poly));
        poly = _mm256_mul_pd(t, poly);

        // Compute exp(-x^2)
        __m256d x2 = _mm256_mul_pd(abs_x, abs_x);
        __m256d neg_x2 = _mm256_sub_pd(_mm256_setzero_pd(), x2);

        // Call our exp implementation
        alignas(32) double exp_input[AVX_DOUBLE_WIDTH];
        alignas(32) double exp_result[AVX_DOUBLE_WIDTH];
        _mm256_store_pd(exp_input, neg_x2);
        vector_exp_avx(exp_input, exp_result, AVX_DOUBLE_WIDTH);
        __m256d exp_neg_x2 = _mm256_load_pd(exp_result);

        // erf(|x|) = 1 - poly * exp(-x^2)
        __m256d result = _mm256_sub_pd(one, _mm256_mul_pd(poly, exp_neg_x2));

        // For very small |x|, use linear approximation
        __m256d small_result = _mm256_mul_pd(abs_x, two_over_sqrtpi);
        result = _mm256_blendv_pd(result, small_result, is_small);

        // For large |x| >= 6, erf(x) = 1
        result = _mm256_blendv_pd(result, one, is_large);

        // Apply sign
        result = _mm256_or_pd(result, sign);

        _mm256_storeu_pd(&output[i], result);
    }

    // Handle remaining elements with scalar fallback
    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::erf(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats

#ifdef __clang__
    #pragma clang attribute pop
#endif
