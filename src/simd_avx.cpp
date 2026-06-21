#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
// AVX-specific SIMD implementations
// This file is compiled ONLY with AVX flags to ensure safety
//
// Some algorithms and polynomial coefficients in this file are inspired by or
// derived from the SLEEF library (https://github.com/shibatch/sleef), which is
// licensed under the Boost Software License 1.0. The Boost license is fully
// compatible with our MIT License.

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

    // Four-region rational polynomial approximation derived from musl libc erf.c
    // (origin: Sun Microsystems / FreeBSD s_erf.c).
    // Error: < 1 ULP throughout (2^-57.9 in R1, 2^-59.1 in R2, 2^-62.6 in R3, 2^-61.5 in R4).
    // All regions evaluated for every element; results blended by region mask.
    // simd_avx.cpp is compiled with -mavx only (no FMA); uses mul+add pairs.
    //
    // Region 1: |x| < 0.84375  — rational P(x²)/Q(x²),  erf(x) = x + x·R
    // Region 2: 0.84375 ≤ |x| < 1.25  — rational around x=1,  erf(x) = erx + P(s)/Q(s)
    // Region 3: 1.25 ≤ |x| < 2.857  — erfc via exp(-x²-0.5625+R/S)/x
    // Region 4: 2.857 ≤ |x| < 6     — same structure, different coefficients
    // Region 5: |x| ≥ 6             — erf ≈ ±1

    // ---- Region 1 coefficients (rational P/Q in z = x²) ----
    const __m256d pp0 = _mm256_set1_pd( 1.28379167095512558561e-01);
    const __m256d pp1 = _mm256_set1_pd(-3.25042107247001499370e-01);
    const __m256d pp2 = _mm256_set1_pd(-2.84817495755985104766e-02);
    const __m256d pp3 = _mm256_set1_pd(-5.77027029648944159157e-03);
    const __m256d pp4 = _mm256_set1_pd(-2.37630166566501626084e-05);
    const __m256d qq1 = _mm256_set1_pd( 3.97917223959155352819e-01);
    const __m256d qq2 = _mm256_set1_pd( 6.50222499887672944485e-02);
    const __m256d qq3 = _mm256_set1_pd( 5.08130628187576562776e-03);
    const __m256d qq4 = _mm256_set1_pd( 1.32494738004321644526e-04);
    const __m256d qq5 = _mm256_set1_pd(-3.96022827877536812320e-06);

    // ---- Region 2 coefficients (rational P/Q in s = |x|-1) ----
    const __m256d erx = _mm256_set1_pd( 8.45062911510467529297e-01); // erf(~0.84375) rounded to float24
    const __m256d pa0 = _mm256_set1_pd(-2.36211856075265944077e-03);
    const __m256d pa1 = _mm256_set1_pd( 4.14856118683748331666e-01);
    const __m256d pa2 = _mm256_set1_pd(-3.72207876035701323847e-01);
    const __m256d pa3 = _mm256_set1_pd( 3.18346619901161753674e-01);
    const __m256d pa4 = _mm256_set1_pd(-1.10894694282396677476e-01);
    const __m256d pa5 = _mm256_set1_pd( 3.54783043256182359371e-02);
    const __m256d pa6 = _mm256_set1_pd(-2.16637559486879084300e-03);
    const __m256d qa1 = _mm256_set1_pd( 1.06420880400844228286e-01);
    const __m256d qa2 = _mm256_set1_pd( 5.40397917702171048937e-01);
    const __m256d qa3 = _mm256_set1_pd( 7.18286544141962662868e-02);
    const __m256d qa4 = _mm256_set1_pd( 1.26171219808761642112e-01);
    const __m256d qa5 = _mm256_set1_pd( 1.36370839120290507362e-02);
    const __m256d qa6 = _mm256_set1_pd( 1.19844998467991074170e-02);

    // ---- Region 3 coefficients (rational R/S in s = 1/x², 1.25 ≤ |x| < 2.857) ----
    const __m256d ra0 = _mm256_set1_pd(-9.86494403484714822705e-03);
    const __m256d ra1 = _mm256_set1_pd(-6.93858572707181764372e-01);
    const __m256d ra2 = _mm256_set1_pd(-1.05586262253232909814e+01);
    const __m256d ra3 = _mm256_set1_pd(-6.23753324503260060396e+01);
    const __m256d ra4 = _mm256_set1_pd(-1.62396669462573470355e+02);
    const __m256d ra5 = _mm256_set1_pd(-1.84605092906711035994e+02);
    const __m256d ra6 = _mm256_set1_pd(-8.12874355063065934246e+01);
    const __m256d ra7 = _mm256_set1_pd(-9.81432934416914548592e+00);
    const __m256d sa1 = _mm256_set1_pd( 1.96512716674392571292e+01);
    const __m256d sa2 = _mm256_set1_pd( 1.37657754143519042600e+02);
    const __m256d sa3 = _mm256_set1_pd( 4.34565877475229228821e+02);
    const __m256d sa4 = _mm256_set1_pd( 6.45387271733267880336e+02);
    const __m256d sa5 = _mm256_set1_pd( 4.29008140027567833386e+02);
    const __m256d sa6 = _mm256_set1_pd( 1.08635005541779435134e+02);
    const __m256d sa7 = _mm256_set1_pd( 6.57024977031928170135e+00);
    const __m256d sa8 = _mm256_set1_pd(-6.04244152148580987438e-02);

    // ---- Region 4 coefficients (rational R/S in s = 1/x², 2.857 ≤ |x| < 6) ----
    const __m256d rb0 = _mm256_set1_pd(-9.86494292470009928597e-03);
    const __m256d rb1 = _mm256_set1_pd(-7.99283237680523006574e-01);
    const __m256d rb2 = _mm256_set1_pd(-1.77579549177547519889e+01);
    const __m256d rb3 = _mm256_set1_pd(-1.60636384855821916062e+02);
    const __m256d rb4 = _mm256_set1_pd(-6.37566443368389627722e+02);
    const __m256d rb5 = _mm256_set1_pd(-1.02509513161107724954e+03);
    const __m256d rb6 = _mm256_set1_pd(-4.83519191608651397019e+02);
    const __m256d sb1 = _mm256_set1_pd( 3.03380607434824582924e+01);
    const __m256d sb2 = _mm256_set1_pd( 3.25792512996573918826e+02);
    const __m256d sb3 = _mm256_set1_pd( 1.53672958608443695994e+03);
    const __m256d sb4 = _mm256_set1_pd( 3.19985821950859553908e+03);
    const __m256d sb5 = _mm256_set1_pd( 2.55305040643316442583e+03);
    const __m256d sb6 = _mm256_set1_pd( 4.74528541206955367215e+02);
    const __m256d sb7 = _mm256_set1_pd(-2.24409524465858183362e+01);

    const __m256d one       = _mm256_set1_pd(1.0);
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d t1        = _mm256_set1_pd(0.84375);     // R1 / R2 boundary
    const __m256d t2        = _mm256_set1_pd(1.25);        // R2 / R3 boundary
    const __m256d t3        = _mm256_set1_pd(2.857142857); // R3 / R4 boundary (= 1/0.35)
    const __m256d t5        = _mm256_set1_pd(6.0);         // R4 / R5 boundary
    const __m256d c0p5625   = _mm256_set1_pd(0.5625);

    constexpr std::size_t W = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / W) * W;
    alignas(32) double exp_buf[W];  // temp buffer for exp(-x²-0.5625+R/S)

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x     = _mm256_loadu_pd(&input[i]);
        __m256d sign  = _mm256_and_pd(x, sign_mask);
        __m256d ax    = _mm256_andnot_pd(sign_mask, x);  // |x|

        // Region masks
        __m256d m1 = _mm256_cmp_pd(ax, t1, _CMP_LT_OQ);  // |x| < 0.84375
        __m256d m2 = _mm256_cmp_pd(ax, t2, _CMP_LT_OQ);  // |x| < 1.25
        __m256d m3 = _mm256_cmp_pd(ax, t3, _CMP_LT_OQ);  // |x| < 2.857

        // ---- Region 1: erf(x) = x + x·P(z)/Q(z),  z = x² ----
        __m256d z = _mm256_mul_pd(ax, ax);
        __m256d P1 = pp4;
        P1 = _mm256_add_pd(pp3, _mm256_mul_pd(z, P1));
        P1 = _mm256_add_pd(pp2, _mm256_mul_pd(z, P1));
        P1 = _mm256_add_pd(pp1, _mm256_mul_pd(z, P1));
        P1 = _mm256_add_pd(pp0, _mm256_mul_pd(z, P1));
        __m256d Q1 = qq5;
        Q1 = _mm256_add_pd(qq4, _mm256_mul_pd(z, Q1));
        Q1 = _mm256_add_pd(qq3, _mm256_mul_pd(z, Q1));
        Q1 = _mm256_add_pd(qq2, _mm256_mul_pd(z, Q1));
        Q1 = _mm256_add_pd(qq1, _mm256_mul_pd(z, Q1));
        Q1 = _mm256_add_pd(one, _mm256_mul_pd(z, Q1));
        // r1 = ax + ax*(P1/Q1) = ax*(1 + P1/Q1)
        __m256d r1 = _mm256_add_pd(ax, _mm256_mul_pd(ax, _mm256_div_pd(P1, Q1)));

        // ---- Region 2: erf(x) = erx + P(s)/Q(s),  s = |x|-1 ----
        __m256d s2 = _mm256_sub_pd(ax, one);
        __m256d P2 = pa6;
        P2 = _mm256_add_pd(pa5, _mm256_mul_pd(s2, P2));
        P2 = _mm256_add_pd(pa4, _mm256_mul_pd(s2, P2));
        P2 = _mm256_add_pd(pa3, _mm256_mul_pd(s2, P2));
        P2 = _mm256_add_pd(pa2, _mm256_mul_pd(s2, P2));
        P2 = _mm256_add_pd(pa1, _mm256_mul_pd(s2, P2));
        P2 = _mm256_add_pd(pa0, _mm256_mul_pd(s2, P2));
        __m256d Q2 = qa6;
        Q2 = _mm256_add_pd(qa5, _mm256_mul_pd(s2, Q2));
        Q2 = _mm256_add_pd(qa4, _mm256_mul_pd(s2, Q2));
        Q2 = _mm256_add_pd(qa3, _mm256_mul_pd(s2, Q2));
        Q2 = _mm256_add_pd(qa2, _mm256_mul_pd(s2, Q2));
        Q2 = _mm256_add_pd(qa1, _mm256_mul_pd(s2, Q2));
        Q2 = _mm256_add_pd(one, _mm256_mul_pd(s2, Q2));
        __m256d r2 = _mm256_add_pd(erx, _mm256_div_pd(P2, Q2));

        // ---- Regions 3-4: erfc = exp(-x²-0.5625+R/S)/|x|,  erf = 1-erfc ----
        // Clamp |x| to [1.25, ∞) so 1/x² is safe for all lanes (R1/R2 lanes blended away later).
        __m256d sax = _mm256_max_pd(ax, t2);
        __m256d inv_x2 = _mm256_div_pd(one, _mm256_mul_pd(sax, sax));  // s = 1/x²

        // Region 3 R polynomial: R3 = ra0 + s*(ra1 + ... + s*ra7)
        __m256d R3 = ra7;
        R3 = _mm256_add_pd(ra6, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra5, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra4, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra3, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra2, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra1, _mm256_mul_pd(inv_x2, R3));
        R3 = _mm256_add_pd(ra0, _mm256_mul_pd(inv_x2, R3));

        // Region 3 S polynomial: S3 = 1 + s*(sa1 + ... + s*sa8)
        __m256d S3 = sa8;
        S3 = _mm256_add_pd(sa7, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa6, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa5, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa4, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa3, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa2, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(sa1, _mm256_mul_pd(inv_x2, S3));
        S3 = _mm256_add_pd(one, _mm256_mul_pd(inv_x2, S3));

        // Region 4 R polynomial: R4 = rb0 + s*(rb1 + ... + s*rb6)
        __m256d R4 = rb6;
        R4 = _mm256_add_pd(rb5, _mm256_mul_pd(inv_x2, R4));
        R4 = _mm256_add_pd(rb4, _mm256_mul_pd(inv_x2, R4));
        R4 = _mm256_add_pd(rb3, _mm256_mul_pd(inv_x2, R4));
        R4 = _mm256_add_pd(rb2, _mm256_mul_pd(inv_x2, R4));
        R4 = _mm256_add_pd(rb1, _mm256_mul_pd(inv_x2, R4));
        R4 = _mm256_add_pd(rb0, _mm256_mul_pd(inv_x2, R4));

        // Region 4 S polynomial: S4 = 1 + s*(sb1 + ... + s*sb7)
        __m256d S4 = sb7;
        S4 = _mm256_add_pd(sb6, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(sb5, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(sb4, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(sb3, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(sb2, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(sb1, _mm256_mul_pd(inv_x2, S4));
        S4 = _mm256_add_pd(one, _mm256_mul_pd(inv_x2, S4));

        // Blend R/S: use Region 3 coefficients where |x| < 2.857, Region 4 otherwise
        __m256d RS = _mm256_div_pd(_mm256_blendv_pd(R4, R3, m3),
                                   _mm256_blendv_pd(S4, S3, m3));

        // exp_arg = -x² - 0.5625 + R/S  (equivalent to musl's two-exp decomposition)
        __m256d exp_arg = _mm256_sub_pd(_mm256_sub_pd(RS, c0p5625),
                                        _mm256_mul_pd(sax, sax));
        // Clamp to ≤ 0 to prevent overflow (erfc is always ≤ 1 for |x| ≥ 1.25)
        exp_arg = _mm256_min_pd(exp_arg, _mm256_setzero_pd());

        _mm256_store_pd(exp_buf, exp_arg);
        vector_exp_avx(exp_buf, exp_buf, W);
        __m256d exp_val = _mm256_load_pd(exp_buf);

        __m256d r34 = _mm256_sub_pd(one, _mm256_div_pd(exp_val, sax));

        // ---- Blend regions (innermost wins) ----
        __m256d result = one;                                          // R5: |x| >= 6 -> 1
        result = _mm256_blendv_pd(result, r34,
            _mm256_andnot_pd(m2, _mm256_cmp_pd(ax, t5, _CMP_LT_OQ))); // R3+R4: 1.25 <= |x| < 6
        result = _mm256_blendv_pd(result, r2,
            _mm256_andnot_pd(m1, m2));                                 // R2: 0.84375 <= |x| < 1.25
        result = _mm256_blendv_pd(result, r1, m1);                    // R1: |x| < 0.84375

        // Propagate NaN
        __m256d nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
        result = _mm256_blendv_pd(result, x, nan_mask);

        // Restore sign (erf is odd)
        result = _mm256_or_pd(result, sign);

        _mm256_storeu_pd(&output[i], result);
    }

    for (std::size_t i = simd_end; i < size; ++i) output[i] = std::erf(input[i]);
}

void VectorOps::vector_cos_avx(const double* input, double* output, std::size_t size) noexcept {
    if (!stats::arch::supports_avx()) {
        return vector_cos_fallback(input, output, size);
    }

    // Two-step range reduction + 7-term Horner Taylor polynomial.
    // Step 1: y = x - round(x/2π)·2π  →  y ∈ [−π, π]
    // Step 2: if |y| > π/2, reflect and flip sign  →  y ∈ [−π/2, π/2]
    // Step 3: cos(y) ≈ 1 + y²·(c₁ + y²·(c₂ + … + y²·c₇))
    // Max error ≈ 1×10⁻¹⁰ for |y| ≤ π/2. Scalar tail uses std::cos.

    constexpr std::size_t W = arch::simd::AVX_DOUBLES;
    const std::size_t simd_end = (size / W) * W;

    const __m256d inv_two_pi  = _mm256_set1_pd(1.0 / (2.0 * detail::PI));
    const __m256d two_pi      = _mm256_set1_pd(2.0 * detail::PI);
    const __m256d pi          = _mm256_set1_pd(detail::PI);
    const __m256d half_pi     = _mm256_set1_pd(detail::PI_OVER_2);
    const __m256d neg_pi      = _mm256_set1_pd(-detail::PI);
    const __m256d neg_half_pi = _mm256_set1_pd(-detail::PI_OVER_2);
    const __m256d one         = _mm256_set1_pd(1.0);
    const __m256d neg_one     = _mm256_set1_pd(-1.0);

    // Taylor coefficients: c_k = (-1)^k / (2k)!
    const __m256d c1 = _mm256_set1_pd(-0.5);                     // -1/2!
    const __m256d c2 = _mm256_set1_pd(4.166666666666667e-2);     //  1/4!
    const __m256d c3 = _mm256_set1_pd(-1.388888888888889e-3);    // -1/6!
    const __m256d c4 = _mm256_set1_pd(2.480158730158730e-5);     //  1/8!
    const __m256d c5 = _mm256_set1_pd(-2.755731922398589e-7);    // -1/10!
    const __m256d c6 = _mm256_set1_pd(2.087675698786810e-9);     //  1/12!
    const __m256d c7 = _mm256_set1_pd(-1.147074559772973e-11);   // -1/14!

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m256d x = _mm256_loadu_pd(&input[i]);

        // Step 1: reduce to [−π, π]
        __m256d q = _mm256_round_pd(_mm256_mul_pd(x, inv_two_pi),
                                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256d y = _mm256_sub_pd(x, _mm256_mul_pd(q, two_pi));

        // Step 2: reduce to [−π/2, π/2], tracking sign
        __m256d sign    = one;
        __m256d gt_hpi  = _mm256_cmp_pd(y, half_pi,     _CMP_GT_OQ);
        __m256d lt_nhpi = _mm256_cmp_pd(y, neg_half_pi, _CMP_LT_OQ);

        y    = _mm256_blendv_pd(y,    _mm256_sub_pd(pi,     y), gt_hpi);
        sign = _mm256_blendv_pd(sign, neg_one,                  gt_hpi);
        y    = _mm256_blendv_pd(y,    _mm256_sub_pd(neg_pi, y), lt_nhpi);
        sign = _mm256_blendv_pd(sign, neg_one,                  lt_nhpi);

        // Step 3: Horner evaluation  cos(y) = 1 + y²·(c₁ + y²·(… + y²·c₇))
        __m256d y2   = _mm256_mul_pd(y, y);
        __m256d poly = c7;
        poly = _mm256_add_pd(c6, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(c5, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(c4, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(c3, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(c2, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(c1, _mm256_mul_pd(y2, poly));
        poly = _mm256_add_pd(one, _mm256_mul_pd(y2, poly));

        _mm256_storeu_pd(&output[i], _mm256_mul_pd(poly, sign));
    }

    for (std::size_t i = simd_end; i < size; ++i) {
        output[i] = std::cos(input[i]);
    }
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
