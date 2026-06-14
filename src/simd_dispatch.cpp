// Main SIMD dispatch logic — NO SIMD intrinsics in this file.
// The dispatch table (makeDispatchTable) is the single change point for tier selection:
//   Adding a new SIMD tier: edit makeDispatchTable() only.
//   Adding a new op: add a pointer to DispatchTable (simd.h), set it in
//   makeDispatchTable(), add a 2-line public method in the section below.

#include "libstats/core/math_constants.h"
#include "libstats/core/statistical_constants.h"
#include "libstats/platform/cpu_detection.h"
#include "libstats/platform/platform_constants.h"
#include "libstats/platform/simd.h"
#include "libstats/platform/simd_policy.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace stats {
namespace simd {
namespace ops {

//==============================================================================
// makeDispatchTable — selects the best available SIMD tier at process startup.
// Tiers are tested best-first; the first available tier wins and is returned.
// The #ifdef guards remain here (not in the call sites) because the backend
// functions are conditionally compiled. This is the ONLY function that needs
// editing when a new SIMD tier is added to the library.
//==============================================================================

VectorOps::DispatchTable VectorOps::makeDispatchTable() noexcept {
    DispatchTable t{};

    // Tier: Fallback (always available) — populated first as the safe default.
    t.dot_product            = dot_product_fallback;
    t.vector_add             = vector_add_fallback;
    t.vector_subtract        = vector_subtract_fallback;
    t.vector_multiply        = vector_multiply_fallback;
    t.scalar_multiply        = scalar_multiply_fallback;
    t.scalar_add             = scalar_add_fallback;
    t.vector_exp             = vector_exp_fallback;
    t.vector_log             = vector_log_fallback;
    t.vector_pow             = vector_pow_fallback;
    t.vector_pow_elementwise = vector_pow_elementwise_fallback;
    t.vector_erf             = vector_erf_fallback;

#ifdef LIBSTATS_HAS_NEON
    if (stats::arch::supports_neon()) {
        t.dot_product            = dot_product_neon;
        t.vector_add             = vector_add_neon;
        t.vector_subtract        = vector_subtract_neon;
        t.vector_multiply        = vector_multiply_neon;
        t.scalar_multiply        = scalar_multiply_neon;
        t.scalar_add             = scalar_add_neon;
        t.vector_exp             = vector_exp_neon;
        t.vector_log             = vector_log_neon;
        t.vector_pow             = vector_pow_neon;
        t.vector_pow_elementwise = vector_pow_elementwise_neon;
        t.vector_erf             = vector_erf_neon;
        return t;  // ARM: NEON is the only SIMD tier
    }
#endif

    // x86 tiers: tested best-first; each overwrites the previous on success.
    // AVX512 ⊃ AVX2 ⊃ AVX ⊃ SSE2, so testing from worst to best and
    // overwriting means we always land on the highest available tier.
#ifdef LIBSTATS_HAS_SSE2
    if (stats::arch::supports_sse2()) {
        t.dot_product            = dot_product_sse2;
        t.vector_add             = vector_add_sse2;
        t.vector_subtract        = vector_subtract_sse2;
        t.vector_multiply        = vector_multiply_sse2;
        t.scalar_multiply        = scalar_multiply_sse2;
        t.scalar_add             = scalar_add_sse2;
        t.vector_exp             = vector_exp_sse2;
        t.vector_log             = vector_log_sse2;
        t.vector_pow             = vector_pow_sse2;
        t.vector_pow_elementwise = vector_pow_elementwise_sse2;
        t.vector_erf             = vector_erf_sse2;
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (stats::arch::supports_avx()) {
        t.dot_product            = dot_product_avx;
        t.vector_add             = vector_add_avx;
        t.vector_subtract        = vector_subtract_avx;
        t.vector_multiply        = vector_multiply_avx;
        t.scalar_multiply        = scalar_multiply_avx;
        t.scalar_add             = scalar_add_avx;
        t.vector_exp             = vector_exp_avx;
        t.vector_log             = vector_log_avx;
        t.vector_pow             = vector_pow_avx;
        t.vector_pow_elementwise = vector_pow_elementwise_avx;
        t.vector_erf             = vector_erf_avx;
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (stats::arch::supports_avx2()) {
        t.dot_product            = dot_product_avx2;
        t.vector_add             = vector_add_avx2;
        t.vector_subtract        = vector_subtract_avx2;
        t.vector_multiply        = vector_multiply_avx2;
        t.scalar_multiply        = scalar_multiply_avx2;
        t.scalar_add             = scalar_add_avx2;
        t.vector_exp             = vector_exp_avx2;
        t.vector_log             = vector_log_avx2;
        t.vector_pow             = vector_pow_avx2;
        t.vector_pow_elementwise = vector_pow_elementwise_avx2;
        t.vector_erf             = vector_erf_avx2;
    }
#endif

#ifdef LIBSTATS_HAS_AVX512
    if (stats::arch::supports_avx512()) {
        t.dot_product            = dot_product_avx512;
        t.vector_add             = vector_add_avx512;
        t.vector_subtract        = vector_subtract_avx512;
        t.vector_multiply        = vector_multiply_avx512;
        t.scalar_multiply        = scalar_multiply_avx512;
        t.scalar_add             = scalar_add_avx512;
        t.vector_exp             = vector_exp_avx512;
        t.vector_log             = vector_log_avx512;
        t.vector_pow             = vector_pow_avx512;
        t.vector_pow_elementwise = vector_pow_elementwise_avx512;
        t.vector_erf             = vector_erf_avx512;
    }
#endif

    return t;
}

const VectorOps::DispatchTable& VectorOps::getDispatchTable() noexcept {
    static const DispatchTable table = makeDispatchTable();
    return table;
}

//==============================================================================
// Public interface — each method: size check + one dispatch table call.
//==============================================================================

double VectorOps::dot_product(const double* a, const double* b, std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return dot_product_fallback(a, b, size);
    return getDispatchTable().dot_product(a, b, size);
}

void VectorOps::vector_add(const double* a, const double* b, double* result,
                           std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_add_fallback(a, b, result, size);
    getDispatchTable().vector_add(a, b, result, size);
}

void VectorOps::vector_subtract(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_subtract_fallback(a, b, result, size);
    getDispatchTable().vector_subtract(a, b, result, size);
}

void VectorOps::vector_multiply(const double* a, const double* b, double* result,
                                std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_multiply_fallback(a, b, result, size);
    getDispatchTable().vector_multiply(a, b, result, size);
}

void VectorOps::scalar_multiply(const double* a, double scalar, double* result,
                                std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return scalar_multiply_fallback(a, scalar, result, size);
    getDispatchTable().scalar_multiply(a, scalar, result, size);
}

void VectorOps::scalar_add(const double* a, double scalar, double* result,
                           std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return scalar_add_fallback(a, scalar, result, size);
    getDispatchTable().scalar_add(a, scalar, result, size);
}

void VectorOps::vector_exp(const double* values, double* results, std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_exp_fallback(values, results, size);
    getDispatchTable().vector_exp(values, results, size);
}

void VectorOps::vector_log(const double* values, double* results, std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_log_fallback(values, results, size);
    getDispatchTable().vector_log(values, results, size);
}

void VectorOps::vector_pow(const double* base, double exponent, double* results,
                           std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_pow_fallback(base, exponent, results, size);
    getDispatchTable().vector_pow(base, exponent, results, size);
}

void VectorOps::vector_pow_elementwise(const double* base, const double* exponent, double* results,
                                       std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_pow_elementwise_fallback(base, exponent, results, size);
    getDispatchTable().vector_pow_elementwise(base, exponent, results, size);
}

void VectorOps::vector_erf(const double* values, double* results, std::size_t size) noexcept {
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) return vector_erf_fallback(values, results, size);
    getDispatchTable().vector_erf(values, results, size);
}

//========== Runtime Information Functions ==========

std::string VectorOps::get_active_simd_level() noexcept {
    // Return the highest SIMD level currently available at runtime
#ifdef LIBSTATS_HAS_AVX512
    if (stats::arch::supports_avx512()) {
        return "AVX-512";
    }
#endif

#ifdef LIBSTATS_HAS_AVX2
    if (stats::arch::supports_avx2()) {
        return "AVX2";
    }
#endif

#ifdef LIBSTATS_HAS_AVX
    if (stats::arch::supports_avx()) {
        return "AVX";
    }
#endif

#ifdef LIBSTATS_HAS_SSE2
    if (stats::arch::supports_sse2()) {
        return "SSE2";
    }
#endif

#ifdef LIBSTATS_HAS_NEON
    if (stats::arch::supports_neon()) {
        return "NEON";
    }
#endif

    return "Scalar";
}

bool VectorOps::is_simd_available() noexcept {
    return get_active_simd_level() != "Scalar";
}

std::size_t VectorOps::get_optimal_block_size() noexcept {
    return stats::arch::get_optimal_simd_block_size();
}

//========== Enhanced Platform-Aware Dispatch Utilities ==========

namespace {
/// Internal utility: Check if memory alignment is beneficial for current platform
inline bool is_alignment_beneficial(const void* ptr1, const void* ptr2 = nullptr,
                                    const void* ptr3 = nullptr) noexcept {
    const std::size_t alignment = stats::arch::get_optimal_alignment();

    bool aligned = (reinterpret_cast<uintptr_t>(ptr1) % alignment) == detail::ZERO_INT;
    if (ptr2) {
        aligned = aligned && ((reinterpret_cast<uintptr_t>(ptr2) % alignment) == detail::ZERO_INT);
    }
    if (ptr3) {
        aligned = aligned && ((reinterpret_cast<uintptr_t>(ptr3) % alignment) == detail::ZERO_INT);
    }

    return aligned;
}

/// Internal utility: Get platform-specific cache optimization threshold
inline std::size_t get_cache_optimization_threshold() noexcept {
    const auto thresholds = stats::arch::get_cache_thresholds();
    return thresholds.l1_optimal_size / detail::FOUR_INT;  // Use quarter of L1 as threshold
}

/// Internal utility: Choose optimal SIMD path based on data characteristics
template <typename Operation>
inline bool should_use_advanced_simd(std::size_t size, const void* ptr1, const void* ptr2 = nullptr,
                                     const void* ptr3 = nullptr) noexcept {
    // Basic size check
    if (!arch::simd::SIMDPolicy::shouldUseSIMD(size)) {
        return false;
    }

    // For very large datasets, always use SIMD regardless of alignment
    const std::size_t cache_threshold = get_cache_optimization_threshold();
    if (size >= cache_threshold) {
        return true;
    }

    // For medium datasets, check alignment benefits
    if (size >= stats::arch::simd::OPT_MEDIUM_DATASET_MIN_SIZE &&
        is_alignment_beneficial(ptr1, ptr2, ptr3)) {
        return true;
    }

// For high-end SIMD (AVX-512), use for smaller aligned datasets
#ifdef LIBSTATS_HAS_AVX512
    if (stats::arch::supports_avx512() && size >= stats::arch::simd::OPT_AVX512_MIN_ALIGNED_SIZE &&
        is_alignment_beneficial(ptr1, ptr2, ptr3)) {
        return true;
    }
#endif

// For Apple Silicon, be more aggressive with SIMD usage
#if defined(LIBSTATS_APPLE_SILICON)
    if (size >= stats::arch::simd::OPT_APPLE_SILICON_AGGRESSIVE_THRESHOLD) {
        return true;
    }
#endif

    return size >= stats::arch::get_min_simd_size();
}
}  // namespace

//========== Enhanced Public Interface Functions ==========

bool VectorOps::should_use_vectorized_path(std::size_t size, const void* data1, const void* data2,
                                           const void* data3) noexcept {
    // Early return for null pointer - always use scalar path
    if (!data1) {
        return false;
    }
    return should_use_advanced_simd<void>(size, data1, data2, data3);
}

std::string VectorOps::get_platform_optimization_info() noexcept {
    const auto thresholds = stats::arch::get_cache_thresholds();

    std::string info = "Platform: ";

#if defined(LIBSTATS_APPLE_SILICON)
    info += "Apple Silicon (";
#elif defined(__aarch64__) || defined(_M_ARM64)
    info += "ARM64 (";
#elif defined(__x86_64__) || defined(_M_X64)
    info += "x86_64 (";
#elif defined(__i386) || defined(_M_IX86)
    info += "x86_32 (";
#else
    info += "Unknown (";
#endif

    info += get_active_simd_level() + "), ";
    info += "SIMD Width: " + std::to_string(double_vector_width()) + ", ";
    info += "Min SIMD Size: " + std::to_string(min_simd_size()) + ", ";
    info += "L1 Cache Elements: " + std::to_string(thresholds.l1_optimal_size);

    return info;
}

}  // namespace ops
}  // namespace simd
}  // namespace stats
