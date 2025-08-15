#pragma once

// Common platform includes and utilities
#include "platform_common.h"

// Additional includes specific to SIMD operations
#include <new>
#include "simd_policy.h"

/**
 * @file simd.h
 * @brief Comprehensive SIMD operations and vectorized computations for libstats
 * 
 * This header provides a complete SIMD abstraction layer with:
 * - Platform-specific SIMD intrinsics and feature detection
 * - Vectorized mathematical operations (dot product, add, multiply, etc.)
 * - Memory alignment utilities for optimal SIMD performance
 * - Runtime dispatch to the best available SIMD implementation
 * - Compile-time feature detection and optimization constants
 * 
 * ARCHITECTURE:
 * 
 * The SIMD system is designed with multiple layers:
 * 1. Platform Detection: Compile-time detection of available SIMD features
 * 2. Implementation Layer: Separate optimized implementations for each SIMD level
 * 3. Dispatch Layer: Runtime selection of the best available implementation
 * 4. Public API: Clean, simple interface that abstracts SIMD complexity
 * 
 * SUPPORTED PLATFORMS:
 * - x86/x64: SSE2, SSE4.1, AVX, AVX2, AVX-512
 * - ARM64: NEON (including Apple Silicon optimizations)
 * - Scalar fallback for unsupported platforms
 * 
 * USAGE:
 * 
 * Basic vectorized operations:
 *   double result = VectorOps::dot_product(a, b, size);
 *   VectorOps::vector_add(a, b, result, size);
 * 
 * Platform-specific optimizations:
 *   #ifdef LIBSTATS_HAS_AVX
 *       // Use AVX-specific code
 *   #elif defined(LIBSTATS_HAS_NEON)
 *       // Use NEON-specific code
 *   #endif
 * 
 * Memory alignment for SIMD:
 *   std::vector<double, aligned_allocator<double>> aligned_data(size);
 */

//==============================================================================
// Platform Detection and SIMD Intrinsics
//==============================================================================

// Microsoft Visual C++ - Windows
#if defined(_MSC_VER)
    #include <intrin.h>
    #ifndef LIBSTATS_HAS_SSE2
        #define LIBSTATS_HAS_SSE2
    #endif
    #if defined(__AVX__)
        #ifndef LIBSTATS_HAS_AVX
            #define LIBSTATS_HAS_AVX
        #endif
    #endif
    #if defined(__AVX2__)
        #ifndef LIBSTATS_HAS_AVX2
            #define LIBSTATS_HAS_AVX2
        #endif
    #endif

// GCC/Clang - x86/x64 platforms (Intel/AMD)
#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
    #include <immintrin.h>
    #include <x86intrin.h>
    #ifndef LIBSTATS_HAS_SSE2
        #define LIBSTATS_HAS_SSE2  // Available on all modern x86_64
    #endif
    #if defined(__SSE4_1__) && !defined(LIBSTATS_HAS_SSE4_1)
        #define LIBSTATS_HAS_SSE4_1
    #endif
    #if defined(__AVX__) && !defined(LIBSTATS_HAS_AVX)
        #define LIBSTATS_HAS_AVX
    #endif
    #if defined(__AVX2__) && !defined(LIBSTATS_HAS_AVX2)
        #define LIBSTATS_HAS_AVX2
    #endif
    #if defined(__AVX512F__) && !defined(LIBSTATS_HAS_AVX512)
        #define LIBSTATS_HAS_AVX512
    #endif

// ARM platforms (Apple Silicon, ARM servers, embedded ARM)
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    #include <arm_neon.h>
    #ifndef LIBSTATS_HAS_NEON
        #define LIBSTATS_HAS_NEON
    #endif
    // Apple Silicon specific optimizations
    #if defined(__APPLE__) && defined(__aarch64__)
        #define LIBSTATS_APPLE_SILICON
    #endif

// Fallback - No SIMD support detected
#else
    #warning "No SIMD support detected - using scalar fallback implementations"
#endif

//==============================================================================
// Feature Detection Utilities
//==============================================================================

namespace libstats {
namespace simd {

/**
 * @brief Compile-time SIMD capability detection
 * @return true if any SIMD extensions are available at compile time
 */
constexpr bool has_simd_support() noexcept {
#if defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_SSE2) || defined(LIBSTATS_HAS_NEON)
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get the SIMD vector width for double precision operations
 * @return Number of double-precision values that fit in a SIMD register
 */
constexpr std::size_t double_vector_width() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 8;  // AVX-512 can handle 8 doubles
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 4;  // AVX can handle 4 doubles
#elif defined(LIBSTATS_HAS_SSE2)
    return 2;  // SSE2 can handle 2 doubles
#elif defined(LIBSTATS_HAS_NEON)
    return 2;  // ARM NEON can handle 2 doubles (64-bit elements)
#else
    return 1;  // Scalar fallback
#endif
}

/**
 * @brief Get the SIMD vector width for single precision operations
 * @return Number of single-precision values that fit in a SIMD register
 */
constexpr std::size_t float_vector_width() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 16; // AVX-512 can handle 16 floats
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 8;  // AVX can handle 8 floats
#elif defined(LIBSTATS_HAS_SSE2)
    return 4;  // SSE2 can handle 4 floats
#elif defined(LIBSTATS_HAS_NEON)
    return 4;  // ARM NEON can handle 4 floats (32-bit elements)
#else
    return 1;  // Scalar fallback
#endif
}

/**
 * @brief Get the optimal memory alignment for SIMD operations
 * @return Alignment in bytes for optimal SIMD performance
 */
constexpr std::size_t optimal_alignment() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 64; // AVX-512 benefits from 64-byte alignment
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 32; // AVX requires 32-byte alignment
#elif defined(LIBSTATS_HAS_SSE2)
    return 16; // SSE2 requires 16-byte alignment
#elif defined(LIBSTATS_HAS_NEON)
    return 16; // ARM NEON benefits from 16-byte alignment
#else
    return 8;  // Basic double alignment
#endif
}

/**
 * @brief Get a human-readable description of available SIMD features
 * @return String describing the detected SIMD capabilities
 */
inline const char* feature_string() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return "AVX-512";
#elif defined(LIBSTATS_HAS_AVX2)
    return "AVX2";
#elif defined(LIBSTATS_HAS_AVX)
    return "AVX";
#elif defined(LIBSTATS_HAS_SSE4_1)
    return "SSE4.1";
#elif defined(LIBSTATS_HAS_SSE2)
    return "SSE2";
#elif defined(LIBSTATS_HAS_NEON)
    #if defined(LIBSTATS_APPLE_SILICON)
        return "ARM NEON (Apple Silicon)";
    #else
        return "ARM NEON";
    #endif
#else
    return "Scalar (No SIMD)";
#endif
}

/**
 * @brief Check if the current build supports vectorized operations
 * @return true if SIMD is available and beneficial for performance
 */
constexpr bool supports_vectorization() noexcept {
    return has_simd_support() && double_vector_width() >= 2;
}

//==============================================================================
// Platform-Adaptive Constants
//==============================================================================

/**
 * @brief Optimal SIMD alignment based on detected platform capabilities
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * ensuring optimal memory alignment for the available instruction set.
 */
static constexpr std::size_t SIMD_ALIGNMENT = optimal_alignment();

/**
 * @brief SIMD vector width for double precision based on detected platform
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * providing the correct vector width for double precision operations.
 */
static constexpr std::size_t DOUBLE_SIMD_WIDTH = double_vector_width();

/**
 * @brief SIMD vector width for single precision based on detected platform
 * 
 * This constant adapts to the actual SIMD capabilities detected at compile time,
 * providing the correct vector width for single precision operations.
 */
static constexpr std::size_t FLOAT_SIMD_WIDTH = float_vector_width();

//==============================================================================
// Platform-Tuned Performance Constants
//==============================================================================

/**
 * @brief Platform-optimized block sizes for different operations
 * 
 * These constants are tuned based on the detected SIMD capabilities, cache
 * characteristics, and architectural features of the target platform.
 * 
 * Targeting desktop/laptop processors:
 * - Apple Silicon (M1/M2/M3): Excellent cache hierarchy, wide SIMD
 * - Intel x86_64: Strong AVX support, moderate caches
 * - AMD Ryzen: Strong AVX2 support, large L3 cache
 */
namespace tuned {

/**
 * @brief Optimal block size for matrix operations based on cache size
 * @return Block size that fits well in L1 cache for matrix blocking
 */
constexpr std::size_t matrix_block_size() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    // Intel Sapphire Rapids, AMD Zen 4 - large caches, wide vectors
    return 64;
#elif defined(LIBSTATS_HAS_AVX2)
    #if defined(__znver3__) || defined(__znver2__) || defined(__znver1__)
        // AMD Ryzen - excellent L3 cache, can handle larger blocks
        return 48;
    #else
        // Intel AVX2 (Haswell+) - good balance
        return 32;
    #endif
#elif defined(LIBSTATS_HAS_AVX)
    return 32;  // First-gen AVX (Sandy Bridge, Bulldozer)
#elif defined(LIBSTATS_HAS_SSE2)
    return 16;  // Older x86_64 hardware
#elif defined(LIBSTATS_HAS_NEON)
    #if defined(LIBSTATS_APPLE_SILICON)
        // Apple Silicon: M1/M2/M3 have excellent cache hierarchy
        return 48;  // Can handle larger blocks efficiently
    #else
        return 16;  // Other ARM64 devices (servers, embedded)
    #endif
#else
    return 8;   // Scalar fallback
#endif
}

/**
 * @brief Optimal SIMD processing width for loops
 * @return Number of elements to process in SIMD loops for best performance
 */
constexpr std::size_t simd_loop_width() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 8;   // AVX-512 can handle 8 doubles efficiently
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 4;   // AVX handles 4 doubles efficiently
#elif defined(LIBSTATS_HAS_SSE2)
    return 2;   // SSE2 handles 2 doubles
#elif defined(LIBSTATS_HAS_NEON)
    return 2;   // NEON handles 2 doubles
#else
    return 1;   // Scalar
#endif
}

/**
 * @brief Minimum number of states to benefit from SIMD operations
 * @return Threshold where SIMD overhead is justified
 */
constexpr std::size_t min_states_for_simd() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 16;  // AVX-512 setup cost is higher but benefits are greater
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 8;   // AVX has moderate setup cost
#elif defined(LIBSTATS_HAS_SSE2)
    return 4;   // SSE2 has lower setup cost
#elif defined(LIBSTATS_HAS_NEON)
    return 4;   // NEON is efficient
#else
    return SIZE_MAX;  // Never use SIMD in scalar mode
#endif
}

/**
 * @brief Optimal grain size for parallel operations based on core count
 * @return Work items per thread for good load balancing
 */
constexpr std::size_t parallel_grain_size() noexcept {
    // This should ideally be determined at runtime, but we provide
    // reasonable compile-time defaults based on typical core counts
#if defined(LIBSTATS_APPLE_SILICON)
    return 32;  // Apple Silicon has many high-performance cores
#elif defined(__x86_64__) || defined(_M_X64)
    return 64;  // x86_64 typically has 4-16 cores with hyperthreading
#elif defined(__aarch64__) || defined(_M_ARM64)
    return 48;  // ARM64 servers can have many cores
#else
    return 32;  // Conservative default
#endif
}

/**
 * @brief Minimum work items to justify parallel execution
 * @return Threshold where parallel overhead is justified
 */
constexpr std::size_t min_parallel_work() noexcept {
#if defined(LIBSTATS_APPLE_SILICON)
    return 64;   // Apple Silicon thread creation is very fast
#elif defined(__x86_64__) || defined(_M_X64)
    return 128;  // x86_64 moderate thread overhead
#else
    return 256;  // Conservative for unknown architectures
#endif
}

/**
 * @brief Cache-friendly iteration step size
 * @return Step size that aligns with cache line boundaries
 */
constexpr std::size_t cache_friendly_step() noexcept {
    // Most modern systems have 64-byte cache lines
    // For double precision (8 bytes), this is 8 elements per cache line
    return 8;
}

/**
 * @brief Number of elements that fit in typical L1 cache
 * @return Conservative estimate for L1 cache capacity in doubles
 */
constexpr std::size_t l1_cache_doubles() noexcept {
#if defined(LIBSTATS_APPLE_SILICON)
    return 16384;  // Apple Silicon: ~128KB L1D cache
#elif defined(__x86_64__) || defined(_M_X64)
    return 4096;   // Typical x86_64: ~32KB L1D cache
#elif defined(__aarch64__) || defined(_M_ARM64)
    return 8192;   // ARM64 varies: ~64KB L1D cache
#else
    return 2048;   // Conservative: ~16KB L1D cache
#endif
}

/**
 * @brief Prefetch distance for optimal memory access patterns
 * @return Number of cache lines to prefetch ahead
 */
constexpr std::size_t prefetch_distance() noexcept {
#if defined(LIBSTATS_HAS_AVX512)
    return 4;   // High-performance systems can handle more prefetch
#elif defined(LIBSTATS_HAS_AVX) || defined(LIBSTATS_HAS_AVX2)
    return 2;   // Moderate prefetch for AVX systems
#elif defined(LIBSTATS_APPLE_SILICON)
    return 3;   // Apple Silicon has excellent prefetch hardware
#else
    return 1;   // Conservative prefetch
#endif
}

} // namespace tuned

//==============================================================================
// ALIGNED MEMORY ALLOCATION FOR SIMD
//==============================================================================

/// Thread-safe aligned memory allocator for SIMD operations
template<typename T>
class aligned_allocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = aligned_allocator<U>;
    };

    aligned_allocator() noexcept = default;
    
    template<typename U>
    aligned_allocator(const aligned_allocator<U>&) noexcept {}
    
    aligned_allocator(const aligned_allocator&) noexcept = default;
    aligned_allocator& operator=(const aligned_allocator&) noexcept = default;

    pointer allocate(size_type n) {
        if (n == 0) return nullptr;
        
        // Check for overflow
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }
        
        const size_type bytes = n * sizeof(T);
        constexpr size_type alignment = (SIMD_ALIGNMENT < sizeof(void*)) ? sizeof(void*) : SIMD_ALIGNMENT;
        const size_type aligned_bytes = ((bytes + alignment - 1) / alignment) * alignment;
        
        void* ptr = nullptr;
        
#if defined(_WIN32)
        ptr = _aligned_malloc(aligned_bytes, alignment);
        if (!ptr) throw std::bad_alloc();
#elif defined(__APPLE__) || defined(__linux__)
        const int result = posix_memalign(&ptr, alignment, aligned_bytes);
        if (result != 0 || !ptr) throw std::bad_alloc();
#else
        ptr = std::aligned_alloc(alignment, aligned_bytes);
        if (!ptr) throw std::bad_alloc();
#endif
        
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        if (p) {
#if defined(_WIN32)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }
    }

    template<typename U>
    bool operator==(const aligned_allocator<U>&) const noexcept { return true; }
    
    template<typename U>
    bool operator!=(const aligned_allocator<U>&) const noexcept { return false; }
    
    size_type max_size() const noexcept {
        return std::numeric_limits<size_type>::max() / sizeof(T);
    }
};

//==============================================================================
// CORE SIMD VECTOR OPERATIONS
//==============================================================================

/// SIMD-optimized vector operations for statistical computations
class VectorOps {
public:
    /// Vectorized dot product
    /// @param a First vector
    /// @param b Second vector
    /// @param size Number of elements
    /// @return Dot product result
    static double dot_product(const double* a, const double* b, std::size_t size) noexcept;
    
    /// Vectorized vector addition
    /// @param a First vector
    /// @param b Second vector
    /// @param result Output vector (a + b)
    /// @param size Number of elements
    static void vector_add(const double* a, const double* b, double* result, std::size_t size) noexcept;
    
    /// Vectorized vector subtraction
    /// @param a First vector
    /// @param b Second vector
    /// @param result Output vector (a - b)
    /// @param size Number of elements
    static void vector_subtract(const double* a, const double* b, double* result, std::size_t size) noexcept;
    
    /// Vectorized element-wise multiplication
    /// @param a First vector
    /// @param b Second vector
    /// @param result Output vector (a * b)
    /// @param size Number of elements
    static void vector_multiply(const double* a, const double* b, double* result, std::size_t size) noexcept;
    
    /// Vectorized scalar multiplication
    /// @param a Input vector
    /// @param scalar Scalar value
    /// @param result Output vector (a * scalar)
    /// @param size Number of elements
    static void scalar_multiply(const double* a, double scalar, double* result, std::size_t size) noexcept;
    
    /// Vectorized scalar addition
    /// @param a Input vector
    /// @param scalar Scalar value
    /// @param result Output vector (a + scalar)
    /// @param size Number of elements
    static void scalar_add(const double* a, double scalar, double* result, std::size_t size) noexcept;
    
    /// Vectorized exponential computation
    /// @param values Input vector
    /// @param results Output vector (exp(values))
    /// @param size Number of elements
    static void vector_exp(const double* values, double* results, std::size_t size) noexcept;
    
    /// Vectorized natural logarithm computation
    /// @param values Input vector
    /// @param results Output vector (log(values))
    /// @param size Number of elements
    static void vector_log(const double* values, double* results, std::size_t size) noexcept;
    
    /// Vectorized power computation
    /// @param base Base vector
    /// @param exponent Exponent (scalar)
    /// @param results Output vector (base^exponent)
    /// @param size Number of elements
    static void vector_pow(const double* base, double exponent, double* results, std::size_t size) noexcept;
    
    /// Vectorized error function computation
    /// @param values Input vector
    /// @param results Output vector (erf(values))
    /// @param size Number of elements
    /// @note Uses high-precision rational approximation for accuracy
    static void vector_erf(const double* values, double* results, std::size_t size) noexcept;
    
    /// Check if SIMD should be used for given size
    /// @param size Number of elements to process
    /// @return true if SIMD is beneficial for this size
    static bool should_use_simd(std::size_t size) noexcept;
    
    /// Get minimum size threshold for SIMD operations
    /// @return Minimum number of elements where SIMD becomes beneficial
    static std::size_t min_simd_size() noexcept;
    
    /// Get active SIMD level as string
    /// @return String representation of active SIMD level
    static std::string get_active_simd_level() noexcept;
    
    /// Check if any SIMD optimizations are available
    /// @return true if SIMD is available on this system
    static bool is_simd_available() noexcept;
    
    /// Get optimal block size for SIMD operations
    /// @return Optimal number of elements per SIMD block
    static std::size_t get_optimal_block_size() noexcept;
    
    /// Check if vectorization is supported
    /// @return true if any form of vectorization is available
    static bool supports_vectorization() noexcept;
    
    /// Get double vector width
    /// @return Number of doubles that fit in a SIMD register
    static std::size_t double_vector_width() noexcept;
    
    /// Advanced platform-aware decision for vectorized operations
    /// @param size Number of elements to process
    /// @param data1 Pointer to first data array (for alignment checking)
    /// @param data2 Pointer to second data array (optional)
    /// @param data3 Pointer to third data array (optional)
    /// @return true if vectorized path should be used
    static bool should_use_vectorized_path(std::size_t size, const void* data1, const void* data2 = nullptr, const void* data3 = nullptr) noexcept;
    
    /// Get comprehensive platform optimization information
    /// @return String with detailed platform and optimization info
    static std::string get_platform_optimization_info() noexcept;

private:
    // Fallback implementations
    static double dot_product_fallback(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_fallback(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_fallback(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_fallback(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_fallback(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_fallback(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_fallback(const double* values, double* results, std::size_t size) noexcept;
    
    // SIMD-specific implementations
#ifdef LIBSTATS_HAS_AVX512
    static double dot_product_avx512(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_avx512(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_avx512(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_avx512(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_avx512(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_avx512(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_avx512(const double* values, double* results, std::size_t size) noexcept;
#endif
    
#ifdef LIBSTATS_HAS_AVX
    static double dot_product_avx(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_avx(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_avx(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_avx(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_avx(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_avx(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_avx(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_avx(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_avx(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_avx(const double* values, double* results, std::size_t size) noexcept;
#endif
    
#ifdef LIBSTATS_HAS_AVX2
    static double dot_product_avx2(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_avx2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_avx2(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_avx2(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_avx2(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_avx2(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_avx2(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_avx2(const double* values, double* results, std::size_t size) noexcept;
#endif
    
#ifdef LIBSTATS_HAS_SSE2
    static double dot_product_sse2(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_sse2(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_sse2(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_sse2(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_sse2(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_sse2(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_sse2(const double* values, double* results, std::size_t size) noexcept;
#endif
    
#ifdef LIBSTATS_HAS_NEON
    static double dot_product_neon(const double* a, const double* b, std::size_t size) noexcept;
    static void vector_add_neon(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_subtract_neon(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void vector_multiply_neon(const double* a, const double* b, double* result, std::size_t size) noexcept;
    static void scalar_multiply_neon(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void scalar_add_neon(const double* a, double scalar, double* result, std::size_t size) noexcept;
    static void vector_exp_neon(const double* values, double* results, std::size_t size) noexcept;
    static void vector_log_neon(const double* values, double* results, std::size_t size) noexcept;
    static void vector_pow_neon(const double* base, double exponent, double* results, std::size_t size) noexcept;
    static void vector_erf_neon(const double* values, double* results, std::size_t size) noexcept;
#endif
};

//==============================================================================
// MEMORY PREFETCHING AND CACHE OPTIMIZATION
//==============================================================================

/// Memory prefetching hints for better cache performance
inline void prefetch_read(const void* addr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 0, 3); // Read, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

inline void prefetch_write(const void* addr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(addr, 1, 3); // Write, high temporal locality
#elif defined(_MSC_VER)
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
#endif
}

/// Cache line size for modern processors
static constexpr std::size_t CACHE_LINE_SIZE = 64;

/// Cache line alignment utility
template<typename T>
constexpr std::size_t cache_aligned_size(std::size_t size) noexcept {
    const std::size_t total_bytes = size * sizeof(T);
    return ((total_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE) * CACHE_LINE_SIZE / sizeof(T);
}

/// Check if pointer is properly aligned for SIMD operations
template<typename T>
constexpr bool is_aligned(const T* ptr, std::size_t alignment = SIMD_ALIGNMENT) noexcept {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

/// Align size to SIMD boundary
constexpr std::size_t align_size(std::size_t size, std::size_t alignment = SIMD_ALIGNMENT) noexcept {
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace simd
} // namespace libstats
