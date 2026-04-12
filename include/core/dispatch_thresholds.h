#pragma once

/**
 * @file dispatch_thresholds.h
 * @brief Profiling-derived constexpr lookup table for dispatch strategy thresholds
 *
 * Each threshold is the batch size at which parallel execution sustainably
 * beats VECTORIZED for a given (SIMD level, distribution, operation) triple.
 * Values are derived from Release-build profiling bundles captured on four
 * target architectures (see data/profiles/dispatcher/).
 *
 * SIZE_MAX means "never parallel" — VECTORIZED is always preferred.
 *
 * The SCALAR→VECTORIZED boundary is handled separately by SIMDPolicy::getMinThreshold()
 * and is architecture-independent within a SIMD level (typically 4–8 elements).
 */

#include "libstats/platform/simd_policy.h"
#include "performance_dispatcher.h"

#include <cstddef>
#include <limits>

namespace stats {
namespace detail {

/**
 * @brief Operation types for per-operation threshold resolution
 */
enum class OperationType {
    PDF,       ///< Probability density/mass function
    LOG_PDF,   ///< Log-probability density/mass function
    CDF,       ///< Cumulative distribution function
    BATCH_FIT  ///< Parallel batch parameter estimation
};

namespace dispatch_table {

/// Sentinel: VECTORIZED is always preferred over parallel strategies.
constexpr std::size_t NEVER = std::numeric_limits<std::size_t>::max();

/// Minimum datasets for parallel batch fitting (architecture-independent).
constexpr std::size_t BATCH_FIT_MIN = 8;

// ============================================================================
// Per-architecture parallel thresholds: (DistributionType, OperationType) → size
// Derived from strategy_profile Release builds, 2026-04-12.
//
// Reading guide: the value is the smallest batch size at which a parallel
// strategy (PARALLEL or WORK_STEALING) sustainably beats VECTORIZED through
// the largest measured size (500K).  NEVER means it never does.
// ============================================================================

// --- NEON (Apple M1, 128-bit, 8C/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-36-21Z_darwin-arm64_…_sha-6aef918

constexpr std::size_t neon_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;

    switch (dist) {
        case DistributionType::UNIFORM:
            switch (op) {
                case OperationType::PDF:
                    return NEVER;
                case OperationType::LOG_PDF:
                    return NEVER;
                case OperationType::CDF:
                    return 20000;
                default:
                    return NEVER;
            }
        case DistributionType::GAUSSIAN:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return 10000;
                default:
                    return NEVER;
            }
        case DistributionType::EXPONENTIAL:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return 20000;
                default:
                    return NEVER;
            }
        case DistributionType::DISCRETE:
            switch (op) {
                case OperationType::PDF:
                    return 250000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return 100000;
                default:
                    return NEVER;
            }
        case DistributionType::POISSON:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        case DistributionType::GAMMA:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 20000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        case DistributionType::STUDENT_T:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 250000;
                default:
                    return NEVER;
            }
        case DistributionType::CHI_SQUARED:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        default:
            return NEVER;
    }
}

// --- AVX (Intel Ivy Bridge i7-3820QM, 128/256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-55-52Z_darwin-x86_64_…_sha-e75c6e3

constexpr std::size_t avx_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;

    switch (dist) {
        case DistributionType::UNIFORM:
            switch (op) {
                case OperationType::PDF:
                    return NEVER;
                case OperationType::LOG_PDF:
                    return NEVER;
                case OperationType::CDF:
                    return 10000;
                default:
                    return NEVER;
            }
        case DistributionType::GAUSSIAN:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 20000;
                default:
                    return NEVER;
            }
        case DistributionType::EXPONENTIAL:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return 20000;
                default:
                    return NEVER;
            }
        case DistributionType::DISCRETE:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::POISSON:
            switch (op) {
                case OperationType::PDF:
                    return 2000;
                case OperationType::LOG_PDF:
                    return 10000;
                case OperationType::CDF:
                    return 5000;
                default:
                    return NEVER;
            }
        case DistributionType::GAMMA:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 20000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        case DistributionType::STUDENT_T:
            switch (op) {
                case OperationType::PDF:
                    return 100000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return 100000;
                default:
                    return NEVER;
            }
        case DistributionType::CHI_SQUARED:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 20000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        default:
            return NEVER;
    }
}

// --- AVX2 (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-27-04Z_darwin-x86_64_…_sha-0e4e9f1

constexpr std::size_t avx2_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;

    switch (dist) {
        case DistributionType::UNIFORM:
            switch (op) {
                case OperationType::PDF:
                    return NEVER;
                case OperationType::LOG_PDF:
                    return NEVER;
                case OperationType::CDF:
                    return 20000;
                default:
                    return NEVER;
            }
        case DistributionType::GAUSSIAN:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::EXPONENTIAL:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::DISCRETE:
            switch (op) {
                case OperationType::PDF:
                    return 100000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::POISSON:
            switch (op) {
                case OperationType::PDF:
                    return 10000;
                case OperationType::LOG_PDF:
                    return 20000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        case DistributionType::GAMMA:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 5000;
                default:
                    return NEVER;
            }
        case DistributionType::STUDENT_T:
            switch (op) {
                case OperationType::PDF:
                    return 100000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return NEVER;
                default:
                    return NEVER;
            }
        case DistributionType::CHI_SQUARED:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 100000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        default:
            return NEVER;
    }
}

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// data/profiles/dispatcher/2026-04-12T06-02-56Z_windows-x86_64_…_sha-32c0819

constexpr std::size_t avx512_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;

    switch (dist) {
        case DistributionType::UNIFORM:
            switch (op) {
                case OperationType::PDF:
                    return 100000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::GAUSSIAN:
            switch (op) {
                case OperationType::PDF:
                    return 100000;
                case OperationType::LOG_PDF:
                    return NEVER;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::EXPONENTIAL:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return 100000;
                default:
                    return NEVER;
            }
        case DistributionType::DISCRETE:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return 50000;
                default:
                    return NEVER;
            }
        case DistributionType::POISSON:
            switch (op) {
                case OperationType::PDF:
                    return 10000;
                case OperationType::LOG_PDF:
                    return 20000;
                case OperationType::CDF:
                    return 10000;
                default:
                    return NEVER;
            }
        case DistributionType::GAMMA:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 2000;
                default:
                    return NEVER;
            }
        case DistributionType::STUDENT_T:
            switch (op) {
                case OperationType::PDF:
                    return 20000;
                case OperationType::LOG_PDF:
                    return 250000;
                case OperationType::CDF:
                    return NEVER;
                default:
                    return NEVER;
            }
        case DistributionType::CHI_SQUARED:
            switch (op) {
                case OperationType::PDF:
                    return 50000;
                case OperationType::LOG_PDF:
                    return 50000;
                case OperationType::CDF:
                    return 5000;
                default:
                    return NEVER;
            }
        default:
            return NEVER;
    }
}

// --- SSE2 fallback: shares AVX thresholds (similar 128-bit SIMD width) ---

constexpr std::size_t sse2_parallel_threshold(DistributionType dist, OperationType op) {
    return avx_parallel_threshold(dist, op);
}

// --- No SIMD: conservative high thresholds ---

constexpr std::size_t none_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;
    // Without SIMD, VECTORIZED is just a scalar loop via the batch path.
    // Parallel helps earlier because there is no SIMD advantage to protect.
    return 5000;
}

}  // namespace dispatch_table

/**
 * @brief Look up the parallel threshold for a given SIMD level, distribution, and operation.
 *
 * Returns the batch size at which parallel execution sustainably beats VECTORIZED.
 * Returns SIZE_MAX if VECTORIZED is always preferred.
 *
 * @param level  Runtime SIMD level from SIMDPolicy
 * @param dist   Distribution type
 * @param op     Operation type (PDF, LOG_PDF, CDF, BATCH_FIT)
 * @return Minimum batch size for parallel execution
 */
constexpr std::size_t getParallelThreshold(arch::simd::SIMDPolicy::Level level,
                                           DistributionType dist, OperationType op) {
    switch (level) {
        case arch::simd::SIMDPolicy::Level::NEON:
            return dispatch_table::neon_parallel_threshold(dist, op);
        case arch::simd::SIMDPolicy::Level::AVX512:
            return dispatch_table::avx512_parallel_threshold(dist, op);
        case arch::simd::SIMDPolicy::Level::AVX2:
            return dispatch_table::avx2_parallel_threshold(dist, op);
        case arch::simd::SIMDPolicy::Level::AVX:
            return dispatch_table::avx_parallel_threshold(dist, op);
        case arch::simd::SIMDPolicy::Level::SSE2:
            return dispatch_table::sse2_parallel_threshold(dist, op);
        case arch::simd::SIMDPolicy::Level::None:
        default:
            return dispatch_table::none_parallel_threshold(dist, op);
    }
}

}  // namespace detail
}  // namespace stats
