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
// Reading guide: each value is the smallest batch size at which a parallel
// strategy sustainably beats VECTORIZED up to 500K elements.
// NEVER means VECTORIZED always wins on that machine for that operation.
//
// Table layout: {pdf, log_pdf, cdf} per distribution row.
// Beta is excluded (NEVER across all architectures and operations).
// ============================================================================

/**
 * @brief Thresholds for one distribution across the three scalar-batch operations.
 */
struct ThresholdRow {
    std::size_t pdf;
    std::size_t log_pdf;
    std::size_t cdf;
};

/**
 * @brief Empirical parallel thresholds for one SIMD architecture.
 *
 * Rows correspond to distributions in DistributionType enum order;
 * see the per-architecture constexpr instances below for data.
 */
struct ArchTable {
    ThresholdRow uniform;
    ThresholdRow gaussian;
    ThresholdRow exponential;
    ThresholdRow discrete;
    ThresholdRow poisson;
    ThresholdRow gamma;
    ThresholdRow student_t;
    ThresholdRow chi_squared;
};

// --- NEON (Apple M1, 128-bit, 8C/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-36-21Z_darwin-arm64_…_sha-6aef918
constexpr ArchTable kNeon = {
    /* uniform     */ {NEVER,  NEVER,  20000},
    /* gaussian    */ {50000,  100000, 10000},
    /* exponential */ {50000,  100000, 20000},
    /* discrete    */ {250000, 250000, 100000},
    /* poisson     */ {20000,  50000,  2000},
    /* gamma       */ {20000,  20000,  2000},
    /* student_t   */ {20000,  50000,  250000},
    /* chi_squared */ {20000,  50000,  2000},
};

// --- AVX (Intel Ivy Bridge i7-3820QM, 128/256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-55-52Z_darwin-x86_64_…_sha-e75c6e3
constexpr ArchTable kAvx = {
    /* uniform     */ {NEVER,  NEVER,  10000},
    /* gaussian    */ {20000,  50000,  20000},
    /* exponential */ {20000,  100000, 20000},
    /* discrete    */ {50000,  50000,  50000},
    /* poisson     */ {2000,   10000,  5000},
    /* gamma       */ {20000,  20000,  2000},
    /* student_t   */ {100000, 100000, 100000},
    /* chi_squared */ {20000,  20000,  2000},
};

// --- AVX2 (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-04-12T05-27-04Z_darwin-x86_64_…_sha-0e4e9f1
constexpr ArchTable kAvx2 = {
    /* uniform     */ {NEVER,  NEVER,  20000},
    /* gaussian    */ {50000,  250000, 50000},
    /* exponential */ {50000,  250000, 50000},
    /* discrete    */ {100000, 50000,  50000},
    /* poisson     */ {10000,  20000,  2000},
    /* gamma       */ {50000,  50000,  5000},
    /* student_t   */ {100000, 100000, NEVER},
    /* chi_squared */ {50000,  100000, 2000},
};

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// data/profiles/dispatcher/2026-04-12T06-02-56Z_windows-x86_64_…_sha-32c0819
constexpr ArchTable kAvx512 = {
    /* uniform     */ {100000, 50000,  50000},
    /* gaussian    */ {100000, NEVER,  50000},
    /* exponential */ {50000,  250000, 100000},
    /* discrete    */ {50000,  250000, 50000},
    /* poisson     */ {10000,  20000,  10000},
    /* gamma       */ {20000,  50000,  2000},
    /* student_t   */ {20000,  250000, NEVER},
    /* chi_squared */ {50000,  50000,  5000},
};

/**
 * @brief Single shared implementation: look up (dist, op) in one ArchTable.
 *
 * Replaces four structurally identical 35-CCN functions. Only the data
 * (kNeon / kAvx / kAvx2 / kAvx512) differs between architectures.
 */
constexpr std::size_t parallelThresholdFromTable(const ArchTable& table,
                                                  DistributionType dist,
                                                  OperationType op) noexcept {
    if (op == OperationType::BATCH_FIT) return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)  return NEVER;

    const ThresholdRow* row = nullptr;
    switch (dist) {
        case DistributionType::UNIFORM:      row = &table.uniform;      break;
        case DistributionType::GAUSSIAN:     row = &table.gaussian;     break;
        case DistributionType::EXPONENTIAL:  row = &table.exponential;  break;
        case DistributionType::DISCRETE:     row = &table.discrete;     break;
        case DistributionType::POISSON:      row = &table.poisson;      break;
        case DistributionType::GAMMA:        row = &table.gamma;        break;
        case DistributionType::STUDENT_T:    row = &table.student_t;    break;
        case DistributionType::CHI_SQUARED:  row = &table.chi_squared;  break;
        default:                             return NEVER;
    }
    switch (op) {
        case OperationType::PDF:     return row->pdf;
        case OperationType::LOG_PDF: return row->log_pdf;
        case OperationType::CDF:     return row->cdf;
        default:                     return NEVER;
    }
}

// Per-architecture wrappers — thin delegation to the shared implementation.
constexpr std::size_t neon_parallel_threshold(DistributionType dist, OperationType op) {
    return parallelThresholdFromTable(kNeon, dist, op);
}
constexpr std::size_t avx_parallel_threshold(DistributionType dist, OperationType op) {
    return parallelThresholdFromTable(kAvx, dist, op);
}
constexpr std::size_t avx2_parallel_threshold(DistributionType dist, OperationType op) {
    return parallelThresholdFromTable(kAvx2, dist, op);
}
constexpr std::size_t avx512_parallel_threshold(DistributionType dist, OperationType op) {
    return parallelThresholdFromTable(kAvx512, dist, op);
}

// --- SSE2 fallback: shares AVX thresholds (similar 128-bit SIMD width) ---
constexpr std::size_t sse2_parallel_threshold(DistributionType dist, OperationType op) {
    return avx_parallel_threshold(dist, op);
}

// --- No SIMD: conservative high thresholds ---
constexpr std::size_t none_parallel_threshold(DistributionType dist, OperationType op) {
    if (op == OperationType::BATCH_FIT) return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)  return NEVER;
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
