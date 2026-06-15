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
// Derived from strategy_profile Release builds captured in data/profiles/dispatcher/.
//
// Reading guide: each value is the smallest batch size at which a parallel
// strategy sustainably beats VECTORIZED up to 500K elements.
// NEVER means VECTORIZED always wins on that machine for that operation.
//
// Table layout: {pdf, log_pdf, cdf} per distribution row.
// Beta is excluded (NEVER across all architectures and operations).
//
// Measurement resolution caveat (applies to all architectures):
//   Profiler timing resolution floors out around 0.1–0.2 µs. At batch sizes
//   below ~64 elements, many measurements read 0.0 µs regardless of strategy,
//   making the derived crossover unreliable (noise can make parallel appear to
//   win at batch=8 simply because both scalar and SIMD round to the same tick).
//   When recalibrating this table, treat any crossover derived from sizes < 64
//   as suspect and clamp the threshold upward to at least 64 rather than
//   encoding the noise artifact.
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
    ThresholdRow lognormal;
    ThresholdRow pareto;
    ThresholdRow weibull;
    ThresholdRow rayleigh;
    ThresholdRow von_mises;
    ThresholdRow binomial;
    ThresholdRow negative_binomial;
};

// --- NEON (Apple M1, 128-bit, 8C/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-15T04-55-46Z_darwin-arm64_fix-audit-remediation_sha-b583fb3
// data/profiles/dispatcher/2026-06-15T05-04-15Z_darwin-arm64_fix-audit-remediation_sha-b583fb3
//
// Two Release-mode bundles captured on the audit-remediation branch after
// adding 7 new distributions to the profiler. Values are derived from both
// runs; stability was assessed by comparing V→P crossovers across runs.
//
// Key changes vs v1.5.0 Phase 3 baseline (2026-06-14):
//   - Discrete PDF/LogPDF: 128/100000 -> 250000. Native NEON transcendentals
//     make VECTORIZED fast enough that GCD overhead doesn't pay until 250k.
//   - Discrete CDF: 512 -> NEVER. VECTORIZED beats PARALLEL at all sizes.
//   - Poisson PDF/LogPDF: 64 -> 20000. Same cause: fast NEON exp/log paths.
//   - Poisson CDF: 64 -> 2000.
//   - StudentT CDF: 64 -> 64 (unchanged; 32-128 noisy range, clamped).
//   - 7 new distributions added with Release-mode measurements.
//
// Stability note: GCD thread-pool variability makes crossover detection noisy
// for thresholds in the tens-of-thousands range. Where two runs disagreed
// significantly and best_strategy_at_max_size was consistent, the more
// conservative (larger) crossover value was used. Where best_strategy_at_max_size
// itself disagreed across runs (VonMises PDF), NEVER was used.
constexpr ArchTable kNeon = {
    /* uniform     */ {NEVER, NEVER, 64},
    /* gaussian    */ {64, 64, NEVER},
    /* exponential */ {64, 64, 64},
    /* discrete    */ {250000, 250000, NEVER},
    /* poisson     */ {20000, 20000, 2000},
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {64, 64, 64},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, NEVER},
    /* pareto            */ {100000, 100000, 50000},  // PDF=LogPDF (log-only SIMD; PDF run
                                                      // unstable)
    /* weibull           */ {64, 64, 64},
    /* rayleigh          */ {64, 64, 64},
    /* von_mises         */ {NEVER, NEVER, 64},     // PDF: best@500k disagreed across runs
    /* binomial          */ {NEVER, NEVER, NEVER},  // GCD overhead > lgamma benefit up to 500k
    /* negative_binomial */ {NEVER, NEVER, NEVER},  // GCD overhead > lgamma benefit up to 500k
};

// --- AVX (Intel Ivy Bridge i7-3820QM, 128/256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-15T05-25-42Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
// data/profiles/dispatcher/2026-06-15T05-40-12Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
//
// Two Release-mode bundles captured on fix/audit-remediation. Method: where both
// runs agree on best_strategy_at_max_size, use the larger vectorized_to_parallel
// crossover (64 floor). Where best_strategy_at_max_size disagrees, NEVER is used.
//
// Key changes vs v1.5.0 main baseline (2026-06-15T00-33-56Z):
//   - Gaussian CDF: 50000 -> 64. Both runs show crossover at 8 (clamped to 64
//     floor). Branch changes made parallel competitive earlier on erf-heavy path.
//   - Exponential PDF: 64 -> 100000. Conservative upper bound (runs: 100k vs 8;
//     both WORK_STEALING).
//   - Discrete PDF/LogPDF/CDF: 64/1000/250000 -> 128/100000/100000.
//   - Poisson PDF/CDF: 128/64 -> 50000/50000. Both runs consistent at 50000.
//   - StudentT PDF/LogPDF/CDF: 100000/64/64 -> NEVER. Both runs disagree on
//     PARALLEL vs WORK_STEALING at max size across all three operations.
//   - 7 new distributions: replaced PLACEHOLDERs with Release measurements.
constexpr ArchTable kAvx = {
    /* uniform     */ {NEVER, NEVER, 64},
    /* gaussian    */ {64, 64, 64},
    /* exponential */ {100000, 64, 64},
    /* discrete    */ {128, 100000, 100000},
    /* poisson     */ {50000, 50000, 50000},
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {NEVER, NEVER, NEVER},  // PARALLEL vs WORK_STEALING disagreed across both
                                              // runs
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, 64},
    /* pareto            */ {100000, 64, 250000},
    /* weibull           */ {64, 64, 100000},
    /* rayleigh          */ {64, 64, 64},
    /* von_mises         */ {500000, 64, 64},
    /* binomial          */ {NEVER, NEVER, 50000},  // PDF/LogPDF: VECTORIZED wins at 500k
    /* negative_binomial */ {NEVER, NEVER, 50000},  // PDF/LogPDF: VECTORIZED wins at 500k
};

// --- AVX2+FMA (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-14T19-19-02Z_darwin-x86_64_v1.5-erf-accuracy_sha-c91a348
//
// v1.5.0 Phase 1+2 bundle captured after FMA native exp/log/cos (Phase 1)
// and musl rational polynomial erf (Phase 2) landed on Kaby Lake.
//
// Key changes vs April 2026 baseline:
//   - Gaussian PDF: 50000 -> 100000. FMA exp_avx2 is significantly faster;
//     SIMD stays competitive with parallel all the way to 100k.
//   - StudentT PDF: 100000 -> 250000. Same reason (exp-heavy path improved).
//   - StudentT CDF: NEVER -> 64. Native erf made erf-path heavier per-element;
//     parallel becomes competitive at smaller batches than before.
//   - Most distributions: clamped from measured sub-64 crossovers to 64 floor.
constexpr ArchTable kAvx2 = {
    /* uniform     */ {64, 64, 64},
    /* gaussian    */ {100000, 64, 20000},
    /* exponential */ {64, 64, 64},
    /* discrete    */ {64, 64, 50000},
    /* poisson     */ {64, 20000, 64},
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {250000, 64, 64},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* pareto            */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* weibull           */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* rayleigh          */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* von_mises         */ {512, 512, NEVER},      // PLACEHOLDER — profile with strategy_profile
    /* binomial          */ {512, 512, 512},        // PLACEHOLDER — profile with strategy_profile
    /* negative_binomial */ {512, 512, 512},        // PLACEHOLDER — profile with strategy_profile
};

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// data/profiles/dispatcher/2026-06-14T20-36-11Z_windows-x86_64_v1.5-avx512-transcendentals_sha-14bf1ba
//
// v1.5.0 Phase 4 bundle captured after native 8-wide vector_exp_avx512,
// vector_log_avx512, and vector_erf_avx512 replaced the AVX 4-wide delegations.
//
// Key changes vs April 2026 stale table:
//   - Exponential PDF: 50000 -> NEVER. Native 8-wide exp is so fast VECTORIZED
//     beats PARALLEL at all tested batch sizes.
//   - StudentT PDF/LogPDF: -> NEVER. Same reason; exp-dominated paths are
//     now faster than the parallel + scalar baseline.
//   - Gaussian LogPDF: NEVER (retained). VECTORIZED still best at 500k.
//   - Gaussian PDF: 100000 -> 500000. 8-wide exp widened the SIMD advantage.
//   - Exponential LogPDF: -> 500000 (crossover only just visible at max test size).
//   - StudentT CDF, Gamma PDF/CDF, ChiSquared LogPDF/CDF: clamped to 64 floor.
constexpr ArchTable kAvx512 = {
    /* uniform     */ {100000, 5000, 64},
    /* gaussian    */ {500000, NEVER, 20000},
    /* exponential */ {NEVER, 500000, 250000},
    /* discrete    */ {100000, 250000, 64},
    /* poisson     */ {5000, 20000, 10000},
    /* gamma       */ {64, 100000, 64},
    /* student_t   */ {NEVER, NEVER, 64},
    /* chi_squared */ {250000, 64, 64},
    /* lognormal         */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* pareto            */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* weibull           */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* rayleigh          */ {NEVER, NEVER, NEVER},  // PLACEHOLDER — profile with strategy_profile
    /* von_mises         */ {512, 512, NEVER},      // PLACEHOLDER — profile with strategy_profile
    /* binomial          */ {512, 512, 512},        // PLACEHOLDER — profile with strategy_profile
    /* negative_binomial */ {512, 512, 512},        // PLACEHOLDER — profile with strategy_profile
};

/**
 * @brief Single shared implementation: look up (dist, op) in one ArchTable.
 *
 * Replaces four structurally identical 35-CCN functions. Only the data
 * (kNeon / kAvx / kAvx2 / kAvx512) differs between architectures.
 */
constexpr std::size_t parallelThresholdFromTable(const ArchTable& table, DistributionType dist,
                                                 OperationType op) noexcept {
    if (op == OperationType::BATCH_FIT)
        return BATCH_FIT_MIN;
    if (dist == DistributionType::BETA)
        return NEVER;

    const ThresholdRow* row = nullptr;
    switch (dist) {
        case DistributionType::UNIFORM:
            row = &table.uniform;
            break;
        case DistributionType::GAUSSIAN:
            row = &table.gaussian;
            break;
        case DistributionType::EXPONENTIAL:
            row = &table.exponential;
            break;
        case DistributionType::DISCRETE:
            row = &table.discrete;
            break;
        case DistributionType::POISSON:
            row = &table.poisson;
            break;
        case DistributionType::GAMMA:
            row = &table.gamma;
            break;
        case DistributionType::STUDENT_T:
            row = &table.student_t;
            break;
        case DistributionType::CHI_SQUARED:
            row = &table.chi_squared;
            break;
        case DistributionType::LOG_NORMAL:
            row = &table.lognormal;
            break;
        case DistributionType::PARETO:
            row = &table.pareto;
            break;
        case DistributionType::WEIBULL:
            row = &table.weibull;
            break;
        case DistributionType::RAYLEIGH:
            row = &table.rayleigh;
            break;
        case DistributionType::VON_MISES:
            row = &table.von_mises;
            break;
        case DistributionType::BINOMIAL:
            row = &table.binomial;
            break;
        case DistributionType::NEGATIVE_BINOMIAL:
            row = &table.negative_binomial;
            break;
        default:
            return NEVER;
    }
    switch (op) {
        case OperationType::PDF:
            return row->pdf;
        case OperationType::LOG_PDF:
            return row->log_pdf;
        case OperationType::CDF:
            return row->cdf;
        default:
            return NEVER;
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
