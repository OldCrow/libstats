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
// data/profiles/dispatcher/2026-06-22T05-19-18Z_darwin-arm64_feat-v2-architecture_sha-2904d63
// data/profiles/dispatcher/2026-06-22T05-21-44Z_darwin-arm64_feat-v2-architecture_sha-2904d63
// data/profiles/dispatcher/2026-06-22T05-24-10Z_darwin-arm64_feat-v2-architecture_sha-2904d63
//
// Three sequential Release-mode bundles captured on feat/v2-architecture after
// the v2.0.0 remediation and dispatch-path cleanup work.
// Method: see scripts/PROFILING_METHOD.md (canonical; V→P = min(P,WS) < VECT;
// NEVER when best@max = VECTORIZED or SCALAR).
//
// Corrections applied relative to original encoding (buggy PARALLEL-only V→P):
//   - Uniform LogPDF: 1000 → NEVER. All 3 runs: best@max = VECTORIZED (parallel
//     crossed briefly but didn’t sustain). Original encoding used a transient
//     PARALLEL crossover that the sustainability check correctly rejects.
//   - Discrete CDF: 64 → NEVER. All 3 runs: best@max = VECTORIZED.
//   - Pareto CDF: 50000 → 100000. Three-run rule: {20k,50k,100k} all within
//     OOM → max = 100000. Prior encoding was manually held at 50k; removed.
//   - Weibull CDF: 64 → 100000. Three-run rule: {50k,64,100k}; discard outlier
//     64 (warm GCD pool), max of coherent pair {50k,100k} = 100000. Prior hold
//     at 64 misidentified the warm-pool run as the representative value.
//   - Binomial CDF: 64 → NEVER. All 3 runs: best@max = VECTORIZED.
//   - NegBinomial CDF: 128 → NEVER. All 3 runs: best@max = VECTORIZED/SCALAR.
constexpr ArchTable kNeon = {
    /* uniform     */ {64, NEVER, 64},
    /* gaussian    */ {64, 64, NEVER},
    /* exponential */ {64, 64, 64},
    /* discrete    */ {100000, 100000, NEVER},
    /* poisson     */ {2000, 64, 64},
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {64, 64, 64},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, NEVER},
    /* pareto            */ {64, 50000, 100000},
    /* weibull           */ {64, 64, 100000},
    /* rayleigh          */ {64, 64, 64},
    /* von_mises         */ {250000, 250000, 64},
    /* binomial          */ {NEVER, NEVER, NEVER},
    /* negative_binomial */ {NEVER, NEVER, NEVER},
};

// --- AVX (Intel Ivy Bridge i7-3820QM, 128/256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-15T05-25-42Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
// data/profiles/dispatcher/2026-06-15T05-40-12Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
//
// Two Release-mode bundles on fix/audit-remediation (Ivy Bridge retired; hardware
// no longer in the ecosystem).  Original encoding used the buggy PARALLEL-only
// V→P definition and predates the v2.0.0 dispatch-path improvements.
//
// Inferred updates applied (architecture-independent dispatch improvements from
// v2.0.0; same code paths affect AVX as AVX2; see kAvx2 for measured evidence):
//   - Exponential PDF: 100000 → 64  (kAvx2 measured 64; parallel competitive
//     much earlier on fixed dispatch paths)
//   - Discrete PDF: 128 → 100000   (128 was a PARALLEL-only noise artifact;
//     kNeon measured 100000 stably; inferred same for kAvx)
//   - Poisson: {50000,50000,50000} → {64,64,64}  (kAvx2 measured 64; dispatch
//     path fix makes parallel competitive much earlier)
//   - VonMises PDF: 500000 → 64    (kAvx2 measured 64; kAvx512 measured 64;
//     architecture-independent GCD improvement)
//   - Binomial CDF: 50000 → NEVER  (kAvx2 and kNeon both show best@max =
//     VECTORIZED; inferred same sustainability failure for kAvx)
//   - NegBinomial CDF: 50000 → NEVER  (same reasoning as Binomial CDF)
// SSE2 delegates to kAvx (line 364); both are updated together.
constexpr ArchTable kAvx = {
    /* uniform     */ {NEVER, NEVER, 64},
    /* gaussian    */ {64, 64, 64},
    /* exponential */ {64, 64, 64},    // PDF: inferred 100000→64
    /* discrete    */ {100000, 100000, 100000},  // PDF: inferred 128→100000
    /* poisson     */ {64, 64, 64},    // inferred from kAvx2 (was 50000)
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {100000, 64, 64},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, 64},
    /* pareto            */ {100000, 64, 250000},
    /* weibull           */ {64, 64, 100000},
    /* rayleigh          */ {64, 64, 64},
    /* von_mises         */ {64, 64, 64},   // PDF: inferred 500000→64
    /* binomial          */ {NEVER, NEVER, NEVER},  // CDF: inferred 50000→NEVER
    /* negative_binomial */ {NEVER, NEVER, NEVER},  // CDF: inferred 50000→NEVER
};

// --- AVX2+FMA (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-22T21-15-44Z_darwin-x86_64_feat-v2-architecture_sha-909dc9a
// data/profiles/dispatcher/2026-06-22T21-25-20Z_darwin-x86_64_feat-v2-architecture_sha-909dc9a
// data/profiles/dispatcher/2026-06-22T21-33-55Z_darwin-x86_64_feat-v2-architecture_sha-909dc9a
//
// Three sequential Release-mode bundles captured on feat/v2-architecture (909dc9a).
// Method: see scripts/PROFILING_METHOD.md (canonical; V→P = min(P,WS) < VECT;
// NEVER when best@max = VECTORIZED or SCALAR).
//
// Note: initial encoding of this table used a buggy PARALLEL-only V→P definition.
// Corrected values below use min(PARALLEL, WORK_STEALING) < VECTORIZED, which
// revealed WS was beating VECTORIZED at much lower thresholds than PARALLEL was.
// This explains why many entries that appeared as 64 ("blank clamped to floor")
// were actually 50k–250k range: WS was crossing early; PARALLEL was not.
//
// Manual override applied after review:
//   - Discrete LogPDF: bimodal {64, 100k, 100k}; 64 is the warm-pool outlier.
//     Two-run majority at 100k, same as Discrete PDF; encoded as 100000.
//   - VonMises LogPDF: R3 = 500000 (measurement ceiling; advisory printed).
//     500000 is conservative and valid; re-run with --large to refine if needed.
//
// Key changes vs prior kAvx2 (fix/audit-remediation sha-5675c93, 2 runs):
//   - Gaussian CDF: 20000 → 20000  (corrected: WS at {10k,20k,10k} vs old
//     PARALLEL-only {100k,50k,100k}; conserved max within OOM = 20000)
//   - Discrete PDF: NEVER → 100000 (corrected: WS at 100k all 3 runs; old
//     PARALLEL-only showed bimodal {blank/500k/500k} = artifact)
//   - Discrete LogPDF: NEVER → 100000 (corrected bimodal; see manual override)
//   - Discrete CDF: NEVER → 64 (three-run consensus; best@max = WS)
//   - Poisson PDF/LogPDF/CDF: NEVER → 64 (parallel competitive at floor)
//   - StudentT PDF: NEVER → 50000 (corrected WS at {50k,20k,50k}; old
//     PARALLEL-only showed {250k,250k,100k})
//   - Pareto PDF: NEVER → 50000 (corrected WS at {50k,20k,50k})
//   - Pareto LogPDF: NEVER → 64 (WS at floor; consistent 3 runs)
//   - Pareto CDF: NEVER → 100000 (WS at {100k,100k,20k})
//   - Weibull CDF: NEVER → 250000 (corrected WS at {50k,250k,50k})
//   - VonMises PDF: NEVER → 250000 (corrected WS at {250k,250k,100k})
//   - VonMises LogPDF: NEVER → 500000 (WS at {250k,250k,500k}; ceiling hit)
//   - Binomial CDF: 64 → NEVER (best@max = VECTORIZED all 3 runs)
constexpr ArchTable kAvx2 = {
    /* uniform     */ {NEVER, NEVER, 64},
    /* gaussian    */ {64, 64, 20000},
    /* exponential */ {64, 64, 64},
    /* discrete    */ {100000, 100000, 64},
    /* poisson     */ {64, 64, 64},
    /* gamma       */ {64, 64, 64},
    /* student_t   */ {50000, 64, 64},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, 64},
    /* pareto            */ {50000, 64, 100000},
    /* weibull           */ {64, 64, 250000},
    /* rayleigh          */ {64, 64, 64},
    /* von_mises         */ {250000, 500000, 64},  // LogPDF ceiling hit; --large advisory
    /* binomial          */ {NEVER, NEVER, NEVER},
    /* negative_binomial */ {NEVER, NEVER, NEVER},
};

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// data/profiles/dispatcher/2026-06-22T02-52-00Z_windows-x86_64_feat-v2-architecture_sha-9b2c1a3
// data/profiles/dispatcher/2026-06-22T02-55-00Z_windows-x86_64_feat-v2-architecture_sha-9b2c1a3
// data/profiles/dispatcher/2026-06-22T02-59-00Z_windows-x86_64_feat-v2-architecture_sha-9b2c1a3
//
// Three sequential Release-mode runs on feat/v2-architecture (9b2c1a3).
// Method: PARALLEL-only V→P (pre-correction). Needs re-validation with the
// canonical min(P,WS) method (scripts/PROFILING_METHOD.md). The raw bundles
// are missing strategy_profile_results.csv; re-run on Windows to regenerate.
// Values below are best-available and not expected to change dramatically
// (Windows/Thread Pool uses only PARALLEL, so min(P,WS) == PARALLEL for
// Windows), but the sustainability check (NEVER when best@max=VECTORIZED) was
// not applied. See PROFILING_METHOD.md § Known Issues.
//
// Method (as applied at time of encoding): clamp < 64 → 64; all three within 10× → max (conservative);
// two within 10× + one outlier → discard outlier, max of valid pair; else NEVER.
// Windows/Thread Pool always dispatches PARALLEL (not WORK_STEALING); variation
// between PARALLEL and WORK_STEALING at max size is ignored for threshold derivation.
//
// Key changes vs prior kAvx512 (fix/audit-remediation sha-932addd):
//   - Uniform LogPDF: 1000 → 100000 (parallel competitive much later on v2 paths)
//   - Uniform CDF:    256 → 64
//   - Gaussian LogPDF: NEVER → 64; Gaussian CDF: NEVER → 20000
//   - Exponential LogPDF: NEVER → 64; Exponential CDF: NEVER → 500000
//   - Discrete: all NEVER → {64, 250000, 100000} (parallel now consistently competitive)
//   - Poisson: {NEVER, 50000, NEVER} → {64, 128, 256} (parallel competitive much earlier)
//   - ChiSquared PDF: NEVER → 64
//   - LogNormal PDF/LogPDF: NEVER → 64
//   - Rayleigh CDF: 64 → 500000 (vectorized dominant at medium sizes; parallel only at 500k)
//   - VonMises PDF: NEVER → 64; VonMises CDF: 256 → 64
//   - Binomial: {NEVER, NEVER, 64} → {2000, 50000, 128}
//   - NegBinomial CDF: 128 → 512
//   - Weibull LogPDF: 250000 → 64
//   - Gamma {250000, 64, 64}: unchanged — perfectly stable across all three runs
//   - StudentT PDF/LogPDF, Pareto all, Weibull CDF: NEVER unchanged
constexpr ArchTable kAvx512 = {
    /* uniform     */ {NEVER, 100000, 64},
    /* gaussian    */ {NEVER, 64, 20000},
    /* exponential */ {500000, 64, 500000},
    /* discrete    */ {64, 250000, 100000},
    /* poisson     */ {64, 128, 256},
    /* gamma       */ {250000, 64, 64},
    /* student_t   */ {NEVER, NEVER, 256},
    /* chi_squared */ {64, 64, 64},
    /* lognormal         */ {64, 64, 64},
    /* pareto            */ {NEVER, NEVER, NEVER},
    /* weibull           */ {250000, 64, NEVER},
    /* rayleigh          */ {64, 64, 500000},
    /* von_mises         */ {64, 100000, 64},
    /* binomial          */ {2000, 50000, 128},
    /* negative_binomial */ {NEVER, NEVER, 512},
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
