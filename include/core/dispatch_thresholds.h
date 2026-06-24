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

#include "distribution_meta.h"  // kDistributionTypeCount, DistributionType ordering
#include "libstats/platform/simd_policy.h"
#include "performance_dispatcher.h"

#include <array>
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
// BETA, GEOMETRIC, LAPLACE, CAUCHY: NEVER across all entries (no parallel batch).
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
 * Indexed by static_cast<std::size_t>(DistributionType); must match enum order.
 * Adding a distribution requires appending one ThresholdRow to every kXxx
 * table instance below — no switch changes needed.
 * The std::array<ThresholdRow, kDistributionTypeCount> type fixes the table
 * size to exactly kDistributionTypeCount entries. Omitted trailing rows are
 * zero-initialized ({0,0,0}), which is threshold 0 (always parallel) not NEVER;
 * always append explicit {NEVER,NEVER,NEVER} rows for unimplemented distributions.
 */
using ArchTable = std::array<ThresholdRow, kDistributionTypeCount>;

// --- NEON (Apple M1, 128-bit, 8C/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-24T02-38-54Z_darwin-arm64_feat-v2-architecture_sha-fb8e8b6
// data/profiles/dispatcher/2026-06-24T02-44-18Z_darwin-arm64_feat-v2-architecture_sha-fb8e8b6
// data/profiles/dispatcher/2026-06-24T02-49-41Z_darwin-arm64_feat-v2-architecture_sha-fb8e8b6
//
// Three sequential Release-mode bundles on feat/v2-architecture (fb8e8b6).
// Method: see scripts/PROFILING_METHOD.md (canonical; V→P = min(P,WS) < VECT;
// NEVER when best@max = VECTORIZED or SCALAR).
//
// Profiler grid change: previous bundles (sha-2904d63) used a floor of 8 elements;
// these bundles use a floor of 64 (matching the table's minimum encodable threshold).
// Many prior entries encoded as 64 were noise artifacts from sub-64 timing resolution;
// the new grid reveals the true crossovers, which are substantially higher for
// medium-complexity distributions.
//
// GCD warm/cold pool fingerprints identified and discarded:
//   - Beta CDF: {256,2048,128}; R2 cold-pool → discard 2048, result = 256.
//   - ChiSquared CDF: {64,2048,64}; R2 cold-pool → discard 2048, result = 64.
//   - Poisson PDF: {6144,512,512}; R1 warm from build → discard 6144, result = 512.
//   - Rayleigh CDF: {64,25000,10000}; R1 warm from build → discard 64, result = 25000.
//
// Key changes vs prior kNeon (sha-2904d63, 2026-06-22):
//   - Gaussian PDF:        64 → 25000  (prior 64 was sub-floor noise; true crossover)
//   - Exponential PDF:     64 → 50000  (same; {50k,10k,25k} all within OOM → max)
//   - Exponential CDF:     64 → 25000  (same; {25k,25k,10k} → max)
//   - Gamma PDF:           64 → 50000  (same; {50k,10k,8192} within OOM → max)
//   - LogNormal PDF:       64 → 25000  (same; {25k,6144,25k} → max)
//   - StudentT PDF:        64 → 50000  ({25k,50k,25k} → max)
//   - StudentT LogPDF:     64 → 50000  ({25k,50k,25k} → max)
//   - StudentT CDF:        64 → 256    ({64,256,256} → max)
//   - ChiSquared PDF:      64 → 8192   ({8192,4096,4096} → max)
//   - Rayleigh PDF:        64 → 25000  ({25k,25k,6144} → max)
//   - Rayleigh CDF:        64 → 25000  (warm-pool R1 discarded; upper pair {25k,10k} → max)
//   - Rayleigh LogPDF:     64 → 128    ({128,64,64} → max)
//   - Pareto PDF:          64 → 100000 ({50k,100k,25k} → max)
//   - Pareto CDF:       100000 → 50000 ({50k,50k,8192} → max)
//   - Weibull PDF:         64 → 25000  ({25k,25k,8192} → max)
//   - Weibull LogPDF:      64 → 75000  ({25k,75k,8192}; 9.15× within OOM → max)
//   - VonMises PDF:    250000 → 100000 ({100k,50k,100k} → max)
//   - VonMises LogPDF: 250000 → 300000 ({300k,300k,150k} → max)
//   - VonMises CDF:        64 → 128    ({64,64,128} → max)
//   - Discrete PDF:    100000 → 75000  ({50k,75k,50k} → max)
//   - Discrete LogPDF: 100000 → 50000  ({50k,50k,50k} → max)
//   - Poisson PDF:       2000 → 512    (warm-pool R1 discarded; lower pair {512,512} → 512)
//   - Poisson LogPDF:      64 → 1024   ({1024,512,256} → max)
//   - Poisson CDF:         64 → 256    ({128,64,256} → max)
//   - Beta PDF:         NEVER → 512    ({64,512,128}; new GCD-viable crossover)
//   - Beta LogPDF:      NEVER → 256    ({128,256,256} → max)
//   - Beta CDF:         NEVER → 256    (warm-pool R2 discarded; {256,128} lower pair → 256)
//   - Binomial CDF:     NEVER → 64     ({64,64,64}; best@max = parallel all 3 runs)
//   - NegBinomial CDF:  NEVER → 256    ({256,64,256}; best@max = parallel all 3 runs)
constexpr ArchTable kNeon = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {25000, 64, NEVER},
    /* EXPONENTIAL(2)        */ {50000, 64, 25000},
    /* DISCRETE(3)           */ {75000, 50000, NEVER},
    /* POISSON(4)            */ {512, 1024, 256},
    /* GAMMA(5)              */ {50000, 64, 64},
    /* STUDENT_T(6)          */ {50000, 50000, 256},
    /* BETA(7)               */ {512, 256, 256},
    /* CHI_SQUARED(8)        */ {8192, 64, 64},
    /* LOG_NORMAL(9)         */ {25000, 64, NEVER},
    /* PARETO(10)            */ {100000, 50000, 50000},
    /* WEIBULL(11)           */ {25000, 75000, 100000},
    /* RAYLEIGH(12)          */ {25000, 128, 25000},
    /* VON_MISES(13)         */ {100000, 300000, 128},
    /* BINOMIAL(14)          */ {NEVER, NEVER, 64},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 256},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* LAPLACE(17)           */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* CAUCHY(18)            */ {NEVER, NEVER, NEVER},  // not yet profiled
}};

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
constexpr ArchTable kAvx = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {64, 64, 64},
    /* EXPONENTIAL(2)        */ {64, 64, 64},
    /* DISCRETE(3)           */ {100000, 100000, 100000},
    /* POISSON(4)            */ {64, 64, 64},
    /* GAMMA(5)              */ {64, 64, 64},
    /* STUDENT_T(6)          */ {100000, 64, 64},
    /* BETA(7)               */ {NEVER, NEVER, NEVER},
    /* CHI_SQUARED(8)        */ {64, 64, 64},
    /* LOG_NORMAL(9)         */ {64, 64, 64},
    /* PARETO(10)            */ {100000, 64, 250000},
    /* WEIBULL(11)           */ {64, 64, 100000},
    /* RAYLEIGH(12)          */ {64, 64, 64},
    /* VON_MISES(13)         */ {64, 64, 64},
    /* BINOMIAL(14)          */ {NEVER, NEVER, NEVER},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, NEVER},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* LAPLACE(17)           */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* CAUCHY(18)            */ {NEVER, NEVER, NEVER},  // not yet profiled
}};

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
constexpr ArchTable kAvx2 = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {64, 64, 20000},
    /* EXPONENTIAL(2)        */ {64, 64, 64},
    /* DISCRETE(3)           */ {100000, 100000, 64},
    /* POISSON(4)            */ {64, 64, 64},
    /* GAMMA(5)              */ {64, 64, 64},
    /* STUDENT_T(6)          */ {50000, 64, 64},
    /* BETA(7)               */ {NEVER, NEVER, NEVER},
    /* CHI_SQUARED(8)        */ {64, 64, 64},
    /* LOG_NORMAL(9)         */ {64, 64, 64},
    /* PARETO(10)            */ {50000, 64, 100000},
    /* WEIBULL(11)           */ {64, 64, 250000},
    /* RAYLEIGH(12)          */ {64, 64, 64},
    /* VON_MISES(13)         */ {250000, 500000, 64},  // LogPDF ceiling hit; --large advisory
    /* BINOMIAL(14)          */ {NEVER, NEVER, NEVER},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, NEVER},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* LAPLACE(17)           */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* CAUCHY(18)            */ {NEVER, NEVER, NEVER},  // not yet profiled
}};

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// Standard 3-run bundles (64–500k grid):
//   data/profiles/dispatcher/2026-06-23T03-20-53Z_windows-x86_64_feat-v2-architecture_sha-94d522a
//   data/profiles/dispatcher/2026-06-23T03-27-10Z_windows-x86_64_feat-v2-architecture_sha-94d522a
//   data/profiles/dispatcher/2026-06-23T03-34-31Z_windows-x86_64_feat-v2-architecture_sha-94d522a
// Extended 3-run bundles (--large, 64–2M grid):
//   data/profiles/dispatcher/2026-06-23T03-55-00Z_windows-x86_64_feat-v2-architecture_sha-94d522a
//   data/profiles/dispatcher/2026-06-23T04-15-45Z_windows-x86_64_feat-v2-architecture_sha-94d522a
//   data/profiles/dispatcher/2026-06-23T04-36-00Z_windows-x86_64_feat-v2-architecture_sha-94d522a
//
// Method: see scripts/PROFILING_METHOD.md (canonical; sustainability check applied;
// min(P,WS) == PARALLEL on Windows Thread Pool).
// Base: --large derived table (resolves 500k ceiling entries in standard runs).
// Bimodal overrides (warm-pool/cold-pool, per PROFILING_METHOD.md §Bimodal —
// use NEVER or the more conservative threshold):
//   - Exponential LogPDF: standard {NEVER,500k,300k}→500k vs large {64,64,400k}→64;
//     complete flip indicates warm-pool artifact; conservative = NEVER.
//   - Poisson PDF: standard {64,8192,8192}→8192 vs large {8192,64,128}→128;
//     bimodal; conservative = 8192 (standard).
//   - Poisson CDF: standard {256,2048,512}→2048 vs large {64,8192,64}→64;
//     bimodal; conservative = 2048 (standard).
// Ceiling advisories (--large ceiling = 2M):
//   - Pareto PDF: 2M ceiling in 2 of 3 large runs; 2000000 is valid conservative.
//   - Gaussian PDF/LogPDF: resolved at 1M in large runs (were 500k ceiling or NEVER
//     in standard runs).
//   - Weibull CDF: 1500000 emerged only in large runs; not visible below 500k.
//   - Pareto LogPDF: 1000000 emerged only in large runs.
// Beta: first real thresholds on any SIMD tier; Windows Thread Pool overhead
//   amortises earlier than GCD for the expensive incomplete-beta path (kAvx2=NEVER).
// Binomial PDF/LogPDF: NEVER — VECTORIZED dominates at max size in large runs.
constexpr ArchTable kAvx512 = {{
    /* UNIFORM(0)            */ {50000, 50000, 256},
    /* GAUSSIAN(1)           */ {1000000, 1000000, 50000},
    /* EXPONENTIAL(2)        */ {500000, NEVER, 300000},  // LogPDF bimodal → NEVER
    /* DISCRETE(3)           */ {200000, 200000, 75000},
    /* POISSON(4)            */ {8192, 25000, 2048},  // PDF/CDF bimodal → standard
    /* GAMMA(5)              */ {150000, 150000, 64},
    /* STUDENT_T(6)          */ {NEVER, NEVER, 256},
    /* BETA(7)               */ {256, 128, 6144},
    /* CHI_SQUARED(8)        */ {150000, 150000, 128},
    /* LOG_NORMAL(9)         */ {150000, 150000, 50000},
    /* PARETO(10)            */ {2000000, 1000000, NEVER},  // PDF at 2M ceiling
    /* WEIBULL(11)           */ {150000, 150000, 1500000},  // CDF emerged in large
    /* RAYLEIGH(12)          */ {150000, 150000, 300000},
    /* VON_MISES(13)         */ {50000, 100000, 64},
    /* BINOMIAL(14)          */ {NEVER, NEVER, 128},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 2048},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* LAPLACE(17)           */ {NEVER, NEVER, NEVER},  // not yet profiled
    /* CAUCHY(18)            */ {NEVER, NEVER, NEVER},  // not yet profiled
}};

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
    const auto idx = static_cast<std::size_t>(dist);
    if (idx >= table.size())
        return NEVER;
    const ThresholdRow& row = table[idx];
    switch (op) {
        case OperationType::PDF:
            return row.pdf;
        case OperationType::LOG_PDF:
            return row.log_pdf;
        case OperationType::CDF:
            return row.cdf;
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

// --- No SIMD: conservative parallel thresholds ---
// Without SIMD, VECTORIZED is just a scalar loop; parallel helps earlier
// because there is no SIMD advantage to protect.
// Threshold: 8192 doubles = 64 KB, which exceeds L1d cache (32-64 KB) on all
// target CPUs. Below this the scalar loop stays in L1 and threading overhead
// is hard to amortise; above it L2 latency makes parallel competitive.
// Unprofiled — treat as a principled placeholder until a no-SIMD build is
// measured with strategy_profile. BETA: no parallel batch.
// GEOMETRIC/LAPLACE/CAUCHY: pending implementation.
constexpr ArchTable kNone = {{
    /* UNIFORM(0)            */ {8192, 8192, 8192},
    /* GAUSSIAN(1)           */ {8192, 8192, 8192},
    /* EXPONENTIAL(2)        */ {8192, 8192, 8192},
    /* DISCRETE(3)           */ {8192, 8192, 8192},
    /* POISSON(4)            */ {8192, 8192, 8192},
    /* GAMMA(5)              */ {8192, 8192, 8192},
    /* STUDENT_T(6)          */ {8192, 8192, 8192},
    /* BETA(7)               */ {NEVER, NEVER, NEVER},  // no parallel batch
    /* CHI_SQUARED(8)        */ {8192, 8192, 8192},
    /* LOG_NORMAL(9)         */ {8192, 8192, 8192},
    /* PARETO(10)            */ {8192, 8192, 8192},
    /* WEIBULL(11)           */ {8192, 8192, 8192},
    /* RAYLEIGH(12)          */ {8192, 8192, 8192},
    /* VON_MISES(13)         */ {8192, 8192, 8192},
    /* BINOMIAL(14)          */ {8192, 8192, 8192},
    /* NEGATIVE_BINOMIAL(15) */ {8192, 8192, 8192},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // pending implementation
    /* LAPLACE(17)           */ {NEVER, NEVER, NEVER},  // pending implementation
    /* CAUCHY(18)            */ {NEVER, NEVER, NEVER},  // pending implementation
}};
constexpr std::size_t none_parallel_threshold(DistributionType dist, OperationType op) {
    return parallelThresholdFromTable(kNone, dist, op);
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
