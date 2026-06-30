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
// GEOMETRIC: NEVER on measured SIMD tiers when vectorized/delegate path stays dominant.
// LAPLACE/CAUCHY: measured on kAvx2/kAvx512 (sha-1b564ec); kAvx inferred.
// BETA: NEVER on kAvx (retired hardware, no re-measure); real thresholds on
// kNeon, kAvx2, kAvx512 (GCD/Thread-Pool overhead amortises for expensive
// incomplete-beta path at ~256–512 elements).
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
//   - Uniform PDF:         64 → NEVER  (VECTORIZED dominant at all profiled sizes; no crossover in
//   all 3 runs)
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
// V→P definition; values below are inferred from recalibrated kNeon (fb8e8b6)
// and kAvx2 (fb8e8b6) reference tables.
//
// Inference principle:
//   kAvx VECTORIZED lacks FMA — roughly 2× less efficient than kAvx2 for
//   transcendental operations.  GCD pool overhead is identical (same OS).
//   For SIMD-bound distributions: threshold ≈ kAvx2/2.
//   For GCD-overhead-dominated distributions (VonMises, Discrete): ≈ kNeon.
//   For compute-bound distributions (Beta, StudentT, NegBinomial): ≈ kNeon
//   (SIMD advantage smaller for iterative algorithms; GCD cost dominates).
//
// Key inference updates vs prior kAvx (sha-65b1c61, stale references):
//   - Gaussian PDF/CDF:     64 → 25000/10000  (kAvx2 recalibrated to 50k/25k ÷ 2)
//   - Exponential PDF/CDF:  64 → 25000/25000  (kAvx2=50k ÷ 2)
//   - Discrete:      100000 → {75k,50k,25k}   (match kNeon/kAvx2; non-SIMD-bound)
//   - Poisson:           64 → 128             (kAvx2 recalibrated; floor noise)
//   - Gamma PDF:          64 → 10000          (kAvx2=25k ÷ 2)
//   - StudentT:      100000 → 25000/10000     (kAvx2 recalibrated ÷ 2)
//   - Beta:           NEVER → {512,512,512}   (kNeon=kAvx2=512; compute-bound)
//   - ChiSquared PDF:    64 → 25000           (kAvx2=50k ÷ 2)
//   - LogNormal PDF/CDF: 64 → 10000/6144     (kAvx2=25k/10k ÷ 2 approx)
//   - Pareto:    100000/64 → 50000/50000      (kNeon reference; log-pipeline)
//   - Weibull:           64 → 25000/25000/50k (kAvx2 ÷ 2 approx)
//   - Rayleigh:          64 → 25000/64/50000  (kAvx2 ÷ 2 approx)
//   - VonMises:          64 → 100k/300k/128   (GCD-dominated; match kNeon)
//   - Binomial CDF:   NEVER → 128            (kNeon=64, kAvx512=128; infer 128)
//   - NegBinomial CDF:NEVER → 256            (kNeon=256; kAvx closer to kNeon)
//   - Geometric:       stays NEVER            (delegates to NegBinomial; kAvx2=NEVER)
//   - Laplace:       NEVER → {64,64,128}      (kAvx2 64/64/256 inferred ÷2; floor-clamped)
//   - Cauchy:        NEVER → {50k,50k,64}     (kAvx2 75k/75k/128 inferred ÷2)
//   Note: Binomial/NegBinomial PDF/LogPDF remain NEVER (kAvx2=NEVER; kAvx
//   VECTORIZED not weaker enough to change this given GCD overhead).
// SSE2 delegates to kAvx; both updated together.
constexpr ArchTable kAvx = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {25000, 64, 10000},
    /* EXPONENTIAL(2)        */ {25000, 64, 25000},
    /* DISCRETE(3)           */ {75000, 50000, 25000},
    /* POISSON(4)            */ {128, 128, 128},
    /* GAMMA(5)              */ {10000, 64, 64},
    /* STUDENT_T(6)          */ {25000, 10000, 64},
    /* BETA(7)               */ {512, 512, 512},  // compute-bound; kNeon=kAvx2=512
    /* CHI_SQUARED(8)        */ {25000, 64, 64},
    /* LOG_NORMAL(9)         */ {10000, 64, 6144},
    /* PARETO(10)            */ {50000, 50000, 100000},  // log-pipeline; kNeon reference
    /* WEIBULL(11)           */ {25000, 25000, 50000},
    /* RAYLEIGH(12)          */ {25000, 64, 50000},
    /* VON_MISES(13)         */ {100000, 300000, 128},  // GCD-dominated; match kNeon
    /* BINOMIAL(14)          */ {NEVER, NEVER, 128},    // CDF inferred from kNeon/kAvx512
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 256},    // CDF inferred from kNeon
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // inferred from kAvx2/NegBinomial
    /* LAPLACE(17)           */ {64, 64, 128},           // kAvx2 ÷ 2; PDF/LogPDF floor-clamped
    /* CAUCHY(18)            */ {50000, 50000, 64},      // kAvx2 75k/75k/128 inferred ÷ 2
}};

// --- AVX2+FMA (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// Standard 3-run bundles (64–500k grid):
//   data/profiles/dispatcher/2026-06-23T23-22-49Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
//   data/profiles/dispatcher/2026-06-23T23-37-49Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
//   data/profiles/dispatcher/2026-06-23T23-51-53Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
// Extended 3-run bundles (--large, 64–2M grid):
//   data/profiles/dispatcher/2026-06-24T00-28-01Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
//   data/profiles/dispatcher/2026-06-24T01-21-12Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
//   data/profiles/dispatcher/2026-06-24T02-18-20Z_darwin-x86_64_feat-v2-architecture_sha-fb8e8b6
//
// Method: see scripts/PROFILING_METHOD.md (canonical; sustainability check applied).
// Base: --large derived table; standard used conservatively where standard > large
// (standard runs see a colder GCD pool; --large runs see a warming pool from prior
// distributions, yielding lower thresholds that can understate the cold-start cost).
//
// Primary finding: many entries encoded as 64 in sha-909dc9a were GCD warm-pool
// artifacts. Both standard and large runs on this sha consistently show 25k–75k
// true cold-pool crossovers for medium-complexity SIMD distributions.
//
// Bimodal overrides (standard vs large disagree or within-run spread > OOM):
//   - Discrete CDF: standard {25k,25k,25k} vs large {25k,2k,1k};
//     large shows warm-pool drop; use standard → 25000.
//   - Pareto LogPDF: bimodal in both runs; lower-pair method gave 512, but 512 is
//     sub-floor noise (kNeon=50k, kAvx512=1M confirm kAvx2≥50k); manual override 50000.
//   - VonMises PDF: large R2 BEST=VECTORIZED (NEVER); finite pair {25k,100k}→100k;
//     standard {150k,75k,200k}→200k is more conservative → 200000.
//
// Beta: first real threshold on kAvx2 (was NEVER — incorrectly assumed no parallel
//   batch). All 6 runs show BEST = WS or PARALLEL; crossover varies 64–8192 due to
//   GCD cold/warm state. Encoded as conservative max(standard, large) = 512.
//   kNeon (512/256/256) and kAvx512 (256/128/6144) confirm cross-architecture.
//
// VonMises LogPDF: large R1 V→P=500000 (2M run ceiling); standard 2/3 runs at
//   400000, 1 NEVER. Aggregate: {500k,200k,400k} → max=500000 (unchanged).
//
// New-distribution --large calibration bundles (64–2M grid, sha-1b564ec):
//   data/profiles/dispatcher/2026-06-29T22-32-28Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-29T23-28-13Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-30T00-24-23Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//
// Encoded from the 3-run set above:
//   - Geometric: {NEVER,NEVER,NEVER}; matches NegBinomial delegate behavior.
//     PDF/LogPDF best@max=VECTORIZED all runs; CDF best=SCALAR/VECTORIZED → discard.
//   - Laplace:   {64,64,256}; PDF/LogPDF at floor, CDF {256,128,64} → 256.
//   - Cauchy:    {75000,75000,128}; wrapper overhead vs StudentT is plausible.
//
// The same 2026-06-29/30 bundles show broader kAvx2 movement in existing rows
// (not encoded here).  Several deltas are likely real after post-fb8e8b6 SIMD and
// batch-path changes, but rows such as Gamma PDF, ChiSquared PDF, LogNormal CDF,
// Discrete CDF, Weibull LogPDF, and Beta PDF/LogPDF need targeted review because
// profile-order warm-pool effects can propagate inside a bundle even with 60 s
// sleep between bundles.  Keep the current conservative table until a full kAvx2
// recalibration applies kAvx512-style manual overrides.
//
// Key changes vs prior kAvx2 (sha-909dc9a, standard-only 3-run):
//   - Gaussian PDF:     64 → 50000  (cold-pool; {50k,50k,50k} all runs)
//   - Gaussian CDF:  20000 → 25000  (cold-pool; standard max=10k, large max=25k)
//   - Exponential PDF:  64 → 50000  (cold-pool; consistent standard + large)
//   - Exponential CDF:  64 → 50000  (cold-pool; consistent)
//   - Discrete PDF: 100000 → 75000  ({75k,50k,50k} large; standard was 50k → max=75k)
//   - Discrete CDF:     64 → 25000  (standard {25k,25k,25k}; warm-pool large discarded)
//   - Poisson PDF:      64 → 128    (floor noise; max across runs)
//   - Poisson LogPDF:   64 → 128    (floor noise; max across runs)
//   - Poisson CDF:      64 → 128    (floor noise; max across runs)
//   - Gamma PDF:        64 → 25000  (cold-pool; {25k,25k,10k} large)
//   - StudentT LogPDF:  64 → 25000  (cold-pool; {25k,25k,25k} consistent)
//   - Beta:          NEVER → {512,512,512}  (new finding — see note above)
//   - ChiSquared PDF:   64 → 50000  (cold-pool; standard max=50k)
//   - LogNormal PDF:    64 → 25000  (cold-pool; {25k,25k,25k} standard)
//   - LogNormal CDF:    64 → 10000  (cold-pool; {6k,6k,10k} consistent)
//   - Pareto LogPDF:    64 → 50000  (bimodal noise-floor; manual override — see note above)
//   - Pareto CDF:   100000 → 150000 (standard {50k,50k,150k} → max)
//   - Weibull PDF:      64 → 75000  (cold-pool; large {50k,50k,75k} → max)
//   - Weibull LogPDF:   64 → 50000  (cold-pool; large {50k,10k,10k} → max)
//   - Weibull CDF:  250000 → 75000  (was over-conservative; standard {50k,50k,50k},
//     large {50k,75k,50k} → max=75k)
//   - Rayleigh PDF:     64 → 50000  (cold-pool; {25k,25k,50k} standard)
//   - Rayleigh CDF:     64 → 75000  (cold-pool; standard {75k,25k,25k} → max)
//   - VonMises PDF: 250000 → 200000 (standard max; large R2 unreliable)
//   - VonMises CDF:     64 → 128    ({128,64,64} standard → max)
constexpr ArchTable kAvx2 = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {50000, 64, 25000},
    /* EXPONENTIAL(2)        */ {50000, 64, 50000},
    /* DISCRETE(3)           */ {75000, 100000, 25000},  // CDF: warm-pool large discarded
    /* POISSON(4)            */ {128, 128, 128},
    /* GAMMA(5)              */ {25000, 64, 64},
    /* STUDENT_T(6)          */ {50000, 25000, 64},
    /* BETA(7)               */ {512, 512, 512},  // was NEVER — see Beta note above
    /* CHI_SQUARED(8)        */ {50000, 64, 64},
    /* LOG_NORMAL(9)         */ {25000, 64, 10000},
    /* PARETO(10)            */ {50000, 50000, 150000},  // LogPDF: 512 was bimodal noise-floor
    //   artifact; manual override to 50000 (kNeon=50k; kAvx2 VECTORIZED log≥kNeon → threshold≥50k)
    /* WEIBULL(11)           */ {75000, 50000, 75000},  // CDF: 250k was over-conservative
    /* RAYLEIGH(12)          */ {50000, 64, 75000},
    /* VON_MISES(13)         */ {200000, 500000, 128},  // LogPDF: 2M ceiling confirmed
    /* BINOMIAL(14)          */ {NEVER, NEVER, NEVER},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, NEVER},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // delegates to NegBinomial; no sustained parallel win
    /* LAPLACE(17)           */ {64, 64, 256},           // new sha-1b564ec kAvx2 run
    /* CAUCHY(18)            */ {75000, 75000, 128},     // delegates to StudentT(nu=1)
}};

// --- AVX-512 (AMD Ryzen 7 7445HS Zen 4, 512-bit, 6P/12T, Windows/MSVC) ---
// Extended 3-run bundles (--large, 64–2M grid, sha-1b564ec):
//   data/profiles/dispatcher/2026-06-29T23-00-05Z_windows-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-29T23-23-37Z_windows-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-29T23-47-18Z_windows-x86_64_feat-v2-architecture_sha-1b564ec
//
// Three sequential --large Release-mode bundles on feat/v2-architecture (1b564ec).
// Method: see scripts/PROFILING_METHOD.md (canonical; min(P,WS) < VECT definition).
// Supersedes all six sha-94d522a bundles; covers Geometric, Laplace, Cauchy (new
// distributions) and resolves all prior ceiling advisories via --large grid.
//
// Bimodal overrides (per PROFILING_METHOD.md §Bimodal — use conservative threshold):
//   - Uniform PDF:     {256,1024,50000}; 195×; R1/R2 warm-pool within run; hold 50000.
//   - Gaussian LogPDF: {400k,64,64}; 6250×; R2/R3 warm-pool; override 400000 (R1 cold).
//   - Poisson LogPDF:  {64,25k,25k}; 390×; upper pair agrees → 25000 (algorithm applied;
//     unchanged from prior table).
//   - Geometric CDF:   {64,8192,256}; 128×; algorithm=256 (lower pair b/a=4≤10);
//     accepted — delegation wrapper over NegBinomial(r=1); new measurement vs prior NEVER.
//
// Ceiling advisories (--large 2M ceiling):
//   - StudentT PDF/LogPDF: 2M in all 3 runs; crossover ≤2M; encoded 2000000.
//   - Cauchy PDF: {2M,750k,300k}; hi/lo=6.67× → 2000000 (delegates StudentT(ν=1)).
//   - Pareto CDF: {2M,NEVER,2M}; 2-finite hi/lo=1 → 2000000.
//   - Weibull CDF: {750k,2M,1.5M} → 2000000.
//
// Key changes vs prior kAvx512 (sha-94d522a):
//   - Exponential LogPDF: NEVER    → 400000   (prior bimodal override superseded;
//     all 3 large runs: BEST=WS/PAR at {250k,400k,250k}; real crossover confirmed)
//   - StudentT PDF:       NEVER    → 2000000  (crossover emerges at 2M in large runs)
//   - StudentT LogPDF:    NEVER    → 2000000  ({2M,2M,1.5M} → max)
//   - StudentT CDF:         256    → NEVER    (BEST=VECTORIZED in 2/3 large runs)
//   - Pareto CDF:         NEVER    → 2000000  (crossover at 2M in 2/3 large runs)
//   - Pareto LogPDF:    1000000    → 1500000  ({1M,1.5M,1.5M} → max)
//   - Weibull CDF:      1500000    → 2000000  ({750k,2M,1.5M} → max)
//   - Uniform PDF:        50000    → 50000    (bimodal override; see note above)
//   - Uniform CDF:          256    → 128      ({64,128,1024} → 128)
//   - Gaussian LogPDF: 1000000     → 400000   (bimodal override; see note above)
//   - Gaussian CDF:       50000    → 25000    ({25k,256,25k}; upper pair → 25000)
//   - Exponential PDF:   500000    → 250000   ({100k,250k,200k} → max)
//   - Exponential CDF:   300000    → 250000   ({250k,200k,250k} → max)
//   - Discrete PDF:      200000    → 150000   (algorithm gave 512; held — profiling-order
//     warm-pool artifact: PDF phase runs before LogPDF phase within each bundle;
//     LogPDF in same runs reads {150k,75k,100k}; aligning PDF with LogPDF)
//   - Discrete LogPDF:   200000    → 150000   ({150k,75k,100k} → max)
//   - Discrete CDF:       75000    → NEVER    ({128,2048,75k} mutually incoherent)
//   - Poisson PDF:          8192   → 512      ({64,64,512}; prior bimodal superseded)
//   - Poisson CDF:          2048   → 256      ({128,128,256}; prior bimodal superseded)
//   - Gamma PDF:          150000   → 10000    ({1024,10k,1024} → max)
//   - Gamma LogPDF:       150000   → 256      ({64,64,256} → max)
//   - ChiSquared PDF:     150000   → 1024     ({512,1024,512} → max)
//   - ChiSquared LogPDF:  150000   → 2048     ({2048,512,256} → max)
//   - LogNormal CDF:       50000   → 2048     ({128,2048,2048}; upper pair → 2048)
//   - Beta PDF:              256   → 2048     ({2048,512,256} → max)
//   - Beta LogPDF:           128   → 512      ({6144,512,64}; lower pair b/a=8 → 512)
//   - Beta CDF:             6144   → 8192     ({8192,2048,6144} → max)
//   - VonMises PDF:        50000   → 25000    ({25k,10k,25k} → max)
//   - VonMises LogPDF:    100000   → 75000    ({25k,75k,75k} → max)
//   - VonMises CDF:           64   → 128      ({128,64,64} → max)
//   - NegBinomial CDF:      2048   → 512      ({64,6144,512}; lower pair b/a=8 → 512)
//   - Geometric PDF/LogPDF: NEVER  → NEVER    (BEST=VECTORIZED all 3 runs)
//   - Geometric CDF:        NEVER  → 512      (new; 6-run set with NegBinomial CDF:
//     combined {64,6144,512,64,8192,256}; R2 cold-pool cluster {6144,8192} vs R1/R3
//     warm cluster {64,64,256,512}; max of warm cluster = 512 = NegBinomial alg result)
//   - Laplace PDF:          NEVER  → 64       (new; {64,64,64}; warm-pool floor confirmed)
//   - Laplace LogPDF:       NEVER  → 64       (new; same)
//   - Laplace CDF:          NEVER  → 1024     (new; {128,1024,2048}; lower pair → 1024)
//   - Cauchy PDF:           NEVER  → 2000000  (new; {2M,750k,300k} → max)
//   - Cauchy LogPDF:        NEVER  → 750000   (new; {250k,750k,400k} → max)
//   - Cauchy CDF:           NEVER  → NEVER    (new; 6-run set with StudentT CDF:
//     combined {64,NEVER,NEVER,NEVER,64,256}; 50/50 finite/NEVER split with
//     anti-correlated pool state — conservative = NEVER)
constexpr ArchTable kAvx512 = {{
    /* UNIFORM(0)            */ {50000, 50000, 128},      // CDF: 256→128
    /* GAUSSIAN(1)           */ {1000000, 400000, 25000}, // LogPDF bimodal override; CDF: 50k→25k
    /* EXPONENTIAL(2)        */ {250000, 400000, 250000}, // LogPDF: NEVER→400k; PDF/CDF reduced
    /* DISCRETE(3)           */ {150000, 150000, NEVER},   // PDF: held 150000 (512 was profiling-order warm-pool; aligns with LogPDF same runs); CDF: 75k→NEVER
    /* POISSON(4)            */ {512, 25000, 256},        // PDF: 8192→512; CDF: 2048→256
    /* GAMMA(5)              */ {10000, 256, 64},         // PDF: 150k→10k; LogPDF: 150k→256
    /* STUDENT_T(6)          */ {2000000, 2000000, NEVER}, // PDF/LogPDF: NEVER→2M; CDF: 256→NEVER
    /* BETA(7)               */ {2048, 512, 8192},        // PDF: 256→2048; LogPDF: 128→512; CDF: 6144→8192
    /* CHI_SQUARED(8)        */ {1024, 2048, 128},        // PDF: 150k→1024; LogPDF: 150k→2048
    /* LOG_NORMAL(9)         */ {150000, 150000, 2048},   // CDF: 50k→2048
    /* PARETO(10)            */ {2000000, 1500000, 2000000}, // LogPDF: 1M→1.5M; CDF: NEVER→2M
    /* WEIBULL(11)           */ {150000, 150000, 2000000}, // CDF: 1.5M→2M
    /* RAYLEIGH(12)          */ {150000, 150000, 300000},  // unchanged
    /* VON_MISES(13)         */ {25000, 75000, 128},      // PDF: 50k→25k; LogPDF: 100k→75k; CDF: 64→128
    /* BINOMIAL(14)          */ {NEVER, NEVER, 128},      // unchanged
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 512},      // CDF: 2048→512
    /* GEOMETRIC(16)         */ {NEVER, NEVER, 512},      // new: PDF/LogPDF NEVER; CDF 512 (6-run set with NegBinomial; max of lower cluster)
    /* LAPLACE(17)           */ {64, 64, 1024},           // new: PDF/LogPDF at floor; CDF 1024
    /* CAUCHY(18)            */ {2000000, 750000, NEVER},  // new: PDF 2M; LogPDF 750k; CDF NEVER (6-run set with StudentT CDF; 50/50 split → conservative)
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

// --- No SIMD: per-compute-complexity parallel thresholds ---
// VECTORIZED is a plain scalar loop; parallel wins when threading overhead is
// small relative to per-element compute cost.  Three tiers (unprofiled —
// principled placeholders based on approximate per-element cost analysis):
//
//   T1 — 2048: iterative/special-function distributions.  Per-element cost
//     ~1000–10000 ns (incomplete beta, lgamma, digamma).  Threading overhead
//     (~500 µs / 8 cores) amortises at ~500 elements; 2048 is conservative.
//     Distributions: Beta, StudentT, ChiSquared, Gamma, Poisson, Binomial,
//     NegBinomial, VonMises (cos + cached Bessel Z).
//
//   T2 — 8192: elementary transcendental distributions.  Per-element cost
//     ~20–50 ns (exp, log, erf).  Amortises at ~80000 elements; 8192 is the
//     L1d cache boundary and a conservative lower bound.
//     Distributions: Gaussian, Exponential, LogNormal, Weibull, Rayleigh, Pareto.
//
//   T3 — 16384: arithmetic/bandwidth-limited distributions.  Per-element cost
//     ~2–5 ns.  Amortises at ~800000 elements; 16384 is conservative.
//     Distributions: Uniform, Discrete.
//
// GEOMETRIC: T1 (delegates to NegBinomial — lgamma + incomplete beta).
// LAPLACE: T2 (elementary transcendental: fabs + exp).
// CAUCHY: T1 (delegates to StudentT — incomplete beta).
// Re-profile with strategy_profile on an actual no-SIMD build to replace these
// placeholders with measured values.
constexpr ArchTable kNone = {{
    /* UNIFORM(0)            */ {16384, 16384, 16384},  // T3: trivial arithmetic
    /* GAUSSIAN(1)           */ {8192, 8192, 8192},     // T2: exp + erf
    /* EXPONENTIAL(2)        */ {8192, 8192, 8192},     // T2: exp/log
    /* DISCRETE(3)           */ {16384, 16384, 16384},  // T3: table lookup
    /* POISSON(4)            */ {2048, 2048, 2048},     // T1: lgamma
    /* GAMMA(5)              */ {2048, 2048, 2048},     // T1: lgamma + exp
    /* STUDENT_T(6)          */ {2048, 2048, 2048},     // T1: incomplete beta
    /* BETA(7)               */ {2048, 2048, 2048},     // T1: iterative incomplete beta
    /* CHI_SQUARED(8)        */ {2048, 2048, 2048},     // T1: delegates to Gamma
    /* LOG_NORMAL(9)         */ {8192, 8192, 8192},     // T2: log + exp
    /* PARETO(10)            */ {8192, 8192, 8192},     // T2: log + exp pipeline
    /* WEIBULL(11)           */ {8192, 8192, 8192},     // T2: log + exp
    /* RAYLEIGH(12)          */ {8192, 8192, 8192},     // T2: x² + log
    /* VON_MISES(13)         */ {2048, 2048, 2048},     // T1: cos + cached Bessel Z
    /* BINOMIAL(14)          */ {2048, 2048, 2048},     // T1: lgamma + incomplete beta
    /* NEGATIVE_BINOMIAL(15) */ {2048, 2048, 2048},     // T1: lgamma + digamma/trigamma
    /* GEOMETRIC(16)         */ {2048, 2048, 2048},     // T1: delegates to NegBinomial
    /* LAPLACE(17)           */ {8192, 8192, 8192},     // T2: fabs + exp
    /* CAUCHY(18)            */ {2048, 2048, 2048},     // T1: delegates to StudentT
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
