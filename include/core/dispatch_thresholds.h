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
// BETA: NEVER on kAvx (retired hardware, no re-measure); kNeon PDF/LogPDF
// reverted to NEVER in sha-641bf62 (BEST=VECTORIZED at 2M all runs); kNeon
// CDF=256 and kAvx2/kAvx512 retain measured thresholds.
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
// data/profiles/dispatcher/2026-06-30T03-46-48Z_darwin-arm64_feat-v2-architecture_sha-641bf62
// data/profiles/dispatcher/2026-06-30T04-08-19Z_darwin-arm64_feat-v2-architecture_sha-641bf62
// data/profiles/dispatcher/2026-06-30T04-30-11Z_darwin-arm64_feat-v2-architecture_sha-641bf62
//
// Three sequential --large Release-mode bundles on feat/v2-architecture (641bf62).
// Method: see scripts/PROFILING_METHOD.md (canonical; V→P = min(P,WS) < VECT;
// NEVER when best@max = VECTORIZED or SCALAR).
//
// Warm-pool artifacts in this --large run set (pool accumulates heat across
// distributions; each run is ~20 min with 60 s inter-run sleep):
//   - Gamma PDF:       {1024,512,64}; monotonic warm-pool gradient R1→R3; algorithm
//     gave 512 (lower pair); kAvx2=25k provides cross-arch anchor; override → 25000.
//   - Discrete CDF:    {750k,1M,64}; R3=64 warm-pool; upper pair {750k,1M}→1M.
//     Bimodal: 15625×. Encoded 1000000 (upper pair c/b=1.33≤10).
//   - Rayleigh LogPDF: {25k,25k,64}; R3=64 warm-pool; upper pair→25000. Bimodal: 390×.
//   - Laplace PDF:     {64,6k,6k}; R1=64 (pool still warm from preceding distributions
//     within R1 after 60 s sleep); upper pair→6144. 96×.
//   - Binomial CDF:    BEST=VECTORIZED (R1) / SCALAR (R2,R3) at 2M → NEVER.
//     Prior kNeon=64 came from fb8e8b6 standard-grid runs (cold pool); sha-641bf62
//     code changes appear to have eliminated the parallel advantage for this path.
//   - NegBinomial CDF: BEST=VECTORIZED (R2) / SCALAR (R3) at 2M → NEVER.
//   - Geometric CDF:   BEST=SCALAR all 3 runs at 2M → NEVER (delegates to NegBinomial).
//   - Beta PDF/LogPDF: BEST=VECTORIZED all 3 runs at 2M → NEVER.
//     Prior kNeon=512/256 (fb8e8b6); SIMD path now dominates up to 2M.
//
// Key changes vs prior kNeon (sha-fb8e8b6, standard-grid, 2026-06-24):
//   - Gaussian PDF:      25000 → 50000   ({25k,50k,25k} → max=50k)
//   - Exponential PDF:   50000 → 25000   ({25k,8k,25k}; hi/lo=3.0 → max=25k)
//   - Discrete PDF:      75000 → 50000   ({50k,25k,50k} → max=50k)
//   - Discrete CDF:      NEVER → 1000000 (bimodal; R3=64 warm-pool; upper pair {750k,1M}→1M)
//   - Poisson PDF:         512 → 4096    ({2k,4k,512}; hi/lo=8 → max=4096)
//   - Poisson LogPDF:     1024 → 256     ({6k,128,256}; lower pair b/a=2 → 256)
//   - Gamma PDF:         50000 → 25000   (warm-pool override from {1024,512,64}; kAvx2=25k)
//   - Gamma LogPDF:         64 → 256     ({64,256,64} → max=256)
//   - StudentT LogPDF:   50000 → 25000   ({25k,25k,25k} → 25k; consistent)
//   - StudentT CDF:        256 → 64      ({64,64,64} → 64; consistent)
//   - Beta PDF:            512 → NEVER   (BEST=VECTORIZED all 3 runs at 2M)
//   - Beta LogPDF:         256 → NEVER   (BEST=VECTORIZED all 3 runs at 2M)
//   - ChiSquared PDF:     8192 → 512     ({512,256,256} → max=512; consistent)
//   - ChiSquared LogPDF:    64 → 128     ({64,64,128} → max=128)
//   - LogNormal PDF:     25000 → 10000   ({10k,10k,6k} → max=10k)
//   - LogNormal CDF:     NEVER → 256     ({256,256,256} → 256; consistent)
//   - Pareto PDF:       100000 → 75000   ({75k,50k,25k} → max=75k)
//   - Weibull LogPDF:    75000 → 50000   ({50k,25k,50k} → max=50k)
//   - Weibull CDF:      100000 → 50000   ({50k,50k,8k} → max=50k)
//   - Rayleigh PDF:      25000 → 10000   ({10k,6k,6k} → max=10k)
//   - Rayleigh LogPDF:     128 → 25000   (bimodal {25k,25k,64}; R3 warm; upper pair→25k)
//   - VonMises LogPDF:  300000 → 500000  ({400k,400k,500k} → max=500k)
//   - Binomial CDF:         64 → NEVER   (see warm-pool note above)
//   - NegBinomial CDF:     256 → NEVER   (see warm-pool note above)
//   - Geometric CDF:     NEVER → NEVER   (confirmed; CDF BEST=SCALAR all 3 runs)
//   - Laplace PDF:       NEVER → 6144    (new; {64,6k,6k} upper pair; warm R1 discarded)
//   - Laplace LogPDF:    NEVER → 64      (new; {64,64,64} → 64)
//   - Laplace CDF:       NEVER → 256     (new; {256,256,256} → 256)
//   - Cauchy PDF:        NEVER → 25000   (new; {10k,25k,10k} → max=25k)
//   - Cauchy LogPDF:     NEVER → 50000   (new; {25k,25k,50k} → max=50k)
//   - Cauchy CDF:        NEVER → 512     (new; {64,64,512} → max=512)
constexpr ArchTable kNeon = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {50000, 64, NEVER},
    /* EXPONENTIAL(2)        */ {25000, 64, 25000},
    /* DISCRETE(3)           */ {50000, 50000, 1000000},  // CDF: bimodal {750k,1M,64}; R3 warm;
                                                          // upper pair→1M
    /* POISSON(4)            */ {4096, 256, 256},
    /* GAMMA(5)              */ {25000, 256, 64},  // PDF: warm-pool override from {1024,512,64};
                                                   // kAvx2=25k
    /* STUDENT_T(6)          */ {50000, 25000, 64},
    /* BETA(7)               */ {NEVER, NEVER, 256},  // PDF/LogPDF: BEST=VECTORIZED all 3 runs at
                                                      // 2M
    /* CHI_SQUARED(8)        */ {512, 128, 64},
    /* LOG_NORMAL(9)         */ {10000, 64, 256},
    /* PARETO(10)            */ {75000, 50000, 50000},
    /* WEIBULL(11)           */ {25000, 50000, 50000},
    /* RAYLEIGH(12)          */ {10000, 25000, 25000},  // LogPDF: bimodal {25k,25k,64}; upper
                                                        // pair→25k
    /* VON_MISES(13)         */ {100000, 500000, 128},
    /* BINOMIAL(14)          */ {NEVER, NEVER, NEVER},  // CDF: prior 64 reversed;
                                                        // BEST=VECTORIZED/SCALAR
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, NEVER},  // CDF: prior 256 reversed;
                                                        // BEST=VECTORIZED/SCALAR
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},  // PDF/LogPDF VECTORIZED; CDF SCALAR all 3
                                                        // runs
    /* LAPLACE(17)           */ {6144, 64, 256},  // new: PDF upper pair {6k,6k}; LogPDF floor; CDF
                                                  // consistent
    /* CAUCHY(18)            */ {25000, 50000, 512},  // new: PDF {10k,10k,25k}→25k; LogPDF
                                                      // {25k,25k,50k}→50k
}};

// --- AVX (Intel Ivy Bridge i7-3820QM, 128/256-bit, 4P/8T, macOS/GCD) ---
// data/profiles/dispatcher/2026-06-15T05-25-42Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
// data/profiles/dispatcher/2026-06-15T05-40-12Z_darwin-x86_64_fix-audit-remediation_sha-65b1c61
//
// Two Release-mode bundles on fix/audit-remediation (Ivy Bridge retired; hardware
// no longer in the ecosystem).  Re-inferred from recalibrated kNeon (641bf62)
// and kAvx2 (sha-1b564ec) reference tables.
//
// Inference principle:
//   kAvx VECTORIZED lacks FMA — roughly 2× less efficient than kAvx2 for
//   transcendental operations.  GCD pool overhead is identical (same OS).
//   SIMD-bound distributions: threshold ≈ kAvx2/2.
//   GCD-overhead-dominated (VonMises PDF/CDF, Discrete PDF/LogPDF): ≈ kAvx2
//   (same GCD cost; weaker SIMD lowers threshold slightly, balanced by GCD).
//   Compute-bound iterative (Beta, StudentT): use kAvx2/2 (log-space paths
//   are SIMD-accelerated; iterative overhead is architecture-independent).
//
// Key inference updates vs prior kAvx (sha-fb8e8b6/sha-1b564ec reference):
//   - Gaussian CDF:       10000 → 4096    (kAvx2 8192÷2)
//   - Exponential PDF/CDF:25000 → 12500/12500  (kAvx2 25k÷2)
//   - Discrete PDF/CDF:   75000/25000 → 50000/512  (kAvx2 recalibrated)
//   - Gamma PDF:          10000 → 12500   (kAvx2 override 25k÷2)
//   - Gamma LogPDF:          64 → 256     (kAvx2 512÷2)
//   - StudentT PDF/LogPDF:25000/10000 → 12500/12500  (kAvx2 25k÷2)
//   - Beta:           512/512/512 → 128/128/256  (kAvx2 256/256/512÷2)
//   - ChiSquared PDF:    25000 → 1024    (kAvx2 2048÷2)
//   - LogNormal PDF/CDF: 10000/6144 → 12500/64  (kAvx2 25k/128÷2)
//   - Pareto:       50k/50k/100k → 12500/12500/37500  (kAvx2 25k/25k/75k÷2)
//   - Weibull LogPDF:    25000 → 5000    (kAvx2 10k÷2)
//   - Weibull CDF:       50000 → 25000   (kAvx2 50k÷2; was coincidentally same)
//   - Rayleigh PDF/CDF:  25000/50000 → 12500/12500  (kAvx2 25k÷2)
//   - VonMises LogPDF:  300000 → 400000  (kAvx2 recalibrated to 400k; GCD-dom)
//   - Cauchy PDF/LogPDF: 50000 → 37500   (kAvx2 75k÷2)
//   - Binomial CDF: held 128 (kAvx512=128; kNeon/kAvx2=NEVER; retired hardware)
//   - NegBinomial CDF: held 256 (kNeon/kAvx2=NEVER; retired hardware)
//   Note: Binomial/NegBinomial/Geometric PDF/LogPDF remain NEVER.
// SSE2 delegates to kAvx; both updated together.
constexpr ArchTable kAvx = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 64},
    /* GAUSSIAN(1)           */ {25000, 64, 4096},    // CDF: 10000→4096 (kAvx2=8192÷2)
    /* EXPONENTIAL(2)        */ {12500, 64, 12500},   // PDF/CDF: 25000→12500 (kAvx2=25k÷2)
    /* DISCRETE(3)           */ {50000, 50000, 512},  // PDF: kAvx2=50k; CDF: 1024÷2=512
    /* POISSON(4)            */ {128, 128, 128},
    /* GAMMA(5)              */ {12500, 256, 64},       // PDF: 10000→12500; LogPDF: kAvx2=512÷2
    /* STUDENT_T(6)          */ {12500, 12500, 64},     // PDF: 25k→12500; LogPDF: 25k→12500
    /* BETA(7)               */ {128, 128, 256},        // PDF/LogPDF: 512→128; CDF: 512→256
    /* CHI_SQUARED(8)        */ {1024, 64, 64},         // PDF: 25000→1024 (kAvx2=2048÷2)
    /* LOG_NORMAL(9)         */ {12500, 64, 64},        // PDF: 10k→12500; CDF: 6144→64
    /* PARETO(10)            */ {12500, 12500, 37500},  // kAvx2=25k/25k/75k ÷2
    /* WEIBULL(11)           */ {25000, 5000, 25000},   // LogPDF: 25000→5000 (kAvx2=10k÷2)
    /* RAYLEIGH(12)          */ {12500, 64, 12500},     // PDF: 25k→12500; CDF: 50k→12500
    /* VON_MISES(13)         */ {100000, 400000, 128},  // LogPDF: 300k→400k (kAvx2 GCD-dom)
    /* BINOMIAL(14)          */ {NEVER, NEVER, 128},    // CDF: held from kAvx512=128
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 256},    // CDF: held; kNeon/kAvx2=NEVER
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},
    /* LAPLACE(17)           */ {64, 64, 128},       // CDF: kAvx2=256÷2=128
    /* CAUCHY(18)            */ {37500, 37500, 64},  // PDF/LogPDF: kAvx2=75k÷2
}};

// --- AVX2+FMA (Intel Kaby Lake i7-7820HQ, 256-bit, 4P/8T, macOS/GCD) ---
// Three --large Release-mode bundles on feat/v2-architecture (sha-1b564ec):
//   data/profiles/dispatcher/2026-06-29T22-32-28Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-29T23-28-13Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//   data/profiles/dispatcher/2026-06-30T00-24-23Z_darwin-x86_64_feat-v2-architecture_sha-1b564ec
//
// Full recalibration from sha-1b564ec --large bundles (covers all 19 distributions).
// Method: see scripts/PROFILING_METHOD.md (canonical; V→P = min(P,WS) < VECT;
// NEVER when best@max = VECTORIZED or SCALAR).
//
// Warm-pool notes:
//   - Gamma PDF: {1024,64,256}; non-monotonic; algorithm → 256 (lower pair
//     b/a=4≤10); 256 is implausibly low vs kNeon=25k and kAvx512=10k;
//     override → 25000 (prior value; pending targeted standard-grid re-run).
//   - Weibull LogPDF: {10k,64,8k}; R2=64 warm-pool; upper pair {8k,10k} agree
//     → 10000.  Bimodal: 156×.  Encoded with documentation.
//   - Discrete CDF, LogNormal CDF: large drops ({1k,128,256}→1024;
//     {64,64,128}→128) appear real — consistent across all 3 runs;
//     code changes in sha-1b564ec improved these paths.
//
// Key changes vs prior kAvx2 (sha-fb8e8b6/sha-1b564ec partial encoding):
//   - Uniform CDF:          64 → 128     ({64,128,64} → max=128)
//   - Gaussian CDF:      25000 → 8192    ({8192,6144,8192} → max=8192)
//   - Exponential PDF:   50000 → 25000   ({25k,25k,10k} → max=25k)
//   - Exponential CDF:   50000 → 25000   ({25k,25k,25k} → 25k)
//   - Discrete PDF:      75000 → 50000   ({50k,50k,25k} → max=50k)
//   - Discrete LogPDF:  100000 → 50000   ({50k,50k,50k} → 50k)
//   - Discrete CDF:      25000 → 1024    ({1024,128,256} → max=1024)
//   - Poisson CDF:         128 → 256     ({256,64,64} → max=256)
//   - Gamma PDF:         25000 → 25000   (manual override; see note above)
//   - Gamma LogPDF:         64 → 512     ({512,64,128} → max=512)
//   - StudentT PDF:      50000 → 25000   ({4096,25k,10k} → max=25k)
//   - Beta PDF:            512 → 256     ({256,128,256} → max=256)
//   - Beta LogPDF:         512 → 256     ({1024,64,256}; lower pair → 256)
//   - ChiSquared PDF:    50000 → 2048    ({256,2048,256} → max=2048)
//   - ChiSquared LogPDF:    64 → 128     ({128,64,128} → max=128)
//   - LogNormal CDF:     10000 → 128     ({64,64,128} → max=128)
//   - Pareto PDF:        50000 → 25000   ({25k,25k,25k} → 25k)
//   - Pareto LogPDF:     50000 → 25000   ({10k,10k,25k} → max=25k; prior bimodal override
//   superseded)
//   - Pareto CDF:       150000 → 75000   ({75k,50k,25k} → max=75k)
//   - Weibull PDF:       75000 → 50000   ({50k,50k,25k} → max=50k)
//   - Weibull LogPDF:    50000 → 10000   ({10k,64,8k}; bimodal 156×; upper pair→10k)
//   - Weibull CDF:       75000 → 50000   ({50k,25k,25k} → max=50k)
//   - Rayleigh PDF:      50000 → 25000   ({25k,25k,10k} → max=25k)
//   - Rayleigh CDF:      75000 → 25000   ({25k,25k,25k} → 25k)
//   - VonMises LogPDF:  500000 → 400000  ({400k,250k,250k} → max=400k)
//   - VonMises CDF:        128 → 256     ({64,64,256} → max=256)
constexpr ArchTable kAvx2 = {{
    /* UNIFORM(0)            */ {NEVER, NEVER, 128},
    /* GAUSSIAN(1)           */ {50000, 64, 8192},
    /* EXPONENTIAL(2)        */ {25000, 64, 25000},
    /* DISCRETE(3)           */ {50000, 50000, 1024},
    /* POISSON(4)            */ {128, 128, 256},
    /* GAMMA(5)              */ {25000, 512, 64},  // PDF: warm-pool override; see note above
    /* STUDENT_T(6)          */ {25000, 25000, 64},
    /* BETA(7)               */ {256, 256, 512},
    /* CHI_SQUARED(8)        */ {2048, 128, 64},
    /* LOG_NORMAL(9)         */ {25000, 64, 128},
    /* PARETO(10)            */ {25000, 25000, 75000},
    /* WEIBULL(11)           */ {50000, 10000, 50000},  // LogPDF: bimodal {10k,64,8k}→10k
    /* RAYLEIGH(12)          */ {25000, 64, 25000},
    /* VON_MISES(13)         */ {200000, 400000, 256},
    /* BINOMIAL(14)          */ {NEVER, NEVER, NEVER},
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, NEVER},
    /* GEOMETRIC(16)         */ {NEVER, NEVER, NEVER},
    /* LAPLACE(17)           */ {64, 64, 256},
    /* CAUCHY(18)            */ {75000, 75000, 128},
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
    /* UNIFORM(0)            */ {NEVER, NEVER, NEVER},     // PDF/LogPDF: NEVER (trivial SIMD path;
                                                           // parallel never recovers to SIMD throughput
                                                           // in 1k-100k sweep; see issue #50).
                                                           // CDF: 128→50k→NEVER (50k still too early:
                                                           // 960M at N=45k drops to 463M at N=50k and
                                                           // does not recover within measured range)
    /* GAUSSIAN(1)           */ {1000000, 400000, 25000},  // LogPDF bimodal override; CDF: 50k→25k
    /* EXPONENTIAL(2)        */ {250000, 400000, 250000},  // LogPDF: NEVER→400k; PDF/CDF reduced
    /* DISCRETE(3)           */ {150000, 150000, NEVER},   // PDF: held 150000 (512 was
                                                           // profiling-order warm-pool; aligns with
                                                           // LogPDF same runs); CDF: 75k→NEVER
    /* POISSON(4)            */ {512, 25000, 256},         // PDF: 8192→512; CDF: 2048→256
    /* GAMMA(5)              */ {10000, 256, 64},          // PDF: 150k→10k; LogPDF: 150k→256
    /* STUDENT_T(6)          */ {2000000, 2000000, NEVER},  // PDF/LogPDF: NEVER→2M; CDF: 256→NEVER
    /* BETA(7)               */ {2048, 512, 8192},          // PDF: 256→2048; LogPDF: 128→512; CDF:
                                                            // 6144→8192
    /* CHI_SQUARED(8)        */ {1024, 2048, 128},          // PDF: 150k→1024; LogPDF: 150k→2048
    /* LOG_NORMAL(9)         */ {150000, 150000, 2048},     // CDF: 50k→2048
    /* PARETO(10)            */ {2000000, 1500000, 2000000},  // LogPDF: 1M→1.5M; CDF: NEVER→2M
    /* WEIBULL(11)           */ {150000, 150000, 2000000},    // CDF: 1.5M→2M
    /* RAYLEIGH(12)          */ {150000, 150000, 300000},     // unchanged
    /* VON_MISES(13)         */ {25000, 75000, 128},  // PDF: 50k→25k; LogPDF: 100k→75k; CDF: 64→128
    /* BINOMIAL(14)          */ {NEVER, NEVER, 128},  // unchanged
    /* NEGATIVE_BINOMIAL(15) */ {NEVER, NEVER, 512},  // CDF: 2048→512
    /* GEOMETRIC(16)         */ {NEVER, NEVER, 512},  // new: PDF/LogPDF NEVER; CDF 512 (6-run set
                                                      // with NegBinomial; max of lower cluster)
    /* LAPLACE(17)           */ {35000, 50000, 20000},  // PDF: 64→25k→35k (mild N=25k dip 233M→184M;
                                                          // recovers by N=30k; 35k clears it).
                                                          // LogPDF: 64→25k→50k (severe N=25k dip
                                                          // 433M→170M; only amortises at N=45-50k;
                                                          // see issue #50).
                                                          // CDF: 1024→20k (minor; threshold fires at
                                                          // N=20k but overhead amortises by N=30k)
    /* CAUCHY(18)            */ {2000000, 750000, NEVER},  // new: PDF 2M; LogPDF 750k; CDF NEVER
                                                           // (6-run set with StudentT CDF; 50/50
                                                           // split → conservative)
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
