/**
 * @file gather_throughput_probe.cpp
 * @brief Issue #33 Stage 1-2: hardware-gather throughput probe (first kill-gate)
 *
 * This tool answers one narrow question before any table-based exp/log port is
 * attempted: on THIS machine, is hardware gather (`_mm256_i32gather_pd` on AVX2,
 * `_mm512_i64gather_pd` on AVX-512) cheap enough, relative to FMA-only compute, that
 * a 3-term-polynomial-plus-128-entry-table transcendental could plausibly beat the
 * current 10-term SLEEF polynomial (see PLAN.md "Issue #33 Experiment" and the
 * internal plan artifact for full context)?
 *
 * This is NOT a correctness test and does NOT implement exp/log. It measures raw
 * gather throughput against an FMA-only reference under three cache regimes:
 *   - warm:       table stays resident in L1 across the whole run (best case for gather)
 *   - cold:       table is clflush'd before every single gather (worst case)
 *   - interleave: a large scratch buffer is touched between gathers to evict the
 *                 table partially, approximating a real distribution batch loop that
 *                 touches many other coefficients per element
 *
 * Fail-forward-fast gate: if `warm` gather cost is not meaningfully cheaper than
 * the FMA baseline, cold/interleave will only be worse -- stop here and record a
 * null result rather than proceeding to a full table port (see PLAN.md).
 *
 * AVX-512 path added on the Asus TUF A16 (Zen 4) to measure the untested half of
 * Q2 -- AMD's AVX-512 gather implementation is architecturally distinct from
 * Intel's Skylake-derived gather unit, so the Kaby Lake AVX2 null result (closed
 * 2026-07-18) does not generalize to it. Note the AVX-512 gather intrinsic's
 * argument order is reversed relative to the AVX2 one: `_mm512_i64gather_pd(vindex,
 * base_addr, scale)` vs. `_mm256_i32gather_pd(base_addr, vindex, scale)`.
 *
 * Opt-in only: gated behind LIBSTATS_BUILD_SIMD_DEV_TOOLS (default OFF). Not part
 * of the production build or test suite.
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#if defined(LIBSTATS_HAS_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    #define LIBSTATS_GATHER_PROBE_AVAILABLE 1
#else
    #define LIBSTATS_GATHER_PROBE_AVAILABLE 0
#endif

#if defined(LIBSTATS_HAS_AVX512) && (defined(__x86_64__) || defined(_M_X64))
    #define LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE 1
#else
    #define LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE 0
#endif

namespace {

// Mirrors the ~128-entry table size used by ARM glibc's exp_advsimd/log_advsimd
// (see Issue #33). Content is arbitrary for this throughput-only probe -- only the
// footprint (1 KB) and access pattern matter here, not numerical correctness.
constexpr int kTableEntries = 128;
constexpr std::size_t kIterations = 4'000'000;    // amortizes timer overhead
constexpr std::size_t kColdIterations = 200'000;  // clflush'd path is much slower per-op
constexpr std::size_t kInterleaveScratchDoubles = 2'000'000;  // ~16 MB, exceeds typical L2/L3 slice

void printCpuAndMitigationNotice() {
    stats::detail::detail::subsectionHeader("CPU / Mitigation State (record manually)");
    const auto& features = stats::arch::get_features();
    std::cout << "CPU brand: " << (features.brand.empty() ? "(unknown)" : features.brand) << "\n";
    std::cout << "Architecture: " << stats::detail::detail::getActiveArchitecture() << "\n";
    std::cout << "NOTE: Gather throughput can be depressed by microcode mitigations\n"
                 "(e.g. Downfall/GDS on affected Skylake-derived x86 parts). This tool\n"
                 "cannot portably query microcode/mitigation state -- record it manually\n"
                 "(OS security advisories, `sysctl`/`/proc/cpuinfo` microcode field, etc.)\n"
                 "alongside these results before drawing conclusions.\n\n";
}

#if LIBSTATS_GATHER_PROBE_AVAILABLE

// Prevents the compiler from proving the accumulated result is unused and
// eliding the measured loop entirely.
inline void sink(__m256d v) {
    alignas(32) double tmp[4];
    _mm256_store_pd(tmp, v);
    volatile double escape = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    (void)escape;
}

double nsPerOp(std::chrono::steady_clock::duration total, std::size_t op_count) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(total).count()) /
           static_cast<double>(op_count);
}

// Warm-table gather: the table (1 KB) stays resident in L1 for the entire run.
// This is the best case for gather and the first thing that must look competitive.
double benchmarkWarmGather(const double* table, const std::vector<int>& indices) {
    __m256d acc = _mm256_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 4 <= kIterations; i += 4) {
        const std::size_t base = (i / 4) % (indices.size() / 4);
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&indices[base * 4]));
        __m256d gathered = _mm256_i32gather_pd(table, idx, 8);
        acc = _mm256_add_pd(acc, gathered);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink(acc);
    return nsPerOp(elapsed, kIterations);
}

// Cold-table gather: clflush every table cache line before each single gather.
// Isolates worst-case memory latency; flush/mfence overhead is reported separately
// so the reader can judge how much of the delta is attributable to the gather itself.
double benchmarkColdGather(double* table, const std::vector<int>& indices, double* out_flush_ns) {
    constexpr std::size_t line_bytes = 64;
    const std::size_t table_bytes = sizeof(double) * static_cast<std::size_t>(kTableEntries);

    // Calibration: flush cost alone, no gather.
    {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kColdIterations; ++i) {
            for (std::size_t off = 0; off < table_bytes; off += line_bytes) {
                _mm_clflush(reinterpret_cast<const char*>(table) + off);
            }
            _mm_mfence();
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        *out_flush_ns = nsPerOp(elapsed, kColdIterations);
    }

    __m256d acc = _mm256_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kColdIterations; ++i) {
        for (std::size_t off = 0; off < table_bytes; off += line_bytes) {
            _mm_clflush(reinterpret_cast<const char*>(table) + off);
        }
        _mm_mfence();
        const std::size_t base = (i % (indices.size() / 4)) * 4;
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&indices[base]));
        __m256d gathered = _mm256_i32gather_pd(table, idx, 8);
        acc = _mm256_add_pd(acc, gathered);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink(acc);
    return nsPerOp(elapsed, kColdIterations);
}

// Interleaved gather: touch a large (~16 MB) scratch buffer with FMA work between
// gathers, partially evicting the table -- approximates a real batch loop that
// reads many other coefficients per element rather than calling exp back-to-back.
double benchmarkInterleavedGather(const double* table, const std::vector<int>& indices,
                                  std::vector<double>& scratch) {
    __m256d acc = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    std::size_t scratch_pos = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 4 <= kIterations; i += 4) {
        const std::size_t base = (i / 4) % (indices.size() / 4);
        __m128i idx = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&indices[base * 4]));
        __m256d gathered = _mm256_i32gather_pd(table, idx, 8);
        acc = _mm256_add_pd(acc, gathered);

        // Touch scratch to create cache pressure on the table.
        __m256d s = _mm256_loadu_pd(&scratch[scratch_pos]);
        s = _mm256_fmadd_pd(s, one, gathered);
        _mm256_storeu_pd(&scratch[scratch_pos], s);
        scratch_pos = (scratch_pos + 4) % (scratch.size() - 4);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink(acc);
    return nsPerOp(elapsed, kIterations);
}

// Reference: FMA-only compute with no memory traffic beyond registers, sized to
// roughly match the arithmetic intensity a 3-term Horner residual polynomial would
// need. This is the cost the gather approach must beat, not just "be fast".
double benchmarkFmaBaseline(const double* input) {
    __m256d x = _mm256_loadu_pd(input);
    const __m256d c0 = _mm256_set1_pd(0.9999999);
    const __m256d c1 = _mm256_set1_pd(0.5000001);
    const __m256d c2 = _mm256_set1_pd(0.1666667);
    __m256d acc = _mm256_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 4 <= kIterations; i += 4) {
        __m256d poly = c2;
        poly = _mm256_fmadd_pd(poly, x, c1);
        poly = _mm256_fmadd_pd(poly, x, c0);
        acc = _mm256_add_pd(acc, poly);
        x = _mm256_add_pd(x, c0);  // keep values changing without introducing memory traffic
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink(acc);
    return nsPerOp(elapsed, kIterations);
}

void runProbeAvx2() {
    stats::detail::detail::subsectionHeader("AVX2 gather probe (_mm256_i32gather_pd, 4-wide)");

    alignas(64) double table[kTableEntries];
    for (int i = 0; i < kTableEntries; ++i) {
        table[i] = 1.0 + static_cast<double>(i) / static_cast<double>(kTableEntries);
    }

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> dist(0, kTableEntries - 1);
    std::vector<int> indices(4096);
    for (auto& v : indices) {
        v = dist(rng);
    }

    std::vector<double> scratch(kInterleaveScratchDoubles, 1.0);
    alignas(32) double fma_input[4] = {0.1, 0.2, 0.3, 0.4};

    const double warm_ns = benchmarkWarmGather(table, indices);
    double flush_ns = 0.0;
    const double cold_ns = benchmarkColdGather(table, indices, &flush_ns);
    const double interleave_ns = benchmarkInterleavedGather(table, indices, scratch);
    const double fma_ns = benchmarkFmaBaseline(fma_input);

    stats::detail::detail::subsectionHeader("Results (ns per 4-wide gather-or-FMA op)");
    std::cout << "FMA-only baseline (compute, no memory traffic): " << fma_ns << " ns\n";
    std::cout << "Warm gather      (table resident in L1):        " << warm_ns << " ns  ("
              << (warm_ns / fma_ns) << "x FMA baseline)\n";
    std::cout << "Interleave gather(realistic cache pressure):    " << interleave_ns << " ns  ("
              << (interleave_ns / fma_ns) << "x FMA baseline)\n";
    std::cout << "Cold gather      (clflush before every gather): " << cold_ns << " ns  ("
              << (cold_ns / fma_ns) << "x FMA baseline, flush-only calibration: " << flush_ns
              << " ns)\n\n";

    std::cout << "Interpretation guide (see PLAN.md gates -- interleave is the gate):\n"
                 "  - If interleave gather is NOT meaningfully cheaper than FMA baseline,\n"
                 "    this is the fail-forward-fast signal: stop, do not port the table\n"
                 "    kernel, record a null result.\n"
                 "  - If interleave gather IS meaningfully cheaper, proceed to Stage 3\n"
                 "    (aarch64 exp_advsimd port) under the accuracy + 20% performance gates.\n";
}

#endif  // LIBSTATS_GATHER_PROBE_AVAILABLE

#if LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE

// Prevents the compiler from proving the accumulated result is unused and
// eliding the measured loop entirely. 8-wide counterpart of sink().
inline void sink512(__m512d v) {
    alignas(64) double tmp[8];
    _mm512_store_pd(tmp, v);
    volatile double escape =
        tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    (void)escape;
}

double nsPerOp512(std::chrono::steady_clock::duration total, std::size_t op_count) {
    return static_cast<double>(
               std::chrono::duration_cast<std::chrono::nanoseconds>(total).count()) /
           static_cast<double>(op_count);
}

// Warm-table gather: the table (1 KB) stays resident in L1 for the entire run.
// 8-wide counterpart of benchmarkWarmGather() using _mm512_i64gather_pd.
double benchmarkWarmGather512(const double* table, const std::vector<int64_t>& indices) {
    __m512d acc = _mm512_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 8 <= kIterations; i += 8) {
        const std::size_t base = (i / 8) % (indices.size() / 8);
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(&indices[base * 8]));
        __m512d gathered = _mm512_i64gather_pd(idx, table, 8);
        acc = _mm512_add_pd(acc, gathered);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink512(acc);
    return nsPerOp512(elapsed, kIterations);
}

// Cold-table gather: clflush every table cache line before each single gather.
// 8-wide counterpart of benchmarkColdGather().
double benchmarkColdGather512(double* table, const std::vector<int64_t>& indices,
                              double* out_flush_ns) {
    constexpr std::size_t line_bytes = 64;
    const std::size_t table_bytes = sizeof(double) * static_cast<std::size_t>(kTableEntries);

    // Calibration: flush cost alone, no gather.
    {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kColdIterations; ++i) {
            for (std::size_t off = 0; off < table_bytes; off += line_bytes) {
                _mm_clflush(reinterpret_cast<const char*>(table) + off);
            }
            _mm_mfence();
        }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        *out_flush_ns = nsPerOp512(elapsed, kColdIterations);
    }

    __m512d acc = _mm512_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < kColdIterations; ++i) {
        for (std::size_t off = 0; off < table_bytes; off += line_bytes) {
            _mm_clflush(reinterpret_cast<const char*>(table) + off);
        }
        _mm_mfence();
        const std::size_t base = (i % (indices.size() / 8)) * 8;
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(&indices[base]));
        __m512d gathered = _mm512_i64gather_pd(idx, table, 8);
        acc = _mm512_add_pd(acc, gathered);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink512(acc);
    return nsPerOp512(elapsed, kColdIterations);
}

// Interleaved gather: touch a large (~16 MB) scratch buffer with FMA work between
// gathers, partially evicting the table. 8-wide counterpart of
// benchmarkInterleavedGather().
double benchmarkInterleavedGather512(const double* table, const std::vector<int64_t>& indices,
                                     std::vector<double>& scratch) {
    __m512d acc = _mm512_setzero_pd();
    const __m512d one = _mm512_set1_pd(1.0);
    std::size_t scratch_pos = 0;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 8 <= kIterations; i += 8) {
        const std::size_t base = (i / 8) % (indices.size() / 8);
        __m512i idx = _mm512_loadu_si512(reinterpret_cast<const void*>(&indices[base * 8]));
        __m512d gathered = _mm512_i64gather_pd(idx, table, 8);
        acc = _mm512_add_pd(acc, gathered);

        // Touch scratch to create cache pressure on the table.
        __m512d s = _mm512_loadu_pd(&scratch[scratch_pos]);
        s = _mm512_fmadd_pd(s, one, gathered);
        _mm512_storeu_pd(&scratch[scratch_pos], s);
        scratch_pos = (scratch_pos + 8) % (scratch.size() - 8);
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink512(acc);
    return nsPerOp512(elapsed, kIterations);
}

// Reference: FMA-only compute with no memory traffic beyond registers. 8-wide
// counterpart of benchmarkFmaBaseline().
double benchmarkFmaBaseline512(const double* input) {
    __m512d x = _mm512_loadu_pd(input);
    const __m512d c0 = _mm512_set1_pd(0.9999999);
    const __m512d c1 = _mm512_set1_pd(0.5000001);
    const __m512d c2 = _mm512_set1_pd(0.1666667);
    __m512d acc = _mm512_setzero_pd();
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i + 8 <= kIterations; i += 8) {
        __m512d poly = c2;
        poly = _mm512_fmadd_pd(poly, x, c1);
        poly = _mm512_fmadd_pd(poly, x, c0);
        acc = _mm512_add_pd(acc, poly);
        x = _mm512_add_pd(x, c0);  // keep values changing without introducing memory traffic
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    sink512(acc);
    return nsPerOp512(elapsed, kIterations);
}

void runProbeAvx512() {
    stats::detail::detail::subsectionHeader(
        "AVX-512 gather probe (_mm512_i64gather_pd, 8-wide, Zen 4)");

    alignas(64) double table[kTableEntries];
    for (int i = 0; i < kTableEntries; ++i) {
        table[i] = 1.0 + static_cast<double>(i) / static_cast<double>(kTableEntries);
    }

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int64_t> dist(0, kTableEntries - 1);
    std::vector<int64_t> indices(4096);
    for (auto& v : indices) {
        v = dist(rng);
    }

    std::vector<double> scratch(kInterleaveScratchDoubles, 1.0);
    alignas(64) double fma_input[8] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

    const double warm_ns = benchmarkWarmGather512(table, indices);
    double flush_ns = 0.0;
    const double cold_ns = benchmarkColdGather512(table, indices, &flush_ns);
    const double interleave_ns = benchmarkInterleavedGather512(table, indices, scratch);
    const double fma_ns = benchmarkFmaBaseline512(fma_input);

    stats::detail::detail::subsectionHeader("Results (ns per 8-wide gather-or-FMA op)");
    std::cout << "FMA-only baseline (compute, no memory traffic): " << fma_ns << " ns\n";
    std::cout << "Warm gather      (table resident in L1):        " << warm_ns << " ns  ("
              << (warm_ns / fma_ns) << "x FMA baseline)\n";
    std::cout << "Interleave gather(realistic cache pressure):    " << interleave_ns << " ns  ("
              << (interleave_ns / fma_ns) << "x FMA baseline)\n";
    std::cout << "Cold gather      (clflush before every gather): " << cold_ns << " ns  ("
              << (cold_ns / fma_ns) << "x FMA baseline, flush-only calibration: " << flush_ns
              << " ns)\n\n";

    std::cout << "Interpretation guide (see PLAN.md gates -- interleave is the gate):\n"
                 "  - If interleave gather is NOT meaningfully cheaper than FMA baseline,\n"
                 "    this is the fail-forward-fast signal: stop, do not port the table\n"
                 "    kernel, record a null result -- same governance as the closed AVX2/\n"
                 "    Kaby Lake sub-experiment.\n"
                 "  - If interleave gather IS meaningfully cheaper (>=20% per PLAN.md gate),\n"
                 "    proceed to Stage 3 (table port) under the <1 ULP accuracy floor.\n";
}

// ============================================================================
// Issue #33 Stage 3: experimental table-gather exp kernel (non-dispatched)
// ============================================================================
// Faithful two-gather port of ARM optimized-routines' scalar exp (math/exp.c +
// math/exp_data.c, SPDX: MIT OR Apache-2.0 WITH LLVM-exception), vectorized to
// 8-wide AVX-512. The tail-corrected N=128 table is what holds < 1 ULP; the
// ~1.9 ULP single-table exp_advsimd variant does not meet libstats' accuracy
// floor. This kernel lives in the opt-in dev tool ONLY -- production
// vector_exp_avx512 is untouched and this symbol is never added to the dispatch
// table. See PLAN.md "Issue #33 Experiment" and THIRD_PARTY_NOTICES.md.
#include "avx512_exp_data.inc"         // kExpSbitsAvx512[128], kExpTailAvx512[128]
#include "avx512_exp_ulp_vectors.inc"  // kExpUlpVectors[], struct ExpUlpVector

// ARM AOR exp constants (math/exp_data.c, N=128 block).
constexpr double kExpInvLn2N = 0x1.71547652b82fep0 * 128.0;  // N/ln2
constexpr double kExpNegLn2hiN = -0x1.62e42fefa0000p-8;      // -ln2/N (hi)
constexpr double kExpNegLn2loN = -0x1.cf79abc9e3b3ap-47;     // -ln2/N (lo)
constexpr double kExpShift = 0x1.8p52;
constexpr double kExpC2 = 0x1.ffffffffffdbdp-2;
constexpr double kExpC3 = 0x1.555555555543cp-3;
constexpr double kExpC4 = 0x1.55555cf172b91p-5;
constexpr double kExpC5 = 0x1.1111167a4d017p-7;
constexpr double kExpSpecialBound = 704.0;  // |x| beyond this: exact scalar fixup

// Experimental 8-wide table-gather exp. NOT wired into the dispatch table.
void vectorExpAvx512Gather(const double* values, double* results, std::size_t size) {
    const __m512d invln2N = _mm512_set1_pd(kExpInvLn2N);
    const __m512d shift = _mm512_set1_pd(kExpShift);
    const __m512d negln2hiN = _mm512_set1_pd(kExpNegLn2hiN);
    const __m512d negln2loN = _mm512_set1_pd(kExpNegLn2loN);
    const __m512d c2 = _mm512_set1_pd(kExpC2);
    const __m512d c3 = _mm512_set1_pd(kExpC3);
    const __m512d c4 = _mm512_set1_pd(kExpC4);
    const __m512d c5 = _mm512_set1_pd(kExpC5);
    const __m512d special_bound = _mm512_set1_pd(kExpSpecialBound);
    const __m512d abs_mask = _mm512_set1_pd(-0.0);
    const __m512i idx_mask = _mm512_set1_epi64(127);

    constexpr std::size_t W = 8;
    const std::size_t simd_end = (size / W) * W;

    for (std::size_t i = 0; i < simd_end; i += W) {
        __m512d x = _mm512_loadu_pd(&values[i]);

        // n = round(x * N/ln2) via the shift trick; ki = bits(z), kd = n as double.
        __m512d z = _mm512_fmadd_pd(x, invln2N, shift);
        __m512i ki = _mm512_castpd_si512(z);
        __m512d kd = _mm512_sub_pd(z, shift);

        // r = x - n*ln2/N  (two-part reduction; the NegLn2*N constants are negative).
        __m512d r = _mm512_fmadd_pd(kd, negln2hiN, x);
        r = _mm512_fmadd_pd(kd, negln2loN, r);

        // Table index and exponent injection derived at RUNTIME (no hardcoded bits).
        __m512i idx = _mm512_and_si512(ki, idx_mask);
        __m512i top = _mm512_slli_epi64(ki, 45);  // 52 - EXP_TABLE_BITS(7)

        // Two 8-wide gathers: scale base (uint64) and tail residual (double).
        __m512i sbits = _mm512_i64gather_epi64(idx, kExpSbitsAvx512, 8);
        __m512d tail = _mm512_i64gather_pd(idx, kExpTailAvx512, 8);
        __m512d scale = _mm512_castsi512_pd(_mm512_add_epi64(sbits, top));  // 2^(n/N)

        // tmp = tail + r + r^2*(C2 + r*C3) + r^4*(C4 + r*C5)  ~= exp(r) - 1 + tail
        __m512d r2 = _mm512_mul_pd(r, r);
        __m512d r4 = _mm512_mul_pd(r2, r2);
        __m512d plo = _mm512_fmadd_pd(r, c3, c2);
        __m512d phi = _mm512_fmadd_pd(r, c5, c4);
        __m512d tmp = _mm512_add_pd(tail, r);
        tmp = _mm512_fmadd_pd(r2, plo, tmp);
        tmp = _mm512_fmadd_pd(r4, phi, tmp);

        // exp(x) = scale + scale*tmp = 2^(n/N) * (1 + (exp(r) - 1)).
        __m512d res = _mm512_fmadd_pd(scale, tmp, scale);
        _mm512_storeu_pd(&results[i], res);

        // Edge lanes (|x| >= 704, +/-inf, NaN): exact scalar std::exp fixup.
        __m512d ax = _mm512_andnot_pd(abs_mask, x);
        __mmask8 special = _mm512_cmp_pd_mask(ax, special_bound, _CMP_NLT_UQ);
        if (special) {
            alignas(64) double xb[W];
            _mm512_store_pd(xb, x);
            for (std::size_t l = 0; l < W; ++l) {
                if (special & static_cast<__mmask8>(1u << l))
                    results[i + l] = std::exp(xb[l]);
            }
        }
    }
    for (std::size_t i = simd_end; i < size; ++i)
        results[i] = std::exp(values[i]);
}

inline double bitsToF64(std::uint64_t b) {
    double d;
    std::memcpy(&d, &b, sizeof d);
    return d;
}

inline std::uint64_t f64ToBits(double d) {
    std::uint64_t b;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// ULP distance for nonnegative exp results; inf/NaN handled explicitly.
double expUlpError(double got, double ref) {
    if (std::isnan(ref))
        return std::isnan(got) ? 0.0 : 1e18;
    if (std::isinf(ref))
        return (got == ref) ? 0.0 : 1e18;
    if (!std::isfinite(got))
        return 1e18;
    const std::uint64_t g = f64ToBits(got), r = f64ToBits(ref);
    return static_cast<double>(g > r ? g - r : r - g);
}

template <typename Fn>
double benchExpNsPerElem(Fn&& fn, const double* in, double* out, std::size_t n,
                         std::size_t iters) {
    fn(in, out, n);  // warmup + populate caches
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t it = 0; it < iters; ++it)
        fn(in, out, n);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    volatile double keep = out[n - 1];
    (void)keep;
    return nsPerOp512(elapsed, n * iters);
}

void runExpExperimentAvx512() {
    stats::detail::detail::subsectionHeader(
        "Issue #33 Stage 3: table-gather exp vs current polynomial exp (AVX-512)");

    auto current = [](const double* in, double* out, std::size_t n) {
        stats::arch::simd::VectorOps::vector_exp(in, out, n);
    };
    auto gather = [](const double* in, double* out, std::size_t n) {
        vectorExpAvx512Gather(in, out, n);
    };

    // ---- Accuracy gate: ULP vs correctly-rounded mpmath reference ----
    constexpr std::size_t NV = sizeof(kExpUlpVectors) / sizeof(kExpUlpVectors[0]);
    std::vector<double> xin(NV), yg(NV), yc(NV);
    for (std::size_t i = 0; i < NV; ++i)
        xin[i] = bitsToF64(kExpUlpVectors[i].x_bits);
    gather(xin.data(), yg.data(), NV);
    current(xin.data(), yc.data(), NV);

    // Head-to-head over the core range |x| <= 700, where both kernels run their
    // polynomial path. The table kernel routes |x| >= 704 to exact scalar exp, so
    // it is additionally correct at the edges the current kernel clamps (+/-708);
    // those edge points are therefore not a fair precision comparison and are
    // excluded from the current-kernel max.
    double g_core = 0, g_all = 0, g_sum = 0, c_core = 0, gworst = 0;
    std::size_t core_n = 0;
    for (std::size_t i = 0; i < NV; ++i) {
        const double ref = bitsToF64(kExpUlpVectors[i].exp_bits);
        const double ug = expUlpError(yg[i], ref);
        g_all = std::max(g_all, ug);
        if (std::abs(xin[i]) <= 700.0) {
            if (ug > g_core) {
                g_core = ug;
                gworst = xin[i];
            }
            c_core = std::max(c_core, expUlpError(yc[i], ref));
            g_sum += ug;
            ++core_n;
        }
    }
    std::cout << "Accuracy vs correctly-rounded reference (" << NV << " points):\n";
    std::cout << "  table-gather exp : core(|x|<=700) max " << g_core << " ULP, mean "
              << (g_sum / static_cast<double>(core_n)) << " ULP; full-range max " << g_all
              << " ULP (worst core x = " << gworst << ")\n";
    std::cout << "  current poly exp : core(|x|<=700) max " << c_core
              << " ULP (clamps beyond +/-708; edges not compared)\n";

    // Special IEEE inputs through the vectorized path. Note exp(-710) is a
    // subnormal ~4.7e-309 (not 0); use -750 for the true underflow-to-zero case.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    alignas(64) double sx[8] = {inf, -inf, nan, 0.0, 1.0, -1.0, 710.0, -750.0};
    alignas(64) double so[8];
    gather(sx, so, 8);
    const bool specials_ok = (so[0] == inf) && (so[1] == 0.0) && std::isnan(so[2]) &&
                             (so[3] == 1.0) && std::isinf(so[6]) && (so[7] == 0.0);
    std::cout << "  special inputs (+/-inf, NaN, overflow, underflow): "
              << (specials_ok ? "OK" : "MISMATCH") << "\n";
    const bool accuracy_ok = (g_core <= 1.0) && (g_core <= c_core) && specials_ok;
    std::cout << "  Accuracy gate (<=1 ULP, no regression vs current): "
              << (accuracy_ok ? "PASS" : "FAIL") << "\n\n";

    // ---- Performance gate: >=20% at a realistic cache-resident regime ----
    std::mt19937 rng(0x5EED);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    auto measure = [&](const char* label, std::size_t n, std::size_t iters) {
        std::vector<double> in(n), out(n);
        for (auto& v : in)
            v = dist(rng);
        const double c = benchExpNsPerElem(current, in.data(), out.data(), n, iters);
        const double g = benchExpNsPerElem(gather, in.data(), out.data(), n, iters);
        const double speedup = c / g;
        std::cout << "  " << label << ": current " << c << " ns/elem, table-gather " << g
                  << " ns/elem, speedup " << speedup << "x (" << ((speedup - 1.0) * 100.0)
                  << "%)\n";
        return speedup;
    };

    stats::detail::detail::subsectionHeader("Throughput (ns per element, lower is better)");
    measure("hot    ( 8K elems, L1/L2-resident)", 8'192, 20'000);
    const double stream_speedup = measure("stream (256K elems, ~L3, realistic)", 262'144, 600);

    std::cout << "\n  Performance gate (>=20% at stream): "
              << (stream_speedup >= 1.20 ? "PASS" : "FAIL") << "\n";
    std::cout << "  Overall verdict: table-gather exp is "
              << ((accuracy_ok && stream_speedup >= 1.20)
                      ? "a WIN (both gates pass)"
                      : "NOT a clear win (see gates above)")
              << ".\n";
}

#endif  // LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE

}  // namespace

int main() {
    return stats::detail::detail::runTool("gather_throughput_probe", [] {
        stats::detail::detail::displayToolHeader(
            "Gather Throughput Probe (Issue #33, Stage 1-2)",
            "Kill-gate: is hardware gather cheap enough here to justify a table+polynomial "
            "exp/log port? See PLAN.md 'Issue #33 Experiment'.");
        printCpuAndMitigationNotice();

#if LIBSTATS_GATHER_PROBE_AVAILABLE
        if (stats::arch::supports_avx2()) {
            runProbeAvx2();
        } else {
            std::cout << "AVX2 not available at runtime on this CPU -- skipping AVX2 probe.\n";
        }
#endif
#if LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE
        if (stats::arch::supports_avx512()) {
            runProbeAvx512();
            runExpExperimentAvx512();
        } else {
            std::cout
                << "AVX-512 not available at runtime on this CPU -- skipping AVX-512 probe.\n";
        }
#endif
#if !LIBSTATS_GATHER_PROBE_AVAILABLE && !LIBSTATS_GATHER_PROBE_AVX512_AVAILABLE
        std::cout << "Neither AVX2 nor AVX-512 gather probes are applicable on this "
                     "build/architecture. See PLAN.md 'Issue #33 Experiment': the NEON side\n"
                     "(Q1) requires no gather probe of this kind.\n";
#endif
    });
}
