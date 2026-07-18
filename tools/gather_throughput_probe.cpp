/**
 * @file gather_throughput_probe.cpp
 * @brief Issue #33 Stage 1-2: AVX2 hardware-gather throughput probe (first kill-gate)
 *
 * This tool answers one narrow question before any table-based exp/log port is
 * attempted: on THIS machine, is `_mm256_i32gather_pd` cheap enough, relative to
 * FMA-only compute, that a 3-term-polynomial-plus-128-entry-table transcendental
 * could plausibly beat the current 10-term SLEEF polynomial (see PLAN.md "Issue #33
 * Experiment" and the internal plan artifact for full context)?
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
 * Opt-in only: gated behind LIBSTATS_BUILD_SIMD_DEV_TOOLS (default OFF). Not part
 * of the production build or test suite.
 */

// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#if defined(LIBSTATS_HAS_AVX2) && (defined(__x86_64__) || defined(_M_X64))
    #define LIBSTATS_GATHER_PROBE_AVAILABLE 1
#else
    #define LIBSTATS_GATHER_PROBE_AVAILABLE 0
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

int runProbe() {
    stats::detail::detail::displayToolHeader(
        "Gather Throughput Probe (Issue #33, Stage 1-2)",
        "First kill-gate: is AVX2 hardware gather cheap enough here to justify a "
        "table+polynomial exp/log port? See PLAN.md 'Issue #33 Experiment'.");

    printCpuAndMitigationNotice();

    if (!stats::arch::supports_avx2()) {
        std::cout << "AVX2 not available at runtime on this CPU -- cannot probe gather.\n"
                     "Result: N/A (record as null-result precondition, not a gate failure).\n";
        return 0;
    }

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

    return 0;
}

#endif  // LIBSTATS_GATHER_PROBE_AVAILABLE

}  // namespace

int main() {
    return stats::detail::detail::runTool("gather_throughput_probe", [] {
#if LIBSTATS_GATHER_PROBE_AVAILABLE
        runProbe();
#else
        stats::detail::detail::displayToolHeader(
            "Gather Throughput Probe (Issue #33, Stage 1-2)",
            "Requires x86_64 AVX2; not applicable on this architecture.");
        printCpuAndMitigationNotice();
        std::cout << "This probe targets AVX2 hardware gather (_mm256_i32gather_pd) and is not\n"
                     "applicable on this build/architecture. See PLAN.md 'Issue #33 Experiment':\n"
                     "the NEON side (Q1) requires no gather probe of this kind.\n";
#endif
    });
}
