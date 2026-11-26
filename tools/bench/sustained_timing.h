// sustained_timing.h: per-cell timing that is independent of what ran before the cell (#320).
//
// Standalone on purpose (standard library only): tools/strategy_profile.cpp and the tools/bench
// programs include it, and the bench programs are compiled by hand against other trees' headers.
//
// The problem it solves. A core's clock takes 50-100 ms to recover after a change of threading
// mode (Linux schedutil on Ivy Bridge: 2.1-2.8 GHz against 3.5 after a parallel section, during
// which the calling thread sleeps), and a new process starts cold the same way. A cell timed for
// a few calls inside that window reads whatever the previous cell left behind, and because the
// order of cells is fixed, repeating the run reproduces the bias instead of averaging it away.
//
// The method. Each cell first runs its own work for a fixed time (warm-up, discarded), so the
// machine is in that cell's steady state, then is timed for a minimum duration in blocks:
//   - warm-up: call f() until `warmup_s` has passed (at least once);
//   - block size: enough calls that one block lasts about `block_s`, so the clock read costs
//     nothing against the block even for a 64-element batch;
//   - timed: blocks until `timed_s` has passed and at least `min_blocks` blocks are in.
// A call longer than the budgets is still timed `min_blocks` times, after one warm-up call.
//
// timeCellSpaced is the second method: the same cell with serial work between calls, for callers
// whose batches do not arrive back to back. See its comment.
//
// Environment (seconds, for quick smoke runs or longer windows):
//   LIBSTATS_BENCH_CELL_WARMUP_SECONDS (default 0.15), LIBSTATS_BENCH_CELL_SECONDS (default 0.25).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <vector>

namespace bench_timing {

/// An environment variable's value, empty when unset. The Windows CRT deprecates std::getenv, and
/// these programs are compiled without _CRT_SECURE_NO_WARNINGS, so clang-cl warned on it.
inline std::string envValue(const char* name) {
#ifdef _WIN32
    char* buf = nullptr;
    std::size_t len = 0;
    std::string value;
    if (_dupenv_s(&buf, &len, name) == 0 && buf != nullptr)
        value = buf;
    std::free(buf);
    return value;
#else
    const char* s = std::getenv(name);
    return s ? s : "";
#endif
}

struct CellBudget {
    double warmup_s = 0.15;   // this cell's own work, discarded
    double timed_s = 0.25;    // minimum timed duration
    double block_s = 0.0005;  // target length of one timed block
    std::size_t min_blocks = 7;
};

/// The default budget, with the two durations overridable from the environment.
inline CellBudget cellBudgetFromEnv() {
    CellBudget b;
    if (const std::string s = envValue("LIBSTATS_BENCH_CELL_WARMUP_SECONDS"); !s.empty()) {
        const double v = std::atof(s.c_str());
        if (v >= 0.0)
            b.warmup_s = v;
    }
    if (const std::string s = envValue("LIBSTATS_BENCH_CELL_SECONDS"); !s.empty()) {
        const double v = std::atof(s.c_str());
        if (v >= 0.0)
            b.timed_s = v;
    }
    return b;
}

/// Per-call times in nanoseconds over the timed blocks.
struct CellTiming {
    double median_ns = 0.0;  // median block
    double min_ns = 0.0;     // fastest block
    double mean_ns = 0.0;    // total timed time over total calls: the sustained cost
    std::size_t calls = 0;   // timed calls
    std::size_t blocks = 0;  // timed blocks
};

template <class F>
CellTiming timeCell(F&& f, const CellBudget& budget = cellBudgetFromEnv()) {
    using clk = std::chrono::steady_clock;
    const auto seconds = [](clk::duration d) { return std::chrono::duration<double>(d).count(); };

    // Warm-up: this cell's own work, so the cores' clocks and the pool are in its steady state.
    std::size_t warm_calls = 0;
    const auto w0 = clk::now();
    double warm_elapsed = 0.0;
    do {
        f();
        ++warm_calls;
        warm_elapsed = seconds(clk::now() - w0);
    } while (warm_elapsed < budget.warmup_s);

    // The warm-up's late calls are the fair estimate, but its mean is good enough to size a
    // block: a block a little short or long changes nothing in the statistics below.
    const double est_call_s = warm_elapsed / static_cast<double>(warm_calls);
    std::size_t reps = 1;
    if (est_call_s > 0.0 && est_call_s < budget.block_s)
        reps = static_cast<std::size_t>(budget.block_s / est_call_s) + 1;

    std::vector<double> block_ns;
    block_ns.reserve(1024);
    double total_s = 0.0;
    while (block_ns.size() < budget.min_blocks || total_s < budget.timed_s) {
        const auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i)
            f();
        const double s = seconds(clk::now() - t0);
        total_s += s;
        block_ns.push_back(s * 1e9 / static_cast<double>(reps));
    }

    CellTiming out;
    out.blocks = block_ns.size();
    out.calls = out.blocks * reps;
    out.mean_ns = total_s * 1e9 / static_cast<double>(out.calls);
    out.min_ns = *std::min_element(block_ns.begin(), block_ns.end());
    std::nth_element(block_ns.begin(),
                     block_ns.begin() + static_cast<std::ptrdiff_t>(out.blocks / 2),
                     block_ns.end());
    out.median_ns = block_ns[out.blocks / 2];
    return out;
}

/// Serial work on the calling thread for `gap_us` microseconds: a loop that reads the clock, as
/// tools/bench/batch_gap_bench.cpp does. The caller stays busy, so its core's clock stays up;
/// the pool's workers have nothing to do and go to sleep.
inline void busyGap(double gap_us) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    while (std::chrono::duration<double, std::micro>(clk::now() - t0).count() < gap_us) {
    }
}

/// Spaced timing: one call, then `gap_us` of serial work on the calling thread, repeated. Only
/// the time inside the call is counted, one call per block. The pattern runs for `warmup_s`
/// unmeasured, then for at least `timed_s` and kSpacedMinCalls calls; both budgets are wall time
/// of the pattern, gaps included, so a cell costs about what a back-to-back cell costs.
///
/// What it measures that timeCell cannot: back to back, the pool's workers never sleep between
/// parallel sections. With a gap they do, and a parallel batch then pays for waking them and for
/// whatever clock their cores dropped to (PLAN.md, "Pool wake-up and batch spacing").
template <class F>
CellTiming timeCellSpaced(F&& f, double gap_us, const CellBudget& budget = cellBudgetFromEnv()) {
    using clk = std::chrono::steady_clock;
    constexpr std::size_t kSpacedMinCalls = 20;
    const auto seconds = [](clk::duration d) { return std::chrono::duration<double>(d).count(); };

    const auto w0 = clk::now();
    do {
        f();
        busyGap(gap_us);
    } while (seconds(clk::now() - w0) < budget.warmup_s);

    std::vector<double> call_ns;
    call_ns.reserve(1024);
    double total_ns = 0.0;
    for (const auto t0 = clk::now();
         call_ns.size() < kSpacedMinCalls || seconds(clk::now() - t0) < budget.timed_s;) {
        const auto a = clk::now();
        f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - a).count();
        total_ns += ns;
        call_ns.push_back(ns);
        busyGap(gap_us);
    }

    CellTiming out;
    out.blocks = call_ns.size();
    out.calls = out.blocks;
    out.mean_ns = total_ns / static_cast<double>(out.calls);
    out.min_ns = *std::min_element(call_ns.begin(), call_ns.end());
    std::nth_element(call_ns.begin(), call_ns.begin() + static_cast<std::ptrdiff_t>(out.blocks / 2),
                     call_ns.end());
    out.median_ns = call_ns[out.blocks / 2];
    return out;
}

}  // namespace bench_timing
