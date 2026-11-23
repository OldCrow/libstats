// pool_wake_bench: when batches are spaced, do the pool's workers start late or run slowly?
//
// batch_gap_bench shows that forced PARALLEL costs more per call once batches stop arriving back
// to back, and by how much. It cannot say why: a late start and a slow run both read as a longer
// call. This program separates them, with no kernel in the way and no privileges.
//
// Each call hands `n` elements to the library's own pool in 1024-element slices, as the batch
// paths do (ParallelUtils::parallelForSlices, or the work-stealing pool's). The work per element
// is a fixed integer recurrence that touches no memory, so its time is inversely proportional to
// the core's clock and to nothing else. Every slice records which thread ran it and when it
// started and finished. Per call that gives:
//   Start1_us     submit -> the first worker begins            (wake latency, first worker)
//   StartLast_us  submit -> the last worker to join begins     (wake latency, slowest worker)
//   Work_ns       ns per element inside the slices             (run speed: the clock)
//   FirstWork_ns  the same, each worker's first slice only     (a clock still ramping up)
//   Join_us       the last slice ends -> the call returns
// SERIAL runs the same work inline on the caller: its ns per element is the reference clock, and
// its change with the gap is the caller's own state after the gap.
//
// Pool state cannot be undone within a process, so it is a command-line choice, one run each:
//   (no option)  only the thread pool exists
//   --init       stats::initialize_performance_systems() first; this also starts the
//                work-stealing pool, whose idle workers poll
//   --ws         adds a WORK_STEALING row (starts that pool part-way through the run without
//                --init, so read the THREAD_POOL rows from a run without --ws)
//   --sleep      the caller sleeps during the gap instead of working
//   --gaps a,b   gaps in microseconds (default 0,50,500,2000,20000)
//   --sizes a,b  element counts (default 4096,10240,51200,204800)
//   --local-pool a,b  adds a LOCAL_POOL_<n> row per count: a ThreadPool of n threads owned by
//                this program, driven by a copy of ParallelUtils' slice submission (same slices,
//                same grain rule with n threads, one submit per chunk, wait for all). The
//                library's pool takes physical cores under hyper-threading and its
//                work-stealing pool takes logical ones, so this separates thread count from
//                stealing. Give the library's own count too (4 on a 4-core/8-thread machine):
//                that row against THREAD_POOL shows whether the copy is faithful.
//
// A cell runs its pattern for 0.3 s unmeasured, then for at least 0.7 s and 20 calls; the CSV
// reports medians over calls (StartLast also at the 90th percentile). The '#' lines at the end
// give, per mode and size, each gap against gap 0: the extra start delay and the work-time ratio.
// About two minutes per run on a quiet machine.
//
// Build: as distributions_bench (tools/bench/README.md). Uses detail-level pool headers, so it
// compiles against the tree it ships with.
#include "libstats/libstats.h"
#include "libstats/platform/thread_pool.h"
#include "libstats/platform/work_stealing_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

constexpr double kWarmSeconds = 0.3;
constexpr double kTimedSeconds = 0.7;
constexpr std::size_t kMinCalls = 20;
constexpr std::size_t kSlice = 1024;
constexpr int kRoundsPerElement = 8;  // about 14 ns per element on Kaby Lake

std::atomic<int> g_next_thread{0};
int threadIndex() {
    thread_local const int id = g_next_thread.fetch_add(1, std::memory_order_relaxed);
    return id;
}

std::atomic<std::uint64_t> g_sink{0};

/// The work: a dependent xorshift chain, `len` elements. No memory, no SIMD, no branches that
/// depend on data, so the time is cycles and nothing else.
std::uint64_t work(std::size_t begin, std::size_t len) {
    std::uint64_t x = 0x9E3779B97F4A7C15ull + begin;
    for (std::size_t i = 0; i < len * kRoundsPerElement; ++i) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
    }
    return x;
}

double secondsSince(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}
double us(clk::duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

void gap(double gap_us, bool sleep) {
    if (gap_us <= 0.0)
        return;
    if (sleep) {
        std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(gap_us));
        return;
    }
    const auto t0 = clk::now();
    while (us(clk::now() - t0) < gap_us) {
    }
}

struct SliceRecord {
    clk::time_point begin, end;
    std::size_t len = 0;
    int thread = -1;
};

struct CallStats {
    double call_us, start1_us, start_last_us, work_ns, first_work_ns, join_us;
    int workers;
};

enum class Mode { Serial, ThreadPool, WorkStealing, LocalPool };

/// ParallelUtils::parallelForSlices on a pool of our own: the same slices (a short tail folded
/// into the last one), the same grain rule with this pool's thread count, one submit per chunk,
/// every future waited on.
template <class Body>
void localPoolForSlices(stats::ThreadPool& pool, std::size_t threads, std::size_t count,
                        const Body& body) {
    const std::size_t num_slices = std::max<std::size_t>(1, count / kSlice);
    const std::size_t grain = std::max<std::size_t>(1, num_slices / (threads * 4));
    std::vector<std::future<void>> futures;
    futures.reserve((num_slices + grain - 1) / grain);
    for (std::size_t i = 0; i < num_slices; i += grain) {
        const std::size_t chunk_end = std::min(i + grain, num_slices);
        futures.push_back(pool.submit([&body, i, chunk_end, count, num_slices] {
            for (std::size_t ci = i; ci < chunk_end; ++ci) {
                const std::size_t begin = ci * kSlice;
                body(begin, ci + 1 == num_slices ? count - begin : kSlice);
            }
        }));
    }
    for (auto& f : futures)
        f.wait();
}

struct LocalPool {
    std::size_t threads;
    std::unique_ptr<stats::ThreadPool> pool;
};

/// One call: run `n` elements under `mode`, return what the slices recorded.
CallStats oneCall(Mode mode, std::size_t n, std::vector<SliceRecord>& slices, LocalPool* local) {
    for (auto& s : slices)
        s.thread = -1;
    std::atomic<std::uint64_t> acc{0};
    const auto body = [&](std::size_t begin, std::size_t len) {
        SliceRecord& r = slices[begin / kSlice];
        r.thread = threadIndex();
        r.len = len;
        r.begin = clk::now();
        const std::uint64_t v = work(begin, len);
        r.end = clk::now();
        acc.fetch_xor(v, std::memory_order_relaxed);
    };

    const auto submit = clk::now();
    switch (mode) {
        case Mode::Serial:
            body(0, n);
            break;
        case Mode::ThreadPool:
            stats::ParallelUtils::parallelForSlices(n, kSlice, body);
            break;
        case Mode::WorkStealing:
            stats::GlobalWorkStealingPool::getInstance().parallelForSlices(n, kSlice, body);
            break;
        case Mode::LocalPool:
            localPoolForSlices(*local->pool, local->threads, n, body);
            break;
    }
    const auto done = clk::now();
    g_sink.fetch_xor(acc.load(), std::memory_order_relaxed);

    // First slice of each thread, in start order.
    std::map<int, const SliceRecord*> first;
    clk::time_point last_end = submit;
    std::vector<double> work_ns;
    for (const auto& s : slices) {
        if (s.thread < 0)
            continue;
        auto [it, inserted] = first.try_emplace(s.thread, &s);
        if (!inserted && s.begin < it->second->begin)
            it->second = &s;
        last_end = std::max(last_end, s.end);
        work_ns.push_back(std::chrono::duration<double, std::nano>(s.end - s.begin).count() /
                          static_cast<double>(s.len));
    }
    std::vector<double> starts, first_ns;
    for (const auto& [thread, s] : first) {
        starts.push_back(us(s->begin - submit));
        first_ns.push_back(std::chrono::duration<double, std::nano>(s->end - s->begin).count() /
                           static_cast<double>(s->len));
    }
    const auto median = [](std::vector<double>& v) {
        std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2), v.end());
        return v[v.size() / 2];
    };
    CallStats c{};
    c.call_us = us(done - submit);
    c.start1_us = *std::min_element(starts.begin(), starts.end());
    c.start_last_us = *std::max_element(starts.begin(), starts.end());
    c.work_ns = median(work_ns);
    c.first_work_ns = median(first_ns);
    c.join_us = us(done - last_end);
    c.workers = static_cast<int>(first.size());
    return c;
}

struct Cell {
    double call_us, start1_us, start_last_us, start_last_p90_us, work_ns, first_work_ns, join_us,
        workers;
    std::size_t calls;
};

Cell timeCell(Mode mode, std::size_t n, double gap_us, bool sleep, LocalPool* local) {
    std::vector<SliceRecord> slices(std::max<std::size_t>(1, n / kSlice) + 1);
    for (const auto t0 = clk::now(); secondsSince(t0) < kWarmSeconds;) {
        oneCall(mode, n, slices, local);
        gap(gap_us, sleep);
    }
    std::vector<CallStats> calls;
    for (const auto t0 = clk::now();
         secondsSince(t0) < kTimedSeconds || calls.size() < kMinCalls;) {
        calls.push_back(oneCall(mode, n, slices, local));
        gap(gap_us, sleep);
    }
    const auto quantile = [&](double CallStats::* field, double q) {
        std::vector<double> v;
        v.reserve(calls.size());
        for (const auto& c : calls)
            v.push_back(c.*field);
        const auto k = static_cast<std::ptrdiff_t>(q * static_cast<double>(v.size() - 1));
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[static_cast<std::size_t>(k)];
    };
    std::vector<double> workers;
    for (const auto& c : calls)
        workers.push_back(c.workers);
    std::nth_element(workers.begin(),
                     workers.begin() + static_cast<std::ptrdiff_t>(workers.size() / 2),
                     workers.end());
    return Cell{quantile(&CallStats::call_us, 0.5),
                quantile(&CallStats::start1_us, 0.5),
                quantile(&CallStats::start_last_us, 0.5),
                quantile(&CallStats::start_last_us, 0.9),
                quantile(&CallStats::work_ns, 0.5),
                quantile(&CallStats::first_work_ns, 0.5),
                quantile(&CallStats::join_us, 0.5),
                workers[workers.size() / 2],
                calls.size()};
}

template <class T>
std::vector<T> parseList(const char* text) {
    std::vector<T> out;
    for (const char* p = text; *p;) {
        char* end = nullptr;
        out.push_back(static_cast<T>(std::strtod(p, &end)));
        if (end == p)
            return {};
        p = (*end == ',') ? end + 1 : end;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    bool init = false, with_ws = false, sleep = false;
    std::vector<std::size_t> sizes = {4096, 10240, 51200, 204800};
    std::vector<double> gaps_us = {0.0, 50.0, 500.0, 2000.0, 20000.0};
    std::vector<std::size_t> local_threads;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--init") == 0) {
            init = true;
        } else if (std::strcmp(argv[i], "--ws") == 0) {
            with_ws = true;
        } else if (std::strcmp(argv[i], "--sleep") == 0) {
            sleep = true;
        } else if (std::strcmp(argv[i], "--gaps") == 0 && i + 1 < argc) {
            gaps_us = parseList<double>(argv[++i]);
        } else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            sizes = parseList<std::size_t>(argv[++i]);
        } else if (std::strcmp(argv[i], "--local-pool") == 0 && i + 1 < argc) {
            local_threads = parseList<std::size_t>(argv[++i]);
            if (local_threads.empty() ||
                std::find(local_threads.begin(), local_threads.end(), 0) != local_threads.end())
                gaps_us.clear();
        } else {
            gaps_us.clear();
        }
        if (gaps_us.empty() || sizes.empty()) {
            std::fprintf(stderr,
                         "usage: %s [--init] [--ws] [--sleep] [--gaps us,us,...] "
                         "[--sizes n,n,...] [--local-pool threads,...]\n",
                         argv[0]);
            return 2;
        }
    }
    threadIndex();  // the caller is thread 0
    if (init)
        stats::initialize_performance_systems();

    struct ModeRow {
        std::string name;
        Mode mode;
        LocalPool* local;
    };
    std::vector<ModeRow> modes = {{"SERIAL", Mode::Serial, nullptr},
                                  {"THREAD_POOL", Mode::ThreadPool, nullptr}};
    if (with_ws)
        modes.push_back({"WORK_STEALING", Mode::WorkStealing, nullptr});
    // Every local pool exists from here on, as the library's pools do once created: their idle
    // workers sleep on a condition variable and cost nothing.
    std::vector<LocalPool> local_pools;
    local_pools.reserve(local_threads.size());
    for (const std::size_t t : local_threads) {
        local_pools.push_back({t, std::make_unique<stats::ThreadPool>(t)});
        modes.push_back({"LOCAL_POOL_" + std::to_string(t), Mode::LocalPool, &local_pools.back()});
    }

    const char* pool = init ? "init" : "plain";
    const char* gap_kind = sleep ? "sleep" : "busy";
    std::printf(
        "# pool_wake_bench pool=%s gap=%s logical_cpus=%u slice=%zu rounds_per_element=%d\n", pool,
        gap_kind, std::thread::hardware_concurrency(), kSlice, kRoundsPerElement);
    std::printf(
        "Pool,GapKind,Mode,Elements,Gap_us,Calls,Call_us,Workers,Start1_us,StartLast_us,"
        "StartLastP90_us,Work_ns,FirstWork_ns,Join_us\n");

    // results[mode][size][gap]
    std::map<std::string, std::map<std::size_t, std::map<double, Cell>>> results;
    for (const auto& m : modes) {
        for (const double gap_us : gaps_us) {
            for (const std::size_t n : sizes) {
                const Cell c = timeCell(m.mode, n, gap_us, sleep, m.local);
                results[m.name][n][gap_us] = c;
                std::printf("%s,%s,%s,%zu,%.0f,%zu,%.2f,%.0f,%.2f,%.2f,%.2f,%.3f,%.3f,%.2f\n", pool,
                            gap_kind, m.name.c_str(), n, gap_us, c.calls, c.call_us, c.workers,
                            c.start1_us, c.start_last_us, c.start_last_p90_us, c.work_ns,
                            c.first_work_ns, c.join_us);
                std::fflush(stdout);
            }
        }
    }

    std::printf(
        "# against the first gap: extra call time, extra start delay of the last worker, "
        "work-time ratio\n");
    for (const auto& [mode, by_size] : results) {
        for (const auto& [n, by_gap] : by_size) {
            const Cell& base = by_gap.begin()->second;
            for (const auto& [gap_us, c] : by_gap) {
                if (gap_us == by_gap.begin()->first)
                    continue;
                std::printf(
                    "# %-13s n=%-7zu gap %6.0f us: call %+8.1f us  start %+7.1f us  "
                    "work x%.2f  first-slice x%.2f\n",
                    mode.c_str(), n, gap_us, c.call_us - base.call_us,
                    c.start_last_us - base.start_last_us, c.work_ns / base.work_ns,
                    c.first_work_ns / base.first_work_ns);
            }
        }
    }
    std::printf("# (sink %llu)\n", static_cast<unsigned long long>(g_sink.load()));
    return 0;
}
