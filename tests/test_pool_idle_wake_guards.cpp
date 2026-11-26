/**
 * @file test_pool_idle_wake_guards.cpp
 * @brief Guards for the work-stealing pool's wake-up on submit (#333)
 *
 * Until v3.0.0 WorkStealingPool::submit() woke no worker and idle workers polled: ten yields,
 * then a 100 us sleep. Two defects followed, one per guard:
 *
 *   IdleProcessBurnsNoCpu     the polling ran for as long as the pool lived: over two cores on
 *                             the Ivy Bridge machine under Linux, most of a core on the M1.
 *   SubmitWakesAWorker        on Windows a 100 us sleep lasts to the next timer tick (15.6 ms),
 *                             so a task submitted to the idle pool started 13-15 ms late.
 *
 * Both were run against the unfixed pool (b30929c) on Windows (Zen 4, clang-cl) and failed there.
 * On Linux and macOS the unfixed pool is expected to pass SubmitWakesAWorker (its sleep really is
 * about 100 us) and to fail both checks of IdleProcessBurnsNoCpu; neither has been run there.
 *
 * Neither is in the "timing" label: CI's correctness run excludes that label, and a guard that
 * never runs guards nothing. The bounds are far from the fixed behaviour (see each test), and
 * the binary runs serially under ctest.
 */

#include "libstats/libstats.h"
#include "libstats/platform/thread_pool.h"
#include "libstats/platform/work_stealing_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <time.h>
#endif

namespace {

using clk = std::chrono::steady_clock;

/// CPU time this process has used, all threads, user and kernel, in seconds.
double processCpuSeconds() {
#ifdef _WIN32
    FILETIME creation, exit, kernel, user;
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user))
        return -1.0;
    const auto ticks = [](const FILETIME& t) {
        return (static_cast<unsigned long long>(t.dwHighDateTime) << 32) | t.dwLowDateTime;
    };
    return static_cast<double>(ticks(kernel) + ticks(user)) * 1e-7;  // 100 ns units
#else
    timespec ts{};
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0)
        return -1.0;
    return static_cast<double>(ts.tv_sec) + static_cast<double>(ts.tv_nsec) * 1e-9;
#endif
}

/// Serial work on the calling thread. Not a sleep: on Windows a short sleep in the caller raises
/// the process's timer resolution, which hides the tick the latency guard is there to catch.
void busyFor(std::chrono::microseconds duration) {
    const auto t0 = clk::now();
    while (clk::now() - t0 < duration) {
    }
}

/// Both pools started and every worker through one task, then time to go idle.
stats::WorkStealingPool& startedPools() {
    stats::initialize_performance_systems();
    // initialize_performance_systems() does not start this pool since v3.0.0; start it here.
    stats::WorkStealingPool& pool = stats::GlobalWorkStealingPool::getInstance();
    std::atomic<int> ran{0};
    for (std::size_t i = 0; i < 4 * pool.getThreadCount(); ++i)
        pool.submit([&ran] { ran.fetch_add(1); });
    pool.waitForAll();
    stats::ParallelUtils::getGlobalThreadPool().submit([&ran] { ran.fetch_add(1); }).get();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return pool;
}

}  // namespace

// An initialised process with both pools started and nothing to do uses no CPU.
//
// Two checks, so the guard cannot pass on either side of the defect:
//   1. Exact. No worker makes a pass of its loop while the pool is idle: the pool's counters
//      (tasks, steals, failed steals) do not move. A polling worker counts a failed steal on
//      every pass, thousands in the window.
//   2. Measured. The process's CPU time over a 2 s idle window stays under 10 ms. The polling
//      pool used over two cores on the Ivy Bridge machine and 0.6 of one on the M1. On Windows
//      its long sleeps made it cheap (about 20 ms of CPU per second on Zen 4), and CPU time
//      there is accounted in timer ticks, so check 1 is the one that decides on Windows.
// Sleeping threads use none, whatever the machine's load, so the bound needs no allowance.
TEST(PoolIdleWakeGuards, IdleProcessBurnsNoCpu) {
    stats::WorkStealingPool& pool = startedPools();

    const auto before = pool.getStatistics();
    const double cpuBefore = processCpuSeconds();
    ASSERT_GE(cpuBefore, 0.0) << "no process CPU clock on this platform";
    std::this_thread::sleep_for(std::chrono::seconds(2));
    const double cpuUsed = processCpuSeconds() - cpuBefore;
    const auto after = pool.getStatistics();

    std::cout << "  idle 2 s: " << cpuUsed * 1e3 << " ms of CPU, "
              << after.failedSteals - before.failedSteals << " idle worker passes\n";
    EXPECT_EQ(after.failedSteals, before.failedSteals)
        << "idle workers are polling: they looked for work "
        << after.failedSteals - before.failedSteals << " times in 2 s with none submitted (#333)";
    EXPECT_EQ(after.workSteals, before.workSteals);
    EXPECT_EQ(after.tasksExecuted, before.tasksExecuted);
    EXPECT_LT(cpuUsed, 0.010) << "an idle initialised process used " << cpuUsed * 1e3
                              << " ms of CPU in 2 s";
}

// A task submitted to the idle pool starts promptly: the median delay from submit() to the
// task's first instruction, over 25 trials 2 ms of serial work apart, is under 1 ms.
//
// Fixed, a worker is notified and starts within microseconds; the bound is far above that, and
// a median ignores the trials a loaded machine delays. Unfixed on Windows the delay is what is
// left of a 15.6 ms timer tick (13-15 ms in #333's quiet run). A few trials in each run catch a
// worker awake and start at once, which pulls the median down: hence 1 ms and not a higher bound.
TEST(PoolIdleWakeGuards, SubmitWakesAWorker) {
    stats::WorkStealingPool& pool = startedPools();

    constexpr int kTrials = 25;
    std::vector<double> delayMs;
    delayMs.reserve(kTrials);
    for (int trial = 0; trial < kTrials; ++trial) {
        busyFor(std::chrono::microseconds(2000));  // the workers go idle
        std::atomic<bool> entered{false};
        clk::time_point entry;
        const auto submitted = clk::now();
        pool.submit([&entered, &entry] {
            entry = clk::now();
            entered.store(true, std::memory_order_release);
        });
        while (!entered.load(std::memory_order_acquire))
            std::this_thread::yield();
        delayMs.push_back(std::chrono::duration<double, std::milli>(entry - submitted).count());
        pool.waitForAll();  // the task has left the pool before its captures go out of scope
    }

    std::sort(delayMs.begin(), delayMs.end());
    const double median = delayMs[delayMs.size() / 2];
    std::cout << "  submit -> task entry over " << kTrials << " trials: median " << median
              << " ms, min " << delayMs.front() << " ms, max " << delayMs.back() << " ms\n";
    EXPECT_LT(median, 1.0) << "a task submitted to the idle work-stealing pool waited " << median
                           << " ms (median) for a worker: submit() woke none (#333)";
}
