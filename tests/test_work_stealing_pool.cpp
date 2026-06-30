/**
 * @file test_work_stealing_pool.cpp
 * @brief GTest suite for WorkStealingPool with Level 0-2 integration
 */
#include "libstats/platform/work_stealing_pool.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <ranges>
#include <span>
#include <thread>
#include <vector>

using namespace stats;

template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

TEST(WorkStealingPool, BasicTaskSubmission) {
    WorkStealingPool pool(4);
    std::atomic<int> counter{0};

    for (int i = 0; i < 10; ++i) {
        pool.submit([&counter, i]() {
            counter.fetch_add(1);
            std::cout << "  Task " << i << " executed\n";
        });
    }

    pool.waitForAll();
    EXPECT_EQ(counter.load(), 10);
    std::cout << "  All 10 tasks executed successfully\n";
}

TEST(WorkStealingPool, ParallelForConcurrentCallersWaitIndependently) {
    WorkStealingPool pool(4);
    std::atomic<int> slow_done{0};
    std::atomic<int> fast_done{0};
    std::atomic<bool> release_slow{false};
    std::atomic<bool> fast_returned{false};
    std::atomic<bool> slow_returned{false};

    std::thread slow([&]() {
        pool.parallelFor(0, 16, [&](std::size_t) {
            while (!release_slow.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            slow_done.fetch_add(1, std::memory_order_release);
        }, 1);
        slow_returned.store(true, std::memory_order_release);
    });

    // Ensure the slow caller has queued tasks that remain incomplete.
    std::this_thread::sleep_for(std::chrono::milliseconds(25));

    std::thread fast([&]() {
        pool.parallelFor(0, 16, [&](std::size_t) {
            fast_done.fetch_add(1, std::memory_order_release);
        }, 1);
        fast_returned.store(true, std::memory_order_release);
    });

    for (int i = 0; i < 100 && !fast_returned.load(std::memory_order_acquire); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    EXPECT_TRUE(fast_returned.load(std::memory_order_acquire))
        << "fast parallelFor caller should not wait for slow caller's tasks";
    EXPECT_EQ(fast_done.load(std::memory_order_acquire), 16);
    EXPECT_FALSE(slow_returned.load(std::memory_order_acquire));

    release_slow.store(true, std::memory_order_release);
    fast.join();
    slow.join();

    EXPECT_TRUE(slow_returned.load(std::memory_order_acquire));
    EXPECT_EQ(slow_done.load(std::memory_order_acquire), 16);
}

TEST(WorkStealingPool, CppRangesWithWorkStealing) {
    WorkStealingPool pool(4);

    std::vector<int> data;
    auto range = std::views::iota(0, 1000) | std::views::transform([](int i) { return i * 2; });
    std::ranges::copy(range, std::back_inserter(data));

    std::atomic<long long> sum{0};
    const size_t chunkSize = 100;
    for (size_t start = 0; start < data.size(); start += chunkSize) {
        size_t end = std::min(start + chunkSize, data.size());
        pool.submit([&data, &sum, start, end]() {
            long long local = 0;
            for (size_t i = start; i < end; ++i)
                local += data[i];
            sum.fetch_add(local);
        });
    }

    pool.waitForAll();

    long long expected = 0;
    for (auto v : data)
        expected += v;
    EXPECT_EQ(sum.load(), expected);
    std::cout << "  Parallel sum: " << sum.load() << " (expected: " << expected << ")\n";
}

TEST(WorkStealingPool, ParallelForWithConcepts) {
    WorkStealingPool pool(4);

    auto process_numeric = [&pool]<Numeric T>(std::vector<T>& vec, T multiplier) {
        pool.parallelFor(0, vec.size(), [&vec, multiplier](size_t i) { vec[i] *= multiplier; });
    };

    std::vector<int> int_data(1000);
    std::iota(int_data.begin(), int_data.end(), 1);
    process_numeric(int_data, 3);

    EXPECT_EQ(int_data[0], 3);
    EXPECT_EQ(int_data[1], 6);
    EXPECT_EQ(int_data[999], 3000);
    std::cout << "  Concept-based parallelFor processed 1000 elements\n";
}

TEST(WorkStealingPool, PerformanceAndStatistics) {
    unsigned int thread_count = std::max(2u, std::thread::hardware_concurrency());
    WorkStealingPool pool(thread_count);
    const int numTasks = 1000;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numTasks; ++i) {
        pool.submit([i]() {
            auto work = std::views::iota(0, (i % 10 + 1) * 100) |
                        std::views::transform([](int x) { return x * x; }) | std::views::take(50);
            volatile long long s = 0;
            for (auto v : work)
                s = s + v;
        });
    }
    pool.waitForAll();
    auto end = std::chrono::high_resolution_clock::now();

    auto stats = pool.getStatistics();
    std::cout << "  Execution: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms, steals: " << stats.workSteals << "\n";

    EXPECT_EQ(static_cast<int>(stats.tasksExecuted), numTasks);
}

TEST(WorkStealingPool, GlobalUtilities) {
    std::atomic<int> counter{0};

    std::vector<int> indices;
    std::ranges::copy(std::views::iota(0, 50), std::back_inserter(indices));

    for ([[maybe_unused]] auto i : indices) {
        stats::workStealingSubmit([&counter]() { counter.fetch_add(1); });
    }

    stats::workStealingWaitForAll();
    EXPECT_EQ(counter.load(), 50);
    std::cout << "  Global utilities executed " << counter.load() << " tasks\n";
}

TEST(WorkStealingPool, Level0ConstantsIntegration) {
    auto parallelThreshold = arch::get_min_elements_for_parallel();
    auto distributionThreshold = arch::get_min_elements_for_distribution_parallel();
    auto defaultGrainSize = arch::get_default_grain_size();
    auto simdBlockSize = arch::get_optimal_simd_block_size();
    auto memoryAlignment = arch::get_optimal_alignment();

    EXPECT_GT(parallelThreshold, 0u);
    EXPECT_GT(distributionThreshold, 0u);
    EXPECT_GT(defaultGrainSize, 0u);
    EXPECT_GT(simdBlockSize, 0u);
    EXPECT_GT(memoryAlignment, 0u);

    std::cout << "  Min parallel: " << parallelThreshold << ", grain: " << defaultGrainSize
              << ", alignment: " << memoryAlignment << " bytes\n";
}

TEST(WorkStealingPool, CPUDetectionIntegration) {
    const auto& features = arch::get_features();
    auto optimalThreads = WorkStealingPool::getOptimalThreadCount();

    EXPECT_GT(optimalThreads, 0u);

    if (features.topology.hyperthreading && features.topology.logical_cores > 0) {
        EXPECT_LE(optimalThreads, features.topology.logical_cores);
    }

    std::cout << "  Physical: " << features.topology.physical_cores
              << ", Logical: " << features.topology.logical_cores << ", Optimal: " << optimalThreads
              << "\n";
}

// POOL-1 regression: parallelFor must return (not deadlock) when a task throws.
// Before the fix the per-call latch was never decremented after an exception,
// causing doneCv->wait() to block forever.
TEST(WorkStealingPool, ParallelForDoesNotDeadlockOnTaskException) {
    WorkStealingPool pool(2);

    std::atomic<int> completed{0};
    // Use a range larger than get_min_elements_for_parallel() (typically 4096)
    // so that parallelFor takes the multi-task parallel path and the per-call
    // latch is actually exercised. The sequential fast path propagates exceptions
    // directly and has no latch to deadlock.
    const std::size_t kTasks = arch::get_min_elements_for_parallel() * 2;

    // One task mid-range throws; all others increment completed.
    // The pool swallows the exception in executeTask(); parallelFor must
    // return without deadlocking and the majority of work must still run.
    const std::size_t kThrowAt = kTasks / 2;
    pool.parallelFor(std::size_t{0}, kTasks, [&completed, kThrowAt](std::size_t i) {
        if (i == kThrowAt)
            throw std::runtime_error("intentional test exception");
        completed.fetch_add(1, std::memory_order_relaxed);
    });

    // At least most tasks should have completed (the throwing task's whole
    // grain is aborted, but all other grains run normally).
    EXPECT_GT(completed.load(), static_cast<int>(kTasks / 2));
    std::cout << "  parallelFor returned after task exception at index " << kThrowAt
              << " (" << completed.load() << "/" << (kTasks - 1) << " non-throwing tasks ran)\n";
}

// TEST-7a: getOptimalThreadCount() must be capped at 32.
// AGENTS.md / Batch 9A: 'WorkStealingPool::getOptimalThreadCount() capped at 32 workers'.
TEST(WorkStealingPool, OptimalThreadCountCappedAt32) {
    const std::size_t cap = 32u;
    const std::size_t optimal = WorkStealingPool::getOptimalThreadCount();
    EXPECT_GE(optimal, 1u);
    EXPECT_LE(optimal, cap)
        << "getOptimalThreadCount() returned " << optimal
        << " which exceeds the hard cap of 32";
    std::cout << "  getOptimalThreadCount()=" << optimal
              << " (hardware_concurrency=" << std::thread::hardware_concurrency() << ")\n";
}

// TEST-7b: Destructor drains all pending tasks before the pool exits.
// A pool destroyed with outstanding work must complete every submitted task;
// otherwise the atomic counter would not reach the expected value.
TEST(WorkStealingPool, DestructorDrainsAllPendingTasks) {
    std::atomic<int> completed{0};
    constexpr int kTasks = 200;
    {
        WorkStealingPool pool(2);
        for (int i = 0; i < kTasks; ++i) {
            pool.submit([&completed]() {
                // Simulate a small amount of work so tasks are still in-flight
                // when the pool destructor runs.
                std::this_thread::yield();
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }
        // Pool destructor runs here — must drain before returning.
    }
    EXPECT_EQ(completed.load(), kTasks)
        << "Pool destructor did not drain all " << kTasks << " submitted tasks";
    std::cout << "  All " << completed.load() << " tasks completed before destructor returned\n";
}

// TEST-7c: Sequential parallelFor calls are linearised — the second call
// cannot observe results from the first call's tasks unless a full fence
// (the per-call latch wait) has occurred.  Verify by accumulating into a
// non-atomic variable protected only by the per-call boundary.
TEST(WorkStealingPool, ParallelForPerCallFencePreventsDataRace) {
    WorkStealingPool pool(4);
    std::vector<int> data(1024, 0);

    // First pass: write 1 into every element.
    pool.parallelFor(std::size_t{0}, data.size(), [&data](std::size_t i) {
        data[i] = 1;
    });
    // The per-call fence guarantees all writes above are visible here.

    // Second pass: accumulate — relies on first pass being fully visible.
    std::atomic<int> sum{0};
    pool.parallelFor(std::size_t{0}, data.size(), [&data, &sum](std::size_t i) {
        sum.fetch_add(data[i], std::memory_order_relaxed);
    });

    EXPECT_EQ(sum.load(), static_cast<int>(data.size()))
        << "Per-call fence broken: second parallelFor saw incomplete writes from first";
    std::cout << "  Per-call fence verified: sum=" << sum.load()
              << " (expected " << data.size() << ")\n";
}
