#include "libstats/platform/thread_pool.h"

#include "libstats/common/cpu_detection_fwd.h"         // Use lightweight forward declarations
#include "libstats/common/distribution_impl_common.h"  // SIMD + parallel (AQ-7)
#include "libstats/common/platform_constants_fwd.h"    // Use lightweight forward declarations
#include "libstats/core/math_constants.h"
#include "libstats/core/statistical_constants.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// Platform-specific includes (conditionally accepted when properly guarded)
#ifdef __APPLE__
    #include <mach/mach.h>
    #include <mach/thread_policy.h>
    #include <sys/sysctl.h>
    #include <sys/types.h>
#elif defined(__linux__)
    #include <fstream>
    #include <sched.h>
    #include <sys/sysinfo.h>
    #include <unistd.h>
#elif defined(_WIN32)
    #include <intrin.h>
    #include <windows.h>
#endif

using namespace stats;
using namespace stats::detail;
using namespace stats::arch::parallel;

namespace stats {

//========== ThreadPool Implementation ==========

ThreadPool::ThreadPool(std::size_t numThreads)
    : stop_(false), activeTasks_(0), tasksAvailable_(false), allTasksComplete_(false) {
    if (numThreads == 0) {
        numThreads = getOptimalThreadCount();
    }

    workers_.reserve(numThreads);
    for (std::size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this] { workerLoop(); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::submitVoid(Task task) {
    {
        std::unique_lock<std::mutex> lock(queueMutex_);

        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped ThreadPool");
        }

        tasks_.emplace(std::move(task));
    }

    condition_.notify_one();
}

std::size_t ThreadPool::getPendingTasks() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return tasks_.size();
}

void ThreadPool::waitForAll() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    finished_.wait(lock, [this] { return tasks_.empty() && activeTasks_.load() == 0; });
}

void ThreadPool::workerLoop() {
    while (true) {
        Task task;

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
            activeTasks_++;
        }

        try {
            task();
        } catch (const std::exception& e) {
            std::cerr << "Exception in thread pool task: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception in thread pool task" << std::endl;
        }

        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            activeTasks_--;
            if (tasks_.empty() && activeTasks_.load() == 0) {
                finished_.notify_all();
            }
        }
    }
}

namespace {

/// LIBSTATS_THREAD_COUNT as a thread count, 0 when unset or not a count in [1, 1024].
/// _dupenv_s on Windows: its CRT deprecates std::getenv, and the build is warning-free.
std::size_t threadCountFromEnvironment() noexcept {
    constexpr std::size_t kMaxThreads = 1024;
#ifdef _WIN32
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, "LIBSTATS_THREAD_COUNT") != 0 || buffer == nullptr) {
        std::free(buffer);
        return 0;
    }
    const char* const text = buffer;
#else
    const char* const text = std::getenv("LIBSTATS_THREAD_COUNT");
    if (text == nullptr)
        return 0;
#endif
    std::size_t value = 0;
    bool valid = *text != '\0';
    for (const char* c = text; valid && *c != '\0'; ++c) {
        valid = *c >= '0' && *c <= '9';
        if (valid) {
            value = value * 10 + static_cast<std::size_t>(*c - '0');
            valid = value <= kMaxThreads;
        }
    }
#ifdef _WIN32
    std::free(buffer);
#endif
    return valid && value >= 1 ? value : 0;
}

std::size_t computeOptimalThreadCount() noexcept {
    // Measurement override (v3.0.0 pool work): the fleet compares physical against logical
    // core counts with it. Read once, here; the global pool and every slice count follow.
    if (const std::size_t forced = threadCountFromEnvironment(); forced != 0) {
        return forced;
    }

    const auto& features = arch::get_features();
    std::size_t threadCount = std::thread::hardware_concurrency();

    if (threadCount == detail::ZERO_INT) {
        return detail::FOUR_INT;  // Fallback default
    }

    // For CPU-intensive tasks, use physical cores if hyperthreading is available
    if (features.topology.hyperthreading) {
        const std::size_t physicalCores = features.topology.physical_cores;
        if (physicalCores > detail::ZERO_INT) {
            threadCount = physicalCores;
        }
    }

    return std::max(threadCount, std::size_t{detail::ONE_INT});
}

}  // namespace

std::size_t ThreadPool::getOptimalThreadCount() noexcept {
    // Computed once. ParallelUtils sizes every parallel loop with this, and glibc's
    // std::thread::hardware_concurrency() reads a file under /sys on each call: about 10 us per
    // batch on the Ivy Bridge machine, a quarter of a 4096-element parallel batch. The global
    // pool is sized from the same value at its creation, so a later change in the CPUs available
    // to the process was never followed by the pool either.
    static const std::size_t count = computeOptimalThreadCount();
    return count;
}

//========== ParallelUtils Implementation ==========

ThreadPool& ParallelUtils::getGlobalThreadPool() {
    static ThreadPool globalPool;
    return globalPool;
}

}  // namespace stats
