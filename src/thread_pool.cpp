#include "../include/platform/thread_pool.h"

#include "../include/platform/platform_constants.h"

// Additional includes for integration
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>

using namespace stats;
using namespace stats::detail;
using namespace stats::arch::parallel;
// Platform-specific includes
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

std::size_t ThreadPool::getOptimalThreadCount() noexcept {
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

//========== ParallelUtils Implementation ==========

ThreadPool& ParallelUtils::getGlobalThreadPool() {
    static ThreadPool globalPool;
    return globalPool;
}

}  // namespace stats
