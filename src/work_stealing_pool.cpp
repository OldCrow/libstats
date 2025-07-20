#include "work_stealing_pool.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <cmath>
#include <random>
#include <chrono>
#include <optional>
#include <future>
#include <memory>

// Platform-specific includes for thread affinity
#ifdef __APPLE__
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <mach/mach.h>
    #include <mach/thread_policy.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <sched.h>
    #include <sys/sysinfo.h>
#elif defined(_WIN32)
    #include <windows.h>
#endif

namespace libstats {

// Level 0-2 Integration Enabling - use specific namespace qualifiers to avoid conflicts
using namespace libstats::safety;
using namespace libstats::constants;
using namespace libstats::simd;

// Alias the result_of_t helper to ensure compatibility
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
    template<typename F, typename... Args>
    using result_of_t = std::invoke_result_t<F, Args...>;
#else
    template<typename F, typename... Args>
    using result_of_t = typename std::result_of<F(Args...)>::type;
#endif

// Thread-local storage for current worker ID
thread_local int WorkStealingPool::currentWorkerId_ = -1;

WorkStealingPool::WorkStealingPool(std::size_t numThreads, bool enableAffinity) {
    if (numThreads == 0) {
        numThreads = getOptimalThreadCount();
    }
    
    workers_.resize(numThreads);
    
    // Create worker threads
    for (std::size_t i = 0; i < numThreads; ++i) {
        workers_[i] = std::make_unique<WorkerData>();
        workers_[i]->worker = std::thread(&WorkStealingPool::workerLoop, this, static_cast<int>(i));
        
        // Set thread affinity if enabled and supported
        if (enableAffinity) {
            setThreadAffinity(workers_[i]->worker, static_cast<int>(i % numThreads));
        }
    }
}

WorkStealingPool::~WorkStealingPool() {
    // Signal shutdown
    shutdown_.store(true, std::memory_order_release);
    
    // Wake up all threads by notifying condition variables
    {
        std::lock_guard<std::mutex> lock(globalMutex_);
        allTasksComplete_.notify_all();
    }
    
    // Wait for all threads to finish
    for (auto& worker : workers_) {
        if (worker && worker->worker.joinable()) {
            worker->worker.join();
        }
    }
}

void WorkStealingPool::submit(Task task) {
    if (shutdown_.load(std::memory_order_acquire)) {
        throw std::runtime_error("Cannot submit task to shutdown WorkStealingPool");
    }
    
    // Try to submit to current worker's local queue first
    const int workerId = currentWorkerId_;
    if (workerId >= 0 && workerId < static_cast<int>(workers_.size())) {
        std::lock_guard<std::mutex> lock(workers_[workerId]->queueMutex);
        workers_[workerId]->localQueue.push_back(std::move(task));
        activeTasks_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    
    // Fall back to round-robin distribution
    static std::atomic<std::size_t> nextWorker{0};
    const std::size_t targetWorker = nextWorker.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    
    std::lock_guard<std::mutex> lock(workers_[targetWorker]->queueMutex);
    workers_[targetWorker]->localQueue.push_back(std::move(task));
    activeTasks_.fetch_add(1, std::memory_order_relaxed);
}

void WorkStealingPool::waitForAll() {
    std::unique_lock<std::mutex> lock(globalMutex_);
    allTasksComplete_.wait(lock, [this] {
        return activeTasks_.load(std::memory_order_acquire) == 0;
    });
}

std::size_t WorkStealingPool::getPendingTasks() const {
    std::size_t total = 0;
    for (const auto& worker : workers_) {
        std::lock_guard<std::mutex> lock(worker->queueMutex);
        total += worker->localQueue.size();
    }
    return total;
}

WorkStealingPool::Statistics WorkStealingPool::getStatistics() const {
    Statistics stats{0, 0, 0, 0.0};
    
    for (const auto& worker : workers_) {
        stats.tasksExecuted += worker->tasksExecuted.load(std::memory_order_relaxed);
        stats.workSteals += worker->workSteals.load(std::memory_order_relaxed);
        stats.failedSteals += worker->failedSteals.load(std::memory_order_relaxed);
    }
    
    const std::size_t totalStealAttempts = stats.workSteals + stats.failedSteals;
    if (totalStealAttempts > 0) {
        stats.stealSuccessRate = static_cast<double>(stats.workSteals) / totalStealAttempts;
    }
    
    return stats;
}

void WorkStealingPool::resetStatistics() {
    for (auto& worker : workers_) {
        worker->tasksExecuted.store(0, std::memory_order_relaxed);
        worker->workSteals.store(0, std::memory_order_relaxed);
        worker->failedSteals.store(0, std::memory_order_relaxed);
    }
}

void WorkStealingPool::workerLoop(int workerId) {
    currentWorkerId_ = workerId;
    
    // Ensure worker data is valid
    if (workerId < 0 || workerId >= static_cast<int>(workers_.size()) || !workers_[workerId]) {
        return;
    }
    
    auto& workerData = *workers_[workerId];
    
    while (!shutdown_.load(std::memory_order_acquire)) {
        Task task;
        bool taskFound = false;
        
        // Try to get task from local queue first
        {
            std::lock_guard<std::mutex> lock(workerData.queueMutex);
            if (!workerData.localQueue.empty()) {
                task = std::move(workerData.localQueue.front());
                workerData.localQueue.pop_front();
                taskFound = true;
            }
        }
        
        // If no local work, try to steal from other workers
        if (!taskFound && !shutdown_.load(std::memory_order_acquire)) {
            task = tryStealWork(workerId);
            if (task) {
                taskFound = true;
                workerData.workSteals.fetch_add(1, std::memory_order_relaxed);
            } else {
                workerData.failedSteals.fetch_add(1, std::memory_order_relaxed);
            }
        }
        
        if (taskFound && task && !shutdown_.load(std::memory_order_acquire)) {
            executeTask(std::move(task), workerId);
        } else {
            // No work available, yield to avoid busy waiting
            std::this_thread::yield();
        }
    }
}

WorkStealingPool::Task WorkStealingPool::tryStealWork(int thiefId) {
    // Check if shutting down
    if (shutdown_.load(std::memory_order_acquire)) {
        return Task{};
    }
    
    const std::size_t numWorkers = workers_.size();
    if (numWorkers <= 1) return Task{};
    
    // Validate thief ID
    if (thiefId < 0 || thiefId >= static_cast<int>(numWorkers) || !workers_[thiefId]) {
        return Task{};
    }
    
    auto& thiefData = *workers_[thiefId];
    
    // Try to steal from a random victim (avoid always stealing from the same worker)
    std::uniform_int_distribution<int> dist(0, static_cast<int>(numWorkers) - 1);
    const int startVictim = dist(thiefData.rng);
    
    for (std::size_t attempt = 0; attempt < numWorkers - 1; ++attempt) {
        const int victimId = (startVictim + attempt) % numWorkers;
        if (victimId == thiefId) continue;
        
        // Validate victim ID and check if still valid
        if (victimId < 0 || victimId >= static_cast<int>(numWorkers) || !workers_[victimId]) {
            continue;
        }
        
        // Check again for shutdown before accessing victim data
        if (shutdown_.load(std::memory_order_acquire)) {
            return Task{};
        }
        
        auto& victimData = *workers_[victimId];
        
        // Try to steal from the back of victim's queue
        std::lock_guard<std::mutex> lock(victimData.queueMutex);
        if (!victimData.localQueue.empty()) {
            Task stolenTask = std::move(victimData.localQueue.back());
            victimData.localQueue.pop_back();
            return stolenTask;
        }
    }
    
    return Task{};
}

void WorkStealingPool::executeTask(Task&& task, int workerId) {
    try {
        task();
        workers_[workerId]->tasksExecuted.fetch_add(1, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        std::cerr << "Exception in work-stealing pool task (worker " << workerId << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in work-stealing pool task (worker " << workerId << ")" << std::endl;
    }
    
    // Decrement active tasks and notify if all complete
    const std::size_t remaining = activeTasks_.fetch_sub(1, std::memory_order_acq_rel) - 1;
    if (remaining == 0) {
        std::lock_guard<std::mutex> lock(globalMutex_);
        allTasksComplete_.notify_all();
    }
}

std::size_t WorkStealingPool::getOptimalThreadCount() noexcept {
    // Use Level 0 CPU detection for accurate thread count determination
    const auto& features = cpu::get_features();
    const auto physicalCores = features.topology.physical_cores;
    const auto logicalCores = features.topology.logical_cores;
    
    // For work-stealing pools, we can use more threads than regular thread pools
    // because work stealing handles load balancing automatically
    if (features.topology.hyperthreading) {
        // Use logical cores for work-stealing - the work stealing algorithm
        // can handle the increased parallelism effectively
        return logicalCores;
    } else {
        // No hyperthreading, use physical cores
        return physicalCores;
    }
}

void WorkStealingPool::setThreadAffinity([[maybe_unused]] std::thread& thread, [[maybe_unused]] int cpuId) {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuId, &cpuset);
    
    const int result = pthread_setaffinity_np(thread.native_handle(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        // Affinity setting failed, but continue without it
        // This is not critical for correctness, only performance
        std::cerr << "Warning: Failed to set thread affinity for CPU " << cpuId << std::endl;
    }
    
#elif defined(__APPLE__)
    // Modern macOS approach: Use thread_policy_set with THREAD_AFFINITY_POLICY
    // Note: macOS affinity is advisory and may not guarantee exclusive CPU binding
    thread_affinity_policy_data_t policy = {cpuId};
    const kern_return_t result = thread_policy_set(
        pthread_mach_thread_np(thread.native_handle()),
        THREAD_AFFINITY_POLICY,
        (thread_policy_t)&policy,
        THREAD_AFFINITY_POLICY_COUNT
    );
    
    if (result != KERN_SUCCESS) {
        // Affinity setting failed, but continue without it
        // This is common on macOS and not critical for correctness
        std::cerr << "Warning: Thread affinity hint failed on macOS (CPU " << cpuId << ")" << std::endl;
    }
    
#elif defined(_WIN32)
    const DWORD_PTR mask = 1ULL << cpuId;
    const DWORD_PTR result = SetThreadAffinityMask(thread.native_handle(), mask);
    if (result == 0) {
        // Affinity setting failed, but continue without it
        // This is not critical for correctness, only performance
        std::cerr << "Warning: Failed to set thread affinity for CPU " << cpuId << std::endl;
    }
    
#else
    // Unsupported platform - skip affinity setting
    // Parameters are marked [[maybe_unused]] in function signature
#endif
}

} // namespace libstats
