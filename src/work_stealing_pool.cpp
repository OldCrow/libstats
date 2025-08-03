#include "../include/platform/work_stealing_pool.h"
#include "../include/platform/platform_constants.h"
#include <algorithm>
#include <iostream>
#include <thread>
#include <cmath>
#include <random>
#include <chrono>
#include <optional>
#include <future>
#include <memory>

// Platform-specific includes for thread optimization
#ifdef __APPLE__
    #include <sys/types.h>
    #include <sys/sysctl.h>
    #include <pthread.h>
    #include <dispatch/dispatch.h>
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
    if (numThreads == constants::math::ZERO_INT) {
        numThreads = getOptimalThreadCount();
    }
    
    workers_.resize(numThreads);
    
    // Create worker threads
    for (std::size_t i = 0; i < numThreads; ++i) {
        workers_[i] = std::make_unique<WorkerData>();
        workers_[i]->worker = std::thread(&WorkStealingPool::workerLoop, this, static_cast<int>(i));
        
        // Store enableAffinity flag for worker thread to use during initialization
        workers_[i]->enableOptimization = enableAffinity;
    }
    
    // macOS QoS Best Practice: Wait for all threads to be ready before returning
    // This prevents race conditions when QoS/affinity setting takes time
    {
        std::unique_lock<std::mutex> lock(readinessMutex_);
        
        // Use timeout to prevent indefinite blocking in case of initialization issues
        const auto timeout = std::chrono::milliseconds(5000); // 5 second max wait
        const bool allReady = readinessCondition_.wait_for(lock, timeout, [this, numThreads] {
            return readyThreads_.load(std::memory_order_acquire) == numThreads;
        });
        
        if (!allReady) {
            // Log warning and continue - this shouldn't happen in normal operation
            const auto actualReady = readyThreads_.load();
            std::cerr << "Warning: WorkStealingPool thread initialization timeout after 5000ms. "
                      << "Ready threads: " << actualReady << "/" << numThreads;
            if (actualReady < numThreads) {
                std::cerr << " (" << (numThreads - actualReady) << " threads still initializing)";
            }
            std::cerr << std::endl;
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
    if (workerId >= constants::math::ZERO_INT && workerId < static_cast<int>(workers_.size())) {
        std::lock_guard<std::mutex> lock(workers_[workerId]->queueMutex);
        workers_[workerId]->localQueue.push_back(std::move(task));
        activeTasks_.fetch_add(constants::math::ONE_INT, std::memory_order_relaxed);
        return;
    }
    
    // Fall back to round-robin distribution
    static std::atomic<std::size_t> nextWorker{0};
    const std::size_t targetWorker = nextWorker.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    
    std::lock_guard<std::mutex> lock(workers_[targetWorker]->queueMutex);
    workers_[targetWorker]->localQueue.push_back(std::move(task));
    activeTasks_.fetch_add(constants::math::ONE_INT, std::memory_order_relaxed);
}

void WorkStealingPool::waitForAll() {
    std::unique_lock<std::mutex> lock(globalMutex_);
    allTasksComplete_.wait(lock, [this] {
        return activeTasks_.load(std::memory_order_acquire) == constants::math::ZERO_INT;
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
    Statistics stats{constants::math::ZERO_INT, constants::math::ZERO_INT, constants::math::ZERO_INT, constants::math::ZERO_DOUBLE};
    
    for (const auto& worker : workers_) {
        stats.tasksExecuted += worker->tasksExecuted.load(std::memory_order_relaxed);
        stats.workSteals += worker->workSteals.load(std::memory_order_relaxed);
        stats.failedSteals += worker->failedSteals.load(std::memory_order_relaxed);
    }
    
    const std::size_t totalStealAttempts = stats.workSteals + stats.failedSteals;
    if (totalStealAttempts > constants::math::ZERO_INT) {
        stats.stealSuccessRate = static_cast<double>(stats.workSteals) / totalStealAttempts;
    }
    
    return stats;
}

void WorkStealingPool::resetStatistics() {
    for (auto& worker : workers_) {
        worker->tasksExecuted.store(constants::math::ZERO_INT, std::memory_order_relaxed);
        worker->workSteals.store(constants::math::ZERO_INT, std::memory_order_relaxed);
        worker->failedSteals.store(constants::math::ZERO_INT, std::memory_order_relaxed);
    }
}

void WorkStealingPool::workerLoop(int workerId) {
    currentWorkerId_ = workerId;
    
    // Ensure worker data is valid
    if (workerId < constants::math::ZERO_INT || workerId >= static_cast<int>(workers_.size()) || !workers_[workerId]) {
        // Still need to signal readiness even if invalid to prevent constructor deadlock
        {
            std::lock_guard<std::mutex> lock(readinessMutex_);
            readyThreads_.fetch_add(1, std::memory_order_release);
            readinessCondition_.notify_one();
        }
        return;
    }
    
    auto& workerData = *workers_[workerId];
    
    // Perform thread self-optimization if enabled (this is where QoS setting happens)
    if (workerData.enableOptimization) {
        optimizeCurrentThread(workerId, static_cast<int>(workers_.size()));
    }
    
    // macOS QoS Best Practice: Signal that this thread is fully initialized and ready
    // This must happen AFTER optimizeCurrentThread() to ensure QoS setting is complete
    {
        std::lock_guard<std::mutex> lock(readinessMutex_);
        [[maybe_unused]] const std::size_t ready = readyThreads_.fetch_add(1, std::memory_order_release) + 1;
        readinessCondition_.notify_one();
        
        // Optional: Log readiness for debugging (can be removed in production)
        #ifdef LIBSTATS_DEBUG_THREADING
            std::cout << "Worker " << workerId << " ready (" << ready << "/" << workers_.size() << ")" << std::endl;
        #endif
    }
    
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
                workerData.workSteals.fetch_add(constants::math::ONE_INT, std::memory_order_relaxed);
            } else {
                workerData.failedSteals.fetch_add(constants::math::ONE_INT, std::memory_order_relaxed);
            }
        }
        
        if (taskFound && task && !shutdown_.load(std::memory_order_acquire)) {
            executeTask(std::move(task), workerId);
        } else {
            // No work available, sleep briefly to avoid busy waiting
            // Use exponential backoff: yield first, then short sleep
            static thread_local int backoffCount = 0;
            if (backoffCount < 10) {
                std::this_thread::yield();
                backoffCount++;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                backoffCount = 0; // Reset backoff counter
            }
        }
    }
}

WorkStealingPool::Task WorkStealingPool::tryStealWork(int thiefId) {
    // Check if shutting down
    if (shutdown_.load(std::memory_order_acquire)) {
        return Task{};
    }
    
    const std::size_t numWorkers = workers_.size();
    if (numWorkers <= constants::math::ONE_INT) return Task{};
    
    // Validate thief ID
    if (thiefId < constants::math::ZERO_INT || thiefId >= static_cast<int>(numWorkers) || !workers_[thiefId]) {
        return Task{};
    }
    
    auto& thiefData = *workers_[thiefId];
    
    // Try to steal from a random victim (avoid always stealing from the same worker)
    std::uniform_int_distribution<int> dist(constants::math::ZERO_INT, static_cast<int>(numWorkers) - constants::math::ONE_INT);
    const int startVictim = dist(thiefData.rng);
    
    for (std::size_t attempt = constants::math::ZERO_INT; attempt < numWorkers - constants::math::ONE_INT; ++attempt) {
        const int victimId = static_cast<int>((startVictim + attempt) % numWorkers);
        if (victimId == thiefId) continue;
        
        // Validate victim ID and check if still valid
        if (victimId < constants::math::ZERO_INT || victimId >= static_cast<int>(numWorkers) || !workers_[victimId]) {
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
        workers_[workerId]->tasksExecuted.fetch_add(constants::math::ONE_INT, std::memory_order_relaxed);
    } catch (const std::exception& e) {
        std::cerr << "Exception in work-stealing pool task (worker " << workerId << "): " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception in work-stealing pool task (worker " << workerId << ")" << std::endl;
    }
    
    // Decrement active tasks and notify if all complete
    const std::size_t remaining = activeTasks_.fetch_sub(constants::math::ONE_INT, std::memory_order_acq_rel) - constants::math::ONE_INT;
    if (remaining == constants::math::ZERO_INT) {
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

void WorkStealingPool::optimizeCurrentThread([[maybe_unused]] int workerId, [[maybe_unused]] int numWorkers) {
#ifdef __linux__
    // On Linux, set CPU affinity from within the thread
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(workerId % sysconf(_SC_NPROCESSORS_ONLN), &cpuset);
    
    const int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        // Affinity setting failed, but continue without it
        // This is not critical for correctness, only performance
        std::cerr << "Warning: Failed to set thread affinity for CPU " << (workerId % sysconf(_SC_NPROCESSORS_ONLN)) << std::endl;
    }
    
#elif defined(__APPLE__)
    // Modern macOS approach: Use QoS classes instead of deprecated thread affinity
    // QoS provides better integration with macOS scheduler and power management
    
    // Determine appropriate QoS class based on worker role
    qos_class_t qos_class;
    int relative_priority = 0;
    
    // For work-stealing pools, we want high-throughput computational work
    // Use USER_INITIATED for responsive performance with good throughput
    qos_class = QOS_CLASS_USER_INITIATED;
    
    // Optionally differentiate priority based on worker ID to help with load balancing
    // Lower worker IDs get slightly higher priority (closer to 0)
    if (numWorkers > 1) {
        // Spread relative priorities from -2 to +2 across workers
        relative_priority = -2 + (4 * workerId) / (numWorkers - 1);
    }
    
    // Apply QoS class to the current thread (self)
    const int result = pthread_set_qos_class_self_np(qos_class, relative_priority);
    if (result != 0) {
        // QoS setting failed, but continue without it
        // This is not critical for correctness, only performance optimization
        // Note: We don't print a warning since this is expected to work on macOS 10.10+
        // and the failure is not concerning enough to spam logs
    }
    
#elif defined(_WIN32)
    // On Windows, set thread affinity from within the thread
    DWORD cpuCount = 1;
    #if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0600
        cpuCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    #else
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        cpuCount = sysinfo.dwNumberOfProcessors;
    #endif
    DWORD_PTR mask = 1ULL << (workerId % cpuCount);
    DWORD_PTR result = SetThreadAffinityMask(GetCurrentThread(), mask);
    if (result == 0) {
        std::cerr << "Warning: Failed to set thread affinity for CPU " << (workerId % cpuCount) << std::endl;
    }
    
#else
    // Unsupported platform - skip thread optimization
    // Parameter is marked [[maybe_unused]] in function signature
#endif
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
    // On macOS, thread optimization is handled inside optimizeCurrentThread()
    // This function doesn't perform any actions on macOS, but is kept for compatibility.
    
#elif defined(_WIN32)
    DWORD cpuCount = 1;
    #if defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0600
        cpuCount = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    #else
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        cpuCount = sysinfo.dwNumberOfProcessors;
    #endif
    DWORD_PTR mask = 1ULL << (cpuId % cpuCount);
    DWORD_PTR result = SetThreadAffinityMask(thread.native_handle(), mask);
    if (result == 0) {
        std::cerr << "Warning: Failed to set thread affinity for CPU " << (cpuId % cpuCount) << std::endl;
    }
    
#else
    // Unsupported platform - skip thread optimization
    // Parameters are marked [[maybe_unused]] in function signature
#endif
}

} // namespace libstats
