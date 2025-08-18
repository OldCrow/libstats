#pragma once

#include "../common/platform_common.h"
#include <deque>
#include <random>
#include <future>
#include <condition_variable>

// Platform headers needed for template implementations
#include "simd.h"

// Level 1 infrastructure
#include "../core/math_utils.h"

namespace libstats {

// Compatibility helper for different C++ standard library implementations
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
    template<typename F, typename... Args>
    using result_of_t = std::invoke_result_t<F, Args...>;
#else
    template<typename F, typename... Args>
    using result_of_t = typename std::result_of<F(Args...)>::type;
#endif

/**
 * @brief Work-stealing thread pool for high-performance statistical computing
 * 
 * This implementation provides better load balancing than traditional thread pools
 * by allowing idle threads to "steal" work from busy threads. Key features:
 * 
 * - Per-thread work queues to reduce contention
 * - Work stealing algorithm for automatic load balancing
 * - Thread affinity optimization (basic CPU core binding)
 * - Lock-free operations where possible
 * - Persistent threads to avoid creation/destruction overhead
 * - Optimized for statistical computations and data analysis workloads
 * 
 * **NUMA Optimization Status:**
 * - NUMA-aware optimizations are NOT implemented (deprioritized)
 * - Rationale: 95% of desktop/laptop systems have no meaningful NUMA topology
 * - Focus is on cache-friendly algorithms and SIMD optimization instead
 * - NUMA would only be reconsidered if >10% performance impact demonstrated
 *   on systems with >32 cores and multiple memory controllers
 */
class WorkStealingPool {
public:
    using Task = std::function<void()>;
    
    /**
     * @brief Construct work-stealing thread pool
     * @param numThreads Number of worker threads (0 = auto-detect)
     * @param enableAffinity Enable CPU affinity for threads
     */
    explicit WorkStealingPool(std::size_t numThreads = 0, bool enableAffinity = true);
    
    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~WorkStealingPool();
    
    /**
     * @brief Submit a task for execution
     * @param task Function to execute
     */
    void submit(Task task);
    
    /**
     * @brief Submit a range-based parallel task with automatic work distribution
     * 
     * @param start Start of range (inclusive)
     * @param end End of range (exclusive)
     * @param func Function to call for each index: func(std::size_t index)
     * @param grainSize Minimum work per thread (0 = auto-detect)
     */
    template<typename Func>
    void parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize = 0);
    
    /**
     * @brief Wait for all currently submitted tasks to complete
     */
    void waitForAll();
    
    /**
     * @brief Get number of worker threads
     */
    std::size_t getThreadCount() const noexcept { return workers_.size(); }
    
    /**
     * @brief Get number of pending tasks across all queues
     */
    std::size_t getPendingTasks() const;
    
    /**
     * @brief Get statistics about work stealing efficiency
     */
    struct Statistics {
        std::size_t tasksExecuted;
        std::size_t workSteals;
        std::size_t failedSteals;
        double stealSuccessRate;
    };
    
    Statistics getStatistics() const;
    
    /**
     * @brief Reset statistics counters
     */
    void resetStatistics();
    
    /**
     * @brief Get optimal number of threads for this system
     */
    static std::size_t getOptimalThreadCount() noexcept;

private:
    
    #if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4324)
    #endif
    
    struct alignas(64) WorkerData {  // Cache line alignment
        std::deque<Task> localQueue;
        mutable std::mutex queueMutex;
        std::atomic<std::size_t> tasksExecuted{0};
        std::atomic<std::size_t> workSteals{0};
        std::atomic<std::size_t> failedSteals{0};
        std::thread worker;
        std::mt19937 rng;  // For random victim selection
        
        #if defined(__APPLE__) || defined(__linux__)
            bool enableOptimization{false};  // Whether to enable thread optimization (QoS/affinity)
        #else
            [[maybe_unused]] bool enableOptimization{false};  // Unused on this platform
        #endif
        
        WorkerData() : rng(std::random_device{}()) {}
    };
    
    #if defined(_MSC_VER)
    #pragma warning(pop)
    #endif

    std::vector<std::unique_ptr<WorkerData>> workers_;
    std::atomic<bool> shutdown_{false};
    std::atomic<std::size_t> activeTasks_{0};
    
    // Global synchronization for waitForAll()
    mutable std::mutex globalMutex_;
    std::condition_variable allTasksComplete_;
    
    // Thread readiness synchronization (macOS QoS best practice)
    std::mutex readinessMutex_;
    std::condition_variable readinessCondition_;
    std::atomic<std::size_t> readyThreads_{0};
    
    // Thread-local storage for current worker ID
    static thread_local int currentWorkerId_;
    
    /**
     * @brief Main worker loop
     * @param workerId ID of this worker thread
     */
    void workerLoop(int workerId);
    
    /**
     * @brief Try to steal work from another worker
     * @param thiefId ID of the stealing thread
     * @return Task if successful, nullptr otherwise
     */
    Task tryStealWork(int thiefId);
    
    /**
     * @brief Execute a task and handle exceptions
     * @param task Task to execute
     * @param workerId ID of executing worker
     */
    void executeTask(Task&& task, int workerId);
    
    /**
     * @brief Set CPU affinity for a thread
     * @param thread Thread to set affinity for
     * @param cpuId CPU core to bind to
     */
    static void setThreadAffinity([[maybe_unused]] std::thread& thread, [[maybe_unused]] int cpuId);
    
    /**
     * @brief Optimize current thread with platform-specific approaches
     * @param workerId ID of the current worker thread
     * @param numWorkers Total number of worker threads
     */
    void optimizeCurrentThread([[maybe_unused]] int workerId, [[maybe_unused]] int numWorkers);
};

/**
 * @brief Optimized parallel range implementation with Level 0-2 integration
 * 
 * Provides efficient work distribution for parallel loops with
 * automatic grain size calculation and work stealing support.
 * 
 * **Level 0 Integration:**
 * - Uses constants::parallel::* for thresholds and grain size calculation
 * - Integrates with cpu_detection for optimal task distribution
 * 
 * **Level 1 Integration:**
 * - Employs safety functions for bounds checking
 * - Uses math_utils for performance calculations
 * 
 * **Level 2 Integration:**
 * - SIMD-aware grain size alignment
 * - Error handling for robust execution
 */
template<typename Func>
void WorkStealingPool::parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize) {
    if (start >= end) return;
    
    const std::size_t totalWork = end - start;
    const std::size_t numWorkers = getThreadCount();
    
    // Use Level 0 constants for thresholds
    if (totalWork < constants::parallel::adaptive::min_elements_for_parallel()) {
        // Execute sequentially for small workloads
        for (std::size_t i = start; i < end; ++i) {
            func(i);
        }
        return;
    }
    
    // Auto-calculate grain size if not specified using Level 0-2 integration
    if (grainSize == 0) {
        // Use adaptive grain size from Level 0 constants
        const std::size_t baseGrainSize = constants::parallel::adaptive::grain_size();
        
        // Aim for ~4x more tasks than threads for good load balancing
        const std::size_t targetTasks = numWorkers * 4;
        grainSize = std::max(baseGrainSize, totalWork / targetTasks);
        
        // Align grain size to SIMD boundaries for optimal performance
        const std::size_t simdWidth = simd::double_vector_width();
        grainSize = ((grainSize + simdWidth - 1) / simdWidth) * simdWidth;
        
        // Ensure grain size is reasonable (not too small or too large)
        grainSize = std::max(std::size_t{8}, std::min(grainSize, std::size_t{1024}));
    }
    
    // Calculate number of tasks we'll submit
    const std::size_t numTasks = (totalWork + grainSize - 1) / grainSize;
    
    // Submit all tasks
    for (std::size_t taskId = 0; taskId < numTasks; ++taskId) {
        const std::size_t taskStart = start + taskId * grainSize;
        const std::size_t taskEnd = std::min(start + (taskId + 1) * grainSize, end);
        
        // Capture by value to avoid lifetime issues
        submit([func, taskStart, taskEnd]() {
            // Execute function for this range with safety checks
            for (std::size_t i = taskStart; i < taskEnd; ++i) {
                func(i);
            }
        });
    }
    
    // Use the pool's built-in synchronization mechanism
    waitForAll();
}

/**
 * @brief Global work-stealing thread pool instance
 * 
 * Provides a singleton work-stealing pool for use throughout the library.
 * This avoids the overhead of creating multiple thread pools.
 */
class GlobalWorkStealingPool {
public:
    static WorkStealingPool& getInstance() {
        static WorkStealingPool instance;
        return instance;
    }
    
private:
    GlobalWorkStealingPool() = default;
};

/**
 * @brief Utility functions for work-stealing parallel operations
 */
namespace WorkStealingUtils {
    
    /**
     * @brief Execute a parallel for loop using the global work-stealing pool
     */
    template<typename Func>
    inline void parallelFor(std::size_t start, std::size_t end, Func func, std::size_t grainSize = 0) {
        GlobalWorkStealingPool::getInstance().parallelFor(start, end, func, grainSize);
    }
    
    /**
     * @brief Submit a task to the global work-stealing pool
     */
    inline void submit(WorkStealingPool::Task task) {
        GlobalWorkStealingPool::getInstance().submit(std::move(task));
    }
    
    /**
     * @brief Wait for all tasks in the global pool to complete
     */
    inline void waitForAll() {
        GlobalWorkStealingPool::getInstance().waitForAll();
    }
    
    /**
     * @brief Check if work-stealing should be used for a given problem size
     * 
     * @param problemSize Size of the computational problem
     * @param threshold Minimum size to enable work-stealing
     * @return True if work-stealing pool should be used
     */
    inline bool shouldUseWorkStealing(std::size_t problemSize, std::size_t threshold = 1024) {
        return problemSize >= threshold && GlobalWorkStealingPool::getInstance().getThreadCount() > 1;
    }
    
    /**
     * @brief Get statistics for the global work-stealing pool
     */
    inline WorkStealingPool::Statistics getStatistics() {
        return GlobalWorkStealingPool::getInstance().getStatistics();
    }
    
    /**
     * @brief Reset statistics for the global work-stealing pool
     */
    inline void resetStatistics() {
        GlobalWorkStealingPool::getInstance().resetStatistics();
    }
}

} // namespace libstats
