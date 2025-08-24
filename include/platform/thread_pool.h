#pragma once

#include "../common/platform_common.h"

#include <future>
#include <optional>
#include <queue>
#include <span>

// Platform headers needed for template implementations
#include "simd.h"

// Level 1 infrastructure
#include "../core/math_utils.h"

namespace stats {

// Compatibility helper for different C++ standard library implementations
// Uses std::invoke_result_t when available (C++20), falls back to std::result_of for older
// compilers
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
template <typename F, typename... Args>
using result_of_t = std::invoke_result_t<F, Args...>;
#else
template <typename F, typename... Args>
using result_of_t = typename std::result_of<F(Args...)>::type;
#endif

/**
 * @brief High-performance thread pool for parallel statistical computations
 *
 * This thread pool is specifically designed for CPU-intensive statistical tasks
 * with minimal synchronization overhead. It integrates deeply with libstats
 * Level 0-2 infrastructure for optimal performance:
 *
 * **Level 0 Integration (Constants & CPU Detection):**
 * - Uses arch::* for optimal thread counts and grain sizes
 * - Integrates with cpu_detection.h for runtime CPU feature detection
 * - Adapts thread count based on hyperthreading detection
 * - Uses platform-specific cache-aware optimization thresholds
 *
 * **Level 1 Integration (Safety & Math Utils):**
 * - Employs detail::* functions for numerical stability checks
 * - Uses math_utils.h for statistical computations within parallel tasks
 * - Provides bounds checking and finite value validation
 *
 * **Level 2 Integration (SIMD & Error Handling):**
 * - SIMD-aware work distribution with arch::simd::double_vector_width()
 * - Uses error_handling.h Result<T> pattern for robust error management
 * - Aligns memory operations to SIMD boundaries for optimal performance
 *
 * @example Basic usage:
 * @code
 * ThreadPool pool;
 *
 * // Submit a statistical computation task
 * auto future = pool.submit([]() {
 *     std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
 *     return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
 * });
 *
 * double mean = future.get();
 * @endcode
 *
 * @example SIMD-aware parallel processing:
 * @code
 * std::vector<double> input(10000);
 * std::vector<double> output(10000);
 *
 * // Parallel transform with SIMD optimization
 * ParallelUtils::parallelTransform(input.data(), output.data(), input.size(),
 *     [](const double* in, double* out, std::size_t size) {
 *         // This function will be called with SIMD-aligned chunks
 *         simd::VectorOps::vector_log(in, out, size);
 *     });
 * @endcode
 */
class ThreadPool {
   public:
    /// Task function type
    using Task = std::function<void()>;

    /// Constructor with specified number of threads
    /// @param numThreads Number of worker threads (0 = auto-detect)
    explicit ThreadPool(std::size_t numThreads = 0);

    /// Destructor - waits for all tasks to complete
    ~ThreadPool();

    /// Submit a task for execution
    /// @param task Task to execute
    /// @return Future that will contain the result
    template <typename F, typename... Args>
    auto submit(F&& task, Args&&... args) -> std::future<result_of_t<F, Args...>> {
        using ReturnType = result_of_t<F, Args...>;

        auto taskPtr = std::make_shared<std::packaged_task<ReturnType()>>(
            std::bind(std::forward<F>(task), std::forward<Args>(args)...));

        std::future<ReturnType> result = taskPtr->get_future();

        {
            std::unique_lock<std::mutex> lock(queueMutex_);

            // Don't allow enqueueing after stopping the pool
            if (stop_) {
                throw std::runtime_error("Cannot submit task to stopped ThreadPool");
            }

            tasks_.emplace([taskPtr]() { (*taskPtr)(); });
        }

        condition_.notify_one();
        return result;
    }

    /// Submit a task without return value
    /// @param task Task to execute
    void submitVoid(Task task);

    /// Get number of worker threads
    /// @return Number of threads in the pool
    std::size_t getNumThreads() const noexcept { return workers_.size(); }

    /// Get number of pending tasks
    /// @return Number of tasks waiting to be executed
    std::size_t getPendingTasks() const;

    /// Wait for all current tasks to complete
    void waitForAll();

    /// Get optimal number of threads for this system
    /// @return Recommended thread count
    static std::size_t getOptimalThreadCount() noexcept;

   private:
    /// Worker threads
    std::vector<std::thread> workers_;

    /// Task queue
    std::queue<Task> tasks_;

    /// Synchronization primitives
    mutable std::mutex queueMutex_;
    std::condition_variable condition_;
    std::condition_variable finished_;

    /// Stop flag and active task counter with C++20 atomic enhancements
    std::atomic<bool> stop_;
    std::atomic<std::size_t> activeTasks_;

    /// C++20 atomic notification support for better performance
    /// @note These improve upon traditional condition variables for simple state changes
    std::atomic<bool> tasksAvailable_{false};
    std::atomic<bool> allTasksComplete_{false};

    /// Worker thread function
    void workerLoop();
};

/**
 * @brief Parallel execution utilities optimized for statistical algorithms
 *
 * This class provides high-level parallel execution patterns specifically
 * designed for statistical computing workloads. It integrates with all
 * levels of the libstats infrastructure:
 *
 * **Performance Features:**
 * - Automatic work distribution based on CPU characteristics
 * - SIMD-aware chunking for optimal vectorization
 * - Cache-conscious grain size calculation
 * - Runtime CPU feature detection for optimization
 *
 * **Safety Features:**
 * - Numerical stability checks using detail::* functions
 * - Bounds checking for array operations
 * - Finite value validation for statistical computations
 *
 * **Integration Features:**
 * - Uses arch::* for optimal thresholds
 * - Leverages simd::* for vectorization opportunities
 * - Employs error_handling.h for robust error management
 * - Integrates with cpu_detection.h for runtime optimization
 *
 * @example Parallel statistical computation:
 * @code
 * std::vector<double> data(100000);
 * // ... fill data ...
 *
 * // Parallel mean calculation
 * double sum = ParallelUtils::parallelReduce(0, data.size(), 0.0,
 *     [&](std::size_t i) { return data[i]; },
 *     [](double a, double b) { return a + b; });
 * double mean = sum / data.size();
 * @endcode
 *
 * @example SIMD-aware parallel transform:
 * @code
 * std::vector<double> input(50000);
 * std::vector<double> output(50000);
 *
 * ParallelUtils::parallelTransform(input.data(), output.data(), input.size(),
 *     [](const double* in, double* out, std::size_t size) {
 *         // Automatically uses SIMD when beneficial
 *         for (std::size_t i = 0; i < size; ++i) {
 *             out[i] = detail::safe_log(in[i]);
 *         }
 *     });
 * @endcode
 */
class ParallelUtils {
   public:
    /// Parallel for loop execution with Level 0-2 integration
    /// @param start Start index (inclusive)
    /// @param end End index (exclusive)
    /// @param task Function to execute for each index
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    template <typename Func>
    static void parallelFor(std::size_t start, std::size_t end, Func&& task,
                            std::size_t grainSize = 0) {
        const std::size_t range = end - start;
        if (range == 0)
            return;

        // Use constants from Level 0 for optimal thresholds
        const std::size_t minParallelSize = arch::parallel::detail::min_elements_for_parallel();
        if (range < minParallelSize) {
            // Execute sequentially for small ranges
            for (std::size_t i = start; i < end; ++i) {
                task(i);
            }
            return;
        }

        // Calculate optimal grain size using Level 0 constants
        const std::size_t actualGrainSize =
            (grainSize == 0) ? arch::parallel::detail::grain_size() : grainSize;
        const std::size_t numThreads = ThreadPool::getOptimalThreadCount();
        const std::size_t optimalGrainSize = std::max(actualGrainSize, range / (numThreads * 4));

        // Parallel execution with Level 0-2 infrastructure
        ThreadPool& pool = getGlobalThreadPool();
        std::vector<std::future<void>> futures;
        futures.reserve((range + optimalGrainSize - 1) / optimalGrainSize);

        for (std::size_t i = start; i < end; i += optimalGrainSize) {
            const std::size_t chunkEnd = std::min(i + optimalGrainSize, end);

            auto future = pool.submit([&task, i, chunkEnd]() {
                for (std::size_t j = i; j < chunkEnd; ++j) {
                    task(j);
                }
            });

            futures.push_back(std::move(future));
        }

        // Wait for all chunks to complete
        for (auto& future : futures) {
            future.wait();
        }
    }

    /// Parallel reduction operation
    /// @param start Start index (inclusive)
    /// @param end End index (exclusive)
    /// @param init Initial value for reduction
    /// @param task Function to compute value for each index
    /// @param reduce Function to combine two values
    /// @param grainSize Minimum work per thread
    /// @return Reduced result
    template <typename T, typename TaskFunc, typename ReduceFunc>
    static T parallelReduce(std::size_t start, std::size_t end, T init, TaskFunc&& task,
                            ReduceFunc&& reduce, std::size_t grainSize = 1) {
        const std::size_t range = end - start;
        if (range == 0)
            return init;

        const std::size_t numThreads = ThreadPool::getOptimalThreadCount();
        const std::size_t actualGrainSize = std::max(grainSize, range / (numThreads * 4));

        if (range <= actualGrainSize) {
            // Execute sequentially for small ranges
            T result = init;
            for (std::size_t i = start; i < end; ++i) {
                result = reduce(result, task(i));
            }
            return result;
        }

        // Parallel execution
        ThreadPool& pool = getGlobalThreadPool();
        std::vector<std::future<T>> futures;

        for (std::size_t i = start; i < end; i += actualGrainSize) {
            const std::size_t chunkEnd = std::min(i + actualGrainSize, end);

            auto future = pool.submit([&task, &reduce, i, chunkEnd, init]() {
                T localResult = init;
                for (std::size_t j = i; j < chunkEnd; ++j) {
                    localResult = reduce(localResult, task(j));
                }
                return localResult;
            });

            futures.push_back(std::move(future));
        }

        // Combine partial results
        T finalResult = init;
        for (auto& future : futures) {
            finalResult = reduce(finalResult, future.get());
        }

        return finalResult;
    }

    /// Get the global thread pool instance
    /// @return Reference to singleton thread pool
    static ThreadPool& getGlobalThreadPool();

    /// SIMD-aware parallel transform operation
    /// @param input Input data array
    /// @param output Output data array
    /// @param size Number of elements to process
    /// @param func Function to apply to each element/chunk
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    template <typename T, typename Func>
    static void parallelTransform(const T* input, T* output, std::size_t size, Func&& func,
                                  std::size_t grainSize = 0) {
        // Use safety checks from Level 1
        detail::check_finite(static_cast<double>(size), "array size");

        const std::size_t minParallelSize = arch::parallel::detail::min_elements_for_parallel();
        if (size < minParallelSize) {
            // Execute sequentially for small arrays
            func(input, output, size);
            return;
        }

        // Calculate SIMD-aware grain size
        const std::size_t simdWidth = arch::simd::double_vector_width();
        const std::size_t baseGrainSize =
            (grainSize == 0) ? arch::parallel::detail::grain_size() : grainSize;

        // Align grain size to SIMD width for optimal performance
        const std::size_t alignedGrainSize =
            ((baseGrainSize + simdWidth - 1) / simdWidth) * simdWidth;

        parallelFor(
            0, size,
            [&](std::size_t i) {
                const std::size_t chunkEnd = std::min(i + alignedGrainSize, size);
                const std::size_t chunkSize = chunkEnd - i;
                func(input + i, output + i, chunkSize);
            },
            alignedGrainSize);
    }

    /// SIMD-aware parallel sum operation (foundation for statistical operations)
    /// @param data Input data span
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    /// @return Sum of all elements
    template <typename T>
    static T parallelSum(std::span<const T> data, std::size_t grainSize = 0) {
        if (data.empty()) {
            return T{0};
        }

        const std::size_t size = data.size();
        const std::size_t minParallelSize =
            arch::parallel::detail::min_elements_for_distribution_parallel();

        if (size < minParallelSize) {
            // Sequential sum for small arrays
            T sum = T{0};
            for (const auto& value : data) {
                sum += value;
            }
            return sum;
        }

        // Use parallel reduce for large arrays
        return parallelReduce(
            0, size, T{0}, [&data](std::size_t i) { return data[i]; },
            [](T a, T b) { return a + b; }, grainSize);
    }

    /// SIMD-aware parallel mean calculation
    /// @param data Input data span
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    /// @return Mean of all elements
    template <typename T>
    static double parallelMean(std::span<const T> data, std::size_t grainSize = 0) {
        if (data.empty()) {
            return 0.0;
        }

        const T sum = parallelSum(data, grainSize);
        return static_cast<double>(sum) / static_cast<double>(data.size());
    }

    /// SIMD-aware parallel variance calculation
    /// @param data Input data span
    /// @param mean Pre-calculated mean (optional, will calculate if not provided)
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    /// @return Variance of all elements
    template <typename T>
    static double parallelVariance(std::span<const T> data,
                                   std::optional<double> mean = std::nullopt,
                                   std::size_t grainSize = 0) {
        if (data.empty()) {
            return 0.0;
        }

        const double actualMean = mean.value_or(parallelMean(data, grainSize));

        const std::size_t size = data.size();
        const std::size_t minParallelSize =
            arch::parallel::detail::min_elements_for_distribution_parallel();

        if (size < minParallelSize) {
            // Sequential variance for small arrays
            double sumSquaredDiffs = 0.0;
            for (const auto& value : data) {
                const double diff = static_cast<double>(value) - actualMean;
                sumSquaredDiffs += diff * diff;
            }
            return sumSquaredDiffs / static_cast<double>(size);
        }

        // Parallel variance calculation
        const double sumSquaredDiffs = parallelReduce(
            0, size, 0.0,
            [&data, actualMean](std::size_t i) {
                const double diff = static_cast<double>(data[i]) - actualMean;
                return diff * diff;
            },
            [](double a, double b) { return a + b; }, grainSize);

        return sumSquaredDiffs / static_cast<double>(size);
    }

    /// SIMD-aware parallel statistical operation with proper aggregation
    /// @param data Input data span
    /// @param operation Statistical operation to perform on chunks
    /// @param combiner Function to combine partial results
    /// @param grainSize Minimum work per thread (0 = auto-detect)
    /// @return Result of the statistical operation
    template <typename T, typename Op, typename Combiner>
    static auto parallelStatOperation(std::span<const T> data, Op&& operation, Combiner&& combiner,
                                      std::size_t grainSize = 0) -> decltype(operation(data)) {
        if (data.empty()) {
            return operation(data.subspan(0, 0));  // Handle empty case
        }

        const std::size_t size = data.size();
        const std::size_t minParallelSize =
            arch::parallel::detail::min_elements_for_distribution_parallel();

        if (size < minParallelSize) {
            return operation(data);
        }

        // Use SIMD-aware chunking
        const std::size_t simdWidth = arch::simd::double_vector_width();
        const std::size_t baseGrainSize =
            (grainSize == 0) ? arch::parallel::detail::grain_size() : grainSize;
        const std::size_t alignedGrainSize =
            ((baseGrainSize + simdWidth - 1) / simdWidth) * simdWidth;

        // Parallel execution with proper reduction
        ThreadPool& pool = getGlobalThreadPool();
        std::vector<std::future<decltype(operation(data))>> futures;

        for (std::size_t i = 0; i < size; i += alignedGrainSize) {
            const std::size_t chunkEnd = std::min(i + alignedGrainSize, size);
            auto chunk = data.subspan(i, chunkEnd - i);

            auto future = pool.submit([&operation, chunk]() { return operation(chunk); });

            futures.push_back(std::move(future));
        }

        // Collect all partial results
        std::vector<decltype(operation(data))> partialResults;
        partialResults.reserve(futures.size());

        for (auto& future : futures) {
            partialResults.push_back(future.get());
        }

        // Combine results using provided combiner
        auto result = partialResults[0];
        for (std::size_t i = 1; i < partialResults.size(); ++i) {
            result = combiner(result, partialResults[i]);
        }

        return result;
    }
};

}  // namespace stats
