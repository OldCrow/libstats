#ifndef LIBSTATS_PARALLEL_EXECUTION_H_
#define LIBSTATS_PARALLEL_EXECUTION_H_

/**
 * @file parallel_execution.h  
 * @brief C++20 parallel execution policy detection and utilities
 * 
 * This header centralizes all C++20 parallel execution support detection
 * and provides utilities for using std::execution policies safely across
 * different compilers and standard library implementations.
 * 
 */

#include <cstddef>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <vector>

// Level 0-2 Infrastructure Integration
#include "constants.h"
#include "cpu_detection.h"
#include "error_handling.h"
#include "safety.h"

// PARALLEL EXECUTION POLICY DETECTION
// Priority order:
// 1. Standard C++20 execution policies (best)
// 2. LLVM experimental PSTL (when stable)
// 3. Platform-specific parallel systems:
//    a. Grand Central Dispatch (GCD) on Apple platforms
//    b. Windows Thread Pool API on Windows
//    c. OpenMP (if available and enabled)
// 4. POSIX threads (pthreads) on Unix/Linux systems
// 5. Fallback to serial execution

// Check for C++20 std::execution policies
#if defined(__cpp_lib_execution)
    #include <execution>
    #define LIBSTATS_HAS_STD_EXECUTION 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
// Check for LLVM experimental PSTL (note: often incomplete)
#elif defined(_LIBCPP_HAS_EXPERIMENTAL_PSTL) && _LIBCPP_STD_VER >= 17
    // Try to include execution header but don't assume policies exist
    #include <execution>
    // On many LLVM implementations, std::execution::par_unseq is not available
    // We'll detect this at compile time and fallback accordingly
    #ifdef __APPLE__
        // On Apple + LLVM, prefer GCD over incomplete PSTL
        #include <dispatch/dispatch.h>
        #define LIBSTATS_HAS_GCD 1
        #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
    #else
        // Non-Apple LLVM: attempt to use experimental PSTL
        #define LIBSTATS_HAS_STD_EXECUTION 1
        #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
    #endif
// Fallback to Grand Central Dispatch on Apple platforms
#elif defined(__APPLE__)
    #include <dispatch/dispatch.h>
    #define LIBSTATS_HAS_GCD 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
// Windows Thread Pool API (Windows Vista+)
#elif defined(_WIN32) && defined(_WIN32_WINNT) && _WIN32_WINNT >= 0x0600
    #include <windows.h>
    #define LIBSTATS_HAS_WIN_THREADPOOL 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
// OpenMP support (cross-platform)
#elif defined(_OPENMP) && _OPENMP >= 200805
    #include <omp.h>
    #define LIBSTATS_HAS_OPENMP 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
// POSIX threads fallback for Unix/Linux systems
#elif defined(__unix__) || defined(__linux__) || (defined(__APPLE__) && defined(__MACH__))
    #include <pthread.h>
    #include <unistd.h>
    #define LIBSTATS_HAS_PTHREADS 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
#else
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 0
#endif

namespace libstats {
namespace parallel {

/**
 * @brief Compile-time check for parallel execution policy support
 * @return true if std::execution policies are available
 */
constexpr bool has_execution_policies() noexcept {
#if LIBSTATS_HAS_PARALLEL_EXECUTION
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get a human-readable description of parallel execution support
 * @return String describing parallel execution capabilities
 */
inline const char* execution_support_string() noexcept {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
    return "C++20 std::execution policies available";
#elif defined(LIBSTATS_HAS_GCD)
    return "Grand Central Dispatch (GCD) parallel execution available";
#elif defined(LIBSTATS_HAS_WIN_THREADPOOL)
    return "Windows Thread Pool API parallel execution available";
#elif defined(LIBSTATS_HAS_OPENMP)
    return "OpenMP parallel execution available";
#elif defined(LIBSTATS_HAS_PTHREADS)
    return "POSIX threads (pthreads) parallel execution available";
#else
    return "No parallel execution available (serial fallback)";
#endif
}

/**
 * @brief Get CPU-aware optimal parallel threshold
 * @return Optimal minimum elements for parallel processing based on CPU features
 */
inline std::size_t get_optimal_parallel_threshold() noexcept {
    return constants::parallel::adaptive::min_elements_for_parallel();
}

/**
 * @brief Get CPU-aware optimal grain size for parallel operations
 * @return Optimal grain size for work distribution based on CPU features
 */
inline std::size_t get_optimal_grain_size() noexcept {
    return constants::parallel::adaptive::grain_size();
}

/**
 * @brief Get platform-adaptive grain size for specific operation types
 * @param operation_type Type of operation (0=memory, 1=computation, 2=mixed)
 * @param data_size Size of data being processed
 * @return Optimized grain size for the operation
 */
inline std::size_t get_adaptive_grain_size(int operation_type = 0, std::size_t data_size = 0) noexcept {
    const auto& features = cpu::get_features();
    const auto base_grain = constants::parallel::adaptive::grain_size();
    
    // Adjust grain size based on operation type and platform capabilities
    std::size_t adjusted_grain = base_grain;
    
    // Platform-specific adjustments
    #if defined(__APPLE__) && defined(__aarch64__)
        // Apple Silicon: Fast thread creation, can handle smaller grains
        adjusted_grain = base_grain / 2;
    #elif defined(__x86_64__) && (defined(__AVX2__) || defined(__AVX512F__))
        // High-end x86_64: Larger grains for better SIMD utilization
        adjusted_grain = base_grain * 2;
    #endif
    
    // Operation type adjustments
    switch (operation_type) {
        case 0: // Memory-bound operations
            if (features.l3_cache_size > 0 && data_size > 0) {
                // Adjust grain to fit well in cache hierarchy
                const std::size_t cache_elements = features.l3_cache_size / (sizeof(double) * 4);
                if (data_size > cache_elements) {
                    adjusted_grain = std::max(adjusted_grain, cache_elements / cpu::get_logical_core_count());
                }
            }
            break;
        case 1: // Computation-bound operations
            // Larger grains for computation to amortize thread overhead
            adjusted_grain *= 2;
            break;
        case 2: // Mixed operations (default)
        default:
            // Use base grain size
            break;
    }
    
    return std::max(adjusted_grain, static_cast<std::size_t>(64)); // Minimum grain size
}

/**
 * @brief Get optimal number of threads for current platform
 * @param workload_size Total amount of work to be parallelized
 * @return Optimal number of threads to use
 */
inline std::size_t get_optimal_thread_count(std::size_t workload_size = 0) noexcept {
    const std::size_t logical_cores = cpu::get_logical_core_count();
    const std::size_t physical_cores = cpu::get_physical_core_count();
    
    // Platform-specific thread count optimization
    #if defined(__APPLE__) && defined(__aarch64__)
        // Apple Silicon: Excellent threading, can use more threads
        std::size_t optimal_threads = logical_cores;
        
        // For very large workloads, consider using more threads
        if (workload_size > 100000) {
            optimal_threads = std::min(logical_cores + 2, logical_cores * 3 / 2);
        }
    #elif defined(__x86_64__)
        // x86_64: Balance between physical and logical cores
        std::size_t optimal_threads = physical_cores;
        
        // Use hyperthreading for memory-bound workloads
        if (workload_size > 50000) {
            optimal_threads = logical_cores;
        }
    #else
        // Conservative default: use physical cores
        std::size_t optimal_threads = physical_cores;
    #endif
    
    return std::max(optimal_threads, static_cast<std::size_t>(1));
}

/**
 * @brief Check if a problem size is large enough to benefit from parallel execution
 * @param problem_size Total number of elements or operations
 * @param threshold Minimum size to consider parallel execution (0 = use CPU-aware default)
 * @return true if parallel execution is likely beneficial
 */
inline bool should_use_parallel(std::size_t problem_size, std::size_t threshold = 0) noexcept {
    const std::size_t actual_threshold = (threshold == 0) ? 
        get_optimal_parallel_threshold() : threshold;
    return has_execution_policies() && (problem_size >= actual_threshold);
}

/**
 * @brief Check if a problem size is large enough for distribution parallel processing
 * @param problem_size Total number of elements or operations
 * @return true if parallel execution is likely beneficial for distribution operations
 */
inline bool should_use_distribution_parallel(std::size_t problem_size) noexcept {
    return has_execution_policies() && 
           (problem_size >= constants::parallel::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL);
}

//==============================================================================
// SAFE PARALLEL EXECUTION MACROS
//==============================================================================

/**
 * @brief Safely execute parallel algorithms with automatic fallback
 * 
 * These macros provide a clean way to use parallel algorithms with automatic
 * fallback to serial algorithms when parallel execution is not available.
 */

/// Execute with parallel unseq policy if available, otherwise serial
#if defined(LIBSTATS_HAS_STD_EXECUTION)
    #define LIBSTATS_PAR_UNSEQ std::execution::par_unseq,
    #define LIBSTATS_PAR std::execution::par,
    #define LIBSTATS_SEQ std::execution::seq,
#else
    #define LIBSTATS_PAR_UNSEQ
    #define LIBSTATS_PAR  
    #define LIBSTATS_SEQ
#endif

//==============================================================================
// GCD PARALLEL ALGORITHM IMPLEMENTATIONS
//==============================================================================

#if defined(LIBSTATS_HAS_GCD)

namespace detail {

/// Helper function to determine optimal chunk size for GCD
inline size_t get_gcd_chunk_size(size_t total_elements) noexcept {
    const size_t num_cores = cpu::get_logical_core_count();
    const size_t min_chunk = get_optimal_grain_size();
    const size_t chunk_size = std::max(min_chunk, total_elements / (num_cores * 4));
    return chunk_size;
}

/// Helper function to calculate number of chunks safely
inline size_t calculate_num_chunks(size_t total_elements, size_t chunk_size) noexcept {
    return (total_elements + chunk_size - 1) / chunk_size;
}

/// Helper function to calculate chunk bounds safely
inline std::pair<size_t, size_t> get_chunk_bounds(size_t chunk_index, size_t chunk_size, size_t total_elements) noexcept {
    const size_t start_idx = chunk_index * chunk_size;
    const size_t end_idx = std::min(start_idx + chunk_size, total_elements);
    return {start_idx, end_idx};
}

/// GCD-based parallel for_each implementation
template<typename Iterator, typename UnaryFunction>
void gcd_for_each(Iterator first, Iterator last, UnaryFunction f) noexcept(noexcept(f(*first))) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    
    dispatch_apply(num_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const auto [start_idx, end_idx] = get_chunk_bounds(chunk_index, chunk_size, total_elements);
                       auto chunk_first = first + start_idx;
                       auto chunk_last = first + end_idx;
                       std::for_each(chunk_first, chunk_last, f);
                   });
}

/// GCD-based parallel transform implementation
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void gcd_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) noexcept(noexcept(op(*first1))) {
    const size_t total_elements = std::distance(first1, last1);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    
    dispatch_apply(num_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const auto [start_idx, end_idx] = get_chunk_bounds(chunk_index, chunk_size, total_elements);
                       auto chunk_first1 = first1 + start_idx;
                       auto chunk_last1 = first1 + end_idx;
                       auto chunk_first2 = first2 + start_idx;
                       std::transform(chunk_first1, chunk_last1, chunk_first2, op);
                   });
}

/// GCD-based parallel fill implementation
template<typename Iterator, typename T>
void gcd_fill(Iterator first, Iterator last, const T& value) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    
    dispatch_apply(total_elements / chunk_size + ((total_elements % chunk_size) ? 1 : 0),
                   dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const size_t start_idx = chunk_index * chunk_size;
                       const size_t end_idx = std::min(start_idx + chunk_size, total_elements);
                       auto chunk_first = first + start_idx;
                       auto chunk_last = first + end_idx;
                       std::fill(chunk_first, chunk_last, value);
                   });
}

/// GCD-based parallel reduce implementation
template<typename Iterator, typename T, typename BinaryOp>
T gcd_reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    const size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    
    std::vector<T> partial_results(num_chunks, init);
    T* results_ptr = partial_results.data();
    
    dispatch_apply(num_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const size_t start_idx = chunk_index * chunk_size;
                       const size_t end_idx = std::min(start_idx + chunk_size, total_elements);
                       auto chunk_first = first + start_idx;
                       auto chunk_last = first + end_idx;
                       results_ptr[chunk_index] = std::accumulate(chunk_first, chunk_last, init, op);
                   });
    
    // Combine partial results
    return std::accumulate(partial_results.begin(), partial_results.end(), init, op);
}

/// GCD-based parallel count implementation  
template<typename Iterator, typename T>
typename std::iterator_traits<Iterator>::difference_type
gcd_count(Iterator first, Iterator last, const T& value) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    const size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    
    std::vector<typename std::iterator_traits<Iterator>::difference_type> partial_results(num_chunks, 0);
    auto* results_ptr = partial_results.data();
    
    dispatch_apply(num_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const size_t start_idx = chunk_index * chunk_size;
                       const size_t end_idx = std::min(start_idx + chunk_size, total_elements);
                       auto chunk_first = first + start_idx;
                       auto chunk_last = first + end_idx;
                       results_ptr[chunk_index] = std::count(chunk_first, chunk_last, value);
                   });
    
    // Sum partial results
    return std::accumulate(partial_results.begin(), partial_results.end(), 
                          typename std::iterator_traits<Iterator>::difference_type{0});
}

/// GCD-based parallel count_if implementation
template<typename Iterator, typename UnaryPredicate>
typename std::iterator_traits<Iterator>::difference_type
gcd_count_if(Iterator first, Iterator last, UnaryPredicate pred) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_gcd_chunk_size(total_elements);
    const size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    
    std::vector<typename std::iterator_traits<Iterator>::difference_type> partial_results(num_chunks, 0);
    auto* results_ptr = partial_results.data();
    
    dispatch_apply(num_chunks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0),
                   ^(size_t chunk_index) {
                       const size_t start_idx = chunk_index * chunk_size;
                       const size_t end_idx = std::min(start_idx + chunk_size, total_elements);
                       auto chunk_first = first + start_idx;
                       auto chunk_last = first + end_idx;
                       results_ptr[chunk_index] = std::count_if(chunk_first, chunk_last, pred);
                   });
    
    // Sum partial results
    return std::accumulate(partial_results.begin(), partial_results.end(), 
                          typename std::iterator_traits<Iterator>::difference_type{0});
}

} // namespace detail

#endif // LIBSTATS_HAS_GCD

//==============================================================================
// WINDOWS THREAD POOL API PARALLEL ALGORITHM IMPLEMENTATIONS
//==============================================================================

#if defined(LIBSTATS_HAS_WIN_THREADPOOL)

namespace detail {

/// Helper structure for Windows Thread Pool work items
struct WorkItem {
    std::function<void()> work_function;
    HANDLE completion_event;
    std::atomic<bool>* completion_flag;
};

/// Windows Thread Pool callback function
VOID CALLBACK ThreadPoolWorkCallback(PTP_CALLBACK_INSTANCE Instance, PVOID Context, PTP_WORK Work) {
    WorkItem* item = static_cast<WorkItem*>(Context);
    try {
        item->work_function();
    } catch (...) {
        // Swallow exceptions to prevent thread pool corruption
    }
    if (item->completion_flag) {
        item->completion_flag->store(true, std::memory_order_release);
    }
    if (item->completion_event) {
        SetEvent(item->completion_event);
    }
}

/// Helper function to determine optimal chunk size for Windows Thread Pool
inline size_t get_wintp_chunk_size(size_t total_elements) noexcept {
    const size_t num_cores = cpu::get_logical_core_count();
    const size_t min_chunk = get_optimal_grain_size();
    const size_t chunk_size = std::max(min_chunk, total_elements / (num_cores * 2));
    return chunk_size;
}

/// Windows Thread Pool-based parallel for_each implementation
template<typename Iterator, typename UnaryFunction>
void wintp_for_each(Iterator first, Iterator last, UnaryFunction f) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_wintp_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    
    if (num_chunks <= 1) {
        std::for_each(first, last, f);
        return;
    }
    
    std::vector<PTP_WORK> work_items(num_chunks);
    std::vector<WorkItem> work_data(num_chunks);
    std::vector<std::atomic<bool>> completion_flags(num_chunks);
    
    // Create work items
    for (size_t i = 0; i < num_chunks; ++i) {
        const auto [start_idx, end_idx] = get_chunk_bounds(i, chunk_size, total_elements);
        auto chunk_first = first + start_idx;
        auto chunk_last = first + end_idx;
        
        completion_flags[i].store(false, std::memory_order_relaxed);
        work_data[i].work_function = [chunk_first, chunk_last, f]() {
            std::for_each(chunk_first, chunk_last, f);
        };
        work_data[i].completion_flag = &completion_flags[i];
        work_data[i].completion_event = nullptr;
        
        work_items[i] = CreateThreadpoolWork(ThreadPoolWorkCallback, &work_data[i], nullptr);
        if (work_items[i]) {
            SubmitThreadpoolWork(work_items[i]);
        }
    }
    
    // Wait for completion
    for (size_t i = 0; i < num_chunks; ++i) {
        if (work_items[i]) {
            WaitForThreadpoolWorkCallbacks(work_items[i], FALSE);
            CloseThreadpoolWork(work_items[i]);
        }
    }
}

/// Windows Thread Pool-based parallel transform implementation
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void wintp_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) {
    const size_t total_elements = std::distance(first1, last1);
    const size_t chunk_size = get_wintp_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    
    if (num_chunks <= 1) {
        std::transform(first1, last1, first2, op);
        return;
    }
    
    std::vector<PTP_WORK> work_items(num_chunks);
    std::vector<WorkItem> work_data(num_chunks);
    
    // Create work items
    for (size_t i = 0; i < num_chunks; ++i) {
        const auto [start_idx, end_idx] = get_chunk_bounds(i, chunk_size, total_elements);
        auto chunk_first1 = first1 + start_idx;
        auto chunk_last1 = first1 + end_idx;
        auto chunk_first2 = first2 + start_idx;
        
        work_data[i].work_function = [chunk_first1, chunk_last1, chunk_first2, op]() {
            std::transform(chunk_first1, chunk_last1, chunk_first2, op);
        };
        work_data[i].completion_flag = nullptr;
        work_data[i].completion_event = nullptr;
        
        work_items[i] = CreateThreadpoolWork(ThreadPoolWorkCallback, &work_data[i], nullptr);
        if (work_items[i]) {
            SubmitThreadpoolWork(work_items[i]);
        }
    }
    
    // Wait for completion
    for (size_t i = 0; i < num_chunks; ++i) {
        if (work_items[i]) {
            WaitForThreadpoolWorkCallbacks(work_items[i], FALSE);
            CloseThreadpoolWork(work_items[i]);
        }
    }
}

/// Windows Thread Pool-based parallel reduce implementation
template<typename Iterator, typename T, typename BinaryOp>
T wintp_reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_wintp_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    
    if (num_chunks <= 1) {
        return std::accumulate(first, last, init, op);
    }
    
    std::vector<T> partial_results(num_chunks, init);
    std::vector<PTP_WORK> work_items(num_chunks);
    std::vector<WorkItem> work_data(num_chunks);
    
    // Create work items
    for (size_t i = 0; i < num_chunks; ++i) {
        const auto [start_idx, end_idx] = get_chunk_bounds(i, chunk_size, total_elements);
        auto chunk_first = first + start_idx;
        auto chunk_last = first + end_idx;
        
        work_data[i].work_function = [chunk_first, chunk_last, &partial_results, i, init, op]() {
            partial_results[i] = std::accumulate(chunk_first, chunk_last, init, op);
        };
        work_data[i].completion_flag = nullptr;
        work_data[i].completion_event = nullptr;
        
        work_items[i] = CreateThreadpoolWork(ThreadPoolWorkCallback, &work_data[i], nullptr);
        if (work_items[i]) {
            SubmitThreadpoolWork(work_items[i]);
        }
    }
    
    // Wait for completion
    for (size_t i = 0; i < num_chunks; ++i) {
        if (work_items[i]) {
            WaitForThreadpoolWorkCallbacks(work_items[i], FALSE);
            CloseThreadpoolWork(work_items[i]);
        }
    }
    
    // Combine partial results
    return std::accumulate(partial_results.begin(), partial_results.end(), init, op);
}

} // namespace detail

#endif // LIBSTATS_HAS_WIN_THREADPOOL

//==============================================================================
// OPENMP PARALLEL ALGORITHM IMPLEMENTATIONS
//==============================================================================

#if defined(LIBSTATS_HAS_OPENMP)

namespace detail {

/// Helper function to determine optimal chunk size for OpenMP
inline size_t get_openmp_chunk_size(size_t total_elements) noexcept {
    const size_t num_threads = omp_get_max_threads();
    const size_t min_chunk = get_optimal_grain_size();
    const size_t chunk_size = std::max(min_chunk, total_elements / (num_threads * 4));
    return chunk_size;
}

/// OpenMP-based parallel for_each implementation
template<typename Iterator, typename UnaryFunction>
void openmp_for_each(Iterator first, Iterator last, UnaryFunction f) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        std::for_each(first, last, f);
        return;
    }
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        f(*(first + i));
    }
}

/// OpenMP-based parallel transform implementation
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void openmp_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) {
    const size_t total_elements = std::distance(first1, last1);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        std::transform(first1, last1, first2, op);
        return;
    }
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        *(first2 + i) = op(*(first1 + i));
    }
}

/// OpenMP-based parallel fill implementation
template<typename Iterator, typename T>
void openmp_fill(Iterator first, Iterator last, const T& value) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        std::fill(first, last, value);
        return;
    }
    
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        *(first + i) = value;
    }
}

/// OpenMP-based parallel reduce implementation
template<typename Iterator, typename T, typename BinaryOp>
T openmp_reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        return std::accumulate(first, last, init, op);
    }
    
    T result = init;
    
    #pragma omp parallel for reduction(+:result) schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        // Note: This assumes op is associative and commutative (like +)
        // For more complex operations, we'd need a different approach
        result = op(result, *(first + i));
    }
    
    return result;
}

/// OpenMP-based parallel count implementation
template<typename Iterator, typename T>
typename std::iterator_traits<Iterator>::difference_type
openmp_count(Iterator first, Iterator last, const T& value) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        return std::count(first, last, value);
    }
    
    typename std::iterator_traits<Iterator>::difference_type result = 0;
    
    #pragma omp parallel for reduction(+:result) schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        if (*(first + i) == value) {
            ++result;
        }
    }
    
    return result;
}

/// OpenMP-based parallel count_if implementation
template<typename Iterator, typename UnaryPredicate>
typename std::iterator_traits<Iterator>::difference_type
openmp_count_if(Iterator first, Iterator last, UnaryPredicate pred) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_openmp_chunk_size(total_elements);
    
    if (total_elements < get_optimal_parallel_threshold()) {
        return std::count_if(first, last, pred);
    }
    
    typename std::iterator_traits<Iterator>::difference_type result = 0;
    
    #pragma omp parallel for reduction(+:result) schedule(dynamic, chunk_size)
    for (size_t i = 0; i < total_elements; ++i) {
        if (pred(*(first + i))) {
            ++result;
        }
    }
    
    return result;
}

} // namespace detail

#endif // LIBSTATS_HAS_OPENMP

//==============================================================================
// POSIX THREADS PARALLEL ALGORITHM IMPLEMENTATIONS
//==============================================================================

#if defined(LIBSTATS_HAS_PTHREADS)

namespace detail {

/// Helper structure for pthread work items
struct PThreadWorkItem {
    std::function<void()> work_function;
    std::atomic<bool> completed{false};
    std::exception_ptr exception_ptr;
};

/// pthread worker function
void* pthread_worker(void* arg) {
    PThreadWorkItem* item = static_cast<PThreadWorkItem*>(arg);
    try {
        item->work_function();
    } catch (...) {
        item->exception_ptr = std::current_exception();
    }
    item->completed.store(true, std::memory_order_release);
    return nullptr;
}

/// Helper function to determine optimal chunk size for pthreads
inline size_t get_pthread_chunk_size(size_t total_elements) noexcept {
    const size_t num_cores = cpu::get_logical_core_count();
    const size_t min_chunk = get_optimal_grain_size();
    const size_t chunk_size = std::max(min_chunk, total_elements / num_cores);
    return chunk_size;
}

/// pthreads-based parallel for_each implementation
template<typename Iterator, typename UnaryFunction>
void pthread_for_each(Iterator first, Iterator last, UnaryFunction f) {
    const size_t total_elements = std::distance(first, last);
    const size_t chunk_size = get_pthread_chunk_size(total_elements);
    const size_t num_chunks = calculate_num_chunks(total_elements, chunk_size);
    const size_t max_threads = std::min(num_chunks, cpu::get_logical_core_count());
    
    if (total_elements < get_optimal_parallel_threshold() || max_threads <= 1) {
        std::for_each(first, last, f);
        return;
    }
    
    std::vector<pthread_t> threads(max_threads);
    std::vector<PThreadWorkItem> work_items(max_threads);
    
    // Create threads
    for (size_t t = 0; t < max_threads; ++t) {
        const size_t start_idx = (t * total_elements) / max_threads;
        const size_t end_idx = ((t + 1) * total_elements) / max_threads;
        auto chunk_first = first + start_idx;
        auto chunk_last = first + end_idx;
        
        work_items[t].work_function = [chunk_first, chunk_last, f]() {
            std::for_each(chunk_first, chunk_last, f);
        };
        
        if (pthread_create(&threads[t], nullptr, pthread_worker, &work_items[t]) != 0) {
            // If thread creation fails, fall back to serial execution for remaining work
            std::for_each(chunk_first, last, f);
            // Wait for already created threads
            for (size_t j = 0; j < t; ++j) {
                pthread_join(threads[j], nullptr);
            }
            return;
        }
    }
    
    // Wait for all threads to complete
    for (size_t t = 0; t < max_threads; ++t) {
        pthread_join(threads[t], nullptr);
        if (work_items[t].exception_ptr) {
            std::rethrow_exception(work_items[t].exception_ptr);
        }
    }
}

/// pthreads-based parallel transform implementation
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void pthread_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) {
    const size_t total_elements = std::distance(first1, last1);
    const size_t max_threads = std::min(total_elements / get_optimal_grain_size(), cpu::get_logical_core_count());
    
    if (total_elements < get_optimal_parallel_threshold() || max_threads <= 1) {
        std::transform(first1, last1, first2, op);
        return;
    }
    
    std::vector<pthread_t> threads(max_threads);
    std::vector<PThreadWorkItem> work_items(max_threads);
    
    // Create threads
    for (size_t t = 0; t < max_threads; ++t) {
        const size_t start_idx = (t * total_elements) / max_threads;
        const size_t end_idx = ((t + 1) * total_elements) / max_threads;
        auto chunk_first1 = first1 + start_idx;
        auto chunk_last1 = first1 + end_idx;
        auto chunk_first2 = first2 + start_idx;
        
        work_items[t].work_function = [chunk_first1, chunk_last1, chunk_first2, op]() {
            std::transform(chunk_first1, chunk_last1, chunk_first2, op);
        };
        
        if (pthread_create(&threads[t], nullptr, pthread_worker, &work_items[t]) != 0) {
            // If thread creation fails, fall back to serial execution
            std::transform(chunk_first1, last1, chunk_first2, op);
            for (size_t j = 0; j < t; ++j) {
                pthread_join(threads[j], nullptr);
            }
            return;
        }
    }
    
    // Wait for all threads to complete
    for (size_t t = 0; t < max_threads; ++t) {
        pthread_join(threads[t], nullptr);
        if (work_items[t].exception_ptr) {
            std::rethrow_exception(work_items[t].exception_ptr);
        }
    }
}

/// pthreads-based parallel reduce implementation
template<typename Iterator, typename T, typename BinaryOp>
T pthread_reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    const size_t total_elements = std::distance(first, last);
    const size_t max_threads = std::min(total_elements / get_optimal_grain_size(), cpu::get_logical_core_count());
    
    if (total_elements < get_optimal_parallel_threshold() || max_threads <= 1) {
        return std::accumulate(first, last, init, op);
    }
    
    std::vector<pthread_t> threads(max_threads);
    std::vector<PThreadWorkItem> work_items(max_threads);
    std::vector<T> partial_results(max_threads, init);
    
    // Create threads
    for (size_t t = 0; t < max_threads; ++t) {
        const size_t start_idx = (t * total_elements) / max_threads;
        const size_t end_idx = ((t + 1) * total_elements) / max_threads;
        auto chunk_first = first + start_idx;
        auto chunk_last = first + end_idx;
        
        work_items[t].work_function = [chunk_first, chunk_last, &partial_results, t, init, op]() {
            partial_results[t] = std::accumulate(chunk_first, chunk_last, init, op);
        };
        
        if (pthread_create(&threads[t], nullptr, pthread_worker, &work_items[t]) != 0) {
            // If thread creation fails, compute remaining serially
            partial_results[t] = std::accumulate(chunk_first, last, init, op);
            for (size_t j = 0; j < t; ++j) {
                pthread_join(threads[j], nullptr);
            }
            break;
        }
    }
    
    // Wait for all threads to complete
    for (size_t t = 0; t < max_threads; ++t) {
        pthread_join(threads[t], nullptr);
        if (work_items[t].exception_ptr) {
            std::rethrow_exception(work_items[t].exception_ptr);
        }
    }
    
    // Combine partial results
    return std::accumulate(partial_results.begin(), partial_results.end(), init, op);
}

} // namespace detail

#endif // LIBSTATS_HAS_PTHREADS

//==============================================================================
// PARALLEL ALGORITHM WRAPPERS
//==============================================================================

/**
 * @brief Safe wrappers for common parallel algorithms
 * 
 * These provide a consistent API that automatically uses parallel execution
 * when available and falls back to serial execution otherwise.
 */

/// Safe parallel fill operation with Level 0-2 integration
template<typename Iterator, typename T>
void safe_fill(Iterator first, Iterator last, const T& value) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::fill(std::execution::par_unseq, first, last, value);
#elif defined(LIBSTATS_HAS_GCD)
        detail::gcd_fill(first, last, value);
#elif defined(LIBSTATS_HAS_WIN_THREADPOOL)
        // Note: Windows Thread Pool doesn't have a direct fill implementation
        std::fill(first, last, value);
#elif defined(LIBSTATS_HAS_OPENMP)
        detail::openmp_fill(first, last, value);
#elif defined(LIBSTATS_HAS_PTHREADS)
        // Note: pthreads doesn't have a direct fill implementation
        std::fill(first, last, value);
#else
        std::fill(first, last, value);
#endif
    } else {
        std::fill(first, last, value);
    }
}

/// Safe parallel transform operation with Level 0-2 integration
template<typename Iterator1, typename Iterator2, typename UnaryOp>
void safe_transform(Iterator1 first1, Iterator1 last1, Iterator2 first2, UnaryOp op) {
    const auto count = std::distance(first1, last1);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::transform(std::execution::par_unseq, first1, last1, first2, op);
#elif defined(LIBSTATS_HAS_GCD)
        detail::gcd_transform(first1, last1, first2, op);
#else
        std::transform(first1, last1, first2, op);
#endif
    } else {
        std::transform(first1, last1, first2, op);
    }
}

/// Safe parallel reduce operation with Level 0-2 integration
template<typename Iterator, typename T>
T safe_reduce(Iterator first, Iterator last, T init) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        return std::reduce(std::execution::par_unseq, first, last, init);
#elif defined(LIBSTATS_HAS_GCD)
        return detail::gcd_reduce(first, last, init, std::plus<T>{});
#else
        return std::accumulate(first, last, init);
#endif
    } else {
        return std::accumulate(first, last, init);
    }
}

/// Safe parallel for_each operation with Level 0-2 integration
template<typename Iterator, typename UnaryFunction>
void safe_for_each(Iterator first, Iterator last, UnaryFunction f) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::for_each(std::execution::par_unseq, first, last, f);
#elif defined(LIBSTATS_HAS_GCD)
        detail::gcd_for_each(first, last, f);
#else
        std::for_each(first, last, f);
#endif
    } else {
        std::for_each(first, last, f);
    }
}

/// Safe parallel sort operation with Level 0-2 integration
template<typename Iterator>
void safe_sort(Iterator first, Iterator last) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::sort(std::execution::par_unseq, first, last);
#else
        // No GCD implementation for sort, use serial
        std::sort(first, last);
#endif
    } else {
        std::sort(first, last);
    }
}

/// Safe parallel sort operation with custom comparator and Level 0-2 integration
template<typename Iterator, typename Compare>
void safe_sort(Iterator first, Iterator last, Compare comp) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::sort(std::execution::par_unseq, first, last, comp);
#else
        // No GCD implementation for sort, use serial
        std::sort(first, last, comp);
#endif
    } else {
        std::sort(first, last, comp);
    }
}

/// Safe parallel partial sort operation with Level 0-2 integration
template<typename Iterator>
void safe_partial_sort(Iterator first, Iterator middle, Iterator last) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::partial_sort(std::execution::par_unseq, first, middle, last);
#else
        // No GCD implementation for partial_sort, use serial
        std::partial_sort(first, middle, last);
#endif
    } else {
        std::partial_sort(first, middle, last);
    }
}

/// Safe parallel inclusive scan operation with Level 0-2 integration
template<typename Iterator1, typename Iterator2>
void safe_inclusive_scan(Iterator1 first, Iterator1 last, Iterator2 result) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::inclusive_scan(std::execution::par_unseq, first, last, result);
#else
        // No GCD implementation for inclusive_scan, use serial
        std::inclusive_scan(first, last, result);
#endif
    } else {
        std::inclusive_scan(first, last, result);
    }
}

/// Safe parallel exclusive scan operation with Level 0-2 integration
template<typename Iterator1, typename Iterator2, typename T>
void safe_exclusive_scan(Iterator1 first, Iterator1 last, Iterator2 result, T init) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        std::exclusive_scan(std::execution::par_unseq, first, last, result, init);
#else
        // No GCD implementation for exclusive_scan, use serial
        std::exclusive_scan(first, last, result, init);
#endif
    } else {
        std::exclusive_scan(first, last, result, init);
    }
}

/// Safe parallel find operation with Level 0-2 integration
template<typename Iterator, typename T>
Iterator safe_find(Iterator first, Iterator last, const T& value) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        return std::find(std::execution::par_unseq, first, last, value);
#else
        // No GCD implementation for find, use serial
        return std::find(first, last, value);
#endif
    } else {
        return std::find(first, last, value);
    }
}

/// Safe parallel find_if operation with Level 0-2 integration
template<typename Iterator, typename UnaryPredicate>
Iterator safe_find_if(Iterator first, Iterator last, UnaryPredicate pred) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        return std::find_if(std::execution::par_unseq, first, last, pred);
#else
        // No GCD implementation for find_if, use serial
        return std::find_if(first, last, pred);
#endif
    } else {
        return std::find_if(first, last, pred);
    }
}

/// Safe parallel count operation with Level 0-2 integration
template<typename Iterator, typename T>
typename std::iterator_traits<Iterator>::difference_type 
safe_count(Iterator first, Iterator last, const T& value) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        return std::count(std::execution::par_unseq, first, last, value);
#elif defined(LIBSTATS_HAS_GCD)
        return detail::gcd_count(first, last, value);
#else
        return std::count(first, last, value);
#endif
    } else {
        return std::count(first, last, value);
    }
}

/// Safe parallel count_if operation with Level 0-2 integration
template<typename Iterator, typename UnaryPredicate>
typename std::iterator_traits<Iterator>::difference_type 
safe_count_if(Iterator first, Iterator last, UnaryPredicate pred) {
    const auto count = std::distance(first, last);
    safety::check_finite(static_cast<double>(count), "element count");
    
    if (should_use_parallel(static_cast<std::size_t>(count))) {
#if defined(LIBSTATS_HAS_STD_EXECUTION)
        return std::count_if(std::execution::par_unseq, first, last, pred);
#elif defined(LIBSTATS_HAS_GCD)
        return detail::gcd_count_if(first, last, pred);
#else
        return std::count_if(first, last, pred);
#endif
    } else {
        return std::count_if(first, last, pred);
    }
}

} // namespace parallel
} // namespace libstats

#endif // LIBSTATS_PARALLEL_EXECUTION_H_

