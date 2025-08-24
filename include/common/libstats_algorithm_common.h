#pragma once

/**
 * @file common/libstats_algorithm_common.h
 * @brief Consolidated algorithm header - Phase 2 STL optimization
 *
 * This header consolidates algorithm usage across the library, reducing redundant
 * includes of <algorithm> which is used in 12% of headers (6 headers).
 *
 * Benefits:
 *   - Reduces algorithm template instantiation overhead
 *   - Provides optimized statistical algorithms
 *   - Centralized parallel algorithm dispatch
 *   - SIMD-aware algorithm implementations
 */

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <numeric>

// Conditionally include execution for parallel algorithms
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
    #include <execution>
    #define LIBSTATS_HAS_EXECUTION_POLICY 1
#else
    #define LIBSTATS_HAS_EXECUTION_POLICY 0
#endif

namespace stats {
namespace common {

/// Algorithm utilities optimized for statistical operations
namespace algorithm_utils {

/// Statistical min/max with NaN handling
template <typename Iterator>
inline auto safe_min_element(Iterator first, Iterator last) -> Iterator {
    return std::min_element(first, last, [](const auto& a, const auto& b) {
        // Handle NaN: NaN is considered greater than any finite value
        if (std::isnan(a) && std::isnan(b))
            return false;  // NaN == NaN for min purposes
        if (std::isnan(a))
            return false;  // NaN is not minimum
        if (std::isnan(b))
            return true;  // b is NaN, a is minimum
        return a < b;
    });
}

template <typename Iterator>
inline auto safe_max_element(Iterator first, Iterator last) -> Iterator {
    return std::max_element(first, last, [](const auto& a, const auto& b) {
        // Handle NaN: NaN is considered greater than any finite value
        if (std::isnan(a) && std::isnan(b))
            return false;  // NaN == NaN for max purposes
        if (std::isnan(a))
            return true;  // NaN is maximum
        if (std::isnan(b))
            return false;  // b is NaN, a is not maximum
        return a < b;
    });
}

/// Count finite (non-NaN, non-infinite) values
template <typename Iterator>
inline std::size_t count_finite(Iterator first, Iterator last) {
    return std::count_if(first, last, [](const auto& value) { return std::isfinite(value); });
}

/// Statistical accumulation with NaN filtering
template <typename Iterator, typename T, typename BinaryOperation>
inline T safe_accumulate(Iterator first, Iterator last, T init, BinaryOperation op) {
    return std::accumulate(first, last, init, [&op](const T& acc, const auto& value) {
        if (std::isfinite(value)) {
            return op(acc, value);
        }
        return acc;  // Skip non-finite values
    });
}

/// Optimized sum for statistical data (handles NaN/infinity)
template <typename Iterator>
inline auto safe_sum(Iterator first, Iterator last) {
    using ValueType = typename std::iterator_traits<Iterator>::value_type;
    return safe_accumulate(first, last, ValueType{0}, std::plus<ValueType>{});
}

/// Statistical sorting that handles NaN values
template <typename Iterator>
inline void statistical_sort(Iterator first, Iterator last) {
    // Sort with NaN values pushed to the end
    std::sort(first, last, [](const auto& a, const auto& b) {
        if (std::isnan(a) && std::isnan(b))
            return false;  // NaN == NaN
        if (std::isnan(a))
            return false;  // NaN goes to end
        if (std::isnan(b))
            return true;  // NaN goes to end
        return a < b;
    });
}

/// Partial sort for quantile calculations
template <typename Iterator>
inline void statistical_partial_sort(Iterator first, Iterator nth, Iterator last) {
    std::nth_element(first, nth, last, [](const auto& a, const auto& b) {
        if (std::isnan(a) && std::isnan(b))
            return false;
        if (std::isnan(a))
            return false;
        if (std::isnan(b))
            return true;
        return a < b;
    });
}

/// Parallel algorithms with automatic fallback
namespace arch {

/// Parallel transform with automatic policy selection
template <typename InputIt, typename OutputIt, typename UnaryOperation>
inline OutputIt transform(InputIt first, InputIt last, OutputIt d_first, UnaryOperation op) {
    const auto size = std::distance(first, last);

    // Use parallel execution for large datasets
    if (size > 1000) {
#if LIBSTATS_HAS_EXECUTION_POLICY
        try {
            return std::transform(std::execution::par_unseq, first, last, d_first, op);
        } catch (...) {
            // Fallback to sequential if parallel execution fails
            return std::transform(first, last, d_first, op);
        }
#else
        return std::transform(first, last, d_first, op);
#endif
    } else {
        return std::transform(first, last, d_first, op);
    }
}

/// Parallel for_each with automatic policy selection
template <typename InputIt, typename UnaryFunction>
inline void for_each(InputIt first, InputIt last, UnaryFunction f) {
    const auto size = std::distance(first, last);

    if (size > 1000) {
#if LIBSTATS_HAS_EXECUTION_POLICY
        try {
            std::for_each(std::execution::par_unseq, first, last, f);
        } catch (...) {
            std::for_each(first, last, f);
        }
#else
        std::for_each(first, last, f);
#endif
    } else {
        std::for_each(first, last, f);
    }
}

/// Parallel reduce with automatic policy selection
template <typename InputIt, typename T, typename BinaryOperation>
inline T reduce(InputIt first, InputIt last, T init, BinaryOperation op) {
    const auto size = std::distance(first, last);

    if (size > 1000) {
#if LIBSTATS_HAS_EXECUTION_POLICY
        try {
            return std::reduce(std::execution::par_unseq, first, last, init, op);
        } catch (...) {
            return std::accumulate(first, last, init, op);
        }
#else
        return std::accumulate(first, last, init, op);
#endif
    } else {
        return std::accumulate(first, last, init, op);
    }
}

/// Parallel sort with automatic policy selection
template <typename Iterator, typename Compare>
inline void sort(Iterator first, Iterator last, Compare comp) {
    const auto size = std::distance(first, last);

    if (size > 1000) {
#if LIBSTATS_HAS_EXECUTION_POLICY
        try {
            std::sort(std::execution::par_unseq, first, last, comp);
        } catch (...) {
            std::sort(first, last, comp);
        }
#else
        std::sort(first, last, comp);
#endif
    } else {
        std::sort(first, last, comp);
    }
}

/// Parallel sort with default comparison
template <typename Iterator>
inline void sort(Iterator first, Iterator last) {
    sort(first, last, std::less<typename std::iterator_traits<Iterator>::value_type>{});
}
}  // namespace arch

/// SIMD-friendly algorithms
namespace simd_friendly {

/// Check if range size is SIMD-friendly
template <typename Iterator>
inline bool is_simd_size(Iterator first, Iterator last, std::size_t simd_width = 8) {
    const auto size = std::distance(first, last);
    return size >= simd_width && (size % simd_width == 0);
}

/// Transform with SIMD alignment consideration
template <typename InputIt, typename OutputIt, typename UnaryOperation>
inline OutputIt simd_transform(InputIt first, InputIt last, OutputIt d_first, UnaryOperation op) {
    // For SIMD-friendly sizes, process in chunks
    constexpr std::size_t SIMD_WIDTH = 8;
    const auto total_size = std::distance(first, last);

    if (total_size >= SIMD_WIDTH) {
        // Process SIMD-aligned portion
        const auto simd_count = (total_size / SIMD_WIDTH) * SIMD_WIDTH;
        auto simd_end = std::next(first, simd_count);

        // Use parallel transform for the main portion
        auto result = arch::transform(first, simd_end, d_first, op);

        // Process remaining elements
        return std::transform(simd_end, last, result, op);
    } else {
        return std::transform(first, last, d_first, op);
    }
}

/// Accumulate with SIMD consideration
template <typename Iterator, typename T, typename BinaryOperation>
inline T simd_accumulate(Iterator first, Iterator last, T init, BinaryOperation op) {
    constexpr std::size_t SIMD_WIDTH = 8;
    const auto total_size = std::distance(first, last);

    if (total_size >= SIMD_WIDTH) {
        // Use parallel reduce for SIMD-friendly portion
        return arch::reduce(first, last, init, op);
    } else {
        return std::accumulate(first, last, init, op);
    }
}
}
}  // namespace arch::simd_friendly

/// Statistical algorithm helpers
namespace statistical {

/// Find outliers using IQR method
template <typename Iterator, typename OutputIterator>
inline OutputIterator find_outliers_iqr(Iterator first, Iterator last, OutputIterator out,
                                        double multiplier = 1.5) {
    std::vector<typename std::iterator_traits<Iterator>::value_type> sorted_data(first, last);
    statistical_sort(sorted_data.begin(), sorted_data.end());

    const auto size = sorted_data.size();
    if (size < 4)
        return out;  // Need at least 4 points for IQR

    // Calculate quartiles
    const auto q1_idx = size / 4;
    const auto q3_idx = 3 * size / 4;
    const auto q1 = sorted_data[q1_idx];
    const auto q3 = sorted_data[q3_idx];
    const auto iqr = q3 - q1;

    // Define outlier bounds
    const auto lower_bound = q1 - multiplier * iqr;
    const auto upper_bound = q3 + multiplier * iqr;

    // Find outliers in original data
    return std::copy_if(first, last, out, [=](const auto& value) {
        return !std::isfinite(value) || value < lower_bound || value > upper_bound;
    });
}

/// Remove outliers and return cleaned data
template <typename Container>
inline Container remove_outliers_iqr(const Container& data, double multiplier = 1.5) {
    Container result;
    result.reserve(data.size());

    // Find outliers
    std::vector<typename Container::value_type> outliers;
    find_outliers_iqr(data.begin(), data.end(), std::back_inserter(outliers), multiplier);

    // Copy non-outliers
    std::copy_if(data.begin(), data.end(), std::back_inserter(result),
                 [&outliers](const auto& value) {
                     return std::find(outliers.begin(), outliers.end(), value) == outliers.end();
                 });

    return result;
}

/// Calculate percentile (0-100)
template <typename Iterator>
inline auto percentile(Iterator first, Iterator last, double p) {
    using ValueType = typename std::iterator_traits<Iterator>::value_type;

    if (first == last)
        return ValueType{};

    std::vector<ValueType> sorted_data(first, last);
    statistical_sort(sorted_data.begin(), sorted_data.end());

    // Remove NaN values
    auto finite_end = std::remove_if(sorted_data.begin(), sorted_data.end(),
                                     [](const auto& v) { return !std::isfinite(v); });

    const auto size = std::distance(sorted_data.begin(), finite_end);
    if (size == 0)
        return ValueType{};

    const double index = (p / 100.0) * (size - 1);
    const auto lower_idx = static_cast<std::size_t>(std::floor(index));
    const auto upper_idx = static_cast<std::size_t>(std::ceil(index));

    if (lower_idx == upper_idx) {
        return sorted_data[lower_idx];
    } else {
        const double weight = index - lower_idx;
        return sorted_data[lower_idx] * (1.0 - weight) + sorted_data[upper_idx] * weight;
    }
}
}  // namespace statistical
}  // namespace algorithm_utils

}  // namespace common
}  // namespace stats
