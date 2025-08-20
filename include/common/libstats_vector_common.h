#pragma once

/**
 * @file common/libstats_vector_common.h
 * @brief Consolidated vector header - Phase 2 STL optimization
 *
 * This header consolidates vector usage across the library, reducing redundant
 * includes of <vector> which is used in 20% of headers (10 headers).
 *
 * Benefits:
 *   - Reduces template instantiation overhead
 *   - Provides common vector type aliases
 *   - Enables optimized vector operations
 *   - Centralized memory allocation strategies
 */

#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

namespace libstats {
namespace common {

/// Common vector type aliases for statistical data
using DoubleVector = std::vector<double>;
using FloatVector = std::vector<float>;
using IntVector = std::vector<int>;
using SizeVector = std::vector<std::size_t>;
using BoolVector = std::vector<bool>;

/// Aligned vector types for SIMD optimization
template <typename T, std::size_t Alignment = 32>
class AlignedVector {
   private:
    std::vector<T> data_;

   public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

    AlignedVector() = default;

    explicit AlignedVector(size_type size) : data_(size) {
        // Ensure proper alignment for SIMD operations
        static_assert(Alignment >= alignof(T), "Alignment must be at least alignof(T)");
    }

    AlignedVector(size_type size, const T& value) : data_(size, value) {}

    // Standard vector interface
    T& operator[](size_type pos) { return data_[pos]; }
    const T& operator[](size_type pos) const { return data_[pos]; }

    T& at(size_type pos) { return data_.at(pos); }
    const T& at(size_type pos) const { return data_.at(pos); }

    iterator begin() { return data_.begin(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator cbegin() const { return data_.cbegin(); }

    iterator end() { return data_.end(); }
    const_iterator end() const { return data_.end(); }
    const_iterator cend() const { return data_.cend(); }

    bool empty() const { return data_.empty(); }
    size_type size() const { return data_.size(); }
    size_type capacity() const { return data_.capacity(); }

    void reserve(size_type new_cap) { data_.reserve(new_cap); }
    void resize(size_type count) { data_.resize(count); }
    void resize(size_type count, const T& value) { data_.resize(count, value); }

    void push_back(const T& value) { data_.push_back(value); }
    void push_back(T&& value) { data_.push_back(std::move(value)); }

    template <class... Args>
    void emplace_back(Args&&... args) {
        data_.emplace_back(std::forward<Args>(args)...);
    }

    void pop_back() { data_.pop_back(); }
    void clear() { data_.clear(); }

    T* data() { return data_.data(); }
    const T* data() const { return data_.data(); }
};

/// Commonly used aligned vector types
using AlignedDoubleVector = AlignedVector<double, 32>;
using AlignedFloatVector = AlignedVector<float, 32>;

/// Vector utility functions
namespace vector_utils {

/// Reserve capacity for performance-critical vectors
template <typename T>
inline void reserve_performance(std::vector<T>& vec, std::size_t expected_size) {
    // Reserve with some padding to avoid frequent reallocations
    vec.reserve(expected_size + expected_size / 4);  // 25% padding
}

/// Pre-allocate vector with optimal size for statistical operations
template <typename T>
inline std::vector<T> create_statistical_vector(std::size_t size) {
    std::vector<T> vec;
    reserve_performance(vec, size);
    vec.resize(size);
    return vec;
}

/// Create vector with SIMD-friendly size alignment
template <typename T>
inline std::vector<T> create_simd_aligned_vector(std::size_t size, std::size_t simd_width = 8) {
    // Round up to nearest SIMD-width boundary
    std::size_t aligned_size = ((size + simd_width - 1) / simd_width) * simd_width;
    return create_statistical_vector<T>(aligned_size);
}

/// Check if vector size is SIMD-friendly
inline bool is_simd_friendly_size(std::size_t size, std::size_t simd_width = 8) {
    return size % simd_width == 0 && size >= simd_width;
}

/// Get optimal chunk size for parallel vector operations
inline std::size_t get_optimal_chunk_size(std::size_t total_size, std::size_t num_threads = 0) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }

    // Aim for chunks that are:
    // 1. Large enough to amortize thread overhead
    // 2. Small enough for good load balancing
    // 3. SIMD-aligned when possible

    constexpr std::size_t min_chunk_size = 1024;  // Minimum chunk for thread overhead
    std::size_t base_chunk = std::max(total_size / (num_threads * 4), min_chunk_size);

    // Align to cache line boundary (8 doubles = 64 bytes)
    constexpr std::size_t cache_line_elements = 8;
    return ((base_chunk + cache_line_elements - 1) / cache_line_elements) * cache_line_elements;
}
}  // namespace vector_utils

}  // namespace common
}  // namespace libstats
