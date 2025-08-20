#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <new>
#include <type_traits>
#include <vector>
#ifdef _WIN32
    #include <malloc.h>
#endif

namespace libstats {

// =============================================================================
// MEMORY POOL IMPLEMENTATION
// =============================================================================

/**
 * @brief Memory pool for efficient allocation of temporary objects
 */
class MemoryPool {
   private:
    static constexpr size_t POOL_SIZE = 1024 * 1024;  // 1MB pool
    static constexpr size_t ALIGNMENT = 64;           // Cache line alignment

    alignas(ALIGNMENT) std::byte pool_[POOL_SIZE];
    std::atomic<size_t> offset_{0};
    mutable std::mutex pool_mutex_;

   public:
    /**
     * @brief Allocate aligned memory from pool
     * @param size Size in bytes
     * @param alignment Required alignment (default: 64 bytes for SIMD)
     * @return Pointer to allocated memory or nullptr if insufficient space
     */
    void* allocate(size_t size, size_t alignment = ALIGNMENT) noexcept {
        // Ensure size is aligned
        size = (size + alignment - 1) & ~(alignment - 1);

        size_t current_offset = offset_.load(std::memory_order_acquire);
        size_t new_offset = current_offset + size;

        if (new_offset > POOL_SIZE) {
            return nullptr;  // Pool exhausted
        }

        // Try to atomically update offset
        while (!offset_.compare_exchange_weak(current_offset, new_offset, std::memory_order_release,
                                              std::memory_order_acquire)) {
            new_offset = current_offset + size;
            if (new_offset > POOL_SIZE) {
                return nullptr;
            }
        }

        return &pool_[current_offset];
    }

    /**
     * @brief Reset pool (not thread-safe, use with caution)
     */
    void reset() noexcept {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        offset_.store(0, std::memory_order_release);
    }

    /**
     * @brief Get current pool usage
     * @return Bytes used in pool
     */
    size_t getUsage() const noexcept { return offset_.load(std::memory_order_acquire); }

    /**
     * @brief Check if pool has sufficient space
     * @param size Required size
     * @return true if space available
     */
    bool hasSpace(size_t size) const noexcept {
        return offset_.load(std::memory_order_acquire) + size <= POOL_SIZE;
    }
};

// =============================================================================
// SIMD-ALIGNED ALLOCATOR
// =============================================================================

// Platform-agnostic aligned allocation helpers
inline void* libstats_aligned_alloc(std::size_t alignment, std::size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#elif defined(__unix__) || defined(__APPLE__)
    return std::aligned_alloc(alignment, size);
#else
    #error "No aligned_alloc implementation for this platform."
#endif
}

inline void libstats_aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#elif defined(__unix__) || defined(__APPLE__)
    free(ptr);
#else
    #error "No aligned_free implementation for this platform."
#endif
}

/**
 * @brief SIMD-aligned vector allocator
 */
template <typename T>
class SIMDAllocator {
   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;

    static constexpr size_t SIMD_ALIGNMENT = 64;  // 64-byte alignment for AVX-512

    template <typename U>
    struct rebind {
        using other = SIMDAllocator<U>;
    };

    SIMDAllocator() noexcept = default;

    template <typename U>
    SIMDAllocator(const SIMDAllocator<U>&) noexcept {}

    pointer allocate(size_type n) {
        if (n > std::numeric_limits<size_type>::max() / sizeof(T)) {
            throw std::bad_alloc();
        }

        size_t size = n * sizeof(T);
        void* ptr = libstats_aligned_alloc(SIMD_ALIGNMENT, size);
        if (!ptr) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept { libstats_aligned_free(p); }

    template <typename U>
    bool operator==(const SIMDAllocator<U>&) const noexcept {
        return true;
    }

    template <typename U>
    bool operator!=(const SIMDAllocator<U>&) const noexcept {
        return false;
    }
};

// =============================================================================
// SMALL VECTOR OPTIMIZATION
// =============================================================================

/**
 * @brief Small vector optimization for temporary data
 * @tparam T Element type
 * @tparam N Number of elements in small buffer optimization
 */
template <typename T, size_t N>
class SmallVector {
   private:
    alignas(T) char small_buffer_[N * sizeof(T)];
    std::unique_ptr<T[]> heap_data_;
    size_t size_{0};
    size_t capacity_{N};

    T* data() noexcept {
        return capacity_ == N ? reinterpret_cast<T*>(small_buffer_) : heap_data_.get();
    }

    const T* data() const noexcept {
        return capacity_ == N ? reinterpret_cast<const T*>(small_buffer_) : heap_data_.get();
    }

   public:
    SmallVector() = default;

    explicit SmallVector(size_t count) { resize(count); }

    SmallVector(size_t count, const T& value) { resize(count, value); }

    template <typename InputIt>
    SmallVector(InputIt first, InputIt last) {
        assign(first, last);
    }

    SmallVector(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

    ~SmallVector() { clear(); }

    SmallVector(const SmallVector& other) { assign(other.begin(), other.end()); }

    SmallVector(SmallVector&& other) noexcept {
        if (other.capacity_ == N) {
            // Other uses small buffer, must copy
            assign(other.begin(), other.end());
        } else {
            // Other uses heap, can move
            heap_data_ = std::move(other.heap_data_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.size_ = 0;
            other.capacity_ = N;
        }
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this != &other) {
            assign(other.begin(), other.end());
        }
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept {
        if (this != &other) {
            clear();
            if (other.capacity_ == N) {
                // Other uses small buffer, must copy
                assign(other.begin(), other.end());
            } else {
                // Other uses heap, can move
                heap_data_ = std::move(other.heap_data_);
                size_ = other.size_;
                capacity_ = other.capacity_;
                other.size_ = 0;
                other.capacity_ = N;
            }
        }
        return *this;
    }

    void push_back(const T& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ * 2);
        }
        new (data() + size_) T(value);
        ++size_;
    }

    void push_back(T&& value) {
        if (size_ >= capacity_) {
            reserve(capacity_ * 2);
        }
        new (data() + size_) T(std::move(value));
        ++size_;
    }

    template <typename... Args>
    void emplace_back(Args&&... args) {
        if (size_ >= capacity_) {
            reserve(capacity_ * 2);
        }
        new (data() + size_) T(std::forward<Args>(args)...);
        ++size_;
    }

    void pop_back() {
        if (size_ > 0) {
            --size_;
            data()[size_].~T();
        }
    }

    void resize(size_t new_size) {
        if (new_size > size_) {
            reserve(new_size);
            for (size_t i = size_; i < new_size; ++i) {
                new (data() + i) T();
            }
        } else if (new_size < size_) {
            for (size_t i = new_size; i < size_; ++i) {
                data()[i].~T();
            }
        }
        size_ = new_size;
    }

    void resize(size_t new_size, const T& value) {
        if (new_size > size_) {
            reserve(new_size);
            for (size_t i = size_; i < new_size; ++i) {
                new (data() + i) T(value);
            }
        } else if (new_size < size_) {
            for (size_t i = new_size; i < size_; ++i) {
                data()[i].~T();
            }
        }
        size_ = new_size;
    }

    void reserve(size_t new_capacity) {
        if (new_capacity <= capacity_) {
            return;
        }

        auto new_data = std::make_unique<T[]>(new_capacity);

        for (size_t i = 0; i < size_; ++i) {
            new (new_data.get() + i) T(std::move(data()[i]));
            data()[i].~T();
        }

        heap_data_ = std::move(new_data);
        capacity_ = new_capacity;
    }

    template <typename InputIt>
    void assign(InputIt first, InputIt last) {
        clear();
        for (auto it = first; it != last; ++it) {
            push_back(*it);
        }
    }

    void clear() {
        for (size_t i = 0; i < size_; ++i) {
            data()[i].~T();
        }
        size_ = 0;
    }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    T& operator[](size_t index) noexcept { return data()[index]; }
    const T& operator[](size_t index) const noexcept { return data()[index]; }

    T& at(size_t index) {
        if (index >= size_) {
            throw std::out_of_range("SmallVector index out of range");
        }
        return data()[index];
    }

    const T& at(size_t index) const {
        if (index >= size_) {
            throw std::out_of_range("SmallVector index out of range");
        }
        return data()[index];
    }

    T& front() noexcept { return data()[0]; }
    const T& front() const noexcept { return data()[0]; }

    T& back() noexcept { return data()[size_ - 1]; }
    const T& back() const noexcept { return data()[size_ - 1]; }

    T* begin() noexcept { return data(); }
    T* end() noexcept { return data() + size_; }
    const T* begin() const noexcept { return data(); }
    const T* end() const noexcept { return data() + size_; }
    const T* cbegin() const noexcept { return data(); }
    const T* cend() const noexcept { return data() + size_; }
};

// =============================================================================
// STACK ALLOCATOR
// =============================================================================

/**
 * @brief Stack-based memory allocator for temporary computations
 */
template <size_t StackSize = 4096>
class StackAllocator {
   private:
    alignas(std::max_align_t) char stack_[StackSize];
    char* current_ = stack_;

   public:
    template <typename T>
    T* allocate(size_t count) {
        size_t size = count * sizeof(T);
        size_t alignment = alignof(T);

        // Align current pointer
        uintptr_t ptr = reinterpret_cast<uintptr_t>(current_);
        ptr = (ptr + alignment - 1) & ~(alignment - 1);
        current_ = reinterpret_cast<char*>(ptr);

        if (current_ + size > stack_ + StackSize) {
            throw std::bad_alloc();
        }

        T* result = reinterpret_cast<T*>(current_);
        current_ += size;
        return result;
    }

    void reset() noexcept { current_ = stack_; }

    size_t used() const noexcept { return current_ - stack_; }

    size_t available() const noexcept { return StackSize - used(); }
};

// =============================================================================
// TYPE ALIASES FOR CONVENIENCE
// =============================================================================

/**
 * @brief SIMD-aligned vector type for efficient batch operations
 */
template <typename T>
using simd_vector = std::vector<T, SIMDAllocator<T>>;

/**
 * @brief Thread-local memory pool instance
 */
extern thread_local MemoryPool thread_pool_;

/**
 * @brief Get thread-local memory pool
 * @return Reference to thread-local memory pool
 */
inline MemoryPool& getThreadPool() noexcept {
    return thread_pool_;
}

}  // namespace libstats
