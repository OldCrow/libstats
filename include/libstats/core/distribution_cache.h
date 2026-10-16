#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <thread>

namespace stats {

/**
 * @brief The atomic cache-validity flag, plus a write generation.
 *
 * Every store advances generation(). Each write to a distribution's
 * parameters stores to this flag inside the unique-lock section on
 * cache_mutex_ that performs the write: setters, trySet*, fit, reset and the
 * assignments store false, and a copy assignment that rebuilds its cache
 * stores true. ThreadSafeCacheManager::readConsistent relies on that.
 * A store outside such a section (a reader rebuilding the cache) only costs a
 * reader one retry.
 *
 * The interface is the subset of std::atomic<bool> the distributions use, so
 * the flag replaced that type without touching them.
 */
class CacheValidFlag {
   public:
    [[nodiscard]] bool load(std::memory_order order = std::memory_order_seq_cst) const noexcept {
        return valid_.load(order);
    }

    void store(bool valid, std::memory_order order = std::memory_order_seq_cst) noexcept {
        generation_.fetch_add(1, std::memory_order_relaxed);
        valid_.store(valid, order);
    }

    /// Stores so far. Read it under cache_mutex_, which orders it against the writes.
    [[nodiscard]] std::uint64_t generation() const noexcept {
        return generation_.load(std::memory_order_relaxed);
    }

   private:
    std::atomic<bool> valid_{false};
    std::atomic<std::uint64_t> generation_{0};
};

// =============================================================================
// THREAD-SAFE CACHE MANAGEMENT INTERFACE
// =============================================================================

/**
 * @brief Thread-safe cache management base class for distributions
 *
 * Provides the infrastructure for thread-safe caching in distribution classes
 * using the distribution cache adapter pattern.
 *
 * ### Dual-flag pattern (cache_valid_ + cacheValidAtomic_)
 * Two flags represent the same state deliberately:
 * - cacheValidAtomic_ enables a contention-reduced fast path in
 *   getCachedValue(): the atomic load (acquire) avoids a lock entirely when
 *   the cache is stale (proceeding directly to the unique-lock slow path);
 *   when valid it still acquires shared_lock before reading values.  This is
 *   NOT a fully lock-free read — it is a contention-reduced path that avoids
 *   the exclusive (write) lock on the hot path.
 * - cache_valid_ is the plain bool read under shared_lock; it serves as the
 *   ground-truth flag for code paths that already hold cache_mutex_.
 *
 * ### Per-parameter atomic fast paths (derived classes)
 * Some derived classes (e.g. GaussianDistribution) keep individual parameter
 * atomics (atomicMean_, atomicStdDev_) behind get*Atomic() getters, which only
 * tests call; getMean()/getStdDev() take the shared lock. Each atomic value is
 * individually consistent, but reading two atomics across separate calls does
 * not form an atomic pair: a concurrent setParameters() between the two reads
 * yields a torn snapshot. Callers that need a consistent pair of parameters
 * should use the snapshot pattern (withCacheSnapshot, or readConsistent for a
 * value built from several locked calls).
 *
 * v2.0.0 removed the old derived-class cacheValidAtomic_ shadows; this base
 * member is now the single atomic cache-validity flag for all distributions.
 * The plain bool remains for code paths that already hold cache_mutex_.
 */
class ThreadSafeCacheManager {
   protected:
    /**
     * @brief Thread-safe cache management infrastructure
     */
    mutable std::shared_mutex cache_mutex_;
    mutable bool cache_valid_{false};

    /**
     * @brief Atomic cache validity flag for lock-free fast paths
     * @see Class-level documentation for the dual-flag pattern rationale.
     */
    mutable CacheValidFlag cacheValidAtomic_;

   public:
    /**
     * @brief Virtual destructor for proper polymorphic cleanup
     */
    virtual ~ThreadSafeCacheManager() = default;

    /**
     * @brief Update cached statistical properties (must be overridden)
     * @note Called under unique lock; implementation should set cache_valid_ = true
     */
    virtual void updateCacheUnsafe() const = 0;

    /**
     * @brief Invalidate cache when parameters change
     * @note Thread-safe; call whenever parameters are modified
     */
    void invalidateCache() noexcept {
        std::unique_lock lock(cache_mutex_);
        cache_valid_ = false;
        cacheValidAtomic_.store(false, std::memory_order_release);
    }

    /**
     * @brief Thread-safe cached value access with double-checked locking
     * @param accessor Function to access cached value
     * @return Cached value
     * @note Implementation in src/distribution_cache.cpp with explicit instantiations
     */
    template <typename Func>
    auto getCachedValue(Func&& accessor) const -> decltype(accessor());

    /**
     * @brief Acquire the correct lock, ensure the cache is valid, then call
     *        @p snapFn to copy cached members into the caller's local variables.
     *
     * Encapsulates the double-checked locking pattern repeated in the
     * VECTORIZED / PARALLEL / WORK_STEALING batch lambdas of every distribution:
     * @code
     *   // Before
     *   double a, b;
     *   { shared_lock ...; if (!valid) { unlock; unique_lock...; if (!valid) update; a=a_; b=b_; }
     * else { a=a_; b=b_; } }
     *   // After
     *   double a, b;
     *   dist.withCacheSnapshot([&]{ a = dist.a_; b = dist.b_; });
     * @endcode
     *
     * The snapshot lambda is called while an appropriate lock is held —
     * shared_lock when the cache was already valid, unique_lock otherwise —
     * so there is no TOCTOU gap between the validity check and the reads.
     *
     * @tparam SnapshotFn  Callable void() that assigns from this object's
     *                     mutable cached members into caller-owned variables.
     */
    template <typename SnapshotFn>
    void withCacheSnapshot(SnapshotFn&& snapFn) const {
        {
            std::shared_lock<std::shared_mutex> lock(cache_mutex_);
            if (cache_valid_) {
                std::forward<SnapshotFn>(snapFn)();
                return;
            }
        }  // release shared_lock before acquiring exclusive
        std::unique_lock<std::shared_mutex> ulock(cache_mutex_);
        if (!cache_valid_) {
            updateCacheUnsafe();
        }
        std::forward<SnapshotFn>(snapFn)();
    }

    /**
     * @brief Evaluate @p fn, a run of locking scalar calls on this object, so
     *        that all of them see one parameter state.
     *
     * Seqlock-style: read the generation under a shared lock, call @p fn,
     * read it again under a shared lock, and retry if it moved. Neither read
     * is held across @p fn, so the scalar calls inside it lock as usual (a
     * second shared lock in one thread would be undefined behaviour).
     *
     * Why an equal generation proves one state: a parameter write and its
     * store to cacheValidAtomic_ share one unique-lock section W on
     * cache_mutex_. Mutex order puts W before the first generation read, after
     * the second, or between them. Before: every scalar call starts after W's
     * unlock, so it sees W, whether it locks this mutex or only a delegate's
     * (Bernoulli, Geometric, ChiSquared, Erlang forward to a delegate that W
     * writes while holding this mutex, and whose own lock orders the read).
     * After: every scalar call ended before W's lock. Between: W's store
     * happens-before the second read, and the first read happens-before it,
     * so by coherence the second read sees a larger count and @p fn reruns.
     * Relaxed loads suffice; the mutex supplies the ordering.
     *
     * Each retry needs a parameter write to commit during the previous
     * attempt, so @p fn returns once writes pause for one evaluation. No
     * bounded fallback exists here without locking recursively, so the loop
     * yields rather than returning a value from no single state.
     *
     * @warning Never call this while holding cache_mutex_.
     */
    template <typename Fn>
    auto readConsistent(Fn&& fn) const -> decltype(fn()) {
        for (unsigned attempt = 0;; ++attempt) {
            const std::uint64_t before = lockedGeneration();
            auto result = fn();
            if (lockedGeneration() == before)
                return result;
            if (attempt >= kSpinAttempts)
                std::this_thread::yield();
        }
    }

   private:
    static constexpr unsigned kSpinAttempts = 4;

    [[nodiscard]] std::uint64_t lockedGeneration() const {
        std::shared_lock<std::shared_mutex> lock(cache_mutex_);
        return cacheValidAtomic_.generation();
    }
};

}  // namespace stats
