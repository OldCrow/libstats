#pragma once

#include "../common/platform_common.h"
#include "adaptive_cache.h"
#include <cmath>
#include <string>
#include <optional>
#include <atomic>
#include <mutex>
#include <chrono>
#include <bit>

namespace libstats {
namespace cache {

/**
 * @brief High-performance cache keys for mathematical functions
 */

// Single-argument function key (gamma, erf, log, etc.)
struct SingleArgKey {
    int64_t rounded_value;
    
    static SingleArgKey create(double x, double precision) {
        // Convert to fixed-point integer for ultra-fast comparison
        return {static_cast<int64_t>(std::round(x / precision))};;
    }
    
    bool operator==(const SingleArgKey& other) const noexcept {
        return rounded_value == other.rounded_value;
    }
};

// Two-argument function key (beta, etc.)
struct TwoArgKey {
    int64_t rounded_arg1;
    int64_t rounded_arg2;
    
    static TwoArgKey create(double a, double b, double precision) {
        return {
            static_cast<int64_t>(std::round(a / precision)),
            static_cast<int64_t>(std::round(b / precision))
        };
    }
    
    bool operator==(const TwoArgKey& other) const noexcept {
        return rounded_arg1 == other.rounded_arg1 && 
               rounded_arg2 == other.rounded_arg2;
    }
};

} // namespace cache
} // namespace libstats

// Hash specializations for ultra-fast key hashing
template<>
struct std::hash<libstats::cache::SingleArgKey> {
    size_t operator()(const libstats::cache::SingleArgKey& k) const noexcept {
        return std::hash<int64_t>{}(k.rounded_value);
    }
};

template<>
struct std::hash<libstats::cache::TwoArgKey> {
    size_t operator()(const libstats::cache::TwoArgKey& k) const noexcept {
        // Fast hash combination
        size_t h1 = std::hash<int64_t>{}(k.rounded_arg1);
        size_t h2 = std::hash<int64_t>{}(k.rounded_arg2);
        return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
};

namespace libstats {
namespace cache {

/**
 * @brief Configuration for mathematical function caching with precision control
 */
struct MathFunctionCacheConfig {
    // Cache sizes for different function types
    size_t gamma_cache_size = 2048;           // Gamma function cache entries
    size_t erf_cache_size = 1024;             // Error function cache entries  
    size_t beta_cache_size = 1024;            // Beta function cache entries
    size_t log_cache_size = 512;              // Logarithm cache entries
    
    // Precision settings (smaller = higher precision, lower hit rate)
    double gamma_precision = 0.001;           // Gamma function rounding precision
    double erf_precision = 0.0001;            // Error function rounding precision
    double beta_precision = 0.001;            // Beta function rounding precision
    double log_precision = 0.0001;            // Log function rounding precision
    
    // Cache behavior
    std::chrono::milliseconds ttl{60000};     // 60 second TTL for math functions
    bool enable_statistics = true;            // Track cache statistics
    bool enable_background_cleanup = true;    // Enable background optimization
    
    // Performance tuning
    double target_hit_rate = 0.75;            // Target cache hit rate
    size_t memory_limit_mb = 16;              // Maximum memory usage in MB
};

/**
 * @brief Statistics for mathematical function cache performance
 */
struct MathFunctionCacheStats {
    // Per-function hit rates
    std::atomic<size_t> gamma_hits{0};
    std::atomic<size_t> gamma_misses{0};
    std::atomic<size_t> erf_hits{0};
    std::atomic<size_t> erf_misses{0};
    std::atomic<size_t> beta_hits{0};
    std::atomic<size_t> beta_misses{0};
    std::atomic<size_t> log_hits{0};
    std::atomic<size_t> log_misses{0};
    
    // Cache sizes
    std::atomic<size_t> gamma_cache_size{0};
    std::atomic<size_t> erf_cache_size{0};
    std::atomic<size_t> beta_cache_size{0};
    std::atomic<size_t> log_cache_size{0};
    
    // Performance metrics
    std::atomic<size_t> total_memory_bytes{0};
    std::atomic<double> average_lookup_time_ns{0.0};
    std::chrono::steady_clock::time_point created_at;
    
    MathFunctionCacheStats() : created_at(std::chrono::steady_clock::now()) {}
    
    // Custom copy constructor for atomic members
    MathFunctionCacheStats(const MathFunctionCacheStats& other) noexcept 
        : gamma_hits(other.gamma_hits.load())
        , gamma_misses(other.gamma_misses.load())
        , erf_hits(other.erf_hits.load())
        , erf_misses(other.erf_misses.load())
        , beta_hits(other.beta_hits.load())
        , beta_misses(other.beta_misses.load())
        , log_hits(other.log_hits.load())
        , log_misses(other.log_misses.load())
        , gamma_cache_size(other.gamma_cache_size.load())
        , erf_cache_size(other.erf_cache_size.load())
        , beta_cache_size(other.beta_cache_size.load())
        , log_cache_size(other.log_cache_size.load())
        , total_memory_bytes(other.total_memory_bytes.load())
        , average_lookup_time_ns(other.average_lookup_time_ns.load())
        , created_at(other.created_at) {}
    
    // Delete copy assignment operator since atomic types don't support it
    MathFunctionCacheStats& operator=(const MathFunctionCacheStats&) = delete;
    
    double getGammaHitRate() const noexcept {
        size_t total = gamma_hits.load() + gamma_misses.load();
        return total > 0 ? static_cast<double>(gamma_hits.load()) / static_cast<double>(total) : 0.0;
    }
    
    double getErfHitRate() const noexcept {
        size_t total = erf_hits.load() + erf_misses.load();
        return total > 0 ? static_cast<double>(erf_hits.load()) / static_cast<double>(total) : 0.0;
    }
    
    double getBetaHitRate() const noexcept {
        size_t total = beta_hits.load() + beta_misses.load();
        return total > 0 ? static_cast<double>(beta_hits.load()) / static_cast<double>(total) : 0.0;
    }
    
    double getLogHitRate() const noexcept {
        size_t total = log_hits.load() + log_misses.load();
        return total > 0 ? static_cast<double>(log_hits.load()) / static_cast<double>(total) : 0.0;
    }
    
    double getOverallHitRate() const noexcept {
        size_t total_hits = gamma_hits + erf_hits + beta_hits + log_hits;
        size_t total_misses = gamma_misses + erf_misses + beta_misses + log_misses;
        size_t total = total_hits + total_misses;
        return total > 0 ? static_cast<double>(total_hits) / static_cast<double>(total) : 0.0;
    }
};

/**
 * @brief High-performance mathematical function cache with precision rounding
 * 
 * Provides caching for expensive mathematical functions used across distributions
 * with configurable precision to balance accuracy vs cache hit rates.
 * 
 * Key features:
 * - Precision rounding to improve cache hit rates
 * - Separate caches for different function types
 * - Thread-safe concurrent access
 * - Performance monitoring and statistics
 * - Adaptive cache management
 */
class MathFunctionCache {
private:
    // High-performance cache types with specialized keys
    using SingleArgCache = AdaptiveCache<SingleArgKey, double>;
    using TwoArgCache = AdaptiveCache<TwoArgKey, double>;
    
    // Separate caches for different function types
    static std::unique_ptr<SingleArgCache> gamma_cache_;
    static std::unique_ptr<SingleArgCache> erf_cache_;
    static std::unique_ptr<SingleArgCache> erfc_cache_;
    static std::unique_ptr<TwoArgCache> beta_cache_;
    static std::unique_ptr<SingleArgCache> log_cache_;
    
    // Configuration and statistics
    static MathFunctionCacheConfig config_;
    static MathFunctionCacheStats stats_;
    
    // Thread safety for initialization
    static std::once_flag init_flag_;
    static std::shared_mutex stats_mutex_;
    
    /**
     * @brief Initialize all caches with proper configuration
     */
    static void initializeCaches();
    
    /**
     * @brief Create high-performance cache keys (no longer needed as template)
     */
    
    /**
     * @brief Update cache statistics
     */
    static void updateStats(const std::string& function_type, bool cache_hit, double lookup_time_ns);
    
public:
    /**
     * @brief Initialize cache with custom configuration
     */
    static void initialize(const MathFunctionCacheConfig& config = MathFunctionCacheConfig{});
    
    /**
     * @brief Get cached gamma function result
     * @param x Input value
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed gamma function result
     */
    static double getCachedGamma(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached log-gamma function result  
     * @param x Input value
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed log(gamma(x))
     */
    static double getCachedLGamma(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached error function result
     * @param x Input value
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed erf(x)
     */
    static double getCachedErf(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached complementary error function result
     * @param x Input value
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed erfc(x)
     */
    static double getCachedErfc(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached beta function result
     * @param a First parameter
     * @param b Second parameter
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed beta(a, b)
     */
    static double getCachedBeta(double a, double b, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached natural logarithm result
     * @param x Input value (must be positive)
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed log(x)
     */
    static double getCachedLog(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Get cached logarithm base 10 result
     * @param x Input value (must be positive)
     * @param precision Rounding precision (default uses config)
     * @return Cached or computed log10(x)
     */
    static double getCachedLog10(double x, std::optional<double> precision = std::nullopt);
    
    /**
     * @brief Clear all mathematical function caches
     */
    static void clearAll();
    
    /**
     * @brief Clear specific function cache
     */
    static void clearGammaCache();
    static void clearErfCache();
    static void clearBetaCache();
    static void clearLogCache();
    
    /**
     * @brief Get comprehensive cache statistics
     */
    static MathFunctionCacheStats getStats();
    
    /**
     * @brief Get current cache configuration
     */
    static MathFunctionCacheConfig getConfig();
    
    /**
     * @brief Update cache configuration
     * @param config New configuration settings
     */
    static void updateConfig(const MathFunctionCacheConfig& config);
    
    /**
     * @brief Get cache memory usage in bytes
     */
    static size_t getMemoryUsage();
    
    /**
     * @brief Perform cache optimization and cleanup
     */
    static void optimize();
    
    /**
     * @brief Enable/disable cache statistics collection
     */
    static void setStatisticsEnabled(bool enabled);
    
    /**
     * @brief Check if caches are initialized
     */
    static bool isInitialized();
    
    /**
     * @brief Warm up caches with common values
     */
    static void warmUp();
    
    /**
     * @brief Print cache statistics summary
     */
    static void printStats();
};

// Helper macros for convenient cache usage in distributions
#define CACHED_GAMMA(x) libstats::cache::MathFunctionCache::getCachedGamma(x)
#define CACHED_LGAMMA(x) libstats::cache::MathFunctionCache::getCachedLGamma(x)
#define CACHED_ERF(x) libstats::cache::MathFunctionCache::getCachedErf(x)
#define CACHED_ERFC(x) libstats::cache::MathFunctionCache::getCachedErfc(x)
#define CACHED_BETA(a, b) libstats::cache::MathFunctionCache::getCachedBeta(a, b)
#define CACHED_LOG(x) libstats::cache::MathFunctionCache::getCachedLog(x)
#define CACHED_LOG10(x) libstats::cache::MathFunctionCache::getCachedLog10(x)

} // namespace cache
} // namespace libstats
