#include "../include/cache/math_function_cache.h"
#include "../include/core/math_utils.h"
#include <iostream>
#include <iomanip>

namespace libstats {
namespace cache {

// Static member definitions
std::unique_ptr<MathFunctionCache::SingleArgCache> MathFunctionCache::gamma_cache_;
std::unique_ptr<MathFunctionCache::SingleArgCache> MathFunctionCache::erf_cache_;
std::unique_ptr<MathFunctionCache::SingleArgCache> MathFunctionCache::erfc_cache_;
std::unique_ptr<MathFunctionCache::TwoArgCache> MathFunctionCache::beta_cache_;
std::unique_ptr<MathFunctionCache::SingleArgCache> MathFunctionCache::log_cache_;

MathFunctionCacheConfig MathFunctionCache::config_;
MathFunctionCacheStats MathFunctionCache::stats_;
std::once_flag MathFunctionCache::init_flag_;
std::shared_mutex MathFunctionCache::stats_mutex_;

void MathFunctionCache::initializeCaches() {
    // Create cache configurations for each function type
    AdaptiveCacheConfig gamma_config;
    gamma_config.max_cache_size = config_.gamma_cache_size;
    gamma_config.ttl = config_.ttl;
    gamma_config.enable_background_optimization = config_.enable_background_cleanup;
    gamma_config.max_memory_bytes = (config_.memory_limit_mb * 1024 * 1024) / 5; // Divide among caches
    
    AdaptiveCacheConfig erf_config = gamma_config;
    erf_config.max_cache_size = config_.erf_cache_size;
    
    AdaptiveCacheConfig beta_config = gamma_config;
    beta_config.max_cache_size = config_.beta_cache_size;
    
    AdaptiveCacheConfig log_config = gamma_config;
    log_config.max_cache_size = config_.log_cache_size;
    
    // Initialize all caches with optimized configurations
    gamma_cache_ = std::make_unique<SingleArgCache>(gamma_config);
    erf_cache_ = std::make_unique<SingleArgCache>(erf_config);
    erfc_cache_ = std::make_unique<SingleArgCache>(erf_config); // Same config as erf
    beta_cache_ = std::make_unique<TwoArgCache>(beta_config);
    log_cache_ = std::make_unique<SingleArgCache>(log_config);
}

void MathFunctionCache::updateStats(const std::string& function_type, bool cache_hit, double lookup_time_ns) {
    if (!config_.enable_statistics) return;
    
    std::shared_lock lock(stats_mutex_);
    
    // Update hit/miss statistics atomically
    if (function_type == "gamma") {
        if (cache_hit) stats_.gamma_hits.fetch_add(1, std::memory_order_relaxed);
        else stats_.gamma_misses.fetch_add(1, std::memory_order_relaxed);
    } else if (function_type == "erf") {
        if (cache_hit) stats_.erf_hits.fetch_add(1, std::memory_order_relaxed);
        else stats_.erf_misses.fetch_add(1, std::memory_order_relaxed);
    } else if (function_type == "beta") {
        if (cache_hit) stats_.beta_hits.fetch_add(1, std::memory_order_relaxed);
        else stats_.beta_misses.fetch_add(1, std::memory_order_relaxed);
    } else if (function_type == "log") {
        if (cache_hit) stats_.log_hits.fetch_add(1, std::memory_order_relaxed);
        else stats_.log_misses.fetch_add(1, std::memory_order_relaxed);
    }
    
    // Update average lookup time using exponential moving average
    double current_avg = stats_.average_lookup_time_ns.load();
    double alpha = 0.1; // EMA smoothing factor
    double new_avg = alpha * lookup_time_ns + (1.0 - alpha) * current_avg;
    stats_.average_lookup_time_ns.store(new_avg, std::memory_order_relaxed);
}

void MathFunctionCache::initialize(const MathFunctionCacheConfig& config) {
    config_ = config;
    std::call_once(init_flag_, initializeCaches);
}

double MathFunctionCache::getCachedGamma(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.gamma_precision);
    SingleArgKey key = SingleArgKey::create(x, prec);
    
    // Try to get from cache first
    auto cached_result = gamma_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("gamma", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the gamma function using standard library
    double result = std::tgamma(x);
    
    // Cache the result
    gamma_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("gamma", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedLGamma(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.gamma_precision);
    // Use a different key space for lgamma by adding an offset
    SingleArgKey key = SingleArgKey::create(x + 1000000.0, prec); // Offset to avoid collision
    
    // Try to get from cache first
    auto cached_result = gamma_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("gamma", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the log-gamma function
    double result = std::lgamma(x);
    
    // Cache the result
    gamma_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("gamma", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedErf(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.erf_precision);
    SingleArgKey key = SingleArgKey::create(x, prec);
    
    // Try to get from cache first
    auto cached_result = erf_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("erf", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the error function
    double result = std::erf(x);
    
    // Cache the result
    erf_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("erf", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedErfc(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.erf_precision);
    SingleArgKey key = SingleArgKey::create(x, prec);
    
    // Try to get from cache first
    auto cached_result = erfc_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("erf", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the complementary error function
    double result = std::erfc(x);
    
    // Cache the result
    erfc_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("erf", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedBeta(double a, double b, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.beta_precision);
    TwoArgKey key = TwoArgKey::create(a, b, prec);
    
    // Try to get from cache first
    auto cached_result = beta_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("beta", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the beta function using gamma functions: B(a,b) = Γ(a)Γ(b)/Γ(a+b)
    double result = std::tgamma(a) * std::tgamma(b) / std::tgamma(a + b);
    
    // Cache the result
    beta_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("beta", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedLog(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.log_precision);
    SingleArgKey key = SingleArgKey::create(x, prec);
    
    // Try to get from cache first
    auto cached_result = log_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("log", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the natural logarithm
    double result = std::log(x);
    
    // Cache the result
    log_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("log", false, lookup_time);
    
    return result;
}

double MathFunctionCache::getCachedLog10(double x, std::optional<double> precision) {
    std::call_once(init_flag_, initializeCaches);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    double prec = precision.value_or(config_.log_precision);
    // Use offset to separate log10 from natural log in same cache
    SingleArgKey key = SingleArgKey::create(x + 2000000.0, prec); // Offset to avoid collision
    
    // Try to get from cache first
    auto cached_result = log_cache_->get(key);
    if (cached_result.has_value()) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
        updateStats("log", true, lookup_time);
        return *cached_result;
    }
    
    // Compute the base-10 logarithm
    double result = std::log10(x);
    
    // Cache the result
    log_cache_->put(key, result);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double lookup_time = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    updateStats("log", false, lookup_time);
    
    return result;
}

void MathFunctionCache::clearAll() {
    std::call_once(init_flag_, initializeCaches);
    
    clearGammaCache();
    clearErfCache();
    clearBetaCache();
    clearLogCache();
}

void MathFunctionCache::clearGammaCache() {
    if (gamma_cache_) {
        // Clear cache by recreating it
        AdaptiveCacheConfig gamma_config;
        gamma_config.max_cache_size = config_.gamma_cache_size;
        gamma_config.ttl = config_.ttl;
        gamma_config.enable_background_optimization = config_.enable_background_cleanup;
        gamma_config.max_memory_bytes = (config_.memory_limit_mb * 1024 * 1024) / 5;
        gamma_cache_ = std::make_unique<SingleArgCache>(gamma_config);
    }
}

void MathFunctionCache::clearErfCache() {
    if (erf_cache_ && erfc_cache_) {
        AdaptiveCacheConfig erf_config;
        erf_config.max_cache_size = config_.erf_cache_size;
        erf_config.ttl = config_.ttl;
        erf_config.enable_background_optimization = config_.enable_background_cleanup;
        erf_config.max_memory_bytes = (config_.memory_limit_mb * 1024 * 1024) / 5;
        
        erf_cache_ = std::make_unique<SingleArgCache>(erf_config);
        erfc_cache_ = std::make_unique<SingleArgCache>(erf_config);
    }
}

void MathFunctionCache::clearBetaCache() {
    if (beta_cache_) {
        AdaptiveCacheConfig beta_config;
        beta_config.max_cache_size = config_.beta_cache_size;
        beta_config.ttl = config_.ttl;
        beta_config.enable_background_optimization = config_.enable_background_cleanup;
        beta_config.max_memory_bytes = (config_.memory_limit_mb * 1024 * 1024) / 5;
        beta_cache_ = std::make_unique<TwoArgCache>(beta_config);
    }
}

void MathFunctionCache::clearLogCache() {
    if (log_cache_) {
        AdaptiveCacheConfig log_config;
        log_config.max_cache_size = config_.log_cache_size;
        log_config.ttl = config_.ttl;
        log_config.enable_background_optimization = config_.enable_background_cleanup;
        log_config.max_memory_bytes = (config_.memory_limit_mb * 1024 * 1024) / 5;
        log_cache_ = std::make_unique<SingleArgCache>(log_config);
    }
}

MathFunctionCacheStats MathFunctionCache::getStats() {
    std::shared_lock lock(stats_mutex_);
    // Use the copy constructor to create a snapshot
    return MathFunctionCacheStats(stats_);
}

MathFunctionCacheConfig MathFunctionCache::getConfig() {
    return config_;
}

void MathFunctionCache::updateConfig(const MathFunctionCacheConfig& config) {
    config_ = config;
    // Note: Existing caches will continue using their original config until cleared
}

size_t MathFunctionCache::getMemoryUsage() {
    std::call_once(init_flag_, initializeCaches);
    
    size_t total_memory = 0;
    
    // Estimate memory usage more accurately
    // Each cache entry: key (8-16 bytes) + double value (8 bytes) + metadata (~64 bytes) ≈ 80-88 bytes
    const size_t bytes_per_entry = 88;
    
    if (gamma_cache_) {
        size_t gamma_entries = stats_.gamma_hits.load() + stats_.gamma_misses.load();
        total_memory += std::min(gamma_entries, config_.gamma_cache_size) * bytes_per_entry;
    }
    if (erf_cache_) {
        size_t erf_entries = stats_.erf_hits.load() + stats_.erf_misses.load();
        total_memory += std::min(erf_entries, config_.erf_cache_size) * bytes_per_entry;
    }
    if (erfc_cache_) {
        // erfc shares stats with erf
        total_memory += std::min((stats_.erf_hits.load() + stats_.erf_misses.load()) / 2, config_.erf_cache_size) * bytes_per_entry;
    }
    if (beta_cache_) {
        size_t beta_entries = stats_.beta_hits.load() + stats_.beta_misses.load();
        total_memory += std::min(beta_entries, config_.beta_cache_size) * bytes_per_entry;
    }
    if (log_cache_) {
        size_t log_entries = stats_.log_hits.load() + stats_.log_misses.load();
        total_memory += std::min(log_entries, config_.log_cache_size) * bytes_per_entry;
    }
    
    stats_.total_memory_bytes.store(total_memory, std::memory_order_relaxed);
    return total_memory;
}

void MathFunctionCache::optimize() {
    // Trigger optimization in all caches by clearing expired entries
    std::call_once(init_flag_, initializeCaches);
    
    // The AdaptiveCache instances handle their own optimization through background threads
    // This function can be extended to perform cross-cache optimizations if needed
}

void MathFunctionCache::setStatisticsEnabled(bool enabled) {
    config_.enable_statistics = enabled;
}

bool MathFunctionCache::isInitialized() {
    return gamma_cache_ != nullptr && erf_cache_ != nullptr && 
           erfc_cache_ != nullptr && beta_cache_ != nullptr && log_cache_ != nullptr;
}

void MathFunctionCache::warmUp() {
    std::call_once(init_flag_, initializeCaches);
    
    // Pre-populate caches with common values
    std::vector<double> common_gamma_values = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0};
    for (double x : common_gamma_values) {
        getCachedGamma(x);
        getCachedLGamma(x);
    }
    
    std::vector<double> common_erf_values = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
    for (double x : common_erf_values) {
        getCachedErf(x);
        getCachedErfc(x);
    }
    
    std::vector<std::pair<double, double>> common_beta_values = {
        {1.0, 1.0}, {1.0, 2.0}, {2.0, 1.0}, {2.0, 2.0}, {0.5, 0.5}
    };
    for (const auto& [a, b] : common_beta_values) {
        getCachedBeta(a, b);
    }
    
    std::vector<double> common_log_values = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};
    for (double x : common_log_values) {
        getCachedLog(x);
        getCachedLog10(x);
    }
}

void MathFunctionCache::printStats() {
    auto stats = getStats();
    
    std::cout << "=== Mathematical Function Cache Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Gamma Function Cache:" << std::endl;
    std::cout << "  Hits: " << stats.gamma_hits.load() << std::endl;
    std::cout << "  Misses: " << stats.gamma_misses.load() << std::endl;
    std::cout << "  Hit Rate: " << (stats.getGammaHitRate() * 100) << "%" << std::endl;
    
    std::cout << "Error Function Cache:" << std::endl;
    std::cout << "  Hits: " << stats.erf_hits.load() << std::endl;
    std::cout << "  Misses: " << stats.erf_misses.load() << std::endl;
    std::cout << "  Hit Rate: " << (stats.getErfHitRate() * 100) << "%" << std::endl;
    
    std::cout << "Beta Function Cache:" << std::endl;
    std::cout << "  Hits: " << stats.beta_hits.load() << std::endl;
    std::cout << "  Misses: " << stats.beta_misses.load() << std::endl;
    std::cout << "  Hit Rate: " << (stats.getBetaHitRate() * 100) << "%" << std::endl;
    
    std::cout << "Log Function Cache:" << std::endl;
    std::cout << "  Hits: " << stats.log_hits.load() << std::endl;
    std::cout << "  Misses: " << stats.log_misses.load() << std::endl;
    std::cout << "  Hit Rate: " << (stats.getLogHitRate() * 100) << "%" << std::endl;
    
    std::cout << "Overall Statistics:" << std::endl;
    std::cout << "  Overall Hit Rate: " << (stats.getOverallHitRate() * 100) << "%" << std::endl;
    std::cout << "  Average Lookup Time: " << stats.average_lookup_time_ns.load() << " ns" << std::endl;
    std::cout << "  Total Memory Usage: " << getMemoryUsage() << " bytes (" 
              << (static_cast<double>(getMemoryUsage()) / 1024.0) << " KB)" << std::endl;
    
    auto uptime = std::chrono::steady_clock::now() - stats.created_at;
    auto uptime_seconds = std::chrono::duration_cast<std::chrono::seconds>(uptime).count();
    std::cout << "  Cache Uptime: " << uptime_seconds << " seconds" << std::endl;
    std::cout << "================================================" << std::endl;
}

} // namespace cache
} // namespace libstats
