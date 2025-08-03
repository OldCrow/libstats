#include <iostream>
#include "../include/libstats.h"

#ifdef _WIN32
#include <windows.h>
#endif

void test_adaptive_cache_via_main_header() {
    std::cout << "Testing adaptive cache access through main libstats header..." << std::endl;
    
    // Test that the adaptive cache is available
    libstats::cache::AdaptiveCacheConfig config;
    config.max_memory_bytes = 2048;
    config.enable_prefetching = true;
    config.ttl = std::chrono::milliseconds(5000);
    
    libstats::cache::AdaptiveCache<std::string, int> cache(config);
    
    // Basic operations test
    cache.put("test_key", 42);
    auto result = cache.get("test_key");
    
    if (result.has_value() && result.value() == 42) {
        std::cout << "✓ Adaptive cache accessible and functional via libstats.h\n";
    } else {
        throw std::runtime_error("Cache operations failed");
    }
}

void test_forward_compatibility_aliases_via_main_header() {
    std::cout << "Testing forward-compatibility aliases through main libstats header..." << std::endl;
    
    // Test that DistributionBase aliases are accessible
    using PublicAdvancedCache = libstats::DistributionBase::AdvancedAdaptiveCache<int, double>;
    using PublicAdvancedConfig = libstats::DistributionBase::AdvancedCacheConfig;
    
    PublicAdvancedConfig public_config;
    public_config.max_memory_bytes = 1024 * 1024; // 1MB
    public_config.enable_background_optimization = true;
    
    std::cout << "✓ Forward-compatibility aliases accessible via libstats.h\n";
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    std::cout << "=== libstats.h Integration Test ===" << std::endl;
    std::cout << std::endl;
    
    try {
        test_adaptive_cache_via_main_header();
        std::cout << std::endl;
        
        test_forward_compatibility_aliases_via_main_header();
        std::cout << std::endl;
        
        std::cout << "🎉 All libstats.h integration tests passed! 🎉" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}
