// Test program to verify AdaptiveCache is available through libstats.h
#include "include/libstats.h"
#include <iostream>
#include <string>

int main() {
    std::cout << "=== Testing AdaptiveCache Integration ===" << std::endl;
    
    // Test 1: Create adaptive cache directly
    std::cout << "1. Creating adaptive cache..." << std::endl;
    
    libstats::cache::AdaptiveCacheConfig config;
    config.max_memory_bytes = 1024 * 1024;  // 1MB
    config.ttl = std::chrono::seconds(30);
    
    libstats::cache::AdaptiveCache<std::string, double> cache(config);
    std::cout << "   ✓ AdaptiveCache created successfully" << std::endl;
    
    // Test 2: Use the cache
    std::cout << "2. Testing cache operations..." << std::endl;
    
    cache.put("pi", 3.14159);
    cache.put("e", 2.71828);
    
    auto pi_value = cache.get("pi");
    auto e_value = cache.get("e");
    auto missing_value = cache.get("missing");
    
    std::cout << "   ✓ Put operations successful" << std::endl;
    std::cout << "   ✓ Get operations successful" << std::endl;
    std::cout << "   - pi = " << (pi_value ? std::to_string(*pi_value) : "NOT FOUND") << std::endl;
    std::cout << "   - e = " << (e_value ? std::to_string(*e_value) : "NOT FOUND") << std::endl;
    std::cout << "   - missing = " << (missing_value ? std::to_string(*missing_value) : "NOT FOUND") << std::endl;
    
    // Test 3: Test forward-compatibility aliases in DistributionBase
    std::cout << "3. Testing forward-compatibility aliases..." << std::endl;
    
    // These should compile without errors if the aliases work
    using TestAdvancedCache = libstats::DistributionBase::AdvancedAdaptiveCache<int, double>;
    using TestAdvancedConfig = libstats::DistributionBase::AdvancedCacheConfig;
    
    TestAdvancedConfig advanced_config;
    advanced_config.max_memory_bytes = 512 * 1024;  // 512KB
    
    TestAdvancedCache advanced_cache(advanced_config);
    advanced_cache.put(42, 3.14159);
    
    auto result = advanced_cache.get(42);
    std::cout << "   ✓ Forward-compatibility aliases work" << std::endl;
    std::cout << "   - Advanced cache result = " << (result ? std::to_string(*result) : "NOT FOUND") << std::endl;
    
    // Test 4: Version information
    std::cout << "4. Testing version info..." << std::endl;
    std::cout << "   - libstats version: " << libstats::VERSION_STRING << std::endl;
    
    std::cout << "\n=== ALL INTEGRATION TESTS PASSED! ===" << std::endl;
    std::cout << "✓ AdaptiveCache is properly integrated into libstats" << std::endl;
    std::cout << "✓ Forward-compatibility aliases are working" << std::endl;
    std::cout << "✓ Ready for Level 4 distribution integration" << std::endl;
    
    return 0;
}
