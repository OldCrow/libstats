#include <iostream>
#include "../include/distribution_base.h"

using namespace libstats;

void test_alias_access() {
    std::cout << "Testing access to forward-compatibility aliases..." << std::endl;
    
    using PublicAdvancedCache = DistributionBase::AdvancedAdaptiveCache<int, double>;
    using PublicAdvancedConfig = DistributionBase::AdvancedCacheConfig;
    
    PublicAdvancedConfig public_config;
    public_config.max_memory_bytes = 1024;
    public_config.enable_background_optimization = true;
    
    std::cout << "✓ Forward-compatibility aliases accessible and working\n";
}

int main() {
    try {
        test_alias_access();
        std::cout << "🎉 Alias access test passed! 🎉" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
    
    return 0;
}

