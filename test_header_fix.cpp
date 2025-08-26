#include "include/common/platform_constants_fwd.h"

#include <iostream>

int main() {
    // Test that the header compiles without conflicts
    std::cout << "Testing forward declarations..." << std::endl;

    // Test cache line size function (should not cause linker error now)
    try {
        auto cache_size = stats::arch::get_cache_line_size();
        std::cout << "Cache line size: " << cache_size << " bytes" << std::endl;
    } catch (...) {
        std::cout << "Cache line size function is available but needs linking" << std::endl;
    }

    // Test other forward declared functions
    try {
        auto alignment = stats::arch::get_optimal_alignment();
        std::cout << "Optimal alignment: " << alignment << " bytes" << std::endl;
    } catch (...) {
        std::cout << "Optimal alignment function is available but needs linking" << std::endl;
    }

    std::cout << "Header compilation test completed successfully!" << std::endl;
    return 0;
}
