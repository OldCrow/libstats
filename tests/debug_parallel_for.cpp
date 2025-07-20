#include <iostream>
#include <vector>
#include <cassert>
#include "work_stealing_pool.h"

using namespace libstats;

int main() {
    std::cout << "=== Debugging ParallelFor Segfault ===\n\n";
    
    // Test 1: Basic task submission (this worked before)
    std::cout << "Test 1: Basic task submission\n";
    {
        WorkStealingPool pool(2);
        std::atomic<int> counter{0};
        
        pool.submit([&counter]() {
            counter.fetch_add(1);
            std::cout << "  Basic task executed\n";
        });
        
        pool.waitForAll();
        assert(counter.load() == 1);
        std::cout << "  ✓ Basic task submission works\n\n";
    }
    
    // Test 2: Minimal parallelFor test
    std::cout << "Test 2: Minimal parallelFor (this causes segfault)\n";
    {
        WorkStealingPool pool(2);
        std::vector<int> data(10, 0);
        
        std::cout << "  About to call parallelFor...\n";
        
        // This should cause the segfault
        pool.parallelFor(0, 10, [&data](size_t i) {
            std::cout << "    Processing index: " << i << "\n";
            data[i] = static_cast<int>(i * 2);
        });
        
        std::cout << "  ParallelFor completed\n";
        
        // Verify results if we get here
        for (size_t i = 0; i < 10; ++i) {
            assert(data[i] == static_cast<int>(i * 2));
        }
        std::cout << "  ✓ ParallelFor works correctly\n";
    }
    
    std::cout << "\n🎉 All tests passed!\n";
    return 0;
}
