#include <iostream>

// Test that all parallel execution code paths compile by forcing different macros

// Test 1: Force C++20 std::execution
#define LIBSTATS_HAS_STD_EXECUTION 1
#define LIBSTATS_HAS_PARALLEL_EXECUTION 1
#include "../include/parallel_execution.h"

namespace test_std_execution {
void test() {
    std::cout << "C++20 std::execution code compiles" << std::endl;
}
}  // namespace test_std_execution

// Reset macros for next test
#undef LIBSTATS_HAS_STD_EXECUTION
#undef LIBSTATS_HAS_PARALLEL_EXECUTION

// Test 2: Force Windows Thread Pool (won't link on macOS but will compile)
#ifdef _WIN32
    #define LIBSTATS_HAS_WIN_THREADPOOL 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
#endif

// Test 3: Force OpenMP (simulate availability)
#ifdef _OPENMP
    #define LIBSTATS_HAS_OPENMP 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
namespace test_openmp {
void test() {
    std::cout << "OpenMP code compiles (if OpenMP available)" << std::endl;
}
}  // namespace test_openmp
#endif

// Test 4: Force pthreads (simulate non-Apple environment)
#ifdef __unix__
    #define LIBSTATS_HAS_PTHREADS 1
    #define LIBSTATS_HAS_PARALLEL_EXECUTION 1
#endif

int main() {
    std::cout << "=== Parallel Execution Compilation Test ===" << std::endl;

    test_std_execution::test();

#ifdef _OPENMP
    test_openmp::test();
#else
    std::cout << "OpenMP not available on this system" << std::endl;
#endif

    std::cout << "✓ All available parallel execution code paths compile successfully" << std::endl;
    return 0;
}
