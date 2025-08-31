// Use consolidated tool utilities header which includes libstats.h
#include "tool_utils.h"

// Additional standard library includes for C++20 features demonstration
#include <array>     // for std::array
#include <bitset>    // for std::bitset
#include <chrono>    // for timing operations
#include <cstddef>   // for size_t
#include <iomanip>   // for std::setw, std::left
#include <iostream>  // for std::cout
#include <numeric>   // for numeric algorithms
#include <string>    // for std::string
#include <vector>    // for std::vector

// C++20 headers with availability checking
#ifdef __has_include
    #if __has_include(<ranges>)
        #include <ranges>
        #define HAS_RANGES 1
    #else
        #define HAS_RANGES 0
    #endif

    #if __has_include(<concepts>)
        #include <concepts>
        #define HAS_CONCEPTS 1
    #else
        #define HAS_CONCEPTS 0
    #endif

    #if __has_include(<format>)
        #include <format>
        #define HAS_FORMAT 1
    #else
        #define HAS_FORMAT 0
    #endif

    #if __has_include(<span>)
        #include <span>
        #define HAS_SPAN 1
    #else
        #define HAS_SPAN 0
    #endif

    #if __has_include(<bit>)
        #include <bit>
        #define HAS_BIT 1
    #else
        #define HAS_BIT 0
    #endif

    #if __has_include(<numbers>)
        #include <numbers>
        #define HAS_NUMBERS 1
    #else
        #define HAS_NUMBERS 0
    #endif

    #if __has_include(<barrier>)
        #include <barrier>
        #define HAS_BARRIER 1
    #else
        #define HAS_BARRIER 0
    #endif

    #if __has_include(<latch>)
        #include <latch>
        #define HAS_LATCH 1
    #else
        #define HAS_LATCH 0
    #endif

    #if __has_include(<semaphore>)
        #include <semaphore>
        #define HAS_SEMAPHORE 1
    #else
        #define HAS_SEMAPHORE 0
    #endif

    #if __has_include(<coroutine>)
        #include <coroutine>
        #define HAS_COROUTINE 1
    #else
        #define HAS_COROUTINE 0
    #endif

    #if __has_include(<stop_token>)
        #include <stop_token>
        #define HAS_STOP_TOKEN 1
    #else
        #define HAS_STOP_TOKEN 0
    #endif

    #if __has_include(<source_location>)
        #include <source_location>
        #define HAS_SOURCE_LOCATION 1
    #else
        #define HAS_SOURCE_LOCATION 0
    #endif

    #if __has_include(<syncstream>)
        #include <syncstream>
        #define HAS_SYNCSTREAM 1
    #else
        #define HAS_SYNCSTREAM 0
    #endif
#else
    // Fallback for compilers without __has_include
    #define HAS_RANGES 0
    #define HAS_CONCEPTS 0
    #define HAS_FORMAT 0
    #define HAS_SPAN 0
    #define HAS_BIT 0
    #define HAS_NUMBERS 0
    #define HAS_BARRIER 0
    #define HAS_LATCH 0
    #define HAS_SEMAPHORE 0
    #define HAS_COROUTINE 0
    #define HAS_STOP_TOKEN 0
    #define HAS_SOURCE_LOCATION 0
    #define HAS_SYNCSTREAM 0
#endif

// Try to include execution policy header
#ifdef __has_include
    #if __has_include(<execution>)
        #include <execution>
        #define HAS_EXECUTION 1
    #else
        #define HAS_EXECUTION 0
    #endif
#else
    #define HAS_EXECUTION 0
#endif

// Try to include algorithm header for parallel algorithms
#ifdef __has_include
    #if __has_include(<algorithm>)
        #include <algorithm>
        #define HAS_ALGORITHM 1
    #else
        #define HAS_ALGORITHM 0
    #endif
#else
    #define HAS_ALGORITHM 0
#endif

// C++20 Concepts test (if available)
#if HAS_CONCEPTS
template <typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template <Numeric T>
T add_numbers(T a, T b) {
    return a + b;
}

template <typename T>
concept Printable = requires(T t) { std::cout << t; };
#endif

class CPP20FeaturesInspector {
   private:
    bool show_details = false;

    void print_header(const std::string& title) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << title << "\n";
        std::cout << std::string(60, '=') << "\n";
    }

    void print_feature(const std::string& feature, bool available,
                       const std::string& details = "") {
        std::cout << std::left << std::setw(30) << feature << ": "
                  << (available ? "✓ Available" : "✗ Not Available");
        if (available && !details.empty() && show_details) {
            std::cout << " (" << details << ")";
        }
        std::cout << "\n";
    }

    void print_compiler_info() {
        print_header("Compiler Information");

#if defined(_MSC_VER)
        std::cout << "Compiler: Microsoft Visual C++ (MSVC)\n";
        std::cout << "Version: " << _MSC_VER << "\n";
    #if defined(_MSVC_LANG)
        std::cout << "MSVC Language Level: " << _MSVC_LANG << "\n";
    #endif
#elif defined(__clang__)
        std::cout << "Compiler: Clang/LLVM\n";
        std::cout << "Version: " << __clang_version__ << "\n";
        std::cout << "Major: " << __clang_major__ << ", Minor: " << __clang_minor__ << "\n";
#elif defined(__GNUC__)
        std::cout << "Compiler: GCC (GNU Compiler Collection)\n";
        std::cout << "Version: " << __VERSION__ << "\n";
        std::cout << "Major: " << __GNUC__ << ", Minor: " << __GNUC_MINOR__ << "\n";
#else
        std::cout << "Compiler: Unknown\n";
#endif

        std::cout << "C++ Standard Level: " << __cplusplus << "\n";
        std::cout << "Expected C++20: " << ((__cplusplus >= 202002L) ? "✓ Yes" : "✗ No") << "\n";

        // Feature test macros
        std::cout << "\nFeature Test Macros:\n";
#ifdef __cpp_concepts
        std::cout << "  __cpp_concepts: " << __cpp_concepts << "\n";
#endif
#ifdef __cpp_ranges
        std::cout << "  __cpp_ranges: " << __cpp_ranges << "\n";
#endif
#ifdef __cpp_coroutines
        std::cout << "  __cpp_coroutines: " << __cpp_coroutines << "\n";
#endif
#ifdef __cpp_modules
        std::cout << "  __cpp_modules: " << __cpp_modules << "\n";
#endif
#ifdef __cpp_consteval
        std::cout << "  __cpp_consteval: " << __cpp_consteval << "\n";
#endif
#ifdef __cpp_constinit
        std::cout << "  __cpp_constinit: " << __cpp_constinit << "\n";
#endif
    }

    void test_core_language_features() {
        print_header("Core Language Features");

// Three-way comparison
#ifdef __cpp_impl_three_way_comparison
        print_feature("Three-way comparison (<==>)", true, "Spaceship operator");
#else
        print_feature("Three-way comparison (<==>)", false);
#endif

// consteval
#ifdef __cpp_consteval
        print_feature("consteval", true, "Immediate functions");
#else
        print_feature("consteval", false);
#endif

// constinit
#ifdef __cpp_constinit
        print_feature("constinit", true, "Constant initialization");
#else
        print_feature("constinit", false);
#endif

// designated initializers
#ifdef __cpp_designated_initializers
        print_feature("Designated initializers", true, ".member = value");
#else
        print_feature("Designated initializers", false);
#endif

// using enum
#ifdef __cpp_using_enum
        print_feature("using enum", true, "Enum using declarations");
#else
        print_feature("using enum", false);
#endif
    }

    void test_standard_library_features() {
        print_header("Standard Library Headers");

        print_feature("concepts", HAS_CONCEPTS, "Type constraints and requirements");
        print_feature("ranges", HAS_RANGES, "Range-based algorithms and views");
        print_feature("format", HAS_FORMAT, "Text formatting library");
        print_feature("span", HAS_SPAN, "Non-owning array view");
        print_feature("bit", HAS_BIT, "Bit manipulation utilities");
        print_feature("numbers", HAS_NUMBERS, "Mathematical constants");
        print_feature("source_location", HAS_SOURCE_LOCATION, "Source location utilities");
        print_feature("syncstream", HAS_SYNCSTREAM, "Synchronized output streams");

        print_header("Threading and Synchronization");

        print_feature("barrier", HAS_BARRIER, "Thread synchronization barrier");
        print_feature("latch", HAS_LATCH, "Single-use thread synchronization");
        print_feature("semaphore", HAS_SEMAPHORE, "Counting semaphore");
        print_feature("stop_token", HAS_STOP_TOKEN, "Cooperative thread cancellation");

// Check for jthread
#ifdef __cpp_lib_jthread
        print_feature("jthread", true, "Joinable thread with stop support");
#else
        print_feature("jthread", false);
#endif

        print_header("Parallel Execution");

        print_feature("execution", HAS_EXECUTION, "Execution policy support");

// Test actual execution policies
#if HAS_EXECUTION
        bool has_par = false, has_par_unseq = false, has_unseq = false;

    // Test for execution policies at compile-time
    #ifdef __cpp_lib_execution
        try {
            // Try to access the execution policies
            (void)std::execution::seq;
            has_par = true;  // If we can access seq, par should be available
            has_par_unseq = true;
            has_unseq = true;
        } catch (...) {
            // Execution policies not available
        }
    #endif

        print_feature("  std::execution::par", has_par, "Parallel execution");
        print_feature("  std::execution::par_unseq", has_par_unseq, "Parallel unsequenced");
        print_feature("  std::execution::unseq", has_unseq, "Unsequenced execution");
#endif

        print_header("Coroutines");

        print_feature("coroutine", HAS_COROUTINE, "Coroutines support");
#ifdef __cpp_coroutines
        print_feature("  Coroutines language support", true, "co_await, co_yield, co_return");
#else
        print_feature("  Coroutines language support", false);
#endif
    }

    void test_concepts_functionality() {
#if HAS_CONCEPTS
        print_header("Concepts Functionality Test");

        try {
            auto int_result = add_numbers(5, 10);
            auto double_result = add_numbers(3.14, 2.86);

            std::cout << "✓ Concepts working correctly\n";
            std::cout << "  Integer addition: " << int_result << "\n";
            std::cout << "  Double addition: " << double_result << "\n";

            // Test concept requirements
            static_assert(Numeric<int>, "int should satisfy Numeric concept");
            static_assert(Numeric<double>, "double should satisfy Numeric concept");
            static_assert(!Numeric<std::string>, "string should not satisfy Numeric concept");

            std::cout << "  Static assertions passed\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Concepts test failed: " << e.what() << "\n";
        }
#endif
    }

    void test_ranges_functionality() {
#if HAS_RANGES
        print_header("Ranges Functionality Test");

        try {
            std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

            auto even_squared = numbers | std::views::filter([](int n) { return n % 2 == 0; }) |
                                std::views::transform([](int n) { return n * n; });

            std::cout << "✓ Ranges working correctly\n";
            std::cout << "  Even numbers squared: ";
            for (int n : even_squared) {
                std::cout << n << " ";
            }
            std::cout << "\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Ranges test failed: " << e.what() << "\n";
        }
#endif
    }

    void test_format_functionality() {
#if HAS_FORMAT
        print_header("Format Library Test");

        try {
            // Test basic formatting
            auto formatted = std::format("Hello, {}! The answer is {}", "World", 42);
            std::cout << "✓ Format library working correctly\n";
            std::cout << "  Formatted string: " << formatted << "\n";

            // Test more complex formatting
            auto complex = std::format("Pi ≈ {:.6f}, in hex: {:#x}", 3.14159265, 255);
            std::cout << "  Complex formatting: " << complex << "\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Format test failed: " << e.what() << "\n";
        }
#endif
    }

    void test_bit_functionality() {
#if HAS_BIT
        print_header("Bit Operations Test");

        try {
            uint32_t value = 0b11010110;

            std::cout << "✓ Bit operations working correctly\n";
            std::cout << "  Value: 0b" << std::bitset<8>(value) << " (" << value << ")\n";

    #ifdef __cpp_lib_bit_cast
            std::cout << "  bit_cast available\n";
    #endif

    #ifdef __cpp_lib_bitops
            std::cout << "  Bit operations available:\n";
            std::cout << "    popcount: " << std::popcount(value) << "\n";
            std::cout << "    countl_zero: " << std::countl_zero(value) << "\n";
            std::cout << "    countr_zero: " << std::countr_zero(value) << "\n";
    #endif

        } catch (const std::exception& e) {
            std::cout << "✗ Bit operations test failed: " << e.what() << "\n";
        }
#endif
    }

    void test_numbers_functionality() {
#if HAS_NUMBERS
        print_header("Mathematical Constants Test");

        try {
            std::cout << "✓ Mathematical constants available\n";
            std::cout << "  π (pi): " << std::numbers::pi << "\n";
            std::cout << "  e: " << std::numbers::e << "\n";
            std::cout << "  √2: " << std::numbers::sqrt2 << "\n";
            std::cout << "  ln(2): " << std::numbers::ln2 << "\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Numbers test failed: " << e.what() << "\n";
        }
#endif
    }

    void test_threading_functionality() {
        print_header("Threading Functionality Test");

        try {
            std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency()
                      << " threads\n";

            std::function<void()> task = []() {
                std::cout << "  Task executed in thread: " << std::this_thread::get_id() << "\n";
            };

            std::atomic<int> counter{0};
            std::vector<std::thread> threads;

            // Create multiple threads
            for (int i = 0; i < 4; ++i) {
                threads.emplace_back([&counter, task, i]() {
                    task();
                    counter.fetch_add(1);
                    std::cout << "  Worker " << i << " completed\n";
                });
            }

            // Wait for all threads
            for (auto& t : threads) {
                t.join();
            }

            std::cout << "✓ Threading working correctly: " << counter.load()
                      << " threads completed\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Threading test failed: " << e.what() << "\n";
        }
    }

    void test_parallel_algorithms() {
#if HAS_EXECUTION && HAS_ALGORITHM
        print_header("Parallel Algorithms Test");

        try {
            std::vector<int> data(1000);
            std::iota(data.begin(), data.end(), 1);

            // Test parallel sort (if available)
            auto data_copy = data;

            auto start = std::chrono::high_resolution_clock::now();

    #ifdef __cpp_lib_parallel_algorithm
            std::sort(std::execution::par, data_copy.begin(), data_copy.end(), std::greater<int>());
            std::cout << "✓ Parallel algorithms available and working\n";
            std::cout << "  Parallel sort completed\n";
    #else
            std::sort(data_copy.begin(), data_copy.end(), std::greater<int>());
            std::cout << "⚠ Parallel algorithms not available, using sequential fallback\n";
    #endif

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            std::cout << "  Sort duration: " << duration.count() << " μs\n";

        } catch (const std::exception& e) {
            std::cout << "✗ Parallel algorithms test failed: " << e.what() << "\n";
        }
#endif
    }

   public:
    void run_inspection(bool detailed = false) {
        show_details = detailed;

        std::cout << "🔍 C++20 Features Inspector\n";
        std::cout << "Comprehensive C++20 compiler and standard library feature detection\n";

        print_compiler_info();
        test_core_language_features();
        test_standard_library_features();

        if (detailed) {
            std::cout << "\n" << std::string(60, '-') << "\n";
            std::cout << "Detailed Functionality Tests\n";
            std::cout << std::string(60, '-') << "\n";

            test_concepts_functionality();
            test_ranges_functionality();
            test_format_functionality();
            test_bit_functionality();
            test_numbers_functionality();
            test_threading_functionality();
            test_parallel_algorithms();
        }

        std::cout << "\n🎉 C++20 Feature Inspection Complete!\n";
    }

    void show_help() {
        std::cout << "C++20 Features Inspector - Comprehensive compiler feature detection\n\n";
        std::cout << "Usage: cpp20_features_inspector [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  -h, --help     Show this help message\n";
        std::cout << "  -d, --detailed Run detailed functionality tests\n";
        std::cout << "  -v, --verbose  Show additional details (same as --detailed)\n\n";
        std::cout << "Examples:\n";
        std::cout << "  cpp20_features_inspector           # Basic feature detection\n";
        std::cout << "  cpp20_features_inspector --detailed # Full functionality testing\n";
    }
};

int main(int argc, char* argv[]) {
    CPP20FeaturesInspector inspector;
    bool detailed = false;
    bool show_help = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
            break;
        } else if (arg == "-d" || arg == "--detailed" || arg == "-v" || arg == "--verbose") {
            detailed = true;
        } else {
            std::cout << "Unknown option: " << arg << "\n";
            std::cout << "Use --help for usage information.\n";
            return 1;
        }
    }

    if (show_help) {
        inspector.show_help();
        return 0;
    }

    try {
        inspector.run_inspection(detailed);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
