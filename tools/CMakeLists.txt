# Helper function to create tool executables with consistent configuration
function(add_libstats_tool tool_name source_file)
    add_executable(${tool_name} ${source_file})
    target_link_libraries(${tool_name} PRIVATE libstats_static)
    target_include_directories(${tool_name} PRIVATE ${LIBSTATS_BUILD_INCLUDE_DIRS}
                                                    ${CMAKE_SOURCE_DIR}/src)
    libstats_apply_warnings(${tool_name}) # Phase 3B/B5

    # Link to SIMD interface target for modern CMake approach
    if(TARGET libstats_simd_interface)
        target_link_libraries(${tool_name} PRIVATE libstats::simd)
    endif()

    set_target_properties(${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/tools)
endfunction()

# Helper function for standalone tools that don't need libstats
function(add_standalone_tool tool_name source_file)
    add_executable(${tool_name} ${source_file})
    # No libstats linking - pure standard library tools Still needs include paths for tool_utils.h →
    # libstats headers
    target_include_directories(${tool_name} PRIVATE ${LIBSTATS_BUILD_INCLUDE_DIRS})
    libstats_apply_warnings(${tool_name}) # Phase 3B/B5

    set_target_properties(${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                  ${CMAKE_BINARY_DIR}/tools)
endfunction()

# System Information & Diagnostics Tools
add_libstats_tool(system_inspector system_inspector.cpp)
add_standalone_tool(cpp20_features_inspector cpp20_features_inspector.cpp)

# Performance & Benchmarking Tools
add_libstats_tool(parallel_batch_fitting_benchmark parallel_batch_fitting_benchmark.cpp)
add_libstats_tool(strategy_profile strategy_profile.cpp)
add_libstats_tool(simd_verification simd_verification.cpp)
add_libstats_tool(parallel_correctness_verification parallel_correctness_verification.cpp)
add_libstats_tool(quantile_accuracy quantile_accuracy.cpp)
add_libstats_tool(parameter_recovery_benchmark parameter_recovery_benchmark.cpp)
add_libstats_tool(threshold_validator threshold_validator.cpp)
add_libstats_tool(toctou_validator toctou_validator.cpp)
add_libstats_tool(copy_move_stress copy_move_stress.cpp)
add_libstats_tool(accuracy_sweep accuracy_sweep.cpp)

message(STATUS "Tools enabled:")
message(
    STATUS
        "  - system_inspector: Consolidated system analysis tool with multiple modes (--constants, --performance, --quick, --full)"
)
message(
    STATUS
        "  - cpp20_features_inspector: Comprehensive C++20 compiler and standard library feature detection with detailed functionality tests"
)
message(
    STATUS
        "  - parallel_batch_fitting_benchmark: Comprehensive parallel batch fitting performance analysis across all distributions with scalability testing"
)
message(
    STATUS
        "  - strategy_profile: Canonical forced-strategy profiler for dispatcher threshold tuning across distributions, operations, and batch sizes"
)
message(
    STATUS
        "  - simd_verification: Architecture-agnostic SIMD operations correctness verification across all distributions and edge cases"
)
message(
    STATUS
        "  - parallel_correctness_verification: Platform-aware parallel execution correctness validation across different threading systems (GCD, OpenMP, ThreadPool)"
)
message(
    STATUS
        "  - toctou_validator: Concurrent writer/reader race stress test for TOCTOU cache inconsistency detection across all 19 distributions"
)
message(
    STATUS
        "  - copy_move_stress: Concurrent copy/move throughput stress test across all 19 distributions (replaces test_copy_move_stress)"
)
message(
    STATUS
        "  - accuracy_sweep: Issue #46 deterministic bit-exact scalar/batch pdf/logpdf/cdf/quantile sweep across all 27 distributions for the mpmath oracle comparison"
)
