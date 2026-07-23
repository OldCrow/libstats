# Issue #33: opt-in SIMD experimentation tools (gather-vs-polynomial benchmarks etc.). Default OFF
# -- these are ad hoc research tools, not part of the production build. Requires
# LIBSTATS_BUILD_TOOLS=ON (the add_libstats_tool helper is defined in that scope).
option(
    LIBSTATS_BUILD_SIMD_DEV_TOOLS
    "Build opt-in SIMD experimentation/dev tools for Issue #33 (requires LIBSTATS_BUILD_TOOLS=ON)"
    OFF)
# Helper function to create tool executables with consistent configuration
function(add_libstats_tool tool_name source_file)
    add_executable(${tool_name} ${source_file})
    target_link_libraries(${tool_name} PRIVATE libstats_static)
    target_include_directories(${tool_name} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR}
                                                    ${CMAKE_SOURCE_DIR}/src)

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
    target_include_directories(${tool_name} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})

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

# Issue #33 opt-in dev tools -- gated separately since these are research/experiment tools, not part
# of the standard tool set built by every developer.
if(LIBSTATS_BUILD_SIMD_DEV_TOOLS)
    add_libstats_tool(gather_throughput_probe gather_throughput_probe.cpp)

    # gather_throughput_probe needs the same per-TU ISA flags as the SIMD kernels
    # (src/simd_avx2.cpp, src/simd_avx512.cpp) so gather/FMA intrinsics are permitted by the
    # compiler; the libstats::simd interface (linked by add_libstats_tool) only supplies the
    # LIBSTATS_HAS_AVX2/LIBSTATS_HAS_AVX512 compile *definitions*, not the ISA flags themselves.
    # AVX-512 is a superset of AVX2/FMA on MSVC (only a single /arch: flag is accepted), so prefer
    # it over AVX2 rather than passing both.
    if(LIBSTATS_HAS_AVX512)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            target_compile_options(gather_throughput_probe PRIVATE /arch:AVX512)
        else()
            target_compile_options(gather_throughput_probe PRIVATE -mavx512f -mavx512dq -mavx2
                                                                   -mfma)
        endif()
    elseif(LIBSTATS_HAS_AVX2)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            target_compile_options(gather_throughput_probe PRIVATE /arch:AVX2)
        else()
            target_compile_options(gather_throughput_probe PRIVATE -mavx2 -mfma)
        endif()
    endif()

    # NEON table-gather exp/log probe (Issue #33 Q1). NEON is baseline on AArch64, so -- unlike the
    # AVX2/AVX-512 probe above -- no per-target ISA flag is needed; the arm_neon.h intrinsics are
    # always available. On non-NEON builds the tool compiles to a "not applicable" stub (guarded
    # internally).
    add_libstats_tool(neon_table_transcendental_probe neon_table_transcendental_probe.cpp)

    message(STATUS "SIMD dev tools enabled:")
    message(
        STATUS
            "  - gather_throughput_probe: Issue #33 Stage 1-2 first kill-gate -- AVX2/AVX-512 gather vs FMA-only throughput (warm/cold/interleave)"
    )
    message(
        STATUS
            "  - neon_table_transcendental_probe: Issue #33 Q1 -- NEON table-gather exp/log vs current SLEEF polynomials (accuracy + throughput gates)"
    )
elseif(DEFINED LIBSTATS_BUILD_SIMD_DEV_TOOLS)
    message(STATUS "SIMD dev tools disabled (LIBSTATS_BUILD_SIMD_DEV_TOOLS=OFF)")
endif()

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
