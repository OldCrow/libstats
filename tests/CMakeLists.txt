# Create tests directory in build folder
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# GTest detection — three-step: generic find_package, then Apple/Homebrew probe, then FetchContent
# as a self-contained fallback (matches libhmm test pattern).

# Step 1: plain find_package — covers vcpkg, system installs, and previously cached results.
if(NOT GTest_FOUND)
    find_package(GTest QUIET)
endif()
if(GTest_FOUND OR TARGET GTest::gtest)
    set(GTEST_FOUND TRUE)
    message(STATUS "GTest found via find_package")
endif()

# Step 2: macOS only — probe architecture-appropriate Homebrew path.
if(NOT GTEST_FOUND AND APPLE)
    if(EXISTS "/opt/homebrew/opt/googletest/lib/cmake/GTest")
        set(GTEST_ROOT "/opt/homebrew/opt/googletest")
    elseif(EXISTS "/usr/local/opt/googletest/lib/cmake/GTest")
        set(GTEST_ROOT "/usr/local/opt/googletest")
    endif()
    if(DEFINED GTEST_ROOT)
        find_package(GTest QUIET HINTS "${GTEST_ROOT}/lib/cmake/GTest")
        if(GTest_FOUND OR TARGET GTest::gtest)
            set(GTEST_FOUND TRUE)
            message(STATUS "GTest found via Homebrew at ${GTEST_ROOT}")
        elseif(EXISTS "${GTEST_ROOT}/lib/libgtest.a" AND EXISTS
                                                         "${GTEST_ROOT}/include/gtest/gtest.h")
            set(GTEST_FOUND TRUE)
            set(GTEST_INCLUDE_DIRS "${GTEST_ROOT}/include")
            set(GTEST_LIBRARIES "${GTEST_ROOT}/lib/libgtest.a")
            set(GTEST_MAIN_LIBRARIES "${GTEST_ROOT}/lib/libgtest_main.a")
            message(STATUS "GTest found manually at ${GTEST_ROOT}")
        endif()
    endif()
endif()

# Step 3: FetchContent fallback — self-contained for CI and machines without a system GTest.
if(NOT GTEST_FOUND
   AND NOT TARGET GTest::gtest
   AND NOT TARGET gtest)
    message(STATUS "GTest not found locally - fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0
        GIT_SHALLOW TRUE)
    # Prevent GTest from overriding the project's CRT choice on Windows.
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    set(GTEST_FOUND TRUE)
endif()

# Common test configuration function - reduces code duplication across test types
function(configure_common_test_settings TEST_NAME)
    # Set output directory to build/tests
    set_target_properties(
        ${TEST_NAME}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests
                   RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/tests/Debug
                   RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/tests/Release
                   RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/tests/RelWithDebInfo
                   RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/tests/MinSizeRel)

    # Add include directories
    target_include_directories(
        ${TEST_NAME} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR} ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

    # Platform-specific rpath is inherited from libstats_static/shared via PUBLIC link options. No
    # per-target rpath needed here (adding it causes duplicate -rpath linker warnings).

    # Link to SIMD interface target for modern CMake approach
    if(TARGET libstats_simd_interface)
        target_link_libraries(${TEST_NAME} PRIVATE libstats::simd)
    endif()

    # Add to test suite
    add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)

    # Set working directory for tests
    set_tests_properties(${TEST_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# Function to create a test executable with proper flags (static library)
function(create_libstats_test TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    # Link against static library by default for tests
    target_link_libraries(${TEST_NAME} PRIVATE libstats_static)

    # TBB, when enabled, is linked transitively through libstats_static.

    # Apply common test settings
    configure_common_test_settings(${TEST_NAME})
endfunction()

# Function to create a test executable linked to dynamic library.
#
# On Windows, copy stats.dll next to each dynamic test EXE so runtime loading does not depend on
# PATH. Combined with per-config output directories above, this avoids both STATUS_DLL_NOT_FOUND and
# stale Debug/Release binary collisions.
function(create_libstats_test_dynamic TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    # Link against shared library instead of static
    target_link_libraries(${TEST_NAME} PRIVATE libstats_shared)

    # Apply common test settings
    configure_common_test_settings(${TEST_NAME})

    if(WIN32)
        add_custom_command(
            TARGET ${TEST_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:libstats_shared>
                    $<TARGET_FILE_DIR:${TEST_NAME}>
            COMMENT "Copying shared libstats runtime next to ${TEST_NAME}")
    endif()
endfunction()

# Function to create a GTest-based test executable
function(create_libstats_gtest TEST_NAME SOURCE_FILE)
    if(GTEST_FOUND)
        add_executable(${TEST_NAME} ${SOURCE_FILE})

        # Link against static library and GTest
        target_link_libraries(${TEST_NAME} PRIVATE libstats_static)

        # Link GTest libraries (prefer modern CMake targets to avoid duplicate warnings)
        if(TARGET GTest::gtest AND TARGET GTest::gtest_main)
            # Modern CMake targets - these handle dependencies correctly
            target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main GTest::gtest)
        elseif(TARGET GTest::GTest AND TARGET GTest::Main)
            # Alternative modern target name
            target_link_libraries(${TEST_NAME} PRIVATE GTest::Main GTest::GTest)
        else()
            # Fallback to manual linking - be more careful about duplicates
            if(GTEST_MAIN_LIBRARIES)
                target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_MAIN_LIBRARIES})
                # Only add gtest if it's not already included in main libraries
                if(GTEST_LIBRARIES AND NOT "${GTEST_MAIN_LIBRARIES}" MATCHES "gtest")
                    target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_LIBRARIES})
                endif()
            elseif(GTEST_LIBRARIES)
                target_link_libraries(${TEST_NAME} PRIVATE ${GTEST_LIBRARIES})
            endif()
            target_include_directories(${TEST_NAME} PRIVATE ${GTEST_INCLUDE_DIRS})
        endif()

        # Apply common test settings
        configure_common_test_settings(${TEST_NAME})
    else()
        message(WARNING "GTest not found - skipping GTest-based test: ${TEST_NAME}")
    endif()
endfunction()

# =======================================================================
# LEVEL 0: Foundational Tests (Platform & Constants)
# =======================================================================
# These test platform capabilities and constants - most critical

# Basic constants and platform tests
create_libstats_gtest(test_constants test_constants.cpp)
create_libstats_test(test_cpu_detection test_cpu_detection.cpp)
create_libstats_test(test_simd_comprehensive test_simd_comprehensive.cpp)
create_libstats_gtest(test_platform_optimizations test_platform_optimizations.cpp)
create_libstats_gtest(test_simd_policy test_simd_policy.cpp)

# NEON vector_exp accuracy regression (Issue #33 Q1) -- guards the table+poly kernel's <1 ULP floor.
# Self-skips (not fails) on non-NEON builds/machines.
create_libstats_gtest(test_simd_neon_exp_accuracy test_simd_neon_exp_accuracy.cpp)

# NEON vector_log accuracy regression -- guards the clean-room table+series kernel's <1 ULP floor,
# IEEE edges, and in-place aliasing. Self-skips on non-NEON.
create_libstats_gtest(test_simd_neon_log_accuracy test_simd_neon_log_accuracy.cpp)

# NEON vector_cos accuracy regression -- guards the clean-room quadrant-reduction kernel's <1 ULP
# floor incl. the near-k*pi/2 stress set, even symmetry, domain fallback, and in-place aliasing.
# Self-skips on non-NEON.
create_libstats_gtest(test_simd_neon_cos_accuracy test_simd_neon_cos_accuracy.cpp)

# SIMD benchmark test (on-demand only, not in regular test suite due to runtime)
create_libstats_test(benchmark_simd_all benchmark_simd_all.cpp)

# =======================================================================
# LEVEL 1: Core Infrastructure Tests
# =======================================================================
# Core mathematical and safety functionality

# Consolidated numerical safety (safety.h + log_space_ops.h)
create_libstats_gtest(test_numerical_safety test_numerical_safety.cpp)

# Consolidated error handling (error_handling.h)
create_libstats_gtest(test_error_handling_comprehensive test_error_handling_comprehensive.cpp)

# Consolidated math utilities (math_utils.h)
create_libstats_gtest(test_math_comprehensive test_math_comprehensive.cpp)
create_libstats_gtest(test_statistical_utilities test_statistical_utilities.cpp)
create_libstats_gtest(test_discrete_analysis test_discrete_analysis.cpp)

# =======================================================================
# LEVEL 2: Statistical Infrastructure
# =======================================================================
# Statistical validation and goodness-of-fit

# Goodness-of-fit tests
create_libstats_gtest(test_goodness_of_fit test_goodness_of_fit.cpp)

# LEVEL 2b: Platform Capabilities thread_pool.h and work_stealing_pool.h - Thread pool
# implementations
create_libstats_gtest(test_thread_pool test_thread_pool.cpp)
# <concepts> and <ranges> are guaranteed on the v2.0.0 baseline — always register.
create_libstats_gtest(test_work_stealing_pool test_work_stealing_pool.cpp)
set(LIBSTATS_HAS_WORK_STEALING_POOL_TEST TRUE)

# =======================================================================
# LEVEL 3 TESTS: Advanced Infrastructure
# =======================================================================
# Test advanced infrastructure components that depend on Level 0-2

# Performance Infrastructure

# parallel_execution.h - C++20 parallel execution with automatic fallback
create_libstats_gtest(test_parallel_execution_integration test_parallel_execution_integration.cpp)
create_libstats_gtest(test_parallel_execution_comprehensive
                      test_parallel_execution_comprehensive.cpp)

# benchmark.h - Performance measurement utilities
create_libstats_test(test_benchmark_basic test_benchmark_basic.cpp)

# benchmark.h - Comprehensive benchmark infrastructure testing
create_libstats_gtest(test_benchmark test_benchmark.cpp)

# performance_dispatcher.h - Smart auto-dispatch system
create_libstats_gtest(test_performance_dispatcher test_performance_dispatcher.cpp)

# system_capabilities.cpp - System capability detection and integration
create_libstats_gtest(test_system_capabilities test_system_capabilities.cpp)

# Performance system initialization
create_libstats_gtest(test_performance_initialization test_performance_initialization.cpp)

# =======================================================================
# LEVEL 4 TESTS: Complete Distribution Framework
# =======================================================================
# Test the integrated distribution framework (distribution_base.h)

# distribution_base.h - Complete base class implementation (Tested indirectly through distribution
# implementations, but could add direct tests)

# =======================================================================
# LEVEL 5 TESTS: Concrete Distribution Implementations
# =======================================================================
# Test concrete distribution implementations using the complete framework

# gaussian.h - Gaussian/Normal distribution implementation (CONSOLIDATED)
create_libstats_test(test_gaussian_basic test_gaussian_basic.cpp)
create_libstats_gtest(test_gaussian_enhanced test_gaussian_enhanced.cpp)
create_libstats_gtest(test_gaussian_analysis test_gaussian_analysis.cpp)
create_libstats_gtest(test_poisson_analysis test_poisson_analysis.cpp)
create_libstats_gtest(test_distribution_analysis test_distribution_analysis.cpp)
create_libstats_gtest(test_batch_math_regressions test_batch_math_regressions.cpp)

# exponential.h - Exponential distribution implementation (CONSOLIDATED)
create_libstats_test(test_exponential_basic test_exponential_basic.cpp)
create_libstats_gtest(test_exponential_enhanced test_exponential_enhanced.cpp)

# discrete.h - Discrete uniform distribution implementation (CONSOLIDATED)
create_libstats_test(test_discrete_basic test_discrete_basic.cpp)
create_libstats_gtest(test_discrete_enhanced test_discrete_enhanced.cpp)

# Atomic parameters test - Test lock-free parameter access and invalidation for all distributions
create_libstats_gtest(test_atomic_parameters test_atomic_parameters.cpp)

# uniform.h - Uniform distribution implementation
create_libstats_test(test_uniform_basic test_uniform_basic.cpp)
create_libstats_gtest(test_uniform_enhanced test_uniform_enhanced.cpp)

# poisson.h - Poisson distribution implementation
create_libstats_test(test_poisson_basic test_poisson_basic.cpp)
create_libstats_gtest(test_poisson_enhanced test_poisson_enhanced.cpp)

# gamma.h - Gamma distribution implementation
create_libstats_test(test_gamma_basic test_gamma_basic.cpp)
create_libstats_gtest(test_gamma_enhanced test_gamma_enhanced.cpp)

# chi_squared.h - Chi-squared distribution (delegation wrapper over Gamma)
create_libstats_test(test_chi_squared_basic test_chi_squared_basic.cpp)
create_libstats_gtest(test_chi_squared_enhanced test_chi_squared_enhanced.cpp)

# student_t.h - Student's t distribution (full real-line, log-space SIMD)
create_libstats_test(test_student_t_basic test_student_t_basic.cpp)
create_libstats_gtest(test_student_t_enhanced test_student_t_enhanced.cpp)

# beta.h - Beta distribution (bounded support [0,1], two-log SIMD pipeline)
create_libstats_test(test_beta_basic test_beta_basic.cpp)
create_libstats_gtest(test_beta_enhanced test_beta_enhanced.cpp)

# lognormal.h - Log-normal distribution (log+exp pipeline, positive real-line)
create_libstats_test(test_lognormal_basic test_lognormal_basic.cpp)
create_libstats_gtest(test_lognormal_enhanced test_lognormal_enhanced.cpp)

# pareto.h - Pareto distribution (log-only pipeline, power-law tail)
create_libstats_test(test_pareto_basic test_pareto_basic.cpp)
create_libstats_gtest(test_pareto_enhanced test_pareto_enhanced.cpp)

# weibull.h - Weibull distribution (log+exp pipeline, reliability engineering)
create_libstats_test(test_weibull_basic test_weibull_basic.cpp)
create_libstats_gtest(test_weibull_enhanced test_weibull_enhanced.cpp)

# rayleigh.h - Rayleigh distribution (x² pipeline, signal processing)
create_libstats_test(test_rayleigh_basic test_rayleigh_basic.cpp)
create_libstats_gtest(test_rayleigh_enhanced test_rayleigh_enhanced.cpp)

# von_mises.h - Von Mises distribution (circular, Bessel functions; PARALLEL preferred)
create_libstats_test(test_von_mises_basic test_von_mises_basic.cpp)
create_libstats_gtest(test_von_mises_enhanced test_von_mises_enhanced.cpp)

# binomial.h - Binomial distribution (discrete, lgamma log-space PMF, beta_i CDF)
create_libstats_test(test_binomial_basic test_binomial_basic.cpp)
create_libstats_gtest(test_binomial_enhanced test_binomial_enhanced.cpp)

# negative_binomial.h - Negative Binomial distribution (discrete, Newton-Raphson MLE)
create_libstats_test(test_negative_binomial_basic test_negative_binomial_basic.cpp)
create_libstats_gtest(test_negative_binomial_enhanced test_negative_binomial_enhanced.cpp)

# geometric.h - Geometric distribution (discrete, delegation wrapper over NegBinomial(r=1))
create_libstats_test(test_geometric_basic test_geometric_basic.cpp)
create_libstats_gtest(test_geometric_enhanced test_geometric_enhanced.cpp)

# laplace.h - Laplace distribution (double-exponential, standalone, median/MAD MLE)
create_libstats_test(test_laplace_basic test_laplace_basic.cpp)
create_libstats_gtest(test_laplace_enhanced test_laplace_enhanced.cpp)

# cauchy.h - Cauchy distribution (delegation wrapper over StudentT(nu=1), Fisher-scoring MLE)
create_libstats_test(test_cauchy_basic test_cauchy_basic.cpp)
create_libstats_gtest(test_cauchy_enhanced test_cauchy_enhanced.cpp)

# =======================================================================
# LEVEL 6 TESTS: Integration and cross-distribution workflow
# =======================================================================
# End-to-end API interoperability (factory, fit, moments, survival, batch, KS test, bootstrap,
# NaN/±inf boundary values across all 19 distributions)
create_libstats_gtest(test_integration_workflow test_integration_workflow.cpp)

# =======================================================================
# ADDITIONAL TESTS: Cross-cutting concerns and integration
# =======================================================================
# These test cross-cutting functionality and integration scenarios

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_copy_move_fix.cpp)
    create_libstats_test(test_copy_move_fix test_copy_move_fix.cpp)
    set(HAS_COPY_MOVE_FIX TRUE)
else()
    set(HAS_COPY_MOVE_FIX FALSE)
endif()

# test_new_constants removed - functionality is covered by test_constants --platform --simd

# Dynamic library linking tests
create_libstats_test_dynamic(test_dynamic_linking test_dynamic_linking.cpp)

# Dynamic versions of all _simple tests Note: SIMD tests consolidated into test_simd_comprehensive
create_libstats_test_dynamic(test_gaussian_basic_dynamic test_gaussian_basic.cpp)
create_libstats_test_dynamic(test_exponential_basic_dynamic test_exponential_basic.cpp)

# =======================================================================
# TEST LABELS
# =======================================================================
# Tests containing timing-sensitive assertions (speedup thresholds, overhead ratios) are labelled
# "timing". These assertions can fail under CPU contention when multiple tests run in parallel. All
# tests are correct regardless of label; the label only affects whether timing ASSERTIONS are
# reliable.
#
# Usage: ctest -j8             # All tests; timing assertions may be flaky under load ctest -j8 -LE
# timing  # Correctness only; excludes tests with timing assertions ctest -j1 -L timing   # Timing
# validation; run serially for accurate measurement
#
# benchmark_simd_all is labelled separately as it is never part of the standard correctness suite.
# GTest-based tests only exist when GTest is available. set_tests_properties is a hard CMake error
# on non-existent tests, so guard this block -- unlike DEPENDS which tolerates missing targets.
if(GTEST_FOUND)
    set_tests_properties(
        test_math_comprehensive # vectorized math speedup assertions
        test_gaussian_enhanced
        test_chi_squared_enhanced
        test_student_t_enhanced
        test_beta_enhanced
        test_lognormal_enhanced
        test_pareto_enhanced
        test_weibull_enhanced
        test_rayleigh_enhanced
        test_von_mises_enhanced
        test_binomial_enhanced
        test_negative_binomial_enhanced
        test_performance_dispatcher
        test_system_capabilities
        PROPERTIES LABELS "timing")
    set_tests_properties(
        test_exponential_enhanced
        test_uniform_enhanced
        test_poisson_enhanced
        test_discrete_enhanced
        test_gamma_enhanced
        test_geometric_enhanced
        test_laplace_enhanced
        test_cauchy_enhanced
        PROPERTIES LABELS "timing")
endif()
set_tests_properties(benchmark_simd_all PROPERTIES LABELS "benchmark")

# Test target for running all tests (organized by dependency hierarchy) This target runs ALL tests
# including stress/diagnostic tests
add_custom_target(
    run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS # Level 0: Foundational tests (most critical)
            test_constants
            test_cpu_detection
            test_simd_comprehensive
            test_platform_optimizations
            test_simd_policy
            test_simd_neon_exp_accuracy
            test_simd_neon_log_accuracy
            test_simd_neon_cos_accuracy
            # Level 1: Core infrastructure tests
            test_error_handling_comprehensive
            test_numerical_safety
            test_math_comprehensive
            # Level 2: Statistical infrastructure
            test_goodness_of_fit
            test_gaussian_analysis
            test_poisson_analysis
            test_distribution_analysis
            test_batch_math_regressions
            # Level 3: Parallel infrastructure tests
            test_thread_pool
            $<$<BOOL:${LIBSTATS_HAS_WORK_STEALING_POOL_TEST}>:test_work_stealing_pool>
            test_parallel_execution_integration
            test_benchmark_basic
            # Level 4: Performance framework tests
            test_performance_dispatcher
            test_system_capabilities
            test_performance_initialization
            # Level 5: Basic distribution tests
            test_gaussian_basic
            test_exponential_basic
            test_uniform_basic
            test_poisson_basic
            test_discrete_basic
            test_gamma_basic
            test_chi_squared_basic
            test_student_t_basic
            test_beta_basic
            test_lognormal_basic
            test_pareto_basic
            test_weibull_basic
            test_rayleigh_basic
            test_von_mises_basic
            test_binomial_basic
            test_negative_binomial_basic
            test_atomic_parameters
            # Level 6: Enhanced distribution tests
            test_gaussian_enhanced
            test_exponential_enhanced
            test_uniform_enhanced
            test_poisson_enhanced
            test_discrete_enhanced
            test_gamma_enhanced
            test_student_t_enhanced
            test_beta_enhanced
            test_chi_squared_enhanced
            test_lognormal_enhanced
            test_pareto_enhanced
            test_weibull_enhanced
            test_rayleigh_enhanced
            test_von_mises_enhanced
            test_binomial_enhanced
            test_negative_binomial_enhanced
            test_geometric_basic
            test_geometric_enhanced
            test_laplace_basic
            test_laplace_enhanced
            test_cauchy_basic
            test_cauchy_enhanced
            # Level 6: Integration tests
            test_dynamic_linking
            test_integration_workflow
            # On-demand tests (stress, diagnostic, redundant)
            test_parallel_execution_comprehensive
            test_benchmark
            test_gaussian_basic_dynamic
            test_exponential_basic_dynamic
            $<$<BOOL:${HAS_COPY_MOVE_FIX}>:test_copy_move_fix>
            $<$<BOOL:${HAS_PARALLEL_COMPILATION}>:test_parallel_compilation>
    COMMENT "Running ALL libstats tests in dependency order (Levels 0-7 + on-demand)")

# Separate target for standard test suite (correctness only, parallel-safe) Excludes: on-demand
# tests, timing-labelled tests, benchmarks Use 'make run_tests_timing' to validate performance
# assertions on a quiet machine.
add_custom_target(
    run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -LE "timing\\|benchmark" -E
            "stress\\|comprehensive\\|dynamic\\|_fix\\|parallel_compilation\\|^test_benchmark$"
    COMMENT "Running standard libstats test suite (correctness only, timing tests excluded)")

# Timing validation target — runs serially for accurate measurement
add_custom_target(
    run_tests_timing
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -j1 -L timing
    COMMENT "Running timing-sensitive tests serially (requires a quiet machine)")

# Phase 3B/B3: propagate to the top-level scope for the "libstats Configuration Summary" print
# (CMakeLists.txt, "GTest: Available"/"Not found"), which reads GTEST_FOUND after this
# add_subdirectory(tests) call returns. Normal (non-cache) variables set in a subdirectory do not
# escape it automatically -- this PARENT_SCOPE propagation is required for that top-level message to
# stay correct; it has no other behavioral effect.
set(GTEST_FOUND
    ${GTEST_FOUND}
    PARENT_SCOPE)
