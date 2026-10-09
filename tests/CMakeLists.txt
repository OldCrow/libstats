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
        GIT_SHALLOW TRUE
        SYSTEM)
    # Prevent GTest from overriding the project's CRT choice on Windows.
    set(gtest_force_shared_crt
        ON
        CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    # GoogleTest 1.17.0 builds its own sources with -WX under an MSVC-style driver, and its
    # char8_t printer trips -Wcharacter-conversion, new in Clang 21. Only clang-cl meets both.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
       AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC"
       AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 21)
        foreach(_libstats_gtest_target gtest gtest_main gmock gmock_main)
            if(TARGET ${_libstats_gtest_target})
                target_compile_options(${_libstats_gtest_target}
                                       PRIVATE -Wno-character-conversion)
            endif()
        endforeach()
    endif()
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
        ${TEST_NAME} PRIVATE ${LIBSTATS_BUILD_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src
                             ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    libstats_apply_warnings(${TEST_NAME}) # Phase 3B/B5 -- single insertion point for every test
                                          # target

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
# On Windows, copy stats.dll next to the dynamic test EXEs so runtime loading does not depend on
# PATH. Combined with per-config output directories above, this avoids both STATUS_DLL_NOT_FOUND and
# stale Debug/Release binary collisions.
function(create_libstats_test_dynamic TEST_NAME SOURCE_FILE)
    add_executable(${TEST_NAME} ${SOURCE_FILE})

    # Link against shared library instead of static
    target_link_libraries(${TEST_NAME} PRIVATE libstats_shared)

    # Apply common test settings
    configure_common_test_settings(${TEST_NAME})

    # Every dynamic test shares one output directory, and their parallel POST_BUILD copies of the
    # same DLL raced ("Permission denied"). The first dynamic test copies; the rest build after it.
    if(WIN32)
        get_property(_runtime_owner GLOBAL PROPERTY LIBSTATS_TEST_RUNTIME_OWNER)
        if(NOT _runtime_owner)
            set_property(GLOBAL PROPERTY LIBSTATS_TEST_RUNTIME_OWNER ${TEST_NAME})
            add_custom_command(
                TARGET ${TEST_NAME}
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:libstats_shared>
                        $<TARGET_FILE_DIR:${TEST_NAME}>
                COMMENT "Copying shared libstats runtime next to the dynamic tests")
        else()
            add_dependencies(${TEST_NAME} ${_runtime_owner})
        endif()
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

# #117 guard: the CPUID gates behind tier selection -- AVX-512 requires DQ and XCR0[7:5], AVX2
# requires FMA. Reads CPUID/XCR0 independently and requires the dispatcher to agree. Deliberately
# NOT added to the "timing" label block below -- same reasoning as test_bessel_tier: it carries no
# timing assertions, and a regression gate CI never runs guards nothing.
create_libstats_gtest(test_simd_dispatch_gates test_simd_dispatch_gates.cpp)

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

# #97 guards: which Bessel tier a TU compiles, and that both tiers agree on the
# negative axis. Deliberately NOT added to the "timing" label block below --
# these carry no timing assertions, and CI's correctness run excludes that
# label, so labelling them would silently stop them running where they matter
# most (the libstdc++ std::domain_error path is Linux-only).
create_libstats_gtest(test_bessel_tier test_bessel_tier.cpp)

# #95 guard: per-tier ULP gate for the x86 vector_cos_<tier> kernels
# (vector_sin_<tier> staged in behind LIBSTATS_TRIG_GATES_HAVE_SIN until it
# lands). Deliberately NOT added to the "timing" label block below -- same
# reasoning as test_bessel_tier above: this carries no timing assertions,
# and a regression gate CI never runs guards nothing.
create_libstats_gtest(test_trig_ulp_gates test_trig_ulp_gates.cpp)

# #105 guard: per-tier special-value gate for the x86 vector_log_<tier>
# kernels (NaN/-1/0/+inf/denorm_min in the vector body and the scalar tail,
# specials-first), plus the LogNormal batch NaN consequence through the
# public span API. Deliberately NOT added to the "timing" label block below
# -- same reasoning as test_trig_ulp_gates above: this carries no timing
# assertions, and a regression gate CI never runs guards nothing.
create_libstats_gtest(test_log_special_gates test_log_special_gates.cpp)

# #102 guard: batch-path NaN propagation across all 19 distributions
# (pdf/logpdf/cdf span overloads, FORCE_VECTORIZED, NaN in vector body and
# scalar tail). Deliberately NOT added to the "timing" label block below --
# same reasoning as test_log_special_gates above.
create_libstats_gtest(test_batch_nan_gates test_batch_nan_gates.cpp)

# #163 guard: copy/move construction and assignment across all 27 distributions must carry the
# source's parameters and cache, never a validity flag without the cached values behind it.
create_libstats_gtest(test_copy_move_cache test_copy_move_cache.cpp)

# #161/#164/#165 guard: pdf/logpdf at the support boundary (x = 0 by shape, x = +-inf,
# out-of-support counts) take their limits, scalar and batch, under every forced strategy.
create_libstats_gtest(test_support_boundary_gates test_support_boundary_gates.cpp)

# #158 guard: the probit (inverse_normal_cdf, Gaussian and LogNormal quantiles) finite and within
# budget against mpmath from p = 1e-300 to the centre; the erf_inv(2p - 1) form returned -inf.
create_libstats_gtest(test_probit_accuracy test_probit_accuracy.cpp)

# #166 guard: CDFs on gamma_p/gamma_q/beta_i at large shape within budget of mpmath, scalar and
# batch; the 1e-8 series stop and the cancelling gamma prefactor left 1e-9 to 8e-6.
create_libstats_gtest(test_incomplete_gamma_beta_accuracy test_incomplete_gamma_beta_accuracy.cpp)

# #160 guard: Gamma, Erlang and ChiSquared quantiles and the detail Gamma inverses within budget of
# mpmath from p = 1e-300 to 1 - 1e-15; the linear-CDF Newton solver was 5e139 relative off.
create_libstats_gtest(test_gamma_quantile_accuracy test_gamma_quantile_accuracy.cpp)

# #137 guard: Beta quantiles and detail::inverse_beta_i within the accuracy law of mpmath from
# the smallest subnormal p to 1 - 1e-15, both tails; the absolute-1e-8 Newton solver returned
# 1e-8 for every p below it and 0.9999999999 for Beta(25, 1000).Q(6.57e-5).
create_libstats_gtest(test_beta_quantile_accuracy test_beta_quantile_accuracy.cpp)

# #159 guard: Student-t quantile, CDF, pdf and logpdf in the deep tail, at large nu and past
# |t| ~ 1e154, within the accuracy law of mpmath; scalar and every batch strategy.
create_libstats_gtest(test_student_t_tails test_student_t_tails.cpp)

# Guard: lbeta and the normalisers built on it (Student-t, Beta, Fisher-F) at large arguments,
# scalar and every batch strategy; p-values on the small tail, not 1 - CDF.
create_libstats_gtest(test_log_beta_and_pvalues test_log_beta_and_pvalues.cpp)

# Guard: sweep-found tail and tie defects (Cauchy quantile, Discrete and NegativeBinomial ties,
# Poisson scalar pmf, log1p/expm1 forms), scalar and, for Pareto's CDF, every batch strategy.
create_libstats_gtest(test_tail_and_tie_accuracy test_tail_and_tie_accuracy.cpp)

# Guard: Gamma-family densities at large shape (Stirling-form prefactor), scalar and every batch
# strategy; ChiSquared, Erlang and InverseGamma through their Gamma delegate.
create_libstats_gtest(test_gamma_density_accuracy test_gamma_density_accuracy.cpp)

# VectorOps::vector_log1p (generic over each tier's vector_log): accuracy against std::log1p,
# the IEEE edges, in-place calls and block-boundary sizes.
create_libstats_gtest(test_vector_log1p test_vector_log1p.cpp)

# VectorOps::vector_expm1 (Taylor below |x| = 1/2, vector_exp less 1 above): accuracy against
# std::expm1, the IEEE edges, in-place calls and block-boundary sizes.
create_libstats_gtest(test_vector_expm1 test_vector_expm1.cpp)

# Student-t (large nu) and Beta (large shape) batch densities on vector_log1p, every strategy,
# against mpmath.
create_libstats_gtest(test_log1p_density_batches test_log1p_density_batches.cpp)

# #172 guard: Poisson, Binomial, NegativeBinomial and Geometric log-pmfs at counts 1e3 to 1e7 within
# the conditioning budget of mpmath, scalar and every batch strategy.
create_libstats_gtest(test_discrete_pmf_accuracy test_discrete_pmf_accuracy.cpp)

# Guard: scalar methods racing a setter return one parameter state's value, never a mix or NaN
# (Binomial pmf/CDF, Poisson pmf, Gamma quantile), and one batch call returns one state under every
# strategy (#175: Binomial, NegativeBinomial, von Mises). Correctness, not timing: it asserts values
# only.
create_libstats_gtest(test_snapshot_consistency test_snapshot_consistency.cpp)

# #175/#176 guard: forced PARALLEL beats forced VECTORIZED at n = 1M where the parallel path took a
# lock per element or ran serially. Timing assertions: in the "timing" label block below.
create_libstats_gtest(test_parallel_batch_gates test_parallel_batch_gates.cpp)

# Gates from the 2026-10-06 lock and parallel-path audit: Poisson getMedian's recursive shared
# lock, Uniform's unlocked copy-assignment, the Uniform/Discrete bound setters' TOCTOU, the
# delegating setters' order, parallelTransform's repeated chunks, PARALLEL below the fork threshold,
# self-comparison. Correctness, not timing: it asserts values and progress only.
create_libstats_gtest(test_concurrency_gates test_concurrency_gates.cpp)

# Contract gates from the 2026-10-07 architecture review and defect hunt: atomic getters after
# assignment, stream round trips, NaN quantiles, fit validation, survival complements, tiny-kappa
# von Mises, Geometric closed-form CDF, scale ranges, quantile ties, signgam (#173). Correctness,
# not timing: deliberately NOT in the "timing" label block below.
create_libstats_gtest(test_contract_gates test_contract_gates.cpp)
# AR D19: no `inline` member definitions in src/*.cpp (Poisson's overrides, the #189 class).
add_test(NAME check_inline_member_definitions
         COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${PROJECT_SOURCE_DIR} -P
                 ${CMAKE_CURRENT_SOURCE_DIR}/check_inline_member_definitions.cmake)

# R9 stress test: every distribution's readers raced against writer programs (setters, fit, stream,
# assignment), two-writer rounds, and a strategy differential. Correctness, not timing: it asserts
# values and progress only. LIBSTATS_STRESS_SCALE scales its rounds (lower it under TSan).
create_libstats_gtest(test_concurrency_stress test_concurrency_stress.cpp)
# Its templated rows exceed COFF's 2^16 sections: MSVC stops with C1128 without /bigobj (clang-cl
# switches to the big format by itself).
if(MSVC)
    target_compile_options(test_concurrency_stress PRIVATE /bigobj)
endif()
# Its racing threads need the cores to themselves: under ctest --parallel on a 3-vCPU runner they
# starved and tripped the 120 s watchdog. Same for the gates, which race the same way.
foreach(_t test_concurrency_stress test_concurrency_gates)
    if(TEST ${_t})
        set_tests_properties(${_t} PROPERTIES RUN_SERIAL TRUE)
    endif()
endforeach()

# Oracle-free identities over all 27 distributions (2026-10-07 defect hunt, proposal 1): CDF range
# and monotonicity, Q(F(x)) and F(Q(p)) round trips, discrete quantile minimality, pdf against
# dF/dx, logpdf against log(pdf), scalar against vectorised batch, cross-class identities.
# Correctness, not timing: it asserts values only, so it must stay out of the "timing" label.
create_libstats_gtest(test_distribution_identities test_distribution_identities.cpp)

# Stream round trip at awkward parameter values and fit on degenerate data (defect hunt, proposal
# 3). Correctness, not timing.
create_libstats_gtest(test_io_fit_fuzz test_io_fit_fuzz.cpp)

# #51 guard: absolute-error accuracy gate for VonMisesDistribution::
# getCumulativeProbability against a direct-quadrature oracle (independent of
# the Bessel-series/trapezoidal implementation). Deliberately NOT added to
# the "timing" label block below -- same reasoning as test_trig_ulp_gates
# above: this carries no timing assertions, and a regression gate CI never
# runs guards nothing.
create_libstats_gtest(test_vonmises_cdf_accuracy test_vonmises_cdf_accuracy.cpp)

# Guard: von Mises CDF in both deep tails (relative, not absolute, accuracy) and the quantile to
# the accuracy law, including the left end of the support; scalar and every batch strategy.
create_libstats_gtest(test_von_mises_tails test_von_mises_tails.cpp)

# #49 guard: relative-error accuracy gate for LogNormalDistribution::
# getCumulativeProbability (scalar + batch span overload) against a
# closed-form erfc oracle. Deliberately NOT added to the "timing" label
# block below -- same reasoning as test_vonmises_cdf_accuracy above: this
# carries no timing assertions, and a regression gate CI never runs guards
# nothing.
create_libstats_gtest(test_lognormal_cdf_accuracy test_lognormal_cdf_accuracy.cpp)

# #49-pattern guard for Gaussian: GaussianDistribution's CDF paths never
# routed through detail::normal_cdf and reproduced the same erf cancellation
# bug; same gate structure as test_lognormal_cdf_accuracy above, closed-form
# erfc oracle. Deliberately NOT added to the "timing" label block below --
# same reasoning: this carries no timing assertions, and a regression gate
# CI never runs guards nothing.
create_libstats_gtest(test_gaussian_cdf_accuracy test_gaussian_cdf_accuracy.cpp)

# #118 guard: ParallelUtils::parallelFor propagates a chunk's exception, and
# only after every chunk has finished (the naive wait()->get() fix unwinds the
# caller while siblings still hold references into its frame). Deliberately NOT
# added to the "timing" label block below -- the sleeps here are ordering
# scaffolding, not speedup assertions, and a regression gate CI never runs
# guards nothing.
create_libstats_gtest(test_parallel_exception_propagation test_parallel_exception_propagation.cpp)

# binomial.h - Binomial distribution (discrete, lgamma log-space PMF, beta_i CDF)
create_libstats_test(test_binomial_basic test_binomial_basic.cpp)
create_libstats_gtest(test_binomial_enhanced test_binomial_enhanced.cpp)

# negative_binomial.h - Negative Binomial distribution (discrete, Newton-Raphson MLE)
create_libstats_test(test_negative_binomial_basic test_negative_binomial_basic.cpp)
create_libstats_gtest(test_negative_binomial_enhanced test_negative_binomial_enhanced.cpp)

# geometric.h - Geometric distribution (discrete, delegation wrapper over NegBinomial(r=1))
create_libstats_test(test_geometric_basic test_geometric_basic.cpp)
create_libstats_gtest(test_geometric_enhanced test_geometric_enhanced.cpp)

# #116 guard: NegativeBinomial/Geometric getQuantile search bound past INT_MAX.
# A separate binary rather than cases in test_negative_binomial_enhanced /
# test_geometric_enhanced because BOTH of those carry the "timing" label below,
# and the correctness run is -LE "timing|benchmark" -- a guard added there would
# compile and never run. Deliberately NOT added to the "timing" label block:
# same reasoning as test_bessel_tier and the three CDF accuracy gates above.
create_libstats_gtest(test_discrete_quantile_bounds test_discrete_quantile_bounds.cpp)

# laplace.h - Laplace distribution (double-exponential, standalone, median/MAD MLE)
create_libstats_test(test_laplace_basic test_laplace_basic.cpp)
create_libstats_gtest(test_laplace_enhanced test_laplace_enhanced.cpp)

# cauchy.h - Cauchy distribution (delegation wrapper over StudentT(nu=1), Fisher-scoring MLE)
create_libstats_test(test_cauchy_basic test_cauchy_basic.cpp)
create_libstats_gtest(test_cauchy_enhanced test_cauchy_enhanced.cpp)

# half_normal.h - Half-Normal distribution (|N(0,sigma)|, erf CDF pipeline) (#57)
# Deliberately NOT in the "timing" label block: the enhanced binary carries the
# Born-compliant edge contracts (#103/#104) and mpmath accuracy gates but no
# speedup assertions, and CI's correctness run is -LE "timing|benchmark".
create_libstats_test(test_half_normal_basic test_half_normal_basic.cpp)
create_libstats_gtest(test_half_normal_enhanced test_half_normal_enhanced.cpp)

# truncated_normal.h - Truncated Normal (regime-split erf/erfc normalization) (#57)
# Same reasoning as test_half_normal_*: no speedup assertions, so NOT
# timing-labelled — the same-tail cancellation gates must run in CI's
# correctness pass.
create_libstats_test(test_truncated_normal_basic test_truncated_normal_basic.cpp)
create_libstats_gtest(test_truncated_normal_enhanced test_truncated_normal_enhanced.cpp)

# bernoulli.h - Bernoulli distribution (discrete, delegation wrapper over Binomial(n=1))
create_libstats_test(test_bernoulli_basic test_bernoulli_basic.cpp)
create_libstats_gtest(test_bernoulli_enhanced test_bernoulli_enhanced.cpp)

# erlang.h - Erlang distribution (delegation wrapper over Gamma(k, lambda), integer shape)
create_libstats_test(test_erlang_basic test_erlang_basic.cpp)
create_libstats_gtest(test_erlang_enhanced test_erlang_enhanced.cpp)

# fisher_f.h - F (Fisher-Snedecor) distribution (Beta(d1/2,d2/2) transform wrapper)
create_libstats_test(test_fisher_f_basic test_fisher_f_basic.cpp)
create_libstats_gtest(test_fisher_f_enhanced test_fisher_f_enhanced.cpp)

# inverse_gamma.h - Inverse Gamma distribution (reciprocal-transform wrapper over Gamma)
create_libstats_test(test_inverse_gamma_basic test_inverse_gamma_basic.cpp)
create_libstats_gtest(test_inverse_gamma_enhanced test_inverse_gamma_enhanced.cpp)

# logistic.h - Logistic distribution (sigmoid CDF, symmetric vector_exp pipeline,
# conditional-MLE scale via safeguarded Newton).
# Deliberately NOT added to the "timing" label block below: the enhanced binary
# carries no speedup assertion, only a FORCE_VECTORIZED-vs-FORCE_SCALAR equality
# check, so it must run in the correctness suite (-LE "timing|benchmark").
create_libstats_test(test_logistic_basic test_logistic_basic.cpp)
create_libstats_gtest(test_logistic_enhanced test_logistic_enhanced.cpp)

# gumbel.h - Gumbel distribution (Type I extreme value, max-stable / gumbel_r only;
# double-exponential CDF, method-of-moments fit). Same no-timing-label reasoning.
create_libstats_test(test_gumbel_basic test_gumbel_basic.cpp)
create_libstats_gtest(test_gumbel_enhanced test_gumbel_enhanced.cpp)

# =======================================================================
# LEVEL 6 TESTS: Integration and cross-distribution workflow
# =======================================================================
# End-to-end API interoperability (factory, fit, moments, survival, batch, KS test, bootstrap,
# NaN/±inf boundary values across all 27 distributions)
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
        test_lognormal_enhanced
        test_pareto_enhanced
        test_weibull_enhanced
        test_rayleigh_enhanced
        test_von_mises_enhanced
        test_binomial_enhanced
        test_negative_binomial_enhanced
        test_system_capabilities
        test_parallel_batch_gates # #175/#176: PARALLEL beats VECTORIZED at 1M
        # test_chi_squared_enhanced, test_student_t_enhanced, test_beta_enhanced and
        # test_performance_dispatcher left this list (AR T1, 2026-10-07): none has a clock, duration
        # or speedup assertion, so the label only kept their gates (#103 among them) out of CI's
        # correctness run (-LE "timing|benchmark"), the #97 trap.
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
        # v2.4.0 note: NO new enhanced binary carries the timing label. None of
        # the eight new distributions' enhanced tests asserts a speedup ratio
        # (Bernoulli's VectorizedSpeedup only PRINTS the ratio and asserts
        # parallel==scalar values), and the #103/#104 contract gates live in
        # these binaries — a timing label would exclude them from CI's
        # correctness run (-LE "timing|benchmark"), the #97 guard-that-never-
        # runs trap. test_bernoulli_enhanced/test_erlang_enhanced were labelled
        # at first landing and unlabelled here for exactly that reason.
        PROPERTIES LABELS "timing")

    # test_benchmark is a benchmark, not a correctness test. It used to be kept
    # out of run_tests (now run_tests_correctness) by name, via "^test_benchmark$" in that target's -E
    # regex -- and that literal $ is exactly what made the target unusable
    # under the Ninja generator, which requires it written as $$ and treats the
    # malformed line as invalidating the whole build.ninja. Labelling it
    # removes the $ and is the more consistent treatment anyway:
    # benchmark_simd_all is already excluded this way. The name-anchored form
    # was load-bearing (test_benchmark_basic IS a correctness test and must
    # stay in the suite), so it could not simply be dropped.
    set_tests_properties(test_benchmark PROPERTIES LABELS "benchmark")
endif()
set_tests_properties(benchmark_simd_all PROPERTIES LABELS "benchmark")

# Test target for running all tests (organized by dependency hierarchy) This target runs ALL tests
# including stress/diagnostic tests
add_custom_target(
    run_all_tests
    # -C $<CONFIG>: multi-config generators (VS, Xcode) register tests per configuration;
    # without it CTest sees no tests and label filters are inert (review 2026-08-21).
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure
    DEPENDS # Level 0: Foundational tests (most critical)
            test_constants
            test_cpu_detection
            test_simd_comprehensive
            test_platform_optimizations
            test_simd_policy
            test_simd_dispatch_gates
            test_simd_neon_exp_accuracy
            test_simd_neon_log_accuracy
            test_simd_neon_cos_accuracy
            test_bessel_tier
            test_trig_ulp_gates
            test_vonmises_cdf_accuracy
            test_lognormal_cdf_accuracy
            test_gaussian_cdf_accuracy
            test_log_special_gates
            test_batch_nan_gates
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
            test_parallel_exception_propagation
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
            test_discrete_quantile_bounds
            test_laplace_basic
            test_laplace_enhanced
            test_cauchy_basic
            test_cauchy_enhanced
            test_bernoulli_basic
            test_bernoulli_enhanced
            test_erlang_basic
            test_erlang_enhanced
            test_logistic_basic
            test_logistic_enhanced
            test_gumbel_basic
            test_gumbel_enhanced
            test_half_normal_basic
            test_half_normal_enhanced
            test_truncated_normal_basic
            test_truncated_normal_enhanced
            test_fisher_f_basic
            test_fisher_f_enhanced
            test_inverse_gamma_basic
            test_inverse_gamma_enhanced
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
# Pipes here are PLAIN, not backslash-escaped. ctest -LE/-E take CMake regexes,
# where | is alternation and \| is an escaped literal pipe -- so the previous
# "timing\|benchmark" matched only the literal text "timing|benchmark", which no
# label or test name equals. Both filters were therefore inert and this target
# ran the entire suite (72 tests) rather than the 41 it describes. Verified by
# comparing `ctest -N` counts for the escaped and plain forms.
# Named run_tests until v2.4.2: the Visual Studio generator adds its own RUN_TESTS target, and the
# .slnx solution a fresh VS 2026 configure emits compares project names case-insensitively, so
# MSBuild rejected the whole solution ("Project name 'run_tests' already exists").
add_custom_target(
    run_tests_correctness
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -LE "timing|benchmark" -E
            "stress|comprehensive|dynamic|_fix|parallel_compilation"
    COMMENT "Running standard libstats test suite (correctness only, timing tests excluded)"
    # VERBATIM quotes the regexes for each generator's shell: without it the Visual Studio
    # generator wrote them bare into a cmd.exe rule, which read each | as a pipe (exit 255).
    VERBATIM)

# Timing validation target — runs serially for accurate measurement
add_custom_target(
    run_tests_timing
    COMMAND ${CMAKE_CTEST_COMMAND} -C $<CONFIG> --output-on-failure -j1 -L timing --no-tests=error
    COMMENT "Running timing-sensitive tests serially (requires a quiet machine)")

# Phase 3B/B3: propagate to the top-level scope for the "libstats Configuration Summary" print
# (CMakeLists.txt, "GTest: Available"/"Not found"), which reads GTEST_FOUND after this
# add_subdirectory(tests) call returns. Normal (non-cache) variables set in a subdirectory do not
# escape it automatically -- this PARENT_SCOPE propagation is required for that top-level message to
# stay correct; it has no other behavioral effect.
set(GTEST_FOUND
    ${GTEST_FOUND}
    PARENT_SCOPE)
