# SIMDDetection.cmake - Robust SIMD feature detection for cross-platform builds
# 
# This module provides comprehensive SIMD detection that checks both:
# 1. Compiler support for generating SIMD instructions
# 2. Runtime CPU capability to execute SIMD instructions
#
# It supports cross-compilation by allowing override of runtime detection
# through environment variables or CMake cache variables.

include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Global variables to track SIMD availability
set(LIBSTATS_SIMD_SOURCES "" CACHE INTERNAL "List of SIMD source files to compile")
set(LIBSTATS_SIMD_DEFINITIONS "" CACHE INTERNAL "List of SIMD compile definitions")

# Function to test runtime CPU feature support
function(test_runtime_cpu_feature FEATURE_NAME TEST_CODE RESULT_VAR)
    set(TEST_SOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_${FEATURE_NAME}_runtime.cpp")
    
    # Create test program
    file(WRITE "${TEST_SOURCE_FILE}" "
#include <iostream>
#include <exception>

${TEST_CODE}

int main() {
    try {
        if (test_${FEATURE_NAME}()) {
            std::cout << \"${FEATURE_NAME}: SUPPORTED\" << std::endl;
            return 0;
        } else {
            std::cout << \"${FEATURE_NAME}: NOT_SUPPORTED\" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cout << \"${FEATURE_NAME}: EXCEPTION (\" << e.what() << \")\" << std::endl;
        return 2;
    } catch (...) {
        std::cout << \"${FEATURE_NAME}: UNKNOWN_EXCEPTION\" << std::endl;
        return 3;
    }
}
")

    set(TEST_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/test_${FEATURE_NAME}_runtime")
    
    # Try to compile and run the test
    try_compile(COMPILE_RESULT
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${TEST_SOURCE_FILE}"
        CMAKE_FLAGS 
            "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        COMPILE_DEFINITIONS ${ARGN}  # Additional compile flags
        OUTPUT_VARIABLE COMPILE_OUTPUT
        COPY_FILE "${TEST_EXECUTABLE}"
    )
    
    if(COMPILE_RESULT)
        # Try to run the test
        execute_process(
            COMMAND "${TEST_EXECUTABLE}"
            RESULT_VARIABLE RUN_RESULT
            OUTPUT_VARIABLE RUN_OUTPUT
            ERROR_VARIABLE RUN_ERROR
            TIMEOUT 10  # 10 second timeout
        )
        
        if(RUN_RESULT EQUAL 0)
            set(${RESULT_VAR} TRUE PARENT_SCOPE)
            message(STATUS "Runtime ${FEATURE_NAME} test: PASSED")
        else()
            set(${RESULT_VAR} FALSE PARENT_SCOPE)
            if(RUN_RESULT EQUAL 1)
                message(STATUS "Runtime ${FEATURE_NAME} test: NOT SUPPORTED")
            elseif(RUN_RESULT EQUAL 2 OR RUN_RESULT EQUAL 3)
                message(STATUS "Runtime ${FEATURE_NAME} test: CRASHED (illegal instruction?)")
            else()
                message(STATUS "Runtime ${FEATURE_NAME} test: FAILED (exit code: ${RUN_RESULT})")
            endif()
        endif()
    else()
        set(${RESULT_VAR} FALSE PARENT_SCOPE)
        message(STATUS "Runtime ${FEATURE_NAME} test: COMPILE FAILED")
    endif()
    
    # Cleanup
    file(REMOVE "${TEST_SOURCE_FILE}")
    if(EXISTS "${TEST_EXECUTABLE}")
        file(REMOVE "${TEST_EXECUTABLE}")
    endif()
endfunction()

# Function to detect SSE2 support
function(detect_sse2_support)
    # Check compiler support
    check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
    
    if(COMPILER_SUPPORTS_SSE2)
        # Check runtime support
        test_runtime_cpu_feature("sse2" "
#include <emmintrin.h>
bool test_sse2() {
    __m128d a = _mm_set1_pd(1.0);
    __m128d b = _mm_set1_pd(2.0);
    __m128d c = _mm_add_pd(a, b);
    double result[2];
    _mm_store_pd(result, c);
    return (result[0] == 3.0 && result[1] == 3.0);
}
" RUNTIME_SUPPORTS_SSE2 "-msse2")
        
        if(RUNTIME_SUPPORTS_SSE2)
            set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 support available")
            list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp")
            list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1")
            message(STATUS "SIMD: SSE2 enabled (compiler + runtime)")
        else()
            set(LIBSTATS_HAS_SSE2 FALSE CACHE BOOL "SSE2 support not available at runtime")
            message(STATUS "SIMD: SSE2 disabled (runtime check failed)")
        endif()
    else()
        set(LIBSTATS_HAS_SSE2 FALSE CACHE BOOL "SSE2 compiler support not available")
        message(STATUS "SIMD: SSE2 disabled (compiler not supported)")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Function to detect AVX support
function(detect_avx_support)
    # Check compiler support
    check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
    
    if(COMPILER_SUPPORTS_AVX)
        # Check runtime support
        test_runtime_cpu_feature("avx" "
#include <immintrin.h>
bool test_avx() {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_add_pd(a, b);
    double result[4];
    _mm256_store_pd(result, c);
    return (result[0] == 3.0 && result[1] == 3.0 && result[2] == 3.0 && result[3] == 3.0);
}
" RUNTIME_SUPPORTS_AVX "-mavx")
        
        if(RUNTIME_SUPPORTS_AVX)
            set(LIBSTATS_HAS_AVX TRUE CACHE BOOL "AVX support available")
            list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx.cpp")
            list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX=1")
            message(STATUS "SIMD: AVX enabled (compiler + runtime)")
        else()
            set(LIBSTATS_HAS_AVX FALSE CACHE BOOL "AVX support not available at runtime")
            message(STATUS "SIMD: AVX disabled (runtime check failed)")
        endif()
    else()
        set(LIBSTATS_HAS_AVX FALSE CACHE BOOL "AVX compiler support not available")
        message(STATUS "SIMD: AVX disabled (compiler not supported)")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Function to detect AVX2 support
function(detect_avx2_support)
    # Check compiler support for both AVX2 and FMA
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    
    if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
        # Check runtime support
        test_runtime_cpu_feature("avx2" "
#include <immintrin.h>
bool test_avx2() {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_set1_pd(3.0);
    __m256d result = _mm256_fmadd_pd(a, b, c);  // a*b + c = 1*2 + 3 = 5
    double values[4];
    _mm256_store_pd(values, result);
    return (values[0] == 5.0 && values[1] == 5.0 && values[2] == 5.0 && values[3] == 5.0);
}
" RUNTIME_SUPPORTS_AVX2 "-mavx2" "-mfma")
        
        if(RUNTIME_SUPPORTS_AVX2)
            set(LIBSTATS_HAS_AVX2 TRUE CACHE BOOL "AVX2 support available")
            list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx2.cpp")
            list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX2=1")
            message(STATUS "SIMD: AVX2 enabled (compiler + runtime)")
        else()
            set(LIBSTATS_HAS_AVX2 FALSE CACHE BOOL "AVX2 support not available at runtime")
            message(STATUS "SIMD: AVX2 disabled (runtime check failed)")
        endif()
    else()
        set(LIBSTATS_HAS_AVX2 FALSE CACHE BOOL "AVX2 compiler support not available")
        message(STATUS "SIMD: AVX2 disabled (compiler not supported)")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Function to detect AVX-512 support
function(detect_avx512_support)
    # Check compiler support
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
    
    if(COMPILER_SUPPORTS_AVX512F)
        # Check runtime support
        test_runtime_cpu_feature("avx512" "
#include <immintrin.h>
bool test_avx512() {
    __m512d a = _mm512_set1_pd(1.0);
    __m512d b = _mm512_set1_pd(2.0);
    __m512d c = _mm512_add_pd(a, b);
    double result[8];
    _mm512_store_pd(result, c);
    for (int i = 0; i < 8; i++) {
        if (result[i] != 3.0) return false;
    }
    return true;
}
" RUNTIME_SUPPORTS_AVX512 "-mavx512f")
        
        if(RUNTIME_SUPPORTS_AVX512)
            set(LIBSTATS_HAS_AVX512 TRUE CACHE BOOL "AVX-512 support available")
            list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx512.cpp")
            list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX512=1")
            message(STATUS "SIMD: AVX-512 enabled (compiler + runtime)")
        else()
            set(LIBSTATS_HAS_AVX512 FALSE CACHE BOOL "AVX-512 support not available at runtime")
            message(STATUS "SIMD: AVX-512 disabled (runtime check failed)")
        endif()
    else()
        set(LIBSTATS_HAS_AVX512 FALSE CACHE BOOL "AVX-512 compiler support not available")
        message(STATUS "SIMD: AVX-512 disabled (compiler not supported)")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Function to detect NEON support (ARM)
function(detect_neon_support)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        # Check compiler support
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                # On AArch64, NEON is always available and no special flags are needed
                set(COMPILER_SUPPORTS_NEON TRUE)
                message(STATUS "SIMD: NEON available on AArch64 (no special flags needed)")
            else()
                # On 32-bit ARM, we need -mfpu=neon
                check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
            endif()
        else()
            set(COMPILER_SUPPORTS_NEON TRUE)  # Assume NEON is available on AArch64
        endif()
        
        if(COMPILER_SUPPORTS_NEON)
            # Check runtime support
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                # On AArch64, no special flags needed for runtime test
                test_runtime_cpu_feature("neon" "
#include <arm_neon.h>
bool test_neon() {
    float64x2_t a = vdupq_n_f64(1.0);
    float64x2_t b = vdupq_n_f64(2.0);
    float64x2_t c = vaddq_f64(a, b);
    double result[2];
    vst1q_f64(result, c);
    return (result[0] == 3.0 && result[1] == 3.0);
}
" RUNTIME_SUPPORTS_NEON)
            else()
                # On 32-bit ARM, use -mfpu=neon flag
                test_runtime_cpu_feature("neon" "
#include <arm_neon.h>
bool test_neon() {
    float64x2_t a = vdupq_n_f64(1.0);
    float64x2_t b = vdupq_n_f64(2.0);
    float64x2_t c = vaddq_f64(a, b);
    double result[2];
    vst1q_f64(result, c);
    return (result[0] == 3.0 && result[1] == 3.0);
}
" RUNTIME_SUPPORTS_NEON "-mfpu=neon")
            endif()
            
            if(RUNTIME_SUPPORTS_NEON)
                set(LIBSTATS_HAS_NEON TRUE CACHE BOOL "NEON support available")
                list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_neon.cpp")
                list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_NEON=1")
                message(STATUS "SIMD: NEON enabled (compiler + runtime)")
            else()
                set(LIBSTATS_HAS_NEON FALSE CACHE BOOL "NEON support not available at runtime")
                message(STATUS "SIMD: NEON disabled (runtime check failed)")
            endif()
        else()
            set(LIBSTATS_HAS_NEON FALSE CACHE BOOL "NEON compiler support not available")
            message(STATUS "SIMD: NEON disabled (compiler not supported)")
        endif()
    else()
        set(LIBSTATS_HAS_NEON FALSE CACHE BOOL "NEON not available on this architecture")
        message(STATUS "SIMD: NEON disabled (not ARM architecture)")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Cross-compilation support: allow environment variables to override runtime checks
function(apply_cross_compilation_overrides)
    # Allow environment variables to force enable/disable features
    if(DEFINED ENV{LIBSTATS_FORCE_SSE2})
        set(LIBSTATS_HAS_SSE2 $ENV{LIBSTATS_FORCE_SSE2} CACHE BOOL "SSE2 support (forced)" FORCE)
        message(STATUS "SIMD: SSE2 forced to ${LIBSTATS_HAS_SSE2} via environment")
    endif()
    
    if(DEFINED ENV{LIBSTATS_FORCE_AVX})
        set(LIBSTATS_HAS_AVX $ENV{LIBSTATS_FORCE_AVX} CACHE BOOL "AVX support (forced)" FORCE)
        message(STATUS "SIMD: AVX forced to ${LIBSTATS_HAS_AVX} via environment")
    endif()
    
    if(DEFINED ENV{LIBSTATS_FORCE_AVX2})
        set(LIBSTATS_HAS_AVX2 $ENV{LIBSTATS_FORCE_AVX2} CACHE BOOL "AVX2 support (forced)" FORCE)
        message(STATUS "SIMD: AVX2 forced to ${LIBSTATS_HAS_AVX2} via environment")
    endif()
    
    if(DEFINED ENV{LIBSTATS_FORCE_AVX512})
        set(LIBSTATS_HAS_AVX512 $ENV{LIBSTATS_FORCE_AVX512} CACHE BOOL "AVX-512 support (forced)" FORCE)
        message(STATUS "SIMD: AVX-512 forced to ${LIBSTATS_HAS_AVX512} via environment")
    endif()
    
    if(DEFINED ENV{LIBSTATS_FORCE_NEON})
        set(LIBSTATS_HAS_NEON $ENV{LIBSTATS_FORCE_NEON} CACHE BOOL "NEON support (forced)" FORCE)
        message(STATUS "SIMD: NEON forced to ${LIBSTATS_HAS_NEON} via environment")
    endif()
    
    # Update source lists based on forced settings
    set(LIBSTATS_SIMD_SOURCES "")
    set(LIBSTATS_SIMD_DEFINITIONS "")
    
    if(LIBSTATS_HAS_SSE2)
        list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp")
        list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1")
    endif()
    
    if(LIBSTATS_HAS_AVX)
        list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx.cpp")
        list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX=1")
    endif()
    
    if(LIBSTATS_HAS_AVX2)
        list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx2.cpp")
        list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX2=1")
    endif()
    
    if(LIBSTATS_HAS_AVX512)
        list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_avx512.cpp")
        list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_AVX512=1")
    endif()
    
    if(LIBSTATS_HAS_NEON)
        list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_neon.cpp")
        list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_NEON=1")
    endif()
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
endfunction()

# Main function to detect all SIMD features
function(detect_simd_features)
    message(STATUS "Detecting SIMD features...")
    
    # Initialize lists
    set(LIBSTATS_SIMD_SOURCES "" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "" CACHE INTERNAL "List of SIMD compile definitions")
    
    # Skip runtime checks if cross-compiling (unless explicitly requested)
    if(CMAKE_CROSSCOMPILING AND NOT LIBSTATS_ENABLE_RUNTIME_CHECKS)
        message(STATUS "Cross-compiling detected - skipping runtime SIMD checks")
        message(STATUS "Use LIBSTATS_ENABLE_RUNTIME_CHECKS=ON to force runtime checks")
        message(STATUS "Use environment variables LIBSTATS_FORCE_* to override SIMD settings")
        
        # Only do compiler checks
        check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
        if(COMPILER_SUPPORTS_SSE2)
            set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 compiler support")
            list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp")
            list(APPEND LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1")
            message(STATUS "SIMD: SSE2 enabled (compiler only - cross-compiling)")
        endif()
        
        # Apply any environment variable overrides
        apply_cross_compilation_overrides()
        return()
    endif()
    
    # Detect each SIMD feature with runtime checks
    detect_sse2_support()
    detect_avx_support()
    detect_avx2_support()
    detect_avx512_support()
    detect_neon_support()
    
    # Apply any environment variable overrides
    apply_cross_compilation_overrides()
    
    # Always include fallback implementation
    list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_fallback.cpp")
    list(APPEND LIBSTATS_SIMD_SOURCES "src/simd_dispatch.cpp")
    
    # Update global variables
    set(LIBSTATS_SIMD_SOURCES "${LIBSTATS_SIMD_SOURCES}" CACHE INTERNAL "List of SIMD source files to compile")
    set(LIBSTATS_SIMD_DEFINITIONS "${LIBSTATS_SIMD_DEFINITIONS}" CACHE INTERNAL "List of SIMD compile definitions")
    
    # Report summary
    message(STATUS "SIMD detection complete:")
    message(STATUS "  SSE2: ${LIBSTATS_HAS_SSE2}")
    message(STATUS "  AVX:  ${LIBSTATS_HAS_AVX}")
    message(STATUS "  AVX2: ${LIBSTATS_HAS_AVX2}")
    message(STATUS "  AVX-512: ${LIBSTATS_HAS_AVX512}")
    message(STATUS "  NEON: ${LIBSTATS_HAS_NEON}")
    message(STATUS "  Sources: ${LIBSTATS_SIMD_SOURCES}")
endfunction()

# Function to configure SIMD compilation for a target
function(configure_simd_target TARGET_NAME)
    # Add SIMD-specific compilation flags for each source file
    if(LIBSTATS_HAS_SSE2)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_sse2.cpp" 
            PROPERTIES COMPILE_FLAGS "-msse2")
    endif()
    
    if(LIBSTATS_HAS_AVX)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx.cpp" 
            PROPERTIES COMPILE_FLAGS "-mavx")
    endif()
    
    if(LIBSTATS_HAS_AVX2)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx2.cpp" 
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
    endif()
    
    if(LIBSTATS_HAS_AVX512)
        set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx512.cpp" 
            PROPERTIES COMPILE_FLAGS "-mavx512f")
    endif()
    
    if(LIBSTATS_HAS_NEON)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            # On AArch64, no special flags needed
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp" 
                PROPERTIES COMPILE_FLAGS "")
        else()
            # On 32-bit ARM, use -mfpu=neon flag
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp" 
                PROPERTIES COMPILE_FLAGS "-mfpu=neon")
        endif()
    endif()
    
    # Add compile definitions
    target_compile_definitions(${TARGET_NAME} PRIVATE ${LIBSTATS_SIMD_DEFINITIONS})
endfunction()
