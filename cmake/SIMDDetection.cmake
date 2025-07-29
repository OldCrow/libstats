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

# Function to test runtime CPU feature support
function(test_runtime_cpu_feature FEATURE_NAME TEST_CODE RESULT_VAR)
    set(TEST_SOURCE_FILE "${CMAKE_CURRENT_BINARY_DIR}/test_${FEATURE_NAME}_runtime.cpp")
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
    try_compile(COMPILE_RESULT
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${TEST_SOURCE_FILE}"
        CMAKE_FLAGS
            "-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
            "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        COMPILE_DEFINITIONS ${ARGN}
        OUTPUT_VARIABLE COMPILE_OUTPUT
        COPY_FILE "${TEST_EXECUTABLE}"
    )
    if(COMPILE_RESULT)
        execute_process(
            COMMAND "${TEST_EXECUTABLE}"
            RESULT_VARIABLE RUN_RESULT
            OUTPUT_VARIABLE RUN_OUTPUT
            ERROR_VARIABLE RUN_ERROR
            TIMEOUT 10
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
    file(REMOVE "${TEST_SOURCE_FILE}")
    if(EXISTS "${TEST_EXECUTABLE}")
        file(REMOVE "${TEST_EXECUTABLE}")
    endif()
endfunction()

# Cross-compilation support: allow environment variables to override runtime checks
function(apply_cross_compilation_overrides)
    # Read existing cache values and modify source lists based on forced settings
    get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
    get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
    
    if(DEFINED ENV{LIBSTATS_FORCE_SSE2} AND ENV{LIBSTATS_FORCE_SSE2})
        set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 support (forced)" FORCE)
        if(NOT "src/simd_sse2.cpp" IN_LIST _sources)
            list(APPEND _sources "src/simd_sse2.cpp")
        endif()
        if(NOT "LIBSTATS_HAS_SSE2=1" IN_LIST _definitions)
            list(APPEND _definitions "LIBSTATS_HAS_SSE2=1")
        endif()
        message(STATUS "SIMD: SSE2 forced to TRUE via environment")
    endif()
    if(DEFINED ENV{LIBSTATS_FORCE_AVX} AND ENV{LIBSTATS_FORCE_AVX})
        set(LIBSTATS_HAS_AVX TRUE CACHE BOOL "AVX support (forced)" FORCE)
        if(NOT "src/simd_avx.cpp" IN_LIST _sources)
            list(APPEND _sources "src/simd_avx.cpp")
        endif()
        if(NOT "LIBSTATS_HAS_AVX=1" IN_LIST _definitions)
            list(APPEND _definitions "LIBSTATS_HAS_AVX=1")
        endif()
        message(STATUS "SIMD: AVX forced to TRUE via environment")
    endif()
    if(DEFINED ENV{LIBSTATS_FORCE_AVX2} AND ENV{LIBSTATS_FORCE_AVX2})
        set(LIBSTATS_HAS_AVX2 TRUE CACHE BOOL "AVX2 support (forced)" FORCE)
        if(NOT "src/simd_avx2.cpp" IN_LIST _sources)
            list(APPEND _sources "src/simd_avx2.cpp")
        endif()
        if(NOT "LIBSTATS_HAS_AVX2=1" IN_LIST _definitions)
            list(APPEND _definitions "LIBSTATS_HAS_AVX2=1")
        endif()
        message(STATUS "SIMD: AVX2 forced to TRUE via environment")
    endif()
    if(DEFINED ENV{LIBSTATS_FORCE_AVX512} AND ENV{LIBSTATS_FORCE_AVX512})
        set(LIBSTATS_HAS_AVX512 TRUE CACHE BOOL "AVX-512 support (forced)" FORCE)
        if(NOT "src/simd_avx512.cpp" IN_LIST _sources)
            list(APPEND _sources "src/simd_avx512.cpp")
        endif()
        if(NOT "LIBSTATS_HAS_AVX512=1" IN_LIST _definitions)
            list(APPEND _definitions "LIBSTATS_HAS_AVX512=1")
        endif()
        message(STATUS "SIMD: AVX-512 forced to TRUE via environment")
    endif()
    if(DEFINED ENV{LIBSTATS_FORCE_NEON} AND ENV{LIBSTATS_FORCE_NEON})
        set(LIBSTATS_HAS_NEON TRUE CACHE BOOL "NEON support (forced)" FORCE)
        if(NOT "src/simd_neon.cpp" IN_LIST _sources)
            list(APPEND _sources "src/simd_neon.cpp")
        endif()
        if(NOT "LIBSTATS_HAS_NEON=1" IN_LIST _definitions)
            list(APPEND _definitions "LIBSTATS_HAS_NEON=1")
        endif()
        message(STATUS "SIMD: NEON forced to TRUE via environment")
    endif()

    # Update cache with modified lists
    set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
    set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
endfunction()

# Main function to detect all SIMD features
function(detect_simd_features)
    message(STATUS "Detecting SIMD features...")

    # Initialize all SIMD cache variables to FALSE first
    set(LIBSTATS_HAS_SSE2 FALSE CACHE BOOL "SSE2 support available" FORCE)
    set(LIBSTATS_HAS_AVX FALSE CACHE BOOL "AVX support available" FORCE)
    set(LIBSTATS_HAS_AVX2 FALSE CACHE BOOL "AVX2 support available" FORCE)
    set(LIBSTATS_HAS_AVX512 FALSE CACHE BOOL "AVX-512 support available" FORCE)
    set(LIBSTATS_HAS_NEON FALSE CACHE BOOL "NEON support available" FORCE)
    
    # Initialize empty lists in cache
    set(LIBSTATS_SIMD_SOURCES "" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
    set(LIBSTATS_SIMD_DEFINITIONS "" CACHE INTERNAL "List of SIMD compile definitions" FORCE)

    # Skip runtime checks if cross-compiling (unless explicitly requested)
    if(CMAKE_CROSSCOMPILING AND NOT LIBSTATS_ENABLE_RUNTIME_CHECKS)
        message(STATUS "Cross-compiling detected - skipping runtime SIMD checks")
        message(STATUS "Use LIBSTATS_ENABLE_RUNTIME_CHECKS=ON to force runtime checks")
        message(STATUS "Use environment variables LIBSTATS_FORCE_* to override SIMD settings")
        check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
        if(COMPILER_SUPPORTS_SSE2)
            set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 compiler support" FORCE)
            set(LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
            set(LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
            message(STATUS "SIMD: SSE2 enabled (compiler only - cross-compiling)")
        endif()
        apply_cross_compilation_overrides()
        # Always include fallback implementation
        get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
        list(APPEND _sources "src/simd_fallback.cpp" "src/simd_dispatch.cpp")
        set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
        
        get_property(_sse2 CACHE LIBSTATS_HAS_SSE2 PROPERTY VALUE)
        get_property(_avx CACHE LIBSTATS_HAS_AVX PROPERTY VALUE)
        get_property(_avx2 CACHE LIBSTATS_HAS_AVX2 PROPERTY VALUE)
        get_property(_avx512 CACHE LIBSTATS_HAS_AVX512 PROPERTY VALUE)
        get_property(_neon CACHE LIBSTATS_HAS_NEON PROPERTY VALUE)
        get_property(_final_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
        
        message(STATUS "SIMD detection complete:")
        message(STATUS "  SSE2: ${_sse2}")
        message(STATUS "  AVX:  ${_avx}")
        message(STATUS "  AVX2: ${_avx2}")
        message(STATUS "  AVX-512: ${_avx512}")
        message(STATUS "  NEON: ${_neon}")
        message(STATUS "  Sources: ${_final_sources}")
        return()
    endif()

    # Detect SSE2 support
    message(STATUS "SIMDDetection: CMAKE_CXX_COMPILER_ID='${CMAKE_CXX_COMPILER_ID}' CMAKE_CXX_COMPILER='${CMAKE_CXX_COMPILER}' MSVC='${MSVC}' WIN32='${WIN32}'")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 support available" FORCE)
            set(LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
            set(LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
            message(STATUS "SIMD: SSE2 enabled (MSVC/Clang-cl x64, default support)")
        else()
            message(STATUS "SIMD: SSE2 disabled (not x64)")
        endif()
    else()
        check_cxx_compiler_flag("-msse2" COMPILER_SUPPORTS_SSE2)
        if(COMPILER_SUPPORTS_SSE2)
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
                set(LIBSTATS_HAS_SSE2 TRUE CACHE BOOL "SSE2 support available" FORCE)
                set(LIBSTATS_SIMD_SOURCES "src/simd_sse2.cpp" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "LIBSTATS_HAS_SSE2=1" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: SSE2 enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: SSE2 disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: SSE2 disabled (compiler not supported)")
        endif()
    endif()

    # Detect AVX support
    if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(LIBSTATS_HAS_AVX TRUE CACHE BOOL "AVX support available" FORCE)
            get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
            get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
            list(APPEND _sources "src/simd_avx.cpp")
            list(APPEND _definitions "LIBSTATS_HAS_AVX=1")
            set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
            set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
            message(STATUS "SIMD: AVX enabled (MSVC/Clang-cl x64, default support)")
        else()
            message(STATUS "SIMD: AVX disabled (not x64)")
        endif()
    else()
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        if(COMPILER_SUPPORTS_AVX)
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
                set(LIBSTATS_HAS_AVX TRUE CACHE BOOL "AVX support available" FORCE)
                get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
                get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
                list(APPEND _sources "src/simd_avx.cpp")
                list(APPEND _definitions "LIBSTATS_HAS_AVX=1")
                set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: AVX enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: AVX disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: AVX disabled (compiler not supported)")
        endif()
    endif()

    # Detect AVX2 support
    if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(LIBSTATS_HAS_AVX2 TRUE CACHE BOOL "AVX2 support available" FORCE)
            get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
            get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
            list(APPEND _sources "src/simd_avx2.cpp")
            list(APPEND _definitions "LIBSTATS_HAS_AVX2=1")
            set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
            set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
            message(STATUS "SIMD: AVX2 enabled (MSVC/Clang-cl x64, default support)")
        else()
            message(STATUS "SIMD: AVX2 disabled (not x64)")
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
        if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
            test_runtime_cpu_feature("avx2" "
#include <immintrin.h>
bool test_avx2() {
    __m256d a = _mm256_set1_pd(1.0);
    __m256d b = _mm256_set1_pd(2.0);
    __m256d c = _mm256_set1_pd(3.0);
    __m256d result = _mm256_fmadd_pd(a, b, c);
    double values[4];
    _mm256_store_pd(values, result);
    return (values[0] == 5.0 && values[1] == 5.0 && values[2] == 5.0 && values[3] == 5.0);
}
" RUNTIME_SUPPORTS_AVX2 "-mavx2" "-mfma")
            if(RUNTIME_SUPPORTS_AVX2)
                set(LIBSTATS_HAS_AVX2 TRUE CACHE BOOL "AVX2 support available" FORCE)
                get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
                get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
                list(APPEND _sources "src/simd_avx2.cpp")
                list(APPEND _definitions "LIBSTATS_HAS_AVX2=1")
                set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: AVX2 enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: AVX2 disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: AVX2 disabled (compiler not supported)")
        endif()
    endif()

    # Detect AVX-512 support
    message(STATUS "SIMDDetection: Checking AVX-512 support...")
    if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
        check_cxx_compiler_flag("/arch:AVX512" COMPILER_SUPPORTS_AVX512)
        message(STATUS "SIMDDetection: check_cxx_compiler_flag('/arch:AVX512') result = ${COMPILER_SUPPORTS_AVX512}")
        if(COMPILER_SUPPORTS_AVX512)
            test_runtime_cpu_feature("avx512" "#include <immintrin.h>\nbool test_avx512() {\n    __m512d a = _mm512_set1_pd(1.0);\n    __m512d b = _mm512_set1_pd(2.0);\n    __m512d c = _mm512_add_pd(a, b);\n    double result[8];\n    _mm512_store_pd(result, c);\n    for(int i=0; i<8; ++i) if(result[i] != 3.0) return false;\n    return true;\n}" RUNTIME_SUPPORTS_AVX512 "/arch:AVX512")
            if(RUNTIME_SUPPORTS_AVX512)
                set(LIBSTATS_HAS_AVX512 TRUE CACHE BOOL "AVX-512 support available" FORCE)
                get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
                get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
                list(APPEND _sources "src/simd_avx512.cpp")
                list(APPEND _definitions "LIBSTATS_HAS_AVX512=1")
                set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: AVX-512 enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: AVX-512 disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: AVX-512 disabled (compiler not supported)")
        endif()
    else()
        check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
        if(COMPILER_SUPPORTS_AVX512)
            test_runtime_cpu_feature("avx512" "#include <immintrin.h>\nbool test_avx512() {\n    __m512d a = _mm512_set1_pd(1.0);\n    __m512d b = _mm512_set1_pd(2.0);\n    __m512d c = _mm512_add_pd(a, b);\n    double result[8];\n    _mm512_store_pd(result, c);\n    for(int i=0; i<8; ++i) if(result[i] != 3.0) return false;\n    return true;\n}" RUNTIME_SUPPORTS_AVX512 "-mavx512f")
            if(RUNTIME_SUPPORTS_AVX512)
                set(LIBSTATS_HAS_AVX512 TRUE CACHE BOOL "AVX-512 support available" FORCE)
                get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
                get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
                list(APPEND _sources "src/simd_avx512.cpp")
                list(APPEND _definitions "LIBSTATS_HAS_AVX512=1")
                set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: AVX-512 enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: AVX-512 disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: AVX-512 disabled (compiler not supported)")
        endif()
    endif()

    # Detect NEON support (ARM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(COMPILER_SUPPORTS_NEON TRUE)
                message(STATUS "SIMD: NEON available on AArch64 (no special flags needed)")
            else()
                check_cxx_compiler_flag("-mfpu=neon" COMPILER_SUPPORTS_NEON)
            endif()
        else()
            set(COMPILER_SUPPORTS_NEON TRUE)
        endif()
        if(COMPILER_SUPPORTS_NEON)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
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
                set(LIBSTATS_HAS_NEON TRUE CACHE BOOL "NEON support available" FORCE)
                get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
                get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
                list(APPEND _sources "src/simd_neon.cpp")
                list(APPEND _definitions "LIBSTATS_HAS_NEON=1")
                set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)
                set(LIBSTATS_SIMD_DEFINITIONS "${_definitions}" CACHE INTERNAL "List of SIMD compile definitions" FORCE)
                message(STATUS "SIMD: NEON enabled (compiler + runtime)")
            else()
                message(STATUS "SIMD: NEON disabled (runtime check failed)")
            endif()
        else()
            message(STATUS "SIMD: NEON disabled (compiler not supported)")
        endif()
    else()
        message(STATUS "SIMD: NEON disabled (not ARM architecture)")
    endif()

    # Apply any environment variable overrides
    apply_cross_compilation_overrides()

    # Always include fallback implementation
    get_property(_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)
    list(APPEND _sources "src/simd_fallback.cpp" "src/simd_dispatch.cpp")
    set(LIBSTATS_SIMD_SOURCES "${_sources}" CACHE INTERNAL "List of SIMD source files to compile" FORCE)

    # Read back cache for summary
    get_property(_sse2 CACHE LIBSTATS_HAS_SSE2 PROPERTY VALUE)
    get_property(_avx CACHE LIBSTATS_HAS_AVX PROPERTY VALUE)
    get_property(_avx2 CACHE LIBSTATS_HAS_AVX2 PROPERTY VALUE)
    get_property(_avx512 CACHE LIBSTATS_HAS_AVX512 PROPERTY VALUE)
    get_property(_neon CACHE LIBSTATS_HAS_NEON PROPERTY VALUE)
    get_property(_final_sources CACHE LIBSTATS_SIMD_SOURCES PROPERTY VALUE)

    message(STATUS "SIMD detection complete:")
    message(STATUS "  SSE2: ${_sse2}")
    message(STATUS "  AVX:  ${_avx}")
    message(STATUS "  AVX2: ${_avx2}")
    message(STATUS "  AVX-512: ${_avx512}")
    message(STATUS "  NEON: ${_neon}")
    message(STATUS "  Sources: ${_final_sources}")
endfunction()

# Function to configure SIMD compilation for a target
function(configure_simd_target TARGET_NAME)
    get_property(_sse2 CACHE LIBSTATS_HAS_SSE2 PROPERTY VALUE)
    get_property(_avx CACHE LIBSTATS_HAS_AVX PROPERTY VALUE)
    get_property(_avx2 CACHE LIBSTATS_HAS_AVX2 PROPERTY VALUE)
    get_property(_avx512 CACHE LIBSTATS_HAS_AVX512 PROPERTY VALUE)
    get_property(_neon CACHE LIBSTATS_HAS_NEON PROPERTY VALUE)
    get_property(_definitions CACHE LIBSTATS_SIMD_DEFINITIONS PROPERTY VALUE)
    
    if(_sse2)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_sse2.cpp"
                PROPERTIES COMPILE_FLAGS "/arch:SSE2")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_sse2.cpp"
                PROPERTIES COMPILE_FLAGS "-msse2")
        endif()
    endif()
    
    if(_avx)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx.cpp"
                PROPERTIES COMPILE_FLAGS "/arch:AVX")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx.cpp"
                PROPERTIES COMPILE_FLAGS "-mavx")
        endif()
    endif()
    
    if(_avx2)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx2.cpp"
                PROPERTIES COMPILE_FLAGS "/arch:AVX2")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx2.cpp"
                PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")
        endif()
    endif()
    
    if(_avx512)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx512.cpp"
                PROPERTIES COMPILE_FLAGS "/arch:AVX512")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx512.cpp"
                PROPERTIES COMPILE_FLAGS "-mavx512f")
        endif()
    endif()
    
    if(_neon)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp"
                PROPERTIES COMPILE_FLAGS "")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp"
                PROPERTIES COMPILE_FLAGS "-mfpu=neon")
        endif()
    endif()
    
    # Add compile definitions
    target_compile_definitions(${TARGET_NAME} PRIVATE ${_definitions})
endfunction()
