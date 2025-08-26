# DetectAVX512.cmake - Advanced AVX-512 detection and configuration This module provides
# comprehensive AVX-512 support detection for libstats

# Options for AVX-512 control
option(LIBSTATS_ENABLE_AVX512 "Enable AVX-512 support if available" ON)
option(LIBSTATS_FORCE_AVX512 "Force AVX-512 compilation even without runtime detection" OFF)
option(LIBSTATS_TEST_AVX512_COMPILATION "Test AVX-512 compilation during configuration" ON)

# Function to test AVX-512 compilation
function(test_avx512_compilation)
    if(NOT LIBSTATS_TEST_AVX512_COMPILATION)
        return()
    endif()

    message(STATUS "Testing AVX-512 compilation support...")

    # Create test source
    set(AVX512_TEST_SOURCE
        "
        #include <immintrin.h>
        int main() {
            __m512d a = _mm512_setzero_pd();
            __m512d b = _mm512_set1_pd(1.0);
            __m512d c = _mm512_add_pd(a, b);
            __m512d d = _mm512_fmadd_pd(a, b, c);
            double result[8];
            _mm512_storeu_pd(result, d);
            return (int)result[0];
        }
    ")

    # Test compilation with different flag combinations
    set(AVX512_FLAGS_TO_TEST
        "-mavx512f" "-mavx512f -mavx512dq" "-mavx512f -mavx512dq -mavx512bw -mavx512vl"
        "-march=skylake-avx512" "-march=native")

    foreach(flags ${AVX512_FLAGS_TO_TEST})
        string(REPLACE " " "_" flag_name "${flags}")
        string(REPLACE "-" "_" flag_name "${flag_name}")

        try_compile(
            AVX512_COMPILE_${flag_name} ${CMAKE_BINARY_DIR}/cmake_temp
            SOURCES ${CMAKE_BINARY_DIR}/cmake_temp/avx512_test.cpp
            CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=${flags}")

        if(AVX512_COMPILE_${flag_name})
            message(STATUS "  ✅ AVX-512 compiles with flags: ${flags}")
            set(LIBSTATS_AVX512_COMPILE_FLAGS
                "${flags}"
                PARENT_SCOPE)
            set(LIBSTATS_AVX512_COMPILATION_SUPPORTED
                TRUE
                PARENT_SCOPE)
            break()
        else()
            message(STATUS "  ❌ AVX-512 failed with flags: ${flags}")
        endif()
    endforeach()

    # Write test source to file
    file(WRITE ${CMAKE_BINARY_DIR}/cmake_temp/avx512_test.cpp "${AVX512_TEST_SOURCE}")
endfunction()

# Function to detect AVX-512 runtime support
function(detect_avx512_runtime)
    message(STATUS "Checking for AVX-512 runtime support...")

    # Create runtime detection program
    set(AVX512_RUNTIME_TEST
        "
        #include <iostream>
        #include <immintrin.h>

        #ifdef _WIN32
        #include <intrin.h>
        #endif

        bool check_avx512_support() {
            unsigned int eax, ebx, ecx, edx;

            // Check if CPUID is supported
            #ifdef _WIN32
                int cpuinfo[4];
                __cpuid(cpuinfo, 0);
                if (cpuinfo[0] < 7) return false;

                __cpuidex(cpuinfo, 7, 0);
                return (cpuinfo[1] & (1 << 16)) != 0; // AVX-512F
            #else
                __asm__ (\"cpuid\" : \"=a\"(eax), \"=b\"(ebx), \"=c\"(ecx), \"=d\"(edx) : \"a\"(0));
                if (eax < 7) return false;

                __asm__ (\"cpuid\" : \"=a\"(eax), \"=b\"(ebx), \"=c\"(ecx), \"=d\"(edx) : \"a\"(7), \"c\"(0));
                return (ebx & (1 << 16)) != 0; // AVX-512F
            #endif
        }

        int main() {
            if (check_avx512_support()) {
                std::cout << \"AVX-512 runtime support: YES\" << std::endl;
                return 0;
            } else {
                std::cout << \"AVX-512 runtime support: NO\" << std::endl;
                return 1;
            }
        }
    ")

    # Write and try to run the test
    file(WRITE ${CMAKE_BINARY_DIR}/cmake_temp/avx512_runtime_test.cpp "${AVX512_RUNTIME_TEST}")

    try_compile(
        AVX512_RUNTIME_COMPILE ${CMAKE_BINARY_DIR}/cmake_temp
        SOURCES ${CMAKE_BINARY_DIR}/cmake_temp/avx512_runtime_test.cpp
        CMAKE_FLAGS "-DCMAKE_CXX_FLAGS=${LIBSTATS_AVX512_COMPILE_FLAGS}"
        COPY_FILE ${CMAKE_BINARY_DIR}/cmake_temp/avx512_runtime_test)

    if(AVX512_RUNTIME_COMPILE)
        execute_process(
            COMMAND ${CMAKE_BINARY_DIR}/cmake_temp/avx512_runtime_test
            RESULT_VARIABLE AVX512_RUNTIME_RESULT
            OUTPUT_VARIABLE AVX512_RUNTIME_OUTPUT
            ERROR_QUIET)

        if(AVX512_RUNTIME_RESULT EQUAL 0)
            message(STATUS "  ✅ ${AVX512_RUNTIME_OUTPUT}")
            set(LIBSTATS_AVX512_RUNTIME_SUPPORTED
                TRUE
                PARENT_SCOPE)
        else()
            message(STATUS "  ❌ ${AVX512_RUNTIME_OUTPUT}")
            set(LIBSTATS_AVX512_RUNTIME_SUPPORTED
                FALSE
                PARENT_SCOPE)
        endif()
    else()
        message(STATUS "  ❌ Failed to compile AVX-512 runtime test")
        set(LIBSTATS_AVX512_RUNTIME_SUPPORTED
            FALSE
            PARENT_SCOPE)
    endif()
endfunction()

# Main AVX-512 detection logic
if(LIBSTATS_ENABLE_AVX512 OR LIBSTATS_FORCE_AVX512)
    message(STATUS "=== AVX-512 Detection ===")

    # Ensure temp directory exists
    file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/cmake_temp)

    # Test compilation support
    test_avx512_compilation()

    if(LIBSTATS_AVX512_COMPILATION_SUPPORTED OR LIBSTATS_FORCE_AVX512)
        # Test runtime support (only if we can compile)
        if(NOT CMAKE_CROSSCOMPILING AND NOT LIBSTATS_FORCE_AVX512)
            detect_avx512_runtime()
        else()
            message(STATUS "Skipping AVX-512 runtime detection (cross-compiling or forced)")
            set(LIBSTATS_AVX512_RUNTIME_SUPPORTED TRUE)
        endif()

        # Set final configuration
        if(LIBSTATS_AVX512_RUNTIME_SUPPORTED OR LIBSTATS_FORCE_AVX512)
            set(LIBSTATS_HAS_AVX512
                TRUE
                CACHE BOOL "AVX-512 support available" FORCE)
            message(STATUS "✅ AVX-512 support enabled")

            if(LIBSTATS_AVX512_COMPILE_FLAGS)
                message(STATUS "AVX-512 compile flags: ${LIBSTATS_AVX512_COMPILE_FLAGS}")
            endif()
        else()
            set(LIBSTATS_HAS_AVX512
                FALSE
                CACHE BOOL "AVX-512 support not available" FORCE)
            message(STATUS "❌ AVX-512 runtime not supported")
        endif()
    else()
        set(LIBSTATS_HAS_AVX512
            FALSE
            CACHE BOOL "AVX-512 compilation not supported" FORCE)
        message(STATUS "❌ AVX-512 compilation not supported")
    endif()
else()
    set(LIBSTATS_HAS_AVX512
        FALSE
        CACHE BOOL "AVX-512 support disabled" FORCE)
    message(STATUS "AVX-512 support disabled by configuration")
endif()

# Export results for use in main CMakeLists.txt
if(LIBSTATS_HAS_AVX512)
    add_compile_definitions(LIBSTATS_HAS_AVX512)
    if(LIBSTATS_AVX512_COMPILE_FLAGS)
        # These flags will be applied to AVX-512 specific files
        set(CMAKE_CXX_FLAGS_AVX512
            "${LIBSTATS_AVX512_COMPILE_FLAGS}"
            CACHE STRING "Flags for AVX-512 compilation" FORCE)
    endif()
endif()

message(STATUS "========================")
