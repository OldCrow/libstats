# SIMDApplication.cmake - Apply detected SIMD capabilities to build targets
#
# Companion to cmake/SIMDDetection.cmake: that module detects what the compiler/CPU can do
# (LIBSTATS_HAS_* cache variables, LIBSTATS_SIMD_SOURCES/LIBSTATS_SIMD_DEFINITIONS) and this module
# applies those results — interface target, per-source-file compile flags, and per-target linkage.
# Include this immediately after cmake/SIMDDetection.cmake and after detect_simd_features() has run.

# Create SIMD interface target for centralized SIMD configuration This interface target acts as a
# modern CMake way to propagate SIMD settings
function(create_simd_interface_target)
    # Create interface library for SIMD definitions
    add_library(libstats_simd_interface INTERFACE)

    get_property(
        _definitions
        CACHE LIBSTATS_SIMD_DEFINITIONS
        PROPERTY VALUE)

    # Add all SIMD compile definitions to the interface target
    if(_definitions)
        target_compile_definitions(libstats_simd_interface INTERFACE ${_definitions})
    endif()

    # Create alias for consistency with other interface targets
    add_library(libstats::simd ALIAS libstats_simd_interface)

    message(STATUS "SIMD interface target created with definitions: ${_definitions}")
endfunction()

# Apply per-source-file SIMD compile flags. These are file-global properties and need to be set only
# once; call this function once after create_simd_interface_target().
function(apply_simd_source_flags)
    get_property(
        _sse2
        CACHE LIBSTATS_HAS_SSE2
        PROPERTY VALUE)
    get_property(
        _avx
        CACHE LIBSTATS_HAS_AVX
        PROPERTY VALUE)
    get_property(
        _avx2
        CACHE LIBSTATS_HAS_AVX2
        PROPERTY VALUE)
    get_property(
        _avx512
        CACHE LIBSTATS_HAS_AVX512
        PROPERTY VALUE)
    get_property(
        _neon
        CACHE LIBSTATS_HAS_NEON
        PROPERTY VALUE)

    if(_sse2)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_sse2.cpp"
                                        PROPERTIES COMPILE_OPTIONS "/arch:SSE2")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_sse2.cpp"
                                        PROPERTIES COMPILE_OPTIONS "-msse2")
        endif()
    endif()

    if(_avx)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx.cpp"
                                        PROPERTIES COMPILE_OPTIONS "/arch:AVX")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx.cpp"
                                        PROPERTIES COMPILE_OPTIONS "-mavx")
        endif()
    endif()

    if(_avx2)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx2.cpp"
                                        PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx2.cpp"
                                        PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
        endif()
    endif()

    if(_avx512)
        if(MSVC OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32))
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx512.cpp"
                                        PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_avx512.cpp"
                                        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq")
        endif()
    endif()

    if(_neon)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            # AArch64 has NEON unconditionally — no extra flag needed.
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp"
                                        PROPERTIES COMPILE_OPTIONS "")
        else()
            set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/simd_neon.cpp"
                                        PROPERTIES COMPILE_OPTIONS "-mfpu=neon")
        endif()
    endif()
endfunction()

# Link the SIMD interface target (with its compile definitions) to a specific library or object
# target. Call this for every target that compiles any source in the project.
function(configure_simd_target TARGET_NAME)
    if(TARGET libstats_simd_interface)
        target_link_libraries(${TARGET_NAME} PRIVATE libstats::simd)
    endif()
endfunction()
