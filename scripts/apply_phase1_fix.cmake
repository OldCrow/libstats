# apply_phase1_fix.cmake
# 
# Phase 1 Fix: Replace problematic generator expressions with traditional conditionals
# This script creates a modified CMakeLists.txt that fixes the immediate issues
# while preserving all working functionality.

message(STATUS "Applying Phase 1 CMake modernization fix...")

# Read the current CMakeLists.txt
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" ORIGINAL_CONTENT)

# Define the problematic section to replace (lines 798-828)
set(PROBLEMATIC_SECTION "# Apply SIMD-specific compilation flags using generator expressions
add_compile_options\\(
    # MSVC SIMD \\(x64 only\\)
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:MSVC>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:/arch:AVX2>
    
    # Clang SIMD \\(Windows x64 only for ClangCL\\)
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:Clang>,\\$<PLATFORM_ID:Windows>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:-mavx2>
    
    # GCC SIMD \\(Conservative approach - SSE2 baseline only\\)
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:GNU>,\\$<BOOL:\\$\\{LIBSTATS_HAS_SSE2\\}>>:-msse2>
\\)

# Apply SIMD compile definitions using generator expressions
add_compile_definitions\\(
    # MSVC SIMD definitions \\(x64 only\\)
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:MSVC>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_AVX2=1>
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:MSVC>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_AVX=1>
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:MSVC>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_SSE2=1>
    
    # Clang SIMD definitions \\(Windows x64 only\\)
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:Clang>,\\$<PLATFORM_ID:Windows>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_AVX2=1>
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:Clang>,\\$<PLATFORM_ID:Windows>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_AVX=1>
    \\$<\\$<AND:\\$<CXX_COMPILER_ID:Clang>,\\$<PLATFORM_ID:Windows>,\\$<EQUAL:\\$\\{CMAKE_SIZEOF_VOID_P\\},8>>:LIBSTATS_HAS_SSE2=1>
    
    # GCC/Clang SIMD definitions \\(runtime-dispatched, conservative approach\\)
    \\$<\\$<BOOL:\\$\\{LIBSTATS_HAS_SSE2\\}>:LIBSTATS_HAS_SSE2=1>
    \\$<\\$<BOOL:\\$\\{LIBSTATS_HAS_AVX\\}>:LIBSTATS_HAS_AVX=1>
    \\$<\\$<BOOL:\\$\\{LIBSTATS_HAS_AVX2\\}>:LIBSTATS_HAS_AVX2=1>
    \\$<\\$<BOOL:\\$\\{LIBSTATS_HAS_AVX512\\}>:LIBSTATS_HAS_AVX512=1>
    \\$<\\$<BOOL:\\$\\{LIBSTATS_HAS_NEON\\}>:LIBSTATS_HAS_NEON=1>
\\)")

# Define the replacement section
set(REPLACEMENT_SECTION "# =============================================================================
# SIMD COMPILATION FLAGS - TRADITIONAL APPROACH (PHASE 1 FIX)
# =============================================================================
# Removed problematic nested generator expressions that cause all flags to be
# passed to compilers. Using traditional conditionals that are known to work.
# 
# STRATEGY:
# - MSVC/Clang-cl on Windows: Use global flags (simple, reliable)
# - GCC/Clang on Unix: Use source-file-specific flags (cmake/SIMDDetection.cmake)
# - All platforms: Definitions are set by SIMDDetection.cmake based on detection

# Windows compilers: Use global SIMD flags for compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL \"MSVC\" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    # MSVC x64 has comprehensive SIMD support
    add_compile_options(/arch:AVX2)
    message(STATUS \"Applied MSVC x64 SIMD flags: /arch:AVX2\")
    
elseif(CMAKE_CXX_COMPILER_ID MATCHES \"Clang\" AND WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    # Clang-cl on Windows x64
    add_compile_options(-mavx2)
    message(STATUS \"Applied Clang-cl x64 SIMD flags: -mavx2\")
endif()

# IMPORTANT: SIMD compile definitions are handled by cmake/SIMDDetection.cmake
# That system detects actual CPU capabilities and sets appropriate definitions:
# - LIBSTATS_HAS_SSE2, LIBSTATS_HAS_AVX, LIBSTATS_HAS_AVX2, etc.
# The definitions are applied through the libstats_simd_interface target.

# Unix compilers (GCC/Clang): SIMD flags are applied per-source-file
# via the configure_simd_target() function from cmake/SIMDDetection.cmake
# This provides precise control and avoids flag conflicts.")

# Replace the problematic section
string(REGEX REPLACE "${PROBLEMATIC_SECTION}" "${REPLACEMENT_SECTION}" MODIFIED_CONTENT "${ORIGINAL_CONTENT}")

# Verify the replacement worked
if("${MODIFIED_CONTENT}" STREQUAL "${ORIGINAL_CONTENT}")
    message(WARNING "Phase 1 fix: No changes detected - pattern may need adjustment")
    return()
endif()

# Create backup
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" 
     DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt.phase0.backup")
message(STATUS "Created backup: CMakeLists.txt.phase0.backup")

# Write the modified content
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt" "${MODIFIED_CONTENT}")
message(STATUS "Applied Phase 1 fix to CMakeLists.txt")

message(STATUS "Phase 1 modernization complete!")
message(STATUS "")
message(STATUS "TESTING REQUIRED:")
message(STATUS "1. Clean build: rm -rf build && mkdir build && cd build")
message(STATUS "2. Configure: cmake ..")
message(STATUS "3. Build verbose: cmake --build . -- VERBOSE=1")
message(STATUS "4. Run SIMD tests: ./tests/test_simd_integration")
message(STATUS "5. Verify no duplicate flags in build output")
message(STATUS "")
message(STATUS "If issues occur, restore: cp CMakeLists.txt.phase0.backup CMakeLists.txt")
