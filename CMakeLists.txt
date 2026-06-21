cmake_minimum_required(VERSION 3.20)

# Set CMake policies
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW) # find_package uses upper-case <PACKAGENAME>_ROOT variables
endif()

# Verbose messaging control
option(LIBSTATS_VERBOSE_BUILD "Enable verbose build messages for debugging" OFF)

# Threading system preference control
option(LIBSTATS_FORCE_TBB
       "Force TBB usage even on platforms with native threading (e.g., GCD on macOS)" OFF)

# Automatic parallel job detection for optimal build performance
include(ProcessorCount)
include(CheckCXXSourceCompiles)
include(CheckIncludeFileCXX)
ProcessorCount(CPU_COUNT)
if(CPU_COUNT EQUAL 0)
    # Fallback if detection fails
    set(CPU_COUNT 4)
    if(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "CPU count detection failed, defaulting to 4 parallel jobs")
    endif()
else()
    if(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "Detected ${CPU_COUNT} CPU cores - enabling parallel builds")
    endif()
endif()

# Set parallel options for CMake build command (for use with cmake --build)
if(NOT DEFINED CMAKE_BUILD_PARALLEL_LEVEL)
    set(CMAKE_BUILD_PARALLEL_LEVEL
        ${CPU_COUNT}
        CACHE STRING "Number of parallel build jobs")
endif()
if(LIBSTATS_VERBOSE_BUILD)
    message(STATUS "Set CMAKE_BUILD_PARALLEL_LEVEL=${CPU_COUNT} for cmake --build")
endif()

# Enforce Ventura minimum on macOS — prevents accidental use of pre-13 APIs.
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET
        "13.0"
        CACHE STRING "Minimum macOS deployment target")
endif()

project(
    libstats
    VERSION 2.0.0
    LANGUAGES CXX)

# v2.x compiler baseline. Fail early with actionable messages.
if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.0)
        message(FATAL_ERROR "libstats v2.x requires AppleClang 15+ (Xcode 15+) on macOS")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        message(FATAL_ERROR "libstats v2.x requires GCC 13+")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 17.0)
        message(FATAL_ERROR "libstats v2.x requires Clang 17+")
    endif()
elseif(MSVC)
    if(MSVC_VERSION LESS 1938)
        message(FATAL_ERROR "libstats v2.x requires MSVC 19.38+ (Visual Studio 2022 17.8+)")
    endif()
endif()

# Build examples by default only for top-level builds.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(LIBSTATS_BUILD_EXAMPLES_DEFAULT ON)
else()
    set(LIBSTATS_BUILD_EXAMPLES_DEFAULT OFF)
endif()
option(LIBSTATS_BUILD_EXAMPLES "Build examples" ${LIBSTATS_BUILD_EXAMPLES_DEFAULT})

# Set default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Dev' as none was specified.")
    set(CMAKE_BUILD_TYPE
        Dev
        CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "Dev"
                                                 "Strict")
endif()

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# C++20 <concepts> and <ranges> are available on all v2.0.0 baseline compilers.

# Detect whether std::cyl_bessel_i is available (C++17 §29.9.3). GCC 6.1+, MSVC 2017 15.5+ support
# it; AppleClang / macOS libc++ does not. VonMisesDistribution falls back to A&S polynomial
# approximations when absent.
check_cxx_source_compiles("#include <cmath>
int main() { (void)std::cyl_bessel_i(0.0, 1.0); return 0; }" LIBSTATS_HAS_CXX17_BESSEL)
if(LIBSTATS_HAS_CXX17_BESSEL)
    message(STATUS "std::cyl_bessel_i (C++17 special functions): available")
    # LP-4: defer scoping to the SIMD interface target (applied below after target creation). Do NOT
    # use add_compile_definitions here — that would be global.
else()
    message(STATUS "std::cyl_bessel_i: unavailable -- Bessel A&S fallback active")
endif()

# Platform-specific configurations
# =============================================================================
# 2B: -stdlib=libc++ — AppleClang 15 defaults to libc++ on Ventura 13+. Use
# add_compile_options/add_link_options instead of CMAKE_CXX_FLAGS mutation so the flag is a proper
# list entry rather than a string accumulation.
if(APPLE)
    add_compile_options(-stdlib=libc++)
    add_link_options(-stdlib=libc++)
    message(STATUS "macOS: system AppleClang with libc++ (GCD parallel path active)")
elseif(UNIX AND NOT APPLE)
    # -fPIC is required for shared library compilation on Linux.
    add_compile_options(-fPIC)
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-soname,libstats.so.${PROJECT_VERSION_MAJOR}")
elseif(WIN32)
    # Windows compatibility macros (LP-3: add_compile_definitions, not add_definitions)
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# =============================================================================
# 2A: Unified threading system detection — one function, platform guards inside. Replaces the three
# separate detect_threading_systems_*() functions.
# =============================================================================
function(detect_threading_systems)
    if(DEFINED CACHE{LIBSTATS_THREADING_DETECTION_COMPLETE})
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Using cached threading detection results")
        endif()
        return()
    endif()

    message(STATUS "Detecting threading capabilities...")

    # ── OpenMP (all platforms) ──────────────────────────────────────────────
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        set(LIBSTATS_HAS_OPENMP
            TRUE
            CACHE BOOL "OpenMP available")
        message(STATUS "  ✓ OpenMP found")
    else()
        set(LIBSTATS_HAS_OPENMP
            FALSE
            CACHE BOOL "OpenMP not available")
        message(STATUS "  ✗ OpenMP not found")
    endif()

    # ── POSIX threads (Unix) ────────────────────────────────────────────────
    if(NOT WIN32)
        if(UNIX AND NOT APPLE)
            find_package(Threads REQUIRED)
        else()
            find_package(Threads QUIET)
        endif()
        if(Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
            set(LIBSTATS_HAS_PTHREADS
                TRUE
                CACHE BOOL "POSIX threads available")
            message(STATUS "  ✓ POSIX threads found")
        else()
            set(LIBSTATS_HAS_PTHREADS
                FALSE
                CACHE BOOL "POSIX threads not available")
            message(STATUS "  ✗ POSIX threads not found")
        endif()
    else()
        set(LIBSTATS_HAS_PTHREADS
            FALSE
            CACHE BOOL "POSIX threads not natively available on Windows")
    endif()

    # ── Grand Central Dispatch (macOS only) ────────────────────────────────
    if(APPLE)
        set(LIBSTATS_HAS_GCD
            TRUE
            CACHE BOOL "Grand Central Dispatch available")
        message(STATUS "  ✓ Grand Central Dispatch (GCD) available")
        set(LIBSTATS_HAS_WIN_THREADPOOL
            FALSE
            CACHE BOOL "Windows Thread Pool not available")
        set(LIBSTATS_HAS_WIN32_THREADS
            FALSE
            CACHE BOOL "Win32 threads not applicable on macOS")
    elseif(WIN32)
        # ── Win32 threads and Thread Pool API (Windows) ────────────────────
        set(LIBSTATS_HAS_GCD
            FALSE
            CACHE BOOL "GCD not available on Windows")
        set(LIBSTATS_HAS_WIN32_THREADS
            TRUE
            CACHE BOOL "Win32 threads available")
        message(STATUS "  ✓ Win32 threads available")
        # Thread Pool API: available on Windows Vista+ (all modern Windows targets)
        if(DEFINED CMAKE_SYSTEM_VERSION AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "6.0")
            set(LIBSTATS_HAS_WIN_THREADPOOL
                TRUE
                CACHE BOOL "Windows Thread Pool API available")
            message(STATUS "  ✓ Windows Thread Pool API available")
        elseif(DEFINED _WIN32_WINNT AND _WIN32_WINNT GREATER_EQUAL 0x0600)
            set(LIBSTATS_HAS_WIN_THREADPOOL
                TRUE
                CACHE BOOL "Windows Thread Pool API available")
            message(STATUS "  ✓ Windows Thread Pool API available (_WIN32_WINNT >= 0x0600)")
        else()
            set(LIBSTATS_HAS_WIN_THREADPOOL
                TRUE
                CACHE BOOL "Windows Thread Pool API likely available")
            message(STATUS "  ~ Windows Thread Pool API — assuming Vista+ (modern Windows)")
        endif()
        # TBB detection for Windows is handled by detect_tbb_unified() below
    else()
        # ── Linux ──────────────────────────────────────────────────────────
        set(LIBSTATS_HAS_GCD
            FALSE
            CACHE BOOL "GCD not available on Linux")
        set(LIBSTATS_HAS_WIN_THREADPOOL
            FALSE
            CACHE BOOL "Windows Thread Pool not available")
        set(LIBSTATS_HAS_WIN32_THREADS
            FALSE
            CACHE BOOL "Win32 threads not available")
    endif()

    set(LIBSTATS_THREADING_DETECTION_COMPLETE
        TRUE
        CACHE BOOL "Threading detection complete")
    message(
        STATUS
            "Threading: OpenMP=${LIBSTATS_HAS_OPENMP} Pthreads=${LIBSTATS_HAS_PTHREADS} GCD=${LIBSTATS_HAS_GCD}"
    )
endfunction()

detect_threading_systems()
# SIMD feature detection and compilation flags Use our comprehensive SIMD detection system
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SIMDDetection.cmake")

# Set options for SIMD detection behavior
option(LIBSTATS_ENABLE_RUNTIME_CHECKS "Enable runtime CPU checks even when cross-compiling" OFF)
# Perform comprehensive SIMD detection
detect_simd_features()

# =============================================================================
# CONSOLIDATED TBB DETECTION FUNCTION
# =============================================================================
# Unified TBB detection logic for all platforms
function(detect_tbb_unified)
    if(DEFINED CACHE{LIBSTATS_TBB_DETECTION_COMPLETE})
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Using cached TBB detection results")
        endif()
        return()
    endif()

    if(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "Detecting Intel TBB...")
    endif()
    set(LIBSTATS_HAS_TBB FALSE)

    # Method 1: find_package (preferred for vcpkg, conan, system installs)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        set(LIBSTATS_HAS_TBB TRUE)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  ✓ TBB found via find_package")
        endif()
    else()
        # Method 2: pkg-config (preferred for Homebrew, Linux package managers)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(TBB QUIET tbb)
            if(TBB_FOUND)
                # LP-2: propagate to parent scope; target_include_directories added after targets.
                set(LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL
                    "${TBB_INCLUDE_DIRS}"
                    PARENT_SCOPE)
                set(LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL
                    "${TBB_LIBRARY_DIRS}"
                    PARENT_SCOPE)
                # BS-4: accumulate TBB compile flags in a scoped variable instead of mutating the
                # global CMAKE_CXX_FLAGS string.
                set(LIBSTATS_TBB_CFLAGS_INTERNAL
                    "${TBB_CFLAGS_OTHER}"
                    PARENT_SCOPE)
                set(LIBSTATS_HAS_TBB TRUE)
                if(LIBSTATS_VERBOSE_BUILD)
                    message(STATUS "  ✓ TBB found via pkg-config")
                endif()
            endif()
        endif()
    endif()

    # Cache result
    set(LIBSTATS_HAS_TBB
        ${LIBSTATS_HAS_TBB}
        CACHE BOOL "Intel TBB support available")
    set(LIBSTATS_TBB_DETECTION_COMPLETE
        TRUE
        CACHE BOOL "TBB detection completed")

    if(LIBSTATS_HAS_TBB)
        message(STATUS "Intel TBB: AVAILABLE - parallel execution policies enhanced")
    else()
        message(
            STATUS "Intel TBB: NOT FOUND - C++20 execution policies may have limited performance")
    endif()
endfunction()

# Execute unified TBB detection with platform preference On macOS, prefer GCD over TBB for better
# integration with the platform unless explicitly overridden by LIBSTATS_FORCE_TBB
if(APPLE
   AND LIBSTATS_HAS_GCD
   AND NOT LIBSTATS_FORCE_TBB)
    message(STATUS "macOS detected with GCD available - TBB detection skipped (GCD preferred)")
    message(STATUS "  To force TBB usage on macOS, set -DLIBSTATS_FORCE_TBB=ON")
    set(LIBSTATS_HAS_TBB
        FALSE
        CACHE BOOL "TBB disabled in favor of GCD on macOS" FORCE)
    set(LIBSTATS_TBB_DETECTION_COMPLETE
        TRUE
        CACHE BOOL "TBB detection skipped for GCD preference" FORCE)
else()
    if(LIBSTATS_FORCE_TBB AND APPLE)
        message(STATUS "LIBSTATS_FORCE_TBB=ON - forcing TBB detection even on macOS with GCD")
    endif()
    detect_tbb_unified()
endif()

# =============================================================================
# SHARED COMPILER FLAG DEFINITIONS
# =============================================================================
# Define reusable flag sets to reduce repetition and improve maintainability

# Common warning flag sets
set(LIBSTATS_COMMON_WARNINGS_UNIX -Wall -Wextra)
set(LIBSTATS_COMMON_WARNINGS_MSVC /W3)

# Strict warning flag sets (shared between compiler-specific modes)
set(LIBSTATS_CLANG_STRICT_WARNINGS
    -Wall
    -Wextra
    -pedantic
    # Type conversion warnings (strict)
    -Wconversion
    -Wfloat-conversion
    -Wimplicit-int-conversion
    -Wshorten-64-to-32
    -Wdouble-promotion
    -Wsign-conversion
    # Additional strict warnings
    -Wshadow
    -Wunused
    -Wcast-align
    -Wcast-qual
    -Wold-style-cast
    -Woverloaded-virtual
    -Wextra-semi
    -Wmissing-declarations
    # Clang-specific warnings
    -Wloop-analysis
    -Wlogical-op-parentheses
    -Wbool-conversion
    -Wint-conversion
    -Wnull-dereference
    -Wuninitialized
    -Wconditional-uninitialized
    -Wstring-conversion
    -Wdeprecated-volatile # Catch deprecated volatile usage (C++20)
    # ODR (One Definition Rule) violation detection - Enhanced
    -Wduplicate-decl-specifier
    -Wodr
    -fno-common
    -Winconsistent-missing-override
    -Wmultiple-move-vbase
    # Allow some warnings that are too strict for practical development
    -Wno-unused-parameter # Common in template/callback code
    -Wno-padded # Not practical for cross-platform development
    -Wno-c++98-compat # We're using C++20
    -Wno-c++98-compat-pedantic)

set(LIBSTATS_GCC_STRICT_WARNINGS
    -Wall
    -Wextra
    -pedantic
    # GCC-specific conversion warnings
    -Wconversion
    -Wfloat-conversion
    -Wdouble-promotion
    -Wsign-conversion
    -Wold-style-cast
    # GCC-specific diagnostic warnings
    -Wlogical-op
    -Wduplicated-cond
    -Wduplicated-branches
    -Wrestrict
    -Wnull-dereference
    -Wjump-misses-init
    # Template and name resolution differences
    -Wtrampolines
    -Wunsafe-loop-optimizations
    -Wvector-operation-performance
    -Wsuggest-override
    -Wsuggest-final-types
    -Wsuggest-final-methods
    # Memory and alignment warnings
    -Wcast-align=strict
    -Wstrict-overflow=2
    -Wformat=2
    -Wformat-overflow=2
    -Wformat-truncation=2
    # GCC's stricter undefined behavior detection
    -Wshift-overflow=2
    -Wstringop-overflow=4
    # ODR (One Definition Rule) violation detection - Enhanced
    -fno-common
    -Wodr
    -Wredundant-decls
    # Allow some warnings that are too pedantic
    -Wno-inline
    -Wno-padded
    -Wno-unused-parameter
    -Wno-sign-conversion # Too noisy with standard library
)

# MSVC-compatible warning flags for cross-compiler testing
set(LIBSTATS_MSVC_COMPAT_WARNINGS
    -Wall
    -Wextra
    # Type conversion warnings (MSVC-like strictness)
    -Wconversion
    -Wfloat-conversion
    -Wimplicit-int-conversion
    -Wshorten-64-to-32
    -Wdouble-promotion
    # Additional strict warnings
    -Wshadow
    -Wunused
    -Wcast-align
    -Wcast-qual
    -Wold-style-cast
    -Woverloaded-virtual
    -Wextra-semi
    -Wmissing-declarations
    # Allow some warnings that are too strict for practical development
    -Wno-sign-conversion # Too many false positives in standard library interactions
    -Wno-unused-parameter # Common in template/callback code
    -Wno-padded # Not practical for cross-platform development
)

# Enhanced MSVC warning sets
set(LIBSTATS_MSVC_ENHANCED_WARNINGS
    /w14242
    /w14254
    /w14263
    /w14265
    /w14287
    /w14289
    /w14296
    /w14311
    /w14545
    /w14546
    /w14547
    /w14549
    /w14555
    /w14619
    /w14640
    /w14826
    /w14905
    /w14906
    /w14928
    # Additional warnings that MSVC shows but our builds miss
    /w14267 # size_t to smaller integer conversion
    /w14244 # possible loss of data (narrowing conversion)
    /w14101 # unreferenced local variable
    # Code analysis warnings
    /analyze:WX- # Don't treat code analysis warnings as errors (for MSVCWarn)
    # ODR (One Definition Rule) violation detection - Enhanced
    /we4020 # Too many actual parameters for function-like macro
    /we4002 # Too many actual parameters for macro
    /we4005 # Macro redefinition (can catch ODR issues)
    /we4229 # Anachronism used: modifiers on data are ignored
    /we4239 # Nonstandard extension used (can catch duplicate definitions)
)

# Optimization levels for different build types
set(LIBSTATS_OPT_NONE_UNIX -O0)
set(LIBSTATS_OPT_LIGHT_UNIX -O1)
set(LIBSTATS_OPT_FULL_UNIX -O3)
set(LIBSTATS_OPT_NONE_MSVC /Od)
set(LIBSTATS_OPT_LIGHT_MSVC /O1)
set(LIBSTATS_OPT_FULL_MSVC /O2)

# Debug information flags
set(LIBSTATS_DEBUG_INFO_UNIX -g)
set(LIBSTATS_DEBUG_INFO_MSVC /Zi)

# =============================================================================
# COMPILER-SPECIFIC BUILD CONFIGURATION WITH GENERATOR EXPRESSIONS
# =============================================================================
# Use generator expressions to reduce repetitive if/elseif chains Standard CMake Build Types
# (Release, Debug) - Cross-platform Custom Compiler-Specific Build Types (ClangStrict, ClangWarn,
# MSVCStrict, MSVCWarn, GCCStrict, GCCWarn)
# =============================================================================

# =============================================================================
# PROVEN TRADITIONAL CMAKE COMPILER CONFIGURATION
# =============================================================================
# Using traditional conditionals that are guaranteed to work reliably Generator expressions have
# proven problematic with complex nested conditions

# MSVC-specific compiler definitions (Windows compatibility)
add_compile_definitions($<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
                        $<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_DEPRECATE>)

# Apply build type and compiler-specific flags using traditional CMake approach This maintains
# reliability while ensuring cross-compiler compatibility

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC compiler flags /utf-8: interpret source files as UTF-8 and emit UTF-8 output. Silences
    # C4566 (universal-character-name cannot be represented in current code page) for Unicode
    # characters in string literals across all tool sources.
    add_compile_options(/utf-8)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/W3 /O2 /DNDEBUG)
        message(STATUS "Applied MSVC Release flags: /W3 /O2 /DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/W3 /Od /Zi /RTC1 /MDd)
        message(STATUS "Applied MSVC Debug flags: /W3 /Od /Zi /RTC1 /MDd")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(/W3 /O1 /Zi /MD /wd4996)
        message(STATUS "Applied MSVC Dev flags: /W3 /O1 /Zi /MD /wd4996")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(/W4 /WX /permissive-)
        add_compile_options(${LIBSTATS_MSVC_ENHANCED_WARNINGS})
        message(STATUS "Applied Strict flags (MSVC): /W4 /WX /permissive-")
    else()
        add_compile_options(/W3)
        message(STATUS "Applied MSVC default flags: /W3")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    # Clang/AppleClang compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied Clang Release flags: ${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied Clang Debug flags: ${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
        message(
            STATUS
                "Applied Clang Dev flags: ${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
        message(STATUS "Applied Strict flags (Clang/AppleClang): strict warnings + -Werror")
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} -O2)
        message(STATUS "Applied Clang default flags: ${LIBSTATS_COMMON_WARNINGS_UNIX} -O2")
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
        message(
            STATUS
                "Applied GCC Release flags: ${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX}"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -fstack-protector-strong)
        message(
            STATUS
                "Applied GCC Debug flags: ${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -fstack-protector-strong"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX}
                            ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
        message(
            STATUS
                "Applied GCC Dev flags: ${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations"
        )
    elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
        add_compile_options(${LIBSTATS_GCC_STRICT_WARNINGS} -Werror)
        message(STATUS "Applied Strict flags (GCC): strict warnings + -Werror")
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} ${LIBSTATS_OPT_FULL_UNIX})
        message(
            STATUS
                "Applied GCC default flags: ${LIBSTATS_COMMON_WARNINGS_UNIX} ${LIBSTATS_OPT_FULL_UNIX}"
        )
    endif()
else()
    # Unknown compiler - use sensible defaults
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
        message(STATUS "Applied unknown compiler Release flags: -O3 -DNDEBUG")
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
        message(STATUS "Applied unknown compiler Debug flags: -O0 -g")
    else()
        add_compile_options(-O2)
        message(STATUS "Applied unknown compiler default flags: -O2")
    endif()
endif()

# =============================================================================
# SIMD COMPILATION FLAGS - TRADITIONAL APPROACH (MODERNIZED)
# =============================================================================
# Replaced problematic nested generator expressions that can cause all flags to be passed to
# compilers. Using traditional conditionals that are proven to work.
#
# STRATEGY: - MSVC/Clang-cl on Windows: Use global flags (simple, reliable) - GCC/Clang on Unix: Use
# source-file-specific flags (cmake/SIMDDetection.cmake) - All platforms: Definitions are set by
# SIMDDetection.cmake based on detection

# Windows compilers: Use highest detected SIMD level as global flag. SIMDDetection.cmake has already
# run by this point and set LIBSTATS_HAS_AVX512 etc.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(LIBSTATS_HAS_AVX512)
        add_compile_options(/arch:AVX512)
        message(STATUS "Applied MSVC x64 SIMD flags: /arch:AVX512")
    else()
        add_compile_options(/arch:AVX2)
        message(STATUS "Applied MSVC x64 SIMD flags: /arch:AVX2")
    endif()

elseif(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND WIN32
    AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(LIBSTATS_HAS_AVX512)
        add_compile_options(-mavx512f)
        message(STATUS "Applied Clang-cl x64 SIMD flags: -mavx512f")
    else()
        add_compile_options(-mavx2)
        message(STATUS "Applied Clang-cl x64 SIMD flags: -mavx2")
    endif()
endif()

# IMPORTANT: SIMD compile definitions are handled by cmake/SIMDDetection.cmake That system detects
# actual CPU capabilities and sets appropriate definitions: - LIBSTATS_HAS_SSE2, LIBSTATS_HAS_AVX,
# LIBSTATS_HAS_AVX2, etc. The definitions are applied through the libstats_simd_interface target.

# Unix compilers (GCC/Clang): SIMD flags are applied per-source-file via the configure_simd_target()
# function from cmake/SIMDDetection.cmake This provides precise control and avoids flag conflicts.

# Display configuration information (evaluate once at configure time) Cross-Compiler Build Type
# Status Messages
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build (${CMAKE_CXX_COMPILER_ID}): Optimized for performance")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build (${CMAKE_CXX_COMPILER_ID}): Full debugging information")
elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
    message(STATUS "Dev build (${CMAKE_CXX_COMPILER_ID}): Balanced optimization and debugging")
elseif(CMAKE_BUILD_TYPE STREQUAL "Strict")
    message(STATUS "Strict build (${CMAKE_CXX_COMPILER_ID}): strict warnings + -Werror")
else()
    message(STATUS "Default build (${CMAKE_CXX_COMPILER_ID}): standard configuration")
endif()

# SIMD Status Messages (compiler-specific)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(LIBSTATS_HAS_AVX512)
        message(STATUS "SIMD: AVX-512/AVX2/AVX/SSE2 enabled (MSVC x64)")
    else()
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (MSVC x64)")
    endif()
elseif(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND WIN32
    AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(LIBSTATS_HAS_AVX512)
        message(STATUS "SIMD: AVX-512/AVX2/AVX/SSE2 enabled (ClangCL x64)")
    else()
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (ClangCL x64)")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # GCC SIMD status messages
    set(SIMD_FEATURES "")
    if(LIBSTATS_HAS_SSE2)
        list(APPEND SIMD_FEATURES "SSE2")
    endif()
    if(LIBSTATS_HAS_AVX)
        list(APPEND SIMD_FEATURES "AVX")
    endif()
    if(LIBSTATS_HAS_AVX2)
        list(APPEND SIMD_FEATURES "AVX2")
    endif()
    if(LIBSTATS_HAS_AVX512)
        list(APPEND SIMD_FEATURES "AVX-512")
    endif()
    if(LIBSTATS_HAS_NEON)
        list(APPEND SIMD_FEATURES "NEON")
    endif()
    if(SIMD_FEATURES)
        string(JOIN ", " SIMD_LIST ${SIMD_FEATURES})
        message(STATUS "SIMD: ${SIMD_LIST} enabled (runtime-dispatched)")
    endif()
elseif(LIBSTATS_HAS_NEON)
    message(STATUS "SIMD: NEON enabled (${CMAKE_CXX_COMPILER_ID} ARM64)")
endif()

# =============================================================================
# DEPENDENCY-AWARE OBJECT LIBRARIES FOR OPTIMAL COMPILATION
# =============================================================================
# Organize sources into object libraries that align with header dependency levels This enables: -
# Parallel compilation of independent components - Minimal recompilation when headers change -
# Better incremental build performance - Clear architectural boundaries

# Level 0-1: Platform Foundation and Constants (No internal dependencies)
set(LIBSTATS_FOUNDATION_SOURCES
    src/cpu_detection.cpp # Runtime CPU feature detection
    src/cpu_vendor_detection.cpp # Phase 3D: CPU vendor-specific detection
    src/simd_policy.cpp # SIMD policy and feature detection
    src/platform_constants_impl.cpp # Platform constants PIMPL implementation (Phase 2)
)

# Level 2: Core Utilities (Depends on Level 0-1)
set(LIBSTATS_CORE_UTILITIES_SOURCES
    src/safety.cpp # Safe numerical operations
    src/math_utils.cpp # Mathematical utilities and special functions
    src/log_space_ops.cpp # Log-space arithmetic for numerical stability
)

# Level 2: Platform Capabilities (Depends on Level 0-1)
set(LIBSTATS_PLATFORM_SOURCES src/thread_pool.cpp # Thread pool implementation
                              src/work_stealing_pool.cpp # Advanced work-stealing thread pool
)

# Level 3: Advanced Infrastructure (Depends on Level 0-2)
set(LIBSTATS_INFRASTRUCTURE_SOURCES
    src/distribution_cache.cpp # Distribution-specific cache optimization
    src/benchmark.cpp # Performance benchmarking utilities
    src/performance_history.cpp # Performance data collection and analysis
    src/performance_dispatcher.cpp # Smart auto-dispatch system
    src/system_capabilities.cpp # System capability detection
    src/libstats_init.cpp # Performance system initialization
)

# Level 4: Distribution Framework (Depends on Level 0-3)
set(LIBSTATS_FRAMEWORK_SOURCES src/distribution_base.cpp # Complete distribution base framework
)

# Level 5: Distribution Implementations (Depends on Level 0-4)
set(LIBSTATS_DISTRIBUTIONS_SOURCES
    src/gaussian.cpp # Gaussian distribution implementation
    src/exponential.cpp # Exponential distribution implementation
    src/uniform.cpp # Uniform distribution implementation
    src/poisson.cpp # Poisson distribution implementation
    src/discrete.cpp # Discrete distribution implementation
    src/gamma.cpp # Gamma distribution implementation
    src/chi_squared.cpp # Chi-squared distribution (delegation wrapper over Gamma)
    src/student_t.cpp # Student's t distribution (full real-line, log-space SIMD)
    src/beta.cpp # Beta distribution (bounded support [0,1], two-log SIMD pipeline)
    src/lognormal.cpp # Log-normal distribution (log+exp pipeline, positive real-line support)
    src/pareto.cpp # Pareto distribution (log-only pipeline, power-law tail, x >= scale)
    src/weibull.cpp # Weibull distribution (log+exp pipeline, reliability engineering)
    src/rayleigh.cpp # Rayleigh distribution (x² pipeline, signal processing)
    src/von_mises.cpp # Von Mises distribution (circular, cos kernel; Bessel functions)
    src/binomial.cpp # Binomial distribution (discrete, lgamma log-space PMF, beta_i CDF)
    src/negative_binomial.cpp # Negative Binomial distribution (discrete, Newton-Raphson MLE,
                              # Gamma-Poisson sampling)
    src/gaussian_analysis.cpp # Gaussian-specific analysis
    src/poisson_analysis.cpp # Poisson-specific analysis
    src/exponential_analysis.cpp # Exponential-specific analysis
    src/gamma_analysis.cpp # Gamma-specific analysis
    src/binomial_analysis.cpp # Binomial-specific analysis
    src/discrete_analysis.cpp # Discrete-uniform-specific analysis
)

# SIMD sources are added separately by the SIMD detection system These compile with specific
# architecture flags and are platform-dependent

# =============================================================================
# INCLUDE PATH SHIM FOR INSTALLED HEADER PREFIX
# =============================================================================
# Headers use #include "libstats/core/foo.h" so the include path is identical in the build tree and
# the install tree (where headers land in include/libstats/).
#
# Implementation strategy (platform-guarded): macOS/Linux — a single directory symlink:
# build/include_shim/libstats → <src>/include Any edit to a header is immediately visible to the
# compiler; zero copy overhead per build. Re-running cmake is not required after header edits.
#
# Windows — a flat copy (symlinks require Developer Mode or elevated privileges and can trigger
# "untrusted mount point" errors): copy_directory at configure time, refreshed by an ALL custom
# target so mid-session header edits are picked up on the next cmake --build.
#
# Long-term: the canonical fix is to restructure include/ to include/libstats/ so the source tree
# mirrors the install tree and no shim is needed at all. Tracked as a post-v2.0 task.
set(LIBSTATS_INCLUDE_SHIM_DIR "${CMAKE_CURRENT_BINARY_DIR}/include_shim")
set(LIBSTATS_INCLUDE_SHIM_LINK "${LIBSTATS_INCLUDE_SHIM_DIR}/libstats")
file(MAKE_DIRECTORY "${LIBSTATS_INCLUDE_SHIM_DIR}")
file(REMOVE_RECURSE "${LIBSTATS_INCLUDE_SHIM_LINK}")
if(WIN32)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${LIBSTATS_INCLUDE_SHIM_LINK}" RESULT_VARIABLE LIBSTATS_INCLUDE_SHIM_RESULT)
    if(NOT LIBSTATS_INCLUDE_SHIM_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to materialise include shim at ${LIBSTATS_INCLUDE_SHIM_LINK}")
    endif()
    # Re-copy on every build so mid-session header edits are picked up without requiring a cmake
    # re-run. cmake -E copy_directory is timestamp-aware.
    add_custom_target(
        libstats_refresh_shim ALL
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${LIBSTATS_INCLUDE_SHIM_LINK}"
        COMMENT "Refreshing include shim"
        VERBATIM)
else()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E create_symlink "${CMAKE_CURRENT_SOURCE_DIR}/include"
                "${LIBSTATS_INCLUDE_SHIM_LINK}" RESULT_VARIABLE LIBSTATS_INCLUDE_SHIM_RESULT)
    if(NOT LIBSTATS_INCLUDE_SHIM_RESULT EQUAL 0)
        message(
            FATAL_ERROR "Failed to create include shim symlink at ${LIBSTATS_INCLUDE_SHIM_LINK}")
    endif()
endif()

# NEW-BS-1: Generate version header from CMake project() metadata so the runtime constants always
# match the version in CMakeLists.txt.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats_version.h.in"
               "${LIBSTATS_INCLUDE_SHIM_DIR}/libstats_version.h" @ONLY)

# =============================================================================
# CREATE OBJECT LIBRARIES FOR PARALLEL COMPILATION
# =============================================================================
# Object libraries allow parallel compilation and better dependency tracking

# Foundation object library (Level 0-1)
add_library(libstats_foundation_obj OBJECT ${LIBSTATS_FOUNDATION_SOURCES})
target_include_directories(libstats_foundation_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target

# Core utilities object library (Level 2)
add_library(libstats_core_utilities_obj OBJECT ${LIBSTATS_CORE_UTILITIES_SOURCES})
target_include_directories(libstats_core_utilities_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_core_utilities_obj libstats_foundation_obj)

# Platform capabilities object library (Level 2)
add_library(libstats_platform_obj OBJECT ${LIBSTATS_PLATFORM_SOURCES})
target_include_directories(libstats_platform_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_platform_obj libstats_foundation_obj)

# Advanced infrastructure object library (Level 3)
add_library(libstats_infrastructure_obj OBJECT ${LIBSTATS_INFRASTRUCTURE_SOURCES})
target_include_directories(libstats_infrastructure_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_infrastructure_obj libstats_foundation_obj libstats_core_utilities_obj
                 libstats_platform_obj)

# Distribution framework object library (Level 4)
add_library(libstats_framework_obj OBJECT ${LIBSTATS_FRAMEWORK_SOURCES})
target_include_directories(libstats_framework_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_framework_obj libstats_foundation_obj libstats_core_utilities_obj
                 libstats_platform_obj libstats_infrastructure_obj)

# Distribution implementations object library (Level 5)
add_library(libstats_distributions_obj OBJECT ${LIBSTATS_DISTRIBUTIONS_SOURCES})
target_include_directories(libstats_distributions_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_distributions_obj libstats_foundation_obj libstats_core_utilities_obj
                 libstats_platform_obj libstats_infrastructure_obj libstats_framework_obj)

# SIMD object library (Platform-dependent, can compile in parallel)
add_library(libstats_simd_obj OBJECT ${LIBSTATS_SIMD_SOURCES})
target_include_directories(libstats_simd_obj PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})
# SIMD definitions will be linked through interface target
add_dependencies(libstats_simd_obj libstats_foundation_obj) # SIMD depends on CPU detection

# =============================================================================
# COMBINED SOURCE LIST FOR FINAL LIBRARIES
# =============================================================================
# Collect all object library outputs for the final static/shared libraries
set(LIBSTATS_ALL_OBJECTS
    $<TARGET_OBJECTS:libstats_foundation_obj>
    $<TARGET_OBJECTS:libstats_core_utilities_obj>
    $<TARGET_OBJECTS:libstats_platform_obj>
    $<TARGET_OBJECTS:libstats_infrastructure_obj>
    $<TARGET_OBJECTS:libstats_framework_obj>
    $<TARGET_OBJECTS:libstats_distributions_obj>
    $<TARGET_OBJECTS:libstats_simd_obj>)

# Header-only library (for header-only components)
add_library(libstats_headers INTERFACE)
target_include_directories(
    libstats_headers INTERFACE $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
                               $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/libstats>)

# Static library using object libraries for better build performance
add_library(libstats_static STATIC ${LIBSTATS_ALL_OBJECTS})
target_include_directories(
    libstats_static PUBLIC $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
                           $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/libstats>)
# LP-13: VERSION/SOVERSION are meaningful only for shared libraries (they affect dylib naming and
# .so symlinks). Setting them on a static library is a no-op that CMake silently ignores, but it is
# misleading. Keep only OUTPUT_NAME.
if(WIN32)
    set_target_properties(libstats_static PROPERTIES OUTPUT_NAME stats_static)
else()
    set_target_properties(libstats_static PROPERTIES OUTPUT_NAME stats)
endif()

# Dynamic/shared library using object libraries for better build performance
add_library(libstats_shared SHARED ${LIBSTATS_ALL_OBJECTS})
target_include_directories(
    libstats_shared PUBLIC $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
                           $<INSTALL_INTERFACE:include> $<INSTALL_INTERFACE:include/libstats>)
set_target_properties(
    libstats_shared
    PROPERTIES OUTPUT_NAME "stats"
               VERSION ${PROJECT_VERSION}
               SOVERSION ${PROJECT_VERSION_MAJOR})

# Platform-specific shared library properties
if(APPLE)
    set_target_properties(libstats_shared PROPERTIES MACOSX_RPATH ON INSTALL_NAME_DIR "@rpath")

    # Ad-hoc code signing for macOS Library Validation. Required on Catalina (10.15) and later:
    # unsigned dylibs are rejected by the system's Library Validation policy. Ad-hoc signing (-s -)
    # applies a local signature without a Developer ID, which satisfies Library Validation for
    # locally-built libraries. This is safe on all macOS versions.
    find_program(CODESIGN_EXECUTABLE codesign)
    if(CODESIGN_EXECUTABLE)
        add_custom_command(
            TARGET libstats_shared
            POST_BUILD
            COMMAND ${CODESIGN_EXECUTABLE} -s - $<TARGET_FILE:libstats_shared>
            COMMENT
                "Ad-hoc signing ${CMAKE_SHARED_LIBRARY_PREFIX}stats${CMAKE_SHARED_LIBRARY_SUFFIX} for macOS Library Validation"
        )
    endif()
elseif(UNIX AND NOT APPLE)
    set_target_properties(libstats_shared PROPERTIES SOVERSION ${PROJECT_VERSION_MAJOR})
endif()

# Create SIMD interface target for modern CMake approach
create_simd_interface_target()

# LP-4: scope LIBSTATS_HAS_CXX17_BESSEL to the SIMD interface target so it propagates to all
# consumers via configure_simd_target() rather than being a global definition.
if(LIBSTATS_HAS_CXX17_BESSEL)
    target_compile_definitions(libstats_simd_interface INTERFACE LIBSTATS_HAS_CXX17_BESSEL)
endif()

# Apply per-source-file SIMD compile flags once (file-global properties).
apply_simd_source_flags()

# Link the SIMD interface target to every object library and final library so all TUs receive the
# LIBSTATS_HAS_* compile definitions.
configure_simd_target(libstats_foundation_obj)
configure_simd_target(libstats_core_utilities_obj)
configure_simd_target(libstats_platform_obj)
configure_simd_target(libstats_infrastructure_obj)
configure_simd_target(libstats_framework_obj)
configure_simd_target(libstats_distributions_obj)
configure_simd_target(libstats_simd_obj)

# Configure SIMD compilation for final targets
configure_simd_target(libstats_static)
configure_simd_target(libstats_shared)

# POSIX thread functions are used by platform sources on Unix-like systems. NEW-BS-4: also link to
# platform_obj whose sources (thread_pool, work_stealing_pool) reference pthread directly and may be
# consumed without the final library targets.
if(TARGET Threads::Threads)
    target_link_libraries(libstats_platform_obj PRIVATE Threads::Threads)
    target_link_libraries(libstats_static PUBLIC Threads::Threads)
    target_link_libraries(libstats_shared PUBLIC Threads::Threads)
endif()

# Platform-specific library linking: rpath and signing are handled per-target above. -stdlib=libc++
# is set via add_compile_options/add_link_options in the platform block.

# LP-2 / BS-4: target-scope TBB includes, link-dirs, and compile flags.
if(LIBSTATS_HAS_TBB AND DEFINED LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL)
    target_include_directories(libstats_static PUBLIC ${LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL})
    target_include_directories(libstats_shared PUBLIC ${LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL})
    if(DEFINED LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL AND NOT "${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL}"
                                                      STREQUAL "")
        target_link_directories(libstats_static PUBLIC ${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL})
        target_link_directories(libstats_shared PUBLIC ${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL})
    endif()
    # BS-4: apply any pkg-config compile flags (e.g. -DTBB_USE_GCC_BUILTINS) to the library targets
    # only, not globally via CMAKE_CXX_FLAGS.
    if(DEFINED LIBSTATS_TBB_CFLAGS_INTERNAL AND NOT "${LIBSTATS_TBB_CFLAGS_INTERNAL}" STREQUAL "")
        target_compile_options(libstats_static PRIVATE ${LIBSTATS_TBB_CFLAGS_INTERNAL})
        target_compile_options(libstats_shared PRIVATE ${LIBSTATS_TBB_CFLAGS_INTERNAL})
    endif()
endif()

# Create aliases for easier consumption
add_library(libstats::headers ALIAS libstats_headers)
add_library(libstats::static ALIAS libstats_static)
add_library(libstats::shared ALIAS libstats_shared)

# Test infrastructure
option(LIBSTATS_BUILD_TESTS "Build tests" ON)

if(LIBSTATS_BUILD_TESTS)
    # Create tests directory in build folder
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)

    # GTest detection — three-step: generic find_package, then Apple/Homebrew probe, then
    # FetchContent as a self-contained fallback (matches libhmm test pattern).

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
            PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
                       RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tests/Debug
                       RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tests/Release
                       RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
                       ${CMAKE_CURRENT_BINARY_DIR}/tests/RelWithDebInfo
                       RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
                       ${CMAKE_CURRENT_BINARY_DIR}/tests/MinSizeRel)

        # Add include directories
        target_include_directories(
            ${TEST_NAME}
            PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/include)

        # Platform-specific rpath is inherited from libstats_static/shared via PUBLIC link options.
        # No per-target rpath needed here (adding it causes duplicate -rpath linker warnings).

        # Link to SIMD interface target for modern CMake approach
        if(TARGET libstats_simd_interface)
            target_link_libraries(${TEST_NAME} PRIVATE libstats::simd)
        endif()

        # Add to test suite
        add_test(NAME ${TEST_NAME} COMMAND $<TARGET_FILE:${TEST_NAME}>)

        # Set working directory for tests
        set_tests_properties(${TEST_NAME} PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endfunction()

    # Function to create a test executable with proper flags (static library)
    function(create_libstats_test TEST_NAME SOURCE_FILE)
        add_executable(${TEST_NAME} ${SOURCE_FILE})

        # Link against static library by default for tests
        target_link_libraries(${TEST_NAME} PRIVATE libstats_static)

        # Link TBB if available
        if(LIBSTATS_HAS_TBB)
            target_link_libraries(${TEST_NAME} PRIVATE ${TBB_LIBRARIES})
        endif()

        # Apply common test settings
        configure_common_test_settings(${TEST_NAME})
    endfunction()

    # Function to create a test executable linked to dynamic library.
    #
    # On Windows, copy stats.dll next to each dynamic test EXE so runtime loading does not depend on
    # PATH. Combined with per-config output directories above, this avoids both STATUS_DLL_NOT_FOUND
    # and stale Debug/Release binary collisions.
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

    # Enable testing
    enable_testing()

    # =======================================================================
    # LEVEL 0: Foundational Tests (Platform & Constants)
    # =======================================================================
    # These test platform capabilities and constants - most critical

    # Basic constants and platform tests
    create_libstats_gtest(test_constants tests/test_constants.cpp)
    create_libstats_test(test_cpu_detection tests/test_cpu_detection.cpp)
    create_libstats_test(test_simd_comprehensive tests/test_simd_comprehensive.cpp)
    create_libstats_gtest(test_platform_optimizations tests/test_platform_optimizations.cpp)
    create_libstats_gtest(test_simd_policy tests/test_simd_policy.cpp)

    # SIMD benchmark test (on-demand only, not in regular test suite due to runtime)
    create_libstats_test(benchmark_simd_all tests/benchmark_simd_all.cpp)

    # =======================================================================
    # LEVEL 1: Core Infrastructure Tests
    # =======================================================================
    # Core mathematical and safety functionality

    # Consolidated numerical safety (safety.h + log_space_ops.h)
    create_libstats_gtest(test_numerical_safety tests/test_numerical_safety.cpp)

    # Consolidated error handling (error_handling.h)
    create_libstats_gtest(test_error_handling_comprehensive
                          tests/test_error_handling_comprehensive.cpp)

    # Consolidated math utilities (math_utils.h)
    create_libstats_gtest(test_math_comprehensive tests/test_math_comprehensive.cpp)

    # =======================================================================
    # LEVEL 2: Statistical Infrastructure
    # =======================================================================
    # Statistical validation and goodness-of-fit

    # Goodness-of-fit tests
    create_libstats_gtest(test_goodness_of_fit tests/test_goodness_of_fit.cpp)

    # LEVEL 2b: Platform Capabilities thread_pool.h and work_stealing_pool.h - Thread pool
    # implementations
    create_libstats_gtest(test_thread_pool tests/test_thread_pool.cpp)
    # <concepts> and <ranges> are guaranteed on the v2.0.0 baseline — always register.
    create_libstats_gtest(test_work_stealing_pool tests/test_work_stealing_pool.cpp)
    set(LIBSTATS_HAS_WORK_STEALING_POOL_TEST TRUE)

    # =======================================================================
    # LEVEL 3 TESTS: Advanced Infrastructure
    # =======================================================================
    # Test advanced infrastructure components that depend on Level 0-2

    # Performance Infrastructure

    # parallel_execution.h - C++20 parallel execution with automatic fallback
    create_libstats_gtest(test_parallel_execution_integration
                          tests/test_parallel_execution_integration.cpp)
    create_libstats_gtest(test_parallel_execution_comprehensive
                          tests/test_parallel_execution_comprehensive.cpp)

    # benchmark.h - Performance measurement utilities
    create_libstats_test(test_benchmark_basic tests/test_benchmark_basic.cpp)

    # benchmark.h - Comprehensive benchmark infrastructure testing
    create_libstats_gtest(test_benchmark tests/test_benchmark.cpp)

    # PHASE 3 PERFORMANCE OPTIMIZATION FRAMEWORK (NEW TESTS) performance_history.h - Performance
    # data collection and analysis
    create_libstats_gtest(test_performance_history tests/test_performance_history.cpp)

    # performance_dispatcher.h - Smart auto-dispatch system
    create_libstats_gtest(test_performance_dispatcher tests/test_performance_dispatcher.cpp)

    # system_capabilities.cpp - System capability detection and integration
    create_libstats_gtest(test_system_capabilities tests/test_system_capabilities.cpp)

    # Performance system initialization
    create_libstats_gtest(test_performance_initialization tests/test_performance_initialization.cpp)

    # =======================================================================
    # LEVEL 4 TESTS: Complete Distribution Framework
    # =======================================================================
    # Test the integrated distribution framework (distribution_base.h)

    # distribution_base.h - Complete base class implementation (Tested indirectly through
    # distribution implementations, but could add direct tests)

    # =======================================================================
    # LEVEL 5 TESTS: Concrete Distribution Implementations
    # =======================================================================
    # Test concrete distribution implementations using the complete framework

    # gaussian.h - Gaussian/Normal distribution implementation (CONSOLIDATED)
    create_libstats_test(test_gaussian_basic tests/test_gaussian_basic.cpp)
    create_libstats_gtest(test_gaussian_enhanced tests/test_gaussian_enhanced.cpp)
    create_libstats_gtest(test_gaussian_analysis tests/test_gaussian_analysis.cpp)
    create_libstats_gtest(test_poisson_analysis tests/test_poisson_analysis.cpp)
    create_libstats_gtest(test_distribution_analysis tests/test_distribution_analysis.cpp)
    create_libstats_gtest(test_batch_math_regressions tests/test_batch_math_regressions.cpp)

    # exponential.h - Exponential distribution implementation (CONSOLIDATED)
    create_libstats_test(test_exponential_basic tests/test_exponential_basic.cpp)
    create_libstats_gtest(test_exponential_enhanced tests/test_exponential_enhanced.cpp)

    # discrete.h - Discrete uniform distribution implementation (CONSOLIDATED)
    create_libstats_test(test_discrete_basic tests/test_discrete_basic.cpp)
    create_libstats_gtest(test_discrete_enhanced tests/test_discrete_enhanced.cpp)

    # Atomic parameters test - Test lock-free parameter access and invalidation for all
    # distributions
    create_libstats_gtest(test_atomic_parameters tests/test_atomic_parameters.cpp)

    # uniform.h - Uniform distribution implementation
    create_libstats_test(test_uniform_basic tests/test_uniform_basic.cpp)
    create_libstats_gtest(test_uniform_enhanced tests/test_uniform_enhanced.cpp)

    # poisson.h - Poisson distribution implementation
    create_libstats_test(test_poisson_basic tests/test_poisson_basic.cpp)
    create_libstats_gtest(test_poisson_enhanced tests/test_poisson_enhanced.cpp)

    # gamma.h - Gamma distribution implementation
    create_libstats_test(test_gamma_basic tests/test_gamma_basic.cpp)
    create_libstats_gtest(test_gamma_enhanced tests/test_gamma_enhanced.cpp)

    # chi_squared.h - Chi-squared distribution (delegation wrapper over Gamma)
    create_libstats_test(test_chi_squared_basic tests/test_chi_squared_basic.cpp)
    create_libstats_gtest(test_chi_squared_enhanced tests/test_chi_squared_enhanced.cpp)

    # student_t.h - Student's t distribution (full real-line, log-space SIMD)
    create_libstats_test(test_student_t_basic tests/test_student_t_basic.cpp)
    create_libstats_gtest(test_student_t_enhanced tests/test_student_t_enhanced.cpp)

    # beta.h - Beta distribution (bounded support [0,1], two-log SIMD pipeline)
    create_libstats_test(test_beta_basic tests/test_beta_basic.cpp)
    create_libstats_gtest(test_beta_enhanced tests/test_beta_enhanced.cpp)

    # lognormal.h - Log-normal distribution (log+exp pipeline, positive real-line)
    create_libstats_test(test_lognormal_basic tests/test_lognormal_basic.cpp)
    create_libstats_gtest(test_lognormal_enhanced tests/test_lognormal_enhanced.cpp)

    # pareto.h - Pareto distribution (log-only pipeline, power-law tail)
    create_libstats_test(test_pareto_basic tests/test_pareto_basic.cpp)
    create_libstats_gtest(test_pareto_enhanced tests/test_pareto_enhanced.cpp)

    # weibull.h - Weibull distribution (log+exp pipeline, reliability engineering)
    create_libstats_test(test_weibull_basic tests/test_weibull_basic.cpp)
    create_libstats_gtest(test_weibull_enhanced tests/test_weibull_enhanced.cpp)

    # rayleigh.h - Rayleigh distribution (x² pipeline, signal processing)
    create_libstats_test(test_rayleigh_basic tests/test_rayleigh_basic.cpp)
    create_libstats_gtest(test_rayleigh_enhanced tests/test_rayleigh_enhanced.cpp)

    # von_mises.h - Von Mises distribution (circular, Bessel functions; PARALLEL preferred)
    create_libstats_test(test_von_mises_basic tests/test_von_mises_basic.cpp)
    create_libstats_gtest(test_von_mises_enhanced tests/test_von_mises_enhanced.cpp)

    # binomial.h - Binomial distribution (discrete, lgamma log-space PMF, beta_i CDF)
    create_libstats_test(test_binomial_basic tests/test_binomial_basic.cpp)
    create_libstats_gtest(test_binomial_enhanced tests/test_binomial_enhanced.cpp)

    # negative_binomial.h - Negative Binomial distribution (discrete, Newton-Raphson MLE)
    create_libstats_test(test_negative_binomial_basic tests/test_negative_binomial_basic.cpp)
    create_libstats_gtest(test_negative_binomial_enhanced tests/test_negative_binomial_enhanced.cpp)

    # =======================================================================
    # ADDITIONAL TESTS: Cross-cutting concerns and integration
    # =======================================================================
    # These test cross-cutting functionality and integration scenarios

    # Copy/move semantics stress tests - Now re-enabled as distributions are complete
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_stress.cpp)
        create_libstats_test(test_copy_move_stress tests/test_copy_move_stress.cpp)
        set(HAS_COPY_MOVE_STRESS TRUE)
    else()
        set(HAS_COPY_MOVE_STRESS FALSE)
    endif()

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_fix.cpp)
        create_libstats_test(test_copy_move_fix tests/test_copy_move_fix.cpp)
        set(HAS_COPY_MOVE_FIX TRUE)
    else()
        set(HAS_COPY_MOVE_FIX FALSE)
    endif()

    # test_new_constants removed - functionality is covered by test_constants --platform --simd

    # Dynamic library linking tests
    create_libstats_test_dynamic(test_dynamic_linking tests/test_dynamic_linking.cpp)

    # Dynamic versions of all _simple tests Note: SIMD tests consolidated into
    # test_simd_comprehensive
    create_libstats_test_dynamic(test_gaussian_basic_dynamic tests/test_gaussian_basic.cpp)
    create_libstats_test_dynamic(test_exponential_basic_dynamic tests/test_exponential_basic.cpp)

    # =======================================================================
    # TEST LABELS
    # =======================================================================
    # Tests containing timing-sensitive assertions (speedup thresholds, overhead ratios) are
    # labelled "timing". These assertions can fail under CPU contention when multiple tests run in
    # parallel. All tests are correct regardless of label; the label only affects whether timing
    # ASSERTIONS are reliable.
    #
    # Usage: ctest -j8             # All tests; timing assertions may be flaky under load ctest -j8
    # -LE timing  # Correctness only; excludes tests with timing assertions ctest -j1 -L timing   #
    # Timing validation; run serially for accurate measurement
    #
    # benchmark_simd_all is labelled separately as it is never part of the standard correctness
    # suite. GTest-based tests only exist when GTest is available. set_tests_properties is a hard
    # CMake error on non-existent tests, so guard this block -- unlike DEPENDS which tolerates
    # missing targets.
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
            test_system_capabilities # runs live SIMD/threading/bandwidth benchmarks
            PROPERTIES LABELS "timing")
        set_tests_properties(test_exponential_enhanced test_uniform_enhanced test_poisson_enhanced
                             test_discrete_enhanced test_gamma_enhanced PROPERTIES LABELS "timing")
    endif()
    set_tests_properties(benchmark_simd_all PROPERTIES LABELS "benchmark")

    # Test target for running all tests (organized by dependency hierarchy) This target runs ALL
    # tests including stress/diagnostic tests
    add_custom_target(
        run_all_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS # Level 0: Foundational tests (most critical)
                test_constants
                test_cpu_detection
                test_simd_comprehensive
                test_platform_optimizations
                test_simd_policy
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
                test_performance_history
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
                # Level 7: Integration tests
                test_dynamic_linking
                # On-demand tests (stress, diagnostic, redundant)
                test_parallel_execution_comprehensive
                test_benchmark
                test_gaussian_basic_dynamic
                test_exponential_basic_dynamic
                $<$<BOOL:${HAS_COPY_MOVE_STRESS}>:test_copy_move_stress>
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

    message(STATUS "Test infrastructure enabled")
else()
    message(STATUS "Test infrastructure disabled")
endif()

# Tools and utilities
option(LIBSTATS_BUILD_TOOLS "Build utility tools" ON)

if(LIBSTATS_BUILD_TOOLS)
    # Helper function to create tool executables with consistent configuration
    function(add_libstats_tool tool_name source_file)
        add_executable(${tool_name} tools/${source_file})
        target_link_libraries(${tool_name} PRIVATE libstats_static)
        target_include_directories(${tool_name} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR}
                                                        ${CMAKE_CURRENT_SOURCE_DIR}/src)

        # Link to SIMD interface target for modern CMake approach
        if(TARGET libstats_simd_interface)
            target_link_libraries(${tool_name} PRIVATE libstats::simd)
        endif()

        set_target_properties(${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      ${CMAKE_CURRENT_BINARY_DIR}/tools)
    endfunction()

    # Helper function for standalone tools that don't need libstats
    function(add_standalone_tool tool_name source_file)
        add_executable(${tool_name} tools/${source_file})
        # No libstats linking - pure standard library tools Still needs include paths for
        # tool_utils.h → libstats headers
        target_include_directories(${tool_name} PRIVATE ${LIBSTATS_INCLUDE_SHIM_DIR})

        set_target_properties(${tool_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                      ${CMAKE_CURRENT_BINARY_DIR}/tools)
    endfunction()

    # System Information & Diagnostics Tools
    add_libstats_tool(system_inspector system_inspector.cpp)
    add_standalone_tool(cpp20_features_inspector cpp20_features_inspector.cpp)

    # Performance & Benchmarking Tools
    add_libstats_tool(parallel_batch_fitting_benchmark parallel_batch_fitting_benchmark.cpp)
    add_libstats_tool(strategy_profile strategy_profile.cpp)
    add_libstats_tool(simd_verification simd_verification.cpp)
    add_libstats_tool(parallel_correctness_verification parallel_correctness_verification.cpp)

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
endif()

# Examples
if(LIBSTATS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install targets
install(
    TARGETS libstats_headers libstats_static libstats_shared libstats_simd_interface
    EXPORT libstats-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES
    DESTINATION include)

# Install headers (exclude editor artifacts and macOS metadata)
install(
    DIRECTORY include/
    DESTINATION include/libstats
    PATTERN ".DS_Store" EXCLUDE
    PATTERN "*.bak" EXCLUDE
    PATTERN "*.bak2" EXCLUDE
    PATTERN ".gitkeep" EXCLUDE)

# Install CMake config files for find_package support
install(
    EXPORT libstats-targets
    FILE libstats-targets.cmake
    NAMESPACE libstats::
    DESTINATION lib/cmake/libstats)

# Create and install config file
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR include)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    INSTALL_DESTINATION lib/cmake/libstats
    PATH_VARS INCLUDE_INSTALL_DIR)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
        DESTINATION lib/cmake/libstats)

# ===============================================================================
# Documentation Generation with Doxygen
# ===============================================================================
option(BUILD_DOCS "Build documentation using Doxygen" OFF)

if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        message(STATUS "Doxygen found: ${DOXYGEN_VERSION}")

        # Configure Doxyfile with project variables
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
                       "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)

        # Add documentation target
        add_custom_target(
            docs
            COMMAND ${DOXYGEN_EXECUTABLE} "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM)

        message(STATUS "Documentation target 'docs' configured")
        message(STATUS "  Build with: cmake --build . --target docs")
        message(STATUS "  Output will be in: ${CMAKE_CURRENT_BINARY_DIR}/docs/html")
    else()
        message(WARNING "Doxygen not found - documentation target will not be available")
    endif()
endif()

# ===============================================================================
# pkg-config file generation
# ===============================================================================
if(UNIX AND NOT APPLE) # pkg-config is primarily used on Linux
    set(LIBSTATS_PC_LIBS_PRIVATE "-pthread")
    if(LIBSTATS_HAS_TBB)
        string(APPEND LIBSTATS_PC_LIBS_PRIVATE " -ltbb")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats.pc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc" @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc" DESTINATION lib/pkgconfig)

    message(STATUS "pkg-config file will be installed to lib/pkgconfig/")
elseif(APPLE)
    # On macOS with Homebrew, also generate pkg-config
    set(LIBSTATS_PC_LIBS_PRIVATE "")
    if(LIBSTATS_HAS_TBB)
        string(APPEND LIBSTATS_PC_LIBS_PRIVATE " -ltbb")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats.pc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc" @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc" DESTINATION lib/pkgconfig)
endif()

# Print summary
message(STATUS "libstats Configuration Summary:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
if(LIBSTATS_BUILD_TESTS)
    message(STATUS "  Testing: Enabled")
    if(GTEST_FOUND)
        message(STATUS "  GTest: Available")
    else()
        message(STATUS "  GTest: Not found - using standalone tests")
    endif()
else()
    message(STATUS "  Testing: Disabled")
endif()
