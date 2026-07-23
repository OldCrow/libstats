cmake_minimum_required(VERSION 3.25)

# Set CMake policies
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW) # find_package uses upper-case <PACKAGENAME>_ROOT variables
endif()

# Verbose messaging control
option(LIBSTATS_VERBOSE_BUILD "Enable verbose build messages for debugging" OFF)
# LIBSTATS_PORTABLE: skip the global /arch:AVX512 / /arch:AVX2 flag on Windows MSVC / Clang-cl
# builds so that the binary is safe to distribute across machines with different ISA capabilities.
# Per-file ISA flags on the SIMD kernel TUs (applied by
# cmake/SIMDDetection.cmake:apply_simd_source_flags) remain active and unaffected.  Default OFF
# (native performance build). Recommended ON for pylibstats wheel builds.  See libstats issue
# tracker for the long-term plan to extend per-file flags to all TUs on Windows.
option(
    LIBSTATS_PORTABLE
    "Skip global /arch: SIMD flag on Windows builds (safe for distributable wheels; OFF by default)"
    OFF)

# Threading system preference control
option(LIBSTATS_FORCE_TBB
       "Force TBB usage even on platforms with native threading (e.g., GCD on macOS)" OFF)
option(
    LIBSTATS_PREFER_PLATFORM_THREADING
    "Disable OpenMP when a platform-native threading system (GCD on macOS, Windows Thread Pool \
on Windows) is available, preventing over-subscription from two independent thread pools. \
Default OFF preserves the existing behaviour."
    OFF)

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

# libstats v2.x supports macOS 13 Ventura and newer.  Do not force every local build to target the
# minimum: when CMAKE_OSX_DEPLOYMENT_TARGET is unset, let AppleClang/CMake use the host/compiler
# default. This avoids linker warnings when local dependencies such as Homebrew GTest were built for
# the host macOS version (for example 15.0).  Packagers who need Ventura-compatible binaries should
# set -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 explicitly.
set(LIBSTATS_MIN_MACOS_DEPLOYMENT_TARGET "13.0")
if(APPLE)
    if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET AND NOT "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
        if(CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS LIBSTATS_MIN_MACOS_DEPLOYMENT_TARGET)
            message(
                FATAL_ERROR
                    "libstats v2.x requires CMAKE_OSX_DEPLOYMENT_TARGET >= "
                    "${LIBSTATS_MIN_MACOS_DEPLOYMENT_TARGET}; got ${CMAKE_OSX_DEPLOYMENT_TARGET}")
        endif()
        message(STATUS "macOS deployment target: ${CMAKE_OSX_DEPLOYMENT_TARGET} "
                       "(minimum supported: ${LIBSTATS_MIN_MACOS_DEPLOYMENT_TARGET})")
    else()
        message(STATUS "macOS deployment target: compiler default "
                       "(minimum supported: ${LIBSTATS_MIN_MACOS_DEPLOYMENT_TARGET}; "
                       "set -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 for Ventura-compatible binaries)")
    endif()
endif()

project(
    libstats
    VERSION 2.1.0
    LANGUAGES CXX)

# GNUInstallDirs gives platform-correct, packager-overridable destinations (e.g. lib64 on some Linux
# distros) instead of hardcoded lib/bin/include. Included here (rather than down by the install()
# rules) because the $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> genexes on
# libstats_headers/libstats_static/libstats_shared substitute CMAKE_INSTALL_INCLUDEDIR as a plain
# variable at the target_include_directories() call site — it must already be set by then, or it
# silently expands to an empty install interface.
include(GNUInstallDirs)

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

# Threading system detection (detect_threading_systems, detect_tbb_unified) lives in
# cmake/Threading.cmake — included here (function definitions only; call sites and the macOS
# GCD-preference logic below remain inline at their original positions so that configure-time
# message ordering relative to SIMD detection is unchanged).
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Threading.cmake")

detect_threading_systems()
# SIMD feature detection and compilation flags Use our comprehensive SIMD detection system
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SIMDDetection.cmake")

# Set options for SIMD detection behavior
option(LIBSTATS_ENABLE_RUNTIME_CHECKS "Enable runtime CPU checks even when cross-compiling" OFF)
# Cap the highest *compiled* x86 SIMD tier. Runtime dispatch always selects the highest tier the CPU
# supports, so on an AVX2 box the AVX and SSE2 kernels never execute. Setting this to SSE2 or AVX
# disables the tiers above it (source + define removed) so the capped tier becomes the dispatch
# ceiling — useful for validating lower-tier kernels natively. Accepts SSE2|AVX|AVX2|AVX512 (empty =
# no cap). Honoured by detect_simd_features() -> apply_simd_tier_cap() in SIMDDetection.cmake.
set(LIBSTATS_MAX_SIMD_TIER
    ""
    CACHE STRING "Cap the highest compiled x86 SIMD tier (SSE2|AVX|AVX2|AVX512; empty = no cap)")
set_property(CACHE LIBSTATS_MAX_SIMD_TIER PROPERTY STRINGS "" SSE2 AVX AVX2 AVX512)
# Perform comprehensive SIMD detection
detect_simd_features()

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

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompilerFlags.cmake")

# =============================================================================
# SIMD COMPILATION FLAGS - TRADITIONAL APPROACH (MODERNIZED)
# =============================================================================
# Replaced problematic nested generator expressions that can cause all flags to be passed to
# compilers. Using traditional conditionals that are proven to work.
#
# STRATEGY: - MSVC/Clang-cl on Windows: Use global flags (simple, reliable) - GCC/Clang on Unix: Use
# source-file-specific flags (cmake/SIMDDetection.cmake) - All platforms: Definitions are set by
# SIMDDetection.cmake based on detection

# Windows compilers: apply the build-machine SIMD level as a global flag. LIBSTATS_PORTABLE=ON skips
# this block so the binary is distributable to machines with lower ISA than the build host. Per-file
# flags from cmake/SIMDDetection.cmake:apply_simd_source_flags() remain active.
if(NOT LIBSTATS_PORTABLE)
    # NOTE: the global flag must follow the (possibly LIBSTATS_MAX_SIMD_TIER-capped) LIBSTATS_HAS_*
    # results from detect_simd_features(), not just "AVX512 or else AVX2". Applying /arch:AVX2
    # globally regardless of a lower cap predefines the compiler's __AVX2__ macro for every TU
    # (including src/simd_dispatch.cpp), which include/platform/simd.h's MSVC fallback then reads to
    # auto-#define LIBSTATS_HAS_AVX2 -- silently overriding an AVX or SSE2 cap and causing
    # getDispatchTable() to reference vector_*_avx2 symbols that were never compiled (simd_avx2.cpp
    # is excluded by the cap), producing LNK2019 unresolved externals.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        if(LIBSTATS_HAS_AVX512)
            add_compile_options(/arch:AVX512)
            message(STATUS "Applied MSVC x64 SIMD flags: /arch:AVX512")
        elseif(LIBSTATS_HAS_AVX2)
            add_compile_options(/arch:AVX2)
            message(STATUS "Applied MSVC x64 SIMD flags: /arch:AVX2")
        elseif(LIBSTATS_HAS_AVX)
            add_compile_options(/arch:AVX)
            message(STATUS "Applied MSVC x64 SIMD flags: /arch:AVX")
        else()
            message(STATUS "Applied MSVC x64 SIMD flags: none (SSE2 is the x64 baseline)")
        endif()
    elseif(
        CMAKE_CXX_COMPILER_ID MATCHES "Clang"
        AND WIN32
        AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        if(LIBSTATS_HAS_AVX512)
            add_compile_options(-mavx512f)
            message(STATUS "Applied Clang-cl x64 SIMD flags: -mavx512f")
        elseif(LIBSTATS_HAS_AVX2)
            add_compile_options(-mavx2)
            message(STATUS "Applied Clang-cl x64 SIMD flags: -mavx2")
        elseif(LIBSTATS_HAS_AVX)
            add_compile_options(-mavx)
            message(STATUS "Applied Clang-cl x64 SIMD flags: -mavx")
        else()
            message(STATUS "Applied Clang-cl x64 SIMD flags: none (SSE2 is the x64 baseline)")
        endif()
    endif()
else()
    message(STATUS "LIBSTATS_PORTABLE=ON: skipping global Windows /arch: flag "
                   "(SIMD kernel TUs still use per-file flags from SIMDDetection.cmake)")
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
    elseif(LIBSTATS_HAS_AVX2)
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (MSVC x64)")
    elseif(LIBSTATS_HAS_AVX)
        message(STATUS "SIMD: AVX/SSE2 enabled (MSVC x64)")
    else()
        message(STATUS "SIMD: SSE2 enabled (MSVC x64)")
    endif()
elseif(
    CMAKE_CXX_COMPILER_ID MATCHES "Clang"
    AND WIN32
    AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    if(LIBSTATS_HAS_AVX512)
        message(STATUS "SIMD: AVX-512/AVX2/AVX/SSE2 enabled (ClangCL x64)")
    elseif(LIBSTATS_HAS_AVX2)
        message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (ClangCL x64)")
    elseif(LIBSTATS_HAS_AVX)
        message(STATUS "SIMD: AVX/SSE2 enabled (ClangCL x64)")
    else()
        message(STATUS "SIMD: SSE2 enabled (ClangCL x64)")
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
    src/cpu_tier.cpp # CPU tier classification (absorbs vendor-string cascade)
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
    src/geometric.cpp # Geometric distribution (discrete delegation wrapper over NegBinomial(r=1))
    src/laplace.cpp # Laplace distribution (double-exponential, standalone)
    src/cauchy.cpp # Cauchy distribution (delegation wrapper over StudentT(nu=1))
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

# Windows: ensure the shim copy finishes before any cl.exe job opens the copied headers. MSBuild
# runs ALL custom targets in parallel with compilation; without this ordering the copy_directory and
# compiler reads race on the same files, producing "Permission denied".
if(WIN32)
    foreach(
        _shim_dep
        libstats_foundation_obj
        libstats_core_utilities_obj
        libstats_platform_obj
        libstats_infrastructure_obj
        libstats_framework_obj
        libstats_distributions_obj
        libstats_simd_obj)
        add_dependencies(${_shim_dep} libstats_refresh_shim)
    endforeach()
endif()

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
    libstats_headers
    INTERFACE $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libstats>)

# Static library using object libraries for better build performance
add_library(libstats_static STATIC ${LIBSTATS_ALL_OBJECTS})
target_include_directories(
    libstats_static
    PUBLIC $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libstats>)
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
    libstats_shared
    PUBLIC $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libstats>)
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

# LP-2 / BS-4: target-scope TBB includes, link-dirs, compile flags, and transitive linkage. C++20
# parallel STL implementations on GCC/Linux route execution-policy algorithms through TBB's backend.
# Link TBB through the final library targets PUBLICLY so every libstats consumer receives the
# required TBB dependency transitively.  Modern find_package(TBB) provides TBB::tbb; pkg-config is
# retained as a fallback for package managers that do not ship CMake config files.
if(LIBSTATS_HAS_TBB)
    if(TARGET TBB::tbb)
        target_link_libraries(libstats_static PUBLIC TBB::tbb)
        target_link_libraries(libstats_shared PUBLIC TBB::tbb)
    else()
        if(DEFINED LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL)
            target_include_directories(libstats_static PUBLIC ${LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL})
            target_include_directories(libstats_shared PUBLIC ${LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL})
        endif()
        if(DEFINED LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL
           AND NOT "${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL}" STREQUAL "")
            target_link_directories(libstats_static PUBLIC ${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL})
            target_link_directories(libstats_shared PUBLIC ${LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL})
        endif()
        if(DEFINED LIBSTATS_TBB_LIBRARIES_INTERNAL AND NOT "${LIBSTATS_TBB_LIBRARIES_INTERNAL}"
                                                       STREQUAL "")
            target_link_libraries(libstats_static PUBLIC ${LIBSTATS_TBB_LIBRARIES_INTERNAL})
            target_link_libraries(libstats_shared PUBLIC ${LIBSTATS_TBB_LIBRARIES_INTERNAL})
        endif()
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
    # enable_testing() stays top-level (Phase 3B/B3) -- test registration itself (GTest detection,
    # create_libstats_test/gtest/dynamic, run_tests/run_all_tests/run_tests_timing) moved to
    # tests/CMakeLists.txt via add_subdirectory below.
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Test infrastructure enabled")
else()
    message(STATUS "Test infrastructure disabled")
endif()

# Tools and utilities
option(LIBSTATS_BUILD_TOOLS "Build utility tools" ON)

if(LIBSTATS_BUILD_TOOLS)
    # Tool registration (add_libstats_tool/add_standalone_tool helpers, the
    # LIBSTATS_BUILD_SIMD_DEV_TOOLS opt-in block, and all tool targets) moved to
    # tools/CMakeLists.txt (Phase 3B/B4).
    add_subdirectory(tools)
endif()

# Examples
if(LIBSTATS_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Install targets
install(
    TARGETS libstats_headers libstats_static libstats_shared libstats_simd_interface
    EXPORT libstats-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers (exclude editor artifacts and macOS metadata)
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libstats
    PATTERN ".DS_Store" EXCLUDE
    PATTERN "*.bak" EXCLUDE
    PATTERN "*.bak2" EXCLUDE
    PATTERN ".gitkeep" EXCLUDE)

# libstats.h #includes "libstats_version.h" (bare, unnamespaced), generated at configure time into
# the include shim (see NEW-BS-1 above) and reachable at build time via
# $<BUILD_INTERFACE:${LIBSTATS_INCLUDE_SHIM_DIR}>. Without this, an installed find_package(libstats)
# consumer fails to compile: the generated header was never part of the installed tree.
install(FILES "${LIBSTATS_INCLUDE_SHIM_DIR}/libstats_version.h"
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libstats)

# Install CMake config files for find_package support
install(
    EXPORT libstats-targets
    FILE libstats-targets.cmake
    NAMESPACE libstats::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libstats)

# Create and install config file
include(CMakePackageConfigHelpers)
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libstats
    PATH_VARS INCLUDE_INSTALL_DIR)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libstats)

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

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

    message(STATUS "pkg-config file will be installed to ${CMAKE_INSTALL_LIBDIR}/pkgconfig/")
elseif(APPLE)
    # On macOS with Homebrew, also generate pkg-config
    set(LIBSTATS_PC_LIBS_PRIVATE "")
    if(LIBSTATS_HAS_TBB)
        string(APPEND LIBSTATS_PC_LIBS_PRIVATE " -ltbb")
    endif()
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats.pc.in"
                   "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc" @ONLY)

    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libstats.pc"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
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
