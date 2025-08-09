cmake_minimum_required(VERSION 3.20)

# Set CMake policies
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)  # find_package uses upper-case <PACKAGENAME>_ROOT variables
endif()

# Verbose messaging control
option(LIBSTATS_VERBOSE_BUILD "Enable verbose build messages for debugging" OFF)

# Threading system preference control
option(LIBSTATS_FORCE_TBB "Force TBB usage even on platforms with native threading (e.g., GCD on macOS)" OFF)

# Automatic parallel job detection for optimal build performance
include(ProcessorCount)
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

# Set MAKEFLAGS environment variable for make-based generators
if(CMAKE_GENERATOR STREQUAL "Unix Makefiles" OR CMAKE_GENERATOR STREQUAL "MinGW Makefiles")
    set(ENV{MAKEFLAGS} "-j${CPU_COUNT}")
    if(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "Set MAKEFLAGS=-j${CPU_COUNT} for make-based build")
    endif()
endif()

# Set parallel options for CMake build command (for use with cmake --build)
set(CMAKE_BUILD_PARALLEL_LEVEL ${CPU_COUNT} CACHE STRING "Number of parallel build jobs" FORCE)
if(LIBSTATS_VERBOSE_BUILD)
    message(STATUS "Set CMAKE_BUILD_PARALLEL_LEVEL=${CPU_COUNT} for cmake --build")
endif()

# Optimized Homebrew LLVM detection with caching (macOS/Linux only)
# Cache results to avoid repeated filesystem checks during reconfiguration
set(HOMEBREW_LLVM_ROOT_INTEL "/usr/local/opt/llvm")
set(HOMEBREW_LLVM_ROOT_ARM "/opt/homebrew/opt/llvm")

# Function to detect and cache Homebrew LLVM installation
function(detect_homebrew_llvm)
    # Check if detection has already been cached
    if(DEFINED CACHE{LIBSTATS_HOMEBREW_LLVM_ROOT})
        # Use cached result
        set(LLVM_ROOT "${LIBSTATS_HOMEBREW_LLVM_ROOT}" PARENT_SCOPE)
        if(NOT "${LIBSTATS_HOMEBREW_LLVM_ROOT}" STREQUAL "")
            set(USING_HOMEBREW_LLVM TRUE PARENT_SCOPE)
            if(LIBSTATS_VERBOSE_BUILD)
                message(STATUS "Using cached Homebrew LLVM detection: ${LIBSTATS_HOMEBREW_LLVM_ROOT}")
            endif()
        else()
            set(USING_HOMEBREW_LLVM FALSE PARENT_SCOPE)
            if(LIBSTATS_VERBOSE_BUILD)
                message(STATUS "Using cached result: Homebrew LLVM not available")
            endif()
        endif()
        return()
    endif()
    
    # Perform detection and cache result
    if(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "Detecting Homebrew LLVM installation...")
    endif()
    
    # Check ARM64 path first (more common on modern Macs)
    if(EXISTS "${HOMEBREW_LLVM_ROOT_ARM}/bin/clang++")
        set(LIBSTATS_HOMEBREW_LLVM_ROOT "${HOMEBREW_LLVM_ROOT_ARM}" CACHE STRING "Detected Homebrew LLVM root path" FORCE)
        set(LLVM_ROOT "${HOMEBREW_LLVM_ROOT_ARM}" PARENT_SCOPE)
        set(USING_HOMEBREW_LLVM TRUE PARENT_SCOPE)
        message(STATUS "Found Homebrew LLVM (ARM64): ${HOMEBREW_LLVM_ROOT_ARM}")
    # Check Intel path as fallback
    elseif(EXISTS "${HOMEBREW_LLVM_ROOT_INTEL}/bin/clang++")
        set(LIBSTATS_HOMEBREW_LLVM_ROOT "${HOMEBREW_LLVM_ROOT_INTEL}" CACHE STRING "Detected Homebrew LLVM root path" FORCE)
        set(LLVM_ROOT "${HOMEBREW_LLVM_ROOT_INTEL}" PARENT_SCOPE)
        set(USING_HOMEBREW_LLVM TRUE PARENT_SCOPE)
        message(STATUS "Found Homebrew LLVM (Intel): ${HOMEBREW_LLVM_ROOT_INTEL}")
    else()
        # Cache negative result to avoid repeated checks
        set(LIBSTATS_HOMEBREW_LLVM_ROOT "" CACHE STRING "Homebrew LLVM not found" FORCE)
        set(USING_HOMEBREW_LLVM FALSE PARENT_SCOPE)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Homebrew LLVM not found, will use system compiler")
        endif()
    endif()
endfunction()

# Apply Homebrew LLVM detection (macOS/Linux only)
if(APPLE OR (UNIX AND NOT WIN32))
    detect_homebrew_llvm()
    
    if(USING_HOMEBREW_LLVM)
        # Configure Homebrew LLVM
        set(CMAKE_C_COMPILER "${LLVM_ROOT}/bin/clang")
        set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang++")
        set(CMAKE_PREFIX_PATH "${LLVM_ROOT}")
        message(STATUS "Using Homebrew LLVM from ${LLVM_ROOT}")
    else()
        # Fallback to system compiler
        find_program(CMAKE_C_COMPILER clang)
        find_program(CMAKE_CXX_COMPILER clang++)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Using system compiler: ${CMAKE_CXX_COMPILER}")
        endif()
    endif()
endif()

project(libstats VERSION 0.8.1 LANGUAGES CXX)

# Set default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Dev' as none was specified.")
    set(CMAKE_BUILD_TYPE Dev CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "Dev" "ClangStrict" "ClangWarn" "MSVCStrict" "MSVCWarn" "GCCStrict" "GCCWarn")
endif()

# C++20 requirement
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Platform-specific configurations
if(APPLE)
    # macOS specific configuration
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    
    if(USING_HOMEBREW_LLVM)
        # Homebrew LLVM specific configuration - Force LLVM libc++ to get C++20 execution policies
        # Note: rpath will be set per-target to avoid duplication warnings
        
        # CRITICAL: Include LLVM libc++ headers BEFORE system headers to get C++20 execution policies
        include_directories(BEFORE ${LLVM_ROOT}/include/c++/v1)
        include_directories(BEFORE ${LLVM_ROOT}/include)
        
        # Add compiler flags to ensure we use LLVM headers
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -nostdinc++")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${LLVM_ROOT}/include/c++/v1")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${LLVM_ROOT}/include")
        
        # Enable experimental PSTL support in LLVM libc++
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_LIBCPP_HAS_EXPERIMENTAL_PSTL=1")
    else()
        # System compiler configuration - use system libc++
        message(STATUS "Using system libc++ with Apple Clang")
    endif()
    
    # Comprehensive Threading System Detection with Caching
    # Cache results to avoid repeated detection during reconfiguration
    
    # Function to detect and cache comprehensive threading capabilities
    function(detect_threading_systems)
        # Check if detection has already been cached
        if(DEFINED CACHE{LIBSTATS_THREADING_DETECTION_COMPLETE})
            if(LIBSTATS_VERBOSE_BUILD)
                message(STATUS "Using cached threading system detection results")
            endif()
            return()
        endif()
        
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Detecting comprehensive threading system capabilities...")
        endif()
        
        # 1. TBB Detection (Enhanced)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  Checking for Intel TBB...")
        endif()
        find_package(PkgConfig QUIET)
        set(LIBSTATS_HAS_TBB FALSE)
        
        if(PkgConfig_FOUND)
            pkg_check_modules(TBB QUIET tbb)
            if(TBB_FOUND)
                include_directories(${TBB_INCLUDE_DIRS})
                link_directories(${TBB_LIBRARY_DIRS})
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_CFLAGS_OTHER}")
                set(LIBSTATS_HAS_TBB TRUE)
                message(STATUS "    ✓ TBB found via pkg-config - parallel execution policies enhanced")
            endif()
        endif()
        
        # Fallback to find_package for TBB if pkg-config failed
        if(NOT LIBSTATS_HAS_TBB)
            find_package(TBB QUIET)
            if(TBB_FOUND)
                set(LIBSTATS_HAS_TBB TRUE)
                message(STATUS "    ✓ TBB found via find_package - parallel execution policies enhanced")
            else()
                message(STATUS "    ✗ TBB not found - C++20 execution policies may have limited performance")
            endif()
        endif()
        
        # 2. OpenMP Detection
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  Checking for OpenMP...")
        endif()
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(LIBSTATS_HAS_OPENMP TRUE CACHE BOOL "OpenMP support available")
            # Don't automatically link OpenMP - let parallel_execution.h handle detection
            message(STATUS "    ✓ OpenMP found - available as parallel execution fallback")
        else()
            set(LIBSTATS_HAS_OPENMP FALSE CACHE BOOL "OpenMP support not available")
            message(STATUS "    ✗ OpenMP not found")
        endif()
        
        # 3. Pthread Detection (Unix/Linux systems)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  Checking for POSIX Threads (pthreads)...")
        endif()
        find_package(Threads QUIET)
        if(Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
            set(LIBSTATS_HAS_PTHREADS TRUE CACHE BOOL "POSIX threads support available")
            message(STATUS "    ✓ POSIX threads found - available as parallel execution fallback")
        else()
            set(LIBSTATS_HAS_PTHREADS FALSE CACHE BOOL "POSIX threads not available")
            message(STATUS "    ✗ POSIX threads not found or not using pthread implementation")
        endif()
        
        # 4. C++20 Execution Policy Detection (Compile-time)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  Checking for C++20 execution policy support...")
        endif()
        # This will be detected at compile-time by parallel_execution.h
        # but we can do some basic feature testing here
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_STANDARD EQUAL 20)
            if(USING_HOMEBREW_LLVM)
                message(STATUS "    ~ C++20 execution policies: Enhanced with LLVM experimental PSTL")
            else()
                message(STATUS "    ~ C++20 execution policies: May be available with Apple Clang")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_STANDARD EQUAL 20)
            message(STATUS "    ~ C++20 execution policies: May be available with GCC (requires libstdc++9+)")
        elseif(MSVC AND CMAKE_CXX_STANDARD EQUAL 20)
            message(STATUS "    ~ C++20 execution policies: Available with MSVC 2019+ and C++20")
        else()
            message(STATUS "    ~ C++20 execution policies: Detection deferred to runtime")
        endif()
        
        # 5. Platform-Specific Threading (Detection Only - Implementation in parallel_execution.h)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "  Platform-specific threading systems:")
        endif()
        if(APPLE)
            message(STATUS "    ✓ Grand Central Dispatch (GCD) - Available on macOS")
            set(LIBSTATS_HAS_GCD TRUE CACHE BOOL "Grand Central Dispatch available")
        else()
            set(LIBSTATS_HAS_GCD FALSE CACHE BOOL "Grand Central Dispatch not available")
        endif()
        
        if(WIN32)
            # Check for Windows Vista+ (required for Thread Pool API)
            if(DEFINED _WIN32_WINNT)
                if(_WIN32_WINNT GREATER_EQUAL 0x0600)
                    message(STATUS "    ✓ Windows Thread Pool API - Available (Windows Vista+)")
                    set(LIBSTATS_HAS_WIN_THREADPOOL TRUE CACHE BOOL "Windows Thread Pool API available")
                else()
                    message(STATUS "    ✗ Windows Thread Pool API - Requires Windows Vista+ (_WIN32_WINNT >= 0x0600)")
                    set(LIBSTATS_HAS_WIN_THREADPOOL FALSE CACHE BOOL "Windows Thread Pool API not available")
                endif()
            else()
                message(STATUS "    ~ Windows Thread Pool API - Available if _WIN32_WINNT >= 0x0600 (not set)")
                set(LIBSTATS_HAS_WIN_THREADPOOL TRUE CACHE BOOL "Windows Thread Pool API likely available")
            endif()
        else()
            set(LIBSTATS_HAS_WIN_THREADPOOL FALSE CACHE BOOL "Windows Thread Pool API not available")
        endif()
        
        # Cache all TBB variables for consistency
        set(LIBSTATS_HAS_TBB ${LIBSTATS_HAS_TBB} CACHE BOOL "Intel TBB support available")
        
        # Mark detection as complete
        set(LIBSTATS_THREADING_DETECTION_COMPLETE TRUE CACHE BOOL "Threading detection completed")
        
        # Summary
        message(STATUS "Threading System Detection Summary:")
        message(STATUS "  - Intel TBB: ${LIBSTATS_HAS_TBB}")
        message(STATUS "  - OpenMP: ${LIBSTATS_HAS_OPENMP}")
        message(STATUS "  - POSIX Threads: ${LIBSTATS_HAS_PTHREADS}")
        message(STATUS "  - Grand Central Dispatch: ${LIBSTATS_HAS_GCD}")
        message(STATUS "  - Windows Thread Pool: ${LIBSTATS_HAS_WIN_THREADPOOL}")
        message(STATUS "  - C++20 execution policies: Runtime detection enabled")
    endfunction()
    
    # Execute comprehensive threading detection
    detect_threading_systems()

elseif(UNIX AND NOT APPLE)
    # Linux specific configuration
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-soname,libstats.so.${PROJECT_VERSION_MAJOR}")
    
    # Comprehensive Threading System Detection for Linux
    function(detect_threading_systems_linux)
        if(DEFINED CACHE{LIBSTATS_THREADING_DETECTION_COMPLETE})
            message(STATUS "Using cached threading system detection results")
            return()
        endif()
        
        message(STATUS "Detecting comprehensive threading system capabilities (Linux)...")
        
        # 1. TBB Detection
        message(STATUS "  Checking for Intel TBB...")
        find_package(TBB QUIET)
        if(TBB_FOUND)
            set(LIBSTATS_HAS_TBB TRUE CACHE BOOL "Intel TBB support available")
            message(STATUS "    ✓ TBB found via find_package - parallel execution policies enhanced")
        else()
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(TBB QUIET tbb)
                if(TBB_FOUND)
                    include_directories(${TBB_INCLUDE_DIRS})
                    link_directories(${TBB_LIBRARY_DIRS})
                    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_CFLAGS_OTHER}")
                    set(LIBSTATS_HAS_TBB TRUE CACHE BOOL "Intel TBB support available")
                    message(STATUS "    ✓ TBB found via pkg-config - parallel execution policies enhanced")
                else()
                    set(LIBSTATS_HAS_TBB FALSE CACHE BOOL "Intel TBB support not available")
                    message(STATUS "    ✗ TBB not found - C++20 execution policies may have limited performance")
                endif()
            else()
                set(LIBSTATS_HAS_TBB FALSE CACHE BOOL "Intel TBB support not available")
                message(STATUS "    ✗ TBB not found - C++20 execution policies may have limited performance")
            endif()
        endif()
        
        # 2. OpenMP Detection
        message(STATUS "  Checking for OpenMP...")
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(LIBSTATS_HAS_OPENMP TRUE CACHE BOOL "OpenMP support available")
            message(STATUS "    ✓ OpenMP found - available as parallel execution fallback")
        else()
            set(LIBSTATS_HAS_OPENMP FALSE CACHE BOOL "OpenMP support not available")
            message(STATUS "    ✗ OpenMP not found")
        endif()
        
        # 3. Pthread Detection (Standard on Linux)
        message(STATUS "  Checking for POSIX Threads (pthreads)...")
        find_package(Threads REQUIRED)  # Required on Linux
        if(Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
            set(LIBSTATS_HAS_PTHREADS TRUE CACHE BOOL "POSIX threads support available")
            message(STATUS "    ✓ POSIX threads found - available as parallel execution fallback")
        else()
            set(LIBSTATS_HAS_PTHREADS FALSE CACHE BOOL "POSIX threads not available")
            message(STATUS "    ✗ POSIX threads not found - this is unexpected on Linux")
        endif()
        
        # 4. C++20 Execution Policy Detection
        message(STATUS "  Checking for C++20 execution policy support...")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_STANDARD EQUAL 20)
            message(STATUS "    ~ C++20 execution policies: May be available with GCC (requires libstdc++9+)")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_STANDARD EQUAL 20)
            message(STATUS "    ~ C++20 execution policies: May be available with Clang (requires libc++14+)")
        else()
            message(STATUS "    ~ C++20 execution policies: Detection deferred to runtime")
        endif()
        
        # 5. Platform-Specific Threading (Linux-specific)
        message(STATUS "  Platform-specific threading systems:")
        set(LIBSTATS_HAS_GCD FALSE CACHE BOOL "Grand Central Dispatch not available on Linux")
        set(LIBSTATS_HAS_WIN_THREADPOOL FALSE CACHE BOOL "Windows Thread Pool API not available")
        message(STATUS "    ✗ Grand Central Dispatch - Not available on Linux")
        message(STATUS "    ✗ Windows Thread Pool API - Not available on Linux")
        
        set(LIBSTATS_THREADING_DETECTION_COMPLETE TRUE CACHE BOOL "Threading detection completed")
        
        # Summary
        message(STATUS "Threading System Detection Summary (Linux):")
        message(STATUS "  - Intel TBB: ${LIBSTATS_HAS_TBB}")
        message(STATUS "  - OpenMP: ${LIBSTATS_HAS_OPENMP}")
        message(STATUS "  - POSIX Threads: ${LIBSTATS_HAS_PTHREADS}")
        message(STATUS "  - Grand Central Dispatch: ${LIBSTATS_HAS_GCD}")
        message(STATUS "  - Windows Thread Pool: ${LIBSTATS_HAS_WIN_THREADPOOL}")
        message(STATUS "  - C++20 execution policies: Runtime detection enabled")
    endfunction()
    
    detect_threading_systems_linux()

elseif(WIN32)
    # Windows specific configuration
    # Enable Windows compatibility macros
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    
    # Comprehensive Threading System Detection for Windows
    function(detect_threading_systems_windows)
        if(DEFINED CACHE{LIBSTATS_THREADING_DETECTION_COMPLETE})
            message(STATUS "Using cached threading system detection results")
            return()
        endif()
        
        message(STATUS "Detecting comprehensive threading system capabilities (Windows)...")
        
        # 1. TBB Detection (Windows)
        message(STATUS "  Checking for Intel TBB...")
        find_package(TBB QUIET)
        if(TBB_FOUND)
            set(LIBSTATS_HAS_TBB TRUE CACHE BOOL "Intel TBB support available")
            message(STATUS "    ✓ TBB found via find_package - parallel execution policies enhanced")
        else()
            set(LIBSTATS_HAS_TBB FALSE CACHE BOOL "Intel TBB support not available")
            message(STATUS "    ✗ TBB not found - C++20 execution policies may have limited performance")
        endif()
        
        # 2. OpenMP Detection (Windows)
        message(STATUS "  Checking for OpenMP...")
        find_package(OpenMP QUIET)
        if(OpenMP_CXX_FOUND)
            set(LIBSTATS_HAS_OPENMP TRUE CACHE BOOL "OpenMP support available")
            message(STATUS "    ✓ OpenMP found - available as parallel execution fallback")
        else()
            set(LIBSTATS_HAS_OPENMP FALSE CACHE BOOL "OpenMP support not available")
            message(STATUS "    ✗ OpenMP not found")
        endif()
        
        # 3. Windows Threading API Detection
        message(STATUS "  Checking for Windows threading APIs...")
        # Standard Win32 threads are always available on Windows
        set(LIBSTATS_HAS_WIN32_THREADS TRUE CACHE BOOL "Win32 threads support available")
        message(STATUS "    ✓ Win32 threads - Available on Windows")
        
        # Windows Thread Pool API (Vista+)
        if(DEFINED CMAKE_SYSTEM_VERSION AND CMAKE_SYSTEM_VERSION VERSION_GREATER_EQUAL "6.0")
            set(LIBSTATS_HAS_WIN_THREADPOOL TRUE CACHE BOOL "Windows Thread Pool API available")
            message(STATUS "    ✓ Windows Thread Pool API - Available (detected Windows Vista+)")
        elseif(DEFINED _WIN32_WINNT)
            if(_WIN32_WINNT GREATER_EQUAL 0x0600)
                set(LIBSTATS_HAS_WIN_THREADPOOL TRUE CACHE BOOL "Windows Thread Pool API available")
                message(STATUS "    ✓ Windows Thread Pool API - Available (_WIN32_WINNT >= 0x0600)")
            else()
                set(LIBSTATS_HAS_WIN_THREADPOOL FALSE CACHE BOOL "Windows Thread Pool API not available")
                message(STATUS "    ✗ Windows Thread Pool API - Requires Windows Vista+ (_WIN32_WINNT >= 0x0600)")
            endif()
        else()
            # Default to available for modern Windows builds
            set(LIBSTATS_HAS_WIN_THREADPOOL TRUE CACHE BOOL "Windows Thread Pool API likely available")
            message(STATUS "    ~ Windows Thread Pool API - Likely available (Windows Vista+ assumed)")
        endif()
        
        # 4. C++20 Execution Policy Detection (Windows)
        message(STATUS "  Checking for C++20 execution policy support...")
        if(MSVC AND CMAKE_CXX_STANDARD EQUAL 20)
            if(MSVC_VERSION GREATER_EQUAL 1920)  # VS 2019+
                message(STATUS "    ✓ C++20 execution policies: Available with MSVC 2019+ and C++20")
            else()
                message(STATUS "    ~ C++20 execution policies: May be available (MSVC version check failed)")
            endif()
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_STANDARD EQUAL 20)
            message(STATUS "    ~ C++20 execution policies: May be available with Clang-cl")
        else()
            message(STATUS "    ~ C++20 execution policies: Detection deferred to runtime")
        endif()
        
        # 5. Platform-Specific Threading (Windows-specific)
        message(STATUS "  Platform-specific threading systems:")
        set(LIBSTATS_HAS_GCD FALSE CACHE BOOL "Grand Central Dispatch not available on Windows")
        set(LIBSTATS_HAS_PTHREADS FALSE CACHE BOOL "POSIX threads not natively available on Windows")
        message(STATUS "    ✗ Grand Central Dispatch - Not available on Windows")
        message(STATUS "    ✗ POSIX threads - Not natively available on Windows")
        
        set(LIBSTATS_THREADING_DETECTION_COMPLETE TRUE CACHE BOOL "Threading detection completed")
        
        # Summary
        message(STATUS "Threading System Detection Summary (Windows):")
        message(STATUS "  - Intel TBB: ${LIBSTATS_HAS_TBB}")
        message(STATUS "  - OpenMP: ${LIBSTATS_HAS_OPENMP}")
        message(STATUS "  - Win32 Threads: ${LIBSTATS_HAS_WIN32_THREADS}")
        message(STATUS "  - Windows Thread Pool API: ${LIBSTATS_HAS_WIN_THREADPOOL}")
        message(STATUS "  - Grand Central Dispatch: ${LIBSTATS_HAS_GCD}")
        message(STATUS "  - POSIX Threads: ${LIBSTATS_HAS_PTHREADS}")
        message(STATUS "  - C++20 execution policies: Runtime detection enabled")
    endfunction()
    
    # Execute comprehensive threading detection for Windows
    detect_threading_systems_windows()
    
    # Find GTest using find_package (no Homebrew or pkg-config on Windows)
    find_package(GTest QUIET)
    if(GTest_FOUND)
        set(GTEST_FOUND TRUE)
        message(STATUS "GTest found via find_package (Windows)")
    else()
        set(GTEST_FOUND FALSE)
        message(STATUS "GTest not found - GTest-based tests will be skipped (Windows)")
    endif()
endif()

# SIMD feature detection and compilation flags
# Use our comprehensive SIMD detection system
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/SIMDDetection.cmake")

# Set options for SIMD detection behavior
option(LIBSTATS_ENABLE_RUNTIME_CHECKS "Enable runtime CPU checks even when cross-compiling" OFF)
option(LIBSTATS_CONSERVATIVE_SIMD "Use conservative SIMD settings (disable newer instruction sets)" OFF)

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
                include_directories(${TBB_INCLUDE_DIRS})
                link_directories(${TBB_LIBRARY_DIRS})
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TBB_CFLAGS_OTHER}")
                set(LIBSTATS_HAS_TBB TRUE)
                if(LIBSTATS_VERBOSE_BUILD)
                    message(STATUS "  ✓ TBB found via pkg-config")
                endif()
            endif()
        endif()
    endif()
    
    # Cache result
    set(LIBSTATS_HAS_TBB ${LIBSTATS_HAS_TBB} CACHE BOOL "Intel TBB support available")
    set(LIBSTATS_TBB_DETECTION_COMPLETE TRUE CACHE BOOL "TBB detection completed")
    
    if(LIBSTATS_HAS_TBB)
        message(STATUS "Intel TBB: AVAILABLE - parallel execution policies enhanced")
    else()
        message(STATUS "Intel TBB: NOT FOUND - C++20 execution policies may have limited performance")
    endif()
endfunction()

# Execute unified TBB detection with platform preference
# On macOS, prefer GCD over TBB for better integration with the platform
# unless explicitly overridden by LIBSTATS_FORCE_TBB
if(APPLE AND LIBSTATS_HAS_GCD AND NOT LIBSTATS_FORCE_TBB)
    message(STATUS "macOS detected with GCD available - TBB detection skipped (GCD preferred)")
    message(STATUS "  To force TBB usage on macOS, set -DLIBSTATS_FORCE_TBB=ON")
    set(LIBSTATS_HAS_TBB FALSE CACHE BOOL "TBB disabled in favor of GCD on macOS" FORCE)
    set(LIBSTATS_TBB_DETECTION_COMPLETE TRUE CACHE BOOL "TBB detection skipped for GCD preference" FORCE)
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
    -Wall -Wextra -pedantic
    # Type conversion warnings (strict)
    -Wconversion -Wfloat-conversion -Wimplicit-int-conversion -Wshorten-64-to-32
    -Wdouble-promotion -Wsign-conversion
    # Additional strict warnings
    -Wshadow -Wunused -Wcast-align -Wcast-qual
    -Wold-style-cast -Woverloaded-virtual
    -Wextra-semi -Wmissing-declarations
    # Clang-specific warnings
    -Wloop-analysis -Wlogical-op-parentheses -Wbool-conversion
    -Wint-conversion -Wnull-dereference -Wuninitialized
    -Wconditional-uninitialized -Wstring-conversion
    # Allow some warnings that are too strict for practical development
    -Wno-unused-parameter  # Common in template/callback code
    -Wno-padded           # Not practical for cross-platform development
    -Wno-c++98-compat     # We're using C++20
    -Wno-c++98-compat-pedantic
)

set(LIBSTATS_GCC_STRICT_WARNINGS
    -Wall -Wextra -pedantic
    # GCC-specific conversion warnings
    -Wconversion -Wfloat-conversion -Wdouble-promotion
    -Wsign-conversion -Wold-style-cast
    # GCC-specific diagnostic warnings
    -Wlogical-op -Wduplicated-cond -Wduplicated-branches
    -Wrestrict -Wnull-dereference -Wjump-misses-init
    # Template and name resolution differences
    -Wtrampolines -Wunsafe-loop-optimizations
    -Wvector-operation-performance -Wsuggest-override
    -Wsuggest-final-types -Wsuggest-final-methods
    # Memory and alignment warnings
    -Wcast-align=strict -Wstrict-overflow=2
    -Wformat=2 -Wformat-overflow=2 -Wformat-truncation=2
    # GCC's stricter undefined behavior detection
    -Wshift-overflow=2 -Wstringop-overflow=4
    # Allow some warnings that are too pedantic
    -Wno-inline -Wno-padded -Wno-unused-parameter
    -Wno-sign-conversion  # Too noisy with standard library
)

# MSVC-compatible warning flags for cross-compiler testing
set(LIBSTATS_MSVC_COMPAT_WARNINGS
    -Wall -Wextra
    # Type conversion warnings (MSVC-like strictness)
    -Wconversion -Wfloat-conversion -Wimplicit-int-conversion -Wshorten-64-to-32
    -Wdouble-promotion
    # Additional strict warnings
    -Wshadow -Wunused -Wcast-align -Wcast-qual
    -Wold-style-cast -Woverloaded-virtual
    -Wextra-semi -Wmissing-declarations
    # Allow some warnings that are too strict for practical development
    -Wno-sign-conversion  # Too many false positives in standard library interactions
    -Wno-unused-parameter  # Common in template/callback code
    -Wno-padded           # Not practical for cross-platform development
)

# Enhanced MSVC warning sets
set(LIBSTATS_MSVC_ENHANCED_WARNINGS
    /w14242 /w14254 /w14263 /w14265 /w14287 /w14289 /w14296 /w14311
    /w14545 /w14546 /w14547 /w14549 /w14555 /w14619 /w14640 /w14826
    /w14905 /w14906 /w14928
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
# Use generator expressions to reduce repetitive if/elseif chains
# Standard CMake Build Types (Release, Debug) - Cross-platform
# Custom Compiler-Specific Build Types (ClangStrict, ClangWarn, MSVCStrict, MSVCWarn, GCCStrict, GCCWarn)
# =============================================================================

# =============================================================================
# MODERN CMAKE COMPILER CONFIGURATION WITH GENERATOR EXPRESSIONS
# =============================================================================
# This section replaces repetitive if/elseif chains with modern generator expressions
# and uses shared flag lists for better maintainability

# MSVC-specific compiler definitions (Windows compatibility)
add_compile_definitions(
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_SECURE_NO_WARNINGS>
    $<$<CXX_COMPILER_ID:MSVC>:_CRT_NONSTDC_NO_DEPRECATE>
)

# Apply build type and compiler-specific flags using traditional CMake approach
# This maintains the generator expression benefits while ensuring reliability

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(/W3 /O2 /DNDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/W3 /Od /Zi /RTC1 /MDd)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(/W3 /O1 /Zi /MD /wd4996)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCStrict")
        add_compile_options(/W4 /WX /permissive- /we4289)
        add_compile_options(${LIBSTATS_MSVC_ENHANCED_WARNINGS})
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCWarn")
        add_compile_options(/W4 /permissive- /wd4996)
        add_compile_options(${LIBSTATS_MSVC_ENHANCED_WARNINGS})
    else()
        add_compile_options(/W3)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    # Clang/AppleClang compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "ClangStrict")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "ClangWarn")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "GCCStrict")
        # GCC build type with Clang compiler - use Clang equivalent flags
        message(WARNING "GCCStrict build type requested with Clang compiler. Using Clang-compatible flags instead.")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "GCCWarn")
        # GCC build type with Clang compiler - use Clang equivalent flags
        message(WARNING "GCCWarn build type requested with Clang compiler. Using Clang-compatible flags instead.")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCStrict")
        add_compile_options(${LIBSTATS_MSVC_COMPAT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCWarn")
        add_compile_options(${LIBSTATS_MSVC_COMPAT_WARNINGS} -Wno-deprecated-declarations)
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} -O2)
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC compiler flags
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(${LIBSTATS_OPT_FULL_UNIX} -DNDEBUG ${LIBSTATS_COMMON_WARNINGS_UNIX})
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(${LIBSTATS_OPT_NONE_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -fstack-protector-strong)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
        add_compile_options(${LIBSTATS_OPT_LIGHT_UNIX} ${LIBSTATS_DEBUG_INFO_UNIX} ${LIBSTATS_COMMON_WARNINGS_UNIX} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "ClangStrict")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "ClangWarn")
        add_compile_options(${LIBSTATS_CLANG_STRICT_WARNINGS} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "GCCStrict")
        add_compile_options(${LIBSTATS_GCC_STRICT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "GCCWarn")
        add_compile_options(${LIBSTATS_GCC_STRICT_WARNINGS} -Wno-deprecated-declarations)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCStrict")
        add_compile_options(${LIBSTATS_MSVC_COMPAT_WARNINGS} -Werror)
    elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCWarn")
        add_compile_options(${LIBSTATS_MSVC_COMPAT_WARNINGS} -Wno-deprecated-declarations)
    else()
        add_compile_options(${LIBSTATS_COMMON_WARNINGS_UNIX} ${LIBSTATS_OPT_FULL_UNIX})
    endif()
else()
    # Unknown compiler - use sensible defaults
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -DNDEBUG)
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-O0 -g)
    else()
        add_compile_options(-O2)
    endif()
endif()

# Apply SIMD-specific compilation flags using generator expressions
add_compile_options(
    # MSVC SIMD (x64 only)
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:/arch:AVX2>
    
    # Clang SIMD (Windows x64 only for ClangCL)
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:-mavx2>
    
    # GCC SIMD (Conservative approach - SSE2 baseline only)
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${LIBSTATS_HAS_SSE2}>>:-msse2>
)

# Apply SIMD compile definitions using generator expressions
add_compile_definitions(
    # MSVC SIMD definitions (x64 only)
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_AVX2=1>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_AVX=1>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_SSE2=1>
    
    # Clang SIMD definitions (Windows x64 only)
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_AVX2=1>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_AVX=1>
    $<$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Windows>,$<EQUAL:${CMAKE_SIZEOF_VOID_P},8>>:LIBSTATS_HAS_SSE2=1>
    
    # GCC/Clang SIMD definitions (runtime-dispatched, conservative approach)
    $<$<BOOL:${LIBSTATS_HAS_SSE2}>:LIBSTATS_HAS_SSE2=1>
    $<$<BOOL:${LIBSTATS_HAS_AVX}>:LIBSTATS_HAS_AVX=1>
    $<$<BOOL:${LIBSTATS_HAS_AVX2}>:LIBSTATS_HAS_AVX2=1>
    $<$<BOOL:${LIBSTATS_HAS_AVX512}>:LIBSTATS_HAS_AVX512=1>
    $<$<BOOL:${LIBSTATS_HAS_NEON}>:LIBSTATS_HAS_NEON=1>
)

# Display configuration information (evaluate once at configure time)
# Cross-Compiler Build Type Status Messages
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Release build (${CMAKE_CXX_COMPILER_ID}): Optimized for performance")
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug build (${CMAKE_CXX_COMPILER_ID}): Full debugging information")
elseif(CMAKE_BUILD_TYPE STREQUAL "Dev")
    message(STATUS "Dev build (${CMAKE_CXX_COMPILER_ID}): Balanced optimization and debugging")
# Cross-Compiler Compatibility Modes  
elseif(CMAKE_BUILD_TYPE STREQUAL "ClangStrict")
    message(STATUS "Clang-strict mode (${CMAKE_CXX_COMPILER_ID}): Clang-style warnings as ERRORS - cross-compiler compatible")
elseif(CMAKE_BUILD_TYPE STREQUAL "ClangWarn")
    message(STATUS "Clang-warn mode (${CMAKE_CXX_COMPILER_ID}): Clang-style warnings as WARNINGS - cross-compiler compatible")
elseif(CMAKE_BUILD_TYPE STREQUAL "GCCStrict")
    message(STATUS "GCC-strict mode (${CMAKE_CXX_COMPILER_ID}): GCC-style warnings as ERRORS - cross-compiler compatible")
elseif(CMAKE_BUILD_TYPE STREQUAL "GCCWarn")
    message(STATUS "GCC-warn mode (${CMAKE_CXX_COMPILER_ID}): GCC-style warnings as WARNINGS - cross-compiler compatible")
elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCStrict")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "MSVC-strict mode (MSVC native): MSVC warnings as ERRORS with enhanced diagnostics")
    else()
        message(STATUS "MSVC-strict mode (${CMAKE_CXX_COMPILER_ID} emulating MSVC): MSVC-compatible warnings as ERRORS")
    endif()
elseif(CMAKE_BUILD_TYPE STREQUAL "MSVCWarn")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(STATUS "MSVC-warn mode (MSVC native): MSVC warnings with enhanced diagnostics")
    else()
        message(STATUS "MSVC-warn mode (${CMAKE_CXX_COMPILER_ID} emulating MSVC): MSVC-compatible warnings")
    endif()
else()
    message(STATUS "Default build (${CMAKE_CXX_COMPILER_ID}): Standard configuration")
endif()

# SIMD Status Messages (compiler-specific)
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (MSVC x64)")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "SIMD: AVX2/AVX/SSE2 enabled (ClangCL x64)")
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
# Organize sources into object libraries that align with header dependency levels
# This enables:
# - Parallel compilation of independent components
# - Minimal recompilation when headers change
# - Better incremental build performance
# - Clear architectural boundaries

# Level 0-1: Platform Foundation and Constants (No internal dependencies)
set(LIBSTATS_FOUNDATION_SOURCES
    src/cpu_detection.cpp              # Runtime CPU feature detection
    src/simd_policy.cpp               # SIMD policy and feature detection
)

# Level 2: Core Utilities (Depends on Level 0-1)
set(LIBSTATS_CORE_UTILITIES_SOURCES
    src/safety.cpp                    # Safe numerical operations
    src/validation.cpp                # Parameter validation and constraint checking
    src/math_utils.cpp                # Mathematical utilities and special functions
    src/log_space_ops.cpp             # Log-space arithmetic for numerical stability
)

# Level 2: Platform Capabilities (Depends on Level 0-1)
set(LIBSTATS_PLATFORM_SOURCES
    src/parallel_thresholds.cpp       # Architecture-specific parallel thresholds
    src/thread_pool.cpp               # Thread pool implementation
    src/work_stealing_pool.cpp        # Advanced work-stealing thread pool
)

# Level 3: Advanced Infrastructure (Depends on Level 0-2)
set(LIBSTATS_INFRASTRUCTURE_SOURCES
    src/adaptive_cache.cpp            # Memory-aware adaptive caching
    src/distribution_cache.cpp        # Distribution-specific cache optimization
    src/benchmark.cpp                 # Performance benchmarking utilities
    src/performance_history.cpp       # Performance data collection and analysis
    src/performance_dispatcher.cpp    # Smart auto-dispatch system
    src/system_capabilities.cpp       # System capability detection
    src/libstats_init.cpp             # Performance system initialization
)

# Level 4: Distribution Framework (Depends on Level 0-3)
set(LIBSTATS_FRAMEWORK_SOURCES
    src/distribution_memory.cpp       # Memory management for distributions (if exists)
    src/distribution_base.cpp         # Complete distribution base framework
)

# Level 5: Distribution Implementations (Depends on Level 0-4)
set(LIBSTATS_DISTRIBUTIONS_SOURCES
    src/gaussian.cpp                  # Gaussian distribution implementation
    src/exponential.cpp               # Exponential distribution implementation
    src/uniform.cpp                   # Uniform distribution implementation
    src/poisson.cpp                   # Poisson distribution implementation
    src/discrete.cpp                  # Discrete distribution implementation
    src/gamma.cpp                     # Gamma distribution implementation
)

# SIMD sources are added separately by the SIMD detection system
# These compile with specific architecture flags and are platform-dependent

# =============================================================================
# CREATE OBJECT LIBRARIES FOR PARALLEL COMPILATION
# =============================================================================
# Object libraries allow parallel compilation and better dependency tracking

# Foundation object library (Level 0-1)
add_library(libstats_foundation_obj OBJECT ${LIBSTATS_FOUNDATION_SOURCES})
target_include_directories(libstats_foundation_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target

# Core utilities object library (Level 2)
add_library(libstats_core_utilities_obj OBJECT ${LIBSTATS_CORE_UTILITIES_SOURCES})
target_include_directories(libstats_core_utilities_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_core_utilities_obj libstats_foundation_obj)

# Platform capabilities object library (Level 2)
add_library(libstats_platform_obj OBJECT ${LIBSTATS_PLATFORM_SOURCES})
target_include_directories(libstats_platform_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_platform_obj libstats_foundation_obj)

# Advanced infrastructure object library (Level 3)
add_library(libstats_infrastructure_obj OBJECT ${LIBSTATS_INFRASTRUCTURE_SOURCES})
target_include_directories(libstats_infrastructure_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_infrastructure_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj)

# Distribution framework object library (Level 4)
add_library(libstats_framework_obj OBJECT ${LIBSTATS_FRAMEWORK_SOURCES})
target_include_directories(libstats_framework_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_framework_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj libstats_infrastructure_obj)

# Distribution implementations object library (Level 5)
add_library(libstats_distributions_obj OBJECT ${LIBSTATS_DISTRIBUTIONS_SOURCES})
target_include_directories(libstats_distributions_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_distributions_obj libstats_foundation_obj libstats_core_utilities_obj libstats_platform_obj libstats_infrastructure_obj libstats_framework_obj)

# SIMD object library (Platform-dependent, can compile in parallel)
add_library(libstats_simd_obj OBJECT ${LIBSTATS_SIMD_SOURCES})
target_include_directories(libstats_simd_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SIMD definitions will be linked through interface target
add_dependencies(libstats_simd_obj libstats_foundation_obj)  # SIMD depends on CPU detection

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
    $<TARGET_OBJECTS:libstats_simd_obj>
)

# Header-only library (for header-only components)
add_library(libstats_headers INTERFACE)
target_include_directories(libstats_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)

# Static library using object libraries for better build performance
add_library(libstats_static STATIC ${LIBSTATS_ALL_OBJECTS})
target_include_directories(libstats_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)
if(WIN32)
    set_target_properties(libstats_static PROPERTIES
        OUTPUT_NAME stats_static
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
else()
    set_target_properties(libstats_static PROPERTIES
        OUTPUT_NAME stats
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Dynamic/shared library using object libraries for better build performance
add_library(libstats_shared SHARED ${LIBSTATS_ALL_OBJECTS})
target_include_directories(libstats_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/libstats>
)
set_target_properties(libstats_shared PROPERTIES
    OUTPUT_NAME "stats"
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

# Platform-specific shared library properties
if(APPLE)
    set_target_properties(libstats_shared PROPERTIES
        MACOSX_RPATH ON
        INSTALL_NAME_DIR "@rpath"
    )
elseif(UNIX AND NOT APPLE)
    set_target_properties(libstats_shared PROPERTIES
        SOVERSION ${PROJECT_VERSION_MAJOR}
    )
endif()

# Create SIMD interface target for modern CMake approach
create_simd_interface_target()

# Configure SIMD compilation for all object libraries and final targets
# This ensures SIMD-specific compile flags are applied correctly to each component
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

# Platform-specific library linking configuration
if(APPLE AND USING_HOMEBREW_LLVM)
    # Apply Homebrew LLVM libc++ linking to the shared library
    target_link_options(libstats_shared PRIVATE
        -L${LLVM_ROOT}/lib/c++
        -Wl,-rpath,${LLVM_ROOT}/lib/c++
    )
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
    
    # Find GTest (Homebrew installation)
    # First, try to use Homebrew's GTest installation directly
    # Check both Intel and ARM64 paths
    set(GTEST_ROOT_INTEL "/usr/local/opt/googletest")
    set(GTEST_ROOT_ARM "/opt/homebrew/opt/googletest")
    
    # Determine which Homebrew path to use
    if(EXISTS "${GTEST_ROOT_ARM}/lib/cmake/GTest")
        set(GTEST_ROOT "${GTEST_ROOT_ARM}")
        message(STATUS "Using ARM64 Homebrew GTest path")
    elseif(EXISTS "${GTEST_ROOT_INTEL}/lib/cmake/GTest")
        set(GTEST_ROOT "${GTEST_ROOT_INTEL}")
        message(STATUS "Using Intel Homebrew GTest path")
    else()
        set(GTEST_ROOT "${GTEST_ROOT_ARM}")
        message(STATUS "Using default ARM64 Homebrew GTest path (may not exist)")
    endif()
    
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${GTEST_ROOT}")
    
    # Try find_package first with explicit path
    find_package(GTest QUIET HINTS ${GTEST_ROOT}/lib/cmake/GTest)
    
    if(GTest_FOUND OR TARGET GTest::gtest)
        set(GTEST_FOUND TRUE)
        message(STATUS "GTest found via find_package")
    else()
        # Fallback to pkg-config
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            # Set PKG_CONFIG_PATH to include Homebrew's pkgconfig directory
            set(ENV{PKG_CONFIG_PATH} "${GTEST_ROOT}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
            pkg_check_modules(GTEST QUIET gtest)
            pkg_check_modules(GTEST_MAIN QUIET gtest_main)
            if(GTEST_FOUND)
                message(STATUS "GTest found via pkg-config")
            endif()
        endif()
        
        # Manual fallback if both methods fail
        if(NOT GTEST_FOUND)
            # Check if the Homebrew installation exists manually
            if(EXISTS "${GTEST_ROOT}/lib/libgtest.a" AND EXISTS "${GTEST_ROOT}/include/gtest/gtest.h")
                set(GTEST_FOUND TRUE)
                set(GTEST_INCLUDE_DIRS "${GTEST_ROOT}/include")
                set(GTEST_LIBRARIES "${GTEST_ROOT}/lib/libgtest.a")
                set(GTEST_MAIN_LIBRARIES "${GTEST_ROOT}/lib/libgtest_main.a")
                message(STATUS "GTest found manually at ${GTEST_ROOT}")
            else()
                message(STATUS "GTest not found - GTest-based tests will be skipped")
            endif()
        endif()
    endif()
    
    # Common test configuration function - reduces code duplication across test types
    function(configure_common_test_settings TEST_NAME)
        # Set output directory to build/tests
        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/tests
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/tests
        )
        
        # Apply same compiler flags as main library
        if(APPLE)
            target_compile_options(${TEST_NAME} PRIVATE -stdlib=libc++)
        endif()
        
        # Add include directories
        target_include_directories(${TEST_NAME} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        
        # Platform-specific linking
        if(APPLE AND USING_HOMEBREW_LLVM)
            target_link_options(${TEST_NAME} PRIVATE
                -L${LLVM_ROOT}/lib/c++
                -Wl,-rpath,${LLVM_ROOT}/lib/c++
            )
        endif()
        
        # Link to SIMD interface target for modern CMake approach
        if(TARGET libstats_simd_interface)
            target_link_libraries(${TEST_NAME} PRIVATE libstats::simd)
        endif()
        
        # Add to test suite
        add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CURRENT_BINARY_DIR}/tests/${TEST_NAME})
        
        # Set working directory for tests
        set_tests_properties(${TEST_NAME} PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        )
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
    
    # Function to create a test executable linked to dynamic library
    function(create_libstats_test_dynamic TEST_NAME SOURCE_FILE)
        add_executable(${TEST_NAME} ${SOURCE_FILE})
        
        # Link against shared library instead of static
        target_link_libraries(${TEST_NAME} PRIVATE libstats_shared)
        
        # Apply common test settings
        configure_common_test_settings(${TEST_NAME})
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
    # LEVEL 0 TESTS: Foundational Headers (Standard Library Only)
    # =======================================================================
    # These test headers with no internal dependencies - only standard library
    
    # C++20 features and basic constants (no internal dependencies)
    create_libstats_test(test_cpp20_features tests/test_cpp20_features.cpp)
    create_libstats_test(test_constants tests/test_constants.cpp)
    
    # =======================================================================
    # LEVEL 1 TESTS: Platform Foundation and Aggregated Constants
    # =======================================================================
    # Tests for runtime capabilities and aggregated constant headers
    
    # cpu_detection.h - Runtime CPU feature detection
    create_libstats_test(test_cpu_detection tests/test_cpu_detection.cpp)
    
    # simd.h - SIMD compile-time detection and basic operations
    create_libstats_test(test_simd_integration_simple tests/test_simd_integration_simple.cpp)
    create_libstats_test(test_simd_integration tests/test_simd_integration.cpp)
    create_libstats_test(test_simd_operations tests/test_simd_operations.cpp)
    create_libstats_test(test_platform_optimizations tests/test_platform_optimizations.cpp)
    
    # =======================================================================
    # LEVEL 2 TESTS: Core Utilities and Platform Capabilities
    # =======================================================================
    # Test core utilities (Level 2a) and platform capabilities (Level 2b)
    
    # LEVEL 2a: Core Utilities (Platform-Independent)
    # safety.h - Safe numerical operations
    create_libstats_test(test_safety tests/test_safety.cpp)
    
    # math_utils.h - Mathematical utilities and special functions
    create_libstats_gtest(test_math_utils tests/test_math_utils.cpp)
    create_libstats_test(test_vectorized_math tests/test_vectorized_math.cpp)
    
    # validation.h - Parameter validation and constraint checking
    create_libstats_test(test_goodness_of_fit tests/test_goodness_of_fit.cpp)
    create_libstats_test(test_validation_enhanced tests/test_validation_enhanced.cpp)
    
    # error_handling.h - Exception-free error handling
    create_libstats_test(test_safe_factory tests/test_safe_factory.cpp)
    create_libstats_test(test_dual_api tests/test_dual_api.cpp)
    
    # LEVEL 2b: Platform Capabilities
    # thread_pool.h and work_stealing_pool.h - Thread pool implementations  
    create_libstats_test(test_thread_pool tests/test_thread_pool.cpp)
    create_libstats_test(test_work_stealing_pool tests/test_work_stealing_pool.cpp)
    
    # =======================================================================
    # LEVEL 3 TESTS: Advanced Infrastructure
    # =======================================================================
    # Test advanced infrastructure components that depend on Level 0-2
    
    # Performance and Caching Infrastructure
    # adaptive_cache.h - Memory-aware adaptive caching
    create_libstats_test(test_adaptive_cache tests/test_adaptive_cache.cpp)
    
    # parallel_execution.h - C++20 parallel execution with automatic fallback
    create_libstats_test(test_parallel_execution_integration tests/test_parallel_execution_integration.cpp)
    create_libstats_test(test_parallel_execution_comprehensive tests/test_parallel_execution_comprehensive.cpp)
    
    # benchmark.h - Performance measurement utilities
    create_libstats_test(test_benchmark_basic tests/test_benchmark_basic.cpp)
    
    # PHASE 3 PERFORMANCE OPTIMIZATION FRAMEWORK (NEW TESTS)
    # performance_history.h - Performance data collection and analysis
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
    
    # distribution_base.h - Complete base class implementation
    # (Tested indirectly through distribution implementations, but could add direct tests)
    
    # =======================================================================
    # LEVEL 5 TESTS: Concrete Distribution Implementations
    # =======================================================================
    # Test concrete distribution implementations using the complete framework
    
    # gaussian.h - Gaussian/Normal distribution implementation (CONSOLIDATED)
    create_libstats_test(test_gaussian_basic tests/test_gaussian_basic.cpp)
    create_libstats_gtest(test_gaussian_enhanced tests/test_gaussian_enhanced.cpp)
    
    # exponential.h - Exponential distribution implementation (CONSOLIDATED)
    create_libstats_test(test_exponential_basic tests/test_exponential_basic.cpp)
    create_libstats_gtest(test_exponential_enhanced tests/test_exponential_enhanced.cpp)
    
    # discrete.h - Discrete uniform distribution implementation (CONSOLIDATED)
    create_libstats_test(test_discrete_basic tests/test_discrete_basic.cpp)
    create_libstats_gtest(test_discrete_enhanced tests/test_discrete_enhanced.cpp)
    
    # Atomic parameters test - Test lock-free parameter access and invalidation for all distributions
    create_libstats_test(test_atomic_parameters tests/test_atomic_parameters.cpp)
    
    # uniform.h - Uniform distribution implementation
    create_libstats_test(test_uniform_basic tests/test_uniform_basic.cpp)
    create_libstats_gtest(test_uniform_enhanced tests/test_uniform_enhanced.cpp)
    
    # poisson.h - Poisson distribution implementation
    create_libstats_test(test_poisson_basic tests/test_poisson_basic.cpp)
    create_libstats_gtest(test_poisson_enhanced tests/test_poisson_enhanced.cpp)
    
    # gamma.h - Gamma distribution implementation
    create_libstats_test(test_gamma_basic tests/test_gamma_basic.cpp)
    create_libstats_gtest(test_gamma_enhanced tests/test_gamma_enhanced.cpp)
    
    # =======================================================================
    # ADDITIONAL TESTS: Cross-cutting concerns and integration
    # =======================================================================
    # These test cross-cutting functionality and integration scenarios
    
    # Copy/move semantics stress tests
    # TODO: Temporarily commented out due to exponential/uniform distribution dependencies
    # if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_stress.cpp)
    #     create_libstats_test(test_copy_move_stress tests/test_copy_move_stress.cpp)
    #     set(HAS_COPY_MOVE_STRESS TRUE)
    # else()
    #     set(HAS_COPY_MOVE_STRESS FALSE)
    # endif()
    set(HAS_COPY_MOVE_STRESS FALSE)
    
    # if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_copy_move_fix.cpp)
    #     create_libstats_test(test_copy_move_fix tests/test_copy_move_fix.cpp)
    #     set(HAS_COPY_MOVE_FIX TRUE)
    # else()
    #     set(HAS_COPY_MOVE_FIX FALSE)
    # endif()
    set(HAS_COPY_MOVE_FIX FALSE)
    
    # Dynamic library linking tests
    create_libstats_test_dynamic(test_dynamic_linking tests/test_dynamic_linking.cpp)
    
    # Dynamic versions of all _simple tests
    create_libstats_test_dynamic(test_simd_integration_simple_dynamic tests/test_simd_integration_simple.cpp)
    create_libstats_test_dynamic(test_gaussian_basic_dynamic tests/test_gaussian_basic.cpp)
    create_libstats_test_dynamic(test_exponential_basic_dynamic tests/test_exponential_basic.cpp)
    # create_libstats_test_dynamic(test_uniform_simple_dynamic tests/test_uniform_simple.cpp)  # TODO: uncomment after fixing uniform.cpp
    
    # TODO: Add these tests once distributions are implemented
    # create_libstats_test(test_exponential tests/test_exponential.cpp)
    # create_libstats_test(test_uniform tests/test_uniform.cpp)
    # create_libstats_test(test_poisson tests/test_poisson.cpp)
    # create_libstats_test(test_gamma tests/test_gamma.cpp)
    # create_libstats_test(test_validation tests/test_validation.cpp)
    
    # Future GTest-based tests can be added here using create_libstats_gtest()
    # Example: create_libstats_gtest(test_gaussian_gtest tests/test_gaussian_gtest.cpp)
    
    # Test target for running all tests (organized by dependency hierarchy)
    add_custom_target(run_tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS
            # Level 0: Foundational tests (most critical)
            test_cpp20_features
            test_constants
            test_cpu_detection
            test_adaptive_cache
            test_simd_integration_simple
            test_simd_integration
            test_simd_operations
            test_platform_optimizations
            test_safe_factory
            
            # Level 1: Core infrastructure tests
            test_dual_api
            test_safety
            test_math_utils
            test_vectorized_math
            test_goodness_of_fit
            test_validation_enhanced
            
            # Level 2: Framework tests (no direct tests - covered by Level 4)
            
            # Level 3: Parallel infrastructure tests
            test_thread_pool
            test_work_stealing_pool
            test_parallel_execution_integration
            test_parallel_execution_comprehensive
            test_benchmark_basic
            
            # Level 4: Distribution implementation tests (CONSOLIDATED)
            test_gaussian_basic
            test_gaussian_enhanced
            test_exponential_basic
            test_exponential_enhanced
            test_discrete_basic
            test_discrete_enhanced
            test_atomic_parameters
            test_uniform_basic
            test_uniform_enhanced
            test_poisson_basic
            test_poisson_enhanced
            test_gamma_basic
            test_gamma_enhanced
            
            # Additional tests: Cross-cutting concerns
            test_dynamic_linking
            test_simd_integration_simple_dynamic
            test_gaussian_basic_dynamic
            test_exponential_basic_dynamic
            # test_uniform_simple_dynamic      # TODO: uncomment after fixing uniform.cpp
            
            # Conditional tests (added if they exist)
            $<$<BOOL:${HAS_COPY_MOVE_STRESS}>:test_copy_move_stress>
            $<$<BOOL:${HAS_COPY_MOVE_FIX}>:test_copy_move_fix>
        COMMENT "Running all libstats tests in dependency order (Level 0-4)"
    )
    
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
        target_include_directories(${tool_name} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        
        # Apply same compiler flags as main library
        if(APPLE)
            target_compile_options(${tool_name} PRIVATE -stdlib=libc++)
            if(USING_HOMEBREW_LLVM)
                target_link_options(${tool_name} PRIVATE
                    -L${LLVM_ROOT}/lib/c++
                    -Wl,-rpath,${LLVM_ROOT}/lib/c++
                )
            endif()
        endif()
        
        # Link to SIMD interface target for modern CMake approach
        if(TARGET libstats_simd_interface)
            target_link_libraries(${tool_name} PRIVATE libstats::simd)
        endif()
        
        set_target_properties(${tool_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tools
        )
    endfunction()
    
    # System Information & Diagnostics Tools
    add_libstats_tool(cpu_info cpu_info.cpp)
    add_libstats_tool(constants_inspector constants_inspector.cpp)
    
    # Performance & Benchmarking Tools
    add_libstats_tool(performance_benchmark performance_benchmark.cpp)
    add_libstats_tool(parallel_threshold_benchmark parallel_threshold_benchmark.cpp)
    add_libstats_tool(grain_size_optimizer grain_size_optimizer.cpp)
    add_libstats_tool(performance_dispatcher_tool performance_dispatcher_tool.cpp)
    add_libstats_tool(system_capabilities_inspector system_capabilities_inspector.cpp)
    add_libstats_tool(threshold_learning_demo threshold_learning_demo.cpp)
    
    message(STATUS "Tools enabled:")
    message(STATUS "  - cpu_info: System CPU capability detection")
    message(STATUS "  - constants_inspector: Platform-specific constants analysis with Phase 3 integration")
    message(STATUS "  - performance_benchmark: Enhanced parallel operation benchmarks with PerformanceHistory integration")
    message(STATUS "  - parallel_threshold_benchmark: Enhanced distribution-specific threshold optimization with adaptive learning")
    message(STATUS "  - grain_size_optimizer: Comprehensive grain size optimization")
    message(STATUS "  - performance_dispatcher_tool: Interactive Phase 3 performance framework demonstration")
    message(STATUS "  - system_capabilities_inspector: Detailed system capability inspection and performance analysis")
    message(STATUS "  - threshold_learning_demo: Adaptive threshold learning demonstration and performance optimization showcase")
endif()

# Examples
add_subdirectory(examples)

# Install targets
install(TARGETS libstats_headers libstats_static libstats_shared libstats_simd_interface
    EXPORT libstats-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install headers
install(DIRECTORY include/ DESTINATION include/libstats)

# Install CMake config files for find_package support
install(EXPORT libstats-targets
    FILE libstats-targets.cmake
    NAMESPACE libstats::
    DESTINATION lib/cmake/libstats
)

# Create and install config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/libstats-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    INSTALL_DESTINATION lib/cmake/libstats
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libstats-config-version.cmake"
    DESTINATION lib/cmake/libstats
)

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

