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

    # ── Platform-threading preference: suppress OpenMP when GCD/WTP is active ── When
    # LIBSTATS_PREFER_PLATFORM_THREADING is ON and a platform-native pool (GCD on macOS, Windows
    # Thread Pool on Windows) is detected, disable OpenMP to prevent two independent thread pools
    # from over-subscribing the CPU.
    if(LIBSTATS_PREFER_PLATFORM_THREADING)
        if((APPLE AND LIBSTATS_HAS_GCD) OR (WIN32 AND LIBSTATS_HAS_WIN_THREADPOOL))
            if(LIBSTATS_HAS_OPENMP)
                set(LIBSTATS_HAS_OPENMP
                    FALSE
                    CACHE
                        BOOL
                        "OpenMP disabled: platform threading preferred (LIBSTATS_PREFER_PLATFORM_THREADING=ON)"
                        FORCE)
                message(
                    STATUS
                        "  LIBSTATS_PREFER_PLATFORM_THREADING=ON: OpenMP suppressed in favour of platform threading"
                )
            endif()
        endif()
    endif()

    set(LIBSTATS_THREADING_DETECTION_COMPLETE
        TRUE
        CACHE BOOL "Threading detection complete")
    message(
        STATUS
            "Threading: OpenMP=${LIBSTATS_HAS_OPENMP} Pthreads=${LIBSTATS_HAS_PTHREADS} GCD=${LIBSTATS_HAS_GCD}"
    )
endfunction()

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
                # Also add global dirs for transitive-link compatibility (v1.5.3_1 hotfix).
                include_directories(${TBB_INCLUDE_DIRS})
                link_directories(${TBB_LIBRARY_DIRS})
                set(LIBSTATS_TBB_INCLUDE_DIRS_INTERNAL
                    "${TBB_INCLUDE_DIRS}"
                    PARENT_SCOPE)
                set(LIBSTATS_TBB_LIBRARY_DIRS_INTERNAL
                    "${TBB_LIBRARY_DIRS}"
                    PARENT_SCOPE)
                set(LIBSTATS_TBB_LIBRARIES_INTERNAL
                    "${TBB_LIBRARIES}"
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
