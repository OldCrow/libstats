# =============================================================================
# 2A: Unified threading system detection — one function, platform guards inside. Replaces the three
# separate detect_threading_systems_*() functions.
# =============================================================================
function(detect_threading_systems)
    # ── Package lookups run on EVERY configure pass (#90) ───────────────────
    #
    # Cache variables persist across passes; IMPORTED TARGETS DO NOT. They are recreated only by
    # find_package(), so these calls must sit ABOVE the completion guard. When they sat below it,
    # any reconfigure of an existing build dir left Threads::Threads undefined, and the consuming
    # `if(TARGET Threads::Threads)` in CMakeLists.txt went quiet rather than failing — dropping
    # -pthread from the PUBLIC link and therefore from libstats-targets.cmake, so the installed
    # package differed depending on how many times its build dir had been configured.
    #
    # Repeat calls are cheap: each package caches its own probe results, so the second pass rebinds
    # the targets without redoing the compile checks. Everything BELOW the guard is cache-setting
    # and status output, which correctly runs once.
    find_package(OpenMP QUIET)
    if(NOT WIN32)
        if(UNIX AND NOT APPLE)
            find_package(Threads REQUIRED)
        else()
            find_package(Threads QUIET)
        endif()
    endif()

    if(DEFINED CACHE{LIBSTATS_THREADING_DETECTION_COMPLETE})
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Using cached threading detection results")
        endif()
        return()
    endif()

    message(STATUS "Detecting threading capabilities...")

    # ── OpenMP (all platforms) ──────────────────────────────────────────────
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
    #
    # find_package(Threads) itself ran above the guard; this only records the outcome in the cache.
    if(NOT WIN32)
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
    # Detection runs on EVERY configure pass (#90), for the same reason as in
    # detect_threading_systems(): nothing this function produces survives into the next pass. The
    # TBB::tbb imported target, the LIBSTATS_TBB_*_INTERNAL PARENT_SCOPE variables, and the
    # directory-scope include/link paths below are all per-pass state, so a cache-guarded early
    # return left the consuming block in CMakeLists.txt with LIBSTATS_HAS_TBB still TRUE and nothing
    # at all to link. The completion flag now suppresses only the repeat status output; the cache
    # writes stay non-FORCE, so the first pass still decides.
    set(_already_reported FALSE)
    if(DEFINED CACHE{LIBSTATS_TBB_DETECTION_COMPLETE})
        set(_already_reported TRUE)
        if(LIBSTATS_VERBOSE_BUILD)
            message(STATUS "Rebinding cached TBB detection results")
        endif()
    elseif(LIBSTATS_VERBOSE_BUILD)
        message(STATUS "Detecting Intel TBB...")
    endif()

    # Local, deliberately NOT named LIBSTATS_HAS_TBB: that name is a cache variable, and a
    # same-named normal variable would shadow it for the rest of this function, making the reads
    # below ambiguous about which one they meant.
    set(_has_tbb FALSE)

    # Method 1: find_package (preferred for vcpkg, conan, system installs)
    find_package(TBB QUIET)
    if(TBB_FOUND)
        set(_has_tbb TRUE)
        if(LIBSTATS_VERBOSE_BUILD AND NOT _already_reported)
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
                set(_has_tbb TRUE)
                if(LIBSTATS_VERBOSE_BUILD AND NOT _already_reported)
                    message(STATUS "  ✓ TBB found via pkg-config")
                endif()
            endif()
        endif()
    endif()

    # Cache result. Non-FORCE, so the first pass's answer stands and an explicit user override is
    # never stomped.
    set(LIBSTATS_HAS_TBB
        ${_has_tbb}
        CACHE BOOL "Intel TBB support available")
    set(LIBSTATS_TBB_DETECTION_COMPLETE
        TRUE
        CACHE BOOL "TBB detection completed")

    if(NOT _already_reported)
        if(_has_tbb)
            message(STATUS "Intel TBB: AVAILABLE - parallel execution policies enhanced")
        else()
            message(
                STATUS
                    "Intel TBB: NOT FOUND - C++20 execution policies may have limited performance")
        endif()
    endif()
endfunction()
