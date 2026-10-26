# Prefer an installed corvus; fall back to fetching a pinned release. Sets LIBSTATS_CORVUS_PROVIDER
# to "system" or "fetched".
#
# corvus is the special-function engine from v3.0.0 on (erf family, incomplete gamma/beta and their
# inverses, lgamma/lbeta, digamma/trigamma, Bessel I0/I1). Its kernels are validated per SIMD tier
# against an mpmath oracle, so the pin below is an accuracy pin as much as an API pin: bump it only
# together with a characterization-sweep regeneration (docs/ACCURACY_CHARACTERIZATION.md). The
# version floor is the same number for the same reason — an older system corvus predates the
# validated bounds.
#
# Provider matters downstream. corvus reaches libstats_static as $<LINK_ONLY:corvus::corvus>, which
# install(EXPORT) can only express when corvus is a package (an imported target); a fetched corvus
# is a build-tree target outside the export set. So `install` is supported only with a system corvus
# — the same rule corvus applies to Highway — and the wheel path (pylibstats FetchContent → libstats
# → corvus → Highway) never installs, so it never trips it.

find_package(corvus 1.0 CONFIG QUIET)

if(corvus_FOUND)
    set(LIBSTATS_CORVUS_PROVIDER "system")
    message(STATUS "libstats: using system corvus ${corvus_VERSION}")
else()
    include(FetchContent)
    set(CORVUS_BUILD_TESTS
        OFF
        CACHE BOOL "" FORCE)
    set(CORVUS_BUILD_EXAMPLES
        OFF
        CACHE BOOL "" FORCE)

    # SYSTEM: fetched headers must not trip libstats' own warning set.
    FetchContent_Declare(
        corvus
        GIT_REPOSITORY https://github.com/OldCrow/corvus.git
        GIT_TAG v1.0.1
        GIT_SHALLOW TRUE
        SYSTEM)

    # A fetched corvus is a dependency, not the code under test, so it compiles as Release whatever
    # libstats' own configuration or instrumentation [user, 2026-10-25]. Highway kernels do not
    # inline at -O0: the coverage leg (Debug, gcov) timed out at 20 minutes on the first adoption
    # run with test_snapshot_consistency still in its 50,000-call loops, and corvus's own sanitizer
    # leg measured 722 s -> 25 s from -O0 to -O1 (corvus f72a128). Instrumenting corvus also only
    # adds noise to libstats' coverage and sanitizer reports. The function scope hands the
    # subdirectory (and the Highway it fetches) Release's per-config flags with the gcov and
    # sanitizer compile flags removed, then discards the edits; libstats' own flags are untouched.
    # The runtime library and $<CONFIG> stay the build's own, so a Debug libstats still links a
    # /MDd corvus under MSVC.
    function(_libstats_fetch_corvus)
        foreach(_lang C CXX)
            foreach(_flag "--coverage" "-fprofile-arcs" "-ftest-coverage" "-fprofile-update=[^ ]*"
                          "-fsanitize=[^ ]*" "-fsanitize-recover=[^ ]*" "-fno-sanitize=[^ ]*")
                string(REGEX REPLACE "(^| )${_flag}( |$)" " " CMAKE_${_lang}_FLAGS
                                     "${CMAKE_${_lang}_FLAGS}")
            endforeach()
            foreach(_cfg DEBUG RELWITHDEBINFO MINSIZEREL STRICT)
                set(CMAKE_${_lang}_FLAGS_${_cfg} "${CMAKE_${_lang}_FLAGS_RELEASE}")
            endforeach()
        endforeach()
        FetchContent_MakeAvailable(corvus)
    endfunction()
    _libstats_fetch_corvus()
    set(LIBSTATS_CORVUS_PROVIDER "fetched")
    message(STATUS "libstats: fetched corvus v1.0.1 via FetchContent; 'install' target disabled")

    # A fetched corvus is compiled by this build's compiler. Under cl.exe that costs twice: Highway
    # blocklists AVX-512, so corvus stops at AVX2, and the kernels come out 3-19x slower than
    # clang-cl compiles them (docs/bench-evidence/2026-09-30-zen4-clangcl-avx2/). Results are
    # bit-identical either way. Two remedies: build everything with clang-cl, or keep cl.exe for
    # libstats and link a clang-cl corvus (same MSVC ABI). A system corvus cannot be checked here,
    # only a fetched one.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(
            WARNING
                "libstats: corvus is being compiled by MSVC (cl.exe): its kernels are capped at "
                "AVX2 and run 3-19x slower than a clang-cl build. For full speed, configure "
                "with clang-cl (-G Ninja -DCMAKE_C_COMPILER=clang-cl "
                "-DCMAKE_CXX_COMPILER=clang-cl), or install corvus and Highway built with "
                "clang-cl and point CMAKE_PREFIX_PATH at them (tools/bench/README.md, "
                "\"Windows\").")
    endif()
endif()
