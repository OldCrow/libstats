# Prefer an installed corvus; fall back to fetching a pinned release. Sets LIBSTATS_CORVUS_PROVIDER
# to "system" or "fetched".
#
# corvus is the special-function engine from v2.5.0 on (erf family, incomplete gamma/beta and their
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
    FetchContent_MakeAvailable(corvus)
    set(LIBSTATS_CORVUS_PROVIDER "fetched")
    message(STATUS "libstats: fetched corvus v1.0.1 via FetchContent; 'install' target disabled")
endif()
