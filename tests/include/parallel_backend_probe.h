#pragma once

// #179 guard helper. Includes parallel_execution.h and names the backend it selected in the
// including translation unit. Each probe translation unit decides for itself what comes before
// this header, which is the point of the test.

#include "libstats/platform/parallel_execution.h"

#if defined(LIBSTATS_HAS_STD_EXECUTION)
    #define LIBSTATS_TEST_SELECTED_BACKEND 1
#elif defined(LIBSTATS_HAS_GCD)
    #define LIBSTATS_TEST_SELECTED_BACKEND 2
#elif defined(LIBSTATS_HAS_WIN_THREADPOOL)
    #define LIBSTATS_TEST_SELECTED_BACKEND 3
#elif defined(LIBSTATS_HAS_OPENMP)
    #define LIBSTATS_TEST_SELECTED_BACKEND 4
#elif defined(LIBSTATS_HAS_PTHREADS)
    #define LIBSTATS_TEST_SELECTED_BACKEND 5
#else
    #define LIBSTATS_TEST_SELECTED_BACKEND 0
#endif

namespace libstats_test {

inline constexpr int kBackendNone = 0;
inline constexpr int kBackendStdExecution = 1;
inline constexpr int kBackendGcd = 2;
inline constexpr int kBackendWinThreadPool = 3;
inline constexpr int kBackendOpenMp = 4;
inline constexpr int kBackendPthreads = 5;

// One per probe translation unit (tests/parallel_backend_probe_*.cpp).
int backendWithNothingBefore() noexcept;
int backendAfterVersion() noexcept;
int backendAfterString() noexcept;

}  // namespace libstats_test
