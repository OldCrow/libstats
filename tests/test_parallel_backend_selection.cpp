// Guards for #179: parallel_execution.h must select the same backend in every translation unit
// of a program, and the backend it selects must run in parallel.
//
// Before the fix the header tested __cpp_lib_execution without including <version>, so the
// answer depended on what the translation unit had included first. On libstdc++ the three probe
// translation units of this test disagreed (std::execution after <version>, pthreads otherwise),
// and a build without TBB that did reach std::execution ran its policies serially.

#include "include/parallel_backend_probe.h"

#include <atomic>
#include <cstddef>
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <version>

namespace {

using namespace libstats_test;

// The backend this build should select, decided here from the standard library and the
// configure-time fact alone, without reading any macro parallel_execution.h defines. Returns
// kBackendNone where this test has no independent expectation.
int expectedBackend() noexcept {
#if defined(__GLIBCXX__)
    // libstdc++: its policies are parallel only on TBB, which the build must link and this
    // translation unit must see.
    #if defined(LIBSTATS_HAS_TBB) && __has_include(<tbb/tbb.h>)
    return kBackendStdExecution;
    #elif defined(__APPLE__)
    return kBackendGcd;
    #elif defined(_OPENMP)
    return kBackendOpenMp;
    #else
    return kBackendPthreads;
    #endif
#elif defined(_MSVC_STL_VERSION)
    return kBackendStdExecution;
#elif defined(__APPLE__)
    return kBackendGcd;
#else
    return kBackendNone;
#endif
}

TEST(ParallelBackendSelection, Issue179_SameBackendWhateverWasIncludedFirst) {
    const int here = LIBSTATS_TEST_SELECTED_BACKEND;
    EXPECT_EQ(backendWithNothingBefore(), here);
    EXPECT_EQ(backendAfterVersion(), here);
    EXPECT_EQ(backendAfterString(), here);
}

TEST(ParallelBackendSelection, Issue179_BackendMatchesTheBuild) {
    const int expected = expectedBackend();
    if (expected == kBackendNone)
        GTEST_SKIP() << "no independent expectation for this standard library";
    EXPECT_EQ(LIBSTATS_TEST_SELECTED_BACKEND, expected);
    EXPECT_EQ(backendWithNothingBefore(), expected);
}

// A selected backend must use more than one thread on a machine that has more than one. The
// unfixed header could select std::execution on a libstdc++ without TBB, whose policies run
// every element on the calling thread.
TEST(ParallelBackendSelection, Issue179_SafeForEachUsesMoreThanOneThread) {
    if (std::thread::hardware_concurrency() < 2)
        GTEST_SKIP() << "one hardware thread";
    if (LIBSTATS_TEST_SELECTED_BACKEND == kBackendNone)
        GTEST_SKIP() << "no parallel backend on this platform";

    const std::size_t n = 400000;
    ASSERT_TRUE(stats::arch::should_use_parallel(n));
    std::vector<std::size_t> items(n);
    for (std::size_t i = 0; i < n; ++i)
        items[i] = i;

    std::atomic<int> threads_seen{0};
    std::atomic<std::size_t> sink{0};
    stats::arch::safe_for_each(items.begin(), items.end(), [&](std::size_t v) {
        thread_local bool counted = false;
        if (!counted) {
            counted = true;
            threads_seen.fetch_add(1, std::memory_order_relaxed);
        }
        // About a microsecond of work, so the range outlasts the start of the workers.
        std::size_t acc = v;
        for (int k = 0; k < 400; ++k)
            acc = acc * 6364136223846793005ULL + 1442695040888963407ULL;
        if ((acc & 0xFFFFF) == 0)
            sink.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_GE(threads_seen.load(), 2) << "safe_for_each ran " << n << " elements on one thread";
}

}  // namespace
