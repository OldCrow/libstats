/**
 * @file test_parallel_exception_propagation.cpp
 * @brief Guards for #118 — ParallelUtils::parallelFor propagates exceptions
 *        thrown inside a chunk, and does so only after every chunk has
 *        finished running.
 *
 * Two properties are guarded here, and they pull against each other:
 *
 * 1. An exception thrown by the caller's lambda must reach the caller.
 *    Waiting on the chunk futures with `wait()` drops it: a packaged_task
 *    stores the exception in its future and `wait()` never reads it, so the
 *    caller saw success and a partially written output.
 * 2. No chunk may still be executing when the exception leaves parallelFor.
 *    The obvious fix — replacing `wait()` with `get()` in the same loop —
 *    reintroduces this: the first `get()` unwinds the caller's frame while
 *    sibling chunks still hold by-reference captures of the lambda and the
 *    caller's buffers. Correct shape is wait-all, then harvest.
 *
 * Deliberately unlabelled so it runs in the standard correctness suite
 * (`ctest -LE "timing|benchmark"`); a guard CI never runs guards nothing.
 */

#include "libstats/platform/platform_constants.h"
#include "libstats/platform/thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace stats;

namespace {

/// A range guaranteed to clear parallelFor's sequential-execution threshold,
/// so these tests exercise the chunked path rather than the inline loop.
std::size_t parallelRange() {
    const std::size_t minParallel = arch::get_min_elements_for_parallel();
    return (minParallel < 16384u) ? 65536u : (minParallel * 4u);
}

struct ChunkFailure : std::runtime_error {
    explicit ChunkFailure(const char* what) : std::runtime_error(what) {}
};

}  // namespace

TEST(ParallelForExceptions, TestExercisesTheChunkedPath) {
    const std::size_t range = parallelRange();
    ASSERT_GE(range, arch::get_min_elements_for_parallel())
        << "range falls below the sequential-execution threshold, so the "
           "chunked path these tests target would never run";
}

TEST(ParallelForExceptions, PropagatesExceptionFromChunk) {
    const std::size_t range = parallelRange();
    std::vector<double> output(range, 0.0);

    EXPECT_THROW(ParallelUtils::parallelFor(std::size_t{0}, range,
                                            [&](std::size_t i) {
                                                if (i == 0) {
                                                    throw ChunkFailure("chunk 0 failed");
                                                }
                                                output[i] = static_cast<double>(i);
                                            }),
                 ChunkFailure);
}

TEST(ParallelForExceptions, ExceptionCarriesTheOriginalPayload) {
    const std::size_t range = parallelRange();

    try {
        ParallelUtils::parallelFor(std::size_t{0}, range, [](std::size_t i) {
            if (i == 0) {
                throw ChunkFailure("chunk 0 failed");
            }
        });
        FAIL() << "parallelFor returned normally although a chunk threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("chunk 0 failed", e.what());
    }
}

TEST(ParallelForExceptions, NoChunkStillRunningWhenExceptionEscapes) {
    const std::size_t range = parallelRange();
    std::atomic<std::size_t> inFlight{0};
    std::atomic<std::size_t> peakObservedAtThrow{0};

    // The first index throws immediately; the last index of the range sleeps,
    // so a harvest that rethrows before waiting for every chunk would return
    // while that chunk is still holding references into this frame.
    try {
        ParallelUtils::parallelFor(std::size_t{0}, range, [&](std::size_t i) {
            inFlight.fetch_add(1, std::memory_order_acq_rel);
            if (i == 0) {
                inFlight.fetch_sub(1, std::memory_order_acq_rel);
                throw ChunkFailure("chunk 0 failed");
            }
            if (i + 1 == range) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            }
            inFlight.fetch_sub(1, std::memory_order_acq_rel);
        });
        FAIL() << "parallelFor returned normally although a chunk threw";
    } catch (const ChunkFailure&) {
        peakObservedAtThrow.store(inFlight.load(std::memory_order_acquire),
                                  std::memory_order_release);
    }

    EXPECT_EQ(0u, peakObservedAtThrow.load(std::memory_order_acquire))
        << "the exception escaped parallelFor while sibling chunks were still "
           "executing — those chunks hold by-reference captures of the caller's "
           "frame (#118)";
}

TEST(ParallelForExceptions, FirstThrowingChunkInIndexOrderWins) {
    const std::size_t range = parallelRange();

    // Two chunks throw. The contract is that the first one in chunk-submission
    // order — not the first one to fail in wall-clock time — reaches the
    // caller; the rest are discarded during the harvest.
    try {
        ParallelUtils::parallelFor(std::size_t{0}, range, [&](std::size_t i) {
            if (i == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                throw ChunkFailure("first chunk");
            }
            if (i + 1 == range) {
                throw ChunkFailure("last chunk");
            }
        });
        FAIL() << "parallelFor returned normally although two chunks threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("first chunk", e.what());
    }
}

TEST(ParallelForExceptions, SequentialPathPropagates) {
    // Below the threshold parallelFor runs the loop inline; the contract is the
    // same on both sides of that branch.
    const std::size_t range = 4;

    EXPECT_THROW(ParallelUtils::parallelFor(std::size_t{0}, range,
                                            [](std::size_t i) {
                                                if (i == 2) {
                                                    throw ChunkFailure("inline loop failed");
                                                }
                                            }),
                 ChunkFailure);
}

TEST(ParallelForExceptions, NonThrowingRunIsUnaffected) {
    const std::size_t range = parallelRange();
    std::vector<double> output(range, -1.0);

    EXPECT_NO_THROW(ParallelUtils::parallelFor(
        std::size_t{0}, range, [&](std::size_t i) { output[i] = static_cast<double>(i) * 2.0; }));

    for (std::size_t i = 0; i < range; ++i) {
        ASSERT_DOUBLE_EQ(static_cast<double>(i) * 2.0, output[i]) << "at index " << i;
    }
}
