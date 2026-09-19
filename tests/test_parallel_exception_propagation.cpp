/**
 * @file test_parallel_exception_propagation.cpp
 * @brief Guards for #118 and #127 — ParallelUtils::parallelFor propagates exceptions
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
 * #127 extends the same two properties to the value-returning primitives,
 * ParallelUtils::parallelReduce and parallelStatOperation. They had the
 * opposite defect from #118's: they harvested with `get()` inside the combine
 * loop, so the exception did arrive — but before the sibling chunks finished
 * (property 2). Their guards count completed work rather than sampling an
 * in-flight counter: with the chunk count pinned, "everything that could
 * complete has completed" is an exact equality in the catch block, and the
 * unfixed code fails it deterministically instead of by a race.
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
#include <numeric>
#include <span>
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

/// Chunk count the #127 guards pin. Passing grainSize = range / kPinnedChunks
/// makes it the effective grain in both primitives: parallelReduce takes
/// max(grainSize, range / (threads * 4)) and range / (threads * 4) can never
/// exceed range / 4; parallelStatOperation rounds grainSize up to the SIMD
/// width, a no-op because pinnedRange() / kPinnedChunks is a multiple of 8.
constexpr std::size_t kPinnedChunks = 4;

/// A size that clears parallelStatOperation's distribution-parallel threshold
/// and divides into kPinnedChunks chunks of a multiple of 8 elements.
std::size_t pinnedRange() {
    const std::size_t minParallel = arch::get_min_elements_for_distribution_parallel();
    const std::size_t wanted = (minParallel < 16384u) ? 65536u : (minParallel * 4u);
    const std::size_t quantum = kPinnedChunks * 8u;
    return ((wanted + quantum - 1u) / quantum) * quantum;
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

// ---------------------------------------------------------------------------
// #127 — parallelReduce
// ---------------------------------------------------------------------------

TEST(ParallelReduceExceptions, AllSiblingChunksCompleteBeforeExceptionEscapes) {
    const std::size_t range = pinnedRange();
    const std::size_t grain = range / kPinnedChunks;
    std::atomic<std::size_t> completed{0};

    // Index 0 throws at once, so chunk 0 completes nothing. The last index
    // sleeps, so its chunk is still running when an early harvest rethrows.
    try {
        (void)ParallelUtils::parallelReduce(
            std::size_t{0}, range, 0.0,
            [&](std::size_t i) -> double {
                if (i == 0) {
                    throw ChunkFailure("chunk 0 failed");
                }
                if (i + 1 == range) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
                completed.fetch_add(1, std::memory_order_acq_rel);
                return 1.0;
            },
            [](double a, double b) { return a + b; }, grain);
        FAIL() << "parallelReduce returned normally although a chunk threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("chunk 0 failed", e.what());
        EXPECT_EQ(range - grain, completed.load(std::memory_order_acquire))
            << "the exception escaped parallelReduce while sibling chunks were "
               "still executing — they hold by-reference captures of task, "
               "reduce and the caller's frame (#127)";
    }
}

TEST(ParallelReduceExceptions, FirstThrowingChunkInIndexOrderWins) {
    const std::size_t range = pinnedRange();
    const std::size_t grain = range / kPinnedChunks;

    try {
        (void)ParallelUtils::parallelReduce(
            std::size_t{0}, range, 0.0,
            [&](std::size_t i) -> double {
                if (i == 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    throw ChunkFailure("first chunk");
                }
                if (i + 1 == range) {
                    throw ChunkFailure("last chunk");
                }
                return 1.0;
            },
            [](double a, double b) { return a + b; }, grain);
        FAIL() << "parallelReduce returned normally although two chunks threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("first chunk", e.what());
    }
}

TEST(ParallelReduceExceptions, SequentialPathPropagates) {
    // range <= grainSize runs the loop inline.
    EXPECT_THROW((void)ParallelUtils::parallelReduce(
                     std::size_t{0}, std::size_t{4}, 0.0,
                     [](std::size_t i) -> double {
                         if (i == 2) {
                             throw ChunkFailure("inline loop failed");
                         }
                         return 1.0;
                     },
                     [](double a, double b) { return a + b; }, std::size_t{8}),
                 ChunkFailure);
}

TEST(ParallelReduceExceptions, NonThrowingRunIsUnaffected) {
    const std::size_t range = pinnedRange();
    const double sum = ParallelUtils::parallelReduce(
        std::size_t{0}, range, 0.0, [](std::size_t) { return 1.0; },
        [](double a, double b) { return a + b; }, range / kPinnedChunks);
    EXPECT_DOUBLE_EQ(static_cast<double>(range), sum);
}

// ---------------------------------------------------------------------------
// #127 — parallelStatOperation
// ---------------------------------------------------------------------------

TEST(ParallelStatOperationExceptions, TestPinsTheChunkCount) {
    // Premise guard: the completed-chunk equality below is only exact if the
    // operation really is invoked once per pinned chunk, on the chunked path.
    const std::vector<double> data(pinnedRange(), 1.0);
    std::atomic<std::size_t> invocations{0};

    const double sum = ParallelUtils::parallelStatOperation(
        std::span<const double>(data),
        [&](std::span<const double> chunk) {
            invocations.fetch_add(1, std::memory_order_acq_rel);
            return std::accumulate(chunk.begin(), chunk.end(), 0.0);
        },
        [](double a, double b) { return a + b; }, data.size() / kPinnedChunks);

    EXPECT_EQ(kPinnedChunks, invocations.load(std::memory_order_acquire));
    EXPECT_DOUBLE_EQ(static_cast<double>(data.size()), sum);
}

TEST(ParallelStatOperationExceptions, AllSiblingChunksCompleteBeforeExceptionEscapes) {
    const std::vector<double> data(pinnedRange(), 1.0);
    const std::span<const double> all(data);
    std::atomic<std::size_t> completed{0};

    try {
        (void)ParallelUtils::parallelStatOperation(
            all,
            [&](std::span<const double> chunk) -> double {
                if (chunk.data() == all.data()) {
                    throw ChunkFailure("chunk 0 failed");
                }
                if (chunk.data() + chunk.size() == all.data() + all.size()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(150));
                }
                completed.fetch_add(1, std::memory_order_acq_rel);
                return 0.0;
            },
            [](double a, double b) { return a + b; }, data.size() / kPinnedChunks);
        FAIL() << "parallelStatOperation returned normally although a chunk threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("chunk 0 failed", e.what());
        EXPECT_EQ(kPinnedChunks - 1u, completed.load(std::memory_order_acquire))
            << "the exception escaped parallelStatOperation while sibling chunks "
               "were still executing — they hold by-reference captures of "
               "operation and the caller's data (#127)";
    }
}

TEST(ParallelStatOperationExceptions, FirstThrowingChunkInIndexOrderWins) {
    const std::vector<double> data(pinnedRange(), 1.0);
    const std::span<const double> all(data);

    try {
        (void)ParallelUtils::parallelStatOperation(
            all,
            [&](std::span<const double> chunk) -> double {
                if (chunk.data() == all.data()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    throw ChunkFailure("first chunk");
                }
                if (chunk.data() + chunk.size() == all.data() + all.size()) {
                    throw ChunkFailure("last chunk");
                }
                return 0.0;
            },
            [](double a, double b) { return a + b; }, data.size() / kPinnedChunks);
        FAIL() << "parallelStatOperation returned normally although two chunks threw";
    } catch (const ChunkFailure& e) {
        EXPECT_STREQ("first chunk", e.what());
    }
}
