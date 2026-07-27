#pragma once

/**
 * @file core/parallel_batch_fit.h
 * @brief Generic parallel batch-fitting helper for all distributions.
 *
 * detail::batchFitParallel<DistT>(datasets, results) handles all thread-pool
 * submission, per-chunk error recording, serial fallback, and serial error
 * wrapping. Every distribution's static parallelBatchFit delegates here,
 * keeping only the type-specific results parameter at the call site.
 *
 * This replaces the duplicate implementations previously found in Gaussian,
 * Exponential, Uniform, Discrete, Poisson, and Gamma, and provides the
 * standard implementation for Chi-squared, Student's t, and Beta.
 */

#include "libstats/core/dispatch_thresholds.h"
#include "libstats/core/distribution_concepts.h"
#include "libstats/platform/thread_pool.h"

#include <atomic>
#include <future>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace stats {
namespace detail {

/**
 * @brief Fit multiple independent datasets to the same distribution type in parallel.
 *
 * For @p num_datasets >= dispatch_table::BATCH_FIT_MIN: submits chunks to the
 * global ThreadPool with full error recording and a serial fallback on pool
 * failure. For smaller counts: runs serially with per-dataset error wrapping.
 *
 * @tparam DistT  Distribution type satisfying concepts::FittableDistribution.
 * @param datasets Input: one dataset per distribution fit.
 * @param results  In/out: resized to datasets.size() if needed; each element
 *                 receives the parameters fitted to the corresponding dataset.
 *
 * @note Serial fallback is attempted only when the thread pool cannot be
 *       acquired or tasks cannot be submitted. Task-level fit errors are
 *       reported immediately without re-running successful fits.
 */
template <concepts::FittableDistribution DistT>
void batchFitParallel(const std::vector<std::vector<double>>& datasets,
                      std::vector<DistT>& results) {
    if (datasets.empty()) {
        results.clear();
        return;
    }
    if (results.size() != datasets.size()) {
        results.resize(datasets.size());
    }

    const std::size_t num_datasets = datasets.size();

    if (num_datasets >= dispatch_table::BATCH_FIT_MIN) {
        // One mutex per DistT instantiation — serialises pool-pointer acquisition
        // across concurrent callers without blocking unrelated distribution types.
        static std::mutex pool_access_mutex;

        // Declared outside the try block so the catch can inspect them (NEW-AR-2).
        std::atomic<bool> has_error{false};
        std::mutex error_mutex;
        std::string error_message;
        bool tasks_submitted = false;

        try {
            ThreadPool* pool_ptr = nullptr;
            {
                std::lock_guard<std::mutex> pool_lock(pool_access_mutex);
                pool_ptr = &ParallelUtils::getGlobalThreadPool();
            }

            const std::size_t grain_size = std::max(std::size_t{1}, num_datasets / 8);
            const std::size_t num_chunks = (num_datasets + grain_size - 1) / grain_size;

            std::vector<std::future<void>> futures;
            futures.reserve(num_chunks);

            for (std::size_t i = 0; i < num_datasets; i += grain_size) {
                const std::size_t chunk_start = i;
                const std::size_t chunk_end = std::min(i + grain_size, num_datasets);

                futures.push_back(pool_ptr->submit([&datasets, &results, chunk_start, chunk_end,
                                                    &has_error, &error_mutex, &error_message]() {
                    try {
                        for (std::size_t j = chunk_start; j < chunk_end; ++j) {
                            results[j].fit(datasets[j]);
                        }
                    } catch (const std::exception& e) {
                        std::lock_guard<std::mutex> lk(error_mutex);
                        if (!has_error.load()) {
                            error_message = "Parallel batch fit error in chunk [" +
                                            std::to_string(chunk_start) + ", " +
                                            std::to_string(chunk_end) + "): " + e.what();
                            has_error.store(true, std::memory_order_release);
                        }
                    } catch (...) {
                        std::lock_guard<std::mutex> lk(error_mutex);
                        if (!has_error.load()) {
                            error_message = "Unknown error in parallel batch fit chunk [" +
                                            std::to_string(chunk_start) + ", " +
                                            std::to_string(chunk_end) + ")";
                            has_error.store(true, std::memory_order_release);
                        }
                    }
                }));
            }
            tasks_submitted = true;

            bool all_completed = true;
            for (auto& f : futures) {
                try {
                    f.wait();
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(error_mutex);
                    if (!has_error.load()) {
                        error_message = "Future wait error: " + std::string(e.what());
                        has_error.store(true, std::memory_order_release);
                    }
                    all_completed = false;
                }
            }

            if (has_error.load()) {
                std::lock_guard<std::mutex> lk(error_mutex);
                throw std::runtime_error("Parallel batch fitting failed: " + error_message);
            }
            if (!all_completed) {
                throw std::runtime_error(
                    "Some parallel batch fitting tasks failed to complete properly");
            }

        } catch (const std::exception& e) {
            // If tasks were already submitted, the fits ran (some may have
            // written results, others failed). Do not silently re-run the
            // whole batch: propagate the error as-is (NEW-AR-2).
            if (tasks_submitted) {
                throw;
            }
            // Pool acquisition or submission failed before any task ran;
            // fall back to a serial pass with per-dataset error wrapping.
            for (std::size_t i = 0; i < num_datasets; ++i) {
                try {
                    results[i].fit(datasets[i]);
                } catch (const std::exception& fit_error) {
                    throw std::runtime_error("Serial fallback failed for dataset " +
                                             std::to_string(i) + ": " + fit_error.what() +
                                             " (pool setup error: " + e.what() + ")");
                }
            }
        }
    } else {
        // Below parallel threshold: run serially with per-dataset error wrapping.
        for (std::size_t i = 0; i < num_datasets; ++i) {
            try {
                results[i].fit(datasets[i]);
            } catch (const std::exception& e) {
                throw std::runtime_error("Serial batch fit failed for dataset " +
                                         std::to_string(i) + ": " + e.what());
            }
        }
    }
}

}  // namespace detail
}  // namespace stats
