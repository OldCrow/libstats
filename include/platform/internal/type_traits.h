#pragma once

/**
 * @file platform/internal/type_traits.h
 * @brief Shared internal type-trait utilities for platform pool headers
 *
 * Centralises the result_of_t compatibility alias that was previously
 * duplicated in thread_pool.h, work_stealing_pool.h, and
 * work_stealing_pool.cpp. Including a single canonical definition here
 * ensures all three sites stay in sync.
 *
 * This header is an implementation detail; consumers of the library should
 * not include it directly.
 */

#include <type_traits>

namespace stats {

/// @brief Portable function-return-type deduction (C++17 std::invoke_result_t).
/// Catalina-era std::result_of fallback removed in v2.0.0.
template <typename F, typename... Args>
using result_of_t = std::invoke_result_t<F, Args...>;

}  // namespace stats
