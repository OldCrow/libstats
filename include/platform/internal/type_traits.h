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

/// @brief Portable function-return-type deduction.
///
/// Uses std::invoke_result_t when available (C++17+). Falls back to
/// std::result_of for older compilers — retained for cross-platform
/// compatibility with the Catalina-era toolchain until v2.0.0 raises the
/// minimum compiler baseline.
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
template <typename F, typename... Args>
using result_of_t = std::invoke_result_t<F, Args...>;
#else
template <typename F, typename... Args>
using result_of_t = typename std::result_of<F(Args...)>::type;
#endif

}  // namespace stats
