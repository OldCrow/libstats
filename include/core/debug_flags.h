#pragma once

/**
 * @file core/debug_flags.h
 * @brief Canonical location for all libstats compile-time debug flags.
 *
 * Add new flags here. Never gate diagnostic output with a raw `#ifdef`
 * directly at a call site — that makes the dead branch invisible to
 * the compiler, tooling, and static analysers.
 *
 * Usage pattern (always `if constexpr`, never `#ifdef` at the call site):
 * @code
 *   #include "libstats/core/debug_flags.h"
 *   if constexpr (stats::kDebugThreading) {
 *       std::clog << "Worker " << id << " ready\n";
 *   }
 * @endcode
 *
 * Enable a flag by defining the corresponding macro at compile time, e.g.:
 *   cmake -DCMAKE_CXX_FLAGS="-DLIBSTATS_DEBUG_THREADING" ...
 */

namespace stats {

/// Enable verbose thread-pool lifecycle logging (worker ready, steal events).
#ifdef LIBSTATS_DEBUG_THREADING
inline constexpr bool kDebugThreading = true;
#else
inline constexpr bool kDebugThreading = false;
#endif

}  // namespace stats
