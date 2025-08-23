#pragma once

/**
 * @file common/distribution_base_common.h
 * @brief Common dependencies for distribution interface and base classes
 *
 * This header consolidates the common standard library and core includes
 * needed by distribution interface components, following our policy of
 * reducing redundancy while maintaining clear separation of concerns.
 *
 * This header is used by:
 * - distribution_interface.h
 * - distribution_base.h
 * - distribution_memory.h
 * - distribution_validation.h
 *
 * Individual headers still include their specific dependencies as needed.
 */

// Standard library includes common to distribution interfaces
#include <chrono>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

// Core libstats headers common to distribution interfaces
#include "../core/error_handling.h"       // Result types and validation
#include "../core/essential_constants.h"  // Only essential constants

namespace stats {

// Forward declarations to reduce compile-time dependencies
class DistributionInterface;
class DistributionBase;
class ThreadSafeCacheManager;

// Common validation result types used across distribution interfaces
struct ValidationResult;
struct FitResults;

}  // namespace stats
