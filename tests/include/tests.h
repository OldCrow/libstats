#pragma once

/**
 * @file tests/tests.h
 * @brief Unified test infrastructure header
 *
 * Single include for all test infrastructure: constants, fixtures,
 * validators, and benchmarks, all under the stats::tests:: namespace.
 */

// Core test infrastructure components
#include "benchmarks.h"
#include "constants.h"
#include "fixtures.h"
#include "validators.h"

namespace stats {
namespace tests {

//==============================================================================
// Convenience Macros for Common Test Patterns
//==============================================================================

// Quick access to common test utilities
#define TEST_CONSTANTS stats::tests::constants
#define TEST_FIXTURES stats::tests::fixtures
#define TEST_VALIDATORS stats::tests::validators
#define TEST_BENCHMARKS stats::tests::benchmarks

// Convenience macros for architecture-aware validation
#define EXPECT_SIMD_SPEEDUP(measured, batch_size, is_complex)                                      \
    EXPECT_TRUE(TEST_VALIDATORS::validateSIMDSpeedup((measured), (batch_size), (is_complex)))      \
        << "SIMD speedup " << (measured) << "x should exceed adaptive threshold for batch size "   \
        << (batch_size)

#define EXPECT_PARALLEL_SPEEDUP(measured, batch_size, is_complex)                                  \
    EXPECT_TRUE(TEST_VALIDATORS::validateParallelSpeedup((measured), (batch_size), (is_complex)))  \
        << "Parallel speedup " << (measured)                                                       \
        << "x should exceed adaptive threshold for batch size " << (batch_size)

}  // namespace tests
}  // namespace stats

//==============================================================================
// Global Using Declarations for Convenience (Optional)
//==============================================================================

// Uncomment these if you want global access to test utilities
// using TestConstants = stats::tests::constants;
// using TestFixtures = stats::tests::fixtures;
// using TestValidators = stats::tests::validators;
// using TestBenchmarks = stats::tests::benchmarks;
