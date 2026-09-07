---
name: add-distribution
description: Step-by-step checklist for adding a new statistical distribution to libstats (enum registration, dispatch thresholds, header/source implementation, test files, CMake/libstats.h registration, threshold profiling, and docs). Use this whenever asked to add, implement, or register a new distribution in libstats.
---

# Creating New Distributions in libstats

The registration checklist is authoritative in `include/libstats/core/distribution_meta.h`.
The v2.4.0 eight — Logistic through TruncatedNormal (19–26 in enum order) — are the most
recently implemented (2026-09-03); for any future distribution (N+1), follow all 7 steps below.

**Steps for any future distribution (N+1):**

1. **Append** the new `DistributionType` enum value to `include/libstats/core/distribution_type.h`
   (append-only; never reorder — values are used as array indices).
2. **Append** a `DistributionMeta` row to `kDistributionMeta[]` in `include/libstats/core/distribution_meta.h`
   (enum name, display name, `is_discrete`, `is_delegation_wrapper`). Bump the
   `static_assert(kDistributionTypeCount >= N, ...)` minimum to match the new count.
3. **Append** one `ThresholdRow` to each of the five `kXxx` tables in
   `include/libstats/core/dispatch_thresholds.h` (use `{NEVER, NEVER, NEVER}` until profiled;
   the fifth table, `kNone`, takes a T1/T2/T3 tier value per its header comment, not NEVER).
   For delegation wrappers (e.g. Geometric→NegBinomial, Cauchy→StudentT), the delegate's
   thresholds apply — copy them or leave NEVER and profile after implementation.

4. **Implement** the distribution:

   *Header* `include/libstats/distributions/dist.h` — use `exponential.h` as the reference:
   - Inherit from `DistributionBase`.
   - Declare `static constexpr detail::DistributionType kDistributionType = detail::DistributionType::DIST_NAME;`
     and `static constexpr bool kIsDiscrete = false/true;` (must match the metadata row).
   - Declare `noexcept` move constructor and move assignment operator.
   - Declare `static void parallelBatchFit(const std::vector<std::vector<double>>&, std::vector<DistType>&);`
   - Override all pure virtuals from `DistributionInterface`: `getMean`, `getVariance`, `getSkewness`,
     `getKurtosis`, `getNumParameters`, `getDistributionName`, `isDiscrete`,
     `getSupportLowerBound`, `getSupportUpperBound`, `getProbability`, `getLogProbability`,
     `getCumulativeProbability`, `getQuantile`, `sample` (×2), `fit`, `reset`, `toString`.
   - Override `getEntropy()` and `getMedian()` (both have NaN defaults in the interface;
     concrete implementations are required even for wrappers).
   - Declare the three batch span overloads: `getProbability(span, span, hint)`,
     `getLogProbability(span, span, hint)`, `getCumulativeProbability(span, span, hint)`.
   - Declare comparison operators (`==`, `!=`) and friend stream operators (`<<`, `>>`).

   *Source* `src/dist.cpp`: full implementations in the numbered section structure.

   *Basic test* `tests/test_dist_basic.cpp`:
   - `#include "include/basic_test_runner.h"`
   - Define `stats::tests::BasicDistConfig cfg{name, small_values, lo, hi, invalid_scenarios};`
   - Keep Tests 1–5 and 7 per-distribution.
   - Call `stats::tests::runBatchTests(cfg, dist);` for Test 6.
   - Call `stats::tests::runErrorTests(cfg);` for Test 8.

   *Enhanced test* `tests/test_dist_enhanced.cpp`:
   - `#include "include/enhanced_test_suite.h"`
   - Implement `template<> struct stats::tests::DistTraits<DistType> : stats::tests::DistTraitsDefaults { ... };`
     with `make()`, `domain()`, `batch_lo()`, `batch_hi()`, `invalid_creators()`.
     Override tolerances for distributions whose SIMD path has documented approximation error
     (e.g. VonMises pdf_tolerance = 1e-10 for vector_cos).
   - Close with `INSTANTIATE_TYPED_TEST_SUITE_P(Name, DistributionEnhancedTest, ::testing::Types<DistType>);`
   - Add per-distribution tests: known analytical values, moment formulas, special cases,
     VectorizedMatchesScalar, VectorizedSpeedup (timing-labelled), MLEFit.

5. **Register** in four CMakeLists.txt locations (one top-level, three in
   `tests/CMakeLists.txt`) and in `include/libstats/libstats.h`:

   *`CMakeLists.txt` (top-level) — `LIBSTATS_DISTRIBUTIONS_SOURCES`*, in the
   "Level 5: Distribution Implementations" block:
   Add `src/dist.cpp` to the Level-5 distributions source list.

   *`tests/CMakeLists.txt` — Level-5 registration block* (under the
   "LEVEL 5 TESTS: Concrete Distribution Implementations" banner comment):
   ```cmake
   create_libstats_test(test_dist_basic test_dist_basic.cpp)
   create_libstats_gtest(test_dist_enhanced test_dist_enhanced.cpp)
   ```

   *`tests/CMakeLists.txt` — `run_all_tests` DEPENDS block* (the
   `add_custom_target(run_all_tests ...)` near the end of the file):
   Add `test_dist_basic` and `test_dist_enhanced` to the dependency list.

   *`tests/CMakeLists.txt` — timing label* (if the enhanced test has speedup assertions):
   Add `test_dist_enhanced` to the `set_tests_properties(... PROPERTIES LABELS "timing")` call.

   *`include/libstats/libstats.h`* — inside `#ifdef LIBSTATS_FULL_INTERFACE`:
   - Add `#include "distributions/dist.h"`
   - Add `using DistName = DistNameDistribution;` in the `namespace stats { ... }` type-alias block.

6. **Profile and calibrate thresholds** (after correctness tests pass on all target machines):
   - Run `./build-release/tools/strategy_profile --large -o <path.csv>` (release build;
     there is no `--export` flag) to produce a CSV — three quiet-machine runs, read with
     sustained V→P crossovers, not the validator's first-crossing heuristic.
   - For a tier with no native hardware in the fleet (kAvx since the 2012 MBP retired):
     configure a capped Release build (`-DLIBSTATS_MAX_SIMD_TIER=AVX`) on the closest
     machine class, ASSERT the tier before profiling (`system_inspector --quick` active
     tier + no higher-tier `vector_*` kernel symbols in the archive), then profile as
     above. Label the rows as a capped-build measurement — precedent: the 2026-09-04
     kAvx leg (bundle `2026-09-04T23-51-14Z_…`).
   - Run `./build/tools/threshold_validator <csv>` to compare measured crossovers against
     the current NEVER entries and identify which need updating.
   - Update the five `kXxx` tables in `dispatch_thresholds.h` accordingly.
   - For delegation wrappers, verify the delegate's thresholds apply (skip if identical).

7. **Extend the user-facing docs and examples**: add the distribution to the
   README roster (family placement + count) and to
   `examples/distribution_families_demo.cpp` in its family section — part of
   a distribution's definition of done, not release polish.

The `consteval validateMetaOrdering()` in `distribution_meta.h` enforces step 1↔2 alignment at
compile time. A clean build after any enum or table change verifies consistency.
