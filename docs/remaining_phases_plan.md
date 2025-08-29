# Remaining Phases for Namespace Architecture Completion

## Current State Assessment

### ✅ Completed Namespaces:
- `stats::simd::` (ops, dispatch, utils) - Phase 3B ✓
- `stats::arch::memory::` (prefetch, access, allocation) - Phase 3C ✓
- `stats::arch::parallel::` (sse, avx, avx2, avx512, neon, fallback) - Existing ✓
- `stats::detail::` - Existing ✓

### ❌ Namespaces to Eliminate/Reorganize:
1. `stats::constants::` - Move to `arch::` or `tests::`
2. `stats::memory::` - Already moved to `arch::memory::`
3. `stats::performance::` - Move relevant parts to `arch::` or eliminate
4. `stats::safety::` - Move to `detail::`
5. `stats::validation::` - Move to `tests::validators::`
6. `stats::test::` / `stats::testing::` - Consolidate to `tests::`
7. `stats::tools::` - Eliminate or move to appropriate locations

## Phase 3D: CPU-Specific Architecture Namespaces

### Objective:
Create CPU vendor-specific namespaces under `stats::arch::cpu::`

### Tasks:
1. Create namespace structure:
   - `stats::arch::cpu::intel::`
   - `stats::arch::cpu::amd::`
   - `stats::arch::cpu::arm::`
   - `stats::arch::cpu::apple_silicon::`

2. Move CPU-specific tuning parameters from platform_constants.h
3. Create CPU-specific optimization constants
4. Update all references

### Files to Modify:
- `include/platform/platform_constants.h`
- `src/platform_constants_impl.cpp`
- `include/platform/cpu_detection.h`

## Phase 3E: Test Infrastructure Namespace

### Objective:
Create `stats::tests::` namespace hierarchy for test utilities

### Tasks:
1. Create namespace structure:
   - `stats::tests::constants::` - Test-specific thresholds
   - `stats::tests::fixtures::` - Reusable test fixtures
   - `stats::tests::validators::` - Test validation utilities
   - `stats::tests::benchmarks::` - Benchmark utilities

2. Move test utilities from various locations:
   - Move from `stats::validation::` to `tests::validators::`
   - Move from `stats::test::/testing::` to `tests::`
   - Move test constants from `stats::constants::`

### Files to Create/Modify:
- Create `include/tests/test_constants.h`
- Create `include/tests/fixtures.h`
- Create `include/tests/validators.h`
- Create `include/tests/benchmarks.h`
- Update all test files

## Phase 4: Namespace Elimination and Consolidation

### Objective:
Remove all namespaces not in the final architecture

### Tasks:

#### 4A: Eliminate `stats::constants::`
- Move arch-specific constants to `stats::arch::`
- Move test constants to `stats::tests::constants::`
- Update all references

#### 4B: Eliminate `stats::memory::`
- Ensure all memory code is in `stats::arch::memory::`
- Remove any duplicate namespace definitions
- Update all references

#### 4C: Eliminate `stats::performance::`
- Move characteristics to `stats::arch::`
- Move benchmarking utilities to `stats::tests::benchmarks::`
- Update all references

#### 4D: Eliminate `stats::safety::`
- Move to `stats::detail::safety::`
- Update all references

#### 4E: Eliminate `stats::validation::`
- Move to `stats::tests::validators::`
- Update all references

#### 4F: Eliminate `stats::tools::`
- Move display utilities to `stats::detail::display::`
- Move string utilities to `stats::detail::strings::`
- Update all references

## Phase 5: Final Verification and Cleanup

### Tasks:
1. Verify no unwanted namespaces remain
2. Update all forward declarations
3. Update all documentation
4. Run full test suite
5. Update namespace_architecture.md to reflect completion

## Implementation Order

### Priority 1 (Foundation):
1. Phase 3D - CPU-specific namespaces (sets up arch:: hierarchy)
2. Phase 3E - Test infrastructure (sets up tests:: hierarchy)

### Priority 2 (Migration):
3. Phase 4A - Eliminate constants:: (distribute to arch:: and tests::)
4. Phase 4D - Move safety:: to detail::
5. Phase 4E - Move validation:: to tests::

### Priority 3 (Cleanup):
6. Phase 4B - Eliminate memory:: duplicates
7. Phase 4C - Eliminate performance::
8. Phase 4F - Eliminate tools::

### Priority 4 (Verification):
9. Phase 5 - Final verification

## Estimated Effort:
- Phase 3D: 2-3 hours (CPU namespaces)
- Phase 3E: 3-4 hours (Test infrastructure)
- Phase 4: 4-5 hours (Namespace elimination)
- Phase 5: 1-2 hours (Verification)

Total: ~10-14 hours of work remaining

## Next Steps:
1. Start with Phase 3D (CPU-specific namespaces)
2. Then Phase 3E (Test infrastructure)
3. Proceed with Phase 4 eliminations in order
4. Final verification
