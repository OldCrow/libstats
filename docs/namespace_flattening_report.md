# Namespace Flattening Analysis Report

## Safe to Flatten Without Collisions

### 1. Statistical Constants in `stats::detail::`
These constants already have unique prefixes and can be moved directly to `stats::detail::`:

- **normal::** → Move all Z_* constants directly to detail::
  - Z_90, Z_95, Z_99, Z_999, Z_95_ONE_TAIL, Z_99_ONE_TAIL

- **t_distribution::** → Move all T_* constants directly to detail::
  - T_95_DF_*, T_99_DF_* (all have unique names)

- **chi_square::** → Move all CHI2_* constants directly to detail::
  - CHI2_95_DF_*, CHI2_99_DF_* (all have unique names)

- **f_distribution::** → Move all F_* constants directly to detail::
  - F_95_DF_*, F_99_DF_* (all have unique names)

### 2. Method Constants in `stats::detail::`
Can be flattened with minimal prefixing:

- **bayesian::** → Prefix with BAYESIAN_
- **priors::** → Prefix with PRIOR_
- **bootstrap::** → Prefix with BOOTSTRAP_
- **cross_validation::** → Prefix with CV_

### 3. Goodness-of-fit Constants in `stats::detail::`
- **kolmogorov_smirnov::** → Prefix with KS_
- **anderson_darling::** → Prefix with AD_
- **shapiro_wilk::** → Prefix with SW_

### 4. SIMD Sub-namespaces in `stats::arch::simd::`
These can be flattened as most already have unique names:

- **registers::** → Move directly (all end with _DOUBLES)
  - AVX512_DOUBLES, AVX_DOUBLES, SSE_DOUBLES, NEON_DOUBLES, SCALAR_DOUBLES

- **unroll::** → Move directly (all end with _UNROLL)
  - AVX512_UNROLL, AVX_UNROLL, SSE_UNROLL, NEON_UNROLL, SCALAR_UNROLL

- **alignment::** → Keep as-is or add ALIGN_ prefix if needed
  - AVX512_ALIGNMENT, AVX_ALIGNMENT, SSE_ALIGNMENT, NEON_ALIGNMENT

- **cpu::** → Add CPU_ prefix
  - MAX_BACKOFF_NANOSECONDS → CPU_MAX_BACKOFF_NANOSECONDS
  - DEFAULT_CACHE_LINE_SIZE → CPU_DEFAULT_CACHE_LINE_SIZE

- **optimization::** → Add OPT_ prefix
  - MEDIUM_DATASET_MIN_SIZE → OPT_MEDIUM_DATASET_MIN_SIZE
  - ALIGNMENT_BENEFIT_THRESHOLD → OPT_ALIGNMENT_BENEFIT_THRESHOLD

### 5. Utility Namespaces
- **WorkStealingUtils::** → Move to stats::arch:: or stats::detail::

## Cannot Flatten Without Major Refactoring

### 1. Architecture-specific parallel constants (`stats::arch::parallel::`)
These have naming collisions between architectures:

- **sse::, avx::, avx2::, avx512::, neon::**
  - All define: MIN_ELEMENTS_FOR_PARALLEL, DEFAULT_GRAIN_SIZE, MAX_GRAIN_SIZE, etc.
  - Would need architecture prefixes: SSE_MIN_ELEMENTS_FOR_PARALLEL, AVX_MIN_ELEMENTS_FOR_PARALLEL, etc.
  - This would make the code significantly more verbose

### 2. Matrix constants in `stats::arch::simd::matrix::`
- Some collision with MAX_BLOCK_SIZE (appears in both matrix:: and simd::)
- Would need MATRIX_ prefix

## Recommended Approach

### Phase 1: Easy Wins (No Collisions)
1. Flatten all statistical constants in detail:: (they already have prefixes)
2. Flatten register and unroll constants in arch::simd:: (unique suffixes)
3. Move WorkStealingUtils to appropriate parent namespace

### Phase 2: Minimal Prefixing
1. Flatten method constants with short prefixes (BAYESIAN_, BOOTSTRAP_, etc.)
2. Flatten goodness-of-fit constants with standard abbreviations (KS_, AD_, SW_)
3. Flatten cpu:: and optimization:: with prefixes

### Phase 3: Keep As-Is (For Now)
1. Keep architecture-specific parallel:: sub-namespaces (sse::, avx::, etc.)
   - These provide important organization and avoid verbose prefixing
   - They're conceptually different tuning parameters for different architectures
2. Keep matrix:: sub-namespace if it has many constants

## Summary
- **Can flatten safely:** ~70% of sub-namespaces
- **Need prefixing:** ~20% of sub-namespaces
- **Should keep:** ~10% (architecture-specific tuning parameters)

This approach balances the goal of namespace consolidation with code readability and maintainability.

## Current Namespace Structure

### Compliant with Phase 2:
- `stats::`
- `stats::detail::` (most implementations consolidated here)
- `stats::test::`
- `stats::arch::`
- `stats::arch::simd::`

### Remaining Sub-namespaces:
```
stats::detail::
├── normal::
├── t_distribution::
├── chi_square::
├── f_distribution::
├── bayesian::
├── priors::
├── bootstrap::
├── cross_validation::
├── tuning::
├── kolmogorov_smirnov::
├── anderson_darling::
└── shapiro_wilk::

stats::arch::simd::
├── alignment::
├── matrix::
├── registers::
├── unroll::
├── cpu::
└── optimization::

stats::arch::parallel::
├── sse::
├── avx::
│   └── legacy_intel::
│       └── distributions::
├── avx2::
├── avx512::
├── neon::
├── fallback::
├── batch_sizes::
├── detail::
└── tuning::
```

## Build Status
Despite incomplete flattening, the project builds successfully:
- ✅ Core library (static and shared)
- ✅ All test executables
- ✅ All tool executables
- ✅ All example programs
- ✅ 100% build completion
