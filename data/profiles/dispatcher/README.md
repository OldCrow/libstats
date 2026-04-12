# Dispatcher Profiling Data

This directory contains profiling bundles captured by `scripts/capture_dispatcher_profile.sh`.
Each subdirectory is a timestamped bundle from a single architecture run.

## Purpose

The profiling data from all target architectures must be consolidated in one place
to generate the `constexpr` dispatch threshold lookup table (see the plan in issue #14).
Bundles are committed so they can accumulate across machines via normal git workflow.

## Bundle contents

Each bundle contains:

- `metadata.json` — machine, OS, SIMD level, compiler, git state
- `strategy_profile_results.csv` — canonical raw timing data (distribution × operation × batch size × strategy)
- `crossovers.csv` — derived SCALAR→VECTORIZED, VECTORIZED→PARALLEL, PARALLEL→WORK_STEALING crossover points
- `best_strategies.csv` — per-(distribution, operation, batch size) best strategy and speedup vs scalar
- `summary.json` — coverage, strategy win counts, crossover summary
- `logs/` — console output from `system_inspector` and `strategy_profile`

## Target architectures

| Machine | SIMD | Status |
|---|---|---|
| Mac Mini M1 | NEON | ✅ Captured |
| MacBook Pro 9,1 (2012) | AVX | Pending |
| MacBook Pro 14,1 (2017) | AVX2 | Pending |
| Asus TUF A16 (Windows) | AVX-512 | Pending |

## Capturing a new profile

```bash
# Build first, then run the capture script
scripts/capture_dispatcher_profile.sh
# The bundle is saved under build/ and also copied here automatically.
# Commit and push the new bundle.
```
