# Development Tools

Quick reference for the actively useful tools in `tools/`.

## Current Tool Groups

### Runtime and platform inspection
- `system_inspector` — inspect CPU, cache, SIMD, and runtime capability detection
- `cpp20_features_inspector` — inspect compiler and standard-library feature support

### SIMD and performance validation
- `simd_verification` — validate SIMD correctness and measure speedups across distributions
- `strategy_profile` — canonical forced-strategy profiler for dispatcher threshold tuning across distributions, operations, and batch sizes (flags: `-l/--large`, `-o/--output-csv PATH`; there is no `--runs` — that belongs to the capture script)
- `accuracy_sweep` — deterministic bit-exact accuracy sweep over all 27 distributions (CSV consumed by `accuracy_vs_mpmath.py`)
- `threshold_validator` — compares measured crossovers against `dispatch_thresholds.h` (first-crossing heuristic — read with the sustained-crossover rule instead; #146)
- `quantile_accuracy` — quantile accuracy probe
- `toctou_validator` — TOCTOU validation for the cache/dispatch fast paths
- `copy_move_stress` — concurrent copy/move stress over all 27 distributions (replaced `test_copy_move_stress.cpp`)
- `parameter_recovery_benchmark` — MLE parameter-recovery benchmark
- `gather_throughput_probe`, `neon_table_transcendental_probe` — SIMD design probes (#33-era, kept for re-measurement)
- `accuracy_vs_mpmath.py` — the mpmath accuracy oracle (run with a python that has mpmath; regenerates the ACCURACY_CHARACTERIZATION generated blocks)
- `math_cache_benchmark.py` — math-cache micro-benchmark
- `parallel_batch_fitting_benchmark` — benchmark batch fitting behavior across distributions
- `parallel_correctness_verification` — validate batch correctness under parallel execution


### Header-analysis tools
These remain useful for include and compilation-health work:
- `header_dashboard.py`
- `header_insights.py`
- `compilation_benchmark.py`
- `header_analysis.py`
- `static_analysis.py`
- `header_optimization_analysis.py`
- `header_optimization_summary.py`

See `docs/HEADER_TOOLS_GUIDE.md` for the header-analysis workflow.

## Historical or specialized utilities

The following one-time refactoring tools have been **deleted** from the repository
(v2.0.0 cleanup). They are listed here for historical reference only:
`demo_phase1_optimization.py`, `demo_phase2_optimization.py`,
`replace_magic_numbers.py`, `replace_domain_constants.py`,
`analyze_magic_numbers.py`, `find_unsafe_constructor_usage.py`,
`test_include_analysis.cpp`.

The following header-analysis tools are **archival** (the header optimization
work they supported is complete). They remain in the repository but are not
part of the active development workflow:
- `header_optimization_analysis.py`
- `header_optimization_summary.py`
- `header_insights.py`

## Guidance

- Prefer the compiled C++ tools for release validation and performance checks.
- For dispatcher threshold tuning, prefer `strategy_profile` as the canonical raw data source.
- Prefer the Python analysis tools for repo-maintenance work.
- Do not treat every file in `tools/` as part of the primary supported workflow; some are archival.
