# Development Tools

Quick reference for the actively useful tools in `tools/`.

## Current Tool Groups

### Runtime and platform inspection
- `system_inspector` — inspect CPU, cache, SIMD, and runtime capability detection
- `cpp20_features_inspector` — inspect compiler and standard-library feature support

### SIMD and performance validation
- `simd_verification` — validate SIMD correctness and measure speedups across distributions
- `parallel_threshold_benchmark` — inspect architecture-aware threshold behavior
- `parallel_batch_fitting_benchmark` — benchmark batch fitting behavior across distributions
- `parallel_correctness_verification` — validate batch correctness under parallel execution

### Dispatch and learning analysis
- `performance_dispatcher_tool` — inspect dispatch choices and strategy behavior
- `learning_analyzer` — analyze adaptive learning and threshold behavior
- `empirical_characteristics_demo` — inspect empirical complexity assumptions used by dispatch logic

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
Some tools exist to support older refactors, narrow investigations, or one-off cleanup work. Keep them only if they still earn their place; otherwise move them to a clearly historical area or remove them.

Examples:
- `demo_phase1_optimization.py`
- `demo_phase2_optimization.py`
- `replace_magic_numbers.py`
- `replace_domain_constants.py`
- `analyze_magic_numbers.py`
- `find_unsafe_constructor_usage.py`

## Guidance

- Prefer the compiled C++ tools for release validation and performance checks.
- Prefer the Python analysis tools for repo-maintenance work.
- Do not treat every file in `tools/` as part of the primary supported workflow; some are archival.
