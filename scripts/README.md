# Scripts

Quick reference for the scripts in `scripts/`.

## Ongoing maintenance

These are part of the normal development workflow and are expected to be used regularly.

- `build.sh` — standard build invocation wrapper
- `format.sh` — run clang-format across the codebase
- `lint.sh` — run clang-tidy linting passes
- `ci-local.sh` — run the CI suite locally before pushing
- `test-cross-compiler.sh` — exercise libstats warning/build profiles on the current host compiler

  **Note:** On a Clang host, the `GCCStrict`/`GCCWarn`/`MSVCStrict`/`MSVCWarn` build types
  validate the repository's *emulated* warning profiles, not real GCC or MSVC front ends.
  Real compiler validation still requires native GCC or MSVC runs.

- `check-pragma-once.sh` — verify all headers use `#pragma once`
- `check-copyright.sh` — verify copyright headers are present
- `check-no-debug.sh` — detect leftover debug output (`std::cout`-based DEBUG prints, `printf.*DEBUG`, `#define DEBUG`)
- `validate-includes.sh` — check for common include hygiene issues
- `capture_dispatcher_profile.sh` / `capture_dispatcher_profile.ps1` — run
  `tools/strategy_profile` with the current build and bundle the CSV output
  under `build/profiles/dispatcher/`, then copy it to
  `data/profiles/dispatcher/<timestamp>_<platform>_<branch>_sha-<sha>/`.
  Re-run after any change to the SIMD tier or distribution batch path that might
  shift a crossover point.
- `summarize_dispatcher_profile.py` — reads a harness bundle and produces
  `crossovers.csv`, `best_strategies.csv`, and `summary.json`. Companion to
  `capture_dispatcher_profile.sh`. NOTE: threshold derivation now reads
  profiles with the SUSTAINED-crossover rule (PROFILING_METHOD.md 2026-09-04
  amendment; #146 folds it into `threshold_validator`) — the v2.4.0 bundles
  were captured directly via `strategy_profile -o` and analyzed with the
  `analyze_crossovers.py` copy inside each bundle.
- `PROFILING_METHOD.md` — the binding threshold-measurement method doc.
- `gen_neon_erf_table.py` — regenerates `src/neon_erf_data.inc`, the
  1537-entry precomputed erf table used by `vector_erf_neon`. Re-run if the
  NEON erf approximation accuracy target or grid spacing is changed.
- `gen_*` generators (13 more: `gen_neon_exp_table.py`,
  `gen_neon_log_cleanroom_table.py`, `gen_neon_log_table.py`,
  `gen_neon_trig_cleanroom_table.py`, `gen_trig_cleanroom_table.py`,
  `gen_avx512_exp_table.py`, and the `gen_*_ulp_vectors.py` /
  `gen_*_cdf_vectors.py` reference-vector generators) — each regenerates a
  checked-in table or test-vector `.inc`; regenerate only when the method
  or point selection changes, and re-run the matching gates after.

## Setup utilities

Run once per machine or development environment. Not part of the regular build cycle.

- `setup-pre-commit.sh` — install pre-commit hooks
- `setup_env.sh` — configure the local development environment
- `verify-setup.sh` — sanity-check that the development environment is correctly configured

## Historical or ad-hoc

Scripts that supported specific completed refactoring work have been removed
from the repository (their record lives in git history); nothing currently
occupies this category.

## Guidance

- Prefer `ci-local.sh` for a full pre-push check.
- Use `format.sh` and `lint.sh` individually during development.
- Do not treat every file in `scripts/` as part of the primary supported
  workflow; check a script's entry above before relying on it.
