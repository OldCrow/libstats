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
- `check-no-debug.sh` — detect leftover debug output (`std::cerr`, `printf`, etc.)
- `validate-includes.sh` — check for common include hygiene issues
- `capture_dispatcher_profile.sh` — run `tools/strategy_profile` with the current
  build and bundle the CSV output into `data/profiles/dispatcher/<timestamp>/`.
  Re-run after any change to the SIMD tier or distribution batch path that might
  shift a crossover point. The bundle is the canonical input for threshold tuning.
- `summarize_dispatcher_profile.py` — reads a profile bundle and produces
  `crossovers.csv`, `best_strategies.csv`, and `summary.json`. Companion to
  `capture_dispatcher_profile.sh`.
- `gen_neon_erf_table.py` — regenerates `src/neon_erf_data.inc`, the
  769-entry precomputed erf table used by `vector_erf_neon`. Re-run if the
  NEON erf approximation accuracy target or grid spacing is changed.

## Setup utilities

Run once per machine or development environment. Not part of the regular build cycle.

- `setup-pre-commit.sh` — install pre-commit hooks
- `setup_env.sh` — configure the local development environment
- `verify-setup.sh` — sanity-check that the development environment is correctly configured

## Historical or ad-hoc

These supported specific refactoring work that is now complete and have been
removed from the repository. They are documented here for reference only.

## Guidance

- Prefer `ci-local.sh` for a full pre-push check.
- Use `format.sh` and `lint.sh` individually during development.
- Do not treat every file in `scripts/` as part of the primary supported workflow;
  the historical scripts above are retained for reference, not active use.
