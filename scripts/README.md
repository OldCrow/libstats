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

## Setup utilities

Run once per machine or development environment. Not part of the regular build cycle.

- `setup-pre-commit.sh` — install pre-commit hooks
- `setup_env.sh` — configure the local development environment
- `verify-setup.sh` — sanity-check that the development environment is correctly configured

## Historical or ad-hoc

These supported specific phases of development work that is now complete.
Retain them only if there is a concrete reason to run them again.

- `find_magic_numbers.sh` — identified numeric literals for the magic-number elimination
  pass (Phase 5). That work is done; this script is archival.
- `run-iwyu.sh` — ran Include What You Use analysis during the header optimization pass
  (Phase 5). That work is done; this script is archival.
- `check_collisions.py` — checked for symbol and namespace collisions during the
  namespace consolidation pass (Phase 5). That work is done; this script is archival.

## Guidance

- Prefer `ci-local.sh` for a full pre-push check.
- Use `format.sh` and `lint.sh` individually during development.
- Do not treat every file in `scripts/` as part of the primary supported workflow;
  the historical scripts above are retained for reference, not active use.
