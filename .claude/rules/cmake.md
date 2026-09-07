---
paths:
  - "CMakeLists.txt"
  - "**/CMakeLists.txt"
  - "**/*.cmake"
  - "CMakePresets.json"
---

## CMake standard

Full rules: [CMake House Style](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md)
in the fleet standards repo; this section is self-sufficient for this repo. libstats deviations:
- Target-first scoping, `LIBSTATS_`-prefixed options, warnings PRIVATE and
  `PROJECT_IS_TOP_LEVEL`-gated: landed (Phase 3B). Threading detection and
  compiler-flag/warning-set logic live in `cmake/Threading.cmake` and
  `cmake/CompilerFlags.cmake`; tests and tools are registered from their own
  `tests/CMakeLists.txt` and `tools/CMakeLists.txt` via `add_subdirectory`.
  Warnings are applied PRIVATE per-target through `libstats_apply_warnings(target)`
  (defined in `cmake/CompilerFlags.cmake`), called on every object library,
  the final static/shared libs, tests, and tools — GTest is exempt (fetched
  sources never receive our warning flags). Optimization/debug-info flags
  for the custom `Dev`/`Strict` build types come from `CMAKE_CXX_FLAGS_DEV`/
  `CMAKE_CXX_FLAGS_STRICT`, set with the guarded-FORCE idiom
  (`if(NOT var) set(... FORCE)`) — a plain unguarded `set(... CACHE STRING)`
  is a silent no-op here because CMake auto-creates these per-config cache
  entries empty for any custom `CMAKE_BUILD_TYPE` at `project()`, before this
  file is ever included.
- **Grandfathered custom build types**: `Dev` (default) and `Strict`
  (the `-Werror` vehicle) — kept per house-style exception; not to be
  copied into other repos.
- **A configure-time fact that a public header branches on goes in the
  generated `libstats_config.h`, never in `target_compile_definitions`.**
  Template `cmake/libstats_config.h.in`, installed beside the hand-written
  headers, the same mechanism as `libstats_version.h`. Fleet rule:
  [CMake House Style §7](https://github.com/OldCrow/standards/blob/main/CMAKE-HOUSE-STYLE.md#7-install-contract-libhmm-libstats-corvus).
  libstats #97 was the rule's second incident: `$<LINK_ONLY:>` stripped the
  macro from the installed export, so every consumer compiled Tier 2 Bessel
  and an ODR violation against the library's own TUs.
- Install contract conforms: GNUInstallDirs, `libstats-targets` export
  (namespace `libstats::`), kebab `libstats-config.cmake`, `SameMajorVersion`.
- Presets (`CMakePresets.json`, schema 6, min CMake 3.25): `dev` → `build/`
  (default workflow), `release` → `build-release/`, `debug` →
  `build-debug/`, `rel-with-debug` → `build-relwithdebinfo/`, `strict` →
  `build-strict/`. **Deviation from the shared vocabulary**: `release` maps
  to `build-release/` rather than `build/`, because `build/` is already
  claimed by the default `dev` workflow here — grandfathered alongside the
  `Dev` build type.
