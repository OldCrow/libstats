# Header Tools Guide

This guide covers tools and checks that help maintain libstats headers.

## Goals

Header checks should verify:

- headers compile independently where intended
- public headers do not depend on deleted v1.x headers
- umbrella includes expose stable public APIs
- private helpers remain in `stats::detail`

## Important commands

```bash
cmake -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure -LE "timing|benchmark"
```

## Header audit grep checks

```bash
# Deleted v1.x headers should not appear.
# Check the old statistical-utilities stub header and the old distribution-memory header
# by exact filename during release audits.

# Removed v1.x batch APIs should not appear in headers.
# Check strategy-suffix methods and vector-returning batch helpers by their
# exact removed method names during release audits.

# Analysis namespace usage
rg "stats::analysis" include src tests docs
```

## Adding a new public header

When adding a public header:

1. Place it in the appropriate directory.
2. Use `#pragma once`.
3. Include only what the header needs.
4. Use `libstats/...` include paths for installed-header compatibility.
5. Add tests that compile through the installed-style include shim.
6. Update documentation if it changes public API.

## Analysis headers

Generic analysis headers may be added to `stats/analysis/analysis.h`.

Distribution-specific analysis headers must be included explicitly by users. Do not add them to the generic umbrella unless the design changes deliberately.

## Include shim

The build creates:

```text
build/include_shim/libstats/
```

Use this when testing installed-style include paths.
