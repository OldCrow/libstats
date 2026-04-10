# Consumer Example: FetchContent

Demonstrates consuming libstats via CMake's `FetchContent` module — no separate install step needed.

## Build this example

```bash
cd consumer_example_fetchcontent
cmake -S . -B build
cmake --build build --parallel
./build/consumer_demo
```

The first configure will clone the libstats repository. Subsequent builds use the cached source.

## Using a local checkout

For development, point FetchContent at a local directory instead of GitHub:

```cmake
FetchContent_Declare(libstats SOURCE_DIR /path/to/libstats)
```

## What it tests

- `FetchContent_MakeAvailable(libstats)` builds libstats as a subdirectory
- `libstats_static` target is available without `find_package`
- `#include "libstats/libstats.h"` resolves correctly from the build tree
- Gaussian PDF/CDF computation produces correct values
