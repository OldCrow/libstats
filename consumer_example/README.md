# Consumer Example: find_package

Demonstrates consuming libstats from an external project after installation.

## Prerequisites

Build and install libstats:

```bash
cd /path/to/libstats
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
cmake --install build --prefix /tmp/libstats-install
```

## Build this example

```bash
cd consumer_example
cmake -S . -B build -DCMAKE_PREFIX_PATH=/tmp/libstats-install
cmake --build build
./build/consumer_demo
```

## What it tests

- `find_package(libstats REQUIRED)` locates the installed package
- `libstats::libstats_static` target provides headers and static library
- `#include "libstats/libstats.h"` resolves correctly from the install tree
- Gaussian PDF/CDF computation produces correct values
