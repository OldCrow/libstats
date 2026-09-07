---
paths:
  - "include/**/*.h"
  - "src/**/*.cpp"
  - "tests/**/*.cpp"
  - "tools/**/*.cpp"
  - "examples/**/*.cpp"
---

## Code Standards
- **C++20 Required**: Modern features (concepts, spans, execution policies)
- **Header Guards**: Use `#pragma once` (codebase convention)
- **Naming**: CamelCase classes, snake_case functions/variables
- **Memory Management**: Smart pointers, RAII, no raw pointers
- **Error Handling**: Dual API (Result<T> for factories, exceptions for setters)

## Performance Considerations
- Always rebuild after source changes before running tests
- Use `initialize_performance_systems()` for optimal batch performance
- SIMD kernels impose no alignment requirement on caller data: every load/store of a caller buffer is unaligned (`loadu`/`storeu`); aligned ops are used only on internal `alignas` locals
- Large batch operations (>1000 elements) benefit significantly from parallel execution

## Platform-Specific Conventions
- **macOS**: System AppleClang is the default and only supported v2.x compiler path (Ventura 13+).
- **Build artifacts**: Always in `build/tools/` and `build/tests/`, never `bin/`
- **Threading**: GCD preferred on macOS, TBB/OpenMP on Linux/Windows
