## [1.0.0](https://github.com/OldCrow/libstats/compare/v0.10.0...v1.0.0) (2025-09-01)

### ⚠ BREAKING CHANGES

* **optimization:** Test headers relocated from include/tests/ to tests/include/

## Summary
Major header optimization phase completed, improving compilation time and code organization
across the entire libstats project.

## Key Improvements
- Removed unused C++20 headers (<concepts>, <ranges>, <version>) reducing compilation overhead
- Created forward declaration headers for frequently-used types (cpu_detection_fwd.h, platform_constants_fwd.h, simd_policy_fwd.h)
- Introduced common headers to consolidate includes (simd_implementation_common.h, test_common.h)
- Relocated test infrastructure from include/tests/ to tests/include/ for better separation
- Fixed distribution name mismatch in discrete_enhanced test

## Performance Impact
- Clean build time: ~2 minutes (maintained)
- Test pass rate: 89% (34/38 tests passing)
- Reduced header parsing overhead
- Better incremental build performance

## Migration Guide
For developers working on tests:
- Update test includes from 'include/tests/' to '../include/' or 'include/' paths
- New test_common.h header available for common test utilities
- CMakeLists.txt updated to include tests/include/ in test target paths

Closes header optimization work from v0.11.0-header-optimization branch
* Main namespace changed from libstats to stats

- Changed primary namespace from 'libstats' to 'stats' across entire codebase
- Updated ~147 files (headers, source, tests, examples, tools)
- Added backward compatibility alias: namespace libstats = stats
- Resolved LOG_PROBABILITY_EPSILON collision in precision_constants.h
- Version bumped to 0.11.0

This is phase 1 of namespace consolidation to reduce ~160 namespaces to 3-5.
Users can migrate gradually thanks to the compatibility alias.

Test results: 34/39 tests passing (87%)
Failed tests are performance-related (SIMD speedup), not functionality issues.

### ✨ Features

* Add comprehensive AVX-512 testing infrastructure ([1fcb8d2](https://github.com/OldCrow/libstats/commit/1fcb8d28bae3df2f92941c3bd76ba126d317f98d))
* apply IWYU optimizations to distribution_base ([ccca5a7](https://github.com/OldCrow/libstats/commit/ccca5a77813e2da0489f880a16ce03c3acc304f3))
* apply IWYU optimizations to tools, examples, and tests directories ([50e15d4](https://github.com/OldCrow/libstats/commit/50e15d473695abd78fdb2420c242a3f0aa38c390))
* complete Phase 1 preparation for magic number elimination ([363c9ae](https://github.com/OldCrow/libstats/commit/363c9ae2735fea3c238b32535a8c452d44a2054a))
* Complete Phase 2 namespace consolidation and prepare Phase 3 architecture ([0d2b8ac](https://github.com/OldCrow/libstats/commit/0d2b8ac019153b578e2e4d4173d33a29e5b4032c))
* Complete PIMPL refactoring and namespace modernization ([4355ef4](https://github.com/OldCrow/libstats/commit/4355ef42e86b9d4571e143112ba4853eae56fdf7))
* **optimization:** complete header dependency optimization and bump to v0.12.0 ([e0280a6](https://github.com/OldCrow/libstats/commit/e0280a698c1c70729b089d98e6932ddfa113a8c3))
* Optimize header dependencies and compilation time ([b3e7948](https://github.com/OldCrow/libstats/commit/b3e79483f4879cef4ba20a1b0478ea9b66af06fc))

### 🐛 Bug Fixes

* Complete namespace fixes for all SIMD implementation files ([2d2ea9e](https://github.com/OldCrow/libstats/commit/2d2ea9e716e3b50e0586798af825d8848a061450))
* Correct AVX-512 constant namespace in simd_dispatch.cpp ([6f98fa0](https://github.com/OldCrow/libstats/commit/6f98fa036cbe375a238dab5457fd65372318dfb6))
* Cross-platform namespace issues for Linux/Windows builds ([7a38aec](https://github.com/OldCrow/libstats/commit/7a38aec2b19a273aca6db8e9e56adf621aa3107c))
* Redesign AVX-512 workflow for compilation-only testing ([fd68939](https://github.com/OldCrow/libstats/commit/fd68939953cfe0e240db91de23e273e01f0dfb44))
* Remove incorrect cpu:: namespace prefix in simd_policy.cpp ([b8d2557](https://github.com/OldCrow/libstats/commit/b8d2557e9a29047d87db191e7f42af28776a5e33))
* resolve arch::safe_transform compilation errors ([c5d36ff](https://github.com/OldCrow/libstats/commit/c5d36ffa91dc70e56b46f9ee8aa1bded46ed89a4))
* Resolve compilation warnings and AVX-512 function naming consistency ([361000b](https://github.com/OldCrow/libstats/commit/361000b6b8c9bf35f23dc6902087c107914e0e5e))
* Resolve CPU feature detection namespace issues and MSVC std::max type errors ([eeb7e25](https://github.com/OldCrow/libstats/commit/eeb7e25c8dfcddd537ba9c29cc579817bcfff477))

### 📚 Documentation

* clarify pre-1.0 versioning strategy and configure semantic-release ([99995a0](https://github.com/OldCrow/libstats/commit/99995a04e0fc8e26536b93e3d256f11d0c6df133))

### ♻️ Refactoring

* Apply IWYU optimizations to Levels 2a-2c ([02c79bc](https://github.com/OldCrow/libstats/commit/02c79bca3a12ff4021c1b7f7a1fa2f7628eccd54))
* Apply IWYU optimizations to Levels 3-4 (Infrastructure + Framework) ([cc4ae46](https://github.com/OldCrow/libstats/commit/cc4ae46cd634f1489715562bb2821946267e42e4))
* migrate from libstats:: to stats:: namespace (v0.11.0) ([92e5952](https://github.com/OldCrow/libstats/commit/92e595279bfd64248475964c64081f3aca2b70e6))

### 🔧 Maintenance

* apply pre-commit hooks and fix formatting ([d4af2a4](https://github.com/OldCrow/libstats/commit/d4af2a4d1166dc7c622e1655fa5d571985f653d8))
* **release:** 0.11.0 [skip ci] ([b0aa307](https://github.com/OldCrow/libstats/commit/b0aa30738083d7b91a9f317bffa072594dff3d39))
* **release:** 0.11.1 [skip ci] ([c0d98d8](https://github.com/OldCrow/libstats/commit/c0d98d8c66778aacab9e608111940b7c48f50e5e))
* **release:** 0.12.0 [skip ci] ([13c484e](https://github.com/OldCrow/libstats/commit/13c484ec2331c3671bf606bcfc54cffa0348eb6e))

## [2.0.0](https://github.com/OldCrow/libstats/compare/v1.1.0...v2.0.0) (2025-09-01)

### ⚠ BREAKING CHANGES

* **optimization:** Test headers relocated from include/tests/ to tests/include/

## Summary
Major header optimization phase completed, improving compilation time and code organization
across the entire libstats project.

## Key Improvements
- Removed unused C++20 headers (<concepts>, <ranges>, <version>) reducing compilation overhead
- Created forward declaration headers for frequently-used types (cpu_detection_fwd.h, platform_constants_fwd.h, simd_policy_fwd.h)
- Introduced common headers to consolidate includes (simd_implementation_common.h, test_common.h)
- Relocated test infrastructure from include/tests/ to tests/include/ for better separation
- Fixed distribution name mismatch in discrete_enhanced test

## Performance Impact
- Clean build time: ~2 minutes (maintained)
- Test pass rate: 89% (34/38 tests passing)
- Reduced header parsing overhead
- Better incremental build performance

## Migration Guide
For developers working on tests:
- Update test includes from 'include/tests/' to '../include/' or 'include/' paths
- New test_common.h header available for common test utilities
- CMakeLists.txt updated to include tests/include/ in test target paths

Closes header optimization work from v0.11.0-header-optimization branch

### ✨ Features

* **optimization:** complete header dependency optimization and bump to v0.12.0 ([0bb7988](https://github.com/OldCrow/libstats/commit/0bb79881c25b8dfcf721d4221fd778f7a3e1c03b))

## [1.1.0](https://github.com/OldCrow/libstats/compare/v1.0.0...v1.1.0) (2025-09-01)

### ✨ Features

* apply IWYU optimizations to distribution_base ([92bf0e0](https://github.com/OldCrow/libstats/commit/92bf0e03500df06eab890266f55f843d56fc6d6a))
* apply IWYU optimizations to tools, examples, and tests directories ([bf89277](https://github.com/OldCrow/libstats/commit/bf89277292fccd8f9a227f75cb94a7de0db31e2f))
* complete Phase 1 preparation for magic number elimination ([ce03f2c](https://github.com/OldCrow/libstats/commit/ce03f2c5c3324811a3eae80844399f4c51325f2a))
* Optimize header dependencies and compilation time ([27b4c88](https://github.com/OldCrow/libstats/commit/27b4c88410226d3fdfc8a5d9840c9a0910a0f75a))

### 🐛 Bug Fixes

* resolve arch::safe_transform compilation errors ([9174cc2](https://github.com/OldCrow/libstats/commit/9174cc2867b15fe17bf88fdde38dbb093ac7b11c))

### ♻️ Refactoring

* Apply IWYU optimizations to Levels 2a-2c ([7ca15c8](https://github.com/OldCrow/libstats/commit/7ca15c8a146c1a921e00aaf5455d592bdd0659b0))
* Apply IWYU optimizations to Levels 3-4 (Infrastructure + Framework) ([55b5b6f](https://github.com/OldCrow/libstats/commit/55b5b6f57d7430692c4be6198f09e5f2761375af))

## [1.0.0](https://github.com/OldCrow/libstats/compare/v0.10.0...v1.0.0) (2025-08-29)

### ⚠ BREAKING CHANGES

* Main namespace changed from libstats to stats

- Changed primary namespace from 'libstats' to 'stats' across entire codebase
- Updated ~147 files (headers, source, tests, examples, tools)
- Added backward compatibility alias: namespace libstats = stats
- Resolved LOG_PROBABILITY_EPSILON collision in precision_constants.h
- Version bumped to 0.11.0

This is phase 1 of namespace consolidation to reduce ~160 namespaces to 3-5.
Users can migrate gradually thanks to the compatibility alias.

Test results: 34/39 tests passing (87%)
Failed tests are performance-related (SIMD speedup), not functionality issues.

### ✨ Features

* Add comprehensive AVX-512 testing infrastructure ([1fcb8d2](https://github.com/OldCrow/libstats/commit/1fcb8d28bae3df2f92941c3bd76ba126d317f98d))
* Complete Phase 2 namespace consolidation and prepare Phase 3 architecture ([0d2b8ac](https://github.com/OldCrow/libstats/commit/0d2b8ac019153b578e2e4d4173d33a29e5b4032c))
* Complete PIMPL refactoring and namespace modernization ([4355ef4](https://github.com/OldCrow/libstats/commit/4355ef42e86b9d4571e143112ba4853eae56fdf7))

### 🐛 Bug Fixes

* Complete namespace fixes for all SIMD implementation files ([2d2ea9e](https://github.com/OldCrow/libstats/commit/2d2ea9e716e3b50e0586798af825d8848a061450))
* Correct AVX-512 constant namespace in simd_dispatch.cpp ([6f98fa0](https://github.com/OldCrow/libstats/commit/6f98fa036cbe375a238dab5457fd65372318dfb6))
* Cross-platform namespace issues for Linux/Windows builds ([7a38aec](https://github.com/OldCrow/libstats/commit/7a38aec2b19a273aca6db8e9e56adf621aa3107c))
* Redesign AVX-512 workflow for compilation-only testing ([fd68939](https://github.com/OldCrow/libstats/commit/fd68939953cfe0e240db91de23e273e01f0dfb44))
* Remove incorrect cpu:: namespace prefix in simd_policy.cpp ([b8d2557](https://github.com/OldCrow/libstats/commit/b8d2557e9a29047d87db191e7f42af28776a5e33))
* Resolve compilation warnings and AVX-512 function naming consistency ([361000b](https://github.com/OldCrow/libstats/commit/361000b6b8c9bf35f23dc6902087c107914e0e5e))
* Resolve CPU feature detection namespace issues and MSVC std::max type errors ([eeb7e25](https://github.com/OldCrow/libstats/commit/eeb7e25c8dfcddd537ba9c29cc579817bcfff477))

### ♻️ Refactoring

* migrate from libstats:: to stats:: namespace (v0.11.0) ([92e5952](https://github.com/OldCrow/libstats/commit/92e595279bfd64248475964c64081f3aca2b70e6))

### 🔧 Maintenance

* apply pre-commit hooks and fix formatting ([d4af2a4](https://github.com/OldCrow/libstats/commit/d4af2a4d1166dc7c622e1655fa5d571985f653d8))

## [0.10.0](https://github.com/OldCrow/libstats/compare/v0.9.1...v0.10.0) (2025-08-20)

### ✨ Features

* Add CI debugging tools and -Wdeprecated-volatile warning ([0e9cb70](https://github.com/OldCrow/libstats/commit/0e9cb70beec18dd6a1056785d9b369c1b0df9cb9))
* Add CI/CD infrastructure and code quality tools (v0.9.1.5) ([a3667f9](https://github.com/OldCrow/libstats/commit/a3667f9dc7cd20ac8dae4b1cbffd3fda7a25865b))
* **ci:** complete v0.9.1.5 CI/CD and development infrastructure setup ([7c0eda1](https://github.com/OldCrow/libstats/commit/7c0eda1406950ec7db9c212eef87859cc80bf065))

### 🐛 Bug Fixes

* Add --ignore-errors unused flag to lcov to handle missing examples directory ([e222898](https://github.com/OldCrow/libstats/commit/e222898f04bf3d2f17da0e7850d1acebbde42649))
* add clang-format directives to preserve Windows header order ([bd7a481](https://github.com/OldCrow/libstats/commit/bd7a4810d39f4bfd09c5d4e3f4edffb6cbda4f3d))
* Add missing includes to system_capabilities.cpp ([29deab5](https://github.com/OldCrow/libstats/commit/29deab517a1964e14848f12a6a2b728f6c883f21))
* **ci:** add explicit x64 arch to MSVC dev environment setup ([923fe34](https://github.com/OldCrow/libstats/commit/923fe3462b2a99f88340329c0444dd650e7196e8))
* **ci:** specify x64 architecture for Windows MSVC builds ([b935da4](https://github.com/OldCrow/libstats/commit/b935da463f6b1fdcca84c8124bca55f9787b55ca))
* Ensure L3 cache threshold >= L2 on Apple platforms ([330987b](https://github.com/OldCrow/libstats/commit/330987b57f9279f75b61c759bcdea114be9a06d6))
* Fix code coverage job compiler configuration ([2d0eb04](https://github.com/OldCrow/libstats/commit/2d0eb04bf569a3d94b66cdf45c7a8bd9ca94e203))
* Fix std::min type mismatch errors for Linux builds ([75dd3b0](https://github.com/OldCrow/libstats/commit/75dd3b0db81379c0b735677cb097c26e33b32c09))
* Fix Windows CI test failures ([7b22d3a](https://github.com/OldCrow/libstats/commit/7b22d3acf19d577c0969d56c8e7fd734692253d5))
* Improve Windows DLL handling and test robustness ([9adb331](https://github.com/OldCrow/libstats/commit/9adb33137b14d8af0d965f662df54a2d3cd65ecf))
* Replace INT_MIN/INT_MAX with std::numeric_limits for Linux compatibility ([d455780](https://github.com/OldCrow/libstats/commit/d455780142c9429b7f2a1b73e395dffbfcf7fe7c))
* Resolve Linux build errors in CI ([e4af3d1](https://github.com/OldCrow/libstats/commit/e4af3d12caab7a836e61a55c4ac2844702d22299))
* Resolve ODR violation - make pthread_worker inline ([0bc779d](https://github.com/OldCrow/libstats/commit/0bc779d3fceac86552497eab16624ce406522d62))
* Simplify SIMD compilation strategy using traditional CMake approach ([efa656f](https://github.com/OldCrow/libstats/commit/efa656f2e4f5865e6805438abdfe587ad02ebec8))
* Specify bash shell for Windows test step ([6100142](https://github.com/OldCrow/libstats/commit/610014295c3e8b92f8566acc67c24c5d0f152b1b))
* Update README badges and fix clone URL ([29a5b2c](https://github.com/OldCrow/libstats/commit/29a5b2c232b49a206157e087c29ca94ea57ba978))

### 📚 Documentation

* Add header reorganization plan for v0.9.2 ([ab6df66](https://github.com/OldCrow/libstats/commit/ab6df66b88863e3cc438d07b89579a5b6ac37e59))
* Complete v0.9.1.5 milestone - CI/CD infrastructure setup ([fc4cd70](https://github.com/OldCrow/libstats/commit/fc4cd704fd03ddfe59bfaa56741e111a8fac94ff))
* Restructure v0.9.0→v1.0.0 roadmap with comprehensive task lists ([6aea97c](https://github.com/OldCrow/libstats/commit/6aea97c24fb991445edd430c7d2d1d6023ff5a69))
