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
