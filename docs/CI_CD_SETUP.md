# CI/CD Setup for LibStats

## Overview

This document describes the continuous integration and continuous deployment (CI/CD) infrastructure for the LibStats project, introduced in version v0.9.1.5.

## GitHub Actions Workflow

The CI pipeline runs on every push to main/develop branches and on all pull requests. It includes:

### Build Matrix
- **Operating Systems**: Ubuntu, macOS, Windows  
- **Compilers**: GCC 11/12, Clang 14/15, MSVC (Windows), AppleClang (macOS)
- **Build Types**: Debug and Release
- **C++ Standard**: C++20

### Jobs

1. **Build and Test** (`build`)
   - Compiles the library across all platform/compiler combinations
   - Runs unit tests via CTest
   - Executes performance benchmarks (Release builds only)

2. **Code Quality** (`lint`)
   - Checks code formatting with clang-format
   - Performs static analysis with clang-tidy
   - Runs additional checks with cppcheck
   - Currently in warning-only mode to establish baseline

3. **Code Coverage** (`coverage`)
   - Builds with coverage instrumentation (GCC)
   - Runs full test suite
   - Generates LCOV reports
   - Uploads to Codecov for tracking

## Local Development Tools

### Linting Script (`scripts/lint.sh`)
Quick pre-commit checks for developers:
```bash
./scripts/lint.sh
```

Checks:
- Code formatting compliance
- Static analysis on core files
- Common issues (tabs, trailing whitespace)

### Formatting Script (`scripts/format.sh`)
Auto-format all source files:
```bash
./scripts/format.sh
```

Uses the project's `.clang-format` configuration to ensure consistent style.

## Configuration Files

### `.clang-format`
Based on Google style with modifications:
- 4-space indentation
- 100 character line limit
- Automatic header sorting

### `.clang-tidy`
Comprehensive static analysis checks:
- bugprone-* (bug detection)
- concurrency-* (thread safety)
- cppcoreguidelines-* (C++ Core Guidelines)
- modernize-* (modern C++ patterns)
- performance-* (performance issues)
- portability-* (cross-platform compatibility)
- readability-* (code clarity)

Excludes system headers to focus on project code.

## Future Enhancements

### v0.9.1.5 (Current)
- ✅ Basic CI infrastructure
- ✅ Multi-platform builds
- ✅ Code quality tools
- ✅ Coverage reporting

### Future Versions
- [ ] Enforce formatting in CI (fail on violations)
- [ ] Enforce clang-tidy checks (gradual rollout)
- [ ] Add sanitizer builds (ASAN, UBSAN, TSAN)
- [ ] Performance regression testing
- [ ] Automated benchmarking reports
- [ ] Release artifact generation
- [ ] Documentation generation and publishing

## Adding New Checks

To add new static analysis checks:
1. Update `.clang-tidy` with new check categories
2. Run locally first: `./scripts/lint.sh`
3. Fix any issues before committing
4. Update CI to enforce if appropriate

## Troubleshooting

### Clang-tidy False Positives
If clang-tidy reports issues in system headers:
- Ensure `--system-headers=false` is used
- Update `--header-filter` regex if needed
- Add specific suppressions to `.clang-tidy`

### Platform-Specific Issues
- Windows: MSVC may have different warning sets
- macOS: AppleClang versions differ from LLVM Clang
- Linux: Ensure correct compiler versions installed

### Local Tool Versions
Install matching versions locally:
```bash
# Ubuntu/Debian
sudo apt-get install clang-format-15 clang-tidy-15

# macOS
brew install llvm@15
```

## Maintenance

The CI/CD setup should be reviewed and updated:
- When adding new compiler requirements
- Before major releases
- When deprecating platform support
- When updating coding standards
