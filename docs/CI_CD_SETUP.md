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

### IWYU Script (`scripts/run-iwyu.sh`)
Analyze header dependencies and includes:
```bash
# Analyze entire project
./scripts/run-iwyu.sh --all

# Analyze source files only
./scripts/run-iwyu.sh --src

# Analyze specific file
./scripts/run-iwyu.sh --file src/gaussian.cpp
```

### Pre-commit Setup (`scripts/setup-pre-commit.sh`)
Install and configure pre-commit hooks:
```bash
./scripts/setup-pre-commit.sh
```

This will install pre-commit and set up automatic checks before each commit.

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

### `.iwyu_mappings.imp`
Include What You Use mappings:
- Maps internal headers to public interfaces
- Defines symbol-to-header relationships
- Handles platform-specific SIMD headers

### `.pre-commit-config.yaml`
Pre-commit hooks configuration:
- Automatic code formatting (clang-format)
- File integrity checks (trailing whitespace, EOL)
- Custom libstats checks (pragma once, debug code)
- CMake and shell script validation
- Secret detection to prevent credential leaks

### Supporting Configuration Files
- `.cmake-format.yaml` - CMake formatting rules
- `.markdownlint.yaml` - Markdown linting configuration

## Future Enhancements

### v0.9.1.5 (Current)
- ✅ Basic CI infrastructure
- ✅ Multi-platform builds
- ✅ Code quality tools
- ✅ Coverage reporting

### Completed (v0.9.1.5 Final)
- ✅ Include What You Use (IWYU) configuration
- ✅ Pre-commit hooks for local validation

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

## Pre-commit Hooks

### Installation
```bash
# Automatic setup (recommended)
./scripts/setup-pre-commit.sh

# Manual installation
pip install pre-commit
pre-commit install
```

### Usage
```bash
# Run on all files
pre-commit run --all-files

# Run on staged files (automatic on commit)
git commit -m "Your message"

# Skip hooks temporarily (not recommended)
git commit --no-verify

# Update hook versions
pre-commit autoupdate
```

### Custom Hooks
The project includes several custom validation scripts:
- `check-pragma-once.sh` - Ensures headers use #pragma once
- `check-copyright.sh` - Validates copyright headers
- `check-no-debug.sh` - Detects debug code patterns
- `validate-includes.sh` - Checks include order

## Include What You Use (IWYU)

### Installation
```bash
# macOS
brew install include-what-you-use

# Ubuntu/Debian
apt-get install iwyu
```

### Usage
```bash
# Generate compilation database if needed
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Run analysis
./scripts/run-iwyu.sh --all

# Apply suggested fixes (review carefully)
./scripts/run-iwyu.sh --apply
```

### Configuration
The `.iwyu_mappings.imp` file defines:
- Header mappings for the libstats API
- Standard library implementation details
- Platform-specific header relationships

## Maintenance

The CI/CD setup should be reviewed and updated:
- When adding new compiler requirements
- Before major releases
- When deprecating platform support
- When updating coding standards
- After adding new dependencies or tools
