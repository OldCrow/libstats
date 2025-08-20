# Contributing to libstats

Thank you for your interest in contributing to libstats! We welcome contributions from the community and are pleased to have you join us in making this project better.

## 🚀 Quick Start for Contributors

### Prerequisites

- **C++20 compatible compiler**: GCC 10+, Clang 10+, MSVC 2019 16.11+, or LLVM 20+ (recommended)
- **CMake**: 3.15 or later
- **Git**: For version control
- **Optional**: GTest for running tests, Intel TBB for enhanced parallel support

### Setting Up Development Environment

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/yourusername/libstats.git
   cd libstats
   ```

2. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make -j$(nproc)
   ```

3. **Run tests to ensure everything works**:
   ```bash
   ctest --verbose
   ```

## 📋 How to Contribute

### Types of Contributions We Welcome

- 🐛 **Bug reports and fixes**
- ✨ **New features and enhancements**
- 📚 **Documentation improvements**
- 🧪 **Additional tests and benchmarks**
- 🎯 **Performance optimizations**
- 🔧 **Build system and tooling improvements**

### Contribution Process

1. **Check existing issues** to see if your contribution is already being discussed
2. **Create an issue** to discuss new features or major changes before implementing
3. **Fork the repository** and create a feature branch
4. **Implement your changes** following our coding standards
5. **Add tests** for new functionality
6. **Update documentation** as needed
7. **Submit a pull request** with a clear description

## 🎨 Coding Standards

### C++ Style Guidelines

We follow modern C++20 best practices with specific emphasis on:

#### **Code Style**
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `GaussianDistribution`)
  - Functions/methods: `camelCase` (e.g., `getProbability`)
  - Variables: `snake_case` (e.g., `standard_deviation_`)
  - Constants: `UPPER_CASE` (e.g., `MAX_ITERATIONS`)
  - Private members: `snake_case_` with trailing underscore

#### **Modern C++ Features**
- **Use C++20 features**: concepts, ranges, span, likely/unlikely attributes
- **RAII principles**: Always use smart pointers and stack-based resource management
- **Exception safety**: Provide strong exception guarantee where possible
- **Thread safety**: All public APIs should be thread-safe
- **const correctness**: Use const wherever appropriate

#### **Performance Guidelines**
- **Cache efficiency**: Consider memory layout and access patterns
- **SIMD optimization**: Use our SIMD infrastructure for bulk operations
- **Parallel processing**: Leverage thread pools for large computations
- **Minimal allocations**: Prefer stack allocation and reuse containers

### Documentation Requirements

- **Header documentation**: All public APIs must have comprehensive Doxygen comments
- **Implementation comments**: Complex algorithms should be well-explained
- **Usage examples**: Include code examples for new features
- **Performance notes**: Document performance characteristics and trade-offs

### Testing Requirements

All contributions must include appropriate tests:

- **Unit tests**: Test individual components in isolation
- **Integration tests**: Test component interactions
- **Performance tests**: Benchmark performance-critical code
- **Cross-platform tests**: Ensure compatibility across supported platforms

## 🏗️ Development Guidelines

### Project Structure

```
libstats/
├── include/           # Public headers
├── src/              # Implementation files
├── tests/            # Unit and integration tests
├── examples/         # Usage examples
├── docs/             # Project documentation
└── cmake/            # CMake modules and scripts
```

### Adding New Distributions

When contributing new probability distributions:

1. **Inherit from `DistributionBase`**
2. **Implement all pure virtual methods**
3. **Follow the Gaussian implementation pattern**
4. **Include comprehensive statistical methods**
5. **Add parameter validation and safe factory methods**
6. **Implement SIMD-optimized batch operations**
7. **Provide thorough test coverage**

### SIMD Development

For SIMD-related contributions:

- **Use runtime dispatch**: Don't assume specific instruction sets
- **Provide scalar fallbacks**: Always include non-SIMD alternatives
- **Test on multiple architectures**: Ensure cross-platform compatibility
- **Benchmark performance**: Document performance improvements

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
ctest --verbose

# Run specific test categories
ctest -R "test_gaussian"        # Gaussian distribution tests
ctest -R "test_simd"            # SIMD functionality tests
ctest -R "test_thread"          # Thread safety tests
```

### Writing Tests

- **Use descriptive test names**: `test_gaussian_pdf_standard_normal`
- **Test edge cases**: NaN, infinity, extreme values
- **Test error conditions**: Invalid parameters, out-of-range inputs
- **Benchmark performance**: Use our benchmark infrastructure for performance tests
- **Cross-platform compatibility**: Ensure tests pass on all supported platforms

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Clear description** of the problem
2. **Minimal reproducing example**
3. **Expected vs. actual behavior**
4. **Environment information**:
   - Operating system and version
   - Compiler and version
   - CMake version
   - Any relevant hardware info (CPU, SIMD support)
5. **Build configuration** (Debug/Release, compiler flags)

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., macOS 12.0, Ubuntu 20.04]
- Compiler: [e.g., Clang 15, GCC 11]
- CMake: [e.g., 3.20]
- CPU: [e.g., Intel i7, Apple M1]

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

For new features:

1. **Search existing issues** to avoid duplicates
2. **Describe the use case** and motivation
3. **Provide implementation suggestions** if you have them
4. **Consider backwards compatibility**
5. **Discuss performance implications**

## 🔧 Development Tips

### Build Configurations

```bash
# Debug build with all checks
cmake -DCMAKE_BUILD_TYPE=Debug -DLIBSTATS_ENABLE_RUNTIME_CHECKS=ON ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Conservative SIMD (for compatibility testing)
cmake -DLIBSTATS_CONSERVATIVE_SIMD=ON ..
```

### Useful CMake Options

- `LIBSTATS_ENABLE_RUNTIME_CHECKS`: Enable additional runtime validation
- `LIBSTATS_CONSERVATIVE_SIMD`: Use conservative SIMD settings
- `BUILD_TESTING`: Enable/disable test building
- `BUILD_EXAMPLES`: Enable/disable example building

### IDE Setup

The project includes CMake integration for popular IDEs:
- **VSCode**: Use the CMake Tools extension
- **CLion**: Open CMakeLists.txt directly
- **Visual Studio**: Use "Open Folder" with CMake support

## 📜 License

By contributing to libstats, you agree that your contributions will be licensed under the MIT License that governs this project.

## 🤝 Code of Conduct

We are committed to fostering a welcoming and inclusive community. Please:

- **Be respectful** in all interactions
- **Be constructive** in feedback and criticism
- **Be collaborative** and help others learn
- **Be patient** with newcomers and questions

## 🎯 Current Priority Areas

We're particularly interested in contributions in these areas:

1. **Additional Distributions**: Beta, Chi-squared, Student's t, F-distribution
2. **Statistical Tests**: More goodness-of-fit tests and validation methods
3. **Performance Optimization**: SIMD improvements and cache optimizations
4. **Documentation**: API documentation and usage examples
5. **Cross-Platform Support**: Testing and fixes for different platforms

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions and community discussion
- **Code Review**: We provide thorough, constructive code reviews

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/) for clear communication and automated versioning.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Commit Types

- **feat**: New feature (triggers minor version bump)
- **fix**: Bug fix (triggers patch version bump)
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring without changing functionality
- **perf**: Performance improvements
- **test**: Adding or updating tests
- **build**: Build system or dependency changes
- **ci**: CI/CD configuration changes
- **chore**: Other maintenance tasks
- **revert**: Reverts a previous commit

### Examples

```bash
# Feature
git commit -m "feat(gaussian): add parallel batch fitting support"

# Bug fix
git commit -m "fix(simd): correct AVX2 detection on older CPUs"

# Breaking change (triggers major version bump)
git commit -m "feat(api)!: change distribution interface

BREAKING CHANGE: All distribution classes now require explicit template parameters"
```

### Using the Commit Template

Configure git to use our commit message template:

```bash
git config commit.template .gitmessage
```

## 🚀 Release Process

Our release process is automated using semantic versioning based on commit messages.

### Automatic Version Bumping

- **feat commits**: Trigger minor version bump (0.X.0)
- **fix/perf commits**: Trigger patch version bump (0.0.X)
- **BREAKING CHANGE**: Triggers major version bump (X.0.0)

### Release Workflow

1. Commits to `main` trigger automatic version analysis
2. Version is bumped based on commit types since last release
3. CHANGELOG.md is automatically generated
4. GitHub release is created with release notes
5. Documentation is built and deployed to GitHub Pages

### Manual Releases

Maintainers can trigger manual releases:
1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Select release type (patch/minor/major)

---

Thank you for contributing to libstats! Your efforts help make statistical computing in C++ more accessible and powerful for everyone.
