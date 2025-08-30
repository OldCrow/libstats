# phase 1 execution plan: magic number elimination

## overview

based on the analysis, we have identified 827 magic number replacement opportunities across 20 cpp files. this document outlines a systematic approach to eliminate magic numbers and replace them with named constants from the `stats::detail::` namespace.

## analysis summary

### file distribution
- **high-impact files (50+ suggestions)**: 6 files, 653 suggestions (79% of total)
- **medium-impact files (20-49 suggestions)**: 3 files, 90 suggestions (11% of total)
- **low-impact files (5-19 suggestions)**: 5 files, 72 suggestions (9% of total)
- **minimal-impact files (1-4 suggestions)**: 6 files, 12 suggestions (1% of total)

### most common replacements needed
1. `detail::ZERO_DOUBLE` - 466 occurrences (56%)
2. `detail::ONE` - 279 occurrences (34%)
3. `detail::ZERO_INT` - 145 occurrences (18%)
4. `detail::TWO` - 119 occurrences (14%)

## execution strategy

### phase 1a: pilot testing (1 file)
**objective**: validate the complete workflow on a representative file

**target file**: `src/exponential.cpp` (7 suggestions, low-risk)
- test detection, replacement, compilation, and validation
- establish workflow procedures
- identify any edge cases or issues

### phase 1b: high-impact files (6 files)
**objective**: tackle files with the most magic numbers to maximize impact

**processing order** (by suggestion count):
1. `src/gamma.cpp` - 210 suggestions
2. `src/math_utils.cpp` - 136 suggestions
3. `src/poisson.cpp` - 124 suggestions
4. `src/discrete.cpp` - 88 suggestions
5. `src/validation.cpp` - 72 suggestions
6. `src/uniform.cpp` - 58 suggestions

**approach**: interactive mode for first few files, then batch processing

### phase 1c: medium and low-impact files (8 files)
**objective**: complete remaining files with moderate magic number counts

**files**:
- `src/distribution_base.cpp` - 38 suggestions
- `src/benchmark.cpp` - 28 suggestions
- `src/gaussian.cpp` - 24 suggestions
- `src/system_capabilities.cpp` - 9 suggestions
- `src/exponential.cpp` - 7 suggestions (if not done in pilot)
- `src/performance_dispatcher.cpp` - 7 suggestions
- `src/cpu_detection.cpp` - 5 suggestions
- `src/log_space_ops.cpp` - 5 suggestions

### phase 1d: minimal-impact files (6 files)
**objective**: complete remaining files for consistency

**files**:
- `src/performance_history.cpp` - 4 suggestions
- `src/simd_neon.cpp` - 4 suggestions
- `src/platform_constants_impl.cpp` - 3 suggestions
- `src/work_stealing_pool.cpp` - 3 suggestions
- `src/parallel_thresholds.cpp` - 1 suggestions
- `src/simd_fallback.cpp` - 1 suggestions

## execution workflow per file

### 1. pre-processing checks
```bash
# ensure file compiles
make clean && make src/[filename].o

# create backup
cp src/[filename].cpp src/[filename].cpp.backup

# analyze suggestions
python3 tools/replace_magic_numbers.py src/[filename].cpp
```

### 2. replacement application
```bash
# interactive mode (recommended for high-impact files)
python3 tools/replace_magic_numbers.py src/[filename].cpp --interactive

# or batch mode (for low-impact files)
python3 tools/replace_magic_numbers.py src/[filename].cpp --write
```

### 3. post-processing validation
```bash
# verify compilation
make clean && make src/[filename].o

# run relevant tests if available
make test

# visual inspection of changes
git diff src/[filename].cpp
```

### 4. commit and document
```bash
git add src/[filename].cpp
git commit -m "refactor: eliminate magic numbers in [filename]

Replace magic numbers with named constants from stats::detail namespace.
- Applied [X] magic number replacements
- Enhanced code readability and maintainability"
```

## quality assurance

### validation criteria
1. **compilation**: file must compile without errors
2. **functionality**: existing tests must continue to pass
3. **readability**: replacements should improve code clarity
4. **consistency**: use established naming conventions

### risk mitigation
- **backups**: create backup files before processing
- **incremental commits**: commit each file separately
- **testing**: run test suites after each major file
- **review**: manual review of high-impact file changes

## resource requirements

### time estimates
- **pilot testing**: 1 hour
- **high-impact files**: 3-4 hours (30-45 min per file)
- **medium/low-impact files**: 2-3 hours (15-20 min per file)
- **minimal-impact files**: 1 hour (5-10 min per file)

**total estimated time**: 7-9 hours

### prerequisites
- updated `tools/replace_magic_numbers.py` script
- complete constant definitions in `stats::detail` namespace
- working build system
- test suite availability

## success criteria

### quantitative metrics
- **coverage**: 90%+ of identified magic numbers replaced
- **compilation**: 100% of files compile successfully post-replacement
- **testing**: 100% of existing tests continue to pass

### qualitative outcomes
- improved code readability and maintainability
- consistent use of named constants across codebase
- enhanced developer experience with meaningful constant names
- foundation established for phase 2 (iwyu header optimization)

## next phase preparation

once phase 1 is complete:
1. document lessons learned and process improvements
2. prepare for phase 2: iwyu-based header optimization
3. update constant includes where necessary
4. establish ongoing magic number prevention practices
