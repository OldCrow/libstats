# Loop Modernization Plan for libstats

**Created**: 2025-08-21
**Target Version**: v0.11.0
**Estimated Duration**: 1-2 days (concurrent with magic number elimination)
**Priority**: MEDIUM (improves code quality and maintainability)

---

## 🎯 Objectives

Modernize loop constructs throughout the libstats codebase to:
1. Replace traditional index-based loops with range-based for loops where appropriate
2. Use STL algorithms (std::transform, std::accumulate, etc.) for clearer intent
3. Leverage C++20 ranges where beneficial
4. Improve code readability and reduce off-by-one errors
5. Enable better compiler optimizations

---

## 📊 Current State Analysis

### Loop Pattern Distribution
- **201** traditional index-based loops (`for (size_t i = 0; i < ...`)
- **32** range-based for loops (already modernized)
- **63** STL algorithm usages (std::accumulate, std::transform)

### Common Patterns Found

#### 1. Simple Iteration Over Container
```cpp
// OLD: Index-based
for (size_t i = 0; i < data.size(); ++i) {
    process(data[i]);
}

// NEW: Range-based
for (const auto& value : data) {
    process(value);
}
```

#### 2. Accumulation Patterns
```cpp
// OLD: Manual accumulation
double sum = 0.0;
for (size_t i = 0; i < data.size(); ++i) {
    sum += data[i];
}

// NEW: std::accumulate
const double sum = std::accumulate(data.begin(), data.end(), 0.0);
```

#### 3. Transformation Patterns
```cpp
// OLD: Index-based transformation
for (size_t i = 0; i < n; ++i) {
    results[i] = compute(data[i]);
}

// NEW: std::transform
std::transform(data.begin(), data.end(), results.begin(), compute);
```

#### 4. Statistical Calculations
```cpp
// OLD: Manual variance calculation
double variance = 0.0;
for (size_t i = 0; i < data.size(); ++i) {
    const double diff = data[i] - mean;
    variance += diff * diff;
}

// NEW: std::inner_product or std::transform_reduce
const double variance = std::transform_reduce(
    data.begin(), data.end(), 0.0, std::plus<>{},
    [mean](double x) {
        const double diff = x - mean;
        return diff * diff;
    }
);
```

---

## 📋 Modernization Categories

### Category 1: Keep Index-Based (Necessary)

Some loops MUST remain index-based:
- **When index is used in calculation** (e.g., weighted calculations)
- **When accessing multiple containers by index**
- **When index represents mathematical meaning** (e.g., degrees of freedom)
- **SIMD operations requiring aligned access**

Example from validation.cpp:
```cpp
// KEEP AS-IS: Index has mathematical meaning
for (size_t i = 0; i < n; ++i) {
    ecdf[i] = static_cast<double>(i + 1) / static_cast<double>(n);
}
```

### Category 2: Simple Range-Based Conversion

Direct replacements where index is not needed:
```cpp
// OLD
for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] <= 0.0) {
        throw std::invalid_argument("All values must be positive");
    }
}

// NEW
for (double value : data) {
    if (value <= 0.0) {
        throw std::invalid_argument("All values must be positive");
    }
}
```

### Category 3: Algorithm Replacements

Use STL algorithms for common patterns:

#### Accumulation
```cpp
// Replace manual summation
std::accumulate(data.begin(), data.end(), 0.0)
```

#### Finding min/max
```cpp
// Replace manual min/max loops
const auto [min_it, max_it] = std::minmax_element(data.begin(), data.end());
```

#### Counting
```cpp
// Replace manual counting
const auto count = std::count_if(data.begin(), data.end(),
    [threshold](double x) { return x > threshold; });
```

#### All/Any/None checks
```cpp
// Replace early-exit loops
if (std::any_of(data.begin(), data.end(),
    [](double x) { return x < 0.0; })) {
    throw std::invalid_argument("Negative values not allowed");
}
```

### Category 4: C++20 Ranges (Where Available)

For compilers with C++20 ranges support:
```cpp
// Using ranges for pipeline operations
#include <ranges>

auto positive_values = data
    | std::views::filter([](double x) { return x > 0.0; })
    | std::views::transform([](double x) { return std::log(x); });
```

---

## 🛠️ Implementation Plan

### Phase 1: Analysis and Categorization (Day 1 Morning)

#### Step 1.1: Create Loop Analysis Script
```bash
#!/bin/bash
# analyze_loops.sh

echo "=== Analyzing Loop Patterns in libstats ==="

# Find loops that use index in calculation
echo "Loops using index in calculation:"
grep -n "for.*size_t i.*\[i\].*i[^]]" src/*.cpp

# Find simple iteration patterns
echo "Simple iteration candidates:"
grep -n "for.*size_t i = 0.*\.size().*\[i\]" src/*.cpp | \
    grep -v "i\s*[+\-*/]" | grep -v "[+\-*/]\s*i"

# Find accumulation patterns
echo "Accumulation candidates:"
grep -B2 -A2 "sum\s*+=\|total\s*+=" src/*.cpp
```

#### Step 1.2: Create Categorization Spreadsheet
| File | Line | Current Pattern | Category | Proposed Change | Priority |
|------|------|----------------|----------|-----------------|----------|
| validation.cpp | 91 | `for(i<size)` | Keep | Index needed | Low |
| gaussian.cpp | 1205 | `for(i<n)` | Keep | Weighted calc | Low |
| gamma.cpp | 559 | `for(value:data)` | Already modern | None | N/A |

### Phase 2: Safe Modernization (Day 1 Afternoon)

#### Step 2.1: Low-Risk Changes First

**Target: Simple range-based conversions**
```cpp
// Files to update first (validation checks):
- validation.cpp: Simple data validation loops
- safety.cpp: Input validation loops
- Distribution parameter validation loops
```

#### Step 2.2: Algorithm Replacements

**Target: Mathematical operations**
```cpp
// Common patterns to replace:
1. Sum calculations → std::accumulate
2. Min/max finding → std::minmax_element
3. Variance calculations → std::inner_product or std::transform_reduce
4. Count operations → std::count_if
```

### Phase 3: Testing and Validation (Day 2)

#### Step 3.1: Compile and Test After Each File
```bash
# After each file modification:
cmake --build build --target <specific_target>
ctest -R <related_tests>
```

#### Step 3.2: Performance Verification
```bash
# Ensure no performance regression
./build/tools/performance_analyzer --before-after
```

---

## 📝 Modernization Guidelines

### DO Modernize:
✅ Simple iterations where index is not needed
✅ Accumulation/reduction operations
✅ Search and count operations
✅ Validation loops
✅ Range-based operations on whole containers

### DON'T Modernize:
❌ SIMD loops requiring specific memory access patterns
❌ Loops where index has mathematical meaning
❌ Performance-critical tight loops (verify with benchmarks)
❌ Loops accessing multiple containers by shared index
❌ Complex nested loops with interdependencies

### Best Practices:
1. **Use `const auto&` for read-only access**
   ```cpp
   for (const auto& value : data) { /* read only */ }
   ```

2. **Use `auto&` when modification needed**
   ```cpp
   for (auto& value : data) { value *= 2.0; }
   ```

3. **Prefer algorithms for intent clarity**
   ```cpp
   // Clear intent: finding if any negative
   const bool has_negative = std::any_of(data.begin(), data.end(),
       [](double x) { return x < 0.0; });
   ```

4. **Use structured bindings for pairs/tuples**
   ```cpp
   for (const auto& [key, value] : map) { /* ... */ }
   ```

---

## 🔄 Integration with Magic Number Elimination

### Synergies:
1. **Replace loop bounds magic numbers simultaneously**
   ```cpp
   // OLD
   for (int i = 0; i < 100; ++i) { /* ... */ }

   // NEW
   for (int i = 0; i < constants::iterations::SMALL; ++i) { /* ... */ }
   ```

2. **Modernize accumulation with proper initial values**
   ```cpp
   // OLD
   double sum = 0.0;
   for (...) { sum += x; }

   // NEW
   const double sum = std::accumulate(data.begin(), data.end(),
                                      constants::math::ZERO_DOUBLE);
   ```

3. **Clean up complex calculations**
   ```cpp
   // Combine modernization with constant replacement
   const double variance = std::transform_reduce(
       data.begin(), data.end(),
       constants::math::ZERO_DOUBLE,
       std::plus<>{},
       [mean](double x) {
           const double diff = x - mean;
           return diff * diff;
       }
   ) / static_cast<double>(n - constants::math::ONE);
   ```

---

## 📊 Success Metrics

1. **Reduced line count**: ~5-10% reduction in loop-related code
2. **Improved readability**: Clearer intent through algorithm names
3. **Fewer potential bugs**: Elimination of off-by-one errors
4. **Performance maintained**: ±1% of baseline performance
5. **Modern C++ adoption**: >50% of eligible loops modernized

---

## 🚀 Execution Strategy

### Day 1: Analysis and Simple Changes
- **Morning**: Run analysis scripts, categorize all loops
- **Afternoon**: Implement simple range-based conversions
- **Evening**: Test and commit first batch

### Day 2: Algorithms and Complex Changes
- **Morning**: Replace accumulation/reduction patterns
- **Afternoon**: Handle complex transformations
- **Evening**: Final testing and documentation

---

## ⚠️ Risk Mitigation

### Risk 1: Performance Regression
**Mitigation**:
- Benchmark before and after each change
- Keep index-based loops for SIMD operations
- Profile hot paths before modernization

### Risk 2: Semantic Changes
**Mitigation**:
- Careful review of index usage
- Maintain exact numerical behavior
- Comprehensive testing after each change

### Risk 3: Compiler Compatibility
**Mitigation**:
- Use C++17 features as baseline
- C++20 ranges only with feature detection
- Provide fallbacks for older compilers

---

## 📚 Examples from libstats

### Example 1: Validation Loop (validation.cpp)
```cpp
// BEFORE
for (size_t i = 0; i < sorted_data.size(); ++i) {
    ecdf[i] = static_cast<double>(i + 1) / static_cast<double>(sorted_data.size());
}

// KEEP AS-IS: Index has mathematical meaning in ECDF calculation
```

### Example 2: Simple Validation (gamma.cpp)
```cpp
// BEFORE
for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] <= 0.0) {
        throw std::invalid_argument("All values must be positive");
    }
}

// AFTER
if (std::any_of(data.begin(), data.end(),
    [](double value) { return value <= constants::math::ZERO_DOUBLE; })) {
    throw std::invalid_argument("All values must be positive for Gamma distribution");
}
```

### Example 3: Accumulation (gaussian.cpp)
```cpp
// BEFORE
double sum = 0.0;
for (size_t i = 0; i < data.size(); ++i) {
    sum += data[i];
}
const double mean = sum / data.size();

// AFTER
const double mean = std::accumulate(data.begin(), data.end(),
                                    constants::math::ZERO_DOUBLE) /
                   static_cast<double>(data.size());
```

---

## 🔗 Related Work

- Coordinate with magic number elimination (same timeframe)
- Update coding standards to prefer modern constructs
- Consider creating lint rules for future development
- Document patterns in developer guidelines
