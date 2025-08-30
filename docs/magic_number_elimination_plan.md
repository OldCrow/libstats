# Magic Number Elimination Plan for libstats

**Created**: 2025-08-21
**Target Version**: v0.11.0
**Estimated Duration**: 2-3 days
**Priority**: HIGH (blocking header optimization work)

---

## 🎯 Objective

Systematically find and eliminate ALL magic numbers in the libstats codebase, replacing them with named constants to:
1. Enable accurate IWYU analysis (Include What You Use)
2. Improve code readability and maintainability
3. Centralize mathematical and statistical constants
4. Prevent future refactoring issues

---

## 🔍 Current Problem Analysis

### Issues with Previous Attempt
1. **Over-aggressive replacement**: Script replaced numbers in string literals (e.g., "l2 cache" became "lTWO cache")
2. **Incomplete coverage**: Missing many floating-point literals and mathematical constants
3. **Context-unaware**: Didn't distinguish between actual magic numbers vs. array indices, loop counters
4. **Missing constants**: Not all mathematical, statistical, and precision constants were identified

### Types of Magic Numbers Found

#### 1. Mathematical Constants (Most Common)
- `0.0`, `1.0`, `2.0` - Basic arithmetic constants
- `0.5` (HALF), `3.0`, `6.0`, `9.0` - Common divisors/multipliers
- `3.841`, `5.991`, `7.815` - Chi-squared critical values
- `1.645`, `1.96` - Normal distribution quantiles
- `-0.5` (NEG_HALF) - Gaussian exponent term
- Various powers of 2 and 10

#### 2. Statistical Test Constants
- Critical values for tests (chi-squared, t-distribution, etc.)
- Significance levels (0.05, 0.01, 0.99, 0.90)
- Degrees of freedom specific values
- Test statistic thresholds

#### 3. Algorithm-Specific Constants
- Convergence tolerances (1e-12, 1e-10, etc.)
- Iteration limits (100, 200, 1000, 5000)
- Threshold values for algorithm switching
- SIMD-related batch sizes

#### 4. Distribution-Specific Parameters
- Shape parameters for special cases
- Boundary conditions (0.001 as minimum values)
- Approximation coefficients (Lanczos, Taylor series)

---

## 📋 Action Plan

### Phase 1: Audit and Catalog (Day 1)

#### Step 1.1: Create Magic Number Detection Script
```bash
#!/bin/bash
# magic_number_finder.sh

# Find all numeric literals in source files, excluding:
# - Array indices [0], [1], etc.
# - Loop counters in for loops
# - Size calculations with size_t
# - Comments and string literals

# Categories to search:
# 1. Floating-point literals
# 2. Integer literals > 1 (excluding array access)
# 3. Negative numbers
# 4. Scientific notation
```

#### Step 1.2: Categorize Found Numbers
Create a spreadsheet/table with columns:
- File location
- Line number
- Magic number value
- Context (5 lines before/after)
- Category (math/statistical/threshold/etc.)
- Proposed constant name
- Header file location

#### Step 1.3: Identify Missing Constants
Review existing constant headers to find gaps:
- `mathematical_constants.h` - Check completeness
- `statistical_constants.h` - Missing test critical values
- `precision_constants.h` - Missing convergence tolerances
- `threshold_constants.h` - Missing algorithm thresholds

### Phase 2: Constant Definition (Day 1-2)

#### Step 2.1: Expand Existing Constant Headers

**mathematical_constants.h** additions:
```cpp
// Common fractions
constexpr double ONE_THIRD = 1.0 / 3.0;
constexpr double TWO_THIRDS = 2.0 / 3.0;
constexpr double ONE_SIXTH = 1.0 / 6.0;

// Powers and roots
constexpr double SQRT_3 = 1.7320508075688772;
constexpr double CUBE_ROOT_2 = 1.2599210498948731;

// Negative values
constexpr double NEG_ONE = -1.0;
constexpr double NEG_TWO = -2.0;
```

**statistical_constants.h** additions:
```cpp
// Chi-squared critical values at α = 0.05
namespace chi_squared_critical {
    constexpr double DF_1_ALPHA_05 = 3.841;
    constexpr double DF_2_ALPHA_05 = 5.991;
    constexpr double DF_3_ALPHA_05 = 7.815;
    constexpr double DF_4_ALPHA_05 = 9.488;
    constexpr double DF_5_ALPHA_05 = 11.070;
}

// Normal distribution quantiles
namespace normal_quantiles {
    constexpr double Z_90 = 1.645;  // 90% confidence
    constexpr double Z_95 = 1.96;   // 95% confidence
    constexpr double Z_99 = 2.576;  // 99% confidence
}

// Significance levels
namespace significance {
    constexpr double ALPHA_01 = 0.01;
    constexpr double ALPHA_05 = 0.05;
    constexpr double ALPHA_10 = 0.10;
}
```

**precision_constants.h** additions:
```cpp
// Convergence tolerances
namespace convergence {
    constexpr double TIGHT = 1e-12;
    constexpr double STANDARD = 1e-10;
    constexpr double RELAXED = 1e-8;
    constexpr double LOOSE = 1e-6;
}

// Iteration limits
namespace iterations {
    constexpr int SMALL = 100;
    constexpr int MEDIUM = 500;
    constexpr int LARGE = 1000;
    constexpr int VERY_LARGE = 5000;
}
```

#### Step 2.2: Create New Specialized Headers (if needed)
```cpp
// include/core/algorithm_constants.h
namespace libstats::constants::algorithm {
    // Lanczos approximation coefficients
    namespace lanczos {
        constexpr double G = 7.0;
        constexpr double COEFF_0 = 0.99999999999980993;
        // ... etc
    }

    // Series expansion thresholds
    namespace series {
        constexpr int MAX_TERMS = 200;
        constexpr double TERM_CUTOFF = 1e-10;
    }
}
```

### Phase 3: Systematic Replacement (Day 2)

#### Step 3.1: Create Smart Replacement Script
```python
#!/usr/bin/env python3
# smart_magic_replacer.py

import re
import ast

class MagicNumberReplacer:
    def __init__(self):
        self.replacements = {
            # Mathematical constants
            '0.0': 'constants::math::ZERO_DOUBLE',
            '1.0': 'constants::math::ONE',
            '2.0': 'constants::math::TWO',
            '0.5': 'constants::math::HALF',
            '-0.5': 'constants::math::NEG_HALF',

            # Statistical critical values
            '3.841': 'constants::chi_squared_critical::DF_1_ALPHA_05',
            '5.991': 'constants::chi_squared_critical::DF_2_ALPHA_05',

            # Add more mappings...
        }

    def should_replace(self, match, context):
        """Determine if a number should be replaced based on context"""
        # Don't replace in:
        # - String literals
        # - Comments
        # - Array indices
        # - Case labels
        # - Include guards
        return True  # Implement logic

    def add_required_includes(self, file_content, used_constants):
        """Add necessary #include statements"""
        pass
```

#### Step 3.2: File-by-File Replacement Strategy

**Priority Order**:
1. **Core math files** (math_utils.cpp, validation.cpp)
2. **Distribution implementations** (gaussian.cpp, gamma.cpp, etc.)
3. **Statistical test files** (goodness_of_fit tests)
4. **Performance/benchmark files**
5. **SIMD implementations**

**Process per file**:
1. Run detection script
2. Review proposed replacements
3. Apply replacements
4. Add required includes
5. Compile and test
6. Commit changes

### Phase 4: Validation (Day 2-3)

#### Step 4.1: Compilation Testing
```bash
# After each file group:
cmake --build build --clean-first
ctest --output-on-failure
```

#### Step 4.2: Performance Verification
```bash
# Run benchmarks to ensure no performance regression
./build/tools/performance_analyzer
./build/examples/gaussian_performance_benchmark
```

#### Step 4.3: IWYU Verification
```bash
# Verify IWYU now gives correct recommendations
./scripts/run-iwyu.sh --src > iwyu_after.txt
diff iwyu_before.txt iwyu_after.txt
```

---

## 🛠️ Implementation Tools

### Detection Script Template
```bash
#!/bin/bash
# find_magic_numbers.sh

echo "=== Finding Magic Numbers in libstats ==="

# Find floating-point literals (excluding 0.0, 1.0 which might be intentional)
echo "Floating-point literals:"
grep -r -n -E '\b[0-9]+\.[0-9]+\b' src/ --include="*.cpp" | \
    grep -v '// ' | \
    grep -v '"' | \
    grep -v 'version' | \
    sort -u

# Find scientific notation
echo "Scientific notation:"
grep -r -n -E '\b[0-9]+(\.[0-9]+)?[eE][+-]?[0-9]+\b' src/ --include="*.cpp"

# Find negative numbers
echo "Negative literals:"
grep -r -n -E '\-[0-9]+(\.[0-9]+)?\b' src/ --include="*.cpp" | \
    grep -v '// ' | \
    grep -v 'n - 1'  # Exclude common patterns
```

### Validation Checklist
- [ ] All magic numbers documented in catalog
- [ ] Constants defined with clear, descriptive names
- [ ] Constants grouped logically in appropriate headers
- [ ] All source files updated with proper includes
- [ ] No compilation errors or warnings
- [ ] All tests pass
- [ ] No performance regression
- [ ] IWYU provides accurate recommendations

---

## 📊 Success Metrics

1. **Zero undocumented magic numbers** in non-test source files
2. **100% compilation success** across all platforms
3. **IWYU accuracy** improved (no false positives from constants)
4. **Performance maintained** (±1% of baseline)
5. **Code clarity improved** (self-documenting constant names)

---

## ⚠️ Risks and Mitigations

### Risk 1: Breaking Existing Code
**Mitigation**:
- Test after each file group
- Use version control for easy rollback
- Keep original values in comments initially

### Risk 2: Performance Impact
**Mitigation**:
- Use `constexpr` for all constants
- Verify compiler optimization with assembly output
- Run performance benchmarks after changes

### Risk 3: Over-Engineering
**Mitigation**:
- Don't replace obvious loop counters (0, 1 for indices)
- Keep some context-obvious values (like 2 for pair operations)
- Document exceptions clearly

---

## 📅 Timeline

### Day 1: Audit and Define
- Morning: Run detection scripts, catalog magic numbers
- Afternoon: Define new constants, update headers

### Day 2: Replace and Test
- Morning: Replace magic numbers in core files
- Afternoon: Replace in distributions, test thoroughly

### Day 3: Finalize
- Morning: Complete remaining files, final testing
- Afternoon: Documentation, IWYU verification, commit

---

## 🔄 Next Steps

1. **Immediate**: Run comprehensive magic number detection
2. **Today**: Begin cataloging and categorization
3. **Tomorrow**: Start systematic replacement
4. **Day 3**: Complete validation and integration

---

## 📝 Notes

- Coordinate with v0.11.0 header optimization work
- This is a prerequisite for accurate IWYU analysis
- Consider creating a coding standard for future constant usage
- May discover additional optimization opportunities during review
