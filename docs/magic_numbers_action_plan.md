# Magic Number Elimination - Action Plan

**Start Time**: 2025-08-21
**Goal**: Replace 919 magic numbers with named constants

---

## 📋 File Processing Order

### Batch 1: Foundation Files (CRITICAL - Do First!)
These files are used by many others. Fix them first to avoid propagating issues.

#### 1. **validation.cpp** (30 magic numbers) - START HERE
**Why first**: Used by 5 other files, contains chi-squared critical values
**Key magic numbers to replace**:
```cpp
// Chi-squared critical values
3.841  → constants::statistical::chi_square::CHI2_95_DF_1
5.991  → constants::statistical::chi_square::CHI2_95_DF_2
7.815  → constants::statistical::chi_square::CHI2_95_DF_3
9.488  → constants::statistical::chi_square::CHI2_95_DF_4
11.070 → constants::statistical::chi_square::CHI2_95_DF_5

// Normal quantiles
1.645  → constants::statistical::normal::Z_95_ONE_TAIL
1.96   → constants::statistical::normal::Z_95

// Lanczos coefficients (need new constants)
7.0    → constants::algorithm::lanczos::G
0.99999999999980993 → constants::algorithm::lanczos::COEFF_0
// ... etc

// Anderson-Darling critical values (need new constants)
0.576, 0.656, 0.787, 1.248, 1.610, 1.933, 2.492, 3.070, 3.857, 4.500
```

#### 2. **math_utils.cpp** (29 magic numbers)
**Why second**: Used by 7 other files, core mathematical operations
**Key magic numbers**:
```cpp
0.5   → constants::math::HALF
2.0   → constants::math::TWO
3.0   → constants::math::THREE
1e-12 → constants::precision::HIGH_PRECISION_TOLERANCE
```

#### 3. **distribution_base.cpp** (7 magic numbers)
**Why third**: Base class for all distributions
**Quick win**: Only 7 numbers to fix

---

### Batch 2: High-Impact Distribution Files

#### 4. **gamma.cpp** (41 magic numbers - HIGHEST)
**Why next**: Most magic numbers, complex calculations
**Special attention**: Many algorithm-specific constants

#### 5. **discrete.cpp** (36 magic numbers)
**Why**: Second highest count

#### 6. **uniform.cpp** (31 magic numbers)

#### 7. **gaussian.cpp** (19 magic numbers)
**Important**: Core distribution, but fewer magic numbers

#### 8. **poisson.cpp** (11 magic numbers)

#### 9. **exponential.cpp** (2 magic numbers)
**Quick win**: Only 2 to fix

---

### Batch 3: Remaining Files
- performance_dispatcher.cpp (6)
- benchmark.cpp (4)
- system_capabilities.cpp (3)
- performance_history.cpp (1)
- safety.cpp (0 - skip)

---

## 🔧 New Constants Needed

### Create: `include/core/algorithm_constants.h`
```cpp
#pragma once

namespace libstats {
namespace constants {
namespace algorithm {

// Lanczos approximation for gamma function
namespace lanczos {
    inline constexpr double G = 7.0;
    inline constexpr double COEFF_0 = 0.99999999999980993;
    inline constexpr double COEFF_1 = 676.5203681218851;
    inline constexpr double COEFF_2 = -1259.1392167224028;
    inline constexpr double COEFF_3 = 771.32342877765313;
    inline constexpr double COEFF_4 = -176.61502916214059;
    inline constexpr double COEFF_5 = 12.507343278686905;
    inline constexpr double COEFF_6 = -0.13857109526572012;
    inline constexpr double COEFF_7 = 9.9843695780195716e-6;
    inline constexpr double COEFF_8 = 1.5056327351493116e-7;
}

// Anderson-Darling test critical values
namespace anderson_darling {
    inline constexpr double CRIT_50 = 0.576;   // α = 0.50
    inline constexpr double CRIT_40 = 0.656;   // α = 0.40
    inline constexpr double CRIT_30 = 0.787;   // α = 0.30
    inline constexpr double CRIT_25 = 1.248;   // α = 0.25
    inline constexpr double CRIT_15 = 1.610;   // α = 0.15
    inline constexpr double CRIT_10 = 1.933;   // α = 0.10
    inline constexpr double CRIT_05 = 2.492;   // α = 0.05
    inline constexpr double CRIT_025 = 3.070;  // α = 0.025
    inline constexpr double CRIT_01 = 3.857;   // α = 0.01
    inline constexpr double CRIT_005 = 4.500;  // α = 0.005
}

// Significance levels
namespace significance {
    inline constexpr double ALPHA_001 = 0.001;
    inline constexpr double ALPHA_005 = 0.005;
    inline constexpr double ALPHA_01 = 0.01;
    inline constexpr double ALPHA_025 = 0.025;
    inline constexpr double ALPHA_05 = 0.05;
    inline constexpr double ALPHA_10 = 0.10;
    inline constexpr double ALPHA_15 = 0.15;
    inline constexpr double ALPHA_25 = 0.25;
    inline constexpr double ALPHA_30 = 0.30;
    inline constexpr double ALPHA_40 = 0.40;
    inline constexpr double ALPHA_50 = 0.50;
}

// Common iteration limits
namespace iterations {
    inline constexpr int SMALL = 100;
    inline constexpr int MEDIUM = 200;
    inline constexpr int LARGE = 500;
    inline constexpr int VERY_LARGE = 1000;
    inline constexpr int HUGE = 5000;
}

// Convergence tolerances
namespace convergence {
    inline constexpr double ULTRA_TIGHT = 1e-15;
    inline constexpr double VERY_TIGHT = 1e-12;
    inline constexpr double TIGHT = 1e-10;
    inline constexpr double STANDARD = 1e-8;
    inline constexpr double RELAXED = 1e-6;
    inline constexpr double LOOSE = 1e-4;
}

}  // namespace algorithm
}  // namespace constants
}  // namespace libstats
```

### Add to `mathematical_constants.h`:
```cpp
// Missing common values
inline constexpr double SEVEN = 7.0;
inline constexpr double EIGHT = 8.0;
inline constexpr double NINE = 9.0;
inline constexpr double TWENTY = 20.0;
inline constexpr double THIRTY = 30.0;
inline constexpr double FIFTY = 50.0;

// Missing fractions
inline constexpr double TWO_THIRDS = 2.0 / 3.0;
inline constexpr double THREE_HALVES = 1.5;
inline constexpr double FIVE_HALVES = 2.5;

// Special values for validation
inline constexpr double POINT_27 = 0.27;  // KS test threshold
```

---

## 🚀 Execution Steps

### Step 1: Create new constant header
```bash
# Create algorithm_constants.h with the content above
vim include/core/algorithm_constants.h
```

### Step 2: Start with validation.cpp
```bash
# Back up original
cp src/validation.cpp src/validation.cpp.backup

# Edit and replace magic numbers
vim src/validation.cpp

# Key replacements:
# - Line 69: 3.841 → constants::statistical::chi_square::CHI2_95_DF_1
# - Line 71: 5.991 → constants::statistical::chi_square::CHI2_95_DF_2
# - Line 121: 7.0 → constants::algorithm::lanczos::G
# - Lines 322-331: Anderson-Darling values
# etc.

# Add includes at top:
#include "../include/core/algorithm_constants.h"
#include "../include/core/statistical_constants.h"
```

### Step 3: Compile and test after each file
```bash
# After validation.cpp
cmake --build build --target validation
ctest -R validation

# After math_utils.cpp
cmake --build build --target math_utils
ctest -R math

# Continue pattern...
```

### Step 4: Run IWYU to verify
```bash
./scripts/run-iwyu.sh src/validation.cpp
# Should now correctly identify which constant headers are needed
```

---

## ✅ Checklist

### Foundation Files (Day 1 Morning)
- [ ] Create algorithm_constants.h
- [ ] validation.cpp - 30 magic numbers
- [ ] math_utils.cpp - 29 magic numbers
- [ ] distribution_base.cpp - 7 magic numbers
- [ ] Test foundation files

### Distribution Files (Day 1 Afternoon - Day 2 Morning)
- [ ] gamma.cpp - 41 magic numbers
- [ ] discrete.cpp - 36 magic numbers
- [ ] uniform.cpp - 31 magic numbers
- [ ] gaussian.cpp - 19 magic numbers
- [ ] poisson.cpp - 11 magic numbers
- [ ] exponential.cpp - 2 magic numbers
- [ ] Test all distributions

### Remaining Files (Day 2 Afternoon)
- [ ] performance_dispatcher.cpp - 6 magic numbers
- [ ] benchmark.cpp - 4 magic numbers
- [ ] system_capabilities.cpp - 3 magic numbers
- [ ] performance_history.cpp - 1 magic number
- [ ] Final testing

### Validation (Day 2 End)
- [ ] Run full test suite
- [ ] Run performance benchmarks
- [ ] Run IWYU analysis
- [ ] Document changes

---

## 🎯 Success Criteria
- Zero magic numbers in source files
- All tests pass
- No performance regression
- IWYU gives accurate recommendations
- Clean compilation with no warnings
