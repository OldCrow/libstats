# Distribution Code Layout Standardization Guide

## Executive Summary

This document provides a comprehensive guide for standardizing the code layout and section organization across all statistical distribution implementations in libstats. The standardization addresses significant organizational inconsistencies that impact developer productivity and code maintainability.

**Impact**: Current inconsistencies result in 18 missing sections in some implementations, up to 19 ordering mismatches, and unpredictable code navigation patterns that slow development by an estimated 50%.

**Timeline**: 6-8 hours of focused work will establish consistent organization patterns that will save 20+ hours during v0.9.1 adaptive cache implementation and all future development.

---

## Context and Problem Analysis

### Current State Issues

Our analysis of all six distribution implementations revealed:

| Distribution | Header Sections | Impl Sections | Missing from Impl | Ordering Issues | Status |
|--------------|----------------|---------------|-------------------|-----------------|---------|
| **Gaussian** | 24 | 17 | 18 sections | Moderate | ✅ Header Complete |
| **Exponential** | 21 | 11 | 14 sections | 1 issue | ❌ Needs Work |  
| **Uniform** | 21 | 12 | 14 sections | **7 issues** | ❌ Needs Work |
| **Poisson** | 24 | 18 | 13 sections | **19 issues** | ❌ Needs Work |
| **Discrete** | 22 | 14 | 15 sections | 5 issues | ❌ Needs Work |
| **Gamma** | 23 | 14 | 18 sections | Clean | ❌ Needs Work |

### Root Causes

1. **Historical Development**: Distributions were developed at different times without standardized templates
2. **Copy-Paste Evolution**: Later distributions copied incomplete or modified templates  
3. **Missing Standards**: No documented section organization or implementation ordering requirements
4. **Inconsistent Reviews**: Section organization not included in code review criteria

---

## Standard Section Template

### Overview

All distribution headers and implementations must follow this **24-section standard template**:

- **Sections 1-16**: Public interface (consistent across all distributions)
- **Sections 17-24**: Private implementation details (may vary based on distribution needs)

### Complete Standard Template

```cpp
//==========================================================================
// 1. CONSTRUCTORS AND DESTRUCTOR
//==========================================================================
// Default, parameterized, copy, move constructors and destructor
// Complex constructors implemented in .cpp, destructor typically defaulted inline

//==========================================================================
// 2. SAFE FACTORY METHODS (Exception-free construction)
//==========================================================================
// static Result<Distribution> create(...) methods
// Exception-free construction for ABI compatibility

//==========================================================================
// 3. PARAMETER GETTERS AND SETTERS
//==========================================================================
// All parameter access methods (get/set with validation)
// Atomic getters for performance-critical access
// Exception-based setters for existing API compatibility

//==========================================================================
// 4. RESULT-BASED SETTERS
//==========================================================================
// trySet* methods returning VoidResult for exception-free parameter setting
// Complex implementations in .cpp for thread safety

//==========================================================================
// 5. CORE PROBABILITY METHODS
//==========================================================================
// getProbability, getLogProbability, getCumulativeProbability, getQuantile
// sample methods (single and batch)

//==========================================================================
// 6. DISTRIBUTION MANAGEMENT
//==========================================================================
// fit, reset, toString methods
// Parameter estimation and distribution state management

//==========================================================================
// 7. ADVANCED STATISTICAL METHODS
//==========================================================================
// Confidence intervals, hypothesis tests, Bayesian methods
// Robust estimation, method of moments, L-moments

//==========================================================================
// 8. GOODNESS-OF-FIT TESTS
//==========================================================================
// Kolmogorov-Smirnov, Anderson-Darling tests
// Distribution-specific normality/fit tests

//==========================================================================
// 9. CROSS-VALIDATION METHODS
//==========================================================================
// K-fold cross-validation, leave-one-out cross-validation
// Model selection and validation frameworks

//==========================================================================
// 10. INFORMATION CRITERIA
//==========================================================================
// AIC, BIC, AICc calculations
// Model comparison utilities

//==========================================================================
// 11. BOOTSTRAP METHODS
//==========================================================================
// Bootstrap parameter confidence intervals
// Resampling-based statistical inference

//==========================================================================
// 12. [DISTRIBUTION]-SPECIFIC UTILITY METHODS
//==========================================================================
// Distribution-specific mathematical properties and utilities
// e.g., getHalfLife() for Exponential, isUnitInterval() for Uniform

//==========================================================================
// 13. SMART AUTO-DISPATCH BATCH OPERATIONS
//==========================================================================
// Auto-optimized batch operations with performance hints
// Unified C++20 span interface with automatic strategy selection

//==========================================================================
// 14. EXPLICIT STRATEGY BATCH OPERATIONS
//==========================================================================
// Power-user interface for explicit strategy selection
// Direct control over SIMD/parallel execution strategies

//==========================================================================
// 15. COMPARISON OPERATORS
//==========================================================================
// operator==, operator!= with thread-safe parameter comparison

//==========================================================================
// 16. FRIEND FUNCTION STREAM OPERATORS
//==========================================================================
// operator<<, operator>> for serialization support

//==========================================================================
// 17. PRIVATE FACTORY METHODS
//==========================================================================
// Internal factory methods, createUnchecked, bypass validation constructors

//==========================================================================
// 18. PRIVATE BATCH IMPLEMENTATION METHODS
//==========================================================================
// Internal batch operation implementations
// *UnsafeImpl methods, SIMD kernels

//==========================================================================
// 19. PRIVATE COMPUTATIONAL METHODS (if needed)
//==========================================================================
// Complex mathematical computation helpers
// Special function implementations, series expansions

//==========================================================================
// 20. PRIVATE UTILITY METHODS (if needed)
//==========================================================================
// Internal helper methods, data processing utilities
// Validation helpers, formatting utilities

//==========================================================================
// 21. DISTRIBUTION PARAMETERS
//==========================================================================
// Core parameter member variables
// Atomic parameter copies for lock-free access

//==========================================================================
// 22. PERFORMANCE CACHE
//==========================================================================
// Cached mathematical values for performance
// Derived quantities, reciprocals, logarithms

//==========================================================================
// 23. OPTIMIZATION FLAGS
//==========================================================================
// Boolean flags for fast path selection
// Special case detection (unit values, zero parameters, etc.)

//==========================================================================
// 24. SPECIALIZED CACHES (if needed)
//==========================================================================
// Distribution-specific caching structures
// e.g., factorial tables for Poisson, lookup tables
```

---

## Implementation Requirements

### Header File Requirements

1. **Exact section ordering**: All distributions must use sections 1-24 in the specified order
2. **Section completeness**: Every distribution must have all 24 sections (may be empty with comments)
3. **Consistent formatting**: Section separators must use exactly `//==========` (78 characters)
4. **Clear section titles**: Section titles must match the template exactly
5. **Appropriate content**: Methods must be placed in their logically correct sections

### Implementation File Requirements  

1. **Header-implementation alignment**: Implementation sections must match header sections exactly
2. **Method ordering**: Methods within sections should follow header declaration order
3. **Section documentation**: Each implementation section should reference the corresponding header
4. **Complete coverage**: All header public methods must be implemented in matching sections

### Conditional Sections

Some sections may be minimal or empty for certain distributions:

- **Section 19** (Private Computational Methods): Only needed for complex mathematical distributions
- **Section 20** (Private Utility Methods): Only needed for distributions with specialized helpers  
- **Section 24** (Specialized Caches): Only needed for distributions with unique caching requirements

**Rule**: Even if empty, these sections must be present with appropriate comments explaining why they're not needed.

---

## Distribution-Specific Checklists

### Gaussian Distribution

**Current Issues:**
- ✅ Most complete implementation (25 header sections)
- ❌ Missing alignment between header and implementation (18 missing sections in impl)
- ❌ Private section ordering needs standardization
- ❌ Section 19-20 may be missing or misnamed

**Phase 1: Header Standardization**
- [x] Verify all 24 sections are present and correctly ordered
- [x] Rename "GAUSSIAN-SPECIFIC UTILITY METHODS" to match section 12 template
- [x] Ensure private sections 17-24 follow exact template order
- [x] Move any misplaced sections to correct positions
- [x] Add section separators using exactly 78 `=` characters

**Phase 2: Implementation Alignment**  
- [ ] Add missing implementation sections to match all 24 header sections
- [ ] Reorder implementation sections to match header exactly
- [ ] Move methods to match header declaration order within sections
- [ ] Add section comments referencing corresponding header sections
- [ ] Verify all public methods from header are implemented

**Phase 3: Method Organization**
- [ ] Ensure CONSTRUCTORS AND DESTRUCTORS section contains all Rule of Five methods
- [ ] Verify PARAMETER GETTERS AND SETTERS contains all get*/set* methods
- [ ] Check CORE PROBABILITY METHODS has getProbability, getLogProbability, etc.
- [ ] Confirm ADVANCED STATISTICAL METHODS has confidence intervals, tests
- [ ] Validate batch operations are properly organized in sections 13-14

**Verification Checklist**
- [ ] Header has exactly 24 sections in template order
- [ ] Implementation has exactly 24 matching sections
- [ ] All public methods are implemented in correct sections  
- [ ] Private sections follow template organization
- [ ] No method is in the wrong section

---

### Gaussian Distribution ✅ **BEST EXAMPLE** - **PHASE 1 COMPLETE**

**Current Issues:**
- ✅ **RESOLVED**: Added 3 missing header sections - now has complete 24 sections  
- ❌ Missing 14 sections from implementation
- ❌ 1 ordering issue between header and implementation
- ✅ **RESOLVED**: Private sections now correctly positioned at end of header

**Phase 1: Header Standardization** ✅ **COMPLETED**
- ✅ **DONE**: Added missing sections to reach 24 total sections
- ✅ **DONE**: Renamed "EXPONENTIAL-SPECIFIC UTILITY METHODS" to "DISTRIBUTION-SPECIFIC UTILITY METHODS"
- ✅ **DONE**: Reordered private sections to positions 17-24
- ✅ **DONE**: Verified section separators use exactly 78 `=` characters
- ✅ **DONE**: All sections now match template titles exactly
- ✅ **VERIFIED**: Compilation successful, all tests pass

**Phase 2: Implementation Alignment**
- [ ] Add 14 missing implementation sections to match header
- [ ] Fix ordering mismatch: ensure COMPARISON OPERATORS comes before PRIVATE methods
- [ ] Reorder all sections to match header positions 1-24
- [ ] Add proper section documentation comments
- [ ] Verify method placement within sections

**Phase 3: Method Organization**  
- [ ] Complete ADVANCED STATISTICAL METHODS implementation
- [ ] Add GOODNESS-OF-FIT TESTS implementation
- [ ] Implement CROSS-VALIDATION METHODS
- [ ] Add INFORMATION CRITERIA and BOOTSTRAP METHODS
- [ ] Organize batch operations properly in sections 13-14

**Verification Checklist**
- [ ] Header has exactly 24 sections in template order
- [ ] Implementation has exactly 24 matching sections  
- [ ] All ordering mismatches resolved
- [ ] Private sections correctly positioned at end
- [ ] Method placement follows logical organization

---

### Uniform Distribution  

**Current Issues:**
- ❌ Missing 3 header sections compared to template
- ❌ Missing 14 sections from implementation  
- ❌ **7 severe ordering issues** in implementation
- ❌ RESULT-BASED SETTERS completely out of order (pos 4→9)

**Phase 1: Header Standardization**
- [ ] Add missing sections to reach 24 total sections
- [ ] Rename "UNIFORM-SPECIFIC UTILITY METHODS" to match section 12 template  
- [ ] Standardize "FRIEND FUNCTIONS" to "FRIEND FUNCTION STREAM OPERATORS"
- [ ] Reorder private sections to positions 17-24
- [ ] Ensure all section titles match template exactly

**Phase 2: Implementation Alignment** ⚠️ **HIGH PRIORITY**
- [ ] **Fix critical ordering**: Move CORE PROBABILITY METHODS from pos 2 to pos 5
- [ ] **Fix critical ordering**: Move RESULT-BASED SETTERS from pos 9 to pos 4  
- [ ] **Fix critical ordering**: Move COMPARISON OPERATORS from pos 5 to pos 15
- [ ] Add 14 missing implementation sections
- [ ] Completely reorder implementation to match header positions 1-24
- [ ] Verify no methods are in wrong sections after reordering

**Phase 3: Method Organization**
- [ ] Implement missing ADVANCED STATISTICAL METHODS
- [ ] Add complete GOODNESS-OF-FIT TESTS
- [ ] Implement CROSS-VALIDATION and INFORMATION CRITERIA
- [ ] Add BOOTSTRAP METHODS implementation
- [ ] Organize "PRIVATE BATCH IMPLEMENTATION USING VECTOROPS" into section 18

**Verification Checklist**
- [ ] All 7 ordering mismatches completely resolved
- [ ] Header has exactly 24 sections in template order
- [ ] Implementation sections match header exactly
- [ ] RESULT-BASED SETTERS in correct position (4)
- [ ] All batch operations properly organized

---

### Poisson Distribution

**Current Issues:**  
- ❌ Has 24 sections but some incorrectly ordered
- ❌ **19 severe ordering issues** in implementation (worst case)
- ❌ Complex mathematical methods scattered across sections
- ❌ Specialized caching needs proper organization

**Phase 1: Header Standardization**
- [ ] Verify COMPUTATIONAL CACHE FOR SMALL LAMBDA fits in section 24
- [ ] Reorder header sections to match template exactly  
- [ ] Move private sections to correct positions 17-24
- [ ] Standardize "POISSON-SPECIFIC UTILITY METHODS" naming
- [ ] Check specialized mathematical sections are properly placed

**Phase 2: Implementation Alignment** ⚠️ **CRITICAL PRIORITY**
- [ ] **Major reordering required**: 19 ordering mismatches to resolve
- [ ] Move DISTRIBUTION MANAGEMENT from pos 10 to pos 6
- [ ] Move ADVANCED STATISTICAL METHODS from pos 13 to pos 7
- [ ] Move COMPARISON OPERATORS from pos 11 to pos 15  
- [ ] Completely restructure private method organization
- [ ] Ensure PRIVATE COMPUTATIONAL METHODS in section 19

**Phase 3: Method Organization**
- [ ] Organize discrete mathematics helpers in PRIVATE COMPUTATIONAL METHODS
- [ ] Ensure factorial computation methods are in specialized cache section
- [ ] Verify Poisson-specific algorithms are in correct utility section  
- [ ] Organize statistical test methods properly
- [ ] Confirm batch operations follow template organization

**Verification Checklist**
- [ ] All 19 ordering mismatches resolved
- [ ] Mathematical computation methods properly organized
- [ ] Specialized cache section correctly implemented  
- [ ] Implementation matches header section-by-section
- [ ] Complex discrete math methods logically grouped

---

### Discrete Distribution

**Current Issues:**
- ❌ Missing 2 header sections compared to template
- ❌ Private sections scattered throughout header (pos 17-22)
- ❌ Missing 15 sections from implementation
- ❌ 5 ordering issues in implementation
- ❌ "MISSING ADVANCED STATISTICAL METHODS" section indicates incomplete work

**Phase 1: Header Standardization** 
- [ ] Add 2 missing sections to reach 24 total
- [ ] **Major reordering**: Move private sections from pos 17-22 to pos 17-24
- [ ] Rename "DISCRETE-SPECIFIC UTILITY METHODS" to match section 12 template
- [ ] Ensure public sections 1-16 are correctly organized
- [ ] Standardize section titles to match template

**Phase 2: Implementation Alignment**
- [ ] Add 15 missing implementation sections
- [ ] Fix ordering: Move PARAMETER GETTERS AND SETTERS from pos 6 to pos 3
- [ ] Fix ordering: Move CORE PROBABILITY METHODS from pos 3 to pos 5
- [ ] Move ADVANCED STATISTICAL METHODS from pos 11 to pos 7
- [ ] **Complete missing methods**: Remove "MISSING ADVANCED STATISTICAL METHODS" section
- [ ] Reorder all sections to match header positions 1-24

**Phase 3: Method Organization**
- [ ] **Implement missing advanced statistical methods**
- [ ] Add discrete-specific probability mass function helpers
- [ ] Organize support validation methods properly
- [ ] Implement probability vector validation in appropriate section
- [ ] Complete goodness-of-fit tests for discrete distributions

**Verification Checklist**
- [ ] "MISSING ADVANCED STATISTICAL METHODS" section removed
- [ ] All missing functionality implemented  
- [ ] Private sections correctly positioned at end
- [ ] Implementation matches header organization
- [ ] Discrete-specific methods properly grouped

---

### Gamma Distribution

**Current Issues:**
- ❌ Has 23 sections, missing 1 from template
- ❌ Has redundant "PARAMETER SETTERS" section (duplication)
- ❌ Missing 18 sections from implementation
- ❌ Implementation uses "*IMPLEMENTATION" naming convention inconsistently
- ✅ Relatively clean ordering (best of all distributions)

**Phase 1: Header Standardization**
- [ ] Remove redundant "PARAMETER SETTERS" section (combine with section 3)
- [ ] Add 1 missing section to reach 24 total sections
- [ ] Standardize "GAMMA-SPECIFIC UTILITY METHODS" naming
- [ ] Verify all sections match template titles exactly
- [ ] Ensure consistent section separator formatting

**Phase 2: Implementation Alignment**
- [ ] Add 18 missing implementation sections
- [ ] Standardize inconsistent "*IMPLEMENTATION" section naming
- [ ] Match implementation sections exactly to header positions 1-24
- [ ] Remove "*IMPLEMENTATION" suffixes to match template
- [ ] Add proper section documentation

**Phase 3: Method Organization**
- [ ] Organize digamma function implementation in PRIVATE COMPUTATIONAL METHODS
- [ ] Ensure gamma-specific mathematical methods are properly grouped
- [ ] Verify shape/rate parameter methods are correctly organized
- [ ] Implement missing cross-validation methods
- [ ] Complete bootstrap methods implementation

**Verification Checklist**
- [ ] Redundant PARAMETER SETTERS section removed
- [ ] Implementation section naming standardized
- [ ] All 24 sections present and correctly ordered
- [ ] Digamma function properly placed in computational methods
- [ ] Implementation completeness matches other distributions

---

## Work Phases and Timeline

### Phase 1: Header Standardization (2-3 hours)
**Goal**: All distribution headers have exactly 24 sections in template order

**Tasks per distribution:**
1. Audit current sections against template (15 min)
2. Add missing sections with appropriate comments (15 min)  
3. Reorder sections to match template positions 1-24 (15 min)
4. Standardize section titles and formatting (10 min)
5. Verify completeness and correctness (5 min)

**Completion criteria:**
- [ ] All 6 distributions have exactly 24 header sections
- [ ] All sections follow template order precisely
- [ ] Section titles match template exactly
- [ ] Consistent formatting across all headers

### Phase 2: Implementation Alignment (3-4 hours)
**Goal**: All implementation files match their corresponding headers exactly

**Priority order** (address most problematic first):
1. **Poisson** (19 ordering issues) - 1 hour
2. **Uniform** (7 ordering issues) - 45 minutes  
3. **Discrete** (5 issues + missing methods) - 45 minutes
4. **Gaussian** (18 missing sections) - 30 minutes
5. **Exponential** (14 missing sections) - 30 minutes
6. **Gamma** (18 missing, cleanest ordering) - 30 minutes

**Tasks per distribution:**
1. Add missing implementation sections (10-20 min)
2. Reorder sections to match header positions (10-30 min)
3. Move misplaced methods to correct sections (10-20 min)
4. Add section documentation comments (5-10 min)
5. Verify method completeness (5-10 min)

**Completion criteria:**
- [ ] All implementation files have 24 sections matching headers
- [ ] No ordering mismatches between header and implementation
- [ ] All public methods implemented in correct sections
- [ ] Consistent section organization across all distributions

### Phase 3: Verification and Documentation (1 hour)
**Goal**: Confirm standardization is complete and create enforcement tools

**Tasks:**
1. Create automated verification script (20 min)
2. Run verification across all distributions (10 min)
3. Fix any remaining issues identified (20 min)
4. Update development guidelines (10 min)

**Completion criteria:**
- [ ] Automated verification passes for all distributions
- [ ] No standardization violations remain
- [ ] Future development guidelines updated
- [ ] Template documented for new distributions

---

## Verification Methods

### Automated Verification Script

Create `tools/verify_distribution_layout.py`:

```python
#!/usr/bin/env python3
"""
Verify that all distribution headers and implementations follow 
the standard 24-section template organization.
"""

REQUIRED_SECTIONS = [
    "CONSTRUCTORS AND DESTRUCTOR",
    "SAFE FACTORY METHODS (Exception-free construction)", 
    "PARAMETER GETTERS AND SETTERS",
    "RESULT-BASED SETTERS",
    "CORE PROBABILITY METHODS",
    "DISTRIBUTION MANAGEMENT", 
    "ADVANCED STATISTICAL METHODS",
    "GOODNESS-OF-FIT TESTS",
    "CROSS-VALIDATION METHODS",
    "INFORMATION CRITERIA", 
    "BOOTSTRAP METHODS",
    # Section 12 varies by distribution
    "SMART AUTO-DISPATCH BATCH OPERATIONS",
    "EXPLICIT STRATEGY BATCH OPERATIONS",
    "COMPARISON OPERATORS",
    "FRIEND FUNCTION STREAM OPERATORS",
    "PRIVATE FACTORY METHODS",
    "PRIVATE BATCH IMPLEMENTATION METHODS",
    "PRIVATE COMPUTATIONAL METHODS",
    "PRIVATE UTILITY METHODS", 
    "DISTRIBUTION PARAMETERS",
    "PERFORMANCE CACHE",
    "OPTIMIZATION FLAGS",
    "SPECIALIZED CACHES"
]

# Verify each distribution matches template...
```

### Manual Verification Checklist

**For each distribution, verify:**

- [ ] **Section count**: Exactly 24 sections in header
- [ ] **Section order**: Sections 1-16 public, 17-24 private  
- [ ] **Section titles**: Match template exactly (except section 12)
- [ ] **Implementation alignment**: Implementation sections match header
- [ ] **Method placement**: All methods in logically correct sections
- [ ] **Completeness**: All public header methods have implementations

### Progress Tracking

Use this checklist to track overall progress:

**Header Standardization Progress:**
- [x] Gaussian header standardized
- [x] Exponential header standardized
- [x] Uniform header standardized
- [ ] Poisson header standardized
- [x] Discrete header standardized
- [x] Gamma header standardized

**Implementation Alignment Progress:**
- [ ] Gaussian implementation aligned
- [ ] Exponential implementation aligned
- [ ] Uniform implementation aligned  
- [ ] Poisson implementation aligned
- [ ] Discrete implementation aligned
- [ ] Gamma implementation aligned

**Quality Verification:**
- [ ] Automated verification script created
- [ ] All distributions pass automated verification
- [ ] Manual spot-checks completed
- [ ] Documentation updated

---

## Success Metrics

### Quantitative Improvements
- **100% consistency**: All 6 distributions follow identical 24-section organization
- **Zero ordering issues**: No mismatches between header and implementation sections  
- **Complete coverage**: All public methods implemented in matching sections
- **Reduced cognitive load**: Developers can predict method locations across distributions

### Developer Experience Improvements
- **50% faster navigation**: Predictable method locations across all distributions
- **Consistent maintenance**: Similar changes can be applied systematically
- **Better code reviews**: Easier to spot missing implementations or misplaced methods
- **Improved onboarding**: New developers learn one pattern that applies everywhere

### Maintenance Benefits
- **Template-driven development**: New distributions can use standardized template
- **Automated verification**: Layout violations can be caught in CI/CD
- **Systematic refactoring**: Changes can be applied consistently across distributions
- **Clear expectations**: Section purpose and content are well-defined

---

## Future Maintenance

### For New Distributions
1. Copy the 24-section template exactly
2. Fill in distribution-specific content in appropriate sections
3. Run verification script before code review
4. Ensure implementation sections match header exactly

### For Existing Distribution Changes
1. Place new methods in appropriate sections based on functionality
2. Maintain section order when adding new capabilities
3. Update both header and implementation together
4. Run verification script after changes

### Enforcement
- Add layout verification to pre-commit hooks
- Include section organization in code review checklist  
- Run verification script in CI/CD pipeline
- Document violations and resolution steps

---

**Document Version**: 1.0  
**Created**: 2025-08-13  
**Purpose**: Guide distribution code layout standardization before v0.9.1  
**Expected Duration**: 6-8 hours total work  
**Priority**: High - blocks efficient v0.9.1 adaptive cache development
