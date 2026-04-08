# Academic References

This document lists academic sources and algorithms used in libstats implementations. These are algorithmic references for transparency and reproducibility, not third-party code requiring attribution.

## Mathematical Functions

### Error Function and Inverse (`math_utils.cpp`)
- **Inverse Error Function**: Rational approximation based on methods described in:
  - Press, W. H., et al. "Numerical Recipes in C++" (various editions)
  - Moro's method for rational approximations
  - Acklam's refined coefficients for tail regions

### Gamma Functions (`math_utils.cpp`)
- **Incomplete Gamma Functions**: Series expansion and continued fraction methods from:
  - Press, W. H., et al. "Numerical Recipes: The Art of Scientific Computing"
  - NIST Digital Library of Mathematical Functions

### Beta Functions (`math_utils.cpp`)
- **Incomplete Beta Function**: Continued fraction approximation from:
  - Press, W. H., et al. "Numerical Recipes in C++"

## Statistical Distributions

### General Methods
- **Maximum Likelihood Estimation**: Standard statistical inference techniques
- **Goodness-of-Fit Tests**: Classical statistical tests (Kolmogorov-Smirnov, Anderson-Darling, Chi-squared)

### Poisson Distribution (`poisson.cpp`)
- **Knuth's Algorithm**: For small lambda values
- **Transformed rejection method**: For large lambda values
- **Stirling's Approximation**: For factorial calculations in large parameter regime

### Gamma Distribution (`gamma.cpp`)
- **Marsaglia and Tsang Method**: Efficient sampling for shape > 1
- **Transformation methods**: For shape < 1

## Numerical Methods

### Special Functions (`simd_avx.cpp`)
- **Polynomial Approximations**: Coefficients derived from:
  - Abramowitz, M. and Stegun, I. A. (1964). "Handbook of Mathematical Functions"
  - SLEEF library (properly attributed in THIRD_PARTY_NOTICES.md)

### Optimization Techniques
- **SIMD Vectorization**: Modern CPU optimization techniques
- **Cache-aware algorithms**: Standard performance optimization patterns

## Notes

- All implementations are original code written for libstats
- Mathematical formulas and algorithms are implemented from first principles or academic descriptions
- No code was copied from the referenced sources (except SLEEF-inspired SIMD code, which is properly attributed)
- These references are provided for academic integrity and to help users understand the theoretical foundations
