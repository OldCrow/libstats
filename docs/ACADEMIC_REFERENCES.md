# Academic and Algorithm References

This document lists algorithmic references used by libstats.

## Numerical functions and SIMD

- SLEEF Project. Vectorised elementary function implementation techniques for exp, log, and trigonometric functions.
- musl libc. High-accuracy `erf` rational approximation strategy used as inspiration for x86 vector erf paths.
- Agner Fog. *Optimizing software in C++* and instruction tables for SIMD performance analysis.
- Abramowitz, M. and Stegun, I. A. (1964). *Handbook of Mathematical Functions*. Used for several classical approximations and special-function references.

## Statistical inference

- Hosking, J. R. M. (1990). "L-moments: Analysis and estimation of distributions using linear combinations of order statistics." *Journal of the Royal Statistical Society: Series B*, 52(1), 105–124.
- Royston, P. (1992). "Approximating the Shapiro-Wilk W-test for non-normality." *Statistics and Computing*, 2, 117–119.
- Jarque, C. M. and Bera, A. K. (1980). "Efficient tests for normality, homoscedasticity and serial independence of regression residuals." *Economics Letters*, 6(3), 255–259.
- Garwood, F. (1936). "Fiducial limits for the Poisson distribution." *Biometrika*, 28(3/4), 437–442.
- Clopper, C. J. and Pearson, E. S. (1934). "The use of confidence or fiducial limits illustrated in the case of the binomial." *Biometrika*, 26(4), 404–413.
- Wilson, E. B. (1927). "Probable inference, the law of succession, and statistical inference." *Journal of the American Statistical Association*, 22(158), 209–212.

## Distribution algorithms

- Devroye, L. (1986). *Non-Uniform Random Variate Generation*. Springer.
- Johnson, N. L., Kotz, S., and Balakrishnan, N. (1994–1997). *Continuous Univariate Distributions* and *Discrete Multivariate Distributions*.
- Press, W. H. et al. (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd ed.

## Parallelism and dispatch

- Blumofe, R. D. and Leiserson, C. E. (1999). "Scheduling multithreaded computations by work stealing." *Journal of the ACM*, 46(5), 720–748.
- Intel. *Intel 64 and IA-32 Architectures Optimization Reference Manual*.
- Arm. *Arm Architecture Reference Manual* and NEON programmer guidance.
