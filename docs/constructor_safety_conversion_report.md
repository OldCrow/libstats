# Distribution Constructor Safety Conversion Report

This report identifies files using direct distribution constructors that should be converted to safe factory methods.

## Summary

**Total issues found:** 151

- **tools/:** 30 issues
- **tests/:** 121 issues

## Safe Factory Method Examples

The following safe patterns should be used instead of direct constructors:

### GaussianDistribution

**Safe alternatives:**
- `libstats::Gaussian(params)` - Type alias (uses safe construction internally)
- `GaussianDistribution::create(params)` - Exception-free factory method

**Examples of correct usage found:**
- quick_start_tutorial.cpp: libstats::Gaussian
- comparative_distributions_demo.cpp: libstats::Gaussian
- gaussian_performance_benchmark.cpp: libstats::Gaussian

### ExponentialDistribution

**Safe alternatives:**
- `libstats::Exponential(params)` - Type alias (uses safe construction internally)
- `ExponentialDistribution::create(params)` - Exception-free factory method

**Examples of correct usage found:**
- exponential_performance_benchmark.cpp: libstats::Exponential
- quick_start_tutorial.cpp: libstats::Exponential
- comparative_distributions_demo.cpp: libstats::Exponential

### UniformDistribution

**Safe alternatives:**
- `libstats::Uniform(params)` - Type alias (uses safe construction internally)
- `UniformDistribution::create(params)` - Exception-free factory method

**Examples of correct usage found:**
- quick_start_tutorial.cpp: libstats::Uniform
- comparative_distributions_demo.cpp: libstats::Uniform
- error_handling_demo.cpp: libstats::Uniform

### PoissonDistribution

**Safe alternatives:**
- `libstats::Poisson(params)` - Type alias (uses safe construction internally)
- `PoissonDistribution::create(params)` - Exception-free factory method

**Examples of correct usage found:**
- comparative_distributions_demo.cpp: libstats::Poisson
- error_handling_demo.cpp: libstats::Poisson
- error_handling_demo.cpp: PoissonDistribution::create

### GammaDistribution

**Safe alternatives:**
- `libstats::Gamma(params)` - Type alias (uses safe construction internally)
- `GammaDistribution::create(params)` - Exception-free factory method

### DiscreteDistribution

**Safe alternatives:**
- `libstats::Discrete(params)` - Type alias (uses safe construction internally)
- `DiscreteDistribution::create(params)` - Exception-free factory method

## Files Requiring Conversion

### tools/ Directory

#### `tools/simd_verification.cpp`

**Issues found:** 12

**Line 143:** `UniformDistribution`
```cpp
UniformDistribution dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 153:** `GaussianDistribution`
```cpp
GaussianDistribution dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 163:** `ExponentialDistribution`
```cpp
ExponentialDistribution dist(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 173:** `DiscreteDistribution`
```cpp
DiscreteDistribution dist(0, 10);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 183:** `PoissonDistribution`
```cpp
PoissonDistribution dist(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 193:** `GammaDistribution`
```cpp
GammaDistribution dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 387:** `UniformDistribution`
```cpp
{"Uniform", [this]() { testDistributionEdgeCases(UniformDistribution(0.0, 1.0), "Uniform"); }},
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 388:** `GaussianDistribution`
```cpp
{"Gaussian", [this]() { testDistributionEdgeCases(GaussianDistribution(0.0, 1.0), "Gaussian"); }},
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 389:** `ExponentialDistribution`
```cpp
{"Exponential", [this]() { testDistributionEdgeCases(ExponentialDistribution(1.0), "Exponential"); }},
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 390:** `DiscreteDistribution`
```cpp
{"Discrete", [this]() { testDistributionEdgeCases(DiscreteDistribution(0, 10), "Discrete"); }},
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 391:** `PoissonDistribution`
```cpp
{"Poisson", [this]() { testDistributionEdgeCases(PoissonDistribution(3.0), "Poisson"); }},
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 392:** `GammaDistribution`
```cpp
{"Gamma", [this]() { testDistributionEdgeCases(GammaDistribution(2.0, 1.0), "Gamma"); }}
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

#### `tools/parallel_threshold_benchmark.cpp`

**Issues found:** 6

**Line 143:** `UniformDistribution`
```cpp
UniformDistribution uniform(distribution_params::UNIFORM_MIN, distribution_params::UNIFORM_MAX);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 175:** `PoissonDistribution`
```cpp
PoissonDistribution poisson(distribution_params::DEFAULT_POISSON_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 207:** `DiscreteDistribution`
```cpp
DiscreteDistribution discrete(distribution_params::DISCRETE_MIN, distribution_params::DISCRETE_MAX);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 239:** `GaussianDistribution`
```cpp
GaussianDistribution gaussian(distribution_params::GAUSSIAN_MEAN, distribution_params::GAUSSIAN_STDDEV);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 271:** `ExponentialDistribution`
```cpp
ExponentialDistribution exponential(distribution_params::EXPONENTIAL_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 303:** `GammaDistribution`
```cpp
GammaDistribution gamma(distribution_params::GAMMA_ALPHA, distribution_params::GAMMA_BETA);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

#### `tools/learning_analyzer.cpp`

**Issues found:** 12

**Line 437:** `UniformDistribution`
```cpp
UniformDistribution uniform_dist(distribution_params::UNIFORM_MIN, distribution_params::UNIFORM_MAX);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 442:** `GaussianDistribution`
```cpp
GaussianDistribution gaussian_dist(distribution_params::GAUSSIAN_MEAN, distribution_params::GAUSSIAN_STDDEV);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 447:** `ExponentialDistribution`
```cpp
ExponentialDistribution exp_dist(distribution_params::EXPONENTIAL_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 452:** `DiscreteDistribution`
```cpp
DiscreteDistribution disc_dist(distribution_params::DISCRETE_MIN, distribution_params::DISCRETE_MAX);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 457:** `PoissonDistribution`
```cpp
PoissonDistribution poisson_dist(distribution_params::POISSON_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 462:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(distribution_params::GAMMA_ALPHA, distribution_params::GAMMA_BETA);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 565:** `UniformDistribution`
```cpp
UniformDistribution uniform_dist(distribution_params::UNIFORM_MIN, distribution_params::UNIFORM_MAX);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 573:** `GaussianDistribution`
```cpp
GaussianDistribution gaussian_dist(distribution_params::GAUSSIAN_MEAN, distribution_params::GAUSSIAN_STDDEV);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 581:** `ExponentialDistribution`
```cpp
ExponentialDistribution exp_dist(distribution_params::EXPONENTIAL_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 589:** `DiscreteDistribution`
```cpp
DiscreteDistribution disc_dist(distribution_params::DISCRETE_MIN, distribution_params::DISCRETE_MAX);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 597:** `PoissonDistribution`
```cpp
PoissonDistribution poisson_dist(distribution_params::POISSON_LAMBDA);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 605:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(distribution_params::GAMMA_ALPHA, distribution_params::GAMMA_BETA);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

### tests/ Directory

#### `tests/test_gamma_basic.cpp`

**Issues found:** 10

**Line 24:** `GammaDistribution`
```cpp
GammaDistribution param_gamma(2.0, 3.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 29:** `GammaDistribution`
```cpp
GammaDistribution copy_gamma(param_gamma);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 34:** `GammaDistribution`
```cpp
GammaDistribution temp_gamma(5.0, 0.5);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 35:** `GammaDistribution`
```cpp
GammaDistribution move_gamma(std::move(temp_gamma));
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 56:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 103:** `GammaDistribution`
```cpp
GammaDistribution test_gamma(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 186:** `GammaDistribution`
```cpp
GammaDistribution test_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 321:** `GammaDistribution`
```cpp
GammaDistribution dist1(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 322:** `GammaDistribution`
```cpp
GammaDistribution dist2(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 323:** `GammaDistribution`
```cpp
GammaDistribution dist3(3.0, 2.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

#### `tests/test_discrete_basic.cpp`

**Issues found:** 10

**Line 24:** `DiscreteDistribution`
```cpp
DiscreteDistribution param_discrete(0, 1);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 29:** `DiscreteDistribution`
```cpp
DiscreteDistribution copy_discrete(param_discrete);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 34:** `DiscreteDistribution`
```cpp
DiscreteDistribution temp_discrete(10, 15);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 35:** `DiscreteDistribution`
```cpp
DiscreteDistribution move_discrete(std::move(temp_discrete));
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 56:** `DiscreteDistribution`
```cpp
DiscreteDistribution discrete_dist(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 103:** `DiscreteDistribution`
```cpp
DiscreteDistribution dice_dist(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 186:** `DiscreteDistribution`
```cpp
DiscreteDistribution test_dist(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 321:** `DiscreteDistribution`
```cpp
DiscreteDistribution dist1(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 322:** `DiscreteDistribution`
```cpp
DiscreteDistribution dist2(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 323:** `DiscreteDistribution`
```cpp
DiscreteDistribution dist3(0, 10);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

#### `tests/test_exponential_basic.cpp`

**Issues found:** 10

**Line 23:** `ExponentialDistribution`
```cpp
ExponentialDistribution param_exp(2.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 27:** `ExponentialDistribution`
```cpp
ExponentialDistribution copy_exp(param_exp);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 31:** `ExponentialDistribution`
```cpp
ExponentialDistribution temp_exp(0.5);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 32:** `ExponentialDistribution`
```cpp
ExponentialDistribution move_exp(std::move(temp_exp));
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 51:** `ExponentialDistribution`
```cpp
ExponentialDistribution exp_dist(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 83:** `ExponentialDistribution`
```cpp
ExponentialDistribution test_exp(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 164:** `ExponentialDistribution`
```cpp
ExponentialDistribution test_dist(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 299:** `ExponentialDistribution`
```cpp
ExponentialDistribution dist1(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 300:** `ExponentialDistribution`
```cpp
ExponentialDistribution dist2(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 301:** `ExponentialDistribution`
```cpp
ExponentialDistribution dist3(2.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

#### `tests/test_uniform_enhanced.cpp`

**Issues found:** 9

**Line 44:** `UniformDistribution`
```cpp
test_distribution_ = UniformDistribution(test_a_, test_b_);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 60:** `UniformDistribution`
```cpp
UniformDistribution stdUniform(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 77:** `UniformDistribution`
```cpp
UniformDistribution custom(-2.0, 4.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 228:** `UniformDistribution`
```cpp
UniformDistribution stdUniform(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 371:** `UniformDistribution`
```cpp
UniformDistribution uniform_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 407:** `UniformDistribution`
```cpp
UniformDistribution new_dist(0.5, 1.5);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 426:** `UniformDistribution`
```cpp
UniformDistribution uniform_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 525:** `UniformDistribution`
```cpp
UniformDistribution stdUniform(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 690:** `UniformDistribution`
```cpp
UniformDistribution uniform(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

#### `tests/test_gamma_enhanced.cpp`

**Issues found:** 8

**Line 27:** `GammaDistribution`
```cpp
test_distribution_ = GammaDistribution(test_alpha_, test_beta_);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 42:** `GammaDistribution`
```cpp
GammaDistribution gamma1(2.0, 1.0);  // shape=2, rate=1 -> mean=2, var=2
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 52:** `GammaDistribution`
```cpp
GammaDistribution gamma2(1.0, 0.5);  // shape=1, rate=0.5 -> mean=2, var=4 (exponential)
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 325:** `GammaDistribution`
```cpp
GammaDistribution stdGamma(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 440:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 511:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 566:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

**Line 733:** `GammaDistribution`
```cpp
GammaDistribution gamma_dist(2.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gamma(...)`
- `GammaDistribution::create(...)`

#### `tests/test_dynamic_linking.cpp`

**Issues found:** 1

**Line 11:** `GaussianDistribution`
```cpp
GaussianDistribution normal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

#### `tests/test_uniform_basic.cpp`

**Issues found:** 10

**Line 24:** `UniformDistribution`
```cpp
UniformDistribution param_uniform(2.0, 5.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 29:** `UniformDistribution`
```cpp
UniformDistribution copy_uniform(param_uniform);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 34:** `UniformDistribution`
```cpp
UniformDistribution temp_uniform(-1.0, 3.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 35:** `UniformDistribution`
```cpp
UniformDistribution move_uniform(std::move(temp_uniform));
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 56:** `UniformDistribution`
```cpp
UniformDistribution uniform_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 102:** `UniformDistribution`
```cpp
UniformDistribution test_uniform(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 187:** `UniformDistribution`
```cpp
UniformDistribution test_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 322:** `UniformDistribution`
```cpp
UniformDistribution dist1(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 323:** `UniformDistribution`
```cpp
UniformDistribution dist2(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

**Line 324:** `UniformDistribution`
```cpp
UniformDistribution dist3(2.0, 5.0);
```
**Recommended alternatives:**
- `libstats::Uniform(...)`
- `UniformDistribution::create(...)`

#### `tests/test_copy_move_stress.cpp`

**Issues found:** 2

**Line 70:** `GaussianDistribution`
```cpp
GaussianDistribution gauss1(thread_id, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 71:** `GaussianDistribution`
```cpp
GaussianDistribution gauss2(thread_id + 10, 2.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

#### `tests/test_simd_integration_simple.cpp`

**Issues found:** 1

**Line 73:** `GaussianDistribution`
```cpp
GaussianDistribution normal(0.0, 1.0);  // Standard normal
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

#### `tests/test_discrete_enhanced.cpp`

**Issues found:** 9

**Line 35:** `DiscreteDistribution`
```cpp
test_distribution_ = DiscreteDistribution(test_lower_, test_upper_);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 51:** `DiscreteDistribution`
```cpp
DiscreteDistribution dice(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 69:** `DiscreteDistribution`
```cpp
DiscreteDistribution binary(0, 1);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 227:** `DiscreteDistribution`
```cpp
DiscreteDistribution stdDiscrete(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 360:** `DiscreteDistribution`
```cpp
DiscreteDistribution discrete_dist(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 401:** `DiscreteDistribution`
```cpp
DiscreteDistribution new_dist(2, 8);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 420:** `DiscreteDistribution`
```cpp
DiscreteDistribution discrete_dist(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 514:** `DiscreteDistribution`
```cpp
DiscreteDistribution dice(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

**Line 678:** `DiscreteDistribution`
```cpp
DiscreteDistribution dice(1, 6);
```
**Recommended alternatives:**
- `libstats::Discrete(...)`
- `DiscreteDistribution::create(...)`

#### `tests/test_poisson_basic.cpp`

**Issues found:** 10

**Line 23:** `PoissonDistribution`
```cpp
PoissonDistribution param_poisson(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 27:** `PoissonDistribution`
```cpp
PoissonDistribution copy_poisson(param_poisson);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 31:** `PoissonDistribution`
```cpp
PoissonDistribution temp_poisson(5.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 32:** `PoissonDistribution`
```cpp
PoissonDistribution move_poisson(std::move(temp_poisson));
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 51:** `PoissonDistribution`
```cpp
PoissonDistribution poisson_dist(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 84:** `PoissonDistribution`
```cpp
PoissonDistribution test_poisson(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 172:** `PoissonDistribution`
```cpp
PoissonDistribution test_dist(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 307:** `PoissonDistribution`
```cpp
PoissonDistribution dist1(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 308:** `PoissonDistribution`
```cpp
PoissonDistribution dist2(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 309:** `PoissonDistribution`
```cpp
PoissonDistribution dist3(5.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

#### `tests/test_exponential_enhanced.cpp`

**Issues found:** 9

**Line 37:** `ExponentialDistribution`
```cpp
test_distribution_ = ExponentialDistribution(test_lambda_);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 52:** `ExponentialDistribution`
```cpp
ExponentialDistribution unitExp(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 68:** `ExponentialDistribution`
```cpp
ExponentialDistribution custom(2.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 217:** `ExponentialDistribution`
```cpp
ExponentialDistribution stdExp(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 363:** `ExponentialDistribution`
```cpp
ExponentialDistribution exp_dist(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 399:** `ExponentialDistribution`
```cpp
ExponentialDistribution new_dist(2.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 418:** `ExponentialDistribution`
```cpp
ExponentialDistribution exp_dist(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 512:** `ExponentialDistribution`
```cpp
ExponentialDistribution unitExp(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

**Line 676:** `ExponentialDistribution`
```cpp
ExponentialDistribution unitExp(1.0);
```
**Recommended alternatives:**
- `libstats::Exponential(...)`
- `ExponentialDistribution::create(...)`

#### `tests/test_gaussian_enhanced.cpp`

**Issues found:** 8

**Line 33:** `GaussianDistribution`
```cpp
test_distribution_ = GaussianDistribution(test_mean_, test_std_);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 49:** `GaussianDistribution`
```cpp
GaussianDistribution stdNormal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 65:** `GaussianDistribution`
```cpp
GaussianDistribution custom(5.0, 2.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 262:** `GaussianDistribution`
```cpp
GaussianDistribution stdNormal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 358:** `GaussianDistribution`
```cpp
GaussianDistribution gauss_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 432:** `GaussianDistribution`
```cpp
GaussianDistribution gauss_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 487:** `GaussianDistribution`
```cpp
GaussianDistribution stdNormal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 618:** `GaussianDistribution`
```cpp
GaussianDistribution normal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

#### `tests/test_gaussian_basic.cpp`

**Issues found:** 10

**Line 24:** `GaussianDistribution`
```cpp
GaussianDistribution param_gauss(5.0, 2.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 29:** `GaussianDistribution`
```cpp
GaussianDistribution copy_gauss(param_gauss);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 34:** `GaussianDistribution`
```cpp
GaussianDistribution temp_gauss(10.0, 3.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 35:** `GaussianDistribution`
```cpp
GaussianDistribution move_gauss(std::move(temp_gauss));
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 56:** `GaussianDistribution`
```cpp
GaussianDistribution gauss_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 101:** `GaussianDistribution`
```cpp
GaussianDistribution std_normal(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 183:** `GaussianDistribution`
```cpp
GaussianDistribution test_dist(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 318:** `GaussianDistribution`
```cpp
GaussianDistribution dist1(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 319:** `GaussianDistribution`
```cpp
GaussianDistribution dist2(0.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 320:** `GaussianDistribution`
```cpp
GaussianDistribution dist3(1.0, 2.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

#### `tests/test_poisson_enhanced.cpp`

**Issues found:** 9

**Line 44:** `PoissonDistribution`
```cpp
test_distribution_ = PoissonDistribution(test_lambda_);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 59:** `PoissonDistribution`
```cpp
PoissonDistribution stdPoisson(1.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 77:** `PoissonDistribution`
```cpp
PoissonDistribution custom(5.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 217:** `PoissonDistribution`
```cpp
PoissonDistribution stdPoisson(2.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 358:** `PoissonDistribution`
```cpp
PoissonDistribution poisson_dist(4.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 394:** `PoissonDistribution`
```cpp
PoissonDistribution new_dist(8.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 413:** `PoissonDistribution`
```cpp
PoissonDistribution poisson_dist(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 507:** `PoissonDistribution`
```cpp
PoissonDistribution stdPoisson(3.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

**Line 671:** `PoissonDistribution`
```cpp
PoissonDistribution poisson(5.0);
```
**Recommended alternatives:**
- `libstats::Poisson(...)`
- `PoissonDistribution::create(...)`

#### `tests/test_copy_move_fix.cpp`

**Issues found:** 5

**Line 56:** `GaussianDistribution`
```cpp
GaussianDistribution gauss1(1.0, 2.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 57:** `GaussianDistribution`
```cpp
GaussianDistribution gauss2(5.0, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 68:** `GaussianDistribution`
```cpp
GaussianDistribution gauss3(3.0, 4.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 69:** `GaussianDistribution`
```cpp
GaussianDistribution gauss4(7.0, 9.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

**Line 130:** `GaussianDistribution`
```cpp
GaussianDistribution gauss(t, 1.0);
```
**Recommended alternatives:**
- `libstats::Gaussian(...)`
- `GaussianDistribution::create(...)`

