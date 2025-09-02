# Distribution Mutability and Cache Invalidation

## Overview

All distributions in the libstats library are **mutable** - they support parameter modification after construction through setter methods. When parameters are modified, internal caches are automatically invalidated to ensure consistency.

## Distribution Setter Methods

### 1. Gamma Distribution
- `setAlpha(double alpha)` - Set shape parameter
- `setBeta(double beta)` - Set rate parameter
- `setParameters(double alpha, double beta)` - Set both parameters

### 2. Gaussian Distribution
- `setMean(double mean)` - Set mean parameter
- `setStandardDeviation(double stdDev)` - Set standard deviation
- `setParameters(double mean, double stdDev)` - Set both parameters

### 3. Exponential Distribution
- `setLambda(double lambda)` - Set rate parameter
- `setParameters(double lambda)` - Set rate parameter (same as setLambda)

### 4. Poisson Distribution
- `setLambda(double lambda)` - Set rate parameter
- `setParameters(double lambda)` - Set rate parameter (same as setLambda)

### 5. Uniform Distribution
- `setLowerBound(double a)` - Set lower bound
- `setUpperBound(double b)` - Set upper bound
- `setBounds(double a, double b)` - Set both bounds
- `setParameters(double a, double b)` - Set both bounds (same as setBounds)

### 6. Discrete Uniform Distribution
- `setLowerBound(int a)` - Set lower bound
- `setUpperBound(int b)` - Set upper bound
- `setBounds(int a, int b)` - Set both bounds
- `setParameters(int a, int b)` - Set both bounds (same as setBounds)

## Cache Invalidation

All distributions implement automatic cache invalidation when parameters are modified:

1. **Thread-Safe Updates**: All setter methods acquire exclusive locks before modification
2. **Cache Invalidation**: Internal caches for computed statistics (mean, variance, skewness, kurtosis) are automatically invalidated
3. **Atomic Parameter Updates**: Parameters are updated atomically to ensure thread-safe access

### Example Usage

```cpp
// Create a distribution
auto result = stats::GammaDistribution::create(2.0, 1.0);
if (result.isOk()) {
    auto gamma = std::move(result.value);

    // First access - calculates and caches statistics
    double mean1 = gamma.getMean();  // Returns 2.0

    // Modify parameters - automatically invalidates cache
    gamma.setAlpha(3.0);

    // Next access - recalculates with new parameters
    double mean2 = gamma.getMean();  // Returns 3.0
}
```

## Performance Considerations

### Caching Benefits
- **First Access**: Computes and caches statistical properties
- **Subsequent Access**: Returns cached values (near-instant)
- **After Modification**: Recomputes on next access

### Thread Safety
- All getters use shared locks (multiple readers allowed)
- All setters use exclusive locks (single writer)
- Atomic parameters provide lock-free fast path for high-frequency access

### Best Practices
1. **Batch Parameter Updates**: Use `setParameters()` methods when changing multiple parameters
2. **Minimize Modifications**: Cache invalidation has a cost - avoid frequent parameter changes in performance-critical loops
3. **Consider Immutability**: If parameters don't need to change, create new distribution instances instead

## Testing Cache Invalidation

The enhanced test suite verifies cache invalidation for all distributions:

```cpp
// Example from test_gamma_enhanced.cpp
TEST_F(GammaEnhancedTest, CachingSpeedupVerification) {
    auto gamma_dist = stats::GammaDistribution::create(2.0, 1.0).value;

    // Verify cache provides speedup
    double mean_before = gamma_dist.getMean();  // Cache miss
    double mean_cached = gamma_dist.getMean();  // Cache hit (faster)

    // Test cache invalidation
    gamma_dist.setAlpha(3.0);  // Invalidates cache
    double mean_after = gamma_dist.getMean();  // Cache miss, new value

    EXPECT_EQ(mean_after, 3.0 / gamma_dist.getBeta());
}
```

## Design Rationale

### Why Mutable?
1. **Efficiency**: Avoids object creation overhead when parameters change
2. **Flexibility**: Allows parameter optimization and fitting algorithms
3. **Cache Optimization**: Internal caching provides performance benefits for repeated access

### Alternative: Factory Methods
For scenarios requiring immutability, use factory methods to create new instances:

```cpp
auto dist1 = stats::GammaDistribution::create(2.0, 1.0).value;
auto dist2 = stats::GammaDistribution::create(3.0, 1.0).value;
// dist1 and dist2 are independent instances
```

## Summary

All distributions in libstats are mutable with:
- ✅ Parameter setter methods for all distributions
- ✅ Automatic cache invalidation on parameter changes
- ✅ Thread-safe parameter updates with proper locking
- ✅ Atomic parameter access for high-performance scenarios
- ✅ Comprehensive test coverage for cache invalidation

This design provides a balance between performance (through caching) and flexibility (through mutability) while maintaining thread safety.
