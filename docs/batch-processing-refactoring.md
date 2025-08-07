# Batch Processing API Refactoring

## Overview

This document tracks the comprehensive refactoring of libstats' batch processing API to address the "15 methods per distribution" problem while maintaining performance and flexibility for power users.

## Current State (Before Refactoring)

### Existing API Surface per Distribution
- 1 auto-dispatch method (`getProbability(span, span, hint)`)
- 3 SIMD batch methods (`getProbabilityBatch`, `getLogProbabilityBatch`, `getCumulativeProbabilityBatch`)
- 3 parallel methods (`getProbabilityBatchParallel`, etc.)
- 3 work-stealing methods (`getProbabilityBatchWorkStealing`, etc.)
- 3 cache-aware methods (`getProbabilityBatchCacheAware`, etc.)

**Total: 13 public methods per distribution class**

### Problems Identified
1. **API Complexity**: Too many public methods creating cognitive overhead
2. **Code Duplication**: Similar patterns repeated across all distributions
3. **Maintenance Burden**: Changes require updates to 13+ methods per distribution
4. **User Confusion**: Unclear when to use which method
5. **Template Complexity**: Current DispatchUtils has mixing of public/private access

## Target Architecture (Hybrid Solution)

### Design Principles
1. **Simplicity First**: Clean API for 95% of users
2. **Power User Access**: Explicit strategy selection for advanced users
3. **Performance Preservation**: No regression in optimized paths
4. **Maintainability**: Reduce code duplication without over-engineering
5. **Extensibility**: Easy to add new strategies (GPU, distributed, etc.)

### Proposed Public API

```cpp
class DiscreteDistribution {
public:
    // PRIMARY INTERFACE - Auto-dispatch based on hints
    void getProbability(std::span<const double> values, std::span<double> results,
                       const PerformanceHint& hint = {}) const;
    void getLogProbability(std::span<const double> values, std::span<double> results,
                          const PerformanceHint& hint = {}) const;
    void getCumulativeProbability(std::span<const double> values, std::span<double> results,
                                 const PerformanceHint& hint = {}) const;
    
    // POWER USER INTERFACE - Explicit strategy selection
    void getProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                   DispatchStrategy strategy) const;
    void getLogProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                      DispatchStrategy strategy) const;
    void getCumulativeProbabilityWithStrategy(std::span<const double> values, std::span<double> results,
                                             DispatchStrategy strategy) const;

private:
    // IMPLEMENTATION METHODS - Called by DispatchUtils
    void getProbabilityImpl(std::span<const double> values, std::span<double> results,
                           DispatchStrategy strategy) const;
    void getLogProbabilityImpl(std::span<const double> values, std::span<double> results,
                              DispatchStrategy strategy) const;
    void getCumulativeProbabilityImpl(std::span<const double> values, std::span<double> results,
                                     DispatchStrategy strategy) const;
    
    // LOW-LEVEL IMPLEMENTATIONS - Strategy-specific
    void getProbabilityBatch(const double* values, double* results, size_t count) const;
    void getProbabilityBatchParallel(std::span<const double> values, std::span<double> results) const;
    void getProbabilityBatchWorkStealing(std::span<const double> values, std::span<double> results,
                                        WorkStealingPool& pool) const;
    void getProbabilityBatchCacheAware(std::span<const double> values, std::span<double> results,
                                      cache::AdaptiveCache<std::string, double>& cache) const;
    // ... similar for LogProbability and CumulativeProbability
};
```

**Result: 6 public methods per distribution (down from 13)**

### DispatchStrategy Enum

```cpp
enum class DispatchStrategy {
    AUTO,           // Let DispatchUtils choose
    SCALAR,         // Simple loop for tiny batches
    SIMD_BATCH,     // Vectorized operations
    PARALLEL,       // Thread-based parallelism
    WORK_STEALING,  // Dynamic load balancing
    CACHE_AWARE     // Cache-optimized processing
};
```

### DispatchUtils Refactoring

```cpp
class DispatchUtils {
public:
    // PRIMARY AUTO-DISPATCH
    template<typename Distribution, typename ScalarFunc, typename BatchFunc, 
             typename ParallelFunc, typename WorkStealingFunc, typename CacheAwareFunc>
    static void autoDispatch(
        const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        const PerformanceHint& hint,
        DistributionType dist_type,
        ComputationalComplexity complexity,
        ScalarFunc&& scalar_func,
        BatchFunc&& batch_func,
        ParallelFunc&& parallel_func,
        WorkStealingFunc&& work_stealing_func,
        CacheAwareFunc&& cache_aware_func
    );
    
    // EXPLICIT STRATEGY EXECUTION
    template<typename Distribution, typename ScalarFunc, typename BatchFunc,
             typename ParallelFunc, typename WorkStealingFunc, typename CacheAwareFunc>
    static void executeWithStrategy(
        const Distribution& dist,
        std::span<const double> values,
        std::span<double> results,
        DispatchStrategy strategy,
        ScalarFunc&& scalar_func,
        BatchFunc&& batch_func,
        ParallelFunc&& parallel_func,
        WorkStealingFunc&& work_stealing_func,
        CacheAwareFunc&& cache_aware_func
    );

private:
    // STRATEGY SELECTION LOGIC
    static DispatchStrategy selectOptimalStrategy(
        size_t count,
        const PerformanceHint& hint,
        DistributionType dist_type,
        ComputationalComplexity complexity
    );
    
    // STRATEGY EXECUTION HELPERS
    template<typename Distribution, typename Func>
    static void executeScalar(const Distribution& dist, std::span<const double> values,
                             std::span<double> results, Func&& func);
    
    template<typename Distribution, typename Func>
    static void executeParallel(const Distribution& dist, std::span<const double> values,
                               std::span<double> results, Func&& func);
    // ... other execution helpers
};
```

### Base Class Integration (Limited)

```cpp
class DistributionBase {
protected:
    // STRATEGY SELECTION - Can be overridden for distribution-specific tuning
    virtual DispatchStrategy selectOptimalStrategy(size_t count, const PerformanceHint& hint) const {
        // Default implementation - can be overridden
        return DispatchUtils::selectOptimalStrategy(count, hint, getDistributionType(), getComplexity());
    }
    
    // DISTRIBUTION CHARACTERISTICS - Must be implemented by derived classes
    virtual DistributionType getDistributionType() const = 0;
    virtual ComputationalComplexity getComplexity() const = 0;
};
```

## Implementation Phases

### Phase 1: Core Infrastructure ✅ COMPLETED
- [x] Create DispatchStrategy enum
- [x] Implement basic DispatchUtils::autoDispatch
- [x] Test with DiscreteDistribution
- [x] Verify performance parity

**Status**: Completed during initial refactoring session

### Phase 2: API Cleanup 🚧 IN PROGRESS
- [ ] Add `*WithStrategy` methods to DiscreteDistribution
- [ ] Make existing batch methods private
- [ ] Update method signatures to match new patterns
- [ ] Fix compilation issues from private method access
- [ ] Update unit tests

**Current Issues**:
- Lambda capture problems with `this` pointer
- Private method access from DispatchUtils
- Need to implement direct parallel/work-stealing/cache-aware patterns

### Phase 3: Base Class Integration
- [ ] Add virtual methods to DistributionBase
- [ ] Move strategy selection logic to base class
- [ ] Implement getDistributionType() and getComplexity() in derived classes
- [ ] Test strategy selection consistency

### Phase 4: Extended Distribution Support
- [ ] Apply refactoring to NormalDistribution
- [ ] Apply refactoring to ExponentialDistribution
- [ ] Apply refactoring to GammaDistribution
- [ ] Apply refactoring to BetaDistribution
- [ ] Verify consistency across all distributions

### Phase 5: Advanced Features
- [ ] Implement adaptive strategy selection based on profiling
- [ ] Add performance monitoring and metrics
- [ ] Implement strategy caching for repeated patterns
- [ ] Add GPU strategy preparation (placeholder)

### Phase 6: Documentation and Testing
- [ ] Update API documentation
- [ ] Create migration guide for existing code
- [ ] Add comprehensive benchmarks
- [ ] Performance regression testing
- [ ] Update examples and tutorials

## Technical Decisions Made

### 1. Why Hybrid Approach Over Pure Template Solution?
- **Type Safety**: Avoids complex CRTP machinery and potential compilation issues
- **Flexibility**: Each distribution can have specific optimizations
- **Maintainability**: Clear separation between public API and implementation
- **Performance**: No template instantiation overhead in hot paths

### 2. Why Keep Strategy-Specific Methods Private?
- **Encapsulation**: Implementation details hidden from users
- **API Stability**: Can change implementation without breaking user code
- **Testing**: Internal methods can still be tested through friend classes
- **Power Users**: Still accessible via `*WithStrategy` methods

### 3. Why Limited Base Class Integration?
- **Avoiding Over-engineering**: Only move truly common functionality
- **Type Preservation**: Avoid type masking in template hierarchies
- **Performance**: No virtual call overhead in computation paths
- **Flexibility**: Distributions can override strategy selection

## Performance Requirements

### Non-Negotiable Requirements
1. **Zero Regression**: New API must not be slower than current implementation
2. **Memory Efficiency**: No additional allocations in hot paths
3. **Cache Friendly**: Preserve existing cache optimization strategies
4. **SIMD Preservation**: Vectorized paths must remain vectorized

### Benchmarking Plan
1. **Micro-benchmarks**: Individual method performance
2. **Macro-benchmarks**: Real-world usage patterns
3. **Memory Profiling**: Allocation patterns and cache behavior
4. **Scaling Tests**: Performance across different batch sizes

## Migration Strategy

### For Library Users
1. **Backward Compatibility**: Existing auto-dispatch calls continue to work
2. **Deprecation Period**: Old methods marked deprecated with migration guidance
3. **Migration Tools**: Scripts to automatically update common patterns
4. **Documentation**: Clear migration examples

### For Library Developers
1. **Staged Rollout**: Apply to one distribution at a time
2. **Comprehensive Testing**: Each phase fully tested before proceeding
3. **Performance Validation**: Benchmarks run after each major change
4. **Code Review**: Architectural decisions reviewed by team

## Risk Mitigation

### Technical Risks
1. **Performance Regression**: Mitigated by comprehensive benchmarking
2. **Compilation Issues**: Addressed through incremental testing
3. **Template Complexity**: Avoided through hybrid approach
4. **Breaking Changes**: Minimized through careful API design

### Project Risks
1. **Scope Creep**: Controlled through phased approach
2. **Timeline**: Tracked through this document and regular reviews
3. **Quality**: Maintained through extensive testing at each phase

## Success Metrics

### Quantitative Goals
- **API Simplicity**: Reduce public methods from 13 to 6 per distribution
- **Performance**: Zero regression in optimized paths
- **Code Duplication**: Reduce by 60% through shared patterns
- **Build Time**: No significant increase in compilation time

### Qualitative Goals
- **Developer Experience**: Easier to add new distributions
- **User Experience**: Clearer API with better discoverability
- **Maintainability**: Easier to add new strategies and optimizations
- **Extensibility**: Clear path for GPU and distributed strategies

## Current Status

**Overall Progress**: 20% Complete (Phase 1 done, Phase 2 in progress)

**Next Steps**:
1. Fix compilation issues in DiscreteDistribution
2. Implement direct parallel/work-stealing/cache-aware patterns
3. Add `*WithStrategy` methods
4. Update unit tests

**Blockers**:
- Lambda capture issues with `this` pointer
- Need to resolve private method access from DispatchUtils

**Last Updated**: January 2024
**Document Version**: 1.0
**Authors**: Development Team
