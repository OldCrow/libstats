# API Simplification Roadmap for v1.0.0
## Intelligent Auto-Dispatch Performance System

**Document Version:** 1.0  
**Target Release:** v1.0.0  
**Author:** Development Team  
**Date:** 2025-01-24  
**Status:** IN PROGRESS - Optimization Phase

---

## 🎯 Executive Summary

This document outlines the ongoing efforts to simplify the libstats API as part of the optimization phase. The goal is to implement an intelligent auto-dispatch system that automatically selects optimal performance strategies based on problem characteristics. This optimization step will reduce API complexity by 60-70% while maintaining or improving performance and must be completed before the project fully transitions to version 1.0.0.

### Key Goals:
- **Reduce API complexity** from 17 methods per distribution to 6-9 methods
- **Eliminate performance decision burden** from users
- **Implement adaptive learning** system that improves over time
- **Maintain backward compatibility** with existing code
- **Create foundation** for future GPU acceleration

---

## 📊 Current State Analysis

### Current API Complexity (Per Distribution)
The current API exposes **17 different probability calculation methods**:

```cpp
// Single-value methods (3)
double getProbability(double x);
double getLogProbability(double x);
double getCumulativeProbability(double x);

// SIMD batch methods (3)
void getProbabilityBatch(const double*, double*, size_t);
void getLogProbabilityBatch(const double*, double*, size_t);
void getCumulativeProbabilityBatch(const double*, double*, size_t);

// Unsafe batch methods (2)
void getProbabilityBatchUnsafe(const double*, double*, size_t);
void getLogProbabilityBatchUnsafe(const double*, double*, size_t);

// Parallel batch methods (3)
void getProbabilityBatchParallel(span<const double>, span<double>);
void getLogProbabilityBatchParallel(span<const double>, span<double>);
void getCumulativeProbabilityBatchParallel(span<const double>, span<double>);

// Work-stealing methods (3)
void getProbabilityBatchWorkStealing(span<const double>, span<double>, WorkStealingPool&);
void getLogProbabilityBatchWorkStealing(span<const double>, span<double>, WorkStealingPool&);
void getCumulativeProbabilityBatchWorkStealing(span<const double>, span<double>, WorkStealingPool&);

// Cache-aware methods (3)
void getProbabilityBatchCacheAware(span<const double>, span<double>, AdaptiveCache&);
void getLogProbabilityBatchCacheAware(span<const double>, span<double>, AdaptiveCache&);
void getCumulativeProbabilityBatchCacheAware(span<const double>, span<double>, AdaptiveCache&);
```

**Total Current API Surface: 17 methods × 4 distributions = 68 probability-related methods**

### Problems with Current Design:
1. **Overwhelming choice paralysis** - Users must understand performance characteristics
2. **Easy to choose wrong method** - Leads to suboptimal performance
3. **Maintenance burden** - 68 methods to document, test, and support
4. **Knowledge barrier** - Requires deep understanding of CPU features, threading, caching
5. **Future complexity** - Adding GPU support would create 85+ methods

---

## 🎨 Proposed Solution: Intelligent Auto-Dispatch

### New Simplified API
```cpp
// Single-value methods (3) - unchanged
double getProbability(double x);
double getLogProbability(double x);  
double getCumulativeProbability(double x);

// Smart batch methods (3) - auto-dispatch everything
void getProbability(span<const double> values, span<double> results);
void getLogProbability(span<const double> values, span<double> results);
void getCumulativeProbability(span<const double> values, span<double> results);

// Optional: Advanced control for power users (3)
void getProbability(span<const double> values, span<double> results, const PerformanceHint& hint);
void getLogProbability(span<const double> values, span<double> results, const PerformanceHint& hint);
void getCumulativeProbability(span<const double> values, span<double> results, const PerformanceHint& hint);
```

**New API Surface: 6-9 methods vs current 17 methods = 60-70% reduction in complexity**

---

## 🏗 Technical Architecture

### Core Components

#### 1. Performance Decision Engine
```cpp
class PerformanceDispatcher {
public:
    enum class Strategy {
        SCALAR,           // Single element or very small batches
        SIMD_BATCH,       // SIMD vectorized for medium batches
        PARALLEL_SIMD,    // Parallel + SIMD for large batches
        WORK_STEALING,    // Dynamic load balancing for irregular workloads
        CACHE_AWARE       // Cache-optimized for very large batches
    };
    
    Strategy selectOptimalStrategy(
        size_t batch_size,
        DistributionType dist_type,
        ComputationComplexity complexity,
        const SystemCapabilities& system,
        const PerformanceHistory& history
    ) const;
    
private:
    // Decision thresholds (learned from current benchmarks)
    struct Thresholds {
        size_t simd_min = 8;                    // SIMD overhead threshold
        size_t parallel_min = 1000;             // Threading overhead threshold
        size_t work_stealing_min = 10000;       // Work-stealing benefit threshold
        size_t cache_aware_min = 50000;         // Cache optimization threshold
        
        // Distribution-specific overrides
        size_t uniform_parallel_min = 65536;    // Simple operations need higher threshold
        size_t gaussian_parallel_min = 256;     // Complex operations benefit earlier
        size_t exponential_parallel_min = 512;  // Moderate complexity
        size_t discrete_parallel_min = 1024;    // Integer operations
    };
    
    mutable Thresholds thresholds_;
    mutable PerformanceProfiler profiler_;
};
```

#### 2. System Capabilities Detection
```cpp
class SystemCapabilities {
public:
    static const SystemCapabilities& current();
    
    // CPU characteristics
    size_t logical_cores() const { return logical_cores_; }
    size_t physical_cores() const { return physical_cores_; }
    size_t l1_cache_size() const { return l1_cache_size_; }
    size_t l2_cache_size() const { return l2_cache_size_; }
    size_t l3_cache_size() const { return l3_cache_size_; }
    
    // SIMD capabilities
    bool has_sse2() const { return has_sse2_; }
    bool has_avx() const { return has_avx_; }
    bool has_avx2() const { return has_avx2_; }
    bool has_avx512() const { return has_avx512_; }
    bool has_neon() const { return has_neon_; }
    
    // Threading capabilities
    bool has_std_execution() const { return has_std_execution_; }
    bool has_openmp() const { return has_openmp_; }
    bool has_gcd() const { return has_gcd_; }
    
    // Performance characteristics
    double simd_efficiency() const { return simd_efficiency_; }
    double threading_overhead_ns() const { return threading_overhead_ns_; }
    double memory_bandwidth_gb_s() const { return memory_bandwidth_gb_s_; }
    
private:
    SystemCapabilities();
    void detectCapabilities();
    void benchmarkPerformance();
    
    // Cached capability data
    size_t logical_cores_, physical_cores_;
    size_t l1_cache_size_, l2_cache_size_, l3_cache_size_;
    bool has_sse2_, has_avx_, has_avx2_, has_avx512_, has_neon_;
    bool has_std_execution_, has_openmp_, has_gcd_;
    double simd_efficiency_, threading_overhead_ns_, memory_bandwidth_gb_s_;
};
```

#### 3. Adaptive Performance Learning
```cpp
class PerformanceHistory {
public:
    void recordExecution(Strategy strategy, size_t batch_size, 
                        DistributionType dist_type,
                        std::chrono::nanoseconds duration);
    
    std::chrono::nanoseconds getPredictedDuration(Strategy strategy, 
                                                 size_t batch_size,
                                                 DistributionType dist_type) const;
    
    Strategy getBestStrategy(size_t batch_size, DistributionType dist_type) const;
    
    void adaptThresholds(PerformanceDispatcher::Thresholds& thresholds) const;
    
private:
    struct PerformanceRecord {
        std::chrono::nanoseconds average_duration{0};
        std::chrono::nanoseconds best_duration{std::chrono::nanoseconds::max()};
        std::chrono::nanoseconds worst_duration{0};
        size_t sample_count = 0;
        double variance = 0.0;
        
        void update(std::chrono::nanoseconds duration);
        double confidence() const { return std::min(1.0, sample_count / 100.0); }
    };
    
    using RecordKey = std::tuple<Strategy, size_t, DistributionType>;
    std::unordered_map<RecordKey, PerformanceRecord> history_;
    mutable std::shared_mutex mutex_;
    
    // Learning parameters
    static constexpr size_t MIN_SAMPLES_FOR_ADAPTATION = 10;
    static constexpr double ADAPTATION_LEARNING_RATE = 0.1;
    static constexpr double CONFIDENCE_THRESHOLD = 0.8;
};
```

#### 4. Smart Dispatch Implementation
```cpp
// Example implementation for UniformDistribution
void UniformDistribution::getProbability(std::span<const double> values, 
                                        std::span<double> results) const {
    if (values.size() != results.size()) {
        throw std::invalid_argument("Input and output spans must have the same size");
    }
    
    const size_t count = values.size();
    if (count == 0) return;
    
    // Handle single-value case efficiently
    if (count == 1) {
        results[0] = getProbability(values[0]);
        return;
    }
    
    // Get global dispatcher and performance history
    static thread_local PerformanceDispatcher dispatcher;
    static thread_local PerformanceHistory& history = PerformanceHistory::global();
    
    // Smart dispatch based on problem characteristics
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto strategy = dispatcher.selectOptimalStrategy(
        count,
        DistributionType::UNIFORM,
        ComputationComplexity::SIMPLE,
        SystemCapabilities::current(),
        history
    );
    
    // Execute using selected strategy
    switch (strategy) {
        case Strategy::SCALAR:
            // Use simple loop for tiny batches (< 8 elements)
            for (size_t i = 0; i < count; ++i) {
                results[i] = getProbability(values[i]);
            }
            break;
            
        case Strategy::SIMD_BATCH:
            // Use existing SIMD implementation
            getProbabilityBatch(values.data(), results.data(), count);
            break;
            
        case Strategy::PARALLEL_SIMD:
            // Use existing parallel implementation
            getProbabilityBatchParallel(values, results);
            break;
            
        case Strategy::WORK_STEALING:
            // Use work-stealing pool for load balancing
            getProbabilityBatchWorkStealing(values, results, 
                                          WorkStealingPool::global());
            break;
            
        case Strategy::CACHE_AWARE:
            // Use cache-aware implementation
            getProbabilityBatchCacheAware(values, results, 
                                        AdaptiveCache::global());
            break;
    }
    
    // Record performance for future learning
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    
    history.recordExecution(strategy, count, DistributionType::UNIFORM, duration);
}
```

#### 5. Power User Controls
```cpp
// Optional performance hints for advanced users
struct PerformanceHint {
    enum class PreferredStrategy {
        AUTO,              // Let system decide (default)
        FORCE_SCALAR,      // Force scalar implementation
        FORCE_SIMD,        // Force SIMD if available
        FORCE_PARALLEL,    // Force parallel execution
        MINIMIZE_LATENCY,  // Optimize for lowest latency
        MAXIMIZE_THROUGHPUT // Optimize for highest throughput
    };
    
    PreferredStrategy strategy = PreferredStrategy::AUTO;
    bool disable_learning = false;     // Don't record performance data
    bool force_strategy = false;       // Override all safety checks
    std::optional<size_t> thread_count; // Override thread count
    
    static PerformanceHint minimal_latency() {
        return {PreferredStrategy::MINIMIZE_LATENCY, false, false, 1};
    }
    
    static PerformanceHint maximum_throughput() {
        return {PreferredStrategy::MAXIMIZE_THROUGHPUT, false, false};
    }
};

// Enhanced API with hints
void getProbability(std::span<const double> values, 
                   std::span<double> results,
                   const PerformanceHint& hint = {}) const;
```

---

## 📅 Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2) - MODERATE Effort
**Deliverables:**
- [ ] `include/performance_dispatcher.h` - Core dispatch engine
- [ ] `src/performance_dispatcher.cpp` - Implementation
- [ ] `include/system_capabilities.h` - System detection
- [ ] `src/system_capabilities.cpp` - Implementation
- [ ] Basic unit tests for dispatcher logic
- [ ] Integration with existing threshold constants

**Key Tasks:**
1. Extract current performance thresholds into centralizable format
2. Implement basic system capability detection
3. Create decision engine with static thresholds
4. Add comprehensive logging for decision tracing
5. Create performance measurement infrastructure

### Phase 2: Smart Dispatch Implementation (Weeks 3-4) - MODERATE-HIGH Effort
**Deliverables:**
- [ ] Smart dispatch for `UniformDistribution`
- [ ] Performance timing and measurement system
- [ ] Comprehensive unit tests comparing auto vs manual selection
- [ ] Benchmark suite validating performance claims
- [ ] Documentation of decision logic

**Key Tasks:**
1. Implement unified batch methods for UniformDistribution
2. Add performance measurement with minimal overhead
3. Create exhaustive test suite covering all dispatch paths
4. Benchmark against existing manual method selection
5. Validate no performance regression occurs

### Phase 3: Learning & Adaptation (Weeks 5-6) - HIGH Effort
**Deliverables:**
- [ ] `include/performance_history.h` - Learning system
- [ ] `src/performance_history.cpp` - Implementation
- [ ] Adaptive threshold adjustment algorithms
- [ ] Performance history persistence (optional)
- [ ] Extension to all distribution types

**Key Tasks:**
1. Implement performance data collection and analysis
2. Create adaptive threshold learning algorithms
3. Add confidence-based decision making
4. Extend smart dispatch to Gaussian, Exponential, Discrete distributions
5. Create comprehensive integration tests

### Phase 4: Polish & Integration (Weeks 7-8) - MODERATE Effort
**Deliverables:**
- [ ] `PerformanceHint` system for power users
- [ ] Backward compatibility wrappers (deprecated)
- [ ] Updated documentation and examples
- [ ] Migration guide for existing users
- [ ] Final performance validation

**Key Tasks:**
1. Implement advanced performance hint system
2. Create deprecated wrappers for all existing methods
3. Update all examples to use simplified API
4. Write comprehensive migration documentation
5. Final end-to-end testing and benchmarking

---

## 🎯 Success Metrics

### API Simplification Goals
- [ ] **Reduce public API surface by 60-70%** (17 → 6-9 methods per distribution)
- [ ] **Maintain 100% backward compatibility** (all existing code continues to work)
- [ ] **Zero performance regression** (auto-dispatch performs same or better)
- [ ] **Improve average performance by 10-20%** (through better method selection)

### User Experience Goals
- [ ] **Eliminate performance decision burden** - users call simple `getProbability()`
- [ ] **Auto-optimize over time** - performance improves with usage
- [ ] **Reduce documentation complexity** - 70% fewer methods to document
- [ ] **Lower learning curve** - new users can be productive immediately

### Technical Goals
- [ ] **Sub-microsecond dispatch overhead** - decision making must be fast
- [ ] **Thread-safe learning system** - works correctly in multi-threaded applications
- [ ] **Memory efficient** - performance history uses minimal memory
- [ ] **Platform portable** - works consistently across all supported platforms

---

## 🚨 Risk Assessment & Mitigation

### Technical Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| Dispatch overhead reduces performance | Medium | High | Extensive benchmarking, inline dispatch for hot paths |
| Learning system introduces bugs | Medium | Medium | Comprehensive unit tests, gradual rollout |
| Thread safety issues in learning | Low | High | Thorough concurrency testing, lock-free where possible |
| Platform-specific behavior differences | Low | Medium | Extensive testing on all supported platforms |

### Implementation Risks
| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| Development timeline overrun | Medium | Low | Phased implementation, can ship partial features |
| Backward compatibility breakage | Low | High | Extensive regression testing, deprecated wrappers |
| User confusion during transition | Medium | Low | Clear migration guide, gradual deprecation warnings |

### Mitigation Strategies
1. **Incremental rollout** - Implement one distribution at a time
2. **Extensive testing** - Unit tests, integration tests, performance tests
3. **Backward compatibility** - All existing methods remain functional
4. **Performance monitoring** - Continuous benchmarking during development
5. **User feedback** - Early preview releases for feedback

---

## 📚 Future Extensions

### v1.1.0: Enhanced Learning
- **Cross-session learning** - Persist performance data between runs
- **Workload classification** - Detect and optimize for common usage patterns
- **User preference learning** - Adapt to individual application characteristics
- **Advanced profiling** - Integrate with CPU performance counters

### v1.2.0: GPU Auto-Dispatch
- **GPU detection** - Automatically detect and use available GPUs
- **GPU/CPU hybrid** - Intelligently split work between CPU and GPU
- **Memory transfer optimization** - Minimize data movement overhead
- **Multi-GPU support** - Scale across multiple GPUs automatically

### v1.3.0: Distributed Computing
- **Cluster auto-dispatch** - Automatically use distributed computing when available
- **Network-aware decisions** - Consider network bandwidth in dispatch decisions
- **Load balancing** - Distribute work across cluster nodes optimally
- **Fault tolerance** - Handle node failures gracefully

---

## 💡 Implementation Notes

### Current System Integration Points
- **Constants system** - Use existing `constants::parallel::MIN_ELEMENTS_FOR_*_PARALLEL`
- **CPU detection** - Leverage existing `cpu::get_features()` infrastructure
- **Thread pools** - Integrate with existing `WorkStealingPool` and `ParallelUtils`
- **SIMD dispatch** - Build on existing runtime SIMD selection
- **Cache management** - Extend existing `AdaptiveCache` system

### Code Organization
```
include/
├── performance_dispatcher.h      # Core dispatch engine
├── system_capabilities.h         # System detection and benchmarking
├── performance_history.h         # Learning and adaptation
└── performance_hints.h           # Power user controls

src/
├── performance_dispatcher.cpp
├── system_capabilities.cpp
├── performance_history.cpp
└── auto_dispatch/               # Implementation details
    ├── threshold_learning.cpp
    ├── strategy_selection.cpp
    └── performance_measurement.cpp

tests/
├── test_performance_dispatcher.cpp
├── test_system_capabilities.cpp
├── test_performance_history.cpp
└── integration/
    ├── test_auto_dispatch_uniform.cpp
    ├── test_auto_dispatch_gaussian.cpp
    └── test_backward_compatibility.cpp
```

### Performance Measurement Strategy
- **Minimal overhead** - Use RAII timing with thread-local storage
- **Statistical robustness** - Collect multiple samples, handle outliers
- **Contextual awareness** - Consider system load, thermal throttling
- **Precision vs overhead trade-off** - Balance measurement accuracy with performance

---

## 🎉 Expected Benefits

### For End Users
- **Dramatically simpler API** - Just call `getProbability()` and get optimal performance
- **Zero performance expertise required** - No need to understand SIMD, threading, caching
- **Automatic optimization** - Performance improves over time with usage
- **Future-proof** - GPU, distributed computing added transparently later
- **Reduced cognitive load** - Focus on statistical problems, not performance tuning

### For Library Maintainers
- **Cleaner architecture** - Central point for all performance optimizations
- **Easier testing** - Fewer public methods to test and validate
- **Better encapsulation** - Performance implementation details hidden
- **Future flexibility** - Easy to add new optimization strategies
- **Reduced support burden** - Users can't choose wrong methods anymore

### For Performance
- **Optimal strategy selection** - Based on actual system characteristics and history
- **Adaptive improvement** - Gets smarter and faster over time
- **No performance regression** - Can only match or exceed current performance
- **Better average performance** - Eliminates suboptimal manual choices
- **Reduced performance variance** - Consistent optimal performance across different systems

---

## 📖 References

### Current Performance Characteristics (Baseline)
From `test_uniform_enhanced` results:
- **SIMD Batch (50K elements)**: 48 μs baseline
- **Standard Parallel**: 50 μs (0.96x vs SIMD)
- **Work-Stealing**: 43 μs (1.11x vs SIMD)  
- **Cache-Aware**: 60 μs (0.8x vs SIMD)

### Current Thresholds (To Be Centralized)
- `constants::parallel::MIN_ELEMENTS_FOR_SIMPLE_DISTRIBUTION_PARALLEL = 65536`
- `constants::parallel::MIN_ELEMENTS_FOR_DISTRIBUTION_PARALLEL = 256`
- `constants::parallel::DEFAULT_GRAIN_SIZE = 64`
- `constants::simd::MIN_SIMD_SIZE = 4`

### Related Documents
- `docs/level_2-3_review.md` - Current parallel infrastructure
- `docs/API_consistency_analysis.md` - Current API analysis
- `CMakeLists.txt` - Build system integration points
- `include/constants.h` - Performance threshold definitions

---

## ✅ Action Items

### Immediate (This Sprint)
- [ ] Create GitHub issue for API simplification epic
- [ ] Design review meeting with team
- [ ] Finalize technical architecture decisions
- [ ] Set up development branch: `feature/api-simplification`

### Phase 1 (Next Sprint)
- [ ] Implement `SystemCapabilities` class
- [ ] Create basic `PerformanceDispatcher` with static thresholds
- [ ] Add performance measurement infrastructure
- [ ] Create unit tests for dispatch logic

### Before v1.0.0 Release
- [ ] Complete all 4 phases of implementation
- [ ] Achieve all success metrics
- [ ] Complete migration documentation
- [ ] Validate no performance regressions

---

**Document Status:** ✅ APPROVED FOR IMPLEMENTATION  
**Next Review Date:** Weekly during implementation phases  
**Implementation Start:** Next development cycle  
**Target Completion:** Before v1.0.0 release
