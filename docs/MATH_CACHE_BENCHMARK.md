# Mathematical Function Cache Benchmarking Tool

## Overview

The `math_cache_benchmark.py` tool provides comprehensive benchmarking of the mathematical function cache to determine when caching provides benefits versus overhead. This tool helps answer critical questions about cache effectiveness and adaptive cache behavior.

## Usage

### Basic Usage
```bash
cd /path/to/libstats
mkdir -p build && cd build && cmake .. && make -j4  # Ensure library is built
cd .. && python3 tools/math_cache_benchmark.py
```

### Options
- `--build-dir DIR` - Specify build directory (default: `build`)
- `--output FILE` - Save report to file (default: display on console)
- `--verbose` - Enable verbose output during benchmarks
- `--quick` - Run quick subset of benchmarks (first 4 scenarios)

### Examples
```bash
# Run full benchmark suite
python3 tools/math_cache_benchmark.py

# Quick benchmark with verbose output
python3 tools/math_cache_benchmark.py --quick --verbose

# Save detailed report to file
python3 tools/math_cache_benchmark.py --output math_cache_report.md

# Use custom build directory
python3 tools/math_cache_benchmark.py --build-dir release-build
```

## Benchmark Scenarios

The tool tests 9 different scenarios designed to reveal cache performance characteristics:

### 1. High Repetition Scenarios
- **High Repetition - Gamma**: 10,000 calls to 10 unique gamma values
- **High Repetition - Beta**: 10,000 calls to 25 unique beta combinations

These test ideal caching conditions where the same values are computed repeatedly.

### 2. Medium Repetition Scenarios  
- **Medium Repetition - Mixed**: 5,000 calls with mixed functions and 100 unique values

Tests realistic workloads with moderate repetition.

### 3. Low Repetition Scenarios
- **Low Repetition - Gamma**: 1,000 calls with 800 mostly unique values

Tests scenarios where caching may introduce overhead due to low hit rates.

### 4. Cache Stress Tests
- **Small Cache - High Load**: Cache smaller than the working set to test eviction

Tests cache replacement algorithms and memory pressure scenarios.

### 5. Precision Sensitivity Tests  
- **High Precision - Low Hit Rate**: Very high precision reduces cache effectiveness
- **Low Precision - High Hit Rate**: Lower precision increases cache hit rates

Tests the trade-off between numerical precision and cache performance.

### 6. Real-World Simulations
- **Statistical Distribution Fitting**: Simulates parameter estimation workloads
- **Cold Start - No Warmup**: Tests cache performance without pre-warming

## Metrics Analyzed

### Performance Metrics
- **Speedup Ratio**: `direct_time / cached_time` (higher is better)
- **Hit Rate**: Percentage of cache hits (0-100%)
- **Memory Usage**: Cache memory consumption in KB
- **Cache Efficiency**: Hits per KB of memory used
- **Overhead Ratio**: Relative overhead compared to theoretical best case

### Analysis Categories
- **High Performers**: Scenarios with >2x speedup
- **Modest Benefits**: Scenarios with 1.1-2x speedup  
- **Overhead Cases**: Scenarios with <1.1x speedup

## Interpreting Results

### Speedup Ratio Guidelines
- **>2.0x**: Caching is highly beneficial
- **1.3-2.0x**: Caching provides solid benefits
- **1.1-1.3x**: Caching has modest benefits
- **<1.1x**: Caching may not be worth the complexity

### Hit Rate Analysis
- **>80%**: Excellent cache effectiveness
- **60-80%**: Good cache effectiveness
- **40-60%**: Moderate cache effectiveness
- **<40%**: Poor cache effectiveness

### Memory Efficiency
- High cache efficiency (hits/KB) indicates good memory utilization
- Low efficiency may suggest cache sizes need tuning

## Sample Output

```
🚀 Running Mathematical Function Cache Benchmarks
============================================================

[1/9] High Repetition - Gamma
  Many calls to few unique gamma values
  ✓ Speedup: 4.23x
  ✓ Hit Rate: 92.1%
  ✓ Memory: 12.4 KB

[2/9] High Repetition - Beta  
  Many calls to few unique beta combinations
  ✓ Speedup: 6.78x
  ✓ Hit Rate: 88.7%
  ✓ Memory: 18.2 KB

...

🎯 CONCLUSION:
   Average speedup: 2.34x
   🚀 Mathematical function caching is HIGHLY beneficial!
```

## Report Generation

The tool generates comprehensive reports including:

### Executive Summary
- Overall performance statistics
- Average speedup and hit rates
- Memory usage analysis

### Key Insights
- Scenarios where caching excels
- Scenarios with limited benefits
- Correlation patterns

### Recommendations
- Whether to enable caching
- Which functions benefit most
- Optimal configuration suggestions

### Detailed Analysis
- Per-scenario performance breakdown
- Top and worst performers
- Technical methodology details

## Integration with Development Workflow

### Performance Validation
Use this tool to:
- Validate cache implementation changes
- Compare different cache configurations
- Identify performance regressions

### Configuration Tuning
The benchmarks help optimize:
- Cache sizes for different workloads
- Precision vs performance trade-offs
- Memory usage vs hit rate balance

### Decision Making
Results inform decisions about:
- When to enable/disable caching
- Which mathematical functions to cache
- Resource allocation for cache systems

## Technical Implementation

### Benchmark Methodology
- Each scenario runs 3 times for statistical accuracy
- Compiled with `-O3` optimization for release performance
- Uses `volatile` variables to prevent compiler optimizations
- Fixed random seeds for reproducible results
- Measures both cached and direct function performance

### Supported Functions
- **Gamma Functions**: `std::tgamma()`, log-gamma
- **Error Functions**: `std::erf()`, `std::erfc()`
- **Beta Functions**: Computed using gamma function ratios
- **Logarithm Functions**: Natural and base-10 logarithms
- **Mixed Workloads**: Combinations of the above

### Cache Configuration Testing
- Cache sizes from 50 to 1024 entries
- Precision settings from 0.000001 to 0.01
- With and without cache warming
- Various repetition patterns

## Troubleshooting

### Common Issues

**Compilation Errors**
- Ensure `make` has been run to build libstats
- Check that `build/libstats.a` exists
- Verify C++20 compiler support

**Benchmark Failures**
- Use `--verbose` flag for detailed error messages
- Check that all dependencies are properly linked
- Ensure sufficient system memory

**Inconsistent Results**
- Results may vary based on system load
- CPU frequency scaling can affect timing
- Run multiple times and look for trends

## Future Enhancements

The benchmarking framework is designed to be extensible:

- Additional mathematical functions can be easily added
- New benchmark scenarios can be configured
- Custom workload patterns can be implemented  
- Integration with CI/CD for automated performance testing

## Conclusion

This benchmarking tool provides the data needed to make informed decisions about mathematical function caching. It reveals when caching provides significant benefits versus when it introduces unwanted overhead, enabling optimal performance configuration for different use cases.
