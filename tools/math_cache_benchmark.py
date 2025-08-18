#!/usr/bin/env python3
"""
Mathematical Function Cache Benchmarking Tool

This tool provides comprehensive benchmarking of the mathematical function cache
under various conditions to determine when caching provides benefits vs overhead.

Usage:
    python3 math_cache_benchmark.py [options]

Features:
- Benchmarks different mathematical functions (gamma, erf, beta, log)
- Tests various cache configurations and workload patterns
- Compares cached vs direct function calls
- Analyzes cache hit rates and performance metrics
- Generates detailed reports and recommendations
- Tests adaptive cache behavior under different conditions
"""

import subprocess
import json
import time
import statistics
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import tempfile
import os

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run"""
    name: str
    function_type: str  # gamma, erf, beta, log, mixed
    num_iterations: int
    unique_values: int
    cache_size: int
    precision: float
    warmup: bool = True
    description: str = ""

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    config: BenchmarkConfig
    cached_time_ms: float
    direct_time_ms: float
    speedup_ratio: float
    hit_rate: float
    memory_usage_kb: float
    cache_efficiency: float  # hits per KB of memory
    overhead_ratio: float   # (cached_time - best_case_time) / best_case_time
    
    def __post_init__(self):
        # Calculate derived metrics
        if self.direct_time_ms > 0:
            self.speedup_ratio = self.direct_time_ms / self.cached_time_ms
        else:
            self.speedup_ratio = 0.0
            
        if self.memory_usage_kb > 0 and self.hit_rate > 0:
            self.cache_efficiency = (self.hit_rate * self.config.num_iterations) / self.memory_usage_kb
        else:
            self.cache_efficiency = 0.0
            
        # Estimate best-case time (all cache hits with minimal overhead)
        estimated_best_case = self.direct_time_ms * 0.1  # Assume 10x speedup is theoretical best
        if estimated_best_case > 0:
            self.overhead_ratio = max(0, (self.cached_time_ms - estimated_best_case) / estimated_best_case)
        else:
            self.overhead_ratio = 0.0

class MathCacheBenchmark:
    def __init__(self, build_dir: str = "build", verbose: bool = False):
        self.build_dir = Path(build_dir)
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
        
        # Determine project root and set up paths
        self.setup_paths()
        
        # Predefined benchmark configurations
        self.benchmark_configs = [
            # High repetition scenarios (cache should excel)
            BenchmarkConfig(
                name="High Repetition - Gamma",
                function_type="gamma",
                num_iterations=10000,
                unique_values=10,
                cache_size=128,
                precision=0.001,
                description="Many calls to few unique gamma values"
            ),
            BenchmarkConfig(
                name="High Repetition - Beta",
                function_type="beta", 
                num_iterations=10000,
                unique_values=25,  # 5x5 combinations
                cache_size=128,
                precision=0.001,
                description="Many calls to few unique beta combinations"
            ),
            
            # Medium repetition scenarios
            BenchmarkConfig(
                name="Medium Repetition - Mixed",
                function_type="mixed",
                num_iterations=5000,
                unique_values=100,
                cache_size=256,
                precision=0.0001,
                description="Mixed mathematical functions with moderate repetition"
            ),
            
            # Low repetition scenarios (cache may have overhead)
            BenchmarkConfig(
                name="Low Repetition - Gamma",
                function_type="gamma",
                num_iterations=1000,
                unique_values=800,
                cache_size=1024,
                precision=0.00001,
                description="Mostly unique gamma values - cache overhead test"
            ),
            
            # Cache size stress tests
            BenchmarkConfig(
                name="Small Cache - High Load",
                function_type="gamma",
                num_iterations=5000,
                unique_values=200,
                cache_size=50,  # Smaller than unique values
                precision=0.001,
                description="Cache eviction and replacement stress test"
            ),
            
            # Precision sensitivity tests
            BenchmarkConfig(
                name="High Precision - Low Hit Rate",
                function_type="erf",
                num_iterations=2000,
                unique_values=50,
                cache_size=128,
                precision=0.000001,  # Very high precision
                description="High precision reduces cache effectiveness"
            ),
            BenchmarkConfig(
                name="Low Precision - High Hit Rate", 
                function_type="erf",
                num_iterations=2000,
                unique_values=50,
                cache_size=128,
                precision=0.01,  # Lower precision
                description="Low precision increases cache effectiveness"
            ),
            
            # Real-world simulation scenarios
            BenchmarkConfig(
                name="Statistical Distribution Fitting",
                function_type="mixed",
                num_iterations=3000,
                unique_values=150,
                cache_size=512,
                precision=0.001,
                warmup=True,
                description="Simulates parameter estimation workload"
            ),
            
            # Cold start scenarios
            BenchmarkConfig(
                name="Cold Start - No Warmup",
                function_type="gamma",
                num_iterations=1000,
                unique_values=20,
                cache_size=128,
                precision=0.001,
                warmup=False,
                description="Tests cache performance without warmup"
            ),
        ]
    
    def setup_paths(self):
        """Setup project paths based on current working directory"""
        current_dir = Path.cwd()
        
        # Try to find project root by looking for key files
        project_indicators = ['CMakeLists.txt', 'include/libstats.h', 'src/']
        
        # Check current directory first
        if all((current_dir / indicator).exists() for indicator in project_indicators):
            self.project_root = current_dir
        # Check parent directory (if we're in tools/)
        elif all((current_dir.parent / indicator).exists() for indicator in project_indicators):
            self.project_root = current_dir.parent
        # Check if we need to go up more levels
        else:
            # Search up the directory tree
            search_dir = current_dir
            for _ in range(5):  # Limit search depth
                if all((search_dir / indicator).exists() for indicator in project_indicators):
                    self.project_root = search_dir
                    break
                search_dir = search_dir.parent
                if search_dir == search_dir.parent:  # Reached root
                    break
            else:
                # Default to current directory
                self.project_root = current_dir
        
        # Update build_dir to be relative to project root if it's not absolute
        if not self.build_dir.is_absolute():
            self.build_dir = self.project_root / self.build_dir
        
        if self.verbose:
            print(f"Project root: {self.project_root}")
            print(f"Build directory: {self.build_dir}")
    
    def create_benchmark_program(self, config: BenchmarkConfig) -> str:
        """Generate C++ benchmark program for the given configuration"""
        cpp_code = f'''
#include <chrono>
#include "include/cache/math_function_cache.h"
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cmath>

using namespace libstats::cache;

struct BenchmarkResults {{
    double cached_time_ms;
    double direct_time_ms;
    double hit_rate;
    size_t memory_usage_kb;
}};

BenchmarkResults run_benchmark() {{
    // Configuration from Python
    const int num_iterations = {config.num_iterations};
    const int unique_values = {config.unique_values};
    const std::string function_type = "{config.function_type}";
    const double precision = {config.precision};
    const bool warmup = {str(config.warmup).lower()};
    
    // Configure cache
    MathFunctionCacheConfig cache_config;
    cache_config.gamma_cache_size = {config.cache_size};
    cache_config.erf_cache_size = {config.cache_size};
    cache_config.beta_cache_size = {config.cache_size};
    cache_config.log_cache_size = {config.cache_size};
    cache_config.gamma_precision = precision;
    cache_config.erf_precision = precision;
    cache_config.beta_precision = precision;
    cache_config.log_precision = precision;
    cache_config.enable_statistics = true;
    
    MathFunctionCache::initialize(cache_config);
    
    // Generate test values
    std::vector<double> test_values;
    std::vector<std::pair<double, double>> test_pairs;
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    
    if (function_type == "beta") {{
        std::uniform_real_distribution<double> dist_a(0.5, 5.0);
        std::uniform_real_distribution<double> dist_b(0.5, 5.0);
        
        // Generate unique pairs, then repeat to reach num_iterations
        for (int i = 0; i < unique_values && i < 100; ++i) {{
            test_pairs.push_back({{dist_a(rng), dist_b(rng)}});
        }}
        
        // Extend to num_iterations by cycling through unique values
        while (test_pairs.size() < num_iterations) {{
            for (const auto& pair : test_pairs) {{
                if (test_pairs.size() >= num_iterations) break;
                test_pairs.push_back(pair);
            }}
        }}
        test_pairs.resize(num_iterations);
        
    }} else {{
        std::uniform_real_distribution<double> dist(0.1, 10.0);
        
        // Generate unique values
        std::vector<double> unique_vals;
        for (int i = 0; i < unique_values; ++i) {{
            unique_vals.push_back(dist(rng));
        }}
        
        // Extend to num_iterations by cycling
        for (int i = 0; i < num_iterations; ++i) {{
            test_values.push_back(unique_vals[i % unique_vals.size()]);
        }}
    }}
    
    // Warmup if requested
    if (warmup) {{
        MathFunctionCache::warmUp();
    }}
    
    // Clear statistics before benchmark
    MathFunctionCache::clearAll();
    MathFunctionCache::initialize(cache_config);
    
    if (warmup) {{
        MathFunctionCache::warmUp();
    }}
    
    // Benchmark cached functions
    auto start_cached = std::chrono::high_resolution_clock::now();
    
    if (function_type == "gamma") {{
        for (double x : test_values) {{
            volatile double result = MathFunctionCache::getCachedGamma(x);
            (void)result;
        }}
    }} else if (function_type == "erf") {{
        for (double x : test_values) {{
            volatile double result = MathFunctionCache::getCachedErf(x);
            (void)result;
        }}
    }} else if (function_type == "beta") {{
        for (const auto& pair : test_pairs) {{
            volatile double result = MathFunctionCache::getCachedBeta(pair.first, pair.second);
            (void)result;
        }}
    }} else if (function_type == "log") {{
        for (double x : test_values) {{
            if (x > 0) {{
                volatile double result = MathFunctionCache::getCachedLog(x);
                (void)result;
            }}
        }}
    }} else if (function_type == "mixed") {{
        for (size_t i = 0; i < test_values.size(); ++i) {{
            double x = test_values[i];
            
            if (i % 4 == 0) {{
                volatile double result = MathFunctionCache::getCachedGamma(x);
                (void)result;
            }} else if (i % 4 == 1) {{
                volatile double result = MathFunctionCache::getCachedErf(x * 0.5);
                (void)result;
            }} else if (i % 4 == 2) {{
                if (x > 0) {{
                    volatile double result = MathFunctionCache::getCachedLog(x);
                    (void)result;
                }}
            }} else {{
                volatile double result = MathFunctionCache::getCachedBeta(x, x + 0.5);
                (void)result;
            }}
        }}
    }}
    
    auto end_cached = std::chrono::high_resolution_clock::now();
    auto cached_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cached - start_cached);
    
    // Get cache statistics
    auto stats = MathFunctionCache::getStats();
    double overall_hit_rate = stats.getOverallHitRate();
    size_t memory_usage = MathFunctionCache::getMemoryUsage();
    
    // Benchmark direct function calls
    auto start_direct = std::chrono::high_resolution_clock::now();
    
    if (function_type == "gamma") {{
        for (double x : test_values) {{
            volatile double result = std::tgamma(x);
            (void)result;
        }}
    }} else if (function_type == "erf") {{
        for (double x : test_values) {{
            volatile double result = std::erf(x);
            (void)result;
        }}
    }} else if (function_type == "beta") {{
        for (const auto& pair : test_pairs) {{
            volatile double result = std::tgamma(pair.first) * std::tgamma(pair.second) / std::tgamma(pair.first + pair.second);
            (void)result;
        }}
    }} else if (function_type == "log") {{
        for (double x : test_values) {{
            if (x > 0) {{
                volatile double result = std::log(x);
                (void)result;
            }}
        }}
    }} else if (function_type == "mixed") {{
        for (size_t i = 0; i < test_values.size(); ++i) {{
            double x = test_values[i];
            
            if (i % 4 == 0) {{
                volatile double result = std::tgamma(x);
                (void)result;
            }} else if (i % 4 == 1) {{
                volatile double result = std::erf(x * 0.5);
                (void)result;
            }} else if (i % 4 == 2) {{
                if (x > 0) {{
                    volatile double result = std::log(x);
                    (void)result;
                }}
            }} else {{
                volatile double result = std::tgamma(x) * std::tgamma(x + 0.5) / std::tgamma(x + x + 0.5);
                (void)result;
            }}
        }}
    }}
    
    auto end_direct = std::chrono::high_resolution_clock::now();
    auto direct_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_direct - start_direct);
    
    return {{
        cached_duration.count() / 1000.0,  // Convert to milliseconds
        direct_duration.count() / 1000.0,
        overall_hit_rate,
        memory_usage / 1024  // Convert to KB
    }};
}}

int main() {{
    try {{
        auto results = run_benchmark();
        
        // Output results as JSON for Python to parse
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{{"
                  << "\\"cached_time_ms\\": " << results.cached_time_ms << ", "
                  << "\\"direct_time_ms\\": " << results.direct_time_ms << ", "
                  << "\\"hit_rate\\": " << results.hit_rate << ", "
                  << "\\"memory_usage_kb\\": " << results.memory_usage_kb
                  << "}}" << std::endl;
        
        return 0;
    }} catch (const std::exception& e) {{
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }}
}}
'''
        return cpp_code
    
    def run_single_benchmark(self, config: BenchmarkConfig) -> Optional[BenchmarkResult]:
        """Run a single benchmark configuration"""
        if self.verbose:
            print(f"Running benchmark: {config.name}")
            
        # Create temporary C++ file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(self.create_benchmark_program(config))
            cpp_file = f.name
            
        try:
            # Compile the benchmark - use proper compiler configuration for macOS
            exe_file = cpp_file.replace('.cpp', '')
            
            # Use Homebrew LLVM if available, otherwise fall back to system clang++
            compiler = "/usr/local/opt/llvm/bin/clang++" if os.path.exists("/usr/local/opt/llvm/bin/clang++") else "clang++"
            
            # Configure compiler with proper C++20 flags based on compiler type
            if compiler.startswith("/usr/local/opt/llvm"):
                # Homebrew LLVM configuration
                compile_cmd = [
                    compiler, '-std=c++20', '-stdlib=libc++',
                    '-I/usr/local/opt/llvm/include/c++/v1',
                    '-O3', '-DNDEBUG',
                    '-I' + str(self.project_root), cpp_file,
                    '-L/usr/local/opt/llvm/lib/c++',
                    '-L' + str(self.build_dir), 
                    '-Wl,-rpath,/usr/local/opt/llvm/lib/c++',
                    str(self.build_dir / 'libstats.a'),  # Use static library for local linking
                    '-o', exe_file
                ]
            else:
                # System compiler (fallback) - try c++2a first, then c++20
                std_flag = '-std=c++20'
                # Check if c++20 is supported by trying c++2a first
                test_cmd = [compiler, '-std=c++2a', '-E', '-x', 'c++', '-', '-o', '/dev/null']
                try:
                    result = subprocess.run(test_cmd, input='', text=True, capture_output=True, timeout=5)
                    if result.returncode == 0:
                        std_flag = '-std=c++2a'
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass
                
                compile_cmd = [
                    compiler, std_flag, '-O3', '-DNDEBUG',
                    '-I' + str(self.project_root), cpp_file,
                    '-L' + str(self.build_dir), 
                    str(self.build_dir / 'libstats.a'),  # Use static library for local linking
                    '-o', exe_file
                ]
            
            
            if self.verbose:
                print(f"Compiling: {' '.join(compile_cmd)}")
                
            result = subprocess.run(compile_cmd, capture_output=True, text=True, cwd=str(self.project_root))
            if result.returncode != 0:
                print(f"Compilation failed for {config.name}:")
                print(result.stderr)
                return None
                
            # Run the benchmark multiple times for statistical accuracy
            run_times = 3  # Number of runs to average
            cached_times = []
            direct_times = []
            hit_rates = []
            memory_usages = []
            
            for run in range(run_times):
                if self.verbose:
                    print(f"  Run {run + 1}/{run_times}")
                    
                result = subprocess.run([exe_file], capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Benchmark execution failed for {config.name}:")
                    print(result.stderr)
                    continue
                    
                try:
                    data = json.loads(result.stdout.strip())
                    cached_times.append(data['cached_time_ms'])
                    direct_times.append(data['direct_time_ms'])
                    hit_rates.append(data['hit_rate'])
                    memory_usages.append(data['memory_usage_kb'])
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Failed to parse output for {config.name}: {e}")
                    print(f"Output was: {result.stdout}")
                    continue
            
            if not cached_times:
                return None
                
            # Calculate averages
            avg_cached_time = statistics.mean(cached_times)
            avg_direct_time = statistics.mean(direct_times)
            avg_hit_rate = statistics.mean(hit_rates)
            avg_memory_usage = statistics.mean(memory_usages)
            
            return BenchmarkResult(
                config=config,
                cached_time_ms=avg_cached_time,
                direct_time_ms=avg_direct_time,
                speedup_ratio=avg_direct_time / avg_cached_time if avg_cached_time > 0 else 0,
                hit_rate=avg_hit_rate,
                memory_usage_kb=avg_memory_usage,
                cache_efficiency=0,  # Will be calculated in __post_init__
                overhead_ratio=0     # Will be calculated in __post_init__
            )
            
        finally:
            # Cleanup
            try:
                os.unlink(cpp_file)
                if os.path.exists(exe_file):
                    os.unlink(exe_file)
            except OSError:
                pass
    
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmark configurations"""
        print("🚀 Running Mathematical Function Cache Benchmarks")
        print("=" * 60)
        
        results = []
        for i, config in enumerate(self.benchmark_configs):
            print(f"\n[{i+1}/{len(self.benchmark_configs)}] {config.name}")
            print(f"  {config.description}")
            
            result = self.run_single_benchmark(config)
            if result:
                results.append(result)
                
                # Quick summary
                print(f"  ✓ Speedup: {result.speedup_ratio:.2f}x")
                print(f"  ✓ Hit Rate: {result.hit_rate*100:.1f}%")
                print(f"  ✓ Memory: {result.memory_usage_kb:.1f} KB")
            else:
                print(f"  ❌ Benchmark failed")
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict:
        """Analyze benchmark results and provide insights"""
        if not self.results:
            return {}
            
        analysis = {
            'summary': {
                'total_benchmarks': len(self.results),
                'successful_benchmarks': len([r for r in self.results if r.speedup_ratio > 0]),
                'avg_speedup': statistics.mean([r.speedup_ratio for r in self.results]),
                'avg_hit_rate': statistics.mean([r.hit_rate for r in self.results]),
                'avg_memory_usage': statistics.mean([r.memory_usage_kb for r in self.results])
            },
            'best_performers': [],
            'worst_performers': [],
            'insights': [],
            'recommendations': []
        }
        
        # Sort by speedup ratio
        sorted_results = sorted(self.results, key=lambda x: x.speedup_ratio, reverse=True)
        
        # Best and worst performers
        analysis['best_performers'] = [
            {
                'name': r.config.name,
                'speedup': r.speedup_ratio,
                'hit_rate': r.hit_rate,
                'description': r.config.description
            }
            for r in sorted_results[:3]
        ]
        
        analysis['worst_performers'] = [
            {
                'name': r.config.name,
                'speedup': r.speedup_ratio,
                'hit_rate': r.hit_rate,
                'description': r.config.description
            }
            for r in sorted_results[-3:]
        ]
        
        # Generate insights
        high_speedup = [r for r in self.results if r.speedup_ratio > 2.0]
        low_speedup = [r for r in self.results if r.speedup_ratio < 1.1]
        high_hit_rate = [r for r in self.results if r.hit_rate > 0.8]
        
        if high_speedup:
            analysis['insights'].append(
                f"Cache provides significant benefits ({len(high_speedup)} scenarios with >2x speedup)"
            )
            
        if low_speedup:
            analysis['insights'].append(
                f"Cache has limited benefits in {len(low_speedup)} scenarios (minimal speedup)"
            )
            
        if high_hit_rate:
            analysis['insights'].append(
                f"High hit rates ({len(high_hit_rate)} scenarios >80%) correlate with better performance"
            )
        
        # Generate recommendations
        if analysis['summary']['avg_speedup'] > 1.5:
            analysis['recommendations'].append(
                "✅ Mathematical function caching is beneficial overall"
            )
        elif analysis['summary']['avg_speedup'] > 1.1:
            analysis['recommendations'].append(
                "⚠️ Caching provides modest benefits - consider selective use"
            )
        else:
            analysis['recommendations'].append(
                "❌ Caching may not be worth the complexity overhead"
            )
            
        # Specific recommendations based on patterns
        beta_results = [r for r in self.results if r.config.function_type == "beta"]
        if beta_results and all(r.speedup_ratio > 1.5 for r in beta_results):
            analysis['recommendations'].append(
                "✅ Beta function caching is particularly effective"
            )
            
        precision_results = [r for r in self.results if "Precision" in r.config.name]
        if len(precision_results) >= 2:
            high_prec = [r for r in precision_results if "High Precision" in r.config.name]
            low_prec = [r for r in precision_results if "Low Precision" in r.config.name]
            if high_prec and low_prec and low_prec[0].speedup_ratio > high_prec[0].speedup_ratio * 1.5:
                analysis['recommendations'].append(
                    "⚙️ Lower precision settings significantly improve cache effectiveness"
                )
        
        return analysis
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return "No benchmark results available."
            
        analysis = self.analyze_results()
        
        report = []
        report.append("# Mathematical Function Cache Benchmark Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total Benchmarks: {analysis['summary']['total_benchmarks']}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- Average Speedup: {analysis['summary']['avg_speedup']:.2f}x")
        report.append(f"- Average Hit Rate: {analysis['summary']['avg_hit_rate']*100:.1f}%")
        report.append(f"- Average Memory Usage: {analysis['summary']['avg_memory_usage']:.1f} KB")
        report.append("")
        
        # Key Insights
        if analysis['insights']:
            report.append("## Key Insights")
            for insight in analysis['insights']:
                report.append(f"- {insight}")
            report.append("")
        
        # Recommendations
        if analysis['recommendations']:
            report.append("## Recommendations")
            for rec in analysis['recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Detailed Results
        report.append("## Detailed Results")
        report.append("")
        report.append("| Benchmark | Speedup | Hit Rate | Memory (KB) | Function | Description |")
        report.append("|-----------|---------|----------|-------------|----------|-------------|")
        
        for result in sorted(self.results, key=lambda x: x.speedup_ratio, reverse=True):
            report.append(
                f"| {result.config.name[:20]} | "
                f"{result.speedup_ratio:.2f}x | "
                f"{result.hit_rate*100:.1f}% | "
                f"{result.memory_usage_kb:.1f} | "
                f"{result.config.function_type} | "
                f"{result.config.description[:30]}... |"
            )
        
        report.append("")
        
        # Performance Analysis
        report.append("## Performance Analysis")
        report.append("")
        
        # Best performers
        if analysis['best_performers']:
            report.append("### Top Performers")
            for i, perf in enumerate(analysis['best_performers'], 1):
                report.append(f"{i}. **{perf['name']}** - {perf['speedup']:.2f}x speedup ({perf['hit_rate']*100:.1f}% hit rate)")
                report.append(f"   {perf['description']}")
            report.append("")
        
        # Worst performers
        if analysis['worst_performers']:
            report.append("### Areas for Improvement")
            for i, perf in enumerate(analysis['worst_performers'], 1):
                report.append(f"{i}. **{perf['name']}** - {perf['speedup']:.2f}x speedup ({perf['hit_rate']*100:.1f}% hit rate)")
                report.append(f"   {perf['description']}")
            report.append("")
        
        # Technical Details
        report.append("## Technical Details")
        report.append("")
        report.append("### Benchmark Methodology")
        report.append("- Each benchmark runs 3 times and results are averaged")
        report.append("- Compiled with -O3 optimization for release performance")
        report.append("- Uses volatile variables to prevent compiler optimizations")
        report.append("- Measures both cached and direct function call performance")
        report.append("- Tracks cache hit rates and memory usage")
        report.append("")
        
        report.append("### Cache Configuration Details")
        report.append("- Cache sizes varied by benchmark (50-1024 entries)")
        report.append("- Precision settings tested from 0.000001 to 0.01")
        report.append("- Warmup enabled for most scenarios (disabled for cold start tests)")
        report.append("- Statistics collection enabled for all benchmarks")
        report.append("")
        
        full_report = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(full_report)
            print(f"\n📄 Report saved to: {output_file}")
        
        return full_report

def main():
    parser = argparse.ArgumentParser(description="Mathematical Function Cache Benchmark Tool")
    parser.add_argument("--build-dir", default="build", help="Build directory path")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick subset of benchmarks")
    
    args = parser.parse_args()
    
    # Create benchmark instance to setup paths
    benchmark = MathCacheBenchmark(str(args.build_dir), args.verbose)
    
    # Check if build directory exists and contains libstats
    build_dir = benchmark.build_dir
    if not build_dir.exists():
        print(f"Error: Build directory '{build_dir}' does not exist")
        print("Please run 'make' to build the library first")
        sys.exit(1)
        
    lib_file = build_dir / "libstats.a"
    if not lib_file.exists():
        print(f"Error: libstats.a not found in '{build_dir}'")
        print("Please run 'make' to build the library first")
        sys.exit(1)
    
    # benchmark instance already created above for path setup
    
    # Run quick subset if requested
    if args.quick:
        benchmark.benchmark_configs = benchmark.benchmark_configs[:4]  # First 4 benchmarks
        print("🏃‍♂️ Running quick benchmark subset...")
    
    # Run benchmarks
    try:
        results = benchmark.run_all_benchmarks()
        
        if not results:
            print("\n❌ No benchmarks completed successfully")
            sys.exit(1)
            
        # Generate and display report
        print("\n" + "="*60)
        report = benchmark.generate_report(args.output)
        
        if not args.output:
            print(report)
            
        # Summary
        analysis = benchmark.analyze_results()
        print(f"\n🎯 CONCLUSION:")
        print(f"   Average speedup: {analysis['summary']['avg_speedup']:.2f}x")
        
        if analysis['summary']['avg_speedup'] > 2.0:
            print("   🚀 Mathematical function caching is HIGHLY beneficial!")
        elif analysis['summary']['avg_speedup'] > 1.3:
            print("   ✅ Mathematical function caching provides solid benefits")
        elif analysis['summary']['avg_speedup'] > 1.1:
            print("   ⚠️  Mathematical function caching has modest benefits")
        else:
            print("   ❌ Mathematical function caching may not be worth the complexity")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
