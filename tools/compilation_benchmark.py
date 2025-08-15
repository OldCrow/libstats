#!/usr/bin/env python3
"""
Compilation Performance Benchmark Tool for libstats

This tool measures compilation times, memory usage, and preprocessing statistics
to validate the effectiveness of header optimization work.
"""

import os
import re
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

@dataclass
class CompilationMetrics:
    """Metrics from a single compilation."""
    wall_time: float
    cpu_time: float
    memory_peak_kb: int
    preprocessed_lines: int
    preprocessed_size_kb: int
    object_size_kb: int
    
class CompilationBenchmark:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.build_dir = self.project_root / "build"
        self.results = {}
        
    def measure_single_file_compilation(self, 
                                      source_file: str,
                                      include_dirs: List[str],
                                      test_name: str) -> CompilationMetrics:
        """Measure compilation metrics for a single file."""
        print(f"   📊 Measuring {test_name}...")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(f'#include "{source_file}"\nint main(){{ return 0; }}\n')
            test_cpp = f.name
            
        try:
            # Build the compile command
            include_flags = [f"-I{inc}" for inc in include_dirs]
            cmd_base = ["clang++", "-std=c++20", "-O0"] + include_flags
            
            # Measure preprocessing
            preprocess_cmd = cmd_base + ["-E", test_cpp, "-o", "/dev/null"]
            preprocess_start = time.time()
            preprocess_result = subprocess.run(
                preprocess_cmd, 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            preprocess_time = time.time() - preprocess_start
            
            if preprocess_result.returncode != 0:
                print(f"      ❌ Preprocessing failed: {preprocess_result.stderr}")
                raise Exception(f"Preprocessing failed for {source_file}")
            
            # Count preprocessed lines
            preprocess_lines_cmd = cmd_base + ["-E", test_cpp]
            preprocess_output = subprocess.run(
                preprocess_lines_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            preprocessed_lines = len(preprocess_output.stdout.split('\n'))
            preprocessed_size_kb = len(preprocess_output.stdout.encode('utf-8')) // 1024
            
            # Measure compilation with timing
            with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as obj_file:
                obj_path = obj_file.name
                
            compile_cmd = cmd_base + ["-c", test_cpp, "-o", obj_path]
            
            # Use time command to measure resources (macOS version)
            time_cmd = ["/usr/bin/time", "-l"] + compile_cmd
            
            compile_start = time.time()
            time_result = subprocess.run(
                time_cmd,
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            wall_time = time.time() - compile_start
            
            if time_result.returncode != 0:
                print(f"      ❌ Compilation failed: {time_result.stderr}")
                raise Exception(f"Compilation failed for {source_file}")
            
            # Parse time output for memory usage
            time_output = time_result.stderr
            memory_match = re.search(r'(\d+)\s+maximum resident set size', time_output)
            memory_kb = int(memory_match.group(1)) // 1024 if memory_match else 0
            
            # CPU time parsing
            cpu_match = re.search(r'(\d+\.\d+)\s+total', time_output)
            cpu_time = float(cpu_match.group(1)) if cpu_match else wall_time
            
            # Get object file size
            object_size_kb = 0
            if os.path.exists(obj_path):
                object_size_kb = os.path.getsize(obj_path) // 1024
                os.unlink(obj_path)
                
            return CompilationMetrics(
                wall_time=wall_time,
                cpu_time=cpu_time,
                memory_peak_kb=memory_kb,
                preprocessed_lines=preprocessed_lines,
                preprocessed_size_kb=preprocessed_size_kb,
                object_size_kb=object_size_kb
            )
            
        finally:
            # Cleanup
            if os.path.exists(test_cpp):
                os.unlink(test_cpp)
                
    def benchmark_key_headers(self):
        """Benchmark compilation of key headers."""
        print("🏁 Starting compilation benchmarks...")
        
        include_dirs = [
            str(self.project_root / "include"),
            str(self.project_root / "src")
        ]
        
        # Key headers to benchmark
        test_cases = [
            ("libstats.h", "Main library header"),
            ("core/distribution_base.h", "Distribution base class"),
            ("distributions/gaussian.h", "Gaussian distribution"),
            ("distributions/exponential.h", "Exponential distribution"),
            ("platform/simd.h", "SIMD platform header"),
            ("platform/parallel_execution.h", "Parallel execution header"),
        ]
        
        for header, description in test_cases:
            try:
                metrics = self.measure_single_file_compilation(
                    header, include_dirs, description
                )
                self.results[header] = metrics
                
                print(f"      ⏱️  Wall time: {metrics.wall_time:.3f}s")
                print(f"      💾 Memory: {metrics.memory_peak_kb}KB")
                print(f"      📝 Preprocessed: {metrics.preprocessed_lines:,} lines ({metrics.preprocessed_size_kb}KB)")
                print()
                
            except Exception as e:
                print(f"      ❌ Failed to benchmark {header}: {e}")
                print()
    
    def analyze_preprocessing_bloat(self):
        """Analyze preprocessing output to identify bloat sources."""
        print("🔍 Analyzing preprocessing bloat...")
        
        if not self.results:
            print("   No benchmark results available")
            return
            
        print("   Header preprocessing statistics:")
        print("   " + "-" * 70)
        print(f"   {'Header':<35} {'Lines':>8} {'Size (KB)':>10} {'Ratio':>8}")
        print("   " + "-" * 70)
        
        baseline_lines = 1000  # Approximate minimal include count
        
        for header, metrics in self.results.items():
            bloat_ratio = metrics.preprocessed_lines / baseline_lines
            status = "✅" if bloat_ratio < 5.0 else "⚠️" if bloat_ratio < 10.0 else "❌"
            
            print(f"   {header:<35} {metrics.preprocessed_lines:>8,} {metrics.preprocessed_size_kb:>10} {bloat_ratio:>7.1f}x {status}")
    
    def measure_full_build_performance(self):
        """Measure full project build performance."""
        print("🏗️  Measuring full build performance...")
        
        # Clean build first
        if self.build_dir.exists():
            subprocess.run(["rm", "-rf", str(self.build_dir)], check=True)
        
        self.build_dir.mkdir(exist_ok=True)
        
        try:
            # Configure with CMake
            print("   📋 Configuring with CMake...")
            configure_start = time.time()
            configure_result = subprocess.run(
                ["cmake", ".."],
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            configure_time = time.time() - configure_start
            
            if configure_result.returncode != 0:
                print(f"   ❌ CMake configuration failed: {configure_result.stderr}")
                return
                
            print(f"   ⏱️  Configuration time: {configure_time:.2f}s")
            
            # Build with timing
            print("   🔨 Building project...")
            build_start = time.time()
            
            # Use make with timing
            build_cmd = ["/usr/bin/time", "-l", "make", "-j4"]
            build_result = subprocess.run(
                build_cmd,
                cwd=self.build_dir,
                capture_output=True,
                text=True
            )
            build_time = time.time() - build_start
            
            if build_result.returncode != 0:
                print(f"   ❌ Build failed: {build_result.stderr}")
                return
            
            # Parse build statistics
            time_output = build_result.stderr
            memory_match = re.search(r'(\d+)\s+maximum resident set size', time_output)
            peak_memory_mb = int(memory_match.group(1)) // 1024 // 1024 if memory_match else 0
            
            print(f"   ⏱️  Build time: {build_time:.2f}s")
            print(f"   💾 Peak memory: {peak_memory_mb}MB")
            
            # Count built targets
            object_files = list(self.build_dir.rglob("*.o"))
            libraries = list(self.build_dir.rglob("*.a")) + list(self.build_dir.rglob("*.dylib"))
            executables = [f for f in self.build_dir.rglob("*") if f.is_file() and os.access(f, os.X_OK) and not f.suffix]
            
            print(f"   📊 Built targets:")
            print(f"      Object files: {len(object_files)}")
            print(f"      Libraries: {len(libraries)}")
            print(f"      Executables: {len(executables)}")
            
        except Exception as e:
            print(f"   ❌ Build measurement failed: {e}")
    
    def generate_optimization_report(self):
        """Generate a comprehensive optimization report."""
        print("\n" + "="*80)
        print("🎯 HEADER OPTIMIZATION PERFORMANCE REPORT")
        print("="*80)
        
        if not self.results:
            print("No benchmark data available")
            return
            
        # Calculate averages
        avg_wall_time = sum(m.wall_time for m in self.results.values()) / len(self.results)
        avg_memory = sum(m.memory_peak_kb for m in self.results.values()) / len(self.results)
        avg_preprocess_lines = sum(m.preprocessed_lines for m in self.results.values()) / len(self.results)
        
        print(f"📈 Performance Summary:")
        print(f"   Average compilation time: {avg_wall_time:.3f}s")
        print(f"   Average memory usage: {avg_memory:.1f}KB")
        print(f"   Average preprocessed lines: {avg_preprocess_lines:,.0f}")
        
        # Identify optimization opportunities
        print(f"\n🎯 Optimization Opportunities:")
        
        slow_headers = [h for h, m in self.results.items() if m.wall_time > avg_wall_time * 1.5]
        if slow_headers:
            print(f"   📌 Slow-compiling headers ({len(slow_headers)}):")
            for header in slow_headers:
                metrics = self.results[header]
                print(f"      {header}: {metrics.wall_time:.3f}s ({metrics.preprocessed_lines:,} lines)")
        
        bloated_headers = [h for h, m in self.results.items() if m.preprocessed_lines > avg_preprocess_lines * 2]
        if bloated_headers:
            print(f"   🎈 Preprocessing-heavy headers ({len(bloated_headers)}):")
            for header in bloated_headers:
                metrics = self.results[header]
                print(f"      {header}: {metrics.preprocessed_lines:,} lines")
        
        print(f"\n✅ Header optimization assessment:")
        fast_headers = len([h for h, m in self.results.items() if m.wall_time < 0.5])
        total_headers = len(self.results)
        
        if fast_headers / total_headers > 0.7:
            print(f"   🚀 Excellent: {fast_headers}/{total_headers} headers compile quickly")
        elif fast_headers / total_headers > 0.5:
            print(f"   👍 Good: {fast_headers}/{total_headers} headers compile quickly")
        else:
            print(f"   ⚠️  Needs improvement: Only {fast_headers}/{total_headers} headers compile quickly")
    
    def save_results(self, filename: str = "compilation_benchmark.json"):
        """Save benchmark results to JSON file."""
        results_data = {}
        for header, metrics in self.results.items():
            results_data[header] = {
                'wall_time': metrics.wall_time,
                'cpu_time': metrics.cpu_time,
                'memory_peak_kb': metrics.memory_peak_kb,
                'preprocessed_lines': metrics.preprocessed_lines,
                'preprocessed_size_kb': metrics.preprocessed_size_kb,
                'object_size_kb': metrics.object_size_kb
            }
        
        # Save to tools/ directory to avoid cluttering project root
        tools_dir = self.project_root / "tools"
        tools_dir.mkdir(exist_ok=True)
        output_path = tools_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to {output_path}")
    
    def run_benchmark(self):
        """Run the complete compilation benchmark."""
        print("🚀 Starting Compilation Performance Benchmark for libstats")
        print("-" * 70)
        
        self.benchmark_key_headers()
        self.analyze_preprocessing_bloat()
        self.measure_full_build_performance()
        self.generate_optimization_report()
        self.save_results()
        
        print("\n✅ Benchmark complete!")


def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        # Default to current directory's parent (assuming we're in tools/)
        project_root = Path(__file__).parent.parent
    
    benchmark = CompilationBenchmark(project_root)
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
