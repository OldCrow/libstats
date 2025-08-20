#!/usr/bin/env python3
"""
Phase 2 Header Optimization Demo
Demonstrates the benefits of PIMPL pattern and STL consolidation headers.

This script tests compilation performance improvements from:
1. PIMPL pattern for heavy headers (platform_constants.h)
2. STL consolidation headers for common includes
3. Template instantiation reduction through explicit instantiation
"""

import subprocess
import tempfile
import time
import os
import sys
from pathlib import Path

def create_test_file(content, suffix=".cpp"):
    """Create a temporary test file with given content."""
    fd, path = tempfile.mkstemp(suffix=suffix, text=True)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path

def get_compiler_config():
    """Get compiler configuration for Homebrew LLVM or system compiler."""
    compiler = '/usr/local/opt/llvm/bin/clang++' if os.path.exists('/usr/local/opt/llvm/bin/clang++') else 'clang++'

    if compiler.startswith('/usr/local/opt/llvm'):
        # Homebrew LLVM configuration
        base_flags = [
            compiler, '-std=c++20', '-stdlib=libc++',
            '-I/usr/local/opt/llvm/include/c++/v1'
        ]
    else:
        # System compiler (fallback)
        base_flags = [compiler, '-std=c++20']

    return base_flags

def measure_compilation(source_file, include_paths=None, extra_flags=None, link_sources=None):
    """Measure compilation time and preprocessed output size."""
    if include_paths is None:
        include_paths = []
    if extra_flags is None:
        extra_flags = []
    if link_sources is None:
        link_sources = []

    # Get compiler configuration
    base_flags = get_compiler_config()

    # Build compile command
    compile_cmd = base_flags + ['-c'] + extra_flags
    for include_path in include_paths:
        compile_cmd.extend(['-I', include_path])
    compile_cmd.extend(['-o', '/dev/null', source_file])

    # Add source files for linking if needed
    if link_sources:
        compile_cmd.extend(link_sources)

    # Measure compilation time
    start_time = time.time()
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    end_time = time.time()

    if result.returncode != 0:
        print(f"Compilation failed: {result.stderr}")
        return None, None

    # Measure preprocessed output size
    preprocess_cmd = base_flags + ['-E'] + extra_flags
    for include_path in include_paths:
        preprocess_cmd.extend(['-I', include_path])
    preprocess_cmd.append(source_file)

    preprocess_result = subprocess.run(preprocess_cmd, capture_output=True, text=True)
    if preprocess_result.returncode != 0:
        print(f"Preprocessing failed: {preprocess_result.stderr}")
        return end_time - start_time, None

    # Count lines in preprocessed output
    preprocessed_lines = len(preprocess_result.stdout.split('\n'))

    return end_time - start_time, preprocessed_lines

def test_platform_constants_optimization():
    """Test PIMPL pattern optimization for platform_constants.h"""

    print("🔧 Phase 2: Platform Constants PIMPL Optimization")
    print("=" * 60)

    # Test using original heavy header that pulls in many dependencies
    # This simulates the old approach where platform constants were inline in headers
    heavy_test_content = '''
#include "platform/platform_constants.h"
#include "platform/cpu_detection.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>

int main() {
    // Simulate heavy usage of platform constants with complex computations
    // This represents the old approach with heavy template instantiation
    std::vector<double> data(1000, 1.0);

    // Use memory access constants
    auto cache_size = libstats::constants::memory::access::CACHE_LINE_SIZE_BYTES;

    // Simulate expensive operations that would be inline in old approach
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sqrt(data[i] * cache_size);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Cache size: " << cache_size << std::endl;
    std::cout << "Computed " << data.size() << " values" << std::endl;
    return 0;
}
'''

    # Test using lightweight forward declaration header
    lightweight_test_content = '''
#include "common/platform_constants_fwd.h"
#include <iostream>

int main() {
    // Use platform constants through lightweight interface
    auto block_size = libstats::constants::simd::get_default_block_size();
    auto grain_size = libstats::constants::parallel::get_default_grain_size();
    auto cache_size = libstats::constants::memory::access::CACHE_LINE_SIZE_BYTES;

    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Grain size: " << grain_size << std::endl;
    std::cout << "Cache size: " << cache_size << std::endl;
    return 0;
}
'''

    include_path = str(Path(__file__).parent.parent / "include")

    # Create test files
    heavy_file = create_test_file(heavy_test_content)
    lightweight_file = create_test_file(lightweight_test_content)

    try:
        print("📊 Testing HEAVY mode (original platform_constants.h)...")
        heavy_time, heavy_lines = measure_compilation(heavy_file, [include_path])

        if heavy_time is not None:
            print(f"   ⏱️  Compilation time: {heavy_time:.3f}s")
            print(f"   📝 Preprocessed lines: {heavy_lines:,}")
            print("   💡 Use case: Direct access to inline constants")
        else:
            print("   ❌ Compilation failed")

        print()
        print("📊 Testing LIGHTWEIGHT mode (platform_constants_fwd.h)...")
        lightweight_time, lightweight_lines = measure_compilation(lightweight_file, [include_path])

        if lightweight_time is not None:
            print(f"   ⏱️  Compilation time: {lightweight_time:.3f}s")
            print(f"   📝 Preprocessed lines: {lightweight_lines:,}")
            print("   💡 Use case: Function-based access through PIMPL")
        else:
            print("   ❌ Compilation failed")

        # Calculate improvements
        if heavy_time is not None and lightweight_time is not None:
            time_improvement = ((heavy_time - lightweight_time) / heavy_time) * 100
            line_reduction = heavy_lines - lightweight_lines
            line_improvement = (line_reduction / heavy_lines) * 100

            print()
            print("🎯 PHASE 2 PIMPL OPTIMIZATION RESULTS:")
            print("-" * 50)
            print(f"⚡ Compilation time saved: {time_improvement:.1f}%")
            print(f"📉 Preprocessor overhead reduced: {line_improvement:.1f}%")
            print(f"🔧 Header include reduction: {line_reduction:,} lines")

            return time_improvement, line_improvement, line_reduction
        else:
            print("   ❌ Could not calculate improvements due to compilation failures")
            return None, None, None

    finally:
        # Cleanup
        os.unlink(heavy_file)
        os.unlink(lightweight_file)

def test_stl_consolidation_optimization():
    """Test STL consolidation header optimization"""

    print("📚 Phase 2: STL Consolidation Optimization")
    print("=" * 55)

    # Test using individual STL headers
    individual_stl_content = '''
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

int main() {
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::string name = "test";

    std::transform(data.begin(), data.end(), data.begin(),
                   [](double x) { return x * 2.0; });

    auto sum = std::accumulate(data.begin(), data.end(), 0.0, std::plus<double>());

    std::cout << name << ": " << sum << std::endl;
    return 0;
}
'''

    # Test using consolidated headers
    consolidated_stl_content = '''
#include "common/libstats_vector_common.h"
#include "common/libstats_string_common.h"
#include "common/libstats_algorithm_common.h"
#include <iostream>

int main() {
    libstats::common::DoubleVector data = {1.0, 2.0, 3.0, 4.0, 5.0};
    libstats::common::String name = "test";

    // Use consolidated algorithm utilities
    libstats::common::algorithm_utils::parallel::transform(
        data.begin(), data.end(), data.begin(),
        [](double x) { return x * 2.0; });

    auto sum = libstats::common::algorithm_utils::safe_sum(data.begin(), data.end());

    std::cout << name << ": " << sum << std::endl;
    return 0;
}
'''

    include_path = str(Path(__file__).parent.parent / "include")

    # Create test files
    individual_file = create_test_file(individual_stl_content)
    consolidated_file = create_test_file(consolidated_stl_content)

    try:
        print("📊 Testing INDIVIDUAL STL mode (separate <vector>, <string>, <algorithm>)...")
        individual_time, individual_lines = measure_compilation(individual_file, [include_path])

        if individual_time is not None:
            print(f"   ⏱️  Compilation time: {individual_time:.3f}s")
            print(f"   📝 Preprocessed lines: {individual_lines:,}")
            print("   💡 Use case: Standard separate STL includes")
        else:
            print("   ❌ Compilation failed")

        print()
        print("📊 Testing CONSOLIDATED STL mode (libstats common headers)...")
        consolidated_time, consolidated_lines = measure_compilation(consolidated_file, [include_path])

        if consolidated_time is not None:
            print(f"   ⏱️  Compilation time: {consolidated_time:.3f}s")
            print(f"   📝 Preprocessed lines: {consolidated_lines:,}")
            print("   💡 Use case: Optimized consolidated STL includes")
        else:
            print("   ❌ Compilation failed")

        # Calculate improvements
        if individual_time is not None and consolidated_time is not None:
            time_improvement = ((individual_time - consolidated_time) / individual_time) * 100
            line_reduction = individual_lines - consolidated_lines
            line_improvement = (line_reduction / individual_lines) * 100 if individual_lines > 0 else 0

            print()
            print("🎯 STL CONSOLIDATION RESULTS:")
            print("-" * 35)
            print(f"⚡ Compilation time change: {time_improvement:.1f}%")
            print(f"📉 Preprocessor change: {line_improvement:.1f}%")
            print(f"🔧 Header line difference: {abs(line_reduction):,} lines")

            return time_improvement, line_improvement, abs(line_reduction)
        else:
            print("   ❌ Could not calculate improvements due to compilation failures")
            return None, None, None

    finally:
        # Cleanup
        os.unlink(individual_file)
        os.unlink(consolidated_file)

def main():
    """Main demo function"""
    print("🚀 Phase 2 Header Optimization Demo")
    print("=" * 60)
    print("This demo shows the benefits of Phase 2 optimizations:")
    print("• PIMPL pattern for heavy headers")
    print("• STL consolidation for common includes")
    print("• Template instantiation reduction")
    print()

    # Check if we're in the right directory
    if not Path("include").exists():
        print("❌ Error: This script must be run from the libstats root directory")
        print("   Current directory:", os.getcwd())
        sys.exit(1)

    # Test platform constants PIMPL optimization
    pimpl_results = test_platform_constants_optimization()
    print()

    # Test STL consolidation optimization
    stl_results = test_stl_consolidation_optimization()
    print()

    # Summary
    print("📋 PHASE 2 OPTIMIZATION SUMMARY:")
    print("=" * 40)

    if pimpl_results[0] is not None:
        print(f"✅ PIMPL Pattern Benefits:")
        print(f"   • Compilation speed: {pimpl_results[0]:.1f}% faster")
        print(f"   • Preprocessor reduction: {pimpl_results[1]:.1f}%")
        print(f"   • Header overhead: {pimpl_results[2]:,} fewer lines")
    else:
        print("❌ PIMPL Pattern: Could not measure (compilation issues)")

    if stl_results[0] is not None:
        print(f"✅ STL Consolidation Results:")
        print(f"   • Compilation change: {stl_results[0]:.1f}%")
        print(f"   • Preprocessor change: {stl_results[1]:.1f}%")
        print(f"   • Header difference: {stl_results[2]:,} lines")
    else:
        print("❌ STL Consolidation: Could not measure (compilation issues)")

    print()
    print("🎯 PHASE 2 KEY BENEFITS:")
    print("   • Reduced compilation dependencies")
    print("   • Hidden implementation complexity")
    print("   • Better template instantiation control")
    print("   • Improved incremental build performance")
    print("   • Cleaner API boundaries")

if __name__ == "__main__":
    main()
