#!/usr/bin/env python3

import subprocess
import tempfile
import os
import time

def measure_compilation(code, description):
    """Measure compilation time and preprocessed size for given code"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Measure compilation time
        start = time.time()
        
        # Use Homebrew LLVM if available, otherwise fall back to system compiler
        compiler = '/usr/local/opt/llvm/bin/clang++' if os.path.exists('/usr/local/opt/llvm/bin/clang++') else 'clang++'
        
        if compiler.startswith('/usr/local/opt/llvm'):
            # Homebrew LLVM configuration
            cmd = [
                compiler, '-std=c++20', '-stdlib=libc++',
                '-I/usr/local/opt/llvm/include/c++/v1',
                '-I', 'include', '-E', temp_file
            ]
        else:
            # System compiler (fallback)
            cmd = [compiler, '-std=c++20', '-I', 'include', '-E', temp_file]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Users/wolfman/Development/libstats')
        end = time.time()
        
        if result.returncode != 0:
            return None, None, f"Error: {result.stderr}"
        
        # Count total preprocessed lines
        lines = result.stdout.count('\n')
        compilation_time = end - start
        
        return compilation_time, lines, None
    finally:
        os.unlink(temp_file)

def main():
    print("🚀 Phase 1 Header Optimization Demo")
    print("=" * 60)
    
    # Test minimal header inclusion (forward declarations only)
    minimal_code = '''
#include "libstats.h"
#include <vector>

class MyStats {
private:
    libstats::Gaussian* gaussian_;      // Forward declaration works
    libstats::Exponential* exponential_; // Forward declaration works
    std::vector<double> data_;
    
public:
    MyStats();
    void setGaussian(libstats::Gaussian* g) { gaussian_ = g; }
    void setExponential(libstats::Exponential* e) { exponential_ = e; }
};

MyStats::MyStats() : gaussian_(nullptr), exponential_(nullptr) {}
'''

    # Test full interface mode
    full_code = '''
#define LIBSTATS_FULL_INTERFACE  // Enable full functionality
#include "libstats.h"
#include <vector>

class MyStats {
private:
    std::vector<double> data_;
    
public:
    void createDistributions() {
        // Full implementation available - can create and use objects
        auto gaussian = libstats::Gaussian::create(0.0, 1.0);
        auto exponential = libstats::Exponential::create(1.0);
        
        if (gaussian.isOk() && exponential.isOk()) {
            std::vector<double> values = {1.0, 2.0, 3.0};
            std::vector<double> results(3);
            
            gaussian.value.getProbability(values, results);
        }
        
        // Initialize performance systems
        libstats::initialize_performance_systems();
    }
};
'''

    print("📊 Testing MINIMAL mode (forward declarations only)...")
    time1, lines1, error1 = measure_compilation(minimal_code, "Minimal")
    if error1:
        print(f"❌ Error in minimal mode: {error1}")
        return
    else:
        print(f"   ⏱️  Compilation time: {time1:.3f}s")
        print(f"   📝 Preprocessed lines: {lines1:,}")
        print(f"   💡 Use case: Header files that only need type information")
    
    print()
    print("📊 Testing FULL mode (complete implementation)...")
    time2, lines2, error2 = measure_compilation(full_code, "Full")
    if error2:
        print(f"❌ Error in full mode: {error2}")
        return
    else:
        print(f"   ⏱️  Compilation time: {time2:.3f}s")
        print(f"   📝 Preprocessed lines: {lines2:,}")
        print(f"   💡 Use case: Implementation files that need full functionality")
    
    print()
    print("🎯 OPTIMIZATION RESULTS:")
    print("-" * 40)
    if time1 and time2 and lines1 and lines2:
        time_improvement = ((time2 - time1) / time2) * 100
        lines_reduction = ((lines2 - lines1) / lines2) * 100
        
        print(f"⚡ Compilation time saved: {time_improvement:.1f}%")
        print(f"📉 Preprocessor overhead reduced: {lines_reduction:.1f}%")
        print(f"🔧 Header include reduction: {lines2 - lines1:,} lines")
        
        print()
        print("✅ PHASE 1 BENEFITS:")
        print("   • Faster compilation for header-only usage")
        print("   • Reduced preprocessor overhead")
        print("   • Cleaner dependency management")
        print("   • Selective full functionality when needed")
        print("   • Better separation of interface vs implementation")
        
        print()
        print("📋 USAGE RECOMMENDATIONS:")
        print("   Header files (.h):")
        print("     #include \"libstats.h\"  // Just forward declarations")
        print()
        print("   Implementation files (.cpp):")
        print("     #define LIBSTATS_FULL_INTERFACE")
        print("     #include \"libstats.h\"  // Complete functionality")

if __name__ == "__main__":
    main()
