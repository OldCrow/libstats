// CPU detection for SIMD - compiled WITHOUT advanced SIMD flags
// This file must be compiled with minimal instruction sets to avoid chicken-and-egg problems

// Explicitly disable advanced instruction sets during compilation of this file
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC push_options
    #pragma GCC target("no-avx512f,no-avx2,no-avx")
#endif

#include "../include/cpu_detection.h"

namespace libstats {
namespace simd {

// Safe CPU detection functions that don't use advanced SIMD instructions
bool cpu_supports_sse2_safe() {
    return cpu::supports_sse2();
}

bool cpu_supports_avx_safe() {
    return cpu::supports_avx();
}

bool cpu_supports_avx2_safe() {
    return cpu::supports_avx2();
}

bool cpu_supports_avx512_safe() {
    return cpu::supports_avx512();
}

bool cpu_supports_neon_safe() {
    return cpu::supports_neon();
}

} // namespace simd
} // namespace libstats

#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC pop_options
#endif
