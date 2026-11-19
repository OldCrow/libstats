#pragma once

// src/portable_fma.h -- private to the library's translation units; not installed.
//
// detail::fma_portable(a, b, c) is std::fma(a, b, c), bit for bit, without the cost std::fma has
// on a CPU that lacks the FMA instruction set. There the C library's fma is a software routine:
// 175 ns per call on an Ivy Bridge under glibc 2.39, which made two calls per element the whole
// 350 ns of InverseGamma's density (#311).
//
// - Where the translation unit is compiled for FMA hardware (AArch64; x86 with FMA or AVX2
//   enabled) it is std::fma inline, one instruction, as before.
// - Otherwise it is an out-of-line call that asks the CPU once: std::fma where the CPU has FMA
//   (the C library then uses the instruction), detail::fma_software where it does not.
//
// The error-free transforms that call this (the rate-product residuals in gamma.cpp and
// inverse_gamma.cpp, the double-double products in von_mises.cpp and truncated_normal.cpp) need
// the correctly rounded result, so fma_software is a correctly rounded fma and not a
// two-product shortcut: the callers' proofs hold unchanged on every tier.

#include <cmath>

namespace stats {
namespace detail {

/// a * b + c correctly rounded, computed without an FMA instruction (Boldo and Melquiond,
/// "Emulation of a FMA and correctly rounded sums: proved algorithms using rounding to odd",
/// IEEE Trans. Computers 57(4), 2008). Outside the exponent window in which its error-free
/// transforms cannot overflow or underflow it calls std::fma, so the result is std::fma's for
/// every input, non-finite ones included.
[[nodiscard]] double fma_software(double a, double b, double c) noexcept;

#if defined(__FMA__) || defined(__AVX2__) || defined(__aarch64__) || defined(_M_ARM64)
[[nodiscard]] inline double fma_portable(double a, double b, double c) noexcept {
    return std::fma(a, b, c);
}
#else
[[nodiscard]] double fma_portable(double a, double b, double c) noexcept;
#endif

}  // namespace detail
}  // namespace stats
