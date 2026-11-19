// src/portable_fma.cpp -- see portable_fma.h. Compiled with floating-point contraction off
// (CMakeLists.txt): every product below must stay a rounded product.

#include "portable_fma.h"

#include "libstats/platform/cpu_detection.h"

#include <bit>
#include <cstdint>

namespace stats {
namespace detail {

namespace {

// Veltkamp split and Dekker product: p + e == a * b exactly.
inline void twoProduct(double a, double b, double& p, double& e) noexcept {
    constexpr double kSplit = 134217729.0;  // 2^27 + 1
    p = a * b;
    double ah = kSplit * a;
    ah = ah - (ah - a);
    const double al = a - ah;
    double bh = kSplit * b;
    bh = bh - (bh - b);
    const double bl = b - bh;
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
}

// Knuth: s + e == a + b exactly.
inline void twoSum(double a, double b, double& s, double& e) noexcept {
    s = a + b;
    const double bb = s - a;
    e = (a - (s - bb)) + (b - bb);
}

// a + b rounded to odd: the nearest-even sum, moved one representable step toward the exact
// value when it is inexact and its last bit is even.
[[nodiscard]] inline double sumRoundedToOdd(double a, double b) noexcept {
    double s = 0.0;
    double e = 0.0;
    twoSum(a, b, s, e);
    if (e != 0.0) {
        auto bits = std::bit_cast<std::int64_t>(s);
        if ((bits & 1) == 0) {
            bits += ((e > 0.0) == (s > 0.0)) ? 1 : -1;
            s = std::bit_cast<double>(bits);
        }
    }
    return s;
}

// The transforms are exact while no intermediate overflows or leaves the normal range. With
// |a| and |b| in [2^-400, 2^400] the product is in [2^-800, 2^800] and its low part is a multiple
// of 2^-906; with c zero or in [2^-800, 2^900] every sum's low part is a multiple of 2^-906 too.
[[nodiscard]] inline bool inExactWindow(double a, double b, double c) noexcept {
    constexpr double kFactorMin = 0x1p-400;
    constexpr double kFactorMax = 0x1p+400;
    constexpr double kAddendMin = 0x1p-800;
    constexpr double kAddendMax = 0x1p+900;
    const double fa = std::fabs(a);
    const double fb = std::fabs(b);
    const double fc = std::fabs(c);
    return fa >= kFactorMin && fa <= kFactorMax && fb >= kFactorMin && fb <= kFactorMax &&
           (fc == 0.0 || (fc >= kAddendMin && fc <= kAddendMax));
}

}  // namespace

double fma_software(double a, double b, double c) noexcept {
    if (!inExactWindow(a, b, c))
        return std::fma(a, b, c);  // zeros, subnormals, huge values, inf and NaN
    double uh = 0.0, ul = 0.0, th = 0.0, tl = 0.0, vh = 0.0, vl = 0.0;
    twoProduct(a, b, uh, ul);
    twoSum(c, ul, th, tl);
    twoSum(uh, th, vh, vl);
    return vh + sumRoundedToOdd(vl, tl);
}

#if !(defined(__FMA__) || defined(__AVX2__) || defined(__aarch64__) || defined(_M_ARM64))
double fma_portable(double a, double b, double c) noexcept {
    static const bool hardware = arch::supports_fma();
    return hardware ? std::fma(a, b, c) : fma_software(a, b, c);
}
#endif

}  // namespace detail
}  // namespace stats
