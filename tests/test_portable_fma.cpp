// tests/test_portable_fma.cpp
//
// detail::fma_software must return std::fma's bits for every input (#311): the library's
// error-free transforms call it on CPUs without the FMA instruction set, and their proofs assume
// a correctly rounded a * b + c. The reference is std::fma, which is the hardware instruction on
// most machines and the C library's correctly rounded routine elsewhere.
//
// The input classes are the ones a wrong emulation gets wrong. A version that adds the two low
// parts in round-to-nearest instead of round-to-odd passes the first seven classes and fails
// about 1.4% of the tie classes (154,444 of 11,000,000 when this test was written), so those
// classes are what gives the zero below its meaning.

#include "portable_fma.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>

using stats::detail::fma_portable;
using stats::detail::fma_software;

namespace {

[[nodiscard]] bool sameBits(double x, double y) {
    return std::bit_cast<std::uint64_t>(x) == std::bit_cast<std::uint64_t>(y) ||
           (std::isnan(x) && std::isnan(y));
}

}  // namespace

TEST(PortableFma, SoftwareMatchesStdFmaBitForBit) {
    std::mt19937_64 gen(311);
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::uniform_int_distribution<int> wide(-300, 300);
    std::uniform_int_distribution<int> near(-60, 60);
    std::uniform_int_distribution<std::uint64_t> bits;
    const auto scaled = [&](int exponent) { return std::ldexp(unit(gen), exponent); };
    const auto mantissa = [&] { return 1.0 + std::fabs(unit(gen)); };

    constexpr int kCases = 1'100'000;  // 100,000 of each class
    int mismatches = 0;
    for (int i = 0; i < kCases; ++i) {
        double a = 0.0, b = 0.0, c = 0.0;
        switch (i % 11) {
            case 0:  // unrelated magnitudes
                a = scaled(wide(gen));
                b = scaled(wide(gen));
                c = scaled(wide(gen));
                break;
            case 1:  // the exact residual of a rounded product
                a = scaled(near(gen));
                b = scaled(near(gen));
                c = -(a * b);
                break;
            case 2:  // near cancellation
                a = scaled(near(gen));
                b = scaled(near(gen));
                c = -(a * b) * (1.0 + std::ldexp(unit(gen), -40));
                break;
            case 3:  // reciprocal residual, as InverseGamma forms it
                a = scaled(near(gen));
                b = 1.0 / a;
                c = -1.0;
                break;
            case 4:  // the addend at every binary offset from the product
                a = scaled(near(gen));
                b = scaled(near(gen));
                c = std::ldexp(a * b, static_cast<int>(bits(gen) % 120) - 60);
                break;
            case 5:  // short mantissas, so sums land on representable ties
                a = std::ldexp(static_cast<double>((bits(gen) & 0x3ffffff) | 1), near(gen));
                b = std::ldexp(static_cast<double>((bits(gen) & 0x3ffffff) | 1), near(gen));
                c = std::ldexp(static_cast<double>((bits(gen) & 0xfffff) | 1), near(gen));
                break;
            case 6: {  // the addend one step either side of minus the rounded product
                a = scaled(near(gen));
                b = scaled(near(gen));
                const double p = a * b;
                c = std::nextafter(-p, (bits(gen) & 1) != 0 ? 1e300 : -1e300);
                break;
            }
            case 7: {  // the product within 2^-53 of an odd multiple of half an ulp of c
                c = scaled(near(gen));
                if (c == 0.0)
                    c = 1.0;
                const double half_ulp = std::ldexp(1.0, std::ilogb(c) - 53) *
                                        static_cast<double>(2 * (bits(gen) % 8) + 1);
                a = std::ldexp(mantissa(), near(gen));
                b = half_ulp / a;
                if ((bits(gen) & 1) != 0)
                    b = -b;
                break;
            }
            case 8: {  // the same beside a binade edge of c
                c = std::ldexp(1.0, near(gen));
                c = std::nextafter(c, (bits(gen) & 1) != 0 ? 0.0 : 1e300);
                const double half_ulp = std::ldexp(1.0, std::ilogb(c) - 53) *
                                        static_cast<double>(2 * (bits(gen) % 4) + 1);
                a = std::ldexp(mantissa(), near(gen));
                b = std::nextafter(half_ulp / a, (bits(gen) & 1) != 0 ? 0.0 : 1e300);
                if ((bits(gen) & 2) != 0)
                    b = -b;
                break;
            }
            case 9: {  // the product's high part cancels c; a tie forms in what is left
                a = std::ldexp(mantissa(), near(gen));
                b = std::ldexp(mantissa(), near(gen));
                const double p = a * b;
                const double step =
                    std::ldexp(1.0, std::ilogb(p) - 53 - static_cast<int>(bits(gen) % 60));
                c = -p + step * static_cast<double>(2 * (bits(gen) % 8) + 1);
                break;
            }
            default: {  // the addend a few steps from 2^53 ulps of the product
                a = scaled(near(gen));
                b = scaled(near(gen));
                const double p = a * b;
                if (p == 0.0)
                    break;
                c = std::ldexp(std::ldexp(1.0, 53) + static_cast<double>(bits(gen) % 7) - 3.0,
                               std::ilogb(p) - 53 - static_cast<int>(bits(gen) % 4));
                if ((bits(gen) & 1) != 0)
                    c = -c;
                break;
            }
        }
        const double expected = std::fma(a, b, c);
        const double actual = fma_software(a, b, c);
        if (!sameBits(actual, expected)) {
            ++mismatches;
            if (mismatches <= 5)
                ADD_FAILURE() << std::hexfloat << "fma_software(" << a << ", " << b << ", " << c
                              << ") = " << actual << ", std::fma = " << expected;
        }
    }
    EXPECT_EQ(mismatches, 0) << "of " << kCases;
}

TEST(PortableFma, SoftwareMatchesStdFmaOutsideTheExactWindow) {
    constexpr double kInf = std::numeric_limits<double>::infinity();
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    constexpr double kMin = std::numeric_limits<double>::min();
    constexpr double kMax = std::numeric_limits<double>::max();
    constexpr double kTiny = std::numeric_limits<double>::denorm_min();
    constexpr double kValues[] = {0.0,      -0.0,     1.0,      -1.0,     3.0,        kTiny,
                                  -kTiny,   kMin,     -kMin,    kMax,     -kMax,      kInf,
                                  -kInf,    kNaN,     0x1p-400, 0x1p+400, 0x1p-401,   0x1p+401,
                                  0x1p-800, 0x1p-801, 0x1p+900, 0x1p+901, 0x1.8p-537, 0x1.8p+511};
    for (double a : kValues)
        for (double b : kValues)
            for (double c : kValues)
                EXPECT_TRUE(sameBits(fma_software(a, b, c), std::fma(a, b, c)))
                    << std::hexfloat << a << " * " << b << " + " << c;
}

TEST(PortableFma, PortableIsStdFma) {
    std::mt19937_64 gen(312);
    std::uniform_real_distribution<double> value(-4.0, 4.0);
    for (int i = 0; i < 20000; ++i) {
        const double a = value(gen);
        const double b = value(gen);
        const double c = (i % 2 == 0) ? -(a * b) : value(gen);
        ASSERT_TRUE(sameBits(fma_portable(a, b, c), std::fma(a, b, c)))
            << std::hexfloat << a << " * " << b << " + " << c;
    }
}
