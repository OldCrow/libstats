# GCC miscompile: overflow guard on a product with a constant (2026-10-13)

Evidence for #275 and the AGENTS.md "Overflow-guard rule". It also serves as a draft
upstream report; nothing has been filed with GCC.

## Reproducer

```cpp
#include <cmath>
#include <cstdio>
__attribute__((noinline)) double f(double ax) {
    if (ax > 700.0) {
        const double p = 6.283185307179586 * ax;
        return std::isfinite(p) ? std::log(p) : std::log(6.283185307179586) + std::log(ax);
    }
    return 0.0;
}
int main() { volatile double x = 3e307; std::printf("%.17g (expect 709.83...)\n", f(x)); }
```

| Compiler | -O0 | -O1 | -O2 |
|---|---|---|---|
| g++ 16.1.0 (Homebrew, x86-64 macOS) | 709.8301129042494 | inf | inf |
| AppleClang 15.0.0 | 709.8301129042494 | 709.8301129042494 | 709.8301129042494 |

The CI gcc-14 leg (Ubuntu) shows the same failure, in libstats' von Mises and
InverseGamma densities. No fast-math or finite-math flag is set. `-fno-tree-vrp`,
`-fno-tree-dominator-opts`, `-fno-thread-jumps` and `-fno-tree-ccp` each leave it
in place.

## What the generated code does

At -O2, `f` keeps the `isfinite(p)` test (`ucomisd` against DBL_MAX). In the
overflow branch it loads the constant +inf (`0x7FF0000000000000`) and calls `log`
on that, not on `ax`. So GCC infers from `p = c·ax` being infinite that `ax` itself
is infinite. That holds for real numbers but not in floating point, where a finite
`ax` above DBL_MAX/c overflows the product. `1/x` behaves the same way: overflow is
taken to mean `x = 0`.

The probe found no fold when both operands are runtime values. Only a product with
a compile-time constant showed it.

## Workaround used in libstats

Test the operand with a strict comparison:
`x < DBL_MAX / c ? log(c * x) : log(c) + log(x)`. A guard whose fallback does not
reuse the operand (it returns a constant) is unaffected.
