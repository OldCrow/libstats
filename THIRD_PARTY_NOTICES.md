# Third Party Notices

This project incorporates or derives work from the following third-party projects.

---

## corvus

**Project**: https://github.com/OldCrow/corvus
**License**: MIT
**Usage**: The special-function and elementary-function engine since v2.5.0
— erf/erfc and their inverses, lgamma/lbeta, digamma/trigamma, the
regularized incomplete gamma and beta functions and their inverses, the
modified Bessel functions I₀/I₁ (scaled and unscaled), and exp/log/cos/sin
behind `VectorOps`. libstats does not bundle corvus: the build uses a
system-installed copy or fetches the pinned release at configure time
(`cmake/FindOrFetchCorvus.cmake`). libstats ships no binaries, so
attribution is its only obligation; a distribution that includes corvus
object code must retain corvus's copyright notice and MIT text.

---

## Google Highway (through corvus)

**Project**: https://github.com/google/highway
**License**: Apache License 2.0 (dual-licensed with BSD 3-Clause; corvus
elects Apache-2.0, and so does libstats)
**Usage**: corvus's SIMD portability layer; every corvus kernel libstats
calls runs on Highway. Not bundled: corvus uses a system-installed Highway
or fetches its pinned release at configure time. A distribution that
includes Highway object code — any static or shared libstats binary linked
against corvus — must retain Highway's copyright notice and include a copy
of the Apache License 2.0 (section 4). Highway publishes no NOTICE file, so
section 4(d) requires no further attribution text.

---

## Retired lineages (through v2.4.1)

The local kernels these notices covered were removed in v2.5.0; nothing
derived from them ships. The entries are kept in the git history of this
file, and the derivations in `docs/NEON_*_DERIVATION.md` /
`docs/NEON_*_DIVERGENCE_AUDIT.md`, as the provenance record:

- **SLEEF** (Boost Software License 1.0) — polynomial coefficients and
  range-reduction constants in the x86 `vector_exp`/`vector_log` kernels.
- **ARM optimized-routines** (MIT) — the N=128 table-based NEON `exp`
  (Issue #33 Q1).
- **`vector_erf_neon`** — independently derived (Issue #67), documented
  only to record the resolution of its former glibc (LGPL-2.1+) lineage.
- **Abramowitz & Stegun** (public domain) — the Bessel I₀/I₁ Tier 2
  polynomials and the §26.2.23 inverse-normal seed.
- **musl libc / FreeBSD msun `s_erf.c`** (MIT; Sun fdlibm lineage) — the
  four-region rational-polynomial `erf` in the x86 tiers.

---

## Numerical Recipes (algorithmic reference only)

**Reference**: Press, W.H. et al., *Numerical Recipes in C++*, Cambridge
University Press, 3rd edition 2007.

**Usage**: The following functions in `src/math_utils.cpp` use the same
standard numerical methods described in Numerical Recipes:
- `erf_inv` — rational approximation for the inverse error function
  (Moro/Acklam method, also described in NR)
- `beta_continued_fraction` — Lentz modified continued-fraction evaluation
  of the regularized incomplete beta function
- `gamma_p_series` — power-series expansion of the regularized incomplete
  gamma function

These algorithms are classical results (Lentz 1976; series predating NR)
and are described in many references. The implementations in this library
are original code written independently; no code was copied from the NR
publication. NR is listed here as an acknowledgment of the algorithmic
references consulted during development.

---

## Design references (no code derived)

The following libraries were studied for design patterns and performance
techniques during development. No algorithms, coefficients, or code from
these projects appear in libstats.

- **EVE** (Expressive Vector Engine) — https://github.com/jfalcou/eve
- **Google Highway** — https://github.com/google/highway
