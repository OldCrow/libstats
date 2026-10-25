# 2026-09-30 Zen 4 — compiler vs tier: corvus v1.0.1, clang-cl capped at AVX2

Separates the two causes folded into the MSVC-corvus column of
`../2026-09-30-zen4-quiet-warm/`. corvus at tag `v1.0.1`, clang-cl 22.1.3,
Ninja Release, `CORVUS_DISABLED_TARGETS` removing every AVX-512 target
(`build_corvus_avx2cap.ps1`); Highway 1.4.0 from the clang-cl prefix.
`corvus_scaling_bench` compiled with `cl` and the flags of the other bench
binaries, warm-up 25 s. Runner: corvus `tools/quiet_bench.ps1`; gate passed
at 3.56%, noise 4.37% before and 6.87% after — the after-sample is over the
5% gate, so read to two digits. The binary reports `corvus AVX2`.

Per element at n = 65536, ns:

| | clang-cl `AVX3_ZEN4` | clang-cl `AVX2` (this run) | MSVC `AVX2` | tier | compiler |
|---|---:|---:|---:|---:|---:|
| erf | 2.4 | 3.4 | 12.2 | 1.4× | 3.6× |
| exp | 3.1 | 5.1 | 23.0 | 1.6× | 4.5× |
| lgamma | 28.7 | 44.3 | 134 | 1.5× | 3.0× |
| gamma_p | 90 | 142 | 2,722 | 1.6× | 19× |
| beta_p | 652 | 973 | 8,069 | 1.5× | 8.3× |
| gamma_p_inv | 1,094 | 1,664 | 10,699 | 1.5× | 6.4× |

Single call, n = 1: gamma_p 1,134, beta_p 2,303, gamma_p_inv 5,412 ns
(clang-cl AVX-512: 1,403 / 2,772 / 6,754; MSVC AVX2: 10,990 / 19,806 /
31,768).

**The tier costs 1.4–1.6×**, matching corvus's own 2026-07-24 figure.
**MSVC code generation costs 3–19×**, most on the incomplete gamma/beta
kernels. A clang-cl AVX2 corvus on this CPU is also within 7% of Kaby
Lake's native AVX2 per-element numbers for gamma_p, beta_p and gamma_p_inv
(152 / 1,024 / 1,886) and about twice as fast on erf, exp and lgamma.

Accuracy is not part of this trade: the libstats sweep is bit-identical
between the clang-cl and MSVC corvus builds on all 9798 rows (`PLAN.md`
Next Steps 6(c)(i), Zen 4 leg).
