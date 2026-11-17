# Ivy Bridge AVX at the corvus v1.2.0 pin, b766e45 (2026-11-17)

The Ivy Bridge leg of the corvus pin bump (standards workplan task 2), on
`dev/v3.0.0-corvus-adoption` at `b766e45`: corvus v1.2.0 and Highway 1.4.0
fetched, `gamma_p_inv` on corvus. Same machine and toolchain as
`../2026-11-16-ivybridge-avx-baseline/`, which is `main` one day earlier and
the reference for the comparison. Correctness and accuracy only: the machine
was in use and an oracle ran alongside the test runs, so nothing here is a
timing record. Machine and toolchain: `environment.txt`.

## Builds

Ninja, Release, one tree per compiler: `gcc13/` is GCC 13.3, `clang18/` is
Clang 18.1.3 with libstdc++. Both: every target builds, 0 warnings from the
adoption's own changes (Clang still reports `kBudgetTight` unused in
`tests/test_trig_ulp_gates.cpp`, as on `main`). libstats' active tier is AVX
(`system_inspector --quick`); corvus reports version 1.2.0 and active target
`SSE4`.

## Results (identical under both compilers unless stated)

| File | What | Result |
|---|---|---|
| `ctest.txt` | `ctest -LE "timing\|benchmark" -j3` | 132/132. |
| `test_trig_ulp_gates.txt` | cos/sin ULP gate | 14/14. AVX cos max 1 ULP, mean 0.0278; sin max 1 ULP, mean 0.0248. |
| `simd_verification.txt` | `tools/simd_verification` | 140 pass, 0 fail. |
| (not stored) | `tools/accuracy_sweep` | 26,882 rows, banner `commit=b766e45 isa=AVX date=2026-11-17`. |
| `gcc13/oracle.txt` | `tools/accuracy_vs_mpmath.py` (mpmath 1.4.1, dps 50) | 190 self-checks passed, 0 contract violations, 0 oracle fallbacks. |

The sweep CSV is not stored (2.6 MB, over the repo's 1000 KB file limit); it
is deterministic at this commit on this machine. SHA-256 of the GCC sweep:
`4c744eaafef1e05446a99d18d3feb2b8e5d1fefaba9f156ab33a07599469c23a`.
The Clang sweep differs from it only in the four rows of
`clang18/sweep_diff_vs_gcc13.txt` (scalar Beta `pdf`/`logpdf` at x = NaN; the
NaN's sign bit), so the oracle ran on the GCC sweep only. Its generated block
is the `isa=AVX` block in `docs/ACCURACY_CHARACTERIZATION.md`, with the delta
against `main`'s block in that file's "v3.0.0 adoption" section.

## Indicative only

The suite took 340 s (GCC) and 331 s (Clang) against about 126 s on `main`,
nearly all of it `test_concurrency_stress` (289 s against 106 s). Loaded
machine; a number to chase for #156, not a record.

## Not run

The 14 timing tests, the batch speed gates and every bench.
