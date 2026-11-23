# M1 NEON quiet session (L4) at cfbb992, 2026-11-21/22

The M1 leg's L4 on `dev/v3.0.0-corvus-adoption` at `cfbb992` (corvus v1.2.0
fetched, Highway 1.4.0 from Homebrew), against `main` at `99f2175`. Apple M1,
macOS 27.2.1, AppleClang 21.0.0, Release, Ninja; active tier NEON
(`system_inspector.txt`). Driver: `l4_quiet_m_20261121.sh`, under
`caffeinate -is`, each phase gated on ambient CPU below 15% over two 10 s
windows (20% after a 3 h wait). `run.txt` is the phase log,
`gate_samples.txt` every refused gate sample.

## Results

| File | What | Result |
|---|---|---|
| `timing/timing.txt` | `ctest -j1 -L timing` | 14/14. Gate passed at the relaxed 20% (ambient 4.0%). |
| `simd/simd_verification.txt` | `tools/simd_verification` | exit 0, 0 failed. |
| `bench/*_pass{1,2,3}.txt` | `distributions_bench`, `elementary_bench`, main and dev alternately | `compare.txt`: medians of three passes and dev:main ratios. |

Bench binaries: `-O3 -DNDEBUG -std=c++20 -arch arm64 -stdlib=libc++
-DLIBSTATS_HAS_NEON=1`, each linked against its tree's Release `libstats.a`
(dev also `libcorvus.a`, `-lhwy -lhwy_contrib`).

Bench conditions: pass 1 started at the relaxed gate (18.7%) and Spotlight
and `mediaanalysisd` ran through it (`bench/noise_pass1.txt`; load 3 to 18),
so its `elementary_bench` n=1000 figures read about 2x high. Passes 2 and 3
started below 15%; `mediaanalysisd` ran on one core during parts of both.
The medians in `compare.txt` agree with passes 2 and 3 to a few percent.

What the bench shows for #156 (dev:main, median): incomplete-gamma and
incomplete-beta paths cost more on the branch — batch CDF gamma 2.5x, beta
11.7x, Poisson 5.0x, binomial 6.5x; scalar CDF gamma 3.6x, beta 14.4x,
Poisson 5.6x, Student-t 5.1x, binomial 9.5x; quantile gamma 8.8x, beta 12.6x,
Poisson 4.8x, Student-t 2.3x, binomial 7.8x. Densities and the closed-form
distributions are level or faster (Poisson and binomial PMF 0.7x to 0.9x,
Gaussian quantile 0.75x, LogNormal quantile 0.59x). Elementary kernels are
level.

## Profile runs

The three `strategy_profile --large` runs did not start in this session: no
gate sample fell below 20% between 02:30 and the 12:00 deadline on
2026-11-22 (`mediaanalysisd`, with the screen locked). `l4_profile_m_20261122.sh`
relaunched that phase alone at 12:03 with the same gate; `run.txt` here ends
at the relaunch. The profile bundle goes under `data/profiles/dispatcher/`.

## Timing method

Every program here is cfbb992's, before the per-cell steady-state timing of
`8c55bfc` (#320). On Ivy Bridge that method let a cell read the CPU state its
neighbour left behind. Whether the M1 has the same bias is not measured; the
ratios of 2x and more above do not depend on it, and the ones within 0.6x to
1.25x should not be quoted on #156 until they are re-timed.
