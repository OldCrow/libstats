# libstats — Plan / Status

State only: what is decided, open, or next. Release contents are in
`CHANGELOG.md`, per-version validation in `docs/VALIDATION_HISTORY.md`,
conventions and commands in `AGENTS.md`. The history this file used to
carry (shipped releases, closed milestones, the Resolved log, the
2026-08-21 defensive review, travel-era records, the corvus spike) is in
`git show f508b12:PLAN.md`.

## Status [DERIVED] — 2026-10-04
**v2.5.0 in progress [OPEN]** on `dev/v2.4.2` (cut from `main` at
`d8d3388`; milestone #10 "v2.5.0 — Correctness & accuracy", #157–#172, which
the release PR closes). This line ships as **v2.5.0**, not v2.4.2
[user, 2026-10-04; Decided]; the branch keeps its name, and "v2.4.2" in
this file names that branch and its code. All milestone code is in, #162 included
(`8018c5d`). The speed work on the over-budget rows landed on
2026-10-04 (Known Gaps, "Over-budget slowdowns"; `b5a47df` through
`5367999`), with `SQRT_PI` corrected (`4ec1950`) and the Beta
large-shape density fixed (`97c67bf`).

**Freeze policy [user, 2026-10-07].** v2.5.0's code freezes for good at
its release and stays frozen until corvus is integrated (v3.0.0). Until
then the freeze head names the code under validation, not a promise that
it is final: a defect that affects accuracy, a speedup or a contract's
validity is fixed, the head moves, and every machine repeats R1, R2's
sweep compare, R8 and R9 there. v2.5.0 releases from the first head that
passes every machine's gates with no such defect open. **Freeze head:
`b0ac2ad`** (2026-10-08: the review fix round, `6a56806`–`b0ac2ad`, over K's
#191 slice-tail fix `045acd9`); #223–#226 and #228 are being fixed next
[user, 2026-10-08], which moves it again.

History: `ee11e62` [user, 2026-10-06] moved the head from `5367999`
by the Windows build fix (`6cab6ed`: `far` is a windows.h macro; a
rename in `gamma.cpp` and `beta.cpp`), the `log1pmx_series` Horner
written out (`998d985`, so MSVC vectorizes the Beta and Gamma Stirling
batches), clang-cl's global flag made cl.exe's `/arch:` (`606607f`), the
parallel-path fixes (`73ecbfe`: #175, #176; one snapshot per batch,
sliced batch kernels) and the audit's lock, TOCTOU and parallel-path
fixes (`ee11e62`; Known Gaps). The first three are bit-identical on
clang or Windows-only, so K's runs at `5367999` and M's at `2c5230a`
stand for R1–R3 [user, 2026-10-04/05]. `73ecbfe` and `ee11e62` are
bit-identical on Z under MSVC and clang-cl (the sweep's result bits);
every machine reruns R1 with the new gates and re-profiles R8 (R8). Verified on K (AppleClang 15,
2026-10-05): the AVX2 sweep at the freeze code is bit-identical to
`4ceafae`'s. CI first ran on this branch at `bb27040` (manual dispatch),
green on all 10 jobs: GCC 14, Linux clang 17, macOS AppleClang, MSVC,
sanitizers, strict `-Werror`. No change to `src/`,
the public headers, `dispatch_thresholds.h` or build flags without the
user's decision; work outside them (issues, tooling under `tools/` and
`scripts/`, docs, cross-repo) continues. A defect found in validation
is stopped and reported; fixing it moves the freeze head and reruns
every machine. The threshold update (R8) is the planned exception, last,
followed by one R1 per machine. At the freeze on K: correctness 89/89,
AVX2 sweep 0 contract violations, warning-clean; a quiet A/B against
`b1ffc16` (scratch, `ab_bench`) put Student-t, Exponential and Rayleigh
back at baseline (1.01–1.11×), kept Weibull 0.55×, Pareto 0.89×, Beta
(2, 1e5) 0.5–0.7×, and measured Beta with both shapes large at logpdf
2.2×, pdf 1.3× (the accuracy fix's cost).

Every machine now runs R1, R2 and R3 at the freeze head (K's earlier
records stand as history) and takes its R8 captures there. Then the runbook
below, docs, and the release. Every fix has a gate shown to fail on the
code before it.

Fixed on the branch [DERIVED] — detail in the commit messages:
- Milestone #157–#161, #163–#167, #170–#172; the FORCE_PARALLEL span
  lambdas that sent NaN out of support.
- Numerics: `solve_concave` behind `detail::gamma_p_inv` (#160) and
  `inverse_t_cdf` (#159, BGRAT tails); Stirling-form prefactors and a 3ε
  stop (#166); stable `lbeta`; Loader-style log-pmfs (#172, no
  third-party code [user]); small-shape `gamma_q`; subnormal probit;
  log1p/expm1 tails (quantiles of Exponential, Rayleigh, Weibull, Pareto;
  the CDFs of Pareto, Weibull, Exponential, Rayleigh); von Mises tail CDF
  and quantile.
- Threading: scalar methods branch on one parameter snapshot (Binomial,
  Poisson, Gamma and Uniform quantiles); noexcept `trySet*` assign under
  one lock.
- Special parameters exact: value paths take λ, α, β = 1 only when exactly 1.
- Build: `run_tests` → `run_tests_correctness` with VERBATIM (VS 2026
  `.slnx`); the dynamic tests' DLL-copy race.
- Oracle: references that lost a tiny p or t at dps 50; refuses to
  rewrite the doc from a partial CSV.

Not defects, decided [DERIVED]: discrete quantiles near p = 1 that differ
from the oracle by 3 to 5e4 counts (the contract is the smallest k whose
library CDF reaches p; #116, #170); logpdf near its zero crossing (a
relative metric on a near-zero value).

Last release: v2.4.1 (2026-09-19). v2.5.0 (corvus adoption) is in
progress on `dev/v2.5.0-corvus` in its own session; that branch's PLAN.md
is authoritative for v2.5.0 state.

## v2.5.0 release runbook (branch `dev/v2.4.2`) [OPEN]
Machines: **Z** = Zen 4 (Windows 11, MSVC, AVX-512), **K** = Kaby Lake
(macOS Ventura, AppleClang, AVX2), **M** = Mac Mini M1 (macOS 27,
AppleClang, NEON).

Status by machine (each machine edits only its own line):
- **Z:** R0 done (`tools/bench/`, `8cf5550`). At the freeze head
  (2026-10-04): `5367999` did not build on Windows (fixed, `6cab6ed`).
  R1 done with MSVC (VS 2026, fresh configure, 0 warnings, 89/89,
  Release CRT) and clang-cl 22.1.3 (Ninja, 0 warnings after `abd0e86`,
  89/89), AVX-512 on both. R2 done: AVX-512 block at `c09efd0`, 0
  contract violations; Beta pdf/logpdf and the Gamma family 2–10⁴×
  better; Binomial pdf (1.7e-14 → 4.9e-14) and Poisson pdf (5.5e-14 →
  1.2e-13) worse, digit for digit as on K's AVX2 block at `4ceafae`, so
  the code's, within the contract. R5 skipped [user]. The MSVC R3/R8
  overnight of 2026-10-04/05 (`2026-10-04-zen4-v242-cost/`, bundles
  `2026-10-05T0*_windows-*_sha-2c5230a`) ran under power throttling
  (base clock; Rules) and is superseded for R8 (manifests say so); its
  R3 ratios hold. At `998d985`: R1 MSVC and clang-cl 89/89, the sweep
  bit-identical to `6cab6ed` under both, so the AVX-512 block stands. At
  `606607f` (clang-cl `/arch:`): clang-cl 0 warnings, 89/89, its sweep
  0 contract violations through the oracle (to scratch; the block stays
  MSVC's). **R3 and R8 done on clang-cl** (overnight 2026-10-05/06, every
  process opted out of throttling, clock at boost throughout;
  `docs/bench-evidence/2026-10-05-zen4-clangcl-v242-cost/`; v2.4.1 built
  with `v241_clangcl_compat.patch` there, applied uncommitted in
  `../libstats-v2.4.1`). R3: timing 21/22, the one failure a single-shot
  timing that 40 reruns pass (`test_gaussian_enhanced`, #177); von
  Mises quantile 17–23× and Student-t quantile at ν ≥ 1e3 11.6–13.3×
  against v2.4.1, the fastest absolute times of the three machines (the
  ratio is v2.4.1's 18 ns grid lookup); Beta at large shapes 3–4×. R8:
  bundles `2026-10-06T0*_windows-*_sha-69c7be4`, run-to-run spread 9–13
  rows > 2×; the cross-tier comparison and the hypothesis tests are in
  the AVX-512 bundle and under "Trends" below. Those R8 bundles predate
  `73ecbfe`/`ee11e62` and are superseded for R8 (R8, "Next on every
  machine"). At `ee11e62` (2026-10-06): MSVC and clang-cl 0 warnings,
  90/90, sweep bit-identical to `73ecbfe` and `998d985`; speed gate
  PARALLEL/VECTORIZED 0.17–0.22 at 1M; every concurrency gate failed on
  `73ecbfe` (the Poisson one by deadlock) and passes. R8 steps 3–4 run
  overnight 2026-10-06/07 on the clang-cl trees rebuilt at `ee11e62`
  (0 warnings, tiers confirmed): the speed gate alone, the 23 timing
  tests, then three `--large` runs per tier (AVX-512, AVX2, AVX, SSE2),
  every process opted out of throttling. Done, clock at boost throughout
  (127.2–127.8 ms), speed gate passed, timing 23/23; superseded before
  use by #184–#191 (#191 rewired both profiled parallel strategies) and
  kept as Z's pre-#191 R8 reference: bundles
  `2026-10-07T0*_windows-*_sha-8271f6e`, evidence
  `docs/bench-evidence/2026-10-06-zen4-clangcl-ee11e62-r8/`; the
  `69c7be4` bundles marked superseded too. At `3339fc1` (2026-10-07):
  #190 reads 6 physical, 12 logical (correct; Z is 6C/12T); clang-cl
  and MSVC 0 warnings once the stress test got `/bigobj` under MSVC
  (C1128) and both `std::getenv` calls a `_dupenv_s` reader (clang-cl
  lacks `_CRT_SECURE_NO_WARNINGS`; tests and tools only); sweep
  bit-identical to `ee11e62`'s under both; correctness 90/91, the stress
  test failing only its strategy differential (#191's slice tail, R9;
  K fixes it [user]). R8 and the stress rerun wait for K's head.
- **K:** R1 done (AppleClang 15, macOS 13.7.8, CMake 4.4.3; `a0a913e`
  fixed the SDK `label` collision and sign-compare warnings in tests;
  86/86). R2 done: AVX2 block at `a0a913e` (`4f40228`), 0 contract
  violations. R4 done: #162 reproduced on head and fixed (`8018c5d`);
  #167 fail-first under UBSan (`d8d3388` aborts at `poisson.cpp:1240`,
  head clean). Run binaries outside the Claude Code sandbox, which blocks
  the cache-size sysctls. R3 done (2026-10-03,
  `docs/bench-evidence/2026-10-03-kabylake-v242-cost/`): timing 22/22;
  von Mises quantile 52–136× slower; Gamma α ≥ 20 logpdf 10–15× and
  Student-t 4–9× (scalar fall-backs); discrete quantiles 0.2–0.6×. 20 NEW
  AUTO-vs-best gaps, all favouring PARALLEL at n = 1e3–1e5 (Z: 22); 13 in
  both versions. R8 captures done (2026-10-04, `d9384f8`): AVX2, AVX
  and SSE2 bundles `data/profiles/dispatcher/2026-10-04T*`, three quiet
  runs each; findings under R8. Superseded by the 2026-10-04 speed work
  (`b5a47df`–`5367999`). At the freeze head (2026-10-04): R1 done
  (clean `build-release`, 0 warnings, 89/89); R2 done, AVX2 block at
  `4ceafae` (`dce2656`), 0 contract violations, no row worse. R3 done
  (overnight 2026-10-04/05, `docs/bench-evidence/2026-10-04-kabylake-v242-cost/`):
  timing 22/22; per call or forced VECTORIZED, von Mises quantile
  5.4–6.8× (was 52–136×), Student-t quantile at ν ≥ 1e3 6.8–8.0×, Gamma
  α ≥ 20 logpdf 3.5–4.9× (was 10–15×), Beta(25, 25) and (1000, 1000)
  logpdf 4.1–4.2×, Student-t pdf 4.2× (as before); discrete quantiles
  0.2–0.6×. 14 NEW AUTO-vs-best gaps, all but one favouring PARALLEL at
  n = 1e3–1e5; 10 in both versions. R8 done: bundles
  `2026-10-05T0{1,3,4}-*_darwin-*_sha-6f86996` (AVX2, AVX, SSE2; code
  `5367999`), run-to-run spread 11–17 rows > 2×. Against `d9384f8`, the
  speed work moved VonMises CDF to NEVER on every tier and Weibull and
  Pareto CDF later; the comparison is in the AVX2 bundle. Against the tables:
  kAvx2 30 rows, kAvx 10 (AVX) and 27 (SSE2). No cross-machine analysis
  or table change until Z's clang-cl R3/R8 rerun is in [user,
  2026-10-05]. R4 stands. K's R1–R4 and R8 are complete at the freeze head.
  At `bb27040` (2026-10-05): incremental build 0 warnings, 89/89; the
  sweep bit-identical to `4ceafae`'s (10,212 lines), so `998d985`
  changes nothing on AppleClang 15. At `045acd9` (2026-10-07, after
  pulling `e9a7410`): Release 0 warnings; correctness 91/91; sweep
  bit-identical to `bb27040`'s, so `ee11e62`, #184–#191 and the slice
  fix change no result bits on AppleClang 15; stress test after K's fix
  in R9. #184–#191 moved to milestone #10 [user]. Next on K: quiet
  `test_parallel_batch_gates`, R8 (AVX2, AVX, SSE2; the capped trees
  rebuilt), then R9's stress test under TSan.
  **2026-10-07/08 review and fix round** [DERIVED, K; committed 2026-10-08
  as `6a56806` (#201, #137), `a8c125d` (#192–#209, #173), `b0ac2ad` (#183,
  #227, #210), `6920ece` (oracle tie rule); AVX2 block at `6920ece`]: an architecture review and
  a defect hunt (reports in the session scratchpad; skill
  `numerical-defect-hunt`, dotfiles `6c03c2d`); fixes #192–#209, #173,
  #137, #183 and #227, each with a guard shown failing first (#183's is
  TSan: 18 setter races → 0); CI tests `test_distribution_identities`
  and `test_io_fit_fuzz` (#210); `detail::parse_double` so
  full-precision output (#208) reads back subnormals; the oracle scores
  discrete ties by the library-CDF rule. Decided [user, 2026-10-08]:
  keep F15's tie rule (#204); accept F6 (`getSurvival` virtual), F14
  (Rayleigh/HalfNormal σ ≤ 1e10, as Gaussian), F19 (17 significant
  digits); fix #183 and #227 now. Next [user, 2026-10-08]: fix #223–#226 and
  #228 (Beta sampler returns 0.5 at small shapes).
  Verified on the final tree: Release 0 warnings, correctness 99/99,
  stress 124 passed + 12 skipped (#180), TSan stress 0 races, sweep
  bit-identical to the merged tree (Beta, TN, Geometric CDF and
  Discrete-tie rows changed against `045acd9`), oracle 0 contract
  violations. Overnight 2026-10-07/08 (code hash `3a00b1e8208bfe3b`,
  before #183/#227, which touch setters only; sweep unchanged by them):
  `test_parallel_batch_gates` 0.27–0.33 (all < 0.75); R8 AVX2, AVX,
  SSE2 three runs each, #191 moved 25–35 rows per tier to PARALLEL
  sooner (many NEVER → ~4096, the fork threshold), 2–8 later; tables
  now differ on 50/36/47 rows; R9 TSan: only #183 and the test-side
  `std::cout` race in `test_work_stealing_pool`. Bundles: `data(profiles)` at
  `b0ac2ad`.
- **M:** macOS 27.0.1, AppleClang 21.0.0 (clang-2100.3.34.2), CMake
  4.4.3, Ninja 1.13.2. At `2c5230a` (freeze head plus tests/docs,
  2026-10-04): R1 done (clean `build-release`, 0 warnings, NEON, 89/89).
  R2 done: NEON block at `2c5230a`, 0 contract violations (v2.4.1: 32);
  no row worse than the AVX-512 block. Fifteen rows are worse than the
  v2.4.1 NEON block (Cauchy cdf 2.9e-16 → 2.1e-12; Gaussian, LogNormal
  pdf/cdf 7–20×; Pareto quantile, Rayleigh pdf), each by the same
  factor in both x86 blocks, so the branch's, within the contract;
  oracle run with the `libstats-log-cleanroom` venv (mpmath 1.3.0). R4
  skipped (done on K). R8 and R3 ran overnight 2026-10-04/05 at
  `2c5230a` from a detached script, gated on ambient CPU < 15% over two
  10 s windows (relaxing to 20% after 3 h), not on load: the M1's 1-min
  load sits at 4–10 idle. The gate opened in lulls; `mediaanalysisd`
  and Backblaze returned during the R8 runs (noise logged per minute).
  R8 done: NEON bundle `2026-10-05T04-36-25Z_*_sha-2c5230a`, run-to-run
  spread 9 rows > 2× (K: 12–19). 22 of 81 rows differ from kNeon, 20 of
  them as in 2026-09-04 (deliberate overrides); the v2.4.2 movers are
  FisherF PDF 8192 → 2048 and LogPDF 10k → 4096, as on K's kAvx2; 8
  more rows moved since September onto values kNeon already holds. R3
  done (`docs/bench-evidence/2026-10-05-m1-v242-cost/`): timing 22/22
  (ambient 6.1%); von Mises quantile 5.8–8.7×, Student-t quantile at ν ≥
  1e3 7–9×, Gamma(20) logpdf 4.3× forced VECTORIZED; discrete quantiles
  0.23–0.71×. 14 NEW AUTO-vs-best gaps, all favouring PARALLEL at
  n = 1e4–1e5 (Beta 25 and 1000 shapes, Gamma α ≥ 19.9, Binomial and
  NegativeBinomial CDF at large counts); 9 in both versions. At
  `1f999e3` (2026-10-05): incremental build 0 warnings, 89/89; the
  sweep bit-identical to `2c5230a`'s (10,210 rows), so `998d985`
  changes nothing on AppleClang 21 / NEON either. M's R1–R3 and R8 are
  complete at the freeze code. At `ee11e62` (2026-10-06, a Claude session
  whose sandbox bypass was refused, so every binary ran inside the
  sandbox): Release 0 warnings, correctness 90/90. Next-steps 2–4 (sweep
  compare, `test_parallel_batch_gates`, R8) not run: they need the
  bypass (cache sizes read 0 inside) and a quiet machine (Spotlight and
  `backupd` at ~180% CPU each). R9 steps 1–4 done on M:
  `tests/test_concurrency_stress.cpp` and
  `docs/bench-evidence/2026-10-06-m1-concurrency/`, both uncommitted;
  results under R9.

### Rules on every machine
- Commit or push only when the user asks. Commits are signed (YubiKey);
  never disable signing; batch a commit and its push.
- Edit only your own status line above; `git pull --rebase` first.
- Validate at the freeze head `b0ac2ad` (Status: it moves when a
  defect is fixed; validate at the latest); confirm it with `git log -1`
  before R1. Library code changes only by the user's decision: see Status.
- New defects or unexpected oracle rows: stop and report with evidence.
  Fix nothing in library code without the user's decision.
- Quiet runs (R3) alone on the machine: no builds, sweeps or other
  sessions. On K, wait for the load average to fall below 2.0.
- On Z, a locked screen puts running processes under Windows power
  throttling (EcoQoS): base clock, 1.5× slower, and it stays with that
  process after unlock [DERIVED, 2026-10-05]. Opt each measured process
  out (`SetProcessInformation`, ProcessPowerThrottling, EXECUTION_SPEED
  in ControlMask, StateMask 0); children do not inherit it, so launch
  every binary yourself, not through ctest. Helper:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/nothrottle.ps1`.
- Performance numbers from Release builds only.
- On the Macs, run every library binary (tests, tools, sweeps, benches,
  profiles) outside the Claude Code sandbox: it blocks the
  `hw.*cachesize` sysctls, so cache sizes read 0 and the cache-derived
  tuning changes. The sandbox also blocks the SSH agent for `git fetch`.

### Cold start (K, M)
1. `git fetch`; `dev/v2.4.2` must be at the freeze head `ee11e62` or
   later. Work in a
   worktree beside the main checkout: `git worktree add
   ../libstats-v2.4.2 dev/v2.4.2`, or `git pull` in it if it exists.
2. Read `AGENTS.md` and this file. Record the OS, AppleClang and CMake
   versions for R6.

### R1 — native correctness (K, M)
1. `rm -rf build-release`, then `cmake --preset release -G Ninja`, then
   `cmake --build build-release`. Expect a warning-clean build. This is
   the first clang build of the v2.4.2 code: watch `[[maybe_unused]]`,
   the double-double helpers under FP contraction, and
   `lgamma1p_small`'s table.
2. `./build-release/tools/system_inspector --quick`: AVX2+FMA on K,
   NEON on M.
3. `ctest --test-dir build-release -LE "timing|benchmark" -j8`: expect
   89/89. The target `run_tests_correctness` runs a 79-test subset.

### R2 — accuracy sweep and block (K, M)
1. The tree must be clean, at a pushed commit, and built in R1. The sweep
   stamps `git rev-parse` at run time, so the banner then names the code
   it measured.
2. `cmake --build build-release --target accuracy_sweep`, then
   `./build-release/tools/accuracy_sweep sweep.csv`. Write the CSV
   outside the repo.
3. `python3 tools/accuracy_vs_mpmath.py sweep.csv`, using the Python that
   ran the 2026-09-28 regenerations. If `import mpmath` fails, ask the
   user before installing anything. Only a full sweep rewrites the doc,
   and only its own `isa=` block (AVX2 on K, NEON on M); check with
   `git diff` that nothing outside that block changed.
4. Expected: 0 contract violations (v2.4.1: 32 on each). Large rows that
   are known or decided:
   - the Beta quantile, max_rel ~1e259 at p = 0.001 (#137, v2.5.0);
   - logpdf rows with max_rel ≫ 1 but max_abs ≲ 1e-14 (Laplace,
     Exponential, Logistic, Gumbel, Beta near a zero crossing);
   - discrete quantiles near p = 1 (Geometric, NegativeBinomial,
     Poisson, Binomial);
   - quantile rows with a huge max_abs and a small max_rel.

   Anything else — a contract violation, or a row far worse than the
   same row in the AVX-512 block — is a stop-and-report.
5. Commit (user's approval) as `docs(accuracy): regenerate <ISA> block
   at <sha> — 32 → N`, then push.

### R4 — Mac-only gates (done on K, 2026-10-03; M skips R4)
- **#162** (`gh issue view 162`): the POST_BUILD ad-hoc `codesign`
  fails "is already signed" when a reconfigure re-runs the dylib's
  symlink step without relinking. Reproduce first on head, then fix
  `CMakeLists.txt`. Accept: the reproduction builds clean, and R1 passes
  again. Write and verify it in one Mac pass [user].
- **#167 fail-first under UBSan.** x86 MSVC cannot show it. The gate is
  `SupportBoundary.CountsBeyondIntRange` in
  `tests/test_support_boundary_gates.cpp`. That file does not exist at
  `d8d3388`, so:
  1. Make a detached worktree at `d8d3388`.
  2. Copy the file in from head, and append
     `create_libstats_gtest(test_support_boundary_gates
     test_support_boundary_gates.cpp)` to its `tests/CMakeLists.txt`.
  3. In both trees, configure a fresh `build-ubsan` with `-G Ninja
     -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLIBSTATS_BUILD_TOOLS=OFF`. Pass
     `-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all`
     in `CMAKE_CXX_FLAGS`, `CMAKE_EXE_LINKER_FLAGS` and
     `CMAKE_SHARED_LINKER_FLAGS`.
  4. Build that target in each, and run it with
     `--gtest_filter=SupportBoundary.CountsBeyondIntRange`.
  5. Pass if the `d8d3388` run aborts with a float-cast-overflow runtime
     error and head runs clean. Record both outputs for R6.

  The other tests in that file gate other issues and fail at `d8d3388`
  by design.

### R3 — quiet-machine costs and timing (Z, K, M; last on each)
1. A `v2.4.1` build to compare against.
   - K, M: `git worktree add ../libstats-v2.4.1 v2.4.1` (detached), then
     in it `cmake --preset release -G Ninja -DLIBSTATS_BUILD_TESTS=OFF
     -DLIBSTATS_BUILD_TOOLS=OFF` and `cmake --build build-release
     --target libstats_static`.
   - Z: `../libstats-v2.4.1/build-v242cmp` exists.
2. Build `tools/bench/v242_cost_bench.cpp` against each tree. The
   commands are in its header: clang++ on K and M, cl from vcvars64 on Z.
3. Quiet: `./bench_v241 > v241.csv`, then `./bench_v242 > v242.csv`, back
   to back (about 2 min each on Z). On Z, set
   `LIBSTATS_BENCH_WARMUP_SECONDS=20` first. Then
   `python3 tools/bench/v242_compare.py v241.csv v242.csv > compare.txt`.
4. Quiet: `ctest --test-dir build-release -j1 -L timing` (Z:
   `ctest --test-dir build -C Release -j1 -L timing`). On K and M this is
   the first native run of the steady-state speedup gates from #169 (in
   `d8d3388`), unverified there so far.
5. Keep the two CSVs and `compare.txt` for R6 under
   `docs/bench-evidence/<date>-<machine>-v242-cost/`; commit with the
   user's approval.
6. Report section 3 of `compare.txt` (AUTO-vs-best gaps marked NEW). The
   gaps feed R8; do not re-derive thresholds here. INDICATIVE
   smoke run on a busy Z: von Mises quantile 140–320× slower (3–6 µs
   Newton on the quadrature CDF, was a grid); forced VECTORIZED 8–13×
   slower for Gamma α ≥ 20 and Student-t ν ≥ 1e3 (scalar fall-backs); 22
   NEW gaps, all favouring PARALLEL at n = 1e4–1e5.

### R5 — optional capped-tier leg (Z)
v2.4.0 precedent: a `LIBSTATS_MAX_SIMD_TIER` build, correctness and
sweep. R8 builds the same capped trees on Z; run R5's correctness and
sweep on them before their profiles if wanted.

### R8 — dispatch-threshold captures (Z, M) [OPEN]
Decided [user, 2026-10-03/04]: capture `strategy_profile` for every tier
natively, compare across machines, and only then update
`dispatch_thresholds.h`. No table edits yet. Z: AVX-512, AVX2, AVX,
SSE2. M: NEON. K captured at `d9384f8`. Timing: the speed work landed
2026-10-04 and moved many rows beyond von Mises and Gamma (Student-t
and Beta densities now SIMD at every shape; the Exponential, Rayleigh,
Weibull and Pareto CDFs on `vector_expm1`; the discrete log-pmfs and
every incomplete gamma/beta through the new `log1pmx`; the von Mises
CDF now compute-bound, scalar per element). Capture once, at the final
head, on every machine; K's `d9384f8` bundles are the before-picture.

Why [DERIVED, K bundles `2026-10-04T*`]: the v2.4.2 accuracy fixes
shrank the VECTORIZED advantage over SCALAR, so PARALLEL now wins sooner
on these rows. On AVX2 the V/S time ratio went from 0.07–0.12 to
0.24–0.43 for Student-t pdf/logpdf, Cauchy (which delegates to
StudentT(1)) and the Exponential/Pareto/Weibull/Rayleigh CDFs, and from
0.25 to 0.79 for the von Mises CDF. 12 of 81 kAvx2 crossovers moved since
2026-09-04 (also FisherF ×3 and Poisson logpdf), and 7 of 81 kAvx. A
mechanical re-derive would change 35 kAvx2 rows, but 23 of those measure
the same as in September: they are deliberate overrides with per-row
comments, so keep them unless the comments show them stale. SSE2 differs
from AVX on 18 rows, nearly all crossing sooner, which argues against
`sse2_parallel_threshold()` delegating to kAvx. Noise to handle when
deriving: 12–19 rows per tier spread more than 2× across three runs, and
crossovers at n = 64 are thin-margin resolution artifacts (the bundle
README's clamping rule).

Procedure (each tier):
1. Build Release with tests off; for a capped tier add
   `-DLIBSTATS_MAX_SIMD_TIER=<AVX2|AVX|SSE2>`. Z: one build directory per
   tier beside `build/`, `stats.dll` copied beside the tools. Confirm the
   tier from the configure line ("capped at <tier> ... disabled: ...")
   and `system_inspector --quick`, and record both.
2. Three quiet `strategy_profile --large -o strategy_profile_run<i>.csv`
   runs. Start each only once the machine is quiet (K: 1-min load < 2.0)
   and log the start load. A run takes ~27 min on K. Z: `strategy_profile`
   has no warm-up option, so precede each run with ~20 s of load (a
   discarded short pass) to get past the boost drop (VALIDATION_HISTORY,
   "Zen 4 frequency scaling"), and record that.
3. Bundle as K's `2026-10-04T*` bundles do:
   `data/profiles/dispatcher/<UTC run-1 start>_<platform>_dev-v2.4.2_sha-<sha>/`
   with `metadata.json`, `manifest.txt` (capture notes), the three CSVs,
   `analyze_crossovers.py` (copy from a K bundle; paths are relative),
   `sustained_crossovers.txt` and `logs/*.txt` (`.gitignore` drops
   `*.log`).
4. Compare with `cross_tier_assessment.py` from K's AVX2 bundle: copy
   it, point `NEWD` at your bundles, and for AVX-512 compare against
   kAvx512. Report rows moved against the table and against K's same
   tier, and stop there; commit the bundles with the user's approval.

After Z and M [user's decision]: update only the rows v2.4.2 moved in
each table, decide on a kSse2 table of its own, then R1 again on every
machine whose table changed. Inputs: "Trends from the v2.5.0 captures".

Captures complete on K, M and Z at the freeze code (2026-10-06). Decided
[user, 2026-10-06]: change only the rows that moved for a code reason,
not a full re-derivation. But the rows under the two parallel-path
defects found in Z's analysis (Trends; #175, #176) measure
the defects, not the operations: Bernoulli, Binomial, Geometric,
NegativeBinomial and the von Mises CDF (per-element lock), the Student-t
CDF (serial PARALLEL lambda). Their rows are unknown until the fix and a
re-profile, so the table update waits for both. Fixed at `73ecbfe`
(2026-10-06; Z: both gates failed on the previous head and pass,
PARALLEL/VECTORIZED 0.17–0.22 at 1M for all eight operations under
MSVC and clang-cl; sweep bit-identical). The Macs' 1.5–8.9× slower
PARALLEL was measured on the unfixed code; the fix is unrun there.

The audit that followed (2026-10-06, `ee11e62`) also changed what R8
measures: below ParallelUtils' fork threshold, PARALLEL now takes the
batch path instead of a serial per-element scalar loop, so no capture
before `ee11e62` measured PARALLEL under 1024–8192 elements as AUTO runs
it now.

Next on every machine, at the freeze head (Status):
1. Build (R1's commands); correctness ctest, 90 tests (91 with
   `test_parallel_batch_gates`, which is timing-labelled).
   `test_snapshot_consistency` and `test_concurrency_gates` carry the
   new correctness gates; a hang in the latter is the Poisson deadlock
   (its watchdog ends the process after 60 s).
2. The accuracy sweep, compared with your last sweep's result bits
   (`diff` past the two banner lines): expect identical. Any difference
   is a stop-and-report.
3. Quiet: `test_parallel_batch_gates` alone; expect every
   PARALLEL/VECTORIZED ratio below 0.75 (it prints them). Record them.
4. R8 again, every tier (K: AVX2, AVX, SSE2; M: NEON; Z: clang-cl
   AVX-512, AVX2, AVX, SSE2), as before; the new bundles supersede all
   previous v2.5.0 captures.
5. R9 (stress test and ThreadSanitizer): written on M; then run on
   every machine, after that machine's quiet runs.
Then decide the rows.

### R9 — mechanical concurrency checks (M writes; M, K, Z run) [OPEN]
Goal [user, 2026-10-06]: catch mechanically, before the v3 refactor
redesigns locking, the defect classes that so far only code review has
found, so later fixes stop triggering unplanned bugfix rounds. Review
finds instances of the classes it looks for; it cannot show absence.
Three 2026-10-06 rounds still found TOCTOUs and lock errors (Known
Gaps); CI has no TSan leg. R9 succeeds when every class below has a
detector shown to fire, or is recorded as a v3 input with no detector.

**Classes and detectors** [DERIVED from `ee11e62`'s message and
#178–#183; the M/K/Z split: "Per-machine"]:

| Class | Instances so far | Detector | Where it can fire |
|---|---|---|---|
| C1 unlocked shared read/write | Uniform copy-assign; pool construction; #183 | TSan | M, K |
| C2 check-then-act across a lock release | Uniform/Discrete bound setters; 5 delegate setters | stress invariant oracle, aligned rounds | M, K, Z |
| C3 recursive or double lock | Poisson `getMedian`; `d == d` | stress watchdog (deadlock) | M (deadlocked at `6987317`), Z; K untested; `d == d` none |
| C4 one call mixing two parameter states | Binomial, NegBinomial, von Mises batches; Beta boundary; #182 | stress one-state oracle | M, K, Z |
| C5 strategy runs the wrong kernel | PARALLEL below the fork threshold | strategy differential (step 1c) | M, K, Z |
| C6 parallel helper semantics | `parallelTransform` repeated chunks; #181 | call-count gates; #181 needs fault injection | gates all; #181 none |
| C7 work-stealing cross-waiting | #180 | `WorkStealingCrossWaiting`: a WORK_STEALING batch beside a blocked pool task | K only after #190; fires on K (12 rows, 28 ops; known #180) |

No detector, so v3 inputs unless one is found: `d == d` (undefined, no
platform shows it), recursive shared locks off Windows, #181's
submit-failure path.

**Per-machine** [user, 2026-10-06: decided, with the strategy
differential (step 1c) kept]. One test source, not three: the defect
classes belong to the code, and machines differ only in which classes
can show. The test prints what it reached
(strategy per batch reader, core counts, lock backend) so coverage is
read off the log, not assumed:
- **M** (M1, 4P+4E, no SMT): the only weakly ordered CPU, so missing
  acquire/release on atomics can show only here; TSan. Until #190's fix
  `SystemCapabilities` read physical = logical / 2, so M took
  WORK_STEALING; with it, MAXIMIZE_THROUGHPUT and AUTO resolve to
  PARALLEL on M, and no public API reaches WORK_STEALING there. The
  recursive shared lock deadlocked here too (C3).
- **K** (Kaby Lake 4C/8T, SMT): the only WORK_STEALING machine after
  #190, so K verifies every WORK_STEALING lambda (#191's rewiring
  included) and runs TSan on them. Runs after K's R8 quiet runs.
- **Z** (Zen 4 6C/12T, Windows): SRWLOCK; Windows resolves
  MAXIMIZE_THROUGHPUT to PARALLEL, so no WORK_STEALING. No TSan (MSVC and
  clang-cl do not support it on Windows). Run the stress test under both
  MSVC and clang-cl. Runs after Z's R8 overnight.
- No machine has Linux; a Linux TSan CI leg is the only check that would
  stay in place (user's decision, CI-HOUSE-STYLE runner budget).

Run every Mac binary outside the Claude Code sandbox (Rules).

1. **Write `tests/test_concurrency_stress.cpp`** (on M), a correctness
   test (no `timing` label, so CI runs it). Default run under ~60 s;
   `LIBSTATS_STRESS_SCALE` (rounds multiplier, default 1) lets the TSan
   runs shrink it and machine runs grow it. Self-contained: copy
   `runWithWatchdog` and `racedRounds` from
   `tests/test_concurrency_gates.cpp`, which does not exist before
   `ee11e62`. Table-driven, one row per distribution (all 27):
   a. **States.** Two valid parameter states A and B, distinct at every
      probe. A writer flips between them through `setParameters` (one
      lock, so A and B are the only states); a second variant flips
      each single-parameter setter, ordered so every intermediate is
      valid, and adds the intermediates to the allowed set.
   b. **Readers.** Scalar pdf, logpdf, CDF at fixed probes;
      `getQuantile`; `getMean`, `getVariance`, `getSkewness`,
      `getKurtosis`, `getMedian`, `getMode`, `getEntropy` (all 27 have
      them); batch pdf/logpdf/CDF under `FORCE_SCALAR`,
      `FORCE_VECTORIZED`, `FORCE_PARALLEL`, `MAXIMIZE_THROUGHPUT` and
      AUTO, at a size above `arch::get_min_elements_for_parallel()`;
      `sample` with a fresh fixed-seed generator per call; copy
      construction, copy assignment, move; `operator==` against fixed
      A and B. Separately, `fit` and stream `operator>>` as writers
      racing the readers: each produces a state C, computed from a fixed
      instance fitted to the same data or reading the same stream, and C
      joins the allowed set.
   c. **Strategy differential** (C5, no race): for each distribution and
      batch operation, every strategy against `FORCE_VECTORIZED` at
      sizes straddling the fork threshold and the SIMD minimum. Open:
      whether sliced PARALLEL is bit-identical to VECTORIZED above the
      threshold (slice edges move SIMD tails); establish it on M first,
      then gate bitwise or record the tolerance.
   d. **Oracle.** Every racing result equals an allowed state's value
      bit for bit, computed from fixed instances by the same call; every
      batch comes entirely from one state.
   e. **Driving.** Both free-running loops and aligned rounds
      (`racedRounds`): loops missed every one-shot race the aligned
      rounds caught (`ee11e62`'s message). Every reader under a
      watchdog, so a deadlock fails instead of hanging.
   f. **Reach.** Print which strategy each batch reader resolved to,
      `logical_cores`/`physical_cores`, and the scale.
2. **Show it fires before trusting it.** Build it in a detached
   worktree at `6987317` (before #175, #176 and the audit fixes),
   registered in that tree's `tests/CMakeLists.txt`. It must fail on the
   Binomial, NegativeBinomial and von Mises batches, the Beta boundary
   batch, the Uniform and Discrete bound setters, and the Bernoulli,
   Geometric, ChiSquared, Erlang and InverseGamma delegates on every
   machine; on Poisson's `getMedian` (deadlock) on Z. `getMedian` not
   deadlocking on M or K is the lock backend, not a harness gap; record
   it either way. Any other miss means the harness needs work. Record
   per machine which it catches; `test_concurrency_gates.cpp`'s header
   says what each defect looked like.
3. **At `ee11e62`, expect it to fail on #182 (FisherF `sample`) and
   nothing else.** #182 is a defect open at the freeze head, so catching
   it there shows the harness fires on current code. Mark #182's row an
   expected failure (GTest skip naming the issue) only after it is seen
   to fail. Any other failure is a new defect: stop and report with the
   evidence (it moves the freeze head).
4. **ThreadSanitizer (M, then K).** A fresh `build-tsan` (Ninja,
   `RelWithDebInfo`, `-DLIBSTATS_BUILD_TOOLS=OFF`, `-fsanitize=thread`
   in `CMAKE_CXX_FLAGS`, `CMAKE_EXE_LINKER_FLAGS` and
   `CMAKE_SHARED_LINKER_FLAGS`, as R4's UBSan build; a fetched corvus
   and Highway take the same flags). Run with a reduced
   `LIBSTATS_STRESS_SCALE`: `test_concurrency_stress`,
   `test_concurrency_gates`, `test_snapshot_consistency`,
   `test_thread_pool`, `test_work_stealing_pool`,
   `test_parallel_execution_integration`,
   `test_parallel_exception_propagation`, then the whole correctness
   ctest (`-LE "timing|benchmark"`, `-j2`). Expect #183's unlocked reads
   in `validateParameters` (the TSan positive control at head). Every
   other report is a finding (file:line, both stacks); triage benign
   ones explicitly, never suppress silently. Fail-first for the two
   `ee11e62` fixes with no deterministic gate, at `73ecbfe`: the stress
   test (copy assignment under a writer) should report Uniform's
   copy-assignment, and `test_work_stealing_pool` the pool's
   construction.
5. **Report** to the user, per machine: what the stress test caught at
   `6987317`, its result at `ee11e62`, the TSan findings, and the class
   table updated with "shown to fire" evidence or "no detector". Commit
   the test and the evidence (`docs/bench-evidence/<date>-<machine>-concurrency/`)
   with the user's approval.

**R9 on M** [DERIVED, 2026-10-06; evidence
`docs/bench-evidence/2026-10-06-m1-concurrency/summary.txt`; test and
evidence uncommitted, awaiting the user]. Run inside the sandbox (the
bypass was refused); correctness only, which the cache sizes do not
affect.
- Step 1c settled: every strategy returns exactly the batch kernel's or
  the scalar method's bits, never a mix; above the fork threshold
  PARALLEL runs the scalar formula in 33 of 81 (distribution, op) pairs
  (F8), up to 88 ULP from VECTORIZED. The gate is "one known kernel", and
  PARALLEL below the threshold must be the batch kernel.
- Step 2 at `6987317`: caught every listed defect but InverseGamma's
  delegate (unseen on Z too); Poisson's recursive lock deadlocked on
  macOS; Bernoulli and Geometric delegates only at 20× rounds (1–52 of
  60,000), so the same-setter pairs run 20×.
- Step 3 at head: #182 fired (1 hit); new F1–F4 below; nothing else.
  Run time 65 s (budget ~60 s).
- Step 4 TSan: at head F1 and #183 (positive control) only, beyond a
  test-side `std::cout` race in `test_work_stealing_pool`; the whole
  correctness ctest otherwise clean. At `73ecbfe`: Uniform's
  copy-assignment and the pool's construction both reported.
- Decided [user, 2026-10-07]: F1–F8 filed (#184–#191); the stress test
  lands with known-issue skips. A result explained by an open issue
  prints `[ known ] #N` and does not fail; a test whose only failures are
  known is skipped naming them; each issue's fix removes its entry
  (`knownReaderIssue`, `knownProgramIssue`, `knownPairIssue`). At head:
  exit 0, 25 tests skipped (#182, #184–#187), 68 s; at `6987317` the
  same file still fails every listed defect.
- Decided [user, 2026-10-07]: fix #184–#191 before the freeze (it moves
  the freeze head; every machine then repeats R1, R2's sweep compare and
  R8 at the new head); the 68 s run time is accepted; M's pending runs
  go overnight, outside the sandbox, after the fixes. Q1 decided
  [user, 2026-10-07]: B, document per-element SCALAR semantics now;
  one-state batches under every strategy is a v3 design input.
- Fixes, steps 1–3 [DERIVED, M, 2026-10-07; uncommitted, awaiting the
  user]: #189 (Poisson `validateCurrentParameters` exported), #187
  (stream tag length), #188 (no thread_local spare; new per-row
  `SeededSample` test), #184 (move-assignment locks both objects in
  `std::lock` order, as copy-assignment does [user, 2026-10-07: parity];
  `noexcept` kept; base-class contract doc updated), #185 (TruncatedNormal `fit` commits
  only if the bounds are unchanged, else recomputes), #186 (vector
  `sample` draws from a delegate copy taken under the owner's lock).
  Each shown failing first by removing its skip. After: stress 107
  passed, 1 skipped (#182); correctness 91/91; timing 23/23 (loaded
  machine); TSan: only #183's setter reads. Commits staged as: format
  drift (4 files), then #189, #187, #188, #184, #185, #186, then docs
  (AGENTS.md's SCALAR note for Q1, this file); each intermediate state
  built and tested; pushed 2026-10-07 (`5765d68`..`7d9cb0b`).
- #190 and #191 [DERIVED, M, 2026-10-07; uncommitted, awaiting the
  user's YubiKey]. #190: `SystemCapabilities` reads the detector's
  physical count (sysctl on macOS), and the x86 CPUID leaf-0xB path
  divides the core-level logical count by threads per core (unverified
  natively: Z's `system_inspector --quick` must read 6 physical, 12
  logical; Z is 6C/12T, and read 6 and 12 at `3339fc1` under both
  compilers, 2026-10-07). Guard `ConcurrencyStressMachine.CoreTopology` (in the stress
  test; `test_system_capabilities` is timing-labelled, so CI never ran
  it), two-sided against sysctl; failed first (physical 4 vs 8). M now
  resolves MAXIMIZE_THROUGHPUT to PARALLEL. #191: every PARALLEL and
  WORK_STEALING lambda runs its VECTORIZED kernel over 1024-element
  slices (19 files, by a subagent, reviewed; Gamma and Beta CDF and
  Discrete were also scalar; Cauchy CDF's inline loop moved to a helper;
  new code drops the redundant `waitForAll`, existing ones stay for
  #180). The differential now requires PARALLEL and MAXIMIZE_THROUGHPUT
  to equal VECTORIZED's bits at every size; failed first in 14 tests,
  passes 27/27 (43 "batch", 38 "="). Stress 109/109, correctness 91/91,
  TSan only #183. The WORK_STEALING half runs only on K: K's stress
  test and TSan verify it. No #191 timing gate: the old path also beat
  VECTORIZED at 1M, so it could not fail first; R8 shows the gain.
  `test_parallel_batch_gates` on M (first Mac run of the #175/#176 fix,
  load ~14): 0.19–0.23, all 8 under 0.75.
- Next: commit #190, #191 and this file; that head is the new freeze
  head. Then every machine: R1, R2's sweep compare, R8 (M's runs
  overnight, outside the sandbox), the stress test (K under TSan).
- Next on K and Z: pull, build, run the stress test (K also under TSan),
  compare with M's lists; no new harness work needed for that.
- **#191's slice tail breaks the differential on Z** [DERIVED, Z,
  2026-10-07, at `3339fc1`]. clang-cl: six `StrategyDifferential` tests
  fail, 13 (distribution, op) pairs: Beta, StudentT, LogNormal, Rayleigh
  pdf/logpdf; Weibull pdf/logpdf/cdf; Pareto cdf. MSVC: three tests, 5
  pairs (Beta pdf/logpdf, Weibull pdf/logpdf, Pareto cdf); MSVC does not
  contract to FMA, so more scalar fallbacks round as the kernel does. `parallelForSlices(count, 1024, …)` leaves a last slice of
  `count % 1024` elements; when that is 1 to simdMin − 1
  (`SIMDPolicy::computeOptimalThreshold`: 8 on AVX-512, AVX2, AVX; 4 on
  SSE2, NEON), the batch impl's `shouldUseSIMD` is false and the slice
  runs its scalar fallback (e.g. `weibull.cpp:860`), so one call mixes
  kernels. Z's fork threshold is 8192, so the sizes 8193 and 24583 end
  in slices of 1 and 7; M's 1536 gave 513 and 519, which is why M
  passed (`docs/bench-evidence/2026-10-07-zen4-concurrency/`). Values stay accurate; #191's contract (VECTORIZED's bits at
  every size) fails, on any machine at such sizes.
- Decided [user, 2026-10-07]: **K fixes it with its work-stealing
  work.** Fold a short final slice into the previous one in both
  helpers, `ParallelUtils::parallelForSlices` (`thread_pool.h:275`,
  also under `parallelTransform`) and `WorkStealingPool::parallelForSlices`
  (`work_stealing_pool.h:267`), so every kernel's tail lands on the same
  elements as VECTORIZED's. Add sizes 1024·k + 1 and 1024·k + simdMin − 1
  above the fork threshold to the differential, so it fires on every
  machine; show it failing on K before the fix. Moves the freeze head;
  Z then reruns R1, the sweep compare, the stress test and R8.
- **Done on K** [DERIVED, 2026-10-07, `045acd9`]. Both helpers fold a
  partial final slice into the previous one (any remainder [user]), so
  slices start at multiples of 1024 and the last holds 1024–2047
  elements. The differential gains 1024·k + 1 and 1024·k + simdMin − 1
  above the fork threshold (K: 6145, 6151); before the fix it failed 14
  distributions on K under PARALLEL and MAX_THROUGHPUT (WORK_STEALING
  on K), after it all 81 pairs run the batch kernel. **C7 probe written**
  [user]: `WorkStealingCrossWaiting` runs a MAXIMIZE_THROUGHPUT batch
  beside a pool task that blocks until released; the batch must return
  within 1 s. It fires on K for every lambda still calling
  `pool.waitForAll()` (Beta, Binomial, FisherF, NegativeBinomial, Gamma,
  LogNormal, InverseGamma, VonMises) and their delegates (ChiSquared,
  Erlang, Bernoulli, Geometric): 28 operations, known #180 skips until
  #180's fix; it skips where WORK_STEALING is unreachable (M, Z). K's
  stress run: 124 passed, 12 skipped (#180), 128 s (M: 68 s; the #180
  waits are 28 s of it).

### R6 — release docs (after R1–R4 everywhere, R3 everywhere and R8)
Version 2.4.1 → 2.5.0: `CMakeLists.txt:83`, README (status lines and the
stale test counts), AGENTS "Current status", PROJECT_CONCEPT. CHANGELOG
via git-cliff. VALIDATION_HISTORY: the three-machine matrix, the costs,
the R4 records and the R8 table update. ACCURACY_CHARACTERIZATION: the three generated blocks
only. This file.

Docs sweep [user, 2026-10-05; OPEN]: this cycle changed more than the
version. clang-cl is the Windows performance build with cl.exe's
`/arch:` flags; Windows timing needs the power-throttling opt-out;
MSVC's vectorization limits; v2.5.0/v3.0.0 numbering. Candidates:
README, PROJECT_CONCEPT, MIGRATION_GUIDE, AGENTS, `tools/` README,
`docs/` (BUILD_SYSTEM_GUIDE, CI_CD_GUIDE, SIMD_OPTIMIZATION_REFERENCE,
VALIDATION_HISTORY, whose "Zen 4 frequency scaling" needs re-reading
against the throttling finding). Start from an inventory of stale
statements (read-only search agent), not from memory. Cross-repo:
standards WINDOWS-TOOLCHAIN gains the `/arch:` equivalence and the
throttling opt-out.

### R7 — release
PR
`dev/v2.4.2` → `main` closing #157–#172 and #175–#177 (#173 stays with corvus); CI
green, including the sanitizer legs; merge; signed tag `v2.5.0`; GitHub
release from the CHANGELOG section; close milestone #10. Then coordinate
the pylibstats pin bump (Cross-Repo below), and tell the corvus session
about the merge note.

Corvus merge note [DERIVED]: `math_utils.{h,cpp}`, `student_t.cpp`,
`gamma.cpp`, `poisson.cpp`, `binomial.cpp`, `negative_binomial.cpp`,
`von_mises.cpp`, `exponential.cpp`, `weibull.cpp`, `rayleigh.cpp`,
`tests/CMakeLists.txt` and this file will conflict. Keep this branch's
fixes; corvus's definitions replace `gamma_p_inv` and the incomplete
gamma/beta where the corvus release swaps them in. `von_mises.{h,cpp}`:
take this branch's whole (`96b157f`, fixed-cost quadrature; no Bessel
series left to swap). `beta.cpp`, `simd_dispatch.cpp`, `pareto.cpp`
also changed here since this note was first written. `vector_log1p` and
`vector_expm1` point at corvus once corvus #45 lands.

## Decided [DERIVED]
- **Windows performance build is clang-cl [user, 2026-10-05]**, with
  cl.exe's `/arch:` flags (`/arch:AVX2` = AVX2 + FMA, F16C, BMI1/2,
  LZCNT, MOVBE; `/arch:AVX512` adds AVX-512 F, CD, BW, DQ, VL), so both
  compilers build the same ISA. cl.exe is the correctness build
  (standards WINDOWS-TOOLCHAIN §5); Z's R3/R8 and kAvx512 come from
  clang-cl.
- **Release numbering [user, 2026-10-04].** `dev/v2.4.2` ships as
  v2.5.0: no public API change, but consumers see different results
  (0 contract violations, new tail and edge behaviour), a different
  speed profile (slowdowns and re-derived dispatch tables, R8) and a
  new supported compiler (clang-cl, `6245e4e`). Corvus adoption
  (`dev/v2.5.0-corvus`, milestone #6) becomes v3.0.0: the installed
  package gains `find_dependency(corvus 1.0 CONFIG)`, so under
  `SameMajorVersion` a `find_package(libstats 2.x)` consumer would
  accept it and fail to configure without corvus; it also ends
  "zero external dependencies". Branch names stay. Knock-on [user,
  2026-10-04]: New Distributions (#3) becomes v3.1.0, after corvus;
  the Architecture Refactor (#4) becomes "post-v3", v3.x or v4 decided
  later. The accuracy patch (#8) may be overtaken by this release (see
  GitHub Milestones). GitHub milestones retitled 2026-10-04 [user].
- **Accuracy-for-speed budget [user, 2026-10-04].** Slowdowns of 2×, 5×
  and up to ~10× are acceptable for accuracy fixes; 50–150× is not. 10×
  is a guide, not a hard limit: weigh how hot the function is for
  consumers (a Gaussian pdf matters far more than a von Mises quantile)
  against the effort, and maximise speed at the accuracy kept, within
  reasonable effort. Over-budget rows get a performance pass or a
  theoretical case that the cost is intrinsic (Known Gaps, "Over-budget
  slowdowns"). Code on this branch may change again for speed, and R8's
  tables are re-profiled after it.
- **Vectorized log1p [user, 2026-10-04].** A `VectorOps` primitive is
  preferred to an inline composite. Corvus has one (`corvus::log1p`,
  span, correctly rounded) but corvus arrives only in v3.0.0, so v2.5.0
  derives `vector_log1p` once, generically, from each tier's
  `vector_log` (the compensated `log(u) − ((u−1)−z)/u`, `u = 1+z`), and
  v3.0.0 points it at `corvus::log1p`. `vector_expm1` likewise [user,
  2026-10-04] (Taylor below ½, `vector_exp` less 1 above). Neither
  serves log1p(x) − x: its cancellation amplifies their few ulp (1e-12
  at large α), so `detail::log1pmx_series` (atanh form) does. corvus
  exports for `expm1` and `log1pmx` are requested (corvus #45).
- **Numerical kernel promotion [user, 2026-10-04].** A better method
  found here is fixed here, written to be moved (generated, verified
  constants; regime map; independent oracle; fail-first guards), and
  filed with corvus, which decides by its doctrine; after adoption
  libstats calls corvus and drops its copy. Distribution-specific
  kernels (the von Mises CDF) stay; their reusable machinery moves.
  Fleet rule: OldCrow/standards `NUMERICAL-KERNEL-PROMOTION.md`
  (adopted, `bf493ff`); filed: corvus #44 (the von Mises quadrature
  machinery, and fixed-cost incomplete gamma/beta), corvus #45.
- **Support-boundary rule (#161, #165) [user, 2026-10-03].** pdf/logpdf at
  a support boundary take the limit by shape: at x = 0, +inf for
  shape < 1, the finite density for shape = 1 exactly, 0 / −∞ above. Out
  of support is 0 / −∞ everywhere, Poisson included; no distribution
  returns `MIN_LOG_PROBABILITY`.
- **Special parameters [user, 2026-10-03].** A value path takes the
  shortcut for λ, α, β = 1 (or a standard parameter set) only on exact
  equality. The `isCauchy()` / `isStandard()` style queries and
  `operator==` keep `DEFAULT_TOLERANCE`: they report, and no value path
  reads them.
- **Quantile contract (#104) [user, 2026-09-02].** Finite best-effort:
  never NaN for p ∈ (0, 1), ±inf only on true overflow; discrete
  quantiles are min{k : F(k) ≥ p} on the library's own CDF.
- **±inf contract (#103) [user, 2026-09-02].** The mathematical limit
  where one exists, scalar and batch identical; von Mises keeps
  saturation as a documented exception.
- **Clean-room replacement is the remedy for a provenance defect.** An
  isolated child agent authors from a functional spec, with no access to
  the suspect implementation or its upstream. The orchestrator audits
  and integrates, and never authors. Each replacement ships a derivation
  doc and a divergence audit under `docs/` (proven on #67).
- **#172: no code with licence concerns used or derived from [user,
  2026-10-03].** It, and the review fixes after it, are built from
  published mathematics (Stirling's series, A&S 6.5.29) and the library's
  own helpers.
- **corvus constant-argument calls [user, 2026-09-29]** (v2.5.0). Fill a
  constant span per block on the stack, inside the existing `vector_*`
  adapters, rather than requesting a corvus broadcast overload. Settle the
  block size, and `vector_beta_i`'s argument order, in the swap.
- **Project skills** live once, in `.claude/skills/`; `.agents/skills` is
  a tracked relative symlink. Edit only `.claude/skills/`. On Windows it
  needs Developer Mode plus a repository-local `core.symlinks=true`; Git
  for Windows defaults to false, and a clone records the value it found.
  Z is set (2026-10-03).
- **`origin/spike/corvus-bessel`** holds the only Tier 0 corvus Bessel
  code (`a1c71d6`). It is not merged; keep it.

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-10-06 (milestones and
open issues below; no open issue without a milestone; one open PR,
#174, a Dependabot bump of reviewdog/action-actionlint).
- GitHub is the collaborator-facing source for issues and milestones;
  this file is the agent-facing state. Keep both in sync: when creating,
  closing, retitling or moving an issue or milestone, update this section
  in the same change set.
- Re-check when the task reads the backlog or changes it, or when more
  than 7 days have passed since the date above.
- Open milestones are itemized here; closed ones are counts only.

## GitHub Milestones [DERIVED]
Closed: #1 v2.2.0 (5), #5 v2.3.0 (5), #7 v2.3.1 (13), #2 v2.4.0 (6), #9
v2.4.1 (3). Milestone numbers do not sort in version order (two title
renumberings, 2026-07-21 and 2026-08-16).

Release order [user, 2026-10-04]: v2.5.0 (milestone #10, `dev/v2.4.2`)
→ v3.0.0 corvus (#6) → v3.1.0 New Distributions (#3) → Architecture
Refactor (#4, post-v3). The accuracy patch (#8) is under review below.
Retitled 2026-10-04 [user], each description carrying a dated
renumbering note: #10 "v2.5.0 — Correctness & accuracy", #6 "v3.0.0 —
corvus adoption", #3 "v3.1.0 — New Distributions (Extended)", #4
"Post-v3 — Architecture Refactor"; #8 note only.

#8 against this release [DERIVED, 2026-10-04; verify before moving
anything]:
- Likely overtaken: #104 (quantile contract at extreme p): its Decided
  contract stands and v2.5.0 fixed the deep-tail quantiles (#159, #160,
  the log1p quantiles; 0 contract violations on AVX-512 and AVX2).
  Re-check its listed cases, then close.
- Partly overtaken: #103 (±inf inputs): the support-boundary rule
  (#161, #165) covers x = 0 and out-of-support; check the +inf cases it
  lists before closing or narrowing.
- Done by the speed work and R8's tooling, closed 2026-10-05: #146
  (`scripts/analyze_crossovers.py`, `3e9fde3`), #111 (the von Mises
  batch CDF, `96b157f`). #144 (von Mises CDF thresholds) waits for R8's
  table update.
- Unaffected: #152 (Codecov measurement), #114 (review backlog). If the
  above go, #8 is these two; fold them into a post-v3 patch or close #8.
- **#10 v2.5.0 — Correctness & accuracy** (open, 17): #157–#167, #170–#172,
  #175–#177 (filed 2026-10-06: #175 per-element lock in PARALLEL lambdas,
  #176 Student-t CDF serial PARALLEL, #177 single-shot timing test).
  Closed by the release PR.
- **#6 v3.0.0 — corvus adoption** (open, 13): #47, #52, #107, #108, #110,
  #113, #126, #136, #137, #138, #141, #156, #173. Full swap [user,
  2026-09-17]: every `detail::` special function corvus covers. #126 and
  #141 are absorbed, measured against corvus v1.0.0. Corvus is pinned at
  v1.0.1. State and design: that branch's PLAN.md.
- **#8 Accuracy, contracts & kernel hygiene patch** (open, 11): ships after
  corvus (v3.0.0); under review, see above. #178–#183 filed 2026-10-06
  from the lock and parallel-path audit, not fixed for v2.5.0 [user]:
  #178 InverseGamma's serial passes, #179 `parallel_execution.h`'s
  include-order backend choice (Linux), #180 the global `waitForAll()`
  after a latched `parallelFor`, #181 the batch-fit fallback racing
  submitted tasks, #182 FisherF `sample()` mixing a snapshot with the
  resynced delegate, #183 unlocked validation reads and dead helpers.
  Bucketed [user, 2026-09-29]:
  - A, independent of adoption: #152 (Codecov measurement); #146 closed.
  - B, re-scope after the post-swap sweep and timing: #103, #104, #144;
    #111 closed.
  - C, one pass after the swap: #114.
- **#3 v3.1.0 — New Distributions (Extended)** (open, 5): #58 GEV, #59
  LogLogistic, #60 Triangular, #61 Wald, #62 Hypergeometric +
  BetaBinomial + Zipf. Settle #62's Zipf CDF design (summation or the
  Hurwitz-zeta closed form) before planning; it scopes corvus work.
- **#4 Post-v3 — Architecture Refactor** (open, 5): #40, #41, #42, #43,
  #128.

## Trends from the v2.5.0 captures [OPEN, 2026-10-05; tested 2026-10-06]
First extracted from K's three tiers, M's NEON and Z's throttled
overnight as markers for analysis [user]; tested on 2026-10-06 against
the three freeze-code datasets (K's AVX2/AVX/SSE2 at `6f86996`, M's NEON
at `2c5230a`, Z's clang-cl AVX-512/AVX2/AVX/SSE2 at `69c7be4`), per
machine and together [DERIVED]. A difference counts only beyond that
row's own three-run spread (grid steps). Three runs in one night are a
lower bound on noise.

For v2.5.0:
- **"Table noise exceeds table precision": rejected as stated.** Run
  noise is a median 0–1 grid step; only 3–11 of 81 rows per set spread
  > 2 steps. Rows differing from their table beyond that noise: K 24
  (AVX2), 9 (AVX), 27 (SSE2); M 22; Z 41, 42, 28, 38. The tables are
  off because crossovers are per machine (below) while a table serves an
  architecture.
- **"AUTO is too reluctant": true only at costly parameters.** On the
  profile's default parameters the tables cross too early as often as
  too late (the cheapest third: early 8–11, late ~2 on most sets). R3's
  AUTO-vs-best gaps favour PARALLEL (K 23/24, M 23/23, Z 28/33), mostly
  at costly parameters (Beta 25 and 1000 shapes, Gamma α ≥ 20, Student-t
  ν ≥ 1e3, large counts: 18/24, 17/23, 19/33). Per-element cost varies
  up to 10× with the parameters, so one threshold per (distribution,
  operation) cannot serve both.
- **Two parallel-path defects** (#175, #176, v2.5.0; predate v2.5.0,
  v2.4.1 has both). (1) The PARALLEL lambdas of `binomial.cpp`,
  `negative_binomial.cpp` and `von_mises.cpp` call the scalar method per
  element, and each call takes `withCacheSnapshot`'s `shared_lock` on
  the object's one mutex: PARALLEL/VECTORIZED at 2M elements 1.5–12×
  slower, worst on the Macs (GCD) and through Bernoulli and Geometric,
  which delegate. (2) The Student-t CDF's PARALLEL lambda is a serial
  loop: exactly 1.00× on every machine. These are the NEVER rows that
  look costly; their table rows wait for the fix (R8). Fixed at
  `73ecbfe`; the audit it prompted found more, fixed at `ee11e62` or
  filed (Known Gaps, GitHub Milestones).
- **kSse2: confirmed, SSE2 crosses sooner.** K 21 sooner / 0 later, Z 15 /
  1, the rest within noise, following SSE2's cost over AVX (median 1.16×
  K, 1.65× Z). Delegating to kAvx is late on a fifth to a quarter of rows.
- **Remaining over-budget rows:** Student-t quantile at ν ≥ 1e3 (K, M
  7–9×; Z 11.6–13.3×) and the von Mises quantile (K, M 5–9×; Z 17–23×,
  Z's ratio from v2.4.1's cheap grid; in absolute terms Z is fastest).
  Ship as documented gaps, or give Student-t the von Mises treatment (a
  fixed-cost method, normal-limited at large ν; a corvus #44 kernel).

For corvus adoption (v3.0.0) and later:
- **Thresholds follow per-element cost: supported** [DERIVED,
  2026-10-06]. log(crossover) against log(VECTORIZED ns/element) has
  slope −0.5 to −1.03 (−1 predicted; M −1.03, R² 0.85; K AVX −0.97,
  0.70; weakest Z SSE2 −0.51, 0.29), and one constant per machine and
  tier (2^16.8–2^18.2 ns) puts 69% of 648 rows within noise + 1 step,
  against 75% for the hand-tuned tables. K and Z differ by cost, not by
  noise: Z crosses later on 19–27 rows and sooner on 9–11, its
  vectorized cost about half K's, and the crossover ratio follows the
  cost ratio with slope +1.1 (AVX2, AVX; SSE2 +0.36). Exceptions are the
  two defects above. Corvus replaces the
  kernels and invalidates every table, so spend little on v2.5.0 table
  precision. Evaluate a cost model (per-element costs × one per-machine
  overhead, perhaps calibrated at startup) against the tables; it ties
  into the post-v3 Architecture Refactor.
- **Don't depend on auto-vectorization.** Beta's large-shape cost was
  6–11× under MSVC and 2.3–2.8× on M until `998d985` wrote the Horner
  out (Known Gaps, MSVC codegen). Promoted kernels use explicit SIMD, or
  must be shown to vectorize on all three compilers: a candidate
  criterion for standards `NUMERICAL-KERNEL-PROMOTION.md`.
- **One capture harness**, after v2.5.0 and written to whatever
  threshold model v3 adopts [user, 2026-10-05]. Each machine needed its
  own quiet rule (K load < 2.0, M ambient CPU, Z the EcoQoS opt-out),
  and throttling cost Z one overnight. It covers warm-up, the quiet
  gate, the throttling opt-out, noise logging and the bundle layout;
  it could be a standards convention shared with corvus and libhmm.
- **Accuracy gate for each corvus swap:** all three ISA blocks now show
  0 contract violations. Proposed acceptance: 0 violations, and no row
  worse than the v2.5.0 block, on every ISA.

## Known Gaps [OPEN]
- **R9 findings on M** [DERIVED, 2026-10-06; filed 2026-10-07 as
  #184–#191 (F1–F8 in order), milestone "Accuracy, contracts & kernel
  hygiene patch"; detail in
  `docs/bench-evidence/2026-10-06-m1-concurrency/summary.txt`]:
  - F1 (C1) #184: move-assignment takes no lock in all 27 distributions
    (TSan; the stress test sees wrong values in 26). Same class as
    Uniform's copy-assignment, fixed in `ee11e62`.
  - F2 (C2) #185: TruncatedNormal `fit` reads the bounds, computes, then
    writes μ and σ under a later lock (`truncated_normal.cpp:661`);
    racing a bound setter, 741–2998 of 3000 rounds end in neither serial
    order.
  - F3 (C4) #186: InverseGamma and FisherF `sample(rng, n)` loop the locking
    scalar `sample`, so one call mixes states; others snapshot once.
  - F4 #187: Poisson `operator>>` cannot read `operator<<`'s output (`"λ="`
    + 2 bytes lands on `=`; `poisson.cpp:1020`).
  - F5 #188: Gaussian `sample`'s `static thread_local` Box–Muller spare
    (`gaussian.cpp:330`) makes a draw depend on the thread's earlier
    draws; identically seeded generators disagree.
  - F6 #189: Poisson `validateCurrentParameters` is defined `inline` in
    `poisson.cpp:246`; callers fail to link.
  - F7 #190: `SystemCapabilities` hardcodes physical = logical / 2
    (`system_capabilities.cpp:28`); the macOS PARALLEL/WORK_STEALING
    choice reads it, so the M1 is treated as SMT.
  - F8 (perf) #191: PARALLEL above the fork threshold runs the scalar formula,
    not the batch kernel, in 33 of 81 (distribution, op) pairs.
  - Q1 (decided B, 2026-10-07): FORCE_SCALAR batches lock per element
    (`executeStrategy`'s SCALAR case) and can mix states; documented as
    per-element, one state under every strategy deferred to v3.
- **Lock and parallel-path audit** [DERIVED, 2026-10-06; read-only agent,
  findings verified]. Fixed at `ee11e62`, each gate failing first on
  `73ecbfe` except where noted: Poisson `getMedian`'s recursive shared
  lock (deadlocked on Windows); Uniform/Discrete bound-setter TOCTOU;
  Uniform's unlocked copy-assignment (no deterministic gate; TSan, R8
  step 5); delegate setters (Bernoulli, Geometric, ChiSquared, Erlang,
  InverseGamma) updating the delegate outside their lock;
  `parallelTransform`'s repeated chunks; PARALLEL below the fork
  threshold running a serial scalar loop; Beta's boundary values from
  the locking scalar methods; `d == d` locking twice (14 distributions;
  cannot fail first); the work-stealing pool's construction race (TSan).
  Filed, not fixed for v2.5.0 [user]: #178–#183. The 48 PARALLEL lambdas'
  own `should_use_parallel` gates are now dead (the dispatcher routes
  below a higher threshold); left in place.
- **Over-budget slowdowns** (budget: Decided, "Accuracy-for-speed
  budget"). Resolved 2026-10-04 [DERIVED, K, indicative, against
  v2.4.1]:
  - von Mises quantile 52–136× → 5–7×; CDF tail 0.1–1.1×, bulk
    0.14–0.75×, batch 0.27–3.5× (`717564f` levers 1–2, `96b157f` the
    fixed-cost quadrature rewrite; the Bessel series is gone).
  - Gamma α ≥ 20 pdf/logpdf 8–15× → about 2–5× (`b5a47df` log1pmx and
    hoisted constants, `b1ffc16` vectorized Stirling batch).
  - Beta: the batch is SIMD at every shape (`97c67bf`); with one shape
    large 0.5–0.7× `b1ffc16`, with both large (now Stirling form, for
    accuracy) logpdf 2.2× and pdf 1.3× `b1ffc16`, about 3× v2.4.1.
  - Student-t, Exponential, Rayleigh: the generic `vector_log1p` and
    `vector_expm1` lost to their earlier paths, which were kept
    (`9ca52db`; A/B in Status).
  - Within budget, left: Student-t CDF/quantile at large ν (5–8×, the
    incomplete-beta continued fraction; corvus #44's research question),
    discrete log-pmfs (0.9–2×), FisherF (2–3×).
- **MSVC codegen behind clang-cl** [DERIVED, Z, 2026-10-05]. Windows
  performance numbers come from clang-cl [user, 2026-10-05; standards
  WINDOWS-TOOLCHAIN §5]; MSVC is the correctness build. The Beta and
  Gamma Stirling batches were scalar under MSVC (reason 1106, the Horner
  loop in `log1pmx_series`); `998d985` writes it out and MSVC vectorizes
  them, 2.2–3.3× faster, Beta at clang-cl's speed. Left: Gamma at large
  shapes 1.5–1.6× behind clang-cl, since MSVC does not contract FMAs
  (`/fp:contract` is a build-flag change under the freeze, and the
  FP-contraction rule in AGENTS.md); `vector_expm1`'s Taylor loop scalar
  under MSVC (reason 1100, its |x| < ½ select; every select form tried),
  which the Pareto and Weibull batch CDFs use. Evidence:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/`.
- **Beta pdf/logpdf at large shapes**: resolved in `97c67bf` (Stirling
  form from both shapes 20; logpdf 3.6e-10 → 2.8e-14, pdf 1.8e-11 →
  1.0e-13), with the far-from-mode log1p fix it exposed in Gamma and
  Beta (x ≪ mode; Gamma(1000) logpdf at 1e-15 was −inf). Its cost: Known
  Gaps above. corvus #45's span `log1pmx` would carry the series.
- The `exp_max` clamp sits ~30 ULP below the true overflow threshold, so
  in that one-double window the kernels return `exp(exp_max)` where
  `std::exp` is still finite. A deliberate margin; left as is.
- von Mises circular variance (#93 residual): forming the complement
  1 − A costs about 2κ × A's ULP error near the κ = 50 cut. That is close
  to intrinsic in double, and corvus does not remove it: its exports
  return doubles. Only a double-double complement export would.
- The past-INT_MAX CDF guard tolerance (2e-4) comes from `beta_i`'s
  lgamma floor. Tighten it when the v2.5.0 incomplete-beta core lands.
- The pinned clang-format is 20.1.8; the cached pre-commit environment on
  Z has 19.1.7, and CI only reports format (clang-format-17, `|| true`).
  The older drift in `fisher_f.cpp` and `beta.cpp` was cleared in
  `b5a47df`.

## Cross-Repo Dependencies [OPEN]
- **pylibstats** pins this repo by a `find_package` floor and a
  FetchContent `GIT_TAG` in `pylibstats/CMakeLists.txt`, which is the
  only source of the version; its pin-currency canary fails when it falls
  behind. Before a release or an API break, check that pin and coordinate
  the bump. It is at v2.4.1 (pylibstats 0.7.1, PR #22); v2.5.0 is owed
  after R7, and v3.0.0 (corvus) needs a major-version floor change.
- **corvus issues from v2.5.0** [2026-10-04]: #44 (promote the von Mises
  quadrature machinery; fixed-cost incomplete gamma/beta as research),
  #45 (span `expm1`, `log1pmx`). No milestone. The corvus branch's von
  Mises Bessel changes are superseded by `96b157f` (note on that branch).
- **corvus dependency cost to the wheels** [priced 2026-09-17; full
  record: `git show f508b12:PLAN.md`, Cross-Repo]. Open, for the v2.5.0
  swap PR:
  - libstats find-or-fetches corvus pinned at v1.0.1;
  - install is supported only with a system corvus;
  - corvus stays out of installed headers (move the `bessel.h`
    wrappers into a TU);
  - `find_dependency(corvus)`;
  - THIRD_PARTY_NOTICES (corvus MIT, Highway Apache-2.0);
  - a corvus pin canary.

  For pylibstats v0.8.0 (pylibstats #20): the Windows wheel (raise the
  timeout and accept corvus's AVX2 cap, or clang-cl), NOTICE and
  Apache-2.0 text in the wheel, and pylibstats' missing LICENSE file. For
  corvus v1.1.0: Linux aarch64 (GCC) NEON is unvalidated.
- **libhmm** shares rules and reference kernels: libhmm#103 (`errorf_inv`
  saturation class), libhmm#108 (Codecov port after #152).
