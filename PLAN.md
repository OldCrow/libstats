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

**Code freeze at `6cab6ed` [user, 2026-10-04]**, moved from `5367999`
by the Windows build fix (`far` is a windows.h macro; a rename in
`gamma.cpp` and `beta.cpp`, the same code on clang). K's runs at
`5367999` stand [user, 2026-10-04]. No change to `src/`,
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
  the code's, within the contract. R3 and the R8 captures (AVX-512, AVX2,
  AVX, SSE2) run overnight 2026-10-04; bundles and evidence commit after
  review. R5 skipped [user].
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
  `4ceafae` (`dce2656`), 0 contract violations, no row worse. R3 and
  the R8 re-captures (AVX2, AVX, SSE2) run overnight 2026-10-04; their
  bundles and evidence commit after review. R4 stands.
- **M:** macOS 27.0.1, AppleClang 21.0.0 (clang-2100.3.34.2), CMake
  4.4.3, Ninja 1.13.2. At `2c5230a` (freeze head plus tests/docs,
  2026-10-04): R1 done (clean `build-release`, 0 warnings, NEON, 89/89).
  R2 done: NEON block at `2c5230a`, 0 contract violations (v2.4.1: 32);
  no row worse than the AVX-512 block. Fifteen rows are worse than the
  v2.4.1 NEON block (Cauchy cdf 2.9e-16 → 2.1e-12; Gaussian, LogNormal
  pdf/cdf 7–20×; Pareto quantile, Rayleigh pdf), each by the same
  factor in both x86 blocks, so the branch's, within the contract;
  oracle run with the `libstats-log-cleanroom` venv (mpmath 1.3.0). R4
  skipped (done on K). R8 (NEON, three `--large` runs) and R3 run
  overnight 2026-10-04 from `../libstats-r3m/overnight_m1.sh`, gated on
  ambient CPU < 15% over two 10 s windows (relaxing to 20% after 3 h),
  not on load: the M1's 1-min load sits at 4–10 idle. Bundles and
  evidence commit after review.

### Rules on every machine
- Commit or push only when the user asks. Commits are signed (YubiKey);
  never disable signing; batch a commit and its push.
- Edit only your own status line above; `git pull --rebase` first.
- Validate at the freeze head `6cab6ed` (Status); confirm it with
  `git log -1` before R1. Library code is frozen: see Status.
- New defects or unexpected oracle rows: stop and report with evidence.
  Fix nothing in library code without the user's decision.
- Quiet runs (R3) alone on the machine: no builds, sweeps or other
  sessions. On K, wait for the load average to fall below 2.0.
- Performance numbers from Release builds only.
- On the Macs, run every library binary (tests, tools, sweeps, benches,
  profiles) outside the Claude Code sandbox: it blocks the
  `hw.*cachesize` sysctls, so cache sizes read 0 and the cache-derived
  tuning changes. The sandbox also blocks the SSH agent for `git fetch`.

### Cold start (K, M)
1. `git fetch`; `dev/v2.4.2` must be at the freeze head `6cab6ed` or
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
machine whose table changed.

### R6 — release docs (after R1–R4 everywhere, R3 everywhere and R8)
Version 2.4.1 → 2.5.0: `CMakeLists.txt:83`, README (status lines and the
stale test counts), AGENTS "Current status", PROJECT_CONCEPT. CHANGELOG
via git-cliff. VALIDATION_HISTORY: the three-machine matrix, the costs,
the R4 records and the R8 table update. ACCURACY_CHARACTERIZATION: the three generated blocks
only. This file.

### R7 — release
PR
`dev/v2.4.2` → `main` closing #157–#172 (#173 stays with corvus); CI
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
Last reconciled against live GitHub state: 2026-10-03 (milestones and
open issues below; no open issue without a milestone; no open PRs).
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
- Pulled into v2.5.0 by R8 and the speed work: #146 (sustained-crossover
  tooling; bucket A said "before the threshold re-measure", which is
  now), #144 (von Mises CDF thresholds), #111 (von Mises batch CDF
  passes and allocations).
- Unaffected: #152 (Codecov measurement), #114 (review backlog). If the
  above go, #8 is these two; fold them into a post-v3 patch or close #8.
- **#10 v2.5.0 — Correctness & accuracy** (open, 14): #157–#167, #170–#172.
  Closed by the release PR.
- **#6 v3.0.0 — corvus adoption** (open, 13): #47, #52, #107, #108, #110,
  #113, #126, #136, #137, #138, #141, #156, #173. Full swap [user,
  2026-09-17]: every `detail::` special function corvus covers. #126 and
  #141 are absorbed, measured against corvus v1.0.0. Corvus is pinned at
  v1.0.1. State and design: that branch's PLAN.md.
- **#8 Accuracy, contracts & kernel hygiene patch** (open, 7): ships after
  corvus (v3.0.0); under review, see above. Bucketed [user, 2026-09-29]:
  - A, independent of adoption: #146 (before the v2.5.0 threshold
    re-measure), #152 (Codecov measurement).
  - B, re-scope after the post-swap sweep and timing: #103, #104, #111,
    #144.
  - C, one pass after the swap: #114.
- **#3 v3.1.0 — New Distributions (Extended)** (open, 5): #58 GEV, #59
  LogLogistic, #60 Triangular, #61 Wald, #62 Hypergeometric +
  BetaBinomial + Zipf. Settle #62's Zipf CDF design (summation or the
  Hurwitz-zeta closed form) before planning; it scopes corvus work.
- **#4 Post-v3 — Architecture Refactor** (open, 5): #40, #41, #42, #43,
  #128.

## Known Gaps [OPEN]
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
