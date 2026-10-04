# libstats — Plan / Status

State only: what is decided, open, or next. Release contents are in
`CHANGELOG.md`, per-version validation in `docs/VALIDATION_HISTORY.md`,
conventions and commands in `AGENTS.md`. The history this file used to
carry (shipped releases, closed milestones, the Resolved log, the
2026-08-21 defensive review, travel-era records, the corvus spike) is in
`git show f508b12:PLAN.md`.

## Status [DERIVED] — 2026-10-03
**v2.4.2 in progress [OPEN]** on `dev/v2.4.2` (cut from `main` at
`d8d3388`; milestone #10 "v2.4.2 — Correctness patch", #157–#172, which
the release PR closes). All code is in except #162. What remains is the
runbook below: validation on Kaby Lake and the M1, the quiet-machine
costs on all three, docs, and the release. Every fix has a gate shown to
fail on the code before it.

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

## v2.4.2 release runbook [OPEN]
Machines: **Z** = Zen 4 (Windows 11, MSVC, AVX-512), **K** = Kaby Lake
(macOS Ventura, AppleClang, AVX2), **M** = Mac Mini M1 (macOS 27,
AppleClang, NEON).

Status by machine (each machine edits only its own line):
- **Z:** R0 done (`tools/bench/`, `8cf5550`). R1 done (fresh configure,
  clean rebuild, 86/86, Release CRT). R2 done: AVX-512 block regenerated
  at `f89380a` (`f508b12`), 0 contract violations (v2.4.1: 32). Next: R3
  when quiet; R5 optional.
- **K:** R1 done (AppleClang 15, macOS 13.7.8, CMake 4.4.3; `a0a913e`
  fixed the SDK `label` collision and sign-compare warnings in tests;
  86/86). R2 done: AVX2 block at `a0a913e` (`4f40228`), 0 contract
  violations. R4 done: #162 reproduced on head and fixed (`8018c5d`);
  #167 fail-first under UBSan (`d8d3388` aborts at `poisson.cpp:1240`,
  head clean). Run binaries outside the Claude Code sandbox, which blocks
  the cache-size sysctls. Next: R3 when quiet.
- **M:** R1, R2, R3 to do; R4 here or on K.

### Rules on every machine
- Commit or push only when the user asks. Commits are signed (YubiKey);
  never disable signing; batch a commit and its push.
- Edit only your own status line above; `git pull --rebase` first.
- New defects or unexpected oracle rows: stop and report with evidence.
  Fix nothing in library code without the user's decision.
- Quiet runs (R3) alone on the machine: no builds, sweeps or other
  sessions. On K, wait for the load average to fall below 2.0.
- Performance numbers from Release builds only.

### Cold start (K, M)
1. `git fetch`; `dev/v2.4.2` must be at `f508b12` or later. Work in a
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
   86/86. The target `run_tests_correctness` runs a 79-test subset.

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

### R4 — Mac-only gates (K or M, once)
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
6. Report section 3 of `compare.txt` (AUTO-vs-best gaps marked NEW). Any
   re-derivation of `dispatch_thresholds.h` rows (`strategy_profile` on
   the same machine, then R1 again) is the user's decision. INDICATIVE
   smoke run on a busy Z: von Mises quantile 140–320× slower (3–6 µs
   Newton on the quadrature CDF, was a grid); forced VECTORIZED 8–13×
   slower for Gamma α ≥ 20 and Student-t ν ≥ 1e3 (scalar fall-backs); 22
   NEW gaps, all favouring PARALLEL at n = 1e4–1e5.

### R5 — optional capped-tier leg (Z)
v2.4.0 precedent: a `LIBSTATS_MAX_SIMD_TIER` build, correctness and
sweep. Before Z's R3.

### R6 — release docs (after R1–R4 everywhere and R3 everywhere)
Version 2.4.1 → 2.4.2: `CMakeLists.txt:83`, README (status lines and the
stale test counts), AGENTS "Current status", PROJECT_CONCEPT. CHANGELOG
via git-cliff. VALIDATION_HISTORY: the three-machine matrix, the costs
and the R4 records. ACCURACY_CHARACTERIZATION: the three generated blocks
only. This file.

### R7 — release
PR `dev/v2.4.2` → `main` closing #157–#172 (#173 stays on v2.5.0); CI
green, including the sanitizer legs; merge; signed tag `v2.4.2`; GitHub
release from the CHANGELOG section; close milestone #10. Then coordinate
the pylibstats pin bump (Cross-Repo below), and tell the v2.5.0 session
about the merge note.

v2.5.0 merge note [DERIVED]: `math_utils.{h,cpp}`, `student_t.cpp`,
`gamma.cpp`, `poisson.cpp`, `binomial.cpp`, `negative_binomial.cpp`,
`von_mises.cpp`, `exponential.cpp`, `weibull.cpp`, `rayleigh.cpp`,
`tests/CMakeLists.txt` and this file will conflict. Keep v2.4.2's fixes;
corvus's definitions replace `gamma_p_inv` and the incomplete gamma/beta
where v2.5.0 swaps them in.

## Decided [DERIVED]
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

Release order: v2.4.2 → v2.5.0 → the accuracy patch (version assigned at
ship, likely v2.5.1) → v2.6.0 → v3.0.0.
- **#10 v2.4.2 — Correctness patch** (open, 14): #157–#167, #170–#172.
  Closed by the release PR.
- **#6 v2.5.0 — corvus adoption** (open, 13): #47, #52, #107, #108, #110,
  #113, #126, #136, #137, #138, #141, #156, #173. Full swap [user,
  2026-09-17]: every `detail::` special function corvus covers. #126 and
  #141 are absorbed, measured against corvus v1.0.0. Corvus is pinned at
  v1.0.1. State and design: that branch's PLAN.md.
- **#8 Accuracy, contracts & kernel hygiene patch** (open, 7): ships after
  v2.5.0. Bucketed [user, 2026-09-29]:
  - A, independent of adoption: #146 (before the v2.5.0 threshold
    re-measure), #152 (Codecov measurement).
  - B, re-scope after the post-swap sweep and timing: #103, #104, #111,
    #144.
  - C, one pass after the swap: #114.
- **#3 v2.6.0 — New Distributions (Extended)** (open, 5): #58 GEV, #59
  LogLogistic, #60 Triangular, #61 Wald, #62 Hypergeometric +
  BetaBinomial + Zipf. Settle #62's Zipf CDF design (summation or the
  Hurwitz-zeta closed form) before planning; it scopes corvus work.
- **#4 v3.0.0 — Architecture Refactor** (open, 5): #40, #41, #42, #43,
  #128.

## Known Gaps [OPEN]
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
  Older drift remains in `fisher_f.cpp` and `beta.cpp`.

## Cross-Repo Dependencies [OPEN]
- **pylibstats** pins this repo by a `find_package` floor and a
  FetchContent `GIT_TAG` in `pylibstats/CMakeLists.txt`, which is the
  only source of the version; its pin-currency canary fails when it falls
  behind. Before a release or an API break, check that pin and coordinate
  the bump. It is at v2.4.1 (pylibstats 0.7.1, PR #22); v2.4.2 is owed
  after R7.
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
