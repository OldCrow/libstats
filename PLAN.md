# libstats — Plan / Status

State only: what is decided, open, or next. Release contents are in
`CHANGELOG.md`, per-version validation in `docs/VALIDATION_HISTORY.md`,
conventions and commands in `AGENTS.md`.

History is in git, not here. This file was pruned on 2026-11-21: the
adoption branch's rebuild and CI narrative, the session logs of 2026-11-16
to 2026-11-21, the v2.5.0 release record and its runbook are in
`git show e1a11ec:PLAN.md`. Earlier history (shipped releases, closed
milestones, the 2026-08-21 review, the corvus spike) is in
`git show f508b12:PLAN.md`. Each fixed issue carries a status comment
naming its commits.

## v3.0.0 corvus adoption — where it stands [DERIVED, 2026-11-21, Ivy Bridge]

Branch `dev/v3.0.0-corvus-adoption`, built on the v2.5.0 release. corvus
v1.2.0 is the special-function engine (found or fetched, pinned;
`cmake/FindOrFetchCorvus.cmake`). `main` is the v2.5.x line at `99f2175`
and is merged in.

- **`src/` is settled for v3.0.0.** The last code change is `af2e548`;
  `e890dec` after it changes comments only. Nothing further is planned in
  `src/` before release, except threshold rows (each machine's own) and
  whatever a leg finds.
- **Validated so far on one machine only**: Ivy Bridge (AVX, GCC 13.3,
  Linux). Correctness 135/135, warning-free. CI green on every leg at
  `af2e548` (macOS AppleClang, GCC 14 with TBB, Clang 17 without, MSVC,
  clang-cl, ASan/UBSan, TSan, FMA-contraction, install contract). CI on
  `e890dec` was dispatched (run 44947886476) and **not read** when this
  was written.
- **Not yet done anywhere but here**: the accuracy sweep and block, the
  row gate and off-grid sweep, the prototype bit-identity check, quiet
  timing, threshold capture. That is the fleet work below.
- **Project version is still 2.5.0** in `CMakeLists.txt`; the bump is a
  release step.

## Fleet legs for v3.0.0 [OPEN; user, 2026-11-21]

Each machine runs one leg. The legs are independent and may run in
parallel: every machine reads only its own tier's threshold rows, and
nothing a leg needs is still open on another machine.

**Start from** the commit that carries this section, or later. Confirm
with `git log -1` and state the head in the leg's status line. Confirm CI
is green on `e890dec` or later before starting.

**Order if they are not run at once** [DERIVED]: M1, then Zen 4, then
Kaby Lake. The M1 differs most from what is validated (architecture,
AppleClang and libc++, GCD backend, hardware FMA), so a defect there is
the one most likely to change code the other legs would re-run. Zen 4 is
the longest leg (two compilers, the throttling opt-out). Kaby Lake covers
the least new ground but is the only native AVX2 tier. To clear the PR
gate soonest, run Kaby Lake first.

**What gates on the legs**: the dev-branch PR (one other leg complete);
removing the `PROTOTYPE` comment tags (all three report bit-identity);
#156's gates (all four machines' numbers); the release docs.

### Leg status (edit only your own line)

- **I — Ivy Bridge, AVX, Linux, GCC 13.3**: L1 done at `e890dec`
  (135/135). L2's AVX block is stale (banner `b766e45`); regenerate after
  the capture. L2b done: `data/accuracy/allow_AVX.csv`; off-grid sweep
  found #316, #317, #318. L3 done at the time (`90d1bef` against
  `b766e45`, sweep row for row). **L4 capture pending**: the user runs
  `build-ivb-adopt/quiet_run.sh` (local script; timing tests,
  `simd_verification`, the dev-against-`main` benches, three
  `strategy_profile --large` runs, about four hours). Then L5 for kAvx.
- **M — M1, NEON, macOS, AppleClang**: not started.
- **Z — Zen 4, AVX-512, Windows, clang-cl and cl.exe**: not started.
- **K — Kaby Lake, AVX2, macOS, AppleClang**: not started.

### What every leg produces

| Step | Output | Committed as |
|---|---|---|
| L1 native correctness | warning-free Release build; `system_inspector --quick`; correctness suite count; the backend `test_parallel_backend_selection` reports | status line |
| L2 accuracy sweep and block | this ISA's block in `docs/ACCURACY_CHARACTERIZATION.md`, 0 contract violations | `docs(accuracy): regenerate <ISA> block at <sha>` |
| L2b row gate and off-grid sweep | `data/accuracy/allow_<ISA>.csv`; a list of off-grid findings | the allowlist; findings filed as issues |
| L3 prototype bit-identity | sweep at `90d1bef` identical to sweep at `b766e45` | status line |
| L4 quiet session | timing tests, `simd_verification`, `tools/bench` dev against `main`, three `strategy_profile --large` runs | bundle under `data/profiles/dispatcher/`; bench evidence under `docs/bench-evidence/` |
| L5 threshold rows | this tier's rows in `dispatch_thresholds.h`, settled by `table_ab_bench` | header, script and bundle together |
| L6 report | bench numbers stated on #156; status line updated | — |

### Rules on every machine

- Commit, push, comment, label or file only when the user asks. Commits
  are signed; never disable signing.
- **Commit only your own**: your tier's rows in `dispatch_thresholds.h`,
  your `isa=` block, your `allow_<ISA>.csv`, your bundle and evidence
  directories, your status line here. `git pull --rebase` before every
  commit; this file is the one every leg edits.
- **Library code changes only by the user's decision.** A new defect or
  an unexpected oracle row: stop and report with evidence. If the fix
  also applies to `main`, it goes to `main` first on a small branch with
  a PR, then `main` is merged here; adoption-only fixes go straight to
  this branch. A fix that changes results means the other legs re-run L2
  from the new head, so say so in your status line.
- **A new guard must be shown to fail on the unfixed library**, on the
  platform it targets (AGENTS.md, Testing).
- CI: this branch has no push trigger. Dispatch with
  `gh workflow run ci.yml --ref dev/v3.0.0-corvus-adoption` after a
  change to `src/`, tests, tools or the workflow; do not poll.
- Performance numbers from Release builds only. **Only a quiet-machine
  run is a record**; anything measured on a machine in use is reported
  in conversation and written nowhere.
- Quiet runs alone on the machine: no builds, sweeps or other sessions.
  K: wait for the 1-minute load to fall below 2.0. I: the script waits
  for 0.30.
- Z: a locked screen puts running processes under Windows power
  throttling (EcoQoS), 1.5× slower, and it stays with the process after
  unlock. Opt each measured process out (`SetProcessInformation`,
  ProcessPowerThrottling, EXECUTION_SPEED in ControlMask, StateMask 0);
  children do not inherit it, so launch every binary yourself, not
  through ctest. Helper:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/nothrottle.ps1`.
  The CPU also steps down some 8–15 s into sustained load: set
  `LIBSTATS_BENCH_WARMUP_SECONDS=25` for the benches and precede each
  `strategy_profile` run with a discarded short pass.
- Macs: run every library binary outside the Claude Code sandbox. It
  blocks the `hw.*cachesize` sysctls, so cache sizes read 0 and the
  cache-derived tuning changes; it also blocks the SSH agent.

### The steps

**L1 — native correctness.** Fresh Release build (`cmake --preset release
-G Ninja`, or the Windows presets in AGENTS.md); expect no warnings. Run
`system_inspector --quick` and record the tier. Run
`ctest -LE "timing|benchmark"`; the count is 135 on Linux and may differ
by platform. Record what `test_parallel_backend_selection` selects: GCD
on the Macs, `std::execution` on Windows, and on Linux `std::execution`
with TBB or pthreads without. On the Macs this is the first native run of
the #179 change.

**L2 — accuracy sweep and block.** The tree must be clean and at a pushed
commit; the sweep stamps the commit it measured. `accuracy_sweep
sweep.csv` (write it outside the repo), then
`python3 tools/accuracy_vs_mpmath.py sweep.csv` with mpmath installed (ask
the user before installing anything). A full sweep rewrites only its own
`isa=` block; check with `git diff`. Expect 0 contract violations and no
row worse than the same row in that ISA's v2.5.0 block beyond noise.
Keep the sweep CSV and the per-row scores (`--rows-out`) for L2b.

**L2b — row gate and off-grid sweep.** Commands and how to read the
output: `docs/ACCURACY_CHARACTERIZATION.md`, "Row gate" and "Off-grid
sweep".
1. Row gate against v2.5.0, on the same inputs: build `v2.5.0` beside
   the checkout, sweep it, then sweep this branch with `--grid-from` that
   CSV, and score with `--baseline` the v2.5.0 rows. Two builds sweep
   neighbouring doubles otherwise, and the comparison is not row for row.
   Expected here: about 1,100 values better; the worse ones are densities
   below shape 20 (#314). Anything else worse is a stop-and-report.
2. Write this ISA's allowlist with `--allow-out
   data/accuracy/allow_<ISA>.csv` from the L2 sweep.
3. Off-grid: `accuracy_sweep offgrid.csv --offgrid` (default seed 213),
   then the oracle with `--no-doc --row-timeout 20 --budget 4`. Known on
   AVX: #316 (Cauchy quantile), #317 (Fisher-F at subnormal x), #318
   (TruncatedNormal narrow-window density). Report anything else.
   Known oracle gap: 120 NegativeBinomial survival rows exceed the time
   limit and are not scored.

**L3 — prototype bit-identity.** The batch-CDF span routing and the
integer `lgamma` table (`23f6fb3`, merge `90d1bef`) must not change a
bit. Build `b766e45` and `90d1bef` Release, run each one's
`accuracy_sweep`, and `diff` the two CSVs past the banner lines: expect
no difference. Record the batch timings the bench in L4 gives for the
discrete PMFs and the incomplete gamma and beta CDFs. The `PROTOTYPE`
tags in `src/` come out once M, Z and K have all reported.

**L4 — quiet session**, last on each machine, one sitting:
1. `ctest -j1 -L timing` (Z: launch each binary directly).
2. `tools/simd_verification`.
3. `tools/bench/distributions_bench.cpp` and `elementary_bench.cpp`,
   each built against a `main` build and against this branch with the
   library's own flags, run alternately, three passes. Build lines:
   `tools/bench/README.md`. This is the machine's number for #156.
4. Three `strategy_profile --large -o strategy_profile_run<i>.csv` runs.
5. Bundle as `data/profiles/dispatcher/README.md` describes
   (`metadata.json`, `manifest.txt` with the conditions and the spread,
   the three CSVs, `sustained_crossovers.txt`, logs).

**L5 — threshold rows.** Every table is stale: the adoption changed
kernel costs on every tier, and #311 changed them again on CPUs without
FMA. Derive this tier's rows from the L4 bundle with
`scripts/derive_threshold_rows.py` (recipe in the dispatcher README; its
`TABLES` entry for your tier must point at the new bundle), then settle
them with `tools/bench/table_ab_bench.cpp`, old table against new, run
old, new, old, new. Rules [user, 2026-10-14, still in force]: a candidate
at the machine's fork threshold is not a measured crossover; skip a row
whose three runs spread more than 2× unless its maximum still clears the
table by more than one grid step; per-row override comments stay unless
the comment is stale. Then L1 again. Tier ownership: kAvx512 is Z's
(clang-cl), kNeon is M's, kAvx2 is K's, **kAvx is I's** [user,
2026-11-19: native values from the only silicon whose native tier is
AVX; on I a candidate of 6144 stays 6144 where PARALLEL is behind at
4096]. kSse2 delegates to kAvx.

**L6 — report.** State the bench result on #156 and update your status
line. Do not decide #156's gates; that follows all four machines.

### Machine-specific items

- **M**: #107 (one trig table; the change is on the NEON side) is done
  here or not at all. #47 is owed its AppleClang validation (von Mises
  pdf/logpdf ≤ 1.7 ulp on AVX). #156's original numbers are from this
  machine: `gamma_p` 272 and `beta_p` 823 ns per element against targets
  of 40 and 30 that corvus's accuracy contract puts out of reach.
- **Z**: clang-cl is the performance build and supplies kAvx512; cl.exe
  is the correctness build. CI's cl.exe leg excludes
  `test_concurrency_stress` (starvation at cl.exe-built corvus speed);
  say whether it passes natively. A capped-tier leg
  (`LIBSTATS_MAX_SIMD_TIER`) is optional.
- **K**: the only hyper-threaded macOS machine, so the only one where
  AUTO reaches WORK_STEALING. #47's AppleClang validation applies here
  too.
- **I**: after the capture, assemble the bundle, derive kAvx, write the
  table and the documentation column #291 asks for, regenerate the AVX
  block, and state the bench on #156. Unverified here: that mixing
  libstats' VEX code with corvus's non-VEX SSE4 kernels costs nothing on
  this CPU (check `other_assists.avx_to_sse` with `perf stat` if the
  bench reads oddly).

### After the legs

1. Open the dev-branch PR once one other leg is complete.
2. Remove the `PROTOTYPE` tags once all three have reported L3.
3. #156: decide the gates from the four machines' numbers [user].
4. Release docs: version 2.5.0 → 3.0.0, `CHANGELOG.md`,
   `docs/VALIDATION_HISTORY.md` (matrix, costs, the threshold tables),
   `MIGRATION_GUIDE.md`, README and AGENTS status lines.
5. Release: PR to `main` closing every issue labelled
   `fixed-on-branch`; CI green; signed tag; GitHub release; close the
   milestone. Then pylibstats and libhmm (Cross-Repo).

## State of the v3.0.0 issues [DERIVED, 2026-11-21]

Milestone #6: 32 open, 1 closed (#312). Issues stay open until this
branch reaches `main`; `fixed-on-branch` marks the ones that are fixed
and guarded here, each with a status comment.

- **Labelled `fixed-on-branch` (17):** #108, #110, #113, #136, #138,
  #179, #221, #247, #248, #251, #252, #262, #267, #270, #271, #290, #311.
- **Done here, label owed:** #213 and #258 (tools, runbook step and
  allowlist in; commented), #287 (fixed in `e890dec`; label when its CI
  run is read), #126 and #214 (guards in `c0f8c3b`).
- **Wait on the other machines:** #47 (AppleClang), #52 (timings and
  rows on every tier), #103 and #104 (guards pass and the sweep is clean
  on AVX; other ISAs), #141 (`gamma_p`/`gamma_q` ≤ 0.48 ulp on AVX; other
  ISAs), #107 (M), #291 (kAvx capture, I).
- **Resolved halves only:** #266 (CDF side; density prefactors are
  #314) and #272 (Beta and NegativeBinomial CDF NaNs; the rest is #315).
- **Decision owed [user]:** #156, after the legs.

Kept in libstats and not on corvus, by decision: the Student-t tail
kernels (`beta_continued_fraction`, `beta_i_large_a_half`) and
`inverse_t_cdf`'s solver; the density prefactor family
(`log_gamma_prefactor`, `beta_prefactor_constant`, `log1pmx`,
`stirling_remainder`); `inv_survival_normal_log`; the incomplete gamma
below `DBL_MIN` (`gamma_p_from_scaled`); libstats' own `exp`, `log`,
`cos` and `sin` SIMD kernels.

Filed from this work into the patch milestone: #314, #315, #316, #317,
#318, #319. Small and unfiled:
- `scripts/analyze_crossovers.py` reports Uniform and NegBinomial as "not
  in table" (a wrapped row; a name mismatch). The derive script reads both.
- `src/simd_avx.cpp`'s cos/sin comment still reads "UNPROVEN LEG"; the
  native gate reads 1 ULP.
- Clang without an FMA tier warns that `kBudgetTight` is unused
  (`tests/test_trig_ulp_gates.cpp`).
- The Windows CI legs warn that `ilammy/msvc-dev-cmd` v1.13.0 targets
  Node 20. It is that action's latest release; harmless today. The likely
  fix is a script step calling `vcvars64.bat`, probably fleet-wide.

## Decided [DERIVED]

For v3.0.0:
- **Scope and order [user, 2026-10-26, 2026-11-19].** v3.0.0 is the
  corvus adoption plus the issues it needs. The accuracy patch
  (milestone #8) ships after it, as v3.0.1; none of its issues close as
  overtaken, since they are real defects. Then v3.1.0 New Distributions,
  then the Architecture Refactor.
- **Per-call cost does not block v3.0.0 [user, 2026-11-18].** It ships on
  corvus v1.2.0 for the accuracy. Speed is recovered on a v3.1.0 pinned
  to corvus v1.3.0 (corvus #110, #111). Speedups that need no corvus
  change may be scoped into v3.0.0. Measured on Ivy Bridge, corvus's
  worst case in the fleet (two lanes, no FMA): single calls `lgamma(2)`
  18 → 144 ns, `lbeta` 76 → 1,742, `gamma_p` 112 → 1,358, `beta_i`
  342 → 3,531 against v2.5.0.
- **No PR for the dev branch** until one other machine has run its leg
  [user, 2026-11-18].
- **Placement [user, 2026-11-19].** A fix that applies to `main` goes to
  `main` first, then `main` is merged into the dev branch. clang-cl work
  stays on the dev branch: clang-cl is needed only because of corvus and
  Highway under MSVC.
- **A fetched corvus builds as Release and uninstrumented** whatever
  libstats' configuration [user, 2026-10-25]. Instrumenting the
  dependency is noise, and corvus at -O0 starves the concurrency tests.
- **CI budgets [user, 2026-10-26; re-measured 2026-11-18].** Coverage 30
  minutes with the suite whole. cl.exe leg 40 minutes, all four Highway
  tiers, `test_snapshot_consistency` in, `test_concurrency_stress` out
  (it starves at cl.exe-built corvus speed; it runs on every other leg).
  If a slow runner pushes that leg at its budget, the snapshot test is
  the first thing to take out again.
- **`expm1`/`log1pmx` are not adopted from corvus in v3.0.0** [user,
  2026-11-18]. No accuracy to win (the rows that rest on `expm1` already
  read 1e-16), corvus exports them as spans only (56 and 105 ns for one
  call against glibc's 13), and libstats' uses are mostly scalar. On
  spans `expm1` is level with glibc, so the Exponential and Rayleigh
  batch loops are a v3.1.0 candidate.
- **kAvx takes Ivy Bridge's native values** [user, 2026-11-19], from one
  quiet capture after `src/` settles.
- **Trig and coefficient duplication stays** [user, 2026-11-21; #108,
  #221]: AGENTS.md, "SIMD kernel conventions".
- **Parallel backends for the `safe_*` wrappers** [user, 2026-11-21;
  #319]: GCD on macOS; `std::execution` on Windows and on Linux with
  TBB; the library's own pool on Linux without TBB; OpenMP not selected.
  v3.0.0 carries only the selection fixes (#179); the rest is #319.
- **#156's gates** are decided after every machine has post-bump numbers
  [user, 2026-11-18].

Standing:
- **Windows performance build is clang-cl [user, 2026-10-05]**, with
  cl.exe's `/arch:` flags (`/arch:AVX2` = AVX2 + FMA, F16C, BMI1/2,
  LZCNT, MOVBE; `/arch:AVX512` adds AVX-512 F, CD, BW, DQ, VL), so both
  compilers build the same ISA. cl.exe is the correctness build
  (standards WINDOWS-TOOLCHAIN §5); Z's timing and kAvx512 come from
  clang-cl.
- **Why the adoption is a major version [user, 2026-10-04].** The
  installed package gains `find_dependency(corvus CONFIG)`, so under
  `SameMajorVersion` a `find_package(libstats 2.x)` consumer would accept
  it and fail to configure without corvus; it also ends "zero external
  dependencies".
- **Accuracy-for-speed budget [user, 2026-10-04].** Slowdowns of 2×, 5×
  and up to ~10× are acceptable for accuracy fixes; 50–150× is not. 10×
  is a guide, not a hard limit: weigh how hot the function is for
  consumers (a Gaussian pdf matters far more than a von Mises quantile)
  against the effort, and maximise speed at the accuracy kept, within
  reasonable effort. Over-budget rows get a performance pass or a
  theoretical case that the cost is intrinsic (Known Gaps). Per-call
  corvus cost in v3.0.0 is the decided exception above.
- **Vectorized log1p and expm1 [user, 2026-10-04].** A `VectorOps`
  primitive is preferred to an inline composite. `vector_log1p` and
  `vector_expm1` are derived generically from each tier's `vector_log`
  and `vector_exp`; neither serves log1p(x) − x, whose cancellation
  amplifies their few ulp, so `detail::log1pmx_series` (atanh form) does.
  Pointing them at corvus's spans is the v3.1.0 candidate above.
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
- **Project skills** live once, in `.claude/skills/`; `.agents/skills` is
  a tracked relative symlink. Edit only `.claude/skills/`. On Windows it
  needs Developer Mode plus a repository-local `core.symlinks=true`; Git
  for Windows defaults to false, and a clone records the value it found.
  Z is set (2026-10-03).
- **Archived branches** are signed tags: `archive/spike-corvus-bessel`
  (the only Tier 0 corvus Bessel code) and `archive/dev-v2.5.0-corvus`
  (the first adoption branch, ported onto v2.5.0 as this one).
- **CI and tooling [2026-11-11]** (rules and incidents: CI House Style §1,
  §6, §8).
  - Runner labels: jobs that install a versioned tool (gcc-13, gcc-14,
    clang-17, clang-tidy-17) are pinned to `ubuntu-24.04`, so image and
    tool move in one commit; everything else floats, macOS included
    (#294). What moving that pin needs: Known Gaps, "CI on ubuntu-26.04".
  - The canary runs weekly (Monday, 03:25–03:40 UTC across the four
    workflows; `docs.yml` gained a schedule). A `canary-report` job opens
    a "Canary failed" issue, or comments on the open one (#295).
  - Pre-commit is enforced (#296): the `precommit` job runs every hook on
    all files. Before this the config existed and nothing ran it.
  - Project hooks are Python and both can fail: `check_pragma_once.py`
    and `check_no_debug.py` (debug markers; `std::cout` in library code
    outside a DEBUG-gated block). The copyright hook (no file carried a
    header) and the include-order hook (clang-format owns include order)
    were removed. Each hook was shown to fail on a planted case.
  - gersemi replaces cmake-format; clang-format hook is v23.1.5 (#297).
    Formatter output of #296 is `39e8402`, in `.git-blame-ignore-revs`.
  - `docs/bench-evidence/` is excluded from the fixers, clang-format and
    shellcheck: it is captured output, and two files hold lone CRs.
  - `tests/tsan.supp` suppresses the exception-release false positive on
    two frames, one per libstdc++ version (#294, #298).
  - No `actions/cache` in the `precommit` job (zizmor `cache-poisoning`).
  - `reviewdog/action-actionlint` stays at v1.83.0 or newer (#292).

## GitHub Synchronization [DERIVED]
Last reconciled against live GitHub state: 2026-11-21. No open issue
without a milestone; no open PRs.
- GitHub is the collaborator-facing source for issues and milestones;
  this file is the agent-facing state. When creating, closing, retitling
  or moving an issue or milestone, update this file in the same change.
- Re-check when a task reads or changes the backlog, or after 7 days.

## GitHub Milestones [DERIVED]
Closed: #1 v2.2.0, #5 v2.3.0, #7 v2.3.1, #2 v2.4.0, #9 v2.4.1, #10 v2.5.0
(released 2026-10-16). Milestone numbers do not sort in version order.

- **#6 v3.0.0 — corvus adoption** (32 open, 1 closed): itemized in "State
  of the v3.0.0 issues" above.
- **#8 Accuracy, contracts & kernel hygiene patch** (33 open, 8 closed),
  v3.0.1:
  - Accuracy: #215, #243, #244, #245, #246, #249, #250, #263, #273, #277,
    #314, #315, #316, #317, #318. #245 may already be fixed (`cd9a2ab`);
    check before working it. #244's "exact path" premise looks wrong.
  - Oracle and test tooling: #259, #260, #261.
  - Concurrency: #181, #182, #217.
  - Contracts, API and docs: #220, #253, #268, #269.
  - Cleanup and CI: #114, #152, #218, #219.
  - Speed and the parallel layer: #178, #283, #284, #319.
- **#3 v3.1.0 — New Distributions (Extended)** (6 open): #58 GEV, #59
  LogLogistic, #60 Triangular, #61 Wald, #62 Hypergeometric +
  BetaBinomial + Zipf, and #307 (the fixed 4,096-element gate in
  `parallelForSlices`). Settle #62's Zipf CDF design before planning; it
  scopes corvus work.
- **#4 Post-v3 — Architecture Refactor** (8 open): #40, #41, #42, #43,
  #128, #222, #282, #285. #282 (one slice callable per batch site behind
  one scheduling seam) is the first step of #43 and is wanted whatever
  the threading direction [user, 2026-10-26]. #285 is gated on a fleet
  executor concept. #42 is on hold.

## Design inputs from the threshold captures [OPEN]
From the v2.5.0 captures on three machines (tested 2026-10-06) and the
Ivy Bridge capture of 2026-11-18. Full analysis: `git show e1a11ec:PLAN.md`,
"Trends".
- **Crossovers are per machine; a table serves an architecture.** That,
  not run noise (a median 0–1 grid step), is why tables read off.
- **One threshold per (distribution, operation) cannot serve every
  parameter**: per-element cost varies up to 10× with the parameters, and
  AUTO is late mostly at costly ones.
- **SSE2 crosses sooner than AVX** (K 21 rows sooner, 0 later; Z 15, 1),
  so delegating kSse2 to kAvx is late on a fifth to a quarter of rows.
  No kSse2 table yet.
- **Below 4,096 elements PARALLEL does not fork** (#307), so a candidate
  there is not a crossover.
- **Thresholds follow per-element cost: supported** [DERIVED,
  2026-10-06]. log(crossover) against log(VECTORIZED ns/element) has
  slope −0.5 to −1.03 (−1 predicted; M −1.03, R² 0.85; K AVX −0.97,
  0.70; weakest Z SSE2 −0.51, 0.29), and one constant per machine and
  tier (2^16.8–2^18.2 ns) puts 69% of 648 rows within noise + 1 step,
  against 75% for the hand-tuned tables. K and Z differ by cost, not by
  noise: Z crosses later on 19–27 rows and sooner on 9–11, its
  vectorized cost about half K's, and the crossover ratio follows the
  cost ratio with slope +1.1 (AVX2, AVX; SSE2 +0.36). Exceptions are the
  two parallel-path defects fixed in v2.5.0 (#175, #176). Corvus replaced
  the kernels and invalidated every table, hence L5. Evaluate a cost model (per-element costs × one per-machine
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
- **CI on ubuntu-26.04** [2026-11-11]: the jobs that install versioned
  tools are pinned to `ubuntu-24.04` (#294, CI House Style §1). Moving the
  pin needs two things first: lcov 2.4 fails with `inconsistent` under
  `gcov-13` (`tests/test_discrete_quantile_bounds.cpp:57`), and the TSan
  exception-release false positive in
  `test_parallel_exception_propagation` is suppressed on two frames that
  differ between libstdc++ 14 and 15 (`tests/tsan.supp`); it depends on
  timing, so a pass proves little.
- **No CI leg runs the install-contract steps without TBB** since the GCC
  leg gained `libtbb-dev` (#290). Checked once locally; a second install
  leg would cover it at the cost of runner time.
- **`consumer_example_fetchcontent` fetches `main`**, so it cannot
  exercise this branch until it is merged.
- **Concurrency findings of 2026-10-06** are filed as #184–#191 and
  #178–#183; evidence in
  `docs/bench-evidence/2026-10-06-m1-concurrency/summary.txt`. Decided
  then: a FORCE_SCALAR batch locks per element and can mix states; one
  state under every strategy is a v3 design input (AGENTS.md, Batch API
  contracts).
- **Within the accuracy-for-speed budget, left as is**: Student-t CDF and
  quantile at large ν (5–8× against v2.4.1; the continued fraction),
  discrete log-pmfs (0.9–2×), FisherF (2–3×). The cost bench sweeps
  parameter bands at a fixed input range and does not resolve x-regions.
- **MSVC codegen behind clang-cl** [Z, 2026-10-05]: Gamma at large shapes
  1.5–1.6× behind, since MSVC does not contract FMAs; `vector_expm1`'s
  Taylor loop is scalar under MSVC, which the Pareto and Weibull batch
  CDFs use. Evidence:
  `docs/bench-evidence/2026-10-05-zen4-msvc-vs-clangcl/`.
- The `exp_max` clamp sits about 30 ULP below the true overflow
  threshold; in that window the kernels return `exp(exp_max)` where
  `std::exp` is still finite. A deliberate margin.
- von Mises circular variance (#93 residual): forming 1 − A costs about
  2κ × A's ULP error near the κ = 50 cut. Close to intrinsic in double;
  only a double-double complement export from corvus would remove it.

## Cross-Repo Dependencies [OPEN]
- **pylibstats** pins this repo by a `find_package` floor and a
  FetchContent `GIT_TAG` in its `CMakeLists.txt`; its pin-currency canary
  fails when it falls behind. It is at v2.5.0 (pylibstats 0.8.0). v3.0.0
  needs a major-version floor change and brings corvus into the wheels
  (pylibstats #20, #26): the Windows wheel (accept corvus's AVX2 cap
  under cl.exe, or build with clang-cl), the NOTICE and Apache-2.0 text
  for Highway, and Linux aarch64 NEON, which is unvalidated.
- **libhmm** ships the same corvus release (fleet coherence rule, libhmm
  #106) and needs the GoogleTest install fix (libhmm #146, this repo's
  #287). Shared rules and kernels: libhmm #103, #108.
- **corvus**: #110 (scalar entry points for `lgamma` and `lbeta`) and
  #111 (`LgammaDiffDd` cost) are the v1.3.0 work that v3.1.0 waits on;
  #44 (von Mises quadrature machinery, fixed-cost incomplete gamma/beta)
  and #45 (`expm1`, `log1pmx` spans, landed) came from v2.5.0. The
  corvus session is the user's own.
- **standards**: the Node 20 action finding and the capture-harness
  convention (one harness for warm-up, the quiet gate, the throttling
  opt-out and the bundle layout) are candidates for the fleet repo.
